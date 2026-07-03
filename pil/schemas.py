"""Schema rules: ILP-style background-knowledge proposers for the PIC rule learner.

A *schema* is a parametric rule template over the numeric/positional semantics of tokens --
the background knowledge channel of classical ILP, and the generalizing counterpart of
ground token rules. Examples (all pure Datalog once grounded):

  add_mod    decide the token whose value is (val(a) + val(b)) mod m   -- modular arithmetic
  sub_mod    ... (val(a) - val(b)) mod m
  mul_mod    ... (val(a) * val(b)) mod m
  copy(o)    decide the token at context offset o                      -- the pointer/copy
                                                                          head (rosetta's
                                                                          ind_ctxlogit)

Selection is *learned from data*: :func:`propose_schemas` evaluates every library entry's
exact-match rate on the training set and accepts those above a threshold; an accepted
schema joins the program as a source with a learnable weight ``w_s`` (zero-initialized, so
adoption is decode-neutral until SGD earns it). At export, a schema renders as an
arithmetic Datalog clause over ``num(token, value)`` facts, e.g.::

  contrib(I,V,w) :- tok(I,0,A), tok(I,2,B), num(A,Na), num(B,Nb),
                    Nc = (((Na + Nb) % 97) + 97) % 97, num(C,Nc), candtok(V,C).

so the learned program generalizes to *unseen* instances -- the regime where ground token
rules (and un-grokked MLPs) transfer nothing.

Every ``Schema.predict`` returns **token ids** (-1 = abstain); arithmetic schemas map their
computed value back to a token through the inverse value table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# predict(x, values, v2t) -> predicted token id per row (-1 = abstain)
PredictFn = Callable[[Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class Schema:
    """A rule template. ``datalog`` is the exported clause body; it must bind the decided
    *token id* to the variable ``C`` (the exporter joins ``candtok(V,C)`` and the weight)."""

    name: str
    predict: PredictFn
    datalog: str


def arithmetic_library(a_off: int, b_off: int, modulus: int) -> list[Schema]:
    """Binary modular-arithmetic schemas over the tokens at two context offsets."""

    def make(name: str, op: Callable[[Tensor, Tensor], Tensor], sym: str) -> Schema:
        def predict(x: Tensor, values: Tensor, v2t: Tensor) -> Tensor:
            va, vb = values[x[:, a_off]], values[x[:, b_off]]
            ok = (va >= 0) & (vb >= 0)
            out_val = op(va, vb).remainder(modulus)
            tok = v2t[out_val.clamp(min=0, max=v2t.shape[0] - 1)]
            return torch.where(ok, tok, torch.full_like(tok, -1))

        body = (
            f"tok(I,{a_off},A), tok(I,{b_off},B), num(A,Na), num(B,Nb), "
            f"Nc = (((Na {sym} Nb) % {modulus}) + {modulus}) % {modulus}, num(C,Nc)"
        )
        return Schema(f"{name}_mod{modulus}[{a_off},{b_off}]", predict, body)

    return [
        make("add", lambda a, b: a + b, "+"),
        make("sub", lambda a, b: a - b, "-"),
        make("mul", lambda a, b: a * b, "*"),
    ]


def copy_library(window: int) -> list[Schema]:
    """Pointer/copy schemas: decide the token observed at offset o (rosetta ind_ctxlogit)."""

    def make(o: int) -> Schema:
        def predict(x: Tensor, values: Tensor, v2t: Tensor) -> Tensor:  # noqa: ARG001
            return x[:, o]

        return Schema(f"copy[{o}]", predict, f"tok(I,{o},C)")

    return [make(o) for o in range(window)]


class SchemaBank(nn.Module):
    """Accepted schemas with learnable weights: adds ``w_s`` on each schema's prediction."""

    def __init__(self, schemas: list[Schema], vocab_size: int, values: dict[int, int]):
        super().__init__()
        self.schemas = schemas
        self.w = nn.Parameter(torch.zeros(len(schemas)))
        vt = torch.full((vocab_size,), -1, dtype=torch.long)
        max_v = max(values.values(), default=0)
        v2t = torch.full((max_v + 1,), -1, dtype=torch.long)
        for tok, v in values.items():
            vt[tok] = v
            v2t[v] = tok
        self.register_buffer("values", vt)
        self.register_buffer("v2t", v2t)

    def __len__(self) -> int:
        return len(self.schemas)

    def predictions(self, x: Tensor, candidate_ids: Tensor) -> Tensor:
        """(B, S) candidate *indices* per schema; -1 where abstaining / not a candidate."""
        cols = []
        for s in self.schemas:
            tok = s.predict(x, self.values, self.v2t)              # token ids, -1 abstain
            idx = torch.searchsorted(candidate_ids, tok.clamp(min=0))
            idx = idx.clamp(max=candidate_ids.shape[0] - 1)
            hit = (candidate_ids[idx] == tok) & (tok >= 0)
            cols.append(torch.where(hit, idx, torch.full_like(idx, -1)))
        return torch.stack(cols, dim=1)

    def logits(self, x: Tensor, candidate_ids: Tensor, n_cand: int) -> Tensor:
        """(B, V) additive schema contributions ``sum_s w_s * onehot(pred_s)``."""
        preds = self.predictions(x, candidate_ids)                 # (B, S)
        L = torch.zeros(x.shape[0], n_cand, dtype=self.w.dtype, device=x.device)
        mask = preds >= 0
        b, s = torch.nonzero(mask, as_tuple=True)
        L.index_put_((b, preds[b, s]), self.w[s], accumulate=True)
        return L


@torch.no_grad()
def propose_schemas(
    library: list[Schema],
    X: Tensor,
    y_cand_idx: Tensor,
    candidate_ids: Tensor,
    vocab_size: int,
    values: dict[int, int],
    min_hit_rate: float = 0.05,
) -> tuple[list[Schema], list[float]]:
    """Score every library schema by exact-match rate on (X, y); return the accepted ones.

    This *is* the learning step for the background-knowledge channel: schemas are adopted
    on evidence, weighted by SGD afterwards, and prunable like any source.
    """
    bank = SchemaBank(library, vocab_size, values)
    preds = bank.predictions(X, candidate_ids)                     # (B, S)
    rates = (preds == y_cand_idx.unsqueeze(1)).float().mean(dim=0)
    accepted, scores = [], []
    for i, s in enumerate(library):
        r = float(rates[i])
        if r >= min_hit_rate:
            accepted.append(s)
            scores.append(r)
    return accepted, scores
