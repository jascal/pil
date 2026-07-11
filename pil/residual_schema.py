"""Bridge: residual candidates → PIC schemas (first integration slice).

Implements the step-2 *bridge* from ``docs/notes/residual_as_schema.md``: an
admitted residual leaf becomes a :class:`~pil.schemas.Schema`, so the existing
:class:`~pil.schemas.SchemaBank` / :func:`~pil.schemas.propose_schemas` machinery
selects, weights, and exports it — instead of a parallel symbolic admit loop.

**Scope of this slice — token-presence residuals.** A candidate whose firing word
``src[0]`` appears anywhere in the context window fires its single target token
``tgt[0]`` (a Freebase ``ns:`` path for ``relation_atom``, or any single-token
leaf). The tensor ``predict`` and the exported Datalog clause are verified
equivalent (``tests/test_residual_schema.py`` — including a souffle round-trip).

**Still open (see the note, do not overclaim):**
  - CFQ *set-F1* selection for bag prediction — ``propose_schemas`` here scores
    single-token exact-match, which is the honest reuse, not the set metric.
  - n-fold *unit* schemas (multi-token ``tgt``; needs the value table) — skipped.
  - soft-NLL weight training of admitted residual weights.

Symbols must already share a vocabulary: ``stoi`` maps both question words and
target paths to token ids. Building a CFQ ``stoi`` is a later slice; the bridge
itself is vocabulary-agnostic.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from pil.residual_template import ResidualCandidate
from pil.schemas import Schema


def _presence_schema(word_id: int, path_id: int, name: str) -> Schema:
    """A schema that fires ``path_id`` when ``word_id`` is present in the window.

    ``predict`` ignores ``values``/``v2t`` (presence is not an arithmetic
    predicate). The Datalog body matches the word token at *any* position
    (``tok(I,_,word_id)``) and assigns the constant target token to ``C`` — the
    exporter joins ``candtok(V,C)`` and the learned weight.
    """

    def predict(x: Tensor, values: Tensor, v2t: Tensor) -> Tensor:  # noqa: ARG001
        present = (x == word_id).any(dim=1)
        path = torch.full((x.shape[0],), path_id, dtype=torch.long, device=x.device)
        return torch.where(present, path, torch.full_like(path, -1))

    return Schema(name=name, predict=predict, datalog=f"tok(I,_,{word_id}), C={path_id}")


def residual_candidates_to_schemas(
    candidates: Sequence[ResidualCandidate],
    stoi: Mapping[str, int],
) -> tuple[list[Schema], list[dict]]:
    """Bridge residual leaves to token-presence schemas.

    A candidate bridges iff it has a firing word (``src[0]``), a *single* target
    token (``tgt[0]``), and both symbols are in ``stoi``. Everything else is
    skipped with a reason (returned, not swallowed) — this slice is single-token
    presence only. Duplicate ``(word, path)`` pairs collapse to one schema.

    Returns ``(schemas, skipped)`` where each skipped entry is
    ``{"src", "template_id", "reason"}``.
    """
    schemas: list[Schema] = []
    skipped: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for c in candidates:
        if not c.src or len(c.tgt) != 1:
            skipped.append({
                "src": c.src, "template_id": c.template_id,
                "reason": "not-single-token-target",
            })
            continue
        word, path = c.src[0], c.tgt[0]
        if word not in stoi or path not in stoi:
            skipped.append({
                "src": c.src, "template_id": c.template_id,
                "reason": "unknown-symbol",
            })
            continue
        key = (stoi[word], stoi[path])
        if key in seen:
            continue
        seen.add(key)
        schemas.append(_presence_schema(
            key[0], key[1], name=f"{c.template_id}/{word}|{path}",
        ))
    return schemas, skipped


def cfq_stoi_from(words: Iterable[str], paths: Iterable[str]) -> dict[str, int]:
    """Shared CFQ vocab: distinct words then distinct paths, sorted for determinism.

    Ids are assigned 0..N-1 over ``sorted(set(words))`` first, then continuing over
    ``sorted(set(paths))`` (so words and paths share one id space with no overlap).
    Calling this twice with the same *sets* of words/paths (regardless of input order
    or duplicates) MUST return an identical dict.
    """
    stoi: dict[str, int] = {}
    for w in sorted(set(words)):
        stoi[w] = len(stoi)
    for p in sorted(set(paths)):
        if p not in stoi:
            stoi[p] = len(stoi)
    return stoi


def encode_cfq(
    word_lists: Sequence[Sequence[str]],
    stoi: Mapping[str, int],
    window: int,
) -> Tensor:
    """Encode ``word_lists`` (one list of question words per row) into an (B, window)
    long tensor of token ids, using ``stoi``. Words not in ``stoi`` are skipped (not
    given a placeholder id). Row i holds the ids of ``word_lists[i]``'s known words in
    their original order, left-packed, padded with -1 to length ``window`` (truncated
    if there are more than ``window`` known words). -1 must never collide with a real
    id (guaranteed since cfq_stoi_from assigns ids starting at 0).
    """
    B = len(word_lists)
    out = torch.full((B, window), -1, dtype=torch.long)
    for i, words in enumerate(word_lists):
        ids = [stoi[w] for w in words if w in stoi][:window]
        if ids:
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def _set_f1(pred: set[int], gold: set[int]) -> float:
    """Identical branch structure to ``experiments.campaign_cfq_residual.set_f1``."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    p, r = inter / len(pred), inter / len(gold)
    return 2 * p * r / (p + r)


@torch.no_grad()
def schema_set_predict(schemas: Sequence[Schema], X: Tensor) -> list[set[int]]:
    """Per row of X: the set of non-(-1) path ids fired by any schema in ``schemas``.

    Presence schemas ignore ``values``/``v2t`` — pass empty dummy tensors
    (``torch.empty(0, dtype=torch.long, device=X.device)``) as those two arguments to
    each ``schema.predict(X, empty_values, empty_v2t)`` call. Implementation shape:
    loop over ``schemas`` (small), and for each schema do ONE vectorized tensor call
    producing a (B,) tok tensor, then only iterate the (typically few) hit rows —
    do NOT nest a per-row Python loop inside a per-schema Python loop naively; this
    function is called O(rounds * |candidates|) times by propose_schemas_setf1 on
    real CFQ data (hundreds of val rows, tens of schemas, dozens of rounds), so a
    quadratic-in-both-dimensions pure-Python implementation will be too slow.
    """
    B = X.shape[0]
    out: list[set[int]] = [set() for _ in range(B)]
    empty = torch.empty(0, dtype=torch.long, device=X.device)
    for schema in schemas:
        tok = schema.predict(X, empty, empty)
        hits = (tok != -1).nonzero(as_tuple=True)[0]
        for i in hits.tolist():
            out[i].add(int(tok[i].item()))
    return out


@torch.no_grad()
def mean_set_f1_schemas(
    schemas: Sequence[Schema],
    X: Tensor,
    gold_sets: Sequence[set[int]],
) -> float:
    """Mean set-F1 over rows via schema_set_predict + Python-float ``_set_f1``.

    Accumulates in Python floats (same pattern as ``campaign_cfq_residual.mean_set_f1``)
    so there is no float32 tensor-mean drift vs the residual admit path. Empty
    ``gold_sets`` returns 0.0.
    """
    if not gold_sets:
        return 0.0
    preds = schema_set_predict(schemas, X)
    total = 0.0
    for pred, gold in zip(preds, gold_sets, strict=True):
        total += _set_f1(pred, gold)
    return total / len(gold_sets)


@torch.no_grad()
def propose_schemas_setf1(
    base_schemas: list[Schema],
    cand_schemas: list[Schema],
    X_val: Tensor,
    gold_val: Sequence[set[int]],
    *,
    thresh: float = 1e-4,
    max_rules: int = 32,
) -> tuple[list[Schema], list[dict]]:
    """Greedy val-marginal set-F1 selection — mirrors ResidualFamily._admit_naive.

    Control flow matches ``_admit_naive`` line for line (strict ``>`` to admit,
    first-candidate-wins on exact marginal ties, ``base`` recomputed after each
    admission, candidate order preserved). Schemas stand in for residual maps;
    ``mean_set_f1_schemas`` stands in for ``score_fn``.

    **Parity contract** (empirical, verified on synthetic fixture + mcd1 experiment):
    for identical base maps, candidate pool, val examples, thresh, max_rules —
    ``{(word, path) of selected schemas}`` equals the admitted-minus-base keys from
    ``fam.admit``, when ``cand_schemas`` is bridged from ``fam.propose(base_maps)``
    unchanged via ``residual_candidates_to_schemas``. For ``relation_atom``,
    ``src == (word, path)`` always, so residual ``src``-dedup and schema
    ``(stoi[word], stoi[path])``-dedup coincide. Schema ``(word, path)`` is recovered
    from ``name`` as ``f"{template_id}/{word}|{path}"`` (split first ``/``, then first ``|``).
    """
    selected: list[Schema] = []
    base = mean_set_f1_schemas(base_schemas, X_val, gold_val)
    remaining = list(cand_schemas)  # PRESERVE ORDER, do not sort/shuffle
    log: list[dict] = []
    for _ in range(max_rules):
        best: tuple[float, Schema | None] = (thresh, None)  # STRICT > thresh to admit
        for cand in remaining:  # FIRST-WINS on an exact marginal tie
            sc = mean_set_f1_schemas(base_schemas + selected + [cand], X_val, gold_val)
            marg = sc - base
            if marg > best[0]:  # STRICT >, never >=
                best = (marg, cand)
        if best[1] is None:
            break
        cand = best[1]
        selected.append(cand)
        remaining.remove(cand)
        base = mean_set_f1_schemas(base_schemas + selected, X_val, gold_val)  # recompute AFTER
        log.append({"event": "admit", "name": cand.name, "marginal": best[0], "score": base})
    log.append({"event": "stop", "n_selected": len(selected)})
    return selected, log
