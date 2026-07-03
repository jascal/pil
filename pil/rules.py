"""The differentiable PIC rule program: soft weighted-Datalog at ``T > 0``, Datalog at ``T = 0``.

A :class:`RuleProgram` is a layered PIC-LP program (``pic/spec/PIC_LP.md``) made a torch module:

  stratum 1   rules whose body literals are token predicates at context offsets
              (exact-token anchors, sign-able and relevance-gated),
  stratum >=2 rules whose body literals are (possibly negated) firings of earlier rules,
  decode      every rule carries a write direction ``a_k`` in ``H = R^d``; the logit is
              PIC's encoder read-out  ``L(v) = sum_k g_k(x) <a_k, U_v> + b_v``  over a
              candidate token set, evaluated by ``(+, ⊕_T)``: softmax when training,
              argmax when deployed (the same argmax at every T, PIC_SPEC §3.1).

Design invariants:
  * every body literal carries a **relevance** ``ρ = σ(rel) ∈ [0,1]`` (ρ=0 ⇒ wildcard: the
    literal is structurally absent) and a **sign** ``s = σ(sgn)`` (s=0 ⇒ negated literal) --
    body structure is itself a continuous parameter, hardenable to {0,1};
  * gates are ⊗-monomials of literals under a t-norm (product = probabilistic AND, min =
    Gödel/tropical AND); at hard {0,1} parameters both equal Boolean conjunction, so the
    rounded program *is* the Datalog program;
  * heads are additive (the decode is a sum over rules), so per-rule ablations, utilities,
    and the §5.5 margin certificate are exact and O(1);
  * every rule has a **stable id**; deep strata reference the atom ids they read, so
    structural edits (birth/death in any stratum) never silently re-wire other rules.

:meth:`RuleProgram.forward` with ``hard=True`` evaluates the rounded program with Boolean
gates -- the same semantics the Datalog export runs -- so soft-vs-hard agreement is a
measurable invariant, not an aspiration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

_EPS = 1e-7
ABSENT = -4.0     # rel logit for "literal absent" (rho ~ 0.018)
PRESENT = 4.0     # rel logit for "literal present" (rho ~ 0.982)
DECISIVE = 12.0   # post-hardening logit (rho/s numerically 0/1)


@dataclass
class RuleProgramConfig:
    vocab_size: int
    window: int = 8                      # context offsets 0..W-1 (W-1 = decode position)
    frame_dim: int = 32                  # d: the PIC frame / write-direction space
    candidates: list[int] | None = None  # candidate token ids (None = full vocab)
    tnorm: str = "prod"                  # 'prod' | 'min'  (the body AND)
    eq_atoms: bool = False               # relational literals x[o1]==x[o2] (Datalog joins)
    lookup_offsets: tuple[int, ...] = () # offsets with a lookup source family (ngram-table
    #                                      backbone, rosetta-style: identity-gated sources
    #                                      d = A_o[x[o]]; rules learn the composed residual)
    direct_lookup_offsets: tuple[int, ...] = ()  # offsets with a FULL-RANK token->candidate
    #                                      logit table (rosetta's gram2d shape) -- the
    #                                      bigram-capacity backbone the rank-d family lacks
    seed: int = 0
    device: str = "cpu"


def _signed(lit: Tensor, sgn: Tensor, hard: bool) -> Tensor:
    """Signed literal values ``s·m + (1-s)(1-m)`` (s=0 ⇒ negated)."""
    s = (sgn > 0).to(lit.dtype) if hard else torch.sigmoid(sgn)
    return s * lit + (1 - s) * (1 - lit)


def _gate(
    lit: Tensor,
    rel: Tensor,
    sgn: Tensor,
    thr: Tensor,
    is_thresh: Tensor,
    tnorm: str,
    beta: float,
    hard: bool,
) -> Tensor:
    """Per-rule gate: AND rules take the t-norm of relevance-mixed literals; THRESHOLD
    rules (PIC-T3: the ⊗ = + coalition bracket behind a margin turnstile ⊢_γ) fire on
    ``Σ_o ρ_o·lit_o ≥ θ`` -- exportable to Datalog as a ``count`` aggregate.

    ``lit`` (B, K, A) raw matches; ``rel``/``sgn`` (K, A); ``thr``/``is_thresh`` (K,).
    """
    rho = (rel > 0).to(lit.dtype) if hard else torch.sigmoid(rel)
    signed = _signed(lit, sgn.unsqueeze(0), hard)
    # -- AND branch: 1 - ρ + ρ·lit, reduced by the t-norm
    mixed = 1 - rho.unsqueeze(0) + rho.unsqueeze(0) * signed
    if tnorm == "min" or hard:
        g_and = mixed.amin(dim=-1)
    else:
        g_and = mixed.clamp_min(_EPS).log().sum(dim=-1).exp()
    # -- THRESHOLD branch: σ(β(Σ ρ·lit − θ)), hard: integer count ≥ ceil(θ)
    z = (rho.unsqueeze(0) * signed).sum(dim=-1)
    if hard:
        g_thr = (z >= torch.ceil(thr - 1e-6).unsqueeze(0)).to(lit.dtype)
    else:
        g_thr = torch.sigmoid(beta * (z - thr.unsqueeze(0)))
    return torch.where(is_thresh.unsqueeze(0), g_thr, g_and)


class Stratum1(nn.Module):
    """Rules over token atoms. Body literals, in column order:

      columns 0..W-1      token literals  (offset o, exact-token anchor)
      columns W..W+P-1    equality literals ``x[o1] == x[o2]`` over ``eq_pairs`` (if enabled)
                          -- the relational atoms; a positive one exports as the Datalog
                          join ``tok(I,o1,C), tok(I,o2,C)``, a negated one as ``Ca != Cb``.
    """

    def __init__(self, cfg: RuleProgramConfig):
        super().__init__()
        self.cfg = cfg
        W = cfg.window
        pairs = (
            [(a, b) for a in range(W) for b in range(a + 1, W)] if cfg.eq_atoms else []
        )
        self.register_buffer("eq_pairs", torch.tensor(pairs, dtype=torch.long).reshape(-1, 2))
        self.register_buffer("anchor", torch.zeros(0, W, dtype=torch.long))
        # per token-slot literal mode: exact (x == anchor) or ordinal (x <= anchor).
        # Ordinal literals are meaningful when consecutive token ids encode ordered levels
        # (the tabular quantile-bin encoding); export renders `tok(I,o,C), C <= c`.
        self.register_buffer("is_le", torch.zeros(0, W, dtype=torch.bool))
        self.register_buffer("is_thresh", torch.zeros(0, dtype=torch.bool))
        L = W + self.eq_pairs.shape[0]
        self.n_lits = L
        self.rel = nn.Parameter(torch.zeros(0, L))
        self.sgn = nn.Parameter(torch.zeros(0, L))
        self.thr = nn.Parameter(torch.zeros(0))

    @property
    def n_rules(self) -> int:
        return self.anchor.shape[0]

    def add_rules(
        self, anchors: Tensor, rel: Tensor, sgn: Tensor,
        thr: Tensor | None = None, is_thresh: Tensor | None = None,
        is_le: Tensor | None = None,
    ) -> None:
        n, dev = anchors.shape[0], self.rel.device
        thr = torch.zeros(n) if thr is None else thr
        is_thresh = torch.zeros(n, dtype=torch.bool) if is_thresh is None else is_thresh
        W = self.anchor.shape[1]
        is_le = torch.zeros(n, W, dtype=torch.bool) if is_le is None else is_le
        self.anchor = torch.cat([self.anchor, anchors.to(self.anchor.device)], dim=0)
        self.is_le = torch.cat([self.is_le, is_le.to(dev)], dim=0)
        self.is_thresh = torch.cat([self.is_thresh, is_thresh.to(dev)], dim=0)
        self.rel = nn.Parameter(torch.cat([self.rel.data, rel.to(dev)], dim=0))
        self.sgn = nn.Parameter(torch.cat([self.sgn.data, sgn.to(dev)], dim=0))
        self.thr = nn.Parameter(torch.cat([self.thr.data, thr.to(dev)], dim=0))

    def keep(self, mask: Tensor) -> None:
        self.anchor = self.anchor[mask]
        self.is_le = self.is_le[mask]
        self.is_thresh = self.is_thresh[mask]
        self.rel = nn.Parameter(self.rel.data[mask])
        self.sgn = nn.Parameter(self.sgn.data[mask])
        self.thr = nn.Parameter(self.thr.data[mask])

    def gates(self, x: Tensor, beta: float, hard: bool = False) -> Tensor:
        """``x`` (B, W) token ids -> (B, K) gate values in [0, 1]."""
        xa = x.unsqueeze(1)
        an = self.anchor.unsqueeze(0)
        m = torch.where(self.is_le.unsqueeze(0), xa <= an, xa == an).to(self.rel.dtype)
        if self.eq_pairs.shape[0]:
            eq = (x[:, self.eq_pairs[:, 0]] == x[:, self.eq_pairs[:, 1]]).to(m.dtype)  # (B,P)
            m = torch.cat([m, eq.unsqueeze(1).expand(-1, self.n_rules, -1)], dim=-1)
        return _gate(m, self.rel, self.sgn, self.thr, self.is_thresh, self.cfg.tnorm, beta, hard)


class StratumDeep(nn.Module):
    """Rules over earlier rules' firings; literal columns are bound to stable atom ids."""

    def __init__(self, cfg: RuleProgramConfig, atom_ids: list[int]):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("in_ids", torch.tensor(atom_ids, dtype=torch.long))
        self.register_buffer("is_thresh", torch.zeros(0, dtype=torch.bool))
        self.rel = nn.Parameter(torch.zeros(0, len(atom_ids)))
        self.sgn = nn.Parameter(torch.zeros(0, len(atom_ids)))
        self.thr = nn.Parameter(torch.zeros(0))

    @property
    def n_rules(self) -> int:
        return self.rel.shape[0]

    def add_rules(
        self, rel: Tensor, sgn: Tensor,
        thr: Tensor | None = None, is_thresh: Tensor | None = None,
    ) -> None:
        n, dev = rel.shape[0], self.rel.device
        thr = torch.zeros(n) if thr is None else thr
        is_thresh = torch.zeros(n, dtype=torch.bool) if is_thresh is None else is_thresh
        self.is_thresh = torch.cat([self.is_thresh, is_thresh.to(dev)], dim=0)
        self.rel = nn.Parameter(torch.cat([self.rel.data, rel.to(dev)], dim=0))
        self.sgn = nn.Parameter(torch.cat([self.sgn.data, sgn.to(dev)], dim=0))
        self.thr = nn.Parameter(torch.cat([self.thr.data, thr.to(dev)], dim=0))

    def keep(self, mask: Tensor) -> None:
        self.is_thresh = self.is_thresh[mask]
        self.rel = nn.Parameter(self.rel.data[mask])
        self.sgn = nn.Parameter(self.sgn.data[mask])
        self.thr = nn.Parameter(self.thr.data[mask])

    def drop_atoms(self, dead_ids: set[int]) -> None:
        """Remove literal columns whose source rule was pruned."""
        keep = torch.tensor([int(i) not in dead_ids for i in self.in_ids], dtype=torch.bool)
        self.in_ids = self.in_ids[keep]
        self.rel = nn.Parameter(self.rel.data[:, keep])
        self.sgn = nn.Parameter(self.sgn.data[:, keep])

    def extend_atoms(self, new_ids: list[int]) -> None:
        """Expose newer atoms to this stratum's rules (initialized absent: rel = ABSENT)."""
        if not new_ids:
            return
        k, dev = self.n_rules, self.rel.device
        self.in_ids = torch.cat([self.in_ids, torch.tensor(new_ids, device=self.in_ids.device)])
        self.rel = nn.Parameter(
            torch.cat([self.rel.data, torch.full((k, len(new_ids)), ABSENT, device=dev)], dim=1)
        )
        self.sgn = nn.Parameter(
            torch.cat([self.sgn.data, torch.full((k, len(new_ids)), PRESENT, device=dev)], dim=1)
        )

    def gates(self, atoms: Tensor, beta: float, hard: bool = False) -> Tensor:
        """``atoms`` (B, A) firing values for exactly ``self.in_ids`` -> (B, K)."""
        lit = atoms.unsqueeze(1)
        return _gate(lit, self.rel, self.sgn, self.thr, self.is_thresh, self.cfg.tnorm, beta, hard)


class RuleProgram(nn.Module):
    """A layered PIC-LP program with a PIC-frame decode head and stable rule ids."""

    def __init__(self, cfg: RuleProgramConfig):
        super().__init__()
        self.cfg = cfg
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

        cands = cfg.candidates if cfg.candidates is not None else list(range(cfg.vocab_size))
        self.register_buffer("candidate_ids", torch.tensor(sorted(cands), dtype=torch.long))
        V = len(cands)

        U0 = torch.randn(V, cfg.frame_dim, generator=gen)
        self.U = nn.Parameter(U0 / U0.norm(dim=-1, keepdim=True))
        self.bias = nn.Parameter(torch.zeros(V))

        self.strata = nn.ModuleList([Stratum1(cfg)])
        self.heads = nn.ParameterList([nn.Parameter(torch.zeros(0, cfg.frame_dim))])
        self.rule_ids: list[list[int]] = [[]]     # stable id per rule, per stratum
        self._next_id = 0
        self.beta = 4.0                            # threshold-gate slope (annealed by the learner)

        # lookup source families (the retrieved/ngram backbone; zero-init = decode-neutral)
        self.lookup = nn.ModuleDict(
            {
                str(o): nn.Embedding(cfg.vocab_size, cfg.frame_dim)
                for o in cfg.lookup_offsets
            }
        )
        for emb in self.lookup.values():
            nn.init.zeros_(emb.weight)
        self.direct_lookup = nn.ModuleDict(
            {str(o): nn.Embedding(cfg.vocab_size, V) for o in cfg.direct_lookup_offsets}
        )
        for emb in self.direct_lookup.values():
            nn.init.zeros_(emb.weight)
        # background-knowledge schema rules (pil.schemas.SchemaBank), attached by the learner
        self.schema_bank = None
        self.to(cfg.device)

    def attach_schemas(self, bank: nn.Module) -> None:
        """Adopt a SchemaBank (weights zero-init: decode-neutral until SGD earns them)."""
        self.schema_bank = bank.to(self.U.device)

    # -- structure ----------------------------------------------------------
    @property
    def n_rules(self) -> int:
        return sum(s.n_rules for s in self.strata)

    def all_ids(self) -> list[int]:
        return [i for ids in self.rule_ids for i in ids]

    def add_stratum(self) -> int:
        """New deep stratum reading every currently-existing rule; returns its index."""
        self.strata.append(StratumDeep(self.cfg, self.all_ids()))
        self.heads.append(nn.Parameter(torch.zeros(0, self.cfg.frame_dim, device=self.U.device)))
        self.rule_ids.append([])
        return len(self.strata) - 1

    def _new_ids(self, n: int) -> list[int]:
        ids = list(range(self._next_id, self._next_id + n))
        self._next_id += n
        return ids

    def add_rules_stratum1(
        self, anchors: Tensor, rel: Tensor, sgn: Tensor,
        thr: Tensor | None = None, is_thresh: Tensor | None = None,
        is_le: Tensor | None = None,
    ) -> list[int]:
        """Append stratum-1 rules (zero heads: decode-neutral at birth). Returns their ids."""
        s1: Stratum1 = self.strata[0]
        s1.add_rules(anchors, rel, sgn, thr, is_thresh, is_le)
        n = anchors.shape[0]
        self.heads[0] = nn.Parameter(
            torch.cat(
                [self.heads[0].data, torch.zeros(n, self.cfg.frame_dim, device=self.U.device)]
            )
        )
        ids = self._new_ids(n)
        self.rule_ids[0].extend(ids)
        return ids

    def add_rules_deep(
        self, stratum: int, rel: Tensor, sgn: Tensor,
        thr: Tensor | None = None, is_thresh: Tensor | None = None,
    ) -> list[int]:
        sd: StratumDeep = self.strata[stratum]
        if rel.shape[1] != sd.in_ids.shape[0]:
            raise ValueError(f"literal width {rel.shape[1]} != stratum atoms {sd.in_ids.shape[0]}")
        sd.add_rules(rel, sgn, thr, is_thresh)
        n = rel.shape[0]
        self.heads[stratum] = nn.Parameter(
            torch.cat(
                [
                    self.heads[stratum].data,
                    torch.zeros(n, self.cfg.frame_dim, device=self.U.device),
                ]
            )
        )
        ids = self._new_ids(n)
        self.rule_ids[stratum].extend(ids)
        return ids

    def prune(self, dead: set[int]) -> None:
        """Remove rules by stable id everywhere (heads, strata, downstream literal columns)."""
        if not dead:
            return
        for si, stratum in enumerate(self.strata):
            ids = self.rule_ids[si]
            mask = torch.tensor([i not in dead for i in ids], dtype=torch.bool)
            if not bool(mask.all()):
                stratum.keep(mask)
                self.heads[si] = nn.Parameter(self.heads[si].data[mask])
                self.rule_ids[si] = [i for i in ids if i not in dead]
            if isinstance(stratum, StratumDeep):
                stratum.drop_atoms(dead)

    # -- evaluation ---------------------------------------------------------
    def all_gates(self, x: Tensor, hard: bool = False) -> list[Tensor]:
        """Per-stratum gate tensors ``(B, K_l)``. Deep strata gather atoms by stable id."""
        gates: list[Tensor] = []
        by_id: dict[int, Tensor] = {}
        for si, stratum in enumerate(self.strata):
            if isinstance(stratum, Stratum1):
                g = stratum.gates(x, self.beta, hard=hard)
            else:
                if stratum.in_ids.numel():
                    atoms = torch.stack([by_id[int(i)] for i in stratum.in_ids], dim=1)
                else:
                    atoms = x.new_zeros(x.shape[0], 0, dtype=self.U.dtype)
                g = stratum.gates(atoms, self.beta, hard=hard)
            gates.append(g)
            for j, rid in enumerate(self.rule_ids[si]):
                by_id[rid] = g[:, j]
        return gates

    def forward(self, x: Tensor, hard: bool = False) -> dict[str, Tensor]:
        """``x`` (B, W) context token ids -> logits over the candidate set.

        ``L = (sum_k g_k a_k) U^T + b`` -- PIC's encoder + frame read-out (PIC_SPEC §4.3).
        ``hard=True`` evaluates the rounded Boolean program (the Datalog semantics).
        """
        gates = self.all_gates(x, hard=hard)
        G = torch.cat(gates, dim=1) if self.n_rules else x.new_zeros(x.shape[0], 0).float()
        A = torch.cat([h for h in self.heads], dim=0)
        r = G @ A                                            # (B, d)  the rules' residual write
        for o, emb in self.lookup.items():
            r = r + emb(x[:, int(o)])                        # lookup families: identity gates
        L = r @ self.U.T + self.bias                         # (B, V)
        for o, emb in self.direct_lookup.items():
            L = L + emb(x[:, int(o)])                        # gram2d-style direct tables
        if self.schema_bank is not None and len(self.schema_bank):
            L = L + self.schema_bank.logits(x, self.candidate_ids, L.shape[-1])
        return {"gates": G, "residual": r, "logits": L, "per_stratum": gates}

    # -- candidate mapping ----------------------------------------------------
    def target_index(self, token_ids: Tensor) -> Tensor:
        """Map vocabulary token ids to candidate-set indices (must be present)."""
        token_ids = token_ids.to(self.candidate_ids.device)
        idx = torch.searchsorted(self.candidate_ids, token_ids)
        ok = self.candidate_ids[idx.clamp(max=len(self.candidate_ids) - 1)] == token_ids
        if not bool(ok.all()):
            missing = token_ids[~ok][:5].tolist()
            raise KeyError(f"target token ids {missing} not in the candidate set")
        return idx

    # -- exact per-rule accounting (additivity) -------------------------------
    @torch.no_grad()
    def head_incidences(self) -> Tensor:
        """The clause-weight matrix ``c_k(v) = <a_k, U_v>`` (K, V) -- PIC's ▷, materialized."""
        A = torch.cat([h for h in self.heads], dim=0)
        return A @ self.U.T

    # -- hardening -------------------------------------------------------------
    @torch.no_grad()
    def hardness(self) -> float:
        """Mean binary gap of (rho, s): 0 = fully crisp Boolean structure."""
        gaps = []
        for stratum in self.strata:
            for p in (stratum.rel, stratum.sgn):
                if p.numel():
                    q = torch.sigmoid(p)
                    gaps.append((q * (1 - q)).mean().item())
        return float(sum(gaps) / max(len(gaps), 1))

    @torch.no_grad()
    def round_structure(self) -> None:
        """Snap structure to Boolean-decisive values (soft eval == hard eval).

        Relevance/sign logits go to ±DECISIVE; thresholds go to ``ceil(θ) − 0.5`` (the hard
        integer-count test ``≥ ceil(θ)`` is unchanged, and the soft σ sits β/2 from the
        boundary on either side); the learner raises ``beta`` so σ is numerically Boolean.
        """
        for stratum in self.strata:
            stratum.rel.data = torch.where(
                stratum.rel > 0, torch.full_like(stratum.rel, DECISIVE),
                torch.full_like(stratum.rel, -DECISIVE),
            )
            stratum.sgn.data = torch.where(
                stratum.sgn > 0, torch.full_like(stratum.sgn, DECISIVE),
                torch.full_like(stratum.sgn, -DECISIVE),
            )
            stratum.thr.data = torch.ceil(stratum.thr.data - 1e-6) - 0.5
        self.beta = 24.0

    def structural_terms(self) -> tuple[Tensor, Tensor]:
        """(rho_l1, binary_gap): body-sparsity mass and distance-from-Boolean, for the loss."""
        dev = self.U.device
        rho_l1 = torch.zeros((), device=dev)
        gap = torch.zeros((), device=dev)
        for stratum in self.strata:
            if stratum.rel.numel():
                rho = torch.sigmoid(stratum.rel)
                s = torch.sigmoid(stratum.sgn)
                rho_l1 = rho_l1 + rho.sum()
                gap = gap + (rho * (1 - rho)).sum() + (rho * s * (1 - s)).sum()
        return rho_l1, gap


@torch.no_grad()
def certified_fraction(prog: RuleProgram, x: Tensor, delta: float, hard: bool = True) -> float:
    """PIC §5.5 margin certificate, applied to the learned program.

    ``decode_margin_certified`` (kernel-proved, threshold tight): if every per-token logit
    perturbation is bounded by ``delta`` and the decision margin exceeds ``2*delta``, the
    argmax cannot flip. Returns the fraction of instances whose decode is certified robust
    -- e.g. a bound on how far exported clause weights may be quantized/pruned per token
    without changing any certified decision.
    """
    L = prog.forward(x, hard=hard)["logits"]
    top2 = L.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    return float((margin > 2 * delta).float().mean().item())


def program_summary(prog: RuleProgram) -> dict:
    active = [
        int((torch.sigmoid(s.rel) > 0.5).sum().item()) if s.rel.numel() else 0
        for s in prog.strata
    ]
    return {
        "n_rules": prog.n_rules,
        "rules_per_stratum": [s.n_rules for s in prog.strata],
        "active_literals_per_stratum": active,
        "mean_binary_gap": prog.hardness(),
    }
