"""The PIC rule learner: Generate -> Gate -> Refine over *program structure*.

Learns a :class:`pil.rules.RuleProgram` from (context, next-token) data, starting from
random rules. The three algorithmic pieces the design note (docs/notes/pic_rule_learner.md)
names:

  (i)   **initial random states** -- random-in-data-measure rule seeding: bodies are sampled
        from observed contexts (so newborn rules are alive under a Zipfian vocabulary),
        arity ~ 1 + Geometric, offsets recency-biased, heads zero (decode-neutral birth);
  (ii)  **incidence backprop** -- autograd through the soft semiring (t-norm gates, softmax
        ⊕_1 decode) on NLL + margin hinge, with an annealed structural penalty that drives
        relevance/sign parameters to Boolean;
  (iii) **structural edits from feedback** -- birth by boosting (seed on the loss residual),
        death by *exact* per-rule ablation (additivity makes this O(K) per probe batch),
        specialization of high-ambiguity rules (info-gain literal, ID3-style), and
        deepening (a new stratum over rule firings) when the shallow residual plateaus.

The endpoint of :meth:`PICRuleLearner.fit` is a *hardened* program: structure rounded to
Boolean, heads refit under the hard gates -- the exact program the Datalog export emits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from .geometry import margin_to_worst
from .rules import (
    ABSENT,
    PRESENT,
    RuleProgram,
    Stratum1,
    StratumDeep,
    certified_fraction,
    program_summary,
)


@dataclass
class RuleLearnerConfig:
    # refine (ii)
    lr: float = 3e-2
    frame_lr: float = 1e-2
    direct_lookup_lr: float = 5e-3       # full-rank tables: lower lr + decay (they memorize)
    direct_lookup_wd: float = 1e-4
    hard_target_weight: float = 0.0      # under soft distillation, extra hard-CE on the
    #                                      teacher argmax (distribution fit trades against
    #                                      argmax agreement; mix when agreement is the metric)
    batch_size: int = 256
    epochs_per_phase: int = 40
    margin_weight: float = 0.3
    margin_target: float = 2.0
    rho_l1: float = 3e-4                 # body-sparsity pressure (per literal)
    gap_max: float = 3e-3                # final binary-gap pressure (annealed 0 -> this)
    # structure (i, iii)
    n_phases: int = 6
    init_rules: int = 64
    births_per_phase: int = 32
    arity_geom_p: float = 0.5            # arity ~ 1 + Geometric(p), clipped to window
    recency_gamma: float = 0.7           # offset sampling weight gamma^(dist from decode)
    eq_seed_p: float = 0.4               # P(seed one true equality literal) when eq atoms exist
    le_seed_p: float = 0.0               # P(a token literal is ordinal x<=anchor); for ordered
    #                                      token encodings (tabular quantile bins), not text
    max_rules: int = 1024
    thresh_fraction: float = 0.35        # fraction of births that are PIC-T3 threshold rules
    deep_thresh_arity: int = 16          # atoms a deep threshold rule reads at birth
    beta_start: float = 4.0              # threshold-gate slope anneal (soft -> crisp)
    beta_end: float = 12.0
    protect_age_phases: int = 1
    death_nll_gain: float = 1e-4         # prune if removing the rule costs less than this
    min_fire_rate: float = 1e-4
    ambiguity_split_threshold: float = 0.55   # split when top-target purity is below this
    max_splits_per_phase: int = 8
    deepen_patience: int = 2             # phases without val improvement before deepening
    solved_val_acc: float = 0.995        # past this, stop structural edits (only refine/harden)
    death_on_val: bool = False           # measure death-ablation utility on the val slice
    #                                      (memorizing rules look useful on train; on val
    #                                      they don't -- the honest pruning signal for LM)
    max_strata: int = 3
    deep_births: int = 32
    deep_arity: int = 2
    # bookkeeping
    probe_size: int = 2048
    val_fraction: float = 0.15
    seed: int = 0
    verbose: bool = True


@dataclass
class FitReport:
    history: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)

    def log(self, **row) -> None:
        self.history.append(row)


@torch.no_grad()
def init_direct_lookup_from_counts(
    prog: RuleProgram, X: Tensor, y_idx: Tensor, alpha: float = 0.5
) -> None:
    """Seed direct lookup tables from train co-occurrence counts (rosetta's gram mining,
    as an initializer): row(c) = centered smoothed log P(target | token c at offset o).
    SGD refines from a robust statistical estimate instead of memorizing from zero."""
    V = prog.candidate_ids.shape[0]
    for o, emb in prog.direct_lookup.items():
        o = int(o)
        toks = X[:, o]
        uniq, inv = toks.unique(return_inverse=True)
        counts = torch.zeros(uniq.shape[0], V)
        counts.index_put_((inv, y_idx), torch.ones(len(y_idx)), accumulate=True)
        logp = (counts + alpha).log() - (counts.sum(1, keepdim=True) + alpha * V).log()
        emb.weight[uniq] = logp - logp.mean(dim=1, keepdim=True)


class PICRuleLearner:
    """Trains a RuleProgram from (X, y): X (N, W) context token ids, y (N,) target ids."""

    def __init__(self, program: RuleProgram, cfg: RuleLearnerConfig | None = None):
        self.prog = program
        self.cfg = cfg or RuleLearnerConfig()
        self.gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        self.birth_phase: dict[int, int] = {}    # rule id -> phase born (age protection)

    # ------------------------------------------------------------------ (i) init
    def _sample_bodies(
        self, X: Tensor, n: int, rows: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Random-in-data-measure bodies: anchors from observed contexts.

        A ``thresh_fraction`` of the sampled rules are PIC-T3 threshold rules: they read the
        *whole* window against their anchor pattern and fire on ``#matches ≥ θ`` (θ random) --
        the additive ⊗-monomial behind a margin turnstile, the gate family AND-rules can't
        express compactly (parity/majority live here).

        When the program has equality atoms, AND rules also seed (with prob ``eq_seed_p``)
        one equality literal that is *true on the seed context* -- the relational channel.
        """
        s1: Stratum1 = self.prog.strata[0]
        W, L = self.prog.cfg.window, s1.n_lits
        if rows is None:
            rows = torch.randint(0, X.shape[0], (n,), generator=self.gen)
        anchors = X[rows].clone()                                  # (n, W)
        # arity ~ 1 + Geometric(p), clipped
        u = torch.rand(n, generator=self.gen)
        arity = (1 + torch.log1p(-u) / math.log(1 - self.cfg.arity_geom_p)).long().clamp(1, W)
        # recency-biased offsets: weight gamma^(W-1-o) favours offsets near the decode slot
        w = self.cfg.recency_gamma ** torch.arange(W - 1, -1, -1, dtype=torch.float)
        rel = torch.full((n, L), ABSENT)
        for i in range(n):
            k = int(arity[i])
            offs = torch.multinomial(w, k, replacement=False, generator=self.gen)
            rel[i, offs] = PRESENT
        sgn = torch.full((n, L), PRESENT)
        is_thresh = torch.rand(n, generator=self.gen) < self.cfg.thresh_fraction
        rel[is_thresh, :W] = PRESENT                     # threshold rules read the window densely
        rel[is_thresh, W:] = ABSENT
        thr = torch.where(
            is_thresh,
            0.5 + torch.rand(n, generator=self.gen) * (W - 1),
            torch.zeros(n),
        )
        P = s1.eq_pairs.shape[0]
        if P:
            eq_true = anchors[:, s1.eq_pairs[:, 0]] == anchors[:, s1.eq_pairs[:, 1]]  # (n,P)
            for i in range(n):
                if bool(is_thresh[i]) or float(torch.rand(1, generator=self.gen)) > self.cfg.eq_seed_p:
                    continue
                idx = torch.nonzero(eq_true[i]).flatten()
                if idx.numel():
                    p = idx[int(torch.randint(0, idx.numel(), (1,), generator=self.gen))]
                    rel[i, W + p] = PRESENT
        # ordinal literals: x[o] <= observed anchor (true on the seed context by construction)
        is_le = torch.rand(n, W, generator=self.gen) < self.cfg.le_seed_p
        return anchors, rel, sgn, thr, is_thresh, is_le

    def seed_rules(self, X: Tensor, n: int) -> list[int]:
        return self.prog.add_rules_stratum1(*self._sample_bodies(X, n))

    def seed_rules_from_residual(self, X: Tensor, y: Tensor, n: int) -> list[int]:
        """Boosting birth: sample seed contexts from the worst-loss examples."""
        with torch.no_grad():
            L = self._logits_in_chunks(X)
            nll = F.cross_entropy(L, y, reduction="none")
        k = min(4 * n, X.shape[0])
        worst = nll.topk(k).indices
        pick = worst[torch.randint(0, k, (n,), generator=self.gen)]
        return self.prog.add_rules_stratum1(*self._sample_bodies(X, n, rows=pick))

    def seed_deep_rules(self, stratum: int, n: int) -> list[int]:
        """Random signed combinations of existing atoms (the composition proposer).

        AND births are sparse (arity ``deep_arity``); threshold births read up to
        ``deep_thresh_arity`` atoms with random signs and a random integer-ish threshold.
        """
        sd: StratumDeep = self.prog.strata[stratum]
        A = sd.in_ids.shape[0]
        if A == 0:
            return []
        rel = torch.full((n, A), ABSENT)
        sgn = torch.where(
            torch.rand(n, A, generator=self.gen) > 0.5,
            torch.tensor(PRESENT), torch.tensor(-PRESENT),
        )
        is_thresh = torch.rand(n, generator=self.gen) < self.cfg.thresh_fraction
        thr = torch.zeros(n)
        for i in range(n):
            if bool(is_thresh[i]):
                k = min(self.cfg.deep_thresh_arity, A)
                thr[i] = 0.5 + torch.rand(1, generator=self.gen).item() * (k - 1)
            else:
                k = min(self.cfg.deep_arity, A)
            offs = torch.randperm(A, generator=self.gen)[:k]
            rel[i, offs] = PRESENT
        return self.prog.add_rules_deep(stratum, rel, sgn, thr, is_thresh)

    # ------------------------------------------------------------------ (ii) refine
    def loss(
        self,
        out: dict,
        tgt: Tensor,
        gap_weight: float,
        soft: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """``soft`` = (idx (B,K) candidate indices with -1 pad, p (B,K) teacher probs):
        distributional distillation -- CE against the teacher's top-K instead of the
        hard argmax. The margin hinge stays on the hard target either way."""
        L = out["logits"]
        if soft is not None:
            idx, p = soft
            lp = torch.log_softmax(L, dim=-1)
            gathered = lp.gather(-1, idx.clamp_min(0))
            nll = -(p * gathered).sum(dim=-1).mean()
            if self.cfg.hard_target_weight > 0:
                nll = nll + self.cfg.hard_target_weight * F.cross_entropy(L, tgt)
        else:
            nll = F.cross_entropy(L, tgt)
        m = margin_to_worst(L, tgt)
        hinge = torch.relu(self.cfg.margin_target - m).mean()
        rho_l1, gap = self.prog.structural_terms()
        return (
            nll
            + self.cfg.margin_weight * hinge
            + self.cfg.rho_l1 * rho_l1
            + gap_weight * gap
        )

    def _optimizer(self) -> torch.optim.Optimizer:
        frame = {"params": [self.prog.U, self.prog.bias], "lr": self.cfg.frame_lr}
        direct = {
            "params": [p for n, p in self.prog.named_parameters() if "direct_lookup" in n],
            "lr": self.cfg.direct_lookup_lr,
            "weight_decay": self.cfg.direct_lookup_wd,
        }
        rest = {
            "params": [
                p
                for n, p in self.prog.named_parameters()
                if p.requires_grad and n not in ("U", "bias") and "direct_lookup" not in n
            ],
            "lr": self.cfg.lr,
        }
        return torch.optim.Adam([frame, direct, rest])

    def refine(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int,
        gap_weight: float,
        soft: tuple[Tensor, Tensor] | None = None,
    ) -> float:
        opt = self._optimizer()
        N = X.shape[0]
        last = float("nan")
        for _ in range(epochs):
            perm = torch.randperm(N, generator=self.gen)
            for i in range(0, N, self.cfg.batch_size):
                idx = perm[i : i + self.cfg.batch_size]
                out = self.prog(X[idx])
                sb = (soft[0][idx], soft[1][idx]) if soft is not None else None
                loss = self.loss(out, y[idx], gap_weight, soft=sb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = float(loss.item())
        return last

    # ------------------------------------------------------------------ feedback
    @torch.no_grad()
    def _logits_in_chunks(self, X: Tensor, hard: bool = False, chunk: int = 4096) -> Tensor:
        outs = [self.prog(X[i : i + chunk], hard=hard)["logits"] for i in range(0, X.shape[0], chunk)]
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def feedback(self, X: Tensor, y: Tensor) -> dict[str, Tensor]:
        """The per-rule sufficient statistics for structural edits (all exact).

        utility  = mean NLL increase when the rule is ablated (additivity: L - g_k c_k)
        fire     = mean gate value
        purity   = max target frequency among fired examples (1 - ambiguity)
        """
        n = min(self.cfg.probe_size, X.shape[0])
        Xp, yp = X[:n], y[:n]
        out = self.prog(Xp)
        G, L = out["gates"], out["logits"]                    # (B,K), (B,V)
        C = self.prog.head_incidences()                       # (K,V)
        base_nll = F.cross_entropy(L, yp, reduction="none")   # (B,)

        K = G.shape[1]
        utility = torch.zeros(K)
        step = max(1, int(32_000_000 // max(n * L.shape[1], 1)))   # ~32M floats per chunk
        for k0 in range(0, K, step):
            k1 = min(k0 + step, K)
            # ablated logits for rules k0..k1: (B, kk, V)
            La = L.unsqueeze(1) - G[:, k0:k1].unsqueeze(-1) * C[k0:k1].unsqueeze(0)
            nll_a = F.cross_entropy(
                La.reshape(-1, La.shape[-1]),
                yp.repeat_interleave(k1 - k0),
                reduction="none",
            ).reshape(n, k1 - k0)
            utility[k0:k1] = (nll_a - base_nll.unsqueeze(1)).mean(dim=0)

        fired = G > 0.5
        fire = fired.float().mean(dim=0)
        purity = torch.zeros(K)
        for k in range(K):
            tk = yp[fired[:, k]]
            if tk.numel():
                purity[k] = tk.bincount().max().float() / tk.numel()
        return {"utility": utility, "fire": fire, "purity": purity, "gates": G}

    @torch.no_grad()
    def _protected_ids(self) -> set[int]:
        """Rules actively read by some deep literal must not be pruned."""
        protected: set[int] = set()
        for stratum in self.prog.strata:
            if isinstance(stratum, StratumDeep) and stratum.rel.numel():
                active = (torch.sigmoid(stratum.rel) > 0.5).any(dim=0)
                protected |= {int(i) for i in stratum.in_ids[active]}
        return protected

    # ------------------------------------------------------------------ (iii) edits
    @torch.no_grad()
    def deaths(self, stats: dict[str, Tensor], phase: int) -> set[int]:
        ids = self.prog.all_ids()
        protected = self._protected_ids()
        dead: set[int] = set()
        for k, rid in enumerate(ids):
            if rid in protected:
                continue
            if phase - self.birth_phase.get(rid, 0) < self.cfg.protect_age_phases:
                continue
            useless = stats["utility"][k] < self.cfg.death_nll_gain
            silent = stats["fire"][k] < self.cfg.min_fire_rate
            if useless or silent:
                dead.add(rid)
        return dead

    def _register_birth(self, ids: list[int], phase: int) -> list[int]:
        for rid in ids:
            self.birth_phase[rid] = phase
        return ids

    def _extend_downstream(self, from_stratum: int, new_ids: list[int]) -> None:
        """Expose newly-born atoms to every *later* deep stratum (init: absent literals)."""
        for si in range(from_stratum + 1, len(self.prog.strata)):
            self.prog.strata[si].extend_atoms(new_ids)

    @torch.no_grad()
    def specialize(self, X: Tensor, y: Tensor, stats: dict[str, Tensor], phase: int) -> int:
        """Clone high-ambiguity stratum-1 rules with one info-gain literal added."""
        s1: Stratum1 = self.prog.strata[0]
        K1 = s1.n_rules
        n = min(self.cfg.probe_size, X.shape[0])
        Xp, yp = X[:n], y[:n]
        G = stats["gates"][:, :K1]
        made = 0
        order = stats["purity"][:K1].argsort()
        for k in order.tolist():
            if made >= self.cfg.max_splits_per_phase:
                break
            if stats["purity"][k] > self.cfg.ambiguity_split_threshold:
                break
            if stats["fire"][k] < 10.0 / n:
                continue
            fired = G[:, k] > 0.5
            Xf, yf = Xp[fired], yp[fired]
            lit, tok, le = self._best_split(Xf, yf, s1, k)
            if lit is None:
                continue
            for sgn_val in (PRESENT, -PRESENT):
                anchors = s1.anchor[k : k + 1].clone()
                is_le = s1.is_le[k : k + 1].clone()
                if lit < s1.cfg.window:
                    anchors[0, lit] = tok
                    is_le[0, lit] = le
                rel = s1.rel.data[k : k + 1].clone()
                rel[0, lit] = PRESENT
                sgn = s1.sgn.data[k : k + 1].clone()
                sgn[0, lit] = sgn_val
                thr = s1.thr.data[k : k + 1].clone()
                is_thr = s1.is_thresh[k : k + 1].clone()
                born = self.prog.add_rules_stratum1(anchors, rel, sgn, thr, is_thr, is_le)
                self._register_birth(born, phase + 1)
                self._extend_downstream(0, born)
            made += 1
        return made

    def _best_split(
        self, Xf: Tensor, yf: Tensor, s1: Stratum1, k: int
    ) -> tuple[int | None, int, bool]:
        """The literal test with maximal information gain on the fired set.

        Candidates: (offset, token) equality tests; ordinal tests ``x[o] <= v`` when
        ``le_seed_p > 0`` (ordered token encodings); and -- when the program has eq
        atoms -- relational tests ``x[o1] == x[o2]`` (literal index ``W + pair``).
        """
        if Xf.shape[0] < 8:
            return None, 0, False

        def entropy(t: Tensor) -> float:
            if t.numel() == 0:
                return 0.0
            p = t.bincount().float()
            p = p[p > 0] / t.numel()
            return float(-(p * p.log()).sum())

        def gain_of(m: Tensor, base: float) -> float:
            c = int(m.sum())
            if c < 4 or c > Xf.shape[0] - 4:
                return -1.0
            return base - (
                m.float().mean().item() * entropy(yf[m])
                + (~m).float().mean().item() * entropy(yf[~m])
            )

        base = entropy(yf)
        active = torch.sigmoid(s1.rel.data[k]) > 0.5
        W = s1.cfg.window
        ordinal = self.cfg.le_seed_p > 0
        best_gain, best = 1e-3, (None, 0, False)
        for off in range(W):
            if bool(active[off]):
                continue                       # already constrained
            vals = Xf[:, off].unique()
            for v in vals.tolist():
                g = gain_of(Xf[:, off] == v, base)
                if g > best_gain:
                    best_gain, best = g, (off, int(v), False)
                if ordinal:
                    g = gain_of(Xf[:, off] <= v, base)
                    if g > best_gain:
                        best_gain, best = g, (off, int(v), True)
        for p in range(s1.eq_pairs.shape[0]):
            if bool(active[W + p]):
                continue
            a, b = int(s1.eq_pairs[p, 0]), int(s1.eq_pairs[p, 1])
            g = gain_of(Xf[:, a] == Xf[:, b], base)
            if g > best_gain:
                best_gain, best = g, (W + p, 0, False)
        return best

    # ------------------------------------------------------------------ the loop
    def make_soft_targets(
        self, topk_token_ids: Tensor, topk_logits: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Teacher top-K (token ids, logits) -> (candidate-index, prob) soft targets.

        Out-of-candidate teacher entries are dropped and each row renormalized over the
        surviving mass (idx = -1, p = 0 padding where dropped)."""
        cands = self.prog.candidate_ids
        pos = torch.searchsorted(cands, topk_token_ids.clamp_min(0))
        pos = pos.clamp(max=cands.shape[0] - 1)
        hit = cands[pos] == topk_token_ids
        p = torch.softmax(topk_logits, dim=-1) * hit
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        idx = torch.where(hit, pos, torch.full_like(pos, -1))
        return idx, p

    def fit(
        self,
        X: Tensor,
        y_tokens: Tensor,
        report: FitReport | None = None,
        soft: tuple[Tensor, Tensor] | None = None,
    ) -> FitReport:
        """X (N, W) contexts; y_tokens (N,) target *vocabulary* token ids.

        ``soft`` (from :meth:`make_soft_targets`) switches the fit to distributional
        distillation; the hard targets still drive margins, feedback, and eval."""
        cfg = self.cfg
        rep = report or FitReport()
        y = self.prog.target_index(y_tokens)

        n_val = max(1, int(X.shape[0] * cfg.val_fraction))
        perm = torch.randperm(X.shape[0], generator=self.gen)
        Xtr, ytr = X[perm[n_val:]], y[perm[n_val:]]
        Xva, yva = X[perm[:n_val]], y[perm[:n_val]]
        soft_tr = (soft[0][perm[n_val:]], soft[1][perm[n_val:]]) if soft is not None else None

        for rid in self.seed_rules(Xtr, cfg.init_rules):
            self.birth_phase[rid] = 0

        best_val, stall = -1.0, 0
        for phase in range(cfg.n_phases):
            t = phase / max(cfg.n_phases - 1, 1)
            gap_w = cfg.gap_max * t**2
            self.prog.beta = cfg.beta_start + (cfg.beta_end - cfg.beta_start) * t
            self.refine(Xtr, ytr, cfg.epochs_per_phase, gap_w, soft=soft_tr)

            stats = self.feedback(Xva, yva) if cfg.death_on_val else self.feedback(Xtr, ytr)
            val_acc = self._acc(Xva, yva)
            hard_acc = self._acc(Xva, yva, hard=True)
            rep.log(
                phase=phase, val_acc=val_acc, hard_val_acc=hard_acc,
                **program_summary(self.prog),
            )
            if cfg.verbose:
                print(
                    f"[phase {phase}] val {val_acc:.3f} hard {hard_acc:.3f} "
                    f"rules {self.prog.n_rules} gap {self.prog.hardness():.4f}"
                )

            if phase == cfg.n_phases - 1:
                break

            # -- structural edits ------------------------------------------
            dead = self.deaths(stats, phase)
            self.prog.prune(dead)

            if val_acc >= cfg.solved_val_acc:
                if cfg.verbose and dead:
                    print(f"          edits: -{len(dead)} pruned (solved; no growth)")
                continue

            n_split = self.specialize(Xtr, ytr, self.feedback(Xtr, ytr), phase)

            if val_acc > best_val + 1e-3:
                best_val, stall = val_acc, 0
            else:
                stall += 1
            if stall >= cfg.deepen_patience and len(self.prog.strata) < cfg.max_strata:
                si = self.prog.add_stratum()
                born = self._register_birth(self.seed_deep_rules(si, cfg.deep_births), phase + 1)
                self._extend_downstream(si, born)
                stall = 0
            elif len(self.prog.strata) > 1:
                # keep proposing compositions in existing deep strata too
                for si in range(1, len(self.prog.strata)):
                    born = self._register_birth(
                        self.seed_deep_rules(si, cfg.deep_births // 2), phase + 1
                    )
                    self._extend_downstream(si, born)

            room = cfg.max_rules - self.prog.n_rules
            if room > 0:
                born = self._register_birth(
                    self.seed_rules_from_residual(Xtr, ytr, min(cfg.births_per_phase, room)),
                    phase + 1,
                )
                self._extend_downstream(0, born)
            if cfg.verbose:
                print(
                    f"          edits: -{len(dead)} pruned, +{n_split} splits, "
                    f"strata {len(self.prog.strata)}"
                )

        # -- harden + refit (the exportable program) ------------------------
        self.prog.round_structure()
        for stratum in self.prog.strata:
            stratum.rel.requires_grad_(False)
            stratum.sgn.requires_grad_(False)
        self.refine(Xtr, ytr, max(cfg.epochs_per_phase // 2, 5), 0.0, soft=soft_tr)

        rep.final = {
            "val_acc": self._acc(Xva, yva),
            "hard_val_acc": self._acc(Xva, yva, hard=True),
            "train_acc": self._acc(Xtr, ytr),
            "hard_train_acc": self._acc(Xtr, ytr, hard=True),
            "soft_hard_agreement": self._agreement(Xva),
            # PIC §5.5: decisions provably robust to per-token clause-weight drift delta
            "certified_frac_delta_0.25": certified_fraction(self.prog, Xva, 0.25),
            **program_summary(self.prog),
        }
        if cfg.verbose:
            print(f"[final] {rep.final}")
        return rep

    # ------------------------------------------------------------------ metrics
    @torch.no_grad()
    def _acc(self, X: Tensor, y: Tensor, hard: bool = False) -> float:
        L = self._logits_in_chunks(X, hard=hard)
        return float((L.argmax(dim=-1) == y).float().mean().item())

    @torch.no_grad()
    def _agreement(self, X: Tensor) -> float:
        a = self._logits_in_chunks(X, hard=False).argmax(dim=-1)
        b = self._logits_in_chunks(X, hard=True).argmax(dim=-1)
        return float((a == b).float().mean().item())
