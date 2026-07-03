"""Certified P3: the trust-region frame step (objective + enforcement + certification).

One P3 phase widens margins on visited contexts and pushes ``FP(U)`` toward the Welch floor
**without ever flipping a visited teacher decision**. The T-traj discharge (PR #5,
``MARGIN_WIDENING_PROPOSAL.md`` §10) showed the last clause cannot be left to the optimizer:
unconstrained AdamW flips essentially every tracked decision at entry margins, the failure is
margin-limited, and the a-priori T-traj budget is vacuous without clipping. So this module
makes all three parts explicit:

* **objective** — unchanged, the shipped ``raw`` knee (computed by the caller);
* **enforcement (a)** — *exact* inviolability: after each candidate step, no visited decision
  may flip (bank argmax unchanged w.r.t. the frozen ``t_x``), checked exactly;
* **certification (b)** — the per-step ``step_decode_preserved`` premise with headroom:
  ``2·(ρ·ε(α) + β(α)) ≤ η · m_protect``  (theorem: i-orca ``examples/pic_learn/PIC_Learn.thy``,
  main ``735973f``).

The two mechanisms are kept distinct everywhere: (a) is decisive and makes "inviolable" true;
(b) is what is *provable without evaluating* — the certified fraction is the serving-grade
number picard P4 exports. (a) without (b) = safe but uncertified; (b) implies (a) on the
certified subset.

Mechanism: backtracking on the step scale α. Take the full AdamW step + row renorm, then test
``U(α) = normalize(U⁰ + α(U¹ − U⁰))``, ``b(α) = b⁰ + α(b¹ − b⁰)`` at α = 1, ½, ¼, … until both
(a) and (b) pass. α → 0 reproduces ``(U⁰, b⁰)`` exactly (already normalized), δ → 0, no flips —
termination is guaranteed. **Optimizer moments are kept as updated by the full step**
(standard clipped-step practice): the clip rescales the applied parameter change only.

Off by default: nothing in the learner calls this unless a ``CertifiedP3`` is constructed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .geometry import margin_to_worst, normalize_rows


class CertifiedP3:
    """Trust-region wrapper around AdamW-step + ``normalize_U`` for one P3 phase.

    **The two guarantees, distinct by design:**

    * **Enforcement (a)** — *exact* and always on while the mode is active: no visited
      decision flips across an accepted step (bank argmax checked against the frozen
      ``t_x``, evaluated, not bounded). Non-probabilistic; what makes "inviolable" true.
    * **Certification (b)** — the per-step ``step_decode_preserved`` premise with headroom:
      what is *provable without evaluating*. ``certified_frac`` is the **serving-grade
      number** (what picard P4 exports); (a) without (b) is safe but uncertified, (b)
      implies (a) on the certified subset.

    ``sources``: (B, J, dim) visited bank (teacher-anchored: fixed forever); ``t_frozen``:
    (B,) decisions frozen at dump/entry time. ``eta``: certificate headroom (logged in
    ``summary()`` so runs record how conservative the certificate was). ``m_floor``:
    protect-set floor (contexts with ``m ≤ m_floor`` are silent for (b) but still covered
    by (a)). ``exact_floor``: if not ``None``, contexts with ``m ≤ exact_floor`` are excluded
    from the exact check too (the separately-flagged variant) — their flips are *counted and
    reported*, never silently allowed.
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        sources: Tensor,
        t_frozen: Tensor,
        eta: float = 0.9,
        m_floor: float = 0.0,
        exact_floor: float | None = None,
        alpha_min: float = 1.0 / 1024,
    ):
        self.model = model
        self.opt = optimizer
        self.eta = eta
        self.m_floor = m_floor
        self.exact_floor = exact_floor
        self.alpha_min = alpha_min

        self.sources = sources
        self.t = t_frozen.clone()
        self.group = torch.zeros(sources.shape[0], dtype=torch.long)   # T-mass support groups
        self._recompute_rho()
        self.budget = 0.0                       # Σ_s δ_s of ACCEPTED steps (the T-traj budget)
        self.n_steps = 0
        self.records: list[dict] = []
        self.flips_on_excluded = 0              # exact_floor variant: counted, never silent
        with torch.no_grad():
            self.m0 = self._margins().clone()

    # -- bank -----------------------------------------------------------------
    def _recompute_rho(self) -> None:
        self.rho_i = self.sources.sum(dim=-2).norm(dim=-1)
        self.rho = float(self.rho_i.max())

    def add_contexts(self, sources: Tensor, t_frozen: Tensor) -> None:
        """Grow the visited bank mid-phase (P2-shaped). New contexts join the protect set;
        ρ is recomputed (it may grow — the budget's ρ is per-step, so earlier records keep
        the ρ they were measured under). Support group id increments (T-mass tracking)."""
        gid = int(self.group.max()) + 1 if self.group.numel() else 0
        self.sources = torch.cat([self.sources, sources], dim=0)
        self.t = torch.cat([self.t, t_frozen])
        self.group = torch.cat([self.group, torch.full((sources.shape[0],), gid)])
        self._recompute_rho()
        with torch.no_grad():
            self.m0 = torch.cat([self.m0, self._margins()[-sources.shape[0]:]])

    @torch.no_grad()
    def _margins(self) -> Tensor:
        L = self.model.forward(self.sources, target=None)["logits"]
        return margin_to_worst(L, self.t)

    @torch.no_grad()
    def _decided(self) -> Tensor:
        L = self.model.forward(self.sources, target=None)["logits"]
        return L.argmax(dim=-1) == self.t

    # -- the certified step -----------------------------------------------------
    def step(self, loss: Tensor) -> dict:
        """Backward + AdamW + renorm, then backtrack α until (a) and (b) both hold."""
        model = self.model
        with torch.no_grad():
            m_prev = self._margins()
            decided_prev = self._decided()
            protect = m_prev > self.m_floor
            m_protect = float(m_prev[protect].min()) if bool(protect.any()) else float("inf")
            enforce = decided_prev.clone()
            if self.exact_floor is not None:
                enforce &= m_prev > self.exact_floor
            U0 = model.U.detach().clone()
            b0 = model.bias.detach().clone()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        model.normalize_U()                     # moments keep the FULL step (clip is applied-change only)

        with torch.no_grad():
            U1 = model.U.detach().clone()
            b1 = model.bias.detach().clone()

            alpha, tries = 1.0, 0
            while True:
                if alpha < self.alpha_min:      # exact restore: δ = 0, no flips, guaranteed accept
                    alpha = 0.0
                    model.U.data.copy_(U0)
                    model.bias.data.copy_(b0)
                    eps = beta = delta = 0.0
                    decided_now = decided_prev
                    m_now = m_prev
                    break
                model.U.data.copy_(normalize_rows(U0 + alpha * (U1 - U0)))
                model.bias.data.copy_(b0 + alpha * (b1 - b0))
                eps = float((model.U.data - U0).norm(dim=-1).max())
                beta = float((model.bias.data - b0).abs().max())
                delta = self.rho * eps + beta
                L = model.forward(self.sources, target=None)["logits"]
                decided_now = L.argmax(dim=-1) == self.t
                m_now = margin_to_worst(L, self.t)
                cond_a = bool((enforce & ~decided_now).sum() == 0)     # exact inviolability
                cond_b = 2.0 * delta <= self.eta * m_protect           # certificate headroom
                tries += 1
                if cond_a and cond_b:
                    break
                alpha *= 0.5

            if self.exact_floor is not None:
                excl = decided_prev & ~enforce
                self.flips_on_excluded += int((excl & ~decided_now).sum())

            certified = m_prev > 2.0 * delta    # per-context step_decode_preserved premise
            self.budget += delta
            self.n_steps += 1
            rec = {
                "step": self.n_steps,
                "alpha": alpha,
                "clipped": alpha < 1.0,
                "tries": tries,
                "eps": eps,
                "beta": beta,
                "delta": delta,
                "enforced_flips": int((enforce & ~decided_now).sum()),   # must be 0
                "flips_on_excluded_total": self.flips_on_excluded,
                "certified_frac": float(certified.float().mean()),
                "premise_hold_rate": float((m_prev > 2.0 * delta).float().mean()),
                "protect_n": int(protect.sum()),
                "m_protect": m_protect,
                "budget_2sum": 2.0 * self.budget,
                "margin_p10": float(m_now.sort().values[int(0.10 * (len(m_now) - 1))]),
                "margin_p50": float(m_now.median()),
                "decided_rate": float(decided_now.float().mean()),
            }
            self.records.append(rec)
        return rec

    # -- T-mass (discharge experiment 5 data) ------------------------------------
    @torch.no_grad()
    def certified_mass(self, delta_ref: float) -> dict[int, int]:
        """Per support group: ``#{x in group : m_x > 2 δ_ref}`` at the current frame."""
        m = self._margins()
        cert = m > 2.0 * delta_ref
        return {int(g): int(cert[self.group == g].sum()) for g in self.group.unique()}

    def _budget_alive(self) -> dict:
        pos = self.m0[self.m0 > 0]
        if pos.numel() == 0:
            return {"m0_pos_n": 0, "budget_alive_pos_p10": False, "budget_alive_pos_p50": False}
        q = pos.sort().values
        p10 = float(q[int(0.10 * (len(q) - 1))])
        p50 = float(pos.median())
        return {
            "m0_pos_n": int(pos.numel()),
            "m0_pos_p10": p10,
            "m0_pos_p50": p50,
            "budget_alive_pos_p10": 2.0 * self.budget < p10,
            "budget_alive_pos_p50": 2.0 * self.budget < p50,
        }

    @torch.no_grad()
    def summary(self) -> dict:
        if not self.records:
            return {"n_steps": 0}
        alphas = torch.tensor([r["alpha"] for r in self.records])
        cf = torch.tensor([r["certified_frac"] for r in self.records])
        m_now = self._margins()
        return {
            "n_steps": self.n_steps,
            "rho": self.rho,
            "eta": self.eta,                       # effective headroom used (review item)
            "m_floor": self.m_floor,
            "exact_floor": self.exact_floor,
            "enforced_flips_total": int(sum(r["enforced_flips"] for r in self.records)),
            "flips_on_excluded_total": self.flips_on_excluded,
            "clip_frac": float((alphas < 1.0).float().mean()),
            "alpha_mean": float(alphas.mean()),
            "alpha_min_seen": float(alphas.min()),
            "certified_frac_mean": float(cf.mean()),
            "certified_frac_final": float(cf[-1]),
            "budget_2sum": 2.0 * self.budget,
            "m0_p10": float(self.m0.sort().values[int(0.10 * (len(self.m0) - 1))]),
            "m0_p50": float(self.m0.median()),
            # the budget-alive question is only meaningful on entry-positive margins
            # (the protect-shaped subset); negative m0 makes it vacuous trivially
            **self._budget_alive(),
            "final_margin_p10": float(m_now.sort().values[int(0.10 * (len(m_now) - 1))]),
            "final_margin_p50": float(m_now.median()),
            "final_decided_rate": float(self._decided().float().mean()),
        }
