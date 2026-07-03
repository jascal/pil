"""Runtime discharge instrumentation for T-traj (i-orca ``examples/pic_learn/PIC_Learn.thy``).

T-traj (kernel-proved; merged, i-orca PR #21) certifies that a frame-learning trajectory
preserves visited decisions. Per tracked context with residual ``r`` (``‖r‖ ≤ ρ``) and
decision ``t`` frozen at entry, with margin ``m_t`` *before* a step whose effective update
is bounded by ``ε_t = max_v ‖U^{t+1}_v − U^t_v‖`` and ``β_t = max_v |b^{t+1}_v − b^t_v|``:

  * ``step_decode_preserved``:  ``m_t > 2·(ρ·ε_t + β_t)``  ⟹  the step cannot flip ``t``;
  * ``step_margin_survives``:   the margin survives as ``≥ m_t − 2(ρ·ε_t + β_t)``;
  * ``traj_decode_preserved``:  ``2·Σ_{s<T}(ρ·ε_s + β_s) < m_0``  ⟹  ``t`` survives the
    whole trajectory (a-priori budget).

The theorem is unconditional math; this module measures whether **real optimizer steps**
(AdamW + row renormalization — pil's actual P3) live inside its premises, and with how
much slack. It never changes training: pure observation, off by default.

Usage — the Python-facing face of ``step_decode_preserved``::

    model = ProjectiveIncidenceLearner(cfg); model.normalize_U()
    cert = TrajectoryCertificate(model, bank_sources, held_out_mask=ho)   # freezes t here
    opt = torch.optim.AdamW(model.parameters(), ...)
    for step in range(T):
        loss = ...; opt.zero_grad(); loss.backward(); opt.step(); model.normalize_U()
        rec = cert.step()          # post-norm -> post-norm effective update
        # rec["premise_hold_rate"], rec["new_flips"], rec["viol_premise_flip"] (must be 0)
    print(cert.summary())

or, through the learner hook (zero-cost when ``certifier is None``)::

    learner.training_step(sources, target, optimizer, certifier=cert)

Traps this implementation gets right (per the discharge task):

  * ``ΔU`` is measured **post-normalization to post-normalization** — the certificate
    applies to the ``(U, b)`` actually used to decode, so construct the certificate
    *after* ``normalize_U()`` and call :meth:`step` *after* each step's ``normalize_U()``.
  * ``t`` is **frozen at entry** (the argmax when the context enters the tracked set) —
    whether it matches a ground-truth label is recorded, never re-frozen.
  * ``ρ = max ‖Σ_j d_j‖`` over the tracked bank; its constancy is *asserted* each step,
    not assumed.
  * Contexts with ``m_t ≤ 2δ_t`` are counted as **silent** (the certificate says nothing
    about them), never dropped: the transfer inequality has no sign condition.

All claims produced here are tagged **empirical** over the run at hand.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .geometry import margin_to_worst


class TrajectoryCertificate:
    """Track T-traj premises/conclusions for one model over one optimizer trajectory.

    ``sources``: (B, J, dim) fixed evaluation bank (include held-out contexts;
    ``held_out_mask`` marks them for split reporting). Decisions are frozen at
    construction time. All tensors live on the model's device.
    """

    def __init__(
        self,
        model,
        sources: Tensor,
        held_out_mask: Tensor | None = None,
        true_targets: Tensor | None = None,
        tol: float = 1e-4,
    ):
        self.model = model
        self.sources = sources
        self.B = sources.shape[0]
        self.held_out = (
            held_out_mask.clone() if held_out_mask is not None
            else torch.zeros(self.B, dtype=torch.bool)
        )
        # rho = max ||r||, r = sum_j d_j (constant while the bank is fixed -- asserted).
        # rho_i = per-context ||r_i||: the per-step lemma is per-context, so the tighter
        # premise m_i > 2(rho_i eps + beta) is also tracked (reported separately; the
        # global-rho form is the trajectory-theorem premise and stays primary).
        self._r = sources.sum(dim=-2)
        self.rho_i = self._r.norm(dim=-1)
        self.rho = float(self.rho_i.max())
        self.tol = tol * (1.0 + self.rho)

        with torch.no_grad():
            L = self._logits()
            self.t = L.argmax(dim=-1)                      # frozen at entry
            self.m_prev = margin_to_worst(L, self.t)       # (B,)
        self.m0 = self.m_prev.clone()
        self.entry_matches_truth = (
            float((self.t == true_targets).float().mean()) if true_targets is not None else None
        )

        self._U_prev = model.U.detach().clone()
        self._b_prev = model.bias.detach().clone()
        self.budget = 0.0                                  # Σ_s (ρ ε_s + β_s)
        self.flipped_at = torch.full((self.B,), -1, dtype=torch.long)
        self.budget_out_at = torch.full((self.B,), -1, dtype=torch.long)
        self.n_steps = 0
        self.records: list[dict] = []
        # self-check counters: the theorem forbids both; nonzero = measurement bug
        self.viol_premise_flip = 0
        self.viol_transfer = 0

    @torch.no_grad()
    def _logits(self) -> Tensor:
        return self.model.forward(self.sources, target=None)["logits"]

    @torch.no_grad()
    def step(self) -> dict:
        """Record one effective update (call after ``optimizer.step(); normalize_U()``)."""
        r_now = self.sources.sum(dim=-2)
        assert torch.equal(r_now, self._r), "tracked bank changed: rho constancy violated"

        U, b = self.model.U.detach(), self.model.bias.detach()
        eps = float((U - self._U_prev).norm(dim=-1).max())
        beta = float((b - self._b_prev).abs().max())
        delta = self.rho * eps + beta

        L = self._logits()
        m_now = margin_to_worst(L, self.t)
        decided_now = L.argmax(dim=-1) == self.t

        premise = self.m_prev > 2.0 * delta                # step_decode_preserved's guard
        delta_i = self.rho_i * eps + beta                  # per-context form of the lemma
        premise_i = self.m_prev > 2.0 * delta_i
        flipped_this_step = decided_now.logical_not() & (self.flipped_at < 0)

        # -- theorem self-checks (violations = harness bug, not a finding) ------------
        pf = int((premise & flipped_this_step).sum())
        self.viol_premise_flip += pf
        # step_margin_survives: m_now >= m_prev - 2 delta (up to fp tolerance)
        tv = int((m_now < self.m_prev - 2.0 * delta - self.tol).sum())
        self.viol_transfer += tv

        self.n_steps += 1
        newly_flipped = torch.nonzero(flipped_this_step).flatten()
        self.flipped_at[newly_flipped] = self.n_steps
        self.budget += delta
        # a-priori budget crossing: 2 Σ δ ≥ m0 (bound goes vacuous for that context)
        crossed = (2.0 * self.budget >= self.m0) & (self.budget_out_at < 0)
        self.budget_out_at[torch.nonzero(crossed).flatten()] = self.n_steps

        q = m_now.sort().values
        rec = {
            "step": self.n_steps,
            "eps": eps,
            "beta": beta,
            "delta": delta,
            "beta_share": beta / delta if delta > 0 else 0.0,
            "premise_hold_rate": float(premise.float().mean()),
            "premise_hold_rate_perctx": float(premise_i.float().mean()),
            "silent_rate": float((~premise).float().mean()),
            "decided_rate": float(decided_now.float().mean()),
            "new_flips": int(flipped_this_step.sum()),
            "viol_premise_flip": pf,
            "viol_transfer": tv,
            "margin_min": float(q[0]),
            "margin_p10": float(q[int(0.10 * (self.B - 1))]),
            "margin_p50": float(q[int(0.50 * (self.B - 1))]),
            "budget": self.budget,
        }
        self.records.append(rec)
        self.m_prev = m_now
        self._U_prev = U.clone()
        self._b_prev = b.clone()
        return rec

    @torch.no_grad()
    def summary(self) -> dict:
        """Aggregate verdicts over the recorded trajectory (all `empirical`)."""
        if not self.records:
            return {"n_steps": 0}
        L = self._logits()
        m_final = margin_to_worst(L, self.t)
        # telescoped a-priori bound: m0 - 2 Σ δ; slack = actual - bound (conservatism)
        bound = self.m0 - 2.0 * self.budget
        slack = (m_final - bound).sort().values
        never = self.flipped_at < 0
        hold = torch.tensor([r["premise_hold_rate"] for r in self.records])
        hold_i = torch.tensor([r["premise_hold_rate_perctx"] for r in self.records])
        late = hold[int(0.75 * len(hold)):]

        def split(mask: Tensor) -> dict:
            nb = int(mask.sum())
            if nb == 0:
                return {}
            ff = self.flipped_at[mask]
            return {
                "n": nb,
                "never_flipped_frac": float((ff < 0).float().mean()),
                "median_first_flip": int(ff[ff >= 0].median()) if bool((ff >= 0).any()) else -1,
            }

        bo = self.budget_out_at
        return {
            "n_steps": self.n_steps,
            "rho": self.rho,
            "entry_matches_truth": self.entry_matches_truth,
            "premise_hold_rate_mean": float(hold.mean()),
            "premise_hold_rate_last25%": float(late.mean()),
            "premise_hold_rate_perctx_mean": float(hold_i.mean()),
            "premise_hold_rate_perctx_last25%": float(hold_i[int(0.75 * len(hold_i)):].mean()),
            "total_new_flips": int((~never).sum()),
            "never_flipped_frac": float(never.float().mean()),
            "viol_premise_flip": self.viol_premise_flip,     # must be 0
            "viol_transfer": self.viol_transfer,             # must be 0
            "beta_share_mean": float(
                torch.tensor([r["beta_share"] for r in self.records]).mean()
            ),
            "budget_2sum": 2.0 * self.budget,
            "m0_p50": float(self.m0.median()),
            "budget_vacuous_frac": float((bo >= 0).float().mean()),
            "budget_T*_p50": int(bo[bo >= 0].median()) if bool((bo >= 0).any()) else -1,
            "flip_horizon": split(torch.ones(self.B, dtype=torch.bool)),
            "train_split": split(~self.held_out),
            "heldout_split": split(self.held_out),
            "telescope_slack_p10": float(slack[int(0.10 * (self.B - 1))]),
            "telescope_slack_p50": float(slack[int(0.50 * (self.B - 1))]),
            "final_margin_p50": float(m_final.median()),
        }
