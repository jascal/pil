"""Projective Incidence Learner (PIL) core.

The two-phase loop PIC motivates:

  1. **Generative step** -- propose many parallel partial-evidence vectors ``d_j``
     (:meth:`ProjectiveIncidenceLearner.propose_parallel_sources`).
  2. **Learn + gate step** -- score them by incidence ``c_j^v = <d_j,U_v>``, decide
     by margin / power diagram, and update the proposition frame ``U`` so more mass
     becomes high-margin "retrieved" and less stays in the diffuse "computed" regime.

The objective is **improvement / targeting, not reproduction**: PIL is free to move
the logits off any host model's, trading decode-side faithfulness for retrievability,
margin, and frame conditioning. The loss is split into clearly-labelled terms:

    nll          decode-side   fit the target distribution (validation anchor)
    margin       decode-side   push the worst-competitor margin past a target
    frame_pot    frame-side    decorrelate propositions toward the Welch floor (T2)

See ``docs/notes/pil_learning_dynamics.md`` for why the decode/frame split is
load-bearing (decode-side gains can silently degrade encode-side co-firing structure).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .geometry import (
    frame_potential,
    incidences,
    logits_from_incidences,
    margin_to_worst,
    normalize_rows,
    participation_ratio,
    welch_bound,
)


@dataclass
class PILConfig:
    dim: int = 64
    n_propositions: int = 128
    n_sources_per_step: int = 32
    lr: float = 1e-2
    frame_reg: float = 0.05          # weight on the frame-potential (Gram) term
    margin_weight: float = 0.5       # weight on the hinge-margin term
    margin_target: float = 2.0       # desired worst-competitor margin
    temp: float = 1.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ProjectiveIncidenceLearner(nn.Module):
    """Learns proposition frames ``U`` (and bias) and proposes/gates parallel sources."""

    def __init__(self, config: PILConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_prop = config.n_propositions

        gen = torch.Generator(device="cpu").manual_seed(config.seed)
        U0 = torch.randn(self.n_prop, self.dim, generator=gen)
        self.U = nn.Parameter(U0.to(config.device))
        self.bias = nn.Parameter(torch.zeros(self.n_prop, device=config.device))
        self.to(config.device)

    # -- frame upkeep -------------------------------------------------------
    def normalize_U(self) -> None:
        """Project frames back to the unit sphere (keeps margins scale-comparable)."""
        with torch.no_grad():
            self.U.data = normalize_rows(self.U.data)

    # -- inference + loss ---------------------------------------------------
    def forward(self, sources: Tensor, target: Tensor | int | None = None) -> dict[str, Tensor]:
        c = incidences(sources, self.U)               # (..., J, V)
        L = logits_from_incidences(c, self.bias)      # (..., V)

        out: dict[str, Tensor] = {"incidences": c, "logits": L}

        if target is None:
            return out

        if isinstance(target, int):
            target_t = L.new_full(L.shape[:-1], target, dtype=torch.long)
        else:
            target_t = target.to(device=L.device, dtype=torch.long)

        # decode-side: fit the target
        probs = torch.softmax(L / self.config.temp, dim=-1)
        nll = -torch.log(
            probs.gather(-1, target_t.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        ).mean()

        # decode-side: margin to the strongest competitor
        m_worst = margin_to_worst(L, target_t)
        margin_loss = torch.relu(self.config.margin_target - m_worst).mean()

        # frame-side: decorrelate the proposition frame (toward the Welch floor)
        frame_pot = frame_potential(self.U)

        total = nll + self.config.margin_weight * margin_loss + self.config.frame_reg * frame_pot

        out.update(
            probs=probs,
            nll=nll,
            margin_loss=margin_loss,
            frame_potential=frame_pot,
            total_loss=total,
            margin_worst=m_worst,
        )
        return out

    # -- generative step ----------------------------------------------------
    @torch.no_grad()
    def propose_parallel_sources(
        self,
        n_sources: int | None = None,
        strategy: str = "random",
        align_scale: float = 1.5,
        noise: float = 0.1,
    ) -> Tensor:
        """Propose ``n`` parallel partial-evidence vectors ``d_j`` (the generative half).

        Strategies:
          ``"random"``  i.i.d. Gaussian directions -- the null proposer (baseline).
          ``"frame"``   sample near current frames: ``d = align_scale * U[v] + noise`` for
                        random ``v`` -- candidate *partial solutions* that already point at
                        a proposition, the PIC "incidence" a circuit would cast. As ``U``
                        sharpens, proposals sharpen: a minimal self-supervised generator.

        For real fieldrun-seeded sources (observed residual DLAs ``d_j``), use
        :func:`pil.fieldrun_io.load_probe_sources` instead of this synthetic proposer.
        """
        n = n_sources or self.config.n_sources_per_step
        dev = self.config.device
        if strategy == "random":
            return torch.randn(n, self.dim, device=dev)
        if strategy == "frame":
            v = torch.randint(0, self.n_prop, (n,), device=dev)
            return align_scale * self.U.data[v] + noise * torch.randn(n, self.dim, device=dev)
        raise ValueError(f"unknown strategy {strategy!r} (want 'random' or 'frame')")

    # -- training -----------------------------------------------------------
    def training_step(
        self,
        sources: Tensor | None = None,
        target: int | Tensor | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, float]:
        if sources is None:
            sources = self.propose_parallel_sources()

        self.train()
        out = self.forward(sources, target=target)
        loss = out["total_loss"]

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.normalize_U()

        return {
            k: float(v.item())
            for k, v in out.items()
            if isinstance(v, Tensor) and v.numel() == 1
        } | {"loss": float(loss.item())}

    # -- evaluation ---------------------------------------------------------
    @torch.no_grad()
    def evaluate_retrievability(
        self,
        sources: Tensor,
        targets: Tensor,
        margin_threshold: float = 1.5,
    ) -> dict[str, float]:
        """Decode-side retrievability + frame-side conditioning, in one pass.

        ``sources`` is ``(B, J, dim)``, ``targets`` is ``(B,)``. Reports the fraction of
        positions whose worst-competitor margin clears ``margin_threshold`` (retrievable),
        the mean margin, top-1 accuracy, mean participation ratio of the target's source
        support (lower => more concentrated), and the frame potential as a ratio to the
        Welch floor (1.0 == optimally incoherent for this over-completeness).
        """
        self.eval()
        out = self.forward(sources, target=None)
        L = out["logits"]                       # (B, V)
        c = out["incidences"]                   # (B, J, V)
        targets = targets.to(device=L.device, dtype=torch.long)

        margins = margin_to_worst(L, targets)   # (B,)
        retrievable = (margins > margin_threshold).float().mean().item()
        acc = (L.argmax(dim=-1) == targets).float().mean().item()

        # PR of |c_j^t| over sources j, averaged over the batch
        c_t = c.gather(-1, targets.view(-1, 1, 1).expand(-1, c.shape[1], 1)).squeeze(-1)  # (B,J)
        pr = participation_ratio(c_t, dim=-1).mean().item()

        fp = frame_potential(self.U).item()
        wb = welch_bound(self.n_prop, self.dim)
        return {
            "retrievable_fraction": retrievable,
            "top1_acc": acc,
            "avg_margin": margins.mean().item(),
            "mean_support_pr": pr,
            "frame_potential": fp,
            "welch_floor": wb,
            "frame_pot_over_welch": (fp / wb) if wb > 0 else float("nan"),
            "margin_threshold": margin_threshold,
        }


def create_synthetic_problem(
    config: PILConfig,
    n_examples: int = 512,
    sparsity: float = 0.15,
    noise: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """A planted-frame problem: each example's active sources point at a hidden target.

    Returns ``(sources, targets, U_gt)`` with ``sources`` of shape
    ``(n_examples, n_sources_per_step, dim)`` -- a *batched 3-D* tensor that flows
    through :meth:`forward` without reshaping (the starter flattened it, which pooled
    every example into one logit vector). ``U_gt`` is the ground-truth frame for
    recovery diagnostics; PIL is not required to match it (improvement, not reproduction).
    """
    gen = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    dev = config.device
    U_gt = normalize_rows(torch.randn(config.n_propositions, config.dim, generator=gen)).to(dev)

    J = config.n_sources_per_step
    n_active = max(3, int(J * sparsity))
    sources = torch.empty(n_examples, J, config.dim, device=dev)
    targets = torch.randint(0, config.n_propositions, (n_examples,), generator=gen).to(dev)

    for i in range(n_examples):
        t = int(targets[i].item())
        d = noise * torch.randn(J, config.dim, generator=gen).to(dev)
        active = torch.randperm(J, generator=gen)[:n_active].to(dev)
        gain = (1.5 + 0.5 * torch.randn(n_active, generator=gen).to(dev)).unsqueeze(-1)
        d[active] = U_gt[t] * gain + noise * torch.randn(n_active, config.dim, generator=gen).to(dev)
        sources[i] = d

    return sources, targets, U_gt
