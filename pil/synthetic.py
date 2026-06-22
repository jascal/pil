"""Synthetic PIC problems with a tunable *hardness* knob.

The easy planted-frame problem (`learner.create_synthetic_problem`) is too easy to
stress the frame-side objective: the margin term alone already yields a near-incoherent
frame, so `frame_reg` barely moves anything. The hard problem here adds the two levers
that make the Gram genuinely dense and the decision genuinely confusable:

  * **over-completeness** -- `V >> dim`, so the Welch floor is well above 0 and propositions
    *must* overlap;
  * **planted synonym clusters** -- propositions are drawn near a few cluster centres with a
    small angular spread, so within-cluster cosine `rho -> 1` (the PIC T2 common-mode regime
    where differential incidence `D_j -> 0`). An example's natural competitor is then a
    *synonym*: the sources for target `t` (drawn from `U_gt[t]`) look almost identical to the
    sources for `t`'s cluster-mates, an analog of open-class confusability / the forge tax.

This is the regime where the decode-side (margin) and frame-side (frame potential) objectives
are expected to trade: decorrelating the frame separates near-collinear synonyms (margin up),
but pulls the learned frame off the planted source directions (alignment down), so there should
be an interior optimum in `frame_reg` -- unlike the flat easy regime.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .geometry import normalize_rows
from .learner import PILConfig


def create_clustered_problem(
    config: PILConfig,
    n_examples: int = 512,
    n_clusters: int = 16,
    cluster_spread: float = 0.12,
    sparsity: float = 0.12,
    noise: float = 0.08,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Over-complete, synonym-clustered planted-frame problem.

    Returns ``(sources, targets, U_gt, cluster_of)``:
      ``sources``    (n_examples, J, dim)  active sources point at ``U_gt[target]``
      ``targets``    (n_examples,)
      ``U_gt``       (V, dim)              planted frame (block-dense Gram)
      ``cluster_of`` (V,)                  cluster id per proposition (for diagnostics)

    Smaller ``cluster_spread`` and larger ``V/dim`` => harder. Expected within-cluster
    cosine of ``U_gt`` is ~ ``1 / (1 + cluster_spread**2 * dim)``.
    """
    gen = torch.Generator(device="cpu").manual_seed(config.seed + 2)
    V, dim, J = config.n_propositions, config.dim, config.n_sources_per_step

    centers = normalize_rows(torch.randn(n_clusters, dim, generator=gen))
    cluster_of = torch.randint(0, n_clusters, (V,), generator=gen)
    U_gt = normalize_rows(centers[cluster_of] + cluster_spread * torch.randn(V, dim, generator=gen))

    n_active = max(3, int(J * sparsity))
    targets = torch.randint(0, V, (n_examples,), generator=gen)
    sources = torch.empty(n_examples, J, dim)
    for i in range(n_examples):
        t = int(targets[i])
        d = noise * torch.randn(J, dim, generator=gen)
        active = torch.randperm(J, generator=gen)[:n_active]
        gain = (1.5 + 0.5 * torch.randn(n_active, generator=gen)).unsqueeze(-1)
        d[active] = U_gt[t] * gain + noise * torch.randn(n_active, dim, generator=gen)
        sources[i] = d

    dev = config.device
    return (
        sources.to(dev),
        targets.to(dev),
        U_gt.to(dev),
        cluster_of.to(dev),
    )


def within_cluster_cosine(U: Tensor, cluster_of: Tensor) -> float:
    """Mean |cosine| between distinct same-cluster propositions in frame ``U`` (frame-side).

    The synonymy that survives in the *learned* frame. Compare against ``U_gt``'s value to
    see how far the frame objective pulled the frame apart from the planted clusters.
    """
    Un = normalize_rows(U)
    G = (Un @ Un.T).abs()
    same = cluster_of.unsqueeze(0) == cluster_of.unsqueeze(1)
    same = same & ~torch.eye(U.shape[0], dtype=torch.bool, device=U.device)
    if same.sum() == 0:
        return float("nan")
    return G[same].mean().item()


def frame_alignment(U: Tensor, U_gt: Tensor) -> float:
    """Mean cosine between each learned frame row and its planted counterpart (decode/recovery).

    1.0 == the learner kept the planted source directions; lower == it moved off them (which is
    what decorrelating an intrinsically-clustered frame forces). The cost side of the trade.
    """
    return (normalize_rows(U) * normalize_rows(U_gt)).sum(-1).mean().item()
