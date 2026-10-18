"""Which hard clusters a rule moves, and whether that set shares an atom.

Ablation removes one rule by zeroing the source it emits inside ``build_sources``, then
re-reads within-cluster margins on the learner's real forward path. The verdict uses the
generator's atom ownership and nothing softer: cluster ``c`` owns topic atom ``c`` and code
atoms ``(n_clusters + 2c, n_clusters + 2c + 1)``. Those sets are pairwise disjoint, so a
rule that moves two hard clusters is unrelated packing, and a rule that moves one is reuse.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .learner import ProjectiveIncidenceLearner
from .proposer import RuleBank, build_sources

# Above float dust on margins of order 1, below a real routing effect.
MARGIN_MOVE_TOL = 1e-3

# Leader share must be strictly above this. An even split is spread. Not retuned after a run.
CONCENTRATION = 0.5


def cluster_atoms(cluster: int, n_clusters: int) -> frozenset[int]:
    """Topic atom and XOR code-atom pair owned by cluster ``c`` in the compositional generator."""
    code = n_clusters + 2 * cluster
    return frozenset({cluster, code, code + 1})


def shares_atom(clusters: list[int], n_clusters: int) -> bool:
    """True when the moved clusters' generator atoms have a non-empty intersection.

    A single cluster shares its own atoms. Two distinct clusters in this generator never do.
    """
    if not clusters:
        return False
    owned = [cluster_atoms(c, n_clusters) for c in clusters]
    common = set.intersection(*(set(s) for s in owned))
    return len(common) > 0


def rule_verdict(moved: list[int], n_clusters: int) -> str:
    """``reuse`` if the moved clusters share an atom, ``packing`` if they do not, ``idle`` if none move."""
    if not moved:
        return "idle"
    if shares_atom(moved, n_clusters):
        return "reuse"
    return "packing"


def bank_verdict(rule_verdicts: list[str]) -> str:
    """Packing if any rule scatters. Reuse only when some rule is reuse and none pack."""
    if any(v == "packing" for v in rule_verdicts):
        return "packing"
    if any(v == "reuse" for v in rule_verdicts):
        return "reuse"
    return "idle"


def per_cluster_within_margins(logits: Tensor, targets: Tensor, n_clusters: int) -> Tensor:
    """Mean held-out within-cluster margin ``L[t] - L[t xor 1]`` for each cluster. NaN if empty."""
    sib = targets ^ 1
    margin = logits.gather(1, targets[:, None]).squeeze(1) - logits.gather(1, sib[:, None]).squeeze(1)
    clusters = targets // 2
    out = torch.full((n_clusters,), float("nan"), device=logits.device)
    for c in range(n_clusters):
        mask = clusters == c
        if int(mask.sum()) > 0:
            out[c] = margin[mask].mean()
    return out.detach()


def moved_hard_clusters(
    before: Tensor,
    after: Tensor,
    hard_clusters: Tensor,
    tol: float = MARGIN_MOVE_TOL,
) -> list[int]:
    """Hard clusters whose within-cluster margin changes by more than ``tol``."""
    delta = (after - before).abs()
    moved: list[int] = []
    for c in range(hard_clusters.shape[0]):
        if not bool(hard_clusters[c]):
            continue
        if not torch.isfinite(delta[c]):
            continue
        if float(delta[c]) > tol:
            moved.append(c)
    return moved


def logits_with_rule_zeroed(
    model: ProjectiveIncidenceLearner,
    z: Tensor,
    A: Tensor,
    bank: RuleBank,
    rule: int | None,
) -> Tensor:
    """Forward on ``build_sources``. ``rule`` set zeroes that rule's emitted source and no other."""
    sources = build_sources(z, A, bank)
    if rule is not None:
        sources = sources.clone()
        atom_count = A.shape[0]
        sources[:, atom_count + rule, :] = 0
    return model.forward(sources, target=None)["logits"]


def mass_share(
    mass: Tensor,
    idle_at: float = MARGIN_MOVE_TOL,
) -> tuple[int, float, float, str]:
    """Leader, top-1 share, top-3 share, and ``single`` / ``spread`` / ``idle``.

    ``mass`` is a non-negative per-cluster vector. Non-finite entries count as zero.
    ``single`` requires the leader's share to be strictly above ``CONCENTRATION``.
    """
    m = torch.where(torch.isfinite(mass), mass, torch.zeros_like(mass)).detach().flatten().clamp_min(0)
    total = float(m.sum())
    if total <= idle_at or m.numel() == 0:
        return -1, 0.0, 0.0, "idle"
    order = torch.argsort(m, descending=True)
    top1 = float(m[order[0]] / total)
    top3 = float(m[order[: min(3, m.numel())]].sum() / total)
    label = "single" if top1 > CONCENTRATION else "spread"
    return int(order[0]), top1, top3, label


def code_pair_mass(w_row: Tensor, n_clusters: int) -> Tensor:
    """``|W|`` on each cluster's two XOR code atoms. Topic atoms are not included."""
    w = w_row.detach().abs()
    mass = torch.empty(n_clusters, dtype=w.dtype)
    for c in range(n_clusters):
        code = n_clusters + 2 * c
        mass[c] = w[code] + w[code + 1]
    return mass


def topic_mass(w_row: Tensor, n_clusters: int) -> Tensor:
    """``|W|`` on each cluster's topic atom. Printed, and not part of the weight vote."""
    return w_row.detach().abs()[:n_clusters].flatten()


def rule_margin_abs_delta(
    model: ProjectiveIncidenceLearner,
    z: Tensor,
    A: Tensor,
    bank: RuleBank,
    targets: Tensor,
    n_clusters: int,
    rule: int,
) -> Tensor:
    """Absolute per-cluster within-margin change when ``rule`` is zeroed on ``build_sources``."""
    before = per_cluster_within_margins(
        logits_with_rule_zeroed(model, z, A, bank, None), targets, n_clusters
    )
    after = per_cluster_within_margins(
        logits_with_rule_zeroed(model, z, A, bank, rule), targets, n_clusters
    )
    return (before - after).abs()


def rule_agreement(noisy: str, clean: str, weight: str) -> str:
    """``single`` or ``spread`` only when all three measures say so. Noisy-idle rules do not vote."""
    if noisy == "idle":
        return "idle"
    if noisy == clean == weight == "single":
        return "single"
    if noisy == clean == weight == "spread":
        return "spread"
    return "disagree"


def budget_agreement(rule_labels: list[str]) -> str:
    """``agree_single`` / ``agree_spread`` only when every voting rule agrees. Else ``disagree``."""
    voting = [label for label in rule_labels if label != "idle"]
    if voting and all(label == "single" for label in voting):
        return "agree_single"
    if voting and all(label == "spread" for label in voting):
        return "agree_spread"
    return "disagree"


def rule_moved_clusters(
    model: ProjectiveIncidenceLearner,
    z: Tensor,
    A: Tensor,
    bank: RuleBank,
    targets: Tensor,
    hard_clusters: Tensor,
    rule: int,
    tol: float = MARGIN_MOVE_TOL,
) -> list[int]:
    """Hard clusters whose within-cluster margin moves when rule ``rule`` is ablated."""
    n_clusters = int(hard_clusters.shape[0])
    before = per_cluster_within_margins(
        logits_with_rule_zeroed(model, z, A, bank, None), targets, n_clusters
    )
    after = per_cluster_within_margins(
        logits_with_rule_zeroed(model, z, A, bank, rule), targets, n_clusters
    )
    return moved_hard_clusters(before, after, hard_clusters, tol)
