"""Theory-guided scoring for the generative step (propose -> score -> select).

The §5c null showed that in a gradient-rich setting SGD already allocates rules to the
at-risk facets, so margin-style targeting buys little. The one regime where *selecting*
the right rule should matter is when SGD cannot fix a bad rule -- i.e. when the rule's
input-weights are **frozen** (weak gradient on the feature it reads). This module scores
candidate rules by how well their firing resolves the currently-unresolved decision, so a
frozen, score-selected rule carries separating signal a random frozen rule does not.

A *candidate* is an input-weight vector ``w`` over atom activations ``z``; its rule is
``relu(<w, z>)``. We score the **firing pattern** (which examples it activates on), not a
raw direction in ``H`` -- the output direction is learned afterwards. Scoring is marginal on
the current model state (which examples are still ambiguous), per Grok's tropical-envelope
idea, corrected to operate on rule activations and the actually-unresolved label.
"""

from __future__ import annotations

import torch
from torch import Tensor


def candidate_activations(z: Tensor, W: Tensor) -> Tensor:
    """ReLU rule activations for a bank of candidate input-weights. ``z``(N,K), ``W``(C,K) -> (N,C)."""
    return torch.relu(z @ W.T)


def ambiguity_resolution_score(
    acts: Tensor,
    targets: Tensor,
    margins: Tensor,
    n_classes: int,
    frac: float = 0.3,
    eps: float = 1e-8,
) -> Tensor:
    """Correlation-ratio (η²) of each candidate's firing with the target identity, on the
    currently-ambiguous (lowest-margin) examples. High = the rule's firing tracks the
    unresolved decision, so it resolves power-diagram near-ties (the tropical-envelope idea).

    Cluster-agnostic and label-leak-free w.r.t. the planted structure: it only uses the
    model's own current margins to pick the ambiguous subset, then measures between-target
    variance of the candidate firing there. ``acts``(N,C), returns (C,).
    """
    n = acts.shape[0]
    k = max(int(frac * n), 1)
    amb = torch.topk(-margins, k).indices              # lowest-margin examples
    h = acts[amb]                                       # (k, C)
    t = targets[amb]                                    # (k,)

    mu = h.mean(dim=0)                                  # (C,)
    ss_tot = ((h - mu) ** 2).sum(dim=0)                 # (C,)

    C = h.shape[1]
    grp_sum = torch.zeros(n_classes, C, device=h.device).index_add_(0, t, h)
    grp_cnt = torch.zeros(n_classes, device=h.device).index_add_(
        0, t, torch.ones_like(t, dtype=h.dtype)
    )
    present = grp_cnt > 0
    grp_mean = torch.zeros_like(grp_sum)
    grp_mean[present] = grp_sum[present] / grp_cnt[present].unsqueeze(-1)
    ss_between = (grp_cnt.unsqueeze(-1) * (grp_mean - mu.unsqueeze(0)) ** 2).sum(dim=0)  # (C,)

    return ss_between / (ss_tot + eps)


def activation_variance_score(acts: Tensor) -> Tensor:
    """Naive heuristic: prefer candidates that fire with high variance (active, not dead). (C,)."""
    return acts.var(dim=0)


def select_top_k(scores: Tensor, k: int) -> Tensor:
    """Indices of the top-``k`` scoring candidates."""
    return torch.topk(scores, min(k, scores.numel())).indices


def propose_and_select(
    z: Tensor,
    targets: Tensor,
    margins: Tensor,
    n_classes: int,
    n_candidates: int,
    k: int,
    method: str = "ambiguity",
    frac: float = 0.3,
    seed: int = 0,
) -> Tensor:
    """Generate ``n_candidates`` random input-weight vectors, score, return the selected ``k`` (k, K).

    ``method``: ``"ambiguity"`` (η² on ambiguous examples), ``"variance"`` (activation variance),
    or ``"random"`` (no scoring -- the control).
    """
    K = z.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(n_candidates, K, generator=g).to(z.device)
    if method == "random":
        return W[:k]
    acts = candidate_activations(z, W)
    if method == "ambiguity":
        scores = ambiguity_resolution_score(acts, targets, margins, n_classes, frac=frac)
    elif method == "variance":
        scores = activation_variance_score(acts)
    else:
        raise ValueError(f"unknown method {method!r}")
    return W[select_top_k(scores, k)]
