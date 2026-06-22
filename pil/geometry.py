"""Core geometric primitives for Projective Incidence Calculus / Learning.

Hilbert-space incidences via inner products, the Gram kernel, decision margins,
the frame potential (mutual-coherence objective), participation ratio, and
Laguerre / power-diagram utilities (the tropical T=0 view).

Naming follows ``fieldrun/PIC_PROPOSAL.md`` §2 exactly:

    sources   d_j in H         (circuit / partial-evidence vectors)
    frames    U_v in H         (proposition directions; unembedding rows)
    pairing   c_j^v = <d_j,U_v> (direct logit attribution, DLA)
    logits    L_v   = sum_j c_j^v
    Gram      G_vw  = <U_v,U_w> (the structural carrier of non-truth-functionality, T2)

Every function is labelled **decode-side** (depends on the proposition frame and
the readout geometry) or **frame-side** (intrinsic to {U_v}); see
``docs/notes/pil_learning_dynamics.md``. This distinction is load-bearing: a loss
that improves decode-side margins can *degrade* encode-side (co-firing) structure
(measured in polygram PR #113), so PIL keeps the two accounted separately.
"""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_rows(x: Tensor, eps: float = 1e-8) -> Tensor:
    """L2-normalize rows (projective directions)."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def gram_matrix(U: Tensor) -> Tensor:
    """Gram kernel ``G_vw = <U_v, U_w>`` (frame-side)."""
    return U @ U.T


def cosine_gram(U: Tensor, eps: float = 1e-8) -> Tensor:
    """Cosine Gram (unit-diagonal): ``rho_vw = <U_v,U_w> / (||U_v|| ||U_w||)`` (frame-side).

    This is the scale-free competition-hardness kernel of PIC T2: ``rho -> 1`` is the
    near-synonym / common-mode regime; ``rho`` diagonal is the classical (Boolean)
    incidence-calculus limit.
    """
    Un = normalize_rows(U, eps)
    return Un @ Un.T


def incidences(d: Tensor, U: Tensor) -> Tensor:
    """Incidence matrix ``c_j^v = <d_j, U_v>`` for sources ``d`` and frames ``U`` (decode-side).

    Accepts ``d`` of shape ``(J, dim)`` -> ``(J, V)`` or ``(B, J, dim)`` -> ``(B, J, V)``.
    """
    if d.dim() == 2:
        return d @ U.T
    if d.dim() == 3:
        return torch.einsum("bjd,vd->bjv", d, U)
    raise ValueError(f"Unsupported d shape: {tuple(d.shape)} (want (J,dim) or (B,J,dim))")


def logits_from_incidences(c: Tensor, bias: Tensor | None = None) -> Tensor:
    """Aggregate incidences to logits ``L_v = sum_j c_j^v (+ b_v)`` (decode-side).

    Sums over the *source* axis (``dim=-2``): ``(J,V)->(V,)``, ``(B,J,V)->(B,V)``.
    """
    L = c.sum(dim=-2)
    if bias is not None:
        L = L + bias
    return L


def margin_to_worst(L: Tensor, target: Tensor | int) -> Tensor:
    """Decision margin to the strongest competitor: ``L_t - max_{v != t} L_v`` (decode-side).

    Vectorised over arbitrary leading dims. ``target`` may be a Python ``int``
    (broadcast) or a long tensor of shape ``L.shape[:-1]``. Returns shape
    ``L.shape[:-1]``.

    (The original starter sliced ``L[..., :target]`` / ``L[..., target+1:]`` which
    only works for a scalar ``int`` target and silently mis-indexed a batched
    tensor target; this gather/scatter form is the fix.)
    """
    if isinstance(target, int):
        target = L.new_full(L.shape[:-1], target, dtype=torch.long)
    else:
        target = target.to(device=L.device, dtype=torch.long)
    idx = target.unsqueeze(-1)
    correct = L.gather(-1, idx).squeeze(-1)
    masked = L.scatter(-1, idx, float("-inf"))  # out-of-place; L is untouched
    max_wrong = masked.max(dim=-1).values
    return correct - max_wrong


def frame_potential(U: Tensor, exclude_diagonal: bool = True) -> Tensor:
    """Mean squared off-diagonal cosine of the frame (frame-side scalar objective).

    ``FP = mean_{v != w} rho_vw^2`` where ``rho`` is the cosine Gram. This is the
    PIC-native "improve the Gram structure" target: driving it down decorrelates
    propositions (toward the diagonal-G / classical-incidence-calculus limit of
    T2) and reduces destructive interference. Its floor for ``V > dim`` is the
    Welch bound ``(V-dim) / (dim (V-1))`` (cf. the i-orca ``superposition`` corpus),
    so it is honest about over-completeness: you cannot drive it to zero when
    ``V > dim``. Returns a scalar.
    """
    G = cosine_gram(U)
    n = G.shape[0]
    if exclude_diagonal:
        off = G - torch.diag(torch.diagonal(G))
        return off.pow(2).sum() / max(n * (n - 1), 1)
    return G.pow(2).mean()


def welch_bound(n_propositions: int, dim: int) -> float:
    """Welch lower bound on mean-squared coherence for ``V`` vectors in ``R^dim``.

    The achievable floor of :func:`frame_potential` when ``V > dim`` (0 when ``V <= dim``,
    i.e. an orthonormal set is possible). Lets experiments report frame quality as a
    *ratio to the information-theoretic optimum*, not an absolute that is unreachable.
    """
    v = n_propositions
    if v <= dim:
        return 0.0
    return (v - dim) / (dim * (v - 1))


def participation_ratio(weights: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    """Effective number of contributing sources ``(sum w)^2 / sum w^2`` (diagnostic).

    ``weights`` are non-negative per-source magnitudes (e.g. ``|c_j^t|``). This is the
    same PR fieldrun measures on circuit contributions (PIC T4): a *high* PR position
    is the diffuse "computed / forge-tax" regime with no small sufficient support; a
    *low* PR position is the retrievable regime. PIL tracks PR to see whether learning
    is concentrating support (lowering PR -> more retrievable), without ever claiming
    the floor is reducible to 1.
    """
    w = weights.abs()
    s1 = w.sum(dim=dim)
    s2 = w.pow(2).sum(dim=dim)
    return s1.pow(2) / (s2 + eps)


def power_diagram_weights(U: Tensor, bias: Tensor | None = None) -> Tensor:
    """Additive weights of the Laguerre/power diagram (frame-side): ``||U_v||^2 - 2 b_v``.

    The argmax decision ``argmax_v L_v`` is the power cell of ``r = sum_j d_j`` under
    these weights (PIC §3 / FINDINGS §5b: the normalised margin is the exact facet
    distance). This is the tropical T=0 dual of the softmax.
    """
    w = (U * U).sum(dim=-1)
    if bias is not None:
        w = w - 2 * bias
    return w
