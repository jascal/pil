"""Does participation ratio predict anything? — quantities and statistics for
``docs/notes/pr_validation_prereg.md``.

Per position, from a block-by-token incidence matrix ``C`` (``nb x V``: ``C[j, v] = ⟨d̃_j, U_v⟩``, exact
logit contributions) and the decided token ``t``:

- ``r_eff`` — PR of the target incidences centred across a candidate set (the pic §8 /
  ``tau_star_entropy`` measure);
- ``k_suf`` — greedy sufficient-coalition size: blocks ordered by ``C[j,t] − C[j,v₂]``, the smallest prefix
  whose partial sum has ``t`` as argmax (``min_blocks_to_argmax``), an upper bound on the minimal sufficient
  coalition;
- PR-head sufficiency — do the top ``round(r_eff)`` blocks by centred target incidence decide ``t``?

Statistics: average-tie ranks, partial Spearman (rank residuals on a covariate's rank), cluster bootstrap.
numpy only.
"""

from __future__ import annotations

import numpy as np


def pr(w: np.ndarray) -> float:
    """Participation ratio ``(Σ|w|)² / Σw²`` (``nan`` for an all-zero vector)."""
    w = np.abs(np.asarray(w, dtype=np.float64))
    s2 = float((w * w).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else float("nan")


def centred_target(C: np.ndarray, t: int, cands: np.ndarray) -> np.ndarray:
    """``cc_j(t) = C[j,t] − mean_{v∈cands} C[j,v]`` for every block ``j``."""
    return C[:, t] - C[:, cands].mean(axis=1)


def k_suf(C: np.ndarray, t: int, v2: int) -> int:
    """Greedy sufficient-coalition size over columns of ``C`` (``nb + 1`` if the full sum misses ``t``)."""
    order = np.argsort(-(C[:, t] - C[:, v2]), kind="stable")
    acc = np.zeros(C.shape[1], dtype=np.float64)
    for k, j in enumerate(order, start=1):
        acc += C[j]
        if int(np.argmax(acc)) == t:
            return k
    return C.shape[0] + 1


def pr_head_decides(C: np.ndarray, t: int, cc_t: np.ndarray, r_eff: float) -> bool:
    """Do the top ``round(r_eff)`` blocks by ``|cc_j(t)|`` have ``t`` as argmax (over columns of ``C``)?"""
    m = int(min(max(round(r_eff), 1), C.shape[0]))
    top = np.argsort(-np.abs(cc_t), kind="stable")[:m]
    return int(np.argmax(C[top].sum(axis=0))) == t


def merge_layers(C: np.ndarray) -> np.ndarray:
    """Block rows ``[embed, L0.attn, L0.mlp, …]`` → layer rows ``[embed, L0, …]`` (attn + mlp summed)."""
    nb = C.shape[0]
    if (nb - 1) % 2:
        raise ValueError(f"expected 1 + 2·n_layer blocks, got {nb}")
    return np.concatenate([C[:1], C[1::2] + C[2::2]], axis=0)


def split_ratio(cc_t: np.ndarray, parts: int = 4) -> float:
    """``r_eff`` after splitting the largest-``|cc|`` block into ``parts`` equal pieces, over before."""
    w = np.abs(np.asarray(cc_t, dtype=np.float64))
    j = int(np.argmax(w))
    split = np.concatenate([np.delete(w, j), np.full(parts, w[j] / parts)])
    return pr(split) / pr(w)


def erank(D: np.ndarray) -> float:
    """Spectral effective rank ``(Σσ²)²/Σσ⁴`` of a block matrix ``D`` (``nb x d``)."""
    s2 = np.linalg.svd(np.asarray(D, dtype=np.float64), compute_uv=False) ** 2
    return float(s2.sum() ** 2 / (s2 * s2).sum())


# ── statistics ──────────────────────────────────────────────────────────────────────────────────────────────


def rankdata(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties given their average rank."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(rankdata(x), rankdata(y))


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson correlation of the rank residuals of ``x`` and ``y`` after OLS on the rank of ``z``."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return _pearson(ex, ey)


def bootstrap_ci(stat, arrays: tuple[np.ndarray, ...], clusters: np.ndarray | None, reps: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of ``stat(*arrays)`` resampling clusters (or positions if ``clusters`` is None)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    if clusters is None:
        groups = [np.array([i]) for i in range(n)]
    else:
        keys = np.unique(clusters)
        groups = [np.flatnonzero(clusters == k) for k in keys]
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        v = stat(*(a[idx] for a in arrays))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
