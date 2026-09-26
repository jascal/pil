"""Empirical margin-stratified directional early-exit calibration (#131).

This implements the signed preregistration's C0/C1/C2 thresholds. The reused
calibration data and P-every policy do not confer a formal coverage guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pil.early_exit import ExitRecord, conformal_quantile

ALPHA = 0.05
CANDIDATES = ("C0", "C1", "C2")
BIN_COUNT = {"C0": 1, "C1": 3, "C2": 5}
TIE_PRIORITY = {"C0": 0, "C2": 1, "C1": 2}


@dataclass
class Calibrator:
    kind: str
    edges: np.ndarray  # (looks, bins-1)
    q: np.ndarray      # (looks, bins)


def normalized_margin(radius: float, ynorm: float) -> float:
    return float(radius / ynorm) if ynorm > 0 and math.isfinite(ynorm) else math.nan


def bin_index(value: float, edges: np.ndarray) -> int:
    """The left side of an edge owns a tie, as pinned in the prereg."""
    return int(np.searchsorted(edges, value, side="left"))


def fit(records: list[tuple[ExitRecord, np.ndarray]], kind: str) -> Calibrator:
    if kind not in BIN_COUNT:
        raise ValueError(kind)
    if not records:
        raise ValueError("no calibration records")
    looks = len(records[0][0].R) - 1
    bins = BIN_COUNT[kind]
    edges = np.empty((looks, bins - 1), dtype=np.float64)
    q = np.full((looks, bins), math.inf, dtype=np.float64)
    for k in range(looks):
        pairs = [(normalized_margin(r.R[k], r.ynorm[k]), float(a[k] / r.ynorm[k]))
                 for r, a in records if r.ynorm[k] > 0 and math.isfinite(r.ynorm[k])]
        pairs = [(m, s) for m, s in pairs if math.isfinite(m) and math.isfinite(s)]
        if bins > 1:
            edges[k] = (np.quantile([m for m, _ in pairs], np.arange(1, bins) / bins, method="linear")
                        if pairs else np.full(bins - 1, math.inf))
        groups: list[list[float]] = [[] for _ in range(bins)]
        for m, s in pairs:
            groups[bin_index(m, edges[k]) if bins > 1 else 0].append(s)
        for j, scores in enumerate(groups):
            q[k, j] = conformal_quantile(np.asarray(scores), ALPHA) if scores else math.inf
    return Calibrator(kind, edges, q)


def bounds(records: list[tuple[ExitRecord, np.ndarray]], calibrator: Calibrator) -> np.ndarray:
    looks = calibrator.q.shape[0]
    out = np.full((len(records), looks), math.inf, dtype=np.float64)
    for i, (r, _) in enumerate(records):
        for k in range(looks):
            m = normalized_margin(r.R[k], r.ynorm[k])
            if math.isfinite(m):
                j = bin_index(m, calibrator.edges[k]) if calibrator.edges.shape[1] else 0
                out[i, k] = calibrator.q[k, j] * r.ynorm[k]
    return out


def candidate(metrics: dict[str, dict]) -> str | None:
    """3B selection, including the zero-certification exclusion."""
    eligible = []
    for kind in CANDIDATES:
        m = metrics[kind]
        count = round(m["N"] * m["coverage"])
        if count and m["violations"] / count <= ALPHA:
            eligible.append(kind)
    if not eligible:
        return None
    return max(eligible, key=lambda kind: (metrics[kind]["net_every"], TIE_PRIORITY[kind]))


def verdict(metrics: dict, sound: bool) -> str:
    if not sound:
        return "VOID"
    if metrics["coverage"] < 0.10:
        return "FAILED"
    count = round(metrics["N"] * metrics["coverage"])
    if metrics["violations"] / count > 2 * ALPHA:
        return "FAILED"
    if metrics["coverage"] >= 0.25 and metrics["net_every"] >= 1.0:
        return "CONFIRMED"
    return "PARTIAL"


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [centre - half, centre + half]
