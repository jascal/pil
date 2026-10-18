"""Whether a hardened rule's negative incidences are a pole or a residual.

The row is one rule of ``RuleProgram.head_incidences``, the matrix ``export_program``
writes into ``headw`` (zeros dropped). Cuts are locked: a negative side at or below
5% of the L1 mass is thin; a heavier side whose largest candidate holds strictly
more than half of that mass is bipolar; a heavier side split at or below half is
scattered.
"""

from __future__ import annotations

import torch
from torch import Tensor

DEAD_L1 = 1e-6
POLE_FLOOR = 0.05
CONCENTRATION = 0.5


def pole_label(row: Tensor) -> tuple[float, int, float, str]:
    """Return ``(neg_share, leader, leader_share, label)`` for one incidence row.

    ``leader`` is the column index of the largest negative incidence, or ``-1`` when
    the negative mass is zero. ``label`` is ``dead``, ``thin``, ``bipolar``, or
    ``scattered``.
    """
    w = torch.where(torch.isfinite(row), row, torch.zeros_like(row)).detach().flatten()
    pos = w.clamp_min(0).sum()
    neg_w = (-w).clamp_min(0)
    neg = neg_w.sum()
    total = float(pos + neg)
    if total <= DEAD_L1:
        return 0.0, -1, 0.0, "dead"
    neg_share = float(neg / total)
    if float(neg) == 0.0:
        return neg_share, -1, 0.0, "thin"
    leader = int(torch.argmax(neg_w))
    leader_share = float(neg_w[leader] / neg)
    if neg_share <= POLE_FLOOR:
        label = "thin"
    elif leader_share > CONCENTRATION:
        label = "bipolar"
    else:
        label = "scattered"
    return neg_share, leader, leader_share, label
