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


ABLATION_MODES = ("all_neg", "bipolar_neg", "scattered_neg")


def ablated_incidence(rows: Tensor, mode: str) -> Tensor:
    """Copy ``rows`` with selected negative entries set to 0.

    ``all_neg`` clears every negative entry. ``bipolar_neg`` and ``scattered_neg`` clear
    negatives only on rows ``pole_label`` calls by that name. Positive entries stay.
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"unknown ablation mode {mode}")
    out = rows.detach().clone()
    if mode == "all_neg":
        return out.clamp_min(0)
    target = "bipolar" if mode == "bipolar_neg" else "scattered"
    for k in range(out.shape[0]):
        if pole_label(rows[k])[3] == target:
            out[k] = out[k].clamp_min(0)
    return out


def incidence_logits(prog, x: Tensor, incidences: Tensor) -> Tensor:
    """Hard-gate logits ``G @ C + bias`` for a program with no lookup or schema bank.

    ``C`` is a ``(K, V)`` incidence matrix, normally ``head_incidences`` or an ablation of it.
    """
    if len(prog.lookup) or len(prog.direct_lookup) or prog.schema_bank is not None:
        raise RuntimeError("headw ablation scores rule incidences only")
    gates = prog.all_gates(x, hard=True)
    if prog.n_rules:
        active = torch.cat(gates, dim=1)
    else:
        active = x.new_zeros(x.shape[0], 0)
    return active @ incidences + prog.bias


def decide_delta(before: Tensor, after: Tensor) -> tuple[int, float, float]:
    """``(changed, mean_margin_before, mean_margin_after)`` on hard logits.

    ``changed`` is the number of rows whose argmax differs. Mean margin is top-1 minus
    top-2. The margin does not vote.
    """

    def _margin(logits: Tensor) -> float:
        top2 = logits.detach().topk(2, dim=-1).values
        return float((top2[:, 0] - top2[:, 1]).mean())

    changed = int((before.argmax(dim=-1) != after.argmax(dim=-1)).sum())
    return changed, _margin(before), _margin(after)
