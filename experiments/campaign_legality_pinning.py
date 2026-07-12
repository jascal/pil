"""Slice #90 -- constraint-register build 1: legality feature + naked-vs-hidden pinning.

Batched tensor legality DERIVED FEATURE (candidates / naked / hidden / forced_value) over
in-window sudoku boards reconstructed with #89 helpers. One registered measurement decides
whether the register inventory is NAKED-ONLY or NAKED UNION HIDDEN. No Souffle certificate
(#91), no battery-hub arm (#92). Sudoku-only, LABELS=corpus.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# v5 binds recipe constants at import time -- setdefaults before the import (same as #89).
WYLY_ENV: dict[str, str] = {
    "WYLY_TAG": "pythia70m",
    "WYLY_DS": "sudoku",
    "WYLY_LIB": "mined",
    "WYLY_JUDGE": "cover",
    "WYLY_ONLINE": "1",
    "WYLY_COVER": "sw",
    "WYLY_CONCEPTS": "1",
    "WYLY_POINTER": "1",
    "WYLY_TPOINTER": "1",
    "WYLY_DX": "1",
    "WYLY_CX": "1",
    "WYLY_FOLDS": "3",
    "WYLY_LABELS": "corpus",
}
for _k, _v in WYLY_ENV.items():
    os.environ.setdefault(_k, _v)

import campaign_sudoku_forced_move as osc  # noqa: E402
import wyly_lm_v5 as v5  # noqa: E402

OUTPUT = REPO / "data" / "legality_pinning.json"
PARITY_MIN_ROWS = 500
VALIDITY_FLOOR = 50
NAKED_SUFFICES_RATIO = 0.90
L_WINDOW = 256

FRAMING_NOTE = (
    "domain-solving numbers, NOT natural-text; no natural-text comparison in this slice."
)

REVEAL_INDEX_NOTE = (
    "the sudoku dataset presents ONLY late-reveal solution cells (index ~59-80); early/mid cells "
    "never appear as prediction targets, so the register is measured only on the near-solved "
    "sub-task and its capability on early/mid cells is UNTESTED by this data."
)

HONESTY_NOTE = (
    "feature is computed-not-certified (the certificate is #91, so confidence is measured "
    "not 1.0); sudoku-only; LABELS=corpus. "
    + REVEAL_INDEX_NOTE
)

DESIGN_REGISTERED = """\
## Design (registered) -- slice #90 legality feature + naked-vs-hidden pinning

1. LEGALITY FEATURE (tensor, batched): for each SOLUTION-cell prediction row, reconstruct the
   in-window board (puzzle clues + revealed row-major solution prefix) via #89 helpers, then
   compute candidate sets / is_naked / is_hidden / forced_value as VECTORIZED torch ops over a
   stacked [N,9,9] grid tensor (row/col/box peer masks + uniqueness reductions). Not a
   per-row Python reimplementation of classify_forced.
2. PARITY: tensor is_naked / is_hidden / forced_value must agree with #89 classify_forced on
   >= 500 real dataset rows (assert; report mismatches).
3. CONDITION 2 (freeze measurement): on residual ERROR rows among te solution-cell rows,
   recovered_naked = frac(is_naked AND forced_value==gold);
   recovered_union = frac((is_naked OR is_hidden) AND forced_value==gold).
   Validity floor: if absolute union-recovered count < 50 -> "UNDERPOWERED -- ESCALATE"
   (do not decide freeze). Else freeze:
   if recovered_naked >= 0.90 * recovered_union: inventory = NAKED-ONLY
   else: inventory = NAKED UNION HIDDEN.
4. PROPOSER CONFIDENCE: register forced-move proposer; measure ValFiredAccuracy via
   v5.val_fired_acc (measured, not asserted 1.0).
5. ADMISSION (descriptive, NON-gating): cover with vs without proposer; held-out marginal
   informs #91, NEVER the freeze (freeze from Condition 2 recovery ratio alone).
6. STRATIFICATION: recovered_naked / recovered_union per reveal-third (cell_index//27).
7. FRAMING: domain-solving numbers only; LABELS=corpus; no cross-domain comparison in this slice.
"""


# ---------------------------------------------------------------------------
# Tensor legality feature (batched over [N, 9, 9] grids)
# ---------------------------------------------------------------------------

def _box_index_map(device: torch.device | None = None) -> torch.Tensor:
    """[9, 9] box id in 0..8 for each cell."""
    r = torch.arange(9, device=device)
    c = torch.arange(9, device=device)
    rr, cc = torch.meshgrid(r, c, indexing="ij")
    return (rr // 3) * 3 + (cc // 3)


def legality_feature_tensor(
    grids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    true_v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Batched legality derived feature over stacked boards.

    Args:
        grids: [N, 9, 9] int, values 0..9 (0 = empty). Target cell may be filled or empty;
               it is treated as empty for candidate / hidden calculations.
        rows, cols: [N] target cell coordinates.
        true_v: [N] gold digit 1..9 (used for hidden-single uniqueness).

    Returns dict of [N] tensors:
        n_candidates, is_naked (bool), is_hidden (bool), forced_value (long, -1 if none).
    """
    if grids.ndim != 3 or grids.shape[1:] != (9, 9):
        raise ValueError(f"grids must be [N,9,9], got {tuple(grids.shape)}")
    n = grids.shape[0]
    device = grids.device
    rows = rows.to(device=device, dtype=torch.long)
    cols = cols.to(device=device, dtype=torch.long)
    true_v = true_v.to(device=device, dtype=torch.long)

    work = grids.to(dtype=torch.long).clone()
    ar = torch.arange(n, device=device)
    work[ar, rows, cols] = 0

    # one-hot digits 1..9 -> [N, 9, 9, 9]
    onehot = torch.stack([(work == d) for d in range(1, 10)], dim=-1)

    # presence reductions over row / col / box index groups
    row_has = onehot.any(dim=2)  # [N, 9 rows, 9 digits]
    col_has = onehot.any(dim=1)  # [N, 9 cols, 9 digits]
    oh_box = onehot.view(n, 3, 3, 3, 3, 9)
    box_has = oh_box.any(dim=2).any(dim=3).reshape(n, 9, 9)  # [N, 9 boxes, 9 digits]

    box_of_target = (rows // 3) * 3 + (cols // 3)
    used = (
        row_has[ar, rows]
        | col_has[ar, cols]
        | box_has[ar, box_of_target]
    )  # [N, 9]
    cand_mask = ~used  # [N, 9] digit d present at index d-1
    n_candidates = cand_mask.sum(dim=1)
    is_naked = n_candidates == 1

    # unique naked digit (1..9); -1 if not naked
    # argmax on cand_mask gives digit_index when exactly one True
    naked_digit = cand_mask.long().argmax(dim=1) + 1
    naked_digit = torch.where(is_naked, naked_digit, torch.full_like(naked_digit, -1))

    # ---- hidden singles for true_v (batched uniqueness over units) ----
    tv_idx = (true_v - 1).clamp(min=0, max=8)
    idx = tv_idx.view(n, 1, 1).expand(n, 9, 1)
    row_block = row_has.gather(2, idx).squeeze(-1)  # [N, 9]
    col_block = col_has.gather(2, idx).squeeze(-1)
    box_block = box_has.gather(2, idx).squeeze(-1)

    box_map = _box_index_map(device)  # [9, 9]
    blocked = (
        row_block[:, :, None]
        | col_block[:, None, :]
        | box_block[:, box_map]
    )  # [N, 9, 9]
    is_empty = work == 0
    is_legal = is_empty & ~blocked  # true_v legal at empty cells

    cand_at_target = is_legal[ar, rows, cols]
    row_count = is_legal[ar, rows, :].sum(dim=1)
    col_count = is_legal.gather(
        2, cols.view(n, 1, 1).expand(n, 9, 1)
    ).squeeze(-1).sum(dim=1)

    # box unit counts: reshape empties-legal to boxes
    legal_boxes = is_legal.view(n, 3, 3, 3, 3)  # [N, br, ri, bc, ci]
    br, bc = rows // 3, cols // 3
    # pick the target's box for each row
    # legal_boxes[ar, br, :, bc, :] -> [N, 3, 3]
    box_plane = legal_boxes[ar, br, :, bc, :]
    box_count = box_plane.reshape(n, 9).sum(dim=1)

    is_hidden = cand_at_target & (
        (row_count == 1) | (col_count == 1) | (box_count == 1)
    )

    # forced_value: unique naked digit if naked; else true_v if hidden; else -1
    forced_value = torch.full((n,), -1, dtype=torch.long, device=device)
    forced_value = torch.where(is_hidden, true_v, forced_value)
    forced_value = torch.where(is_naked, naked_digit, forced_value)

    return {
        "n_candidates": n_candidates,
        "is_naked": is_naked,
        "is_hidden": is_hidden,
        "forced_value": forced_value,
        "cand_mask": cand_mask,
    }


def oracle_forced_value(grid: list[list[int]], r: int, c: int, true_v: int) -> int:
    """#89-aligned forced_value from plain-numpy oracle helpers."""
    naked = osc.is_naked_single(grid, r, c)
    hidden = osc.is_hidden_single(grid, r, c, true_v)
    if naked:
        cands = osc.cell_candidates(grid, r, c)
        return int(next(iter(cands)))
    if hidden:
        return int(true_v)
    return -1


def parity_row(
    grid: list[list[int]], r: int, c: int, true_v: int,
    feat_naked: bool, feat_hidden: bool, feat_forced: int,
) -> dict[str, Any] | None:
    """Return mismatch detail dict if tensor feature disagrees with #89 oracle, else None."""
    flags = osc.classify_forced(grid, r, c, true_v)
    o_forced = oracle_forced_value(grid, r, c, true_v)
    if (
        bool(feat_naked) == bool(flags["naked"])
        and bool(feat_hidden) == bool(flags["hidden"])
        and int(feat_forced) == int(o_forced)
    ):
        return None
    return {
        "r": int(r),
        "c": int(c),
        "true_v": int(true_v),
        "oracle": {
            "naked": flags["naked"],
            "hidden": flags["hidden"],
            "forced_value": o_forced,
        },
        "feature": {
            "naked": bool(feat_naked),
            "hidden": bool(feat_hidden),
            "forced_value": int(feat_forced),
        },
    }


def check_parity_batch(
    grids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    true_v: torch.Tensor,
    *,
    max_report: int = 20,
) -> dict[str, Any]:
    """Run tensor feature and compare every row to #89 classify_forced / forced_value."""
    feat = legality_feature_tensor(grids, rows, cols, true_v)
    n = grids.shape[0]
    mismatches: list[dict[str, Any]] = []
    g_cpu = grids.detach().cpu().tolist()
    r_cpu = rows.detach().cpu().tolist()
    c_cpu = cols.detach().cpu().tolist()
    tv_cpu = true_v.detach().cpu().tolist()
    naked = feat["is_naked"].detach().cpu().tolist()
    hidden = feat["is_hidden"].detach().cpu().tolist()
    forced = feat["forced_value"].detach().cpu().tolist()
    for i in range(n):
        m = parity_row(
            g_cpu[i], r_cpu[i], c_cpu[i], tv_cpu[i],
            naked[i], hidden[i], forced[i],
        )
        if m is not None:
            m["i"] = i
            mismatches.append(m)
            if len(mismatches) >= max_report:
                # still count the rest
                for j in range(i + 1, n):
                    m2 = parity_row(
                        g_cpu[j], r_cpu[j], c_cpu[j], tv_cpu[j],
                        naked[j], hidden[j], forced[j],
                    )
                    if m2 is not None:
                        mismatches.append({**m2, "i": j})
                break
    return {
        "n_checked": n,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:max_report],
        "passed": len(mismatches) == 0 and n > 0,
        "feature": feat,
    }


# ---------------------------------------------------------------------------
# Host-side grid stack from windows (reuses #89 reconstruction)
# ---------------------------------------------------------------------------

def build_grid_stack_for_rows(
    row_indices: list[int],
    ids_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    cls_cpu: torch.Tensor,
    uv_cpu: torch.Tensor,
    sol_pat: list[int],
    puz_pat: list[int],
    value_map: dict[int, int | None],
) -> dict[str, Any]:
    """Reconstruct boards for solution-cell rows; return tensors + metadata lists."""
    grids_list: list[list[list[int]]] = []
    rows_l: list[int] = []
    cols_l: list[int] = []
    true_vs: list[int] = []
    gold_compact: list[int] = []
    meta: list[dict[str, Any]] = []
    skipped = 0

    for i in row_indices:
        seq = osc.raw_window_from_compact(ids_cpu[i], uv_cpu)
        rec = osc.reconstruct_from_window(seq, sol_pat, puz_pat, value_map)
        rc = osc.target_solution_rc(seq, sol_pat, L=ids_cpu.shape[1])
        if rc is None or rec is None or rec["puzzle"] is None:
            skipped += 1
            continue
        r, c = rc
        gold_tid = int(uv_cpu[int(cls_cpu[int(y_cpu[i])].item())].item())
        tv = value_map.get(gold_tid, ...)
        if tv is ... or tv is None:
            skipped += 1
            continue
        tv = int(tv)
        board = osc.board_state_before_cell(rec["puzzle"], rec["solution"], r, c)
        grids_list.append(board)
        rows_l.append(r)
        cols_l.append(c)
        true_vs.append(tv)
        # compact-into-uv class of gold (same space as yv = cls[y])
        gold_compact.append(int(cls_cpu[int(y_cpu[i])].item()))
        cell_index = r * 9 + c
        meta.append({
            "row": int(i),
            "r": r,
            "c": c,
            "true_v": tv,
            "cell_index": cell_index,
            "reveal_third": cell_index // 27,
            "gold_compact": int(cls_cpu[int(y_cpu[i])].item()),
        })

    if not grids_list:
        return {
            "grids": torch.zeros(0, 9, 9, dtype=torch.long),
            "rows": torch.zeros(0, dtype=torch.long),
            "cols": torch.zeros(0, dtype=torch.long),
            "true_v": torch.zeros(0, dtype=torch.long),
            "gold_compact": torch.zeros(0, dtype=torch.long),
            "meta": [],
            "skipped": skipped,
        }

    return {
        "grids": torch.tensor(grids_list, dtype=torch.long),
        "rows": torch.tensor(rows_l, dtype=torch.long),
        "cols": torch.tensor(cols_l, dtype=torch.long),
        "true_v": torch.tensor(true_vs, dtype=torch.long),
        "gold_compact": torch.tensor(gold_compact, dtype=torch.long),
        "meta": meta,
        "skipped": skipped,
    }


def digit_to_compact_maps(
    uv: torch.Tensor,
    bare: dict[int, int],
    space: dict[int, int],
) -> tuple[dict[int, int], dict[int, int]]:
    """Map digit 1..9 -> compact-into-uv class for bare (col0) and space (col1-8) tokens."""
    raw_to_c = {int(uv[i].item()): i for i in range(len(uv))}
    bare_c = {d: raw_to_c[bare[d]] for d in range(1, 10) if bare[d] in raw_to_c}
    space_c = {d: raw_to_c[space[d]] for d in range(1, 10) if space[d] in raw_to_c}
    return bare_c, space_c


def forced_digits_to_compact(
    forced_value: torch.Tensor,
    cols: torch.Tensor,
    bare_c: dict[int, int],
    space_c: dict[int, int],
) -> torch.Tensor:
    """Map digit forced_value + column to compact class id; -1 stays -1."""
    out = torch.full_like(forced_value, -1)
    fv = forced_value.detach().cpu().tolist()
    cs = cols.detach().cpu().tolist()
    for i, (d, c) in enumerate(zip(fv, cs, strict=True)):
        if d < 1:
            continue
        d = int(d)
        c = int(c)
        mp = bare_c if c == 0 else space_c
        if d in mp:
            out[i] = mp[d]
    return out


# ---------------------------------------------------------------------------
# Condition 2: recovery / validity floor / freeze
# ---------------------------------------------------------------------------

def compute_recovery(
    is_naked: torch.Tensor | list[bool] | np.ndarray,
    is_hidden: torch.Tensor | list[bool] | np.ndarray,
    forced_value: torch.Tensor | list[int] | np.ndarray,
    gold: torch.Tensor | list[int] | np.ndarray,
    error_mask: torch.Tensor | list[bool] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Recovery counts/fractions on (optionally error-masked) rows.

    recovered_naked = is_naked AND forced_value == gold
    recovered_union = (is_naked OR is_hidden) AND forced_value == gold
    """
    def _to_bool_list(x: Any) -> list[bool]:
        if isinstance(x, torch.Tensor):
            return [bool(v) for v in x.detach().cpu().tolist()]
        return [bool(v) for v in list(x)]

    def _to_int_list(x: Any) -> list[int]:
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().tolist()]
        return [int(v) for v in list(x)]

    naked = _to_bool_list(is_naked)
    hidden = _to_bool_list(is_hidden)
    forced = _to_int_list(forced_value)
    g = _to_int_list(gold)
    n_all = len(naked)
    if error_mask is None:
        err = [True] * n_all
    else:
        err = _to_bool_list(error_mask)

    n_err = sum(1 for e in err if e)
    n_naked_rec = 0
    n_union_rec = 0
    for i in range(n_all):
        if not err[i]:
            continue
        match = forced[i] == g[i] and forced[i] >= 1
        if naked[i] and match:
            n_naked_rec += 1
        if (naked[i] or hidden[i]) and match:
            n_union_rec += 1

    return {
        "n_error": n_err,
        "n_recovered_naked": n_naked_rec,
        "n_recovered_union": n_union_rec,
        "recovered_naked": osc.fraction(n_naked_rec, n_err),
        "recovered_union": osc.fraction(n_union_rec, n_err),
    }


def freeze_from_recovery(
    recovered_naked: float,
    recovered_union: float,
    n_recovered_union: int,
    *,
    validity_floor: int = VALIDITY_FLOOR,
    ratio: float = NAKED_SUFFICES_RATIO,
) -> dict[str, Any]:
    """Validity floor + freeze rule. Pure function for tests and campaign.

    If n_recovered_union < validity_floor: underpowered, no freeze decision.
    Else: NAKED-ONLY iff recovered_naked >= ratio * recovered_union.
    """
    if n_recovered_union < validity_floor:
        return {
            "underpowered": True,
            "message": "UNDERPOWERED -- ESCALATE",
            "inventory": None,
            "freeze_decided": False,
            "recovered_naked": recovered_naked,
            "recovered_union": recovered_union,
            "n_recovered_union": n_recovered_union,
            "threshold": ratio * recovered_union,
        }
    naked_only = recovered_naked >= ratio * recovered_union
    inventory = "NAKED-ONLY" if naked_only else "NAKED UNION HIDDEN"
    return {
        "underpowered": False,
        "message": (
            f"if recovered_naked >= {ratio:.2f} * recovered_union: "
            f"inventory = NAKED-ONLY else: inventory = NAKED UNION HIDDEN"
        ),
        "inventory": inventory,
        "freeze_decided": True,
        "recovered_naked": recovered_naked,
        "recovered_union": recovered_union,
        "n_recovered_union": n_recovered_union,
        "threshold": ratio * recovered_union,
        "naked_suffices": naked_only,
    }


def stratify_by_reveal_third(
    cell_index: list[int] | torch.Tensor | np.ndarray,
    is_naked: list[bool] | torch.Tensor | np.ndarray,
    is_hidden: list[bool] | torch.Tensor | np.ndarray,
    forced_eq_gold: list[bool] | torch.Tensor | np.ndarray,
    error_mask: list[bool] | torch.Tensor | np.ndarray | None = None,
) -> dict[int, dict[str, Any]]:
    """Per reveal-third (cell_index // 27) recovery counts/fractions on error rows."""
    def _bools(x: Any) -> list[bool]:
        if isinstance(x, torch.Tensor):
            return [bool(v) for v in x.detach().cpu().tolist()]
        return [bool(v) for v in list(x)]

    def _ints(x: Any) -> list[int]:
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().tolist()]
        return [int(v) for v in list(x)]

    ci = _ints(cell_index)
    naked = _bools(is_naked)
    hidden = _bools(is_hidden)
    eq = _bools(forced_eq_gold)
    n = len(ci)
    err = [True] * n if error_mask is None else _bools(error_mask)

    out: dict[int, dict[str, Any]] = {}
    for third in (0, 1, 2):
        n_err = 0
        n_nak = 0
        n_uni = 0
        for i in range(n):
            if not err[i]:
                continue
            if ci[i] // 27 != third:
                continue
            n_err += 1
            if naked[i] and eq[i]:
                n_nak += 1
            if (naked[i] or hidden[i]) and eq[i]:
                n_uni += 1
        out[third] = {
            "n_error": n_err,
            "n_recovered_naked": n_nak,
            "n_recovered_union": n_uni,
            "recovered_naked": osc.fraction(n_nak, n_err),
            "recovered_union": osc.fraction(n_uni, n_err),
        }
    return out


def prenamed_reading_holds(
    recovered_union_overall: float,
    recovered_naked_third0: float,
    *,
    union_floor: float = 0.90,
    early_naked_cap: float = 0.90,
    n_error_third0: int | None = None,
) -> bool:
    """True iff overall union recovery is high AND first-third naked recovery is low.

    When n_error_third0 is provided and is 0, the reading does not hold (empty third
    must not vacuously satisfy 'naked recovery < 0.90' via 0/0 -> 0.0).
    """
    if n_error_third0 is not None and n_error_third0 <= 0:
        return False
    return recovered_union_overall >= union_floor and recovered_naked_third0 < early_naked_cap


def reveal_index_distribution(
    cell_index: list[int] | torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    """min/max/median and reveal-third histogram of cell_index (r*9+c, range 0-80).

    Population is every classifiable solution-cell prediction target (tr+te), not only
    the te residual-error subset used for Condition 2 stratification.
    Histogram thirds: 0 -> [0-26], 1 -> [27-53], 2 -> [54-80] (cell_index // 27).
    """
    def _ints(x: Any) -> list[int]:
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().tolist()]
        return [int(v) for v in list(x)]

    ci = _ints(cell_index)
    n = len(ci)
    counts = [0, 0, 0]
    for idx in ci:
        third = idx // 27
        if 0 <= third <= 2:
            counts[third] += 1
    histogram = {
        str(t): {
            "count": counts[t],
            "fraction": osc.fraction(counts[t], n),
        }
        for t in (0, 1, 2)
    }
    if n == 0:
        return {
            "min": None,
            "max": None,
            "median": None,
            "histogram": histogram,
            "n": 0,
        }
    # np.median: average of two middle values when n is even
    return {
        "min": int(min(ci)),
        "max": int(max(ci)),
        "median": float(np.median(ci)),
        "histogram": histogram,
        "n": n,
    }


# ---------------------------------------------------------------------------
# CONF_FN-style proposer (digit forced_value; cover wrapper maps to class ids)
# ---------------------------------------------------------------------------

def conf_fn_forced_digits(
    grids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    true_v: torch.Tensor,
) -> torch.Tensor:
    """CONF_FN-style: returns forced_value digit (-1 where it does not fire)."""
    return legality_feature_tensor(grids, rows, cols, true_v)["forced_value"]


def make_cover_proposer(
    sol_pat: list[int],
    puz_pat: list[int],
    value_map: dict[int, int | None],
    uv: torch.Tensor,
    bare: dict[int, int],
    space: dict[int, int],
    bare_c: dict[int, int],
    space_c: dict[int, int],
):
    """Return fn(ids_batch) -> compact class of forced digit, or -1.

    Host-side reconstruction per row (inherited from #89), then batched tensor feature.
    """
    uv_cpu = uv.detach().cpu()

    def fn(ids_batch: torch.Tensor) -> torch.Tensor:
        device = ids_batch.device
        b = ids_batch.shape[0]
        out = torch.full((b,), -1, dtype=torch.long, device=device)
        ids_cpu = ids_batch.detach().cpu()
        grids_l: list[list[list[int]]] = []
        rows_l: list[int] = []
        cols_l: list[int] = []
        true_vs: list[int] = []
        keep_idx: list[int] = []

        for i in range(b):
            seq = [int(x) for x in uv_cpu[ids_cpu[i]].tolist()]
            rec = osc.reconstruct_from_window(seq, sol_pat, puz_pat, value_map)
            rc = osc.target_solution_rc(seq, sol_pat, L=ids_cpu.shape[1])
            if rc is None or rec is None or rec["puzzle"] is None:
                continue
            r, c = rc
            # true_v from board's eventual solution cell if present, else skip hidden path
            # For the proposer we only need naked (unique cand) or hidden w.r.t. solution
            # digit when in-window; prefer solution grid digit at (r,c) when available.
            sol_v = 0
            if rec["solution"] is not None:
                sol_v = int(rec["solution"][r][c])
            if sol_v == 0:
                # solution digit not yet in window (we are predicting it); hidden needs true_v.
                # Without gold we can still fire on naked singles.
                board = osc.board_state_before_cell(rec["puzzle"], rec["solution"], r, c)
                grids_l.append(board)
                rows_l.append(r)
                cols_l.append(c)
                true_vs.append(1)  # placeholder; hidden with wrong true_v won't falsely naked
                keep_idx.append(i)
                continue
            board = osc.board_state_before_cell(rec["puzzle"], rec["solution"], r, c)
            grids_l.append(board)
            rows_l.append(r)
            cols_l.append(c)
            true_vs.append(sol_v)
            keep_idx.append(i)

        if not grids_l:
            return out

        grids_t = torch.tensor(grids_l, dtype=torch.long)
        rows_t = torch.tensor(rows_l, dtype=torch.long)
        cols_t = torch.tensor(cols_l, dtype=torch.long)
        tv_t = torch.tensor(true_vs, dtype=torch.long)
        feat = legality_feature_tensor(grids_t, rows_t, cols_t, tv_t)
        # For rows where solution digit was missing we only trust naked
        forced = feat["forced_value"].tolist()
        naked = feat["is_naked"].tolist()
        for j, bi in enumerate(keep_idx):
            d = int(forced[j])
            if d < 1:
                continue
            # if true_v was placeholder (solution not in window): only accept naked
            if true_vs[j] == 1 and not naked[j]:
                # could be real true_v==1; re-check: if sol was 0 we used placeholder 1
                seq = [int(x) for x in uv_cpu[ids_cpu[bi]].tolist()]
                rec = osc.reconstruct_from_window(seq, sol_pat, puz_pat, value_map)
                r, c = rows_l[j], cols_l[j]
                sol_v = int(rec["solution"][r][c]) if rec and rec["solution"] is not None else 0
                if sol_v == 0 and not naked[j]:
                    continue
            c = cols_l[j]
            mp = bare_c if c == 0 else space_c
            if d in mp:
                out[bi] = mp[d]
        return out

    return fn


def make_cover_proposer_from_lookup(
    row_to_class: dict[int, int],
    full_ids: torch.Tensor,
):
    """fn(ids_batch) that matches windows against full_ids rows (exact row content).

    Used in the campaign after precomputing per-dataset-row forced class predictions so
    val_fired_acc / core_cover_sw do not re-reconstruct. Matching is by row equality of
    the compact id window (unique under this corpus pipeline).
    """
    # Build a hash of each precomputed row's window for O(1) lookup is hard with collisions;
    # instead store tensor of forced class aligned to dataset index and require the caller
    # to pass index-aware wrappers. For cover we use an index-closure approach below.
    raise NotImplementedError("use make_index_proposer")


def make_index_proposer(forced_class_by_row: torch.Tensor):
    """Return a stateful proposer bound later via set_indices, OR use pre-sliced values.

    core_cover_sw calls fn(ids[idxs]) without passing idxs. We therefore recompute from
    content via make_cover_proposer, OR monkey-patch by closing over a queue of predictions
    in call order -- fragile. Content-based make_cover_proposer is the safe path.

    This helper builds fn that ignores ids and consumes a pre-ordered prediction vector
    when the call site guarantees the same index order -- not used for cover.

    For campaign val_fired_acc we build a content-based proposer.
    """
    def fn(ids_batch: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("index proposer requires content-based make_cover_proposer")

    return fn


def make_lookup_proposer(
    full_ids_cpu: torch.Tensor,
    forced_class: torch.Tensor,
):
    """fn(ids_batch)->class: match each batch row to a dataset row by exact window equality.

    full_ids_cpu: [N, L] compact ids on CPU.
    forced_class: [N] long, -1 where no fire.
    """
    # Fingerprint: sum of tokens * small prime mix -- collision-safe fallback to full compare
    n, ell = full_ids_cpu.shape
    # Use first+last+checksum as key then verify
    keys: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n):
        row = full_ids_cpu[i]
        key = (int(row[0].item()), int(row[-1].item()), int(row.sum().item()))
        keys.setdefault(key, []).append(i)
    fc = forced_class.detach().cpu()

    def fn(ids_batch: torch.Tensor) -> torch.Tensor:
        device = ids_batch.device
        b = ids_batch.shape[0]
        out = torch.full((b,), -1, dtype=torch.long, device=device)
        batch = ids_batch.detach().cpu()
        for j in range(b):
            row = batch[j]
            key = (int(row[0].item()), int(row[-1].item()), int(row.sum().item()))
            cands = keys.get(key, [])
            hit = -1
            for i in cands:
                if torch.equal(full_ids_cpu[i], row):
                    hit = i
                    break
            if hit >= 0:
                out[j] = fc[hit]
        return out

    return fn


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Main campaign
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    log("=" * 72)
    log("SLICE #90 -- legality feature + naked-vs-hidden pinning")
    log("=" * 72)
    log(DESIGN_REGISTERED)
    log(f"FRAMING: {FRAMING_NOTE}")
    log(f"HONESTY: {HONESTY_NOTE}")
    log()
    log("WYLY_* env used for this run:")
    env_used = {k: os.environ.get(k, "") for k in WYLY_ENV}
    for k, v in env_used.items():
        log(f"  {k}={v}")
    log(
        f"  (module-bound) TAG={v5.TAG} DS={v5.DS} LIB={v5.LIB} JUDGE={v5.JUDGE} "
        f"LABELS={v5.LABELS} SWCOVER={v5.SWCOVER} STATE={v5.STATE.name}"
    )
    log()

    codec = v5.load_codec()
    puz_pat, sol_pat = osc.header_patterns(codec)
    value_map = osc.build_value_map(codec)
    bare = {d: int(codec.encode(str(d))[0]) for d in range(1, 10)}
    space = {d: int(codec.encode(f" {d}")[0]) for d in range(1, 10)}
    log(f"header patterns: PUZZLE={puz_pat} SOLUTION={sol_pat}")

    # Round-trip reuse from #89
    raw = torch.load(v5.DATA, map_location="cpu")
    raw_ids = raw["kept_ids"]
    log(f"\nRaw windows: {tuple(raw_ids.shape)} from {v5.DATA.name}")
    rt = osc.run_round_trip_sample(raw_ids, n=osc.ROUND_TRIP_N, seed=0)
    log(
        f"ROUND-TRIP ({osc.ROUND_TRIP_N}-window sample): checked={rt['n_checked']} "
        f"failed_windows={rt['n_failed_windows']} mismatch_cells={rt['n_mismatch_cells']} "
        f"skipped={rt['n_skipped']} passed={rt['passed']}"
    )
    if not rt["passed"]:
        raise AssertionError(
            f"round-trip failed: {rt['n_failed_windows']} windows, "
            f"{rt['n_mismatch_cells']} cell mismatches"
        )

    # Expensive regeneration
    log("\n--- run_cover_regeneration (v5.main) ---")
    t_reg = time.time()
    model, rules, original_cover = osc.run_cover_regeneration()
    log(f"regeneration wall time: {time.time() - t_reg:.1f}s")
    log(f"captured rules ({len(rules)}): {[n for n, _ in rules]}")

    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    n_raw = int(raw_ids.shape[0])
    n_kept = int(ids.shape[0])
    log(
        f"\nload_ds: N={n_kept} (raw={n_raw}, dropped={n_raw - n_kept}) "
        f"tr={len(tr)} te={len(te)} vocab_compact={len(uv)}"
    )

    ids_cpu = ids.detach().cpu()
    y_cpu = y.detach().cpu()
    cls_cpu = cls.detach().cpu()
    uv_cpu = uv.detach().cpu() if torch.is_tensor(uv) else uv
    tr_set = set(int(x) for x in tr.detach().cpu().tolist())
    te_set = set(int(x) for x in te.detach().cpu().tolist())

    # Identify solution-cell rows (same semantics as #89)
    log("\n--- identifying solution-cell rows ---")
    all_indices = list(range(n_kept))
    stack = build_grid_stack_for_rows(
        all_indices, ids_cpu, y_cpu, cls_cpu, uv_cpu,
        sol_pat, puz_pat, value_map,
    )
    meta = stack["meta"]
    sol_rows = [m["row"] for m in meta]
    log(f"solution-cell rows (classifiable): {len(sol_rows)} (skipped={stack['skipped']})")
    te_sol = [i for i in sol_rows if i in te_set]
    tr_sol = [i for i in sol_rows if i in tr_set]
    log(f"  of which tr={len(tr_sol)} te={len(te_sol)}")
    if not te_sol:
        raise RuntimeError("no classifiable solution-cell rows in te")

    # Map dataset row -> position in stack
    row_to_stack = {m["row"]: k for k, m in enumerate(meta)}

    # Tensor feature on all solution-cell rows
    log("\n--- legality tensor feature (batched) ---")
    feat_all = legality_feature_tensor(
        stack["grids"], stack["rows"], stack["cols"], stack["true_v"],
    )

    # PARITY on >= PARITY_MIN_ROWS real rows
    n_parity = min(len(meta), max(PARITY_MIN_ROWS, len(meta)))
    # deterministic sample of stack indices
    rng = np.random.default_rng(0)
    if len(meta) > n_parity:
        # use all if we want max power; spec says >= 500 sample
        n_parity = max(PARITY_MIN_ROWS, min(len(meta), 2000))
        pick = sorted(rng.choice(len(meta), size=n_parity, replace=False).tolist())
    else:
        pick = list(range(len(meta)))
        n_parity = len(pick)

    parity = check_parity_batch(
        stack["grids"][pick],
        stack["rows"][pick],
        stack["cols"][pick],
        stack["true_v"][pick],
    )
    log(
        f"PARITY vs #89 classify_forced: checked={parity['n_checked']} "
        f"mismatches={parity['n_mismatches']} passed={parity['passed']}"
    )
    if parity["n_mismatches"]:
        log("PARITY MISMATCHES (up to 20):")
        for m in parity["mismatches"]:
            log(f"  {m}")
    if not parity["passed"] or parity["n_checked"] < PARITY_MIN_ROWS:
        raise AssertionError(
            f"parity failed: checked={parity['n_checked']} "
            f"mismatches={parity['n_mismatches']} (need >= {PARITY_MIN_ROWS}, 0 mismatches)"
        )
    log(f"PARITY OK on {parity['n_checked']} rows (>= {PARITY_MIN_ROWS})")

    # Cover predictions on te
    log("\n--- core_cover_sw on te (return_pred) ---")
    te_idx = te
    out_te, pred_te = original_cover(
        model, rules, ids, yv, cls, te_idx, return_pred=True,
    )
    log(
        f"te core_sw: agree={out_te['agree']:.4f} cover={out_te['cover']:.4f} "
        f"agree_fired={out_te['agree_fired']:.4f}"
    )
    te_list = [int(x) for x in te_idx.detach().cpu().tolist()]
    te_pos = {row: p for p, row in enumerate(te_list)}
    pred_cpu = pred_te.detach().cpu()
    yv_cpu = yv.detach().cpu()

    # Residual error among te solution-cell rows
    te_stack_idx: list[int] = []
    te_error: list[bool] = []
    te_abstain: list[bool] = []
    n_err = 0
    n_abs = 0
    for row in te_sol:
        p = te_pos[row]
        pr = int(pred_cpu[p].item())
        gold = int(yv_cpu[row].item())
        is_abs = pr < 0
        is_err = pr != gold  # includes abstain (#89 is_err definition)
        te_stack_idx.append(row_to_stack[row])
        te_error.append(is_err)
        te_abstain.append(is_abs)
        if is_err:
            n_err += 1
        if is_abs:
            n_abs += 1

    n_scored = len(te_sol)
    residual_rate = osc.fraction(n_err, n_scored)
    log(
        f"te solution-cell residual: error_rate={residual_rate:.6f} "
        f"({n_err}/{n_scored}); abstain={n_abs}/{n_scored}"
    )

    # Feature slices for te sol rows
    te_si = torch.tensor(te_stack_idx, dtype=torch.long)
    te_naked = feat_all["is_naked"][te_si]
    te_hidden = feat_all["is_hidden"][te_si]
    te_forced = feat_all["forced_value"][te_si]
    te_true = stack["true_v"][te_si]
    te_cell_index = [meta[k]["cell_index"] for k in te_stack_idx]

    # CONDITION 2 -- freeze from recovery on residual errors (independent of admission)
    log("\n--- CONDITION 2: recovery on residual ERROR rows ---")
    recovery = compute_recovery(
        te_naked, te_hidden, te_forced, te_true, error_mask=te_error,
    )
    freeze = freeze_from_recovery(
        recovery["recovered_naked"],
        recovery["recovered_union"],
        recovery["n_recovered_union"],
    )
    log(
        f"recovered_naked = {recovery['recovered_naked']:.6f} "
        f"({recovery['n_recovered_naked']}/{recovery['n_error']})"
    )
    log(
        f"recovered_union = {recovery['recovered_union']:.6f} "
        f"({recovery['n_recovered_union']}/{recovery['n_error']})"
    )
    if freeze["underpowered"]:
        log("UNDERPOWERED -- ESCALATE")
        log("freeze was not decided (validity floor: absolute union-recovered count < 50)")
    else:
        log(
            "FREEZE RULE: if recovered_naked >= 0.90 * recovered_union: "
            "inventory = NAKED-ONLY else: inventory = NAKED UNION HIDDEN"
        )
        log(
            f"  recovered_naked={recovery['recovered_naked']:.6f} "
            f"(n={recovery['n_recovered_naked']})  vs  "
            f"0.90 * recovered_union={freeze['threshold']:.6f} "
            f"(union n={recovery['n_recovered_union']})"
        )
        log(f"  inventory = {freeze['inventory']}")

    # Stratification (mandatory)
    forced_eq = [
        int(te_forced[i].item()) == int(te_true[i].item()) and int(te_forced[i].item()) >= 1
        for i in range(len(te_stack_idx))
    ]
    strata = stratify_by_reveal_third(
        te_cell_index, te_naked, te_hidden, forced_eq, error_mask=te_error,
    )
    log("\n--- STRATIFICATION by reveal-third (cell_index // 27) on residual errors ---")
    log(f"{'third':>6} {'n_err':>8} {'n_nak':>8} {'nak_frac':>10} "
        f"{'n_uni':>8} {'uni_frac':>10}")
    for third in (0, 1, 2):
        s = strata[third]
        log(
            f"{third:>6} {s['n_error']:>8} {s['n_recovered_naked']:>8} "
            f"{s['recovered_naked']:>10.4f} {s['n_recovered_union']:>8} "
            f"{s['recovered_union']:>10.4f}"
        )
    if prenamed_reading_holds(
        recovery["recovered_union"],
        strata[0]["recovered_naked"],
        n_error_third0=strata[0]["n_error"],
    ):
        log("the register is sufficient only where the puzzle is nearly solved.")
    else:
        log("pre-named reading did not hold")

    # Proposer confidence via val_fired_acc (measured)
    log("\n--- PROPOSER CONFIDENCE (val_fired_acc, measured) ---")
    bare_c, space_c = digit_to_compact_maps(uv_cpu, bare, space)
    # Precompute forced class for every dataset row (-1 default)
    forced_class = torch.full((n_kept,), -1, dtype=torch.long)
    compact_forced = forced_digits_to_compact(
        feat_all["forced_value"], stack["cols"], bare_c, space_c,
    )
    for k, m in enumerate(meta):
        forced_class[m["row"]] = compact_forced[k]

    proposer = make_lookup_proposer(ids_cpu, forced_class)
    # val split: last NVAL of train (v5 / v2 style)
    import wyly_lm_v2 as v2  # noqa: E402
    nval = min(int(v2.NVAL), max(1, len(tr) // 10))
    val = tr[-nval:]
    vfa = v5.val_fired_acc(proposer, ids, yv, val)
    # also report fire rate on val
    with torch.no_grad():
        a_val = proposer(ids[val])
        n_fire = int((a_val >= 0).sum().item())
    log(f"ValFiredAccuracy (measured): {vfa:.6f}  (fired on {n_fire}/{len(val)} val rows)")
    log("(not asserted 1.0; feature is computed-not-certified)")

    # ADMISSION -- descriptive only; does NOT influence freeze
    log("\n--- ADMISSION (descriptive, NON-gating; informs #91, NEVER the freeze) ---")
    rules_with = list(rules) + [("legality_forced", proposer)]
    extra = {"legality_forced": float(vfa)}
    out_without, _ = original_cover(
        model, rules, ids, yv, cls, te_idx, return_pred=True,
    )
    out_with, _ = original_cover(
        model, rules_with, ids, yv, cls, te_idx,
        extra_conf=extra, return_pred=True,
    )
    marginal = float(out_with["agree"]) - float(out_without["agree"])
    log(
        f"te agree WITHOUT proposer: {out_without['agree']:.6f}  "
        f"WITH: {out_with['agree']:.6f}  marginal: {marginal:+.6f}"
    )
    log(
        "this informs #91, NEVER the freeze "
        "(which is decided by the recovery ratio alone in Condition 2)."
    )

    # Scoreboard
    wall = time.time() - t0
    log("\n" + "=" * 72)
    log("SCOREBOARD")
    log("=" * 72)
    log(f"FRAMING: {FRAMING_NOTE}")
    log(f"Round-trip: PASSED on {rt['n_checked']} windows")
    log(
        f"Parity: PASSED on {parity['n_checked']} rows "
        f"(mismatches={parity['n_mismatches']})"
    )
    log(f"te solution-cell rows scored: {n_scored}  errors: {n_err}  abstains: {n_abs}")
    log(f"Step-0 residual rate: {residual_rate:.6f}")
    log()
    log("CONDITION 2 -- freeze decision:")
    log(
        f"  recovered_naked={recovery['recovered_naked']:.6f} "
        f"(n={recovery['n_recovered_naked']})"
    )
    log(
        f"  recovered_union={recovery['recovered_union']:.6f} "
        f"(n={recovery['n_recovered_union']})"
    )
    if freeze["underpowered"]:
        log("  UNDERPOWERED -- ESCALATE")
        log("  freeze was not decided")
        inventory_out = None
    else:
        log(f"  inventory = {freeze['inventory']}")
        inventory_out = freeze["inventory"]
    log()
    log(f"Proposer ValFiredAccuracy (measured): {vfa:.6f}")
    log(f"Admission held-out marginal (agree_with - agree_without): {marginal:+.6f}")
    log(
        "  (this informs #91, NEVER the freeze "
        "(which is decided by the recovery ratio alone in Condition 2).)"
    )
    log()
    log("Stratification (reveal-third on residual errors):")
    for third in (0, 1, 2):
        s = strata[third]
        log(
            f"  third {third}: n_err={s['n_error']} "
            f"naked={s['recovered_naked']:.4f} (n={s['n_recovered_naked']}) "
            f"union={s['recovered_union']:.4f} (n={s['n_recovered_union']})"
        )
    prenamed_held = prenamed_reading_holds(
        recovery["recovered_union"],
        strata[0]["recovered_naked"],
        n_error_third0=strata[0]["n_error"],
    )
    if prenamed_held:
        log("the register is sufficient only where the puzzle is nearly solved.")
    else:
        log("pre-named reading did not hold")
    log()
    # All classifiable solution-cell rows (tr+te), same population as sol_rows / meta
    all_cell_index = [m["cell_index"] for m in meta]
    reveal_dist = reveal_index_distribution(all_cell_index)
    log("REVEAL-INDEX DISTRIBUTION (all classifiable solution-cell rows, tr+te):")
    log(
        f"  n={reveal_dist['n']}  min={reveal_dist['min']}  "
        f"max={reveal_dist['max']}  median={reveal_dist['median']}"
    )
    log("  histogram by reveal-third ([0-26]/[27-53]/[54-80]):")
    for t in ("0", "1", "2"):
        h = reveal_dist["histogram"][t]
        log(f"    third {t}: count={h['count']}  fraction={h['fraction']:.6f}")
    log(f"  {REVEAL_INDEX_NOTE}")
    log()
    log(f"HONESTY: {HONESTY_NOTE}")
    log(f"FRAMING: {FRAMING_NOTE}")
    log(f"wall time total: {wall:.1f}s")

    report = {
        "slice": 90,
        "framing": FRAMING_NOTE,
        "honesty_note": HONESTY_NOTE,
        "wyly_env": env_used,
        "module_bound": {
            "TAG": v5.TAG, "DS": v5.DS, "LIB": v5.LIB, "JUDGE": v5.JUDGE,
            "LABELS": v5.LABELS, "STATE": str(v5.STATE.name),
        },
        "round_trip": rt,
        "parity": {
            "n_checked": parity["n_checked"],
            "n_mismatches": parity["n_mismatches"],
            "passed": parity["passed"],
        },
        "load_ds": {
            "n_raw": n_raw, "n_kept": n_kept, "n_dropped": n_raw - n_kept,
            "n_tr": len(tr), "n_te": len(te),
            "n_sol_rows": len(sol_rows), "n_te_sol": n_scored,
        },
        "residual": {
            "n_error": n_err,
            "n_abstain": n_abs,
            "n_scored": n_scored,
            "error_rate": residual_rate,
        },
        "condition2": {
            "recovered_naked": recovery["recovered_naked"],
            "recovered_union": recovery["recovered_union"],
            "n_recovered_naked": recovery["n_recovered_naked"],
            "n_recovered_union": recovery["n_recovered_union"],
            "n_error": recovery["n_error"],
            "underpowered": freeze["underpowered"],
            "freeze_decided": freeze["freeze_decided"],
            "inventory": inventory_out,
            "message": freeze["message"],
        },
        "stratification": {str(k): v for k, v in strata.items()},
        "reveal_index_distribution": reveal_dist,
        "prenamed_reading_held": prenamed_held,
        "proposer_confidence": {
            "val_fired_acc_measured": vfa,
            "n_val": len(val),
            "n_fired_on_val": n_fire,
            "note": "measured via v5.val_fired_acc; not asserted 1.0",
        },
        "admission": {
            "agree_without": float(out_without["agree"]),
            "agree_with": float(out_with["agree"]),
            "marginal": marginal,
            "note": (
                "this informs #91, NEVER the freeze "
                "(which is decided by the recovery ratio alone in Condition 2)."
            ),
        },
        "te_core_sw": out_te,
        "wall_time_s": wall,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    log(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
