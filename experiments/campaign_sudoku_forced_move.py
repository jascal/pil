"""Slice #89 -- sudoku forced-move residual probe (measurement only, no register).

Plain-numpy forcedness (naked / hidden singles) from the in-window ground-truth grid,
gated against the LABELS=corpus cover residual.  Not a certified feature.
"""
from __future__ import annotations

import hashlib
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

# v5 binds recipe constants at import time -- setdefaults before the import.
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

import wyly_lm_v5 as v5  # noqa: E402

OUTPUT = REPO / "data" / "sudoku_forced_move.json"
ROUND_TRIP_N = 500
ABORT_RESIDUAL = 0.02
VERDICT_FIRES = 0.30
VERDICT_DEAD = 0.10
L_WINDOW = 256
SOL_BODY = 90  # 9 rows * 10 tokens (9 cells + newline)
PUZ_BODY = 90

HONESTY_NOTE = (
    "plain-numpy forcedness from the known in-window ground-truth solution/puzzle grid; "
    "NOT a certified feature; NOT a register; LABELS=corpus (stream next-token gold); "
    "overlap-adjusted subset reported; measurement only (slice #89)."
)

DESIGN_REGISTERED = """\
## Design (registered)

1. GRID RECONSTRUCTION: for each window, locate the PUZZLE and SOLUTION headers (token search),
   recover the 9x9 puzzle and the emitted-so-far solution cells by affine indexing off the
   headers; token->digit normalization (a value-map: '3' and ' 3' -> 3; '.' and ' .' -> empty).
   ROUND-TRIP TEST (required): reconstructed grid re-serialized == the window's cell tokens, on a
   500-window sample (assert; report mismatches).
2. STEP 0 abort/contamination checks (run and report BEFORE the anatomy):
   (a) cover solution-cell error rate under LABELS=corpus. If < 0.02 -> print "ABORT: residual
       too small" and STOP (report the rate).
   (b) board-level train/test overlap: hash each full solution grid; count test boards whose grid
       also appears among train-window boards; report overlap fraction. Recompute the final gate
       metric ALSO on the non-overlapping-board subset.
3. CLASSIFY each SOLUTION-cell prediction row from its in-window grid (puzzle clues + solution
   cells emitted before this position) under all-different over row/col/box peers: candidate set
   = digits 1-9 not used by any filled peer. NAKED = |candidates|==1. HIDDEN = the cell's true
   value v is a candidate AND v has exactly one legal cell in at least one of the cell's three
   units (row/col/box) among still-empty cells. UNION = naked OR hidden. NON-FORCED = neither.
   (Unit-test all four on hand-built grids.)
4. RESIDUAL: run the sudoku cover (core_cover_sw return_pred, LABELS=corpus) -> per-row pred;
   error rows = solution-cell rows where pred != gold (also track abstain). Reuse the MinedGates.
   mine residual pattern.
5. METRICS (print + JSON): base rates {naked, hidden, union} over all solution-cell rows;
   residual error rows' {naked, hidden, union} fractions (enrichment vs base = descriptive);
   ABSOLUTE union-recoverable fraction of residual ERROR rows = the GATE; recoverable-as-
   core_sw-points = (union-recoverable error rows)/(total scored rows); all of the above ALSO on
   the non-overlapping-board subset.
6. VERDICT (verbatim, on the ABSOLUTE union-recoverable fraction of residual error rows): FIRES
   if >= 0.30; DEAD if < 0.10; MIDDLE (escalate) if 0.10-0.30. Print the naked/hidden/union
   breakdown as the register-scoping guide.
"""

CLASSIFICATION_DEFS = """\
CLASSIFICATION DEFINITIONS (verbatim):
  NAKED = |candidates|==1, where candidates = digits 1-9 not used by any filled peer
          (row/col/box) under all-different.
  HIDDEN = the cell's true value v is a candidate AND v has exactly one legal cell in at
           least one of the cell's three units (row/col/box) among still-empty cells.
  UNION = naked OR hidden.
  NON-FORCED = neither.
"""


# ---------------------------------------------------------------------------
# Token value-map and header patterns
# ---------------------------------------------------------------------------

def bare_digit_token_ids() -> dict[int, int]:
    """Digit -> bare col-0 token id (1..9). Built from the same codec as v5."""
    return {d: _encode_one(str(d)) for d in range(1, 10)}


def space_digit_token_ids() -> dict[int, int]:
    return {d: _encode_one(f" {d}") for d in range(1, 10)}


def _encode_one(text: str) -> int:
    ts = v5.load_codec()
    ids = ts.encode(text)
    if len(ids) != 1:
        raise RuntimeError(f"expected single token for {text!r}, got {ids}")
    return int(ids[0])


def build_value_map(codec=None) -> dict[int, int | None]:
    """Raw token id -> digit 1..9, or None for empty ('.' / ' .'). Unknown tokens omitted."""
    ts = codec if codec is not None else v5.load_codec()
    out: dict[int, int | None] = {}
    for d in range(1, 10):
        bare = ts.encode(str(d))
        space = ts.encode(f" {d}")
        if len(bare) == 1:
            out[int(bare[0])] = d
        if len(space) == 1:
            out[int(space[0])] = d
    for text, val in ((".", None), (" .", None)):
        ids = ts.encode(text)
        if len(ids) == 1:
            out[int(ids[0])] = val
    return out


def token_to_digit(token_id: int, value_map: dict[int, int | None]) -> int | None | type[...]:
    """Return 1..9, None (empty), or Ellipsis if token is not a cell value token."""
    if token_id not in value_map:
        return ...
    return value_map[token_id]


def digit_to_token(digit: int | None, col: int, bare: dict[int, int], space: dict[int, int],
                   dot_bare: int, dot_space: int) -> int:
    """Serialize one cell: col0 bare, cols1-8 space-prefixed; digit None/0 -> empty."""
    if digit is None or digit == 0:
        return dot_bare if col == 0 else dot_space
    return bare[int(digit)] if col == 0 else space[int(digit)]


def header_patterns(codec=None) -> tuple[list[int], list[int]]:
    """Canonical full-header token-id subsequences for PUZZLE\\n and SOLUTION\\n."""
    ts = codec if codec is not None else v5.load_codec()
    puz = [int(x) for x in ts.encode("PUZZLE\n")]
    sol = [int(x) for x in ts.encode("SOLUTION\n")]
    return puz, sol


def find_subsequence(seq: list[int] | np.ndarray, pattern: list[int]) -> int:
    """Exact sublist search; return start index or -1."""
    if not pattern:
        return -1
    n, m = len(seq), len(pattern)
    if m > n:
        return -1
    # Fast path for small patterns
    p0 = pattern[0]
    for i in range(n - m + 1):
        if seq[i] != p0:
            continue
        if all(seq[i + k] == pattern[k] for k in range(1, m)):
            return i
    return -1


def locate_solution_header(token_ids: list[int], sol_pat: list[int]) -> int:
    return find_subsequence(token_ids, sol_pat)


def locate_puzzle_header(token_ids: list[int], puz_pat: list[int]) -> int:
    return find_subsequence(token_ids, puz_pat)


# ---------------------------------------------------------------------------
# Grid reconstruction (affine phase off headers)
# ---------------------------------------------------------------------------

def extract_grid_body(token_ids: list[int], body_start: int, value_map: dict[int, int | None],
                      n_rows: int = 9) -> tuple[list[list[int]], list[int]]:
    """Read n_rows of (9 cells + newline) starting at body_start.

    Returns (grid 9x9 with 0=empty, list of cell token ids in row-major order).
    Raises ValueError if the body would fall outside the window or tokens fail the value-map.
    """
    need = n_rows * 10
    if body_start < 0 or body_start + need > len(token_ids):
        raise ValueError("grid body not fully contained in window")
    grid = [[0] * 9 for _ in range(9)]
    cell_tokens: list[int] = []
    for r in range(n_rows):
        row_base = body_start + r * 10
        for c in range(9):
            tid = int(token_ids[row_base + c])
            cell_tokens.append(tid)
            if tid not in value_map:
                raise ValueError(f"non-cell token {tid} at body ({r},{c})")
            d = value_map[tid]
            grid[r][c] = 0 if d is None else int(d)
        # trailing newline expected but not required for digit recovery
    return grid, cell_tokens


def reconstruct_from_window(
    token_ids: list[int],
    sol_pat: list[int],
    puz_pat: list[int],
    value_map: dict[int, int | None],
) -> dict[str, Any] | None:
    """Locate headers and recover puzzle + in-window solution body.

    Returns None if the window lacks a fully-contained SOLUTION header.
    Puzzle body is taken as the 90 tokens immediately before the SOLUTION header
    when present (affine relationship); otherwise via PUZZLE header if fully in view.
    """
    sol_at = locate_solution_header(token_ids, sol_pat)
    if sol_at < 0:
        return None
    sol_body_start = sol_at + len(sol_pat)
    # Puzzle body ends at sol_at (SOLUTION header follows final puzzle-row newline).
    puz_body_start = sol_at - PUZ_BODY
    puzzle = None
    puz_tokens: list[int] = []
    puz_source = None
    if puz_body_start >= 0:
        try:
            puzzle, puz_tokens = extract_grid_body(token_ids, puz_body_start, value_map)
            puz_source = "before_solution"
        except ValueError:
            puzzle = None
    if puzzle is None:
        puz_at = locate_puzzle_header(token_ids, puz_pat)
        if puz_at >= 0:
            try:
                puzzle, puz_tokens = extract_grid_body(
                    token_ids, puz_at + len(puz_pat), value_map
                )
                puz_source = "puzzle_header"
            except ValueError:
                puzzle = None

    # Solution: as many full rows as fit; partial last row allowed for emitted-so-far.
    n_sol_tokens = len(token_ids) - sol_body_start
    solution = [[0] * 9 for _ in range(9)]
    sol_tokens: list[int] = []
    n_full_cells = 0
    if n_sol_tokens > 0:
        max_rows = min(9, n_sol_tokens // 10)
        extra = n_sol_tokens - max_rows * 10
        for r in range(max_rows):
            row_base = sol_body_start + r * 10
            for c in range(9):
                tid = int(token_ids[row_base + c])
                sol_tokens.append(tid)
                if tid not in value_map:
                    return None
                d = value_map[tid]
                solution[r][c] = 0 if d is None else int(d)
                n_full_cells += 1
        # partial row (cells only, no newline yet)
        if max_rows < 9 and extra > 0:
            row_base = sol_body_start + max_rows * 10
            for c in range(min(9, extra)):
                if extra > c:  # cells before newline slot
                    if c == 9:
                        break
                    tid = int(token_ids[row_base + c])
                    # if this slot is the newline position (c would be wrong): off%10==9 is newline
                    sol_tokens.append(tid)
                    if tid not in value_map:
                        # newline or other -- stop partial
                        break
                    d = value_map[tid]
                    solution[max_rows][c] = 0 if d is None else int(d)
                    n_full_cells += 1

    full_solution = n_sol_tokens >= SOL_BODY
    if full_solution:
        try:
            solution, sol_tokens = extract_grid_body(token_ids, sol_body_start, value_map)
            n_full_cells = 81
        except ValueError:
            full_solution = False

    return {
        "sol_header_at": sol_at,
        "sol_body_start": sol_body_start,
        "puzzle": puzzle,
        "puzzle_tokens": puz_tokens,
        "puzzle_source": puz_source,
        "solution": solution,
        "solution_tokens": sol_tokens,
        "n_sol_tokens_in_window": n_sol_tokens,
        "full_solution": full_solution,
        "n_solution_cells_in_window": n_full_cells,
    }


def serialize_grid_cell_tokens(
    grid: list[list[int]],
    bare: dict[int, int],
    space: dict[int, int],
    dot_bare: int,
    dot_space: int,
) -> list[int]:
    """Re-serialize 9x9 (0=empty) to the 81 cell token ids (no newlines)."""
    out: list[int] = []
    for r in range(9):
        for c in range(9):
            d = grid[r][c]
            out.append(digit_to_token(
                None if d == 0 else d, c, bare, space, dot_bare, dot_space
            ))
    return out


def affine_cell_index(header_body_start: int, r: int, c: int) -> int:
    """Token index of cell (r,c) given the first cell of the body."""
    return header_body_start + r * 10 + c


def target_solution_rc(token_ids: list[int], sol_pat: list[int], L: int = L_WINDOW) -> tuple[int, int] | None:
    """If the next token after the window is a solution-grid digit cell, return (r, c)."""
    sol_at = locate_solution_header(token_ids, sol_pat)
    if sol_at < 0:
        return None
    sol_body_start = sol_at + len(sol_pat)
    off = L - sol_body_start  # offset of TARGET in the solution section
    if off < 0 or off >= SOL_BODY:
        return None
    if off % 10 == 9:
        return None  # newline between rows
    r, c = divmod(off, 10)
    if r >= 9 or c >= 9:
        return None
    return r, c


def board_state_before_cell(
    puzzle: list[list[int]],
    solution: list[list[int]],
    r: int,
    c: int,
) -> list[list[int]]:
    """Puzzle clues + solution cells emitted before (r,c) in solution-section order."""
    board = [row[:] for row in puzzle]
    target_off = r * 10 + c
    for rr in range(9):
        for cc in range(9):
            off = rr * 10 + cc
            if off < target_off:
                v = solution[rr][cc]
                if v != 0:
                    board[rr][cc] = v
    return board


def board_hash(grid: list[list[int]]) -> str:
    """Stable hash of a 9x9 digit grid (0-9)."""
    flat = ",".join(str(grid[r][c]) for r in range(9) for c in range(9))
    return hashlib.sha256(flat.encode("ascii")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Forced-move classification (plain numpy / pure python)
# ---------------------------------------------------------------------------

def _filled_in_row(grid: list[list[int]], r: int) -> set[int]:
    return {grid[r][j] for j in range(9) if grid[r][j] != 0}


def _filled_in_col(grid: list[list[int]], c: int) -> set[int]:
    return {grid[i][c] for i in range(9) if grid[i][c] != 0}


def _filled_in_box(grid: list[list[int]], r: int, c: int) -> set[int]:
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return {grid[i][j] for i in range(br, br + 3) for j in range(bc, bc + 3)
            if grid[i][j] != 0}


def cell_candidates(grid: list[list[int]], r: int, c: int) -> set[int]:
    """Digits 1-9 not used by any filled peer in row/col/box. Cell itself treated as empty."""
    if grid[r][c] != 0:
        g2 = [row[:] for row in grid]
        g2[r][c] = 0
        grid = g2
    used = _filled_in_row(grid, r) | _filled_in_col(grid, c) | _filled_in_box(grid, r, c)
    return set(range(1, 10)) - used


def is_naked_single(grid: list[list[int]], r: int, c: int) -> bool:
    return len(cell_candidates(grid, r, c)) == 1


def is_hidden_single(grid: list[list[int]], r: int, c: int, true_v: int) -> bool:
    """HIDDEN: true_v is a candidate AND uniquely placeable in at least one unit."""
    if true_v not in cell_candidates(grid, r, c):
        return False
    # Treat (r,c) as empty
    g = [row[:] for row in grid]
    g[r][c] = 0

    def empties_in_row(rr: int) -> list[tuple[int, int]]:
        return [(rr, j) for j in range(9) if g[rr][j] == 0]

    def empties_in_col(cc: int) -> list[tuple[int, int]]:
        return [(i, cc) for i in range(9) if g[i][cc] == 0]

    def empties_in_box(rr: int, cc: int) -> list[tuple[int, int]]:
        br, bc = 3 * (rr // 3), 3 * (cc // 3)
        return [(i, j) for i in range(br, br + 3) for j in range(bc, bc + 3)
                if g[i][j] == 0]

    def n_legal(cells: list[tuple[int, int]], v: int) -> int:
        return sum(1 for i, j in cells if v in cell_candidates(g, i, j))

    if n_legal(empties_in_row(r), true_v) == 1:
        return True
    if n_legal(empties_in_col(c), true_v) == 1:
        return True
    if n_legal(empties_in_box(r, c), true_v) == 1:
        return True
    return False


def classify_forced(grid: list[list[int]], r: int, c: int, true_v: int) -> dict[str, bool]:
    """Return NAKED / HIDDEN / UNION / NON-FORCED flags for one cell."""
    naked = is_naked_single(grid, r, c)
    hidden = is_hidden_single(grid, r, c, true_v)
    union = naked or hidden
    return {
        "naked": naked,
        "hidden": hidden,
        "union": union,
        "non_forced": not union,
    }


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def round_trip_window(
    token_ids: list[int],
    sol_pat: list[int],
    puz_pat: list[int],
    value_map: dict[int, int | None],
    bare: dict[int, int],
    space: dict[int, int],
    dot_bare: int,
    dot_space: int,
) -> dict[str, Any]:
    """Reconstruct grids and compare re-serialized cell tokens to the window's tokens."""
    rec = reconstruct_from_window(token_ids, sol_pat, puz_pat, value_map)
    if rec is None:
        return {"ok": False, "reason": "no_solution_header", "mismatches": 1}
    mismatches = 0
    details: list[str] = []
    if rec["puzzle"] is not None and rec["puzzle_tokens"]:
        ser = serialize_grid_cell_tokens(
            rec["puzzle"], bare, space, dot_bare, dot_space
        )
        # puzzle_tokens may be 81 cells
        pt = rec["puzzle_tokens"]
        if len(pt) == 81 and ser != pt:
            mismatches += sum(a != b for a, b in zip(ser, pt, strict=True))
            details.append("puzzle_cells")
        elif len(pt) != 81:
            details.append(f"puzzle_token_count={len(pt)}")
    # solution cells present in window (full rows only for strict round-trip of complete rows)
    n_sol = rec["n_sol_tokens_in_window"]
    if n_sol >= 10 and rec["solution"] is not None:
        full_rows = min(9, n_sol // 10)
        if full_rows > 0:
            sub = [rec["solution"][r][:] for r in range(full_rows)]
            # pad to 9x9 for serializer then take first full_rows*9
            padded = sub + [[0] * 9 for _ in range(9 - full_rows)]
            ser = serialize_grid_cell_tokens(padded, bare, space, dot_bare, dot_space)
            ser = ser[: full_rows * 9]
            # extract expected tokens from window
            body = rec["sol_body_start"]
            expected: list[int] = []
            for r in range(full_rows):
                for c in range(9):
                    expected.append(int(token_ids[body + r * 10 + c]))
            if ser != expected:
                mismatches += sum(a != b for a, b in zip(ser, expected, strict=True))
                details.append("solution_cells")
    ok = mismatches == 0 and rec["sol_header_at"] >= 0
    return {
        "ok": ok,
        "mismatches": mismatches,
        "details": details,
        "sol_header_at": rec["sol_header_at"],
        "full_solution": rec["full_solution"],
    }


def run_round_trip_sample(
    raw_ids: torch.Tensor,
    n: int = ROUND_TRIP_N,
    seed: int = 0,
) -> dict[str, Any]:
    """Round-trip on up to n windows that contain a full SOLUTION header + first cell."""
    codec = v5.load_codec()
    puz_pat, sol_pat = header_patterns(codec)
    value_map = build_value_map(codec)
    bare = {d: int(codec.encode(str(d))[0]) for d in range(1, 10)}
    space = {d: int(codec.encode(f" {d}")[0]) for d in range(1, 10)}
    dot_bare = int(codec.encode(".")[0])
    dot_space = int(codec.encode(" .")[0])

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(raw_ids))
    checked = 0
    total_mismatch_cells = 0
    failed_windows = 0
    skipped = 0
    for idx in order:
        if checked >= n:
            break
        seq = [int(x) for x in raw_ids[int(idx)].tolist()]
        sol_at = locate_solution_header(seq, sol_pat)
        if sol_at < 0:
            skipped += 1
            continue
        # require at least the first solution cell in-window
        if sol_at + len(sol_pat) >= len(seq):
            skipped += 1
            continue
        result = round_trip_window(
            seq, sol_pat, puz_pat, value_map, bare, space, dot_bare, dot_space
        )
        checked += 1
        if not result["ok"]:
            failed_windows += 1
            total_mismatch_cells += int(result["mismatches"])
    return {
        "n_checked": checked,
        "n_failed_windows": failed_windows,
        "n_mismatch_cells": total_mismatch_cells,
        "n_skipped": skipped,
        "passed": failed_windows == 0 and checked > 0,
    }


# ---------------------------------------------------------------------------
# Metrics / verdict / abort
# ---------------------------------------------------------------------------

def fraction(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def compute_rate_block(flags: list[dict[str, bool]]) -> dict[str, float]:
    n = len(flags)
    return {
        "n": n,
        "naked": fraction(sum(1 for f in flags if f["naked"]), n),
        "hidden": fraction(sum(1 for f in flags if f["hidden"]), n),
        "union": fraction(sum(1 for f in flags if f["union"]), n),
        "non_forced": fraction(sum(1 for f in flags if f["non_forced"]), n),
        "n_naked": sum(1 for f in flags if f["naked"]),
        "n_hidden": sum(1 for f in flags if f["hidden"]),
        "n_union": sum(1 for f in flags if f["union"]),
        "n_non_forced": sum(1 for f in flags if f["non_forced"]),
    }


def compute_forced_metrics(
    all_flags: list[dict[str, bool]],
    error_mask: list[bool],
    n_scored: int,
) -> dict[str, Any]:
    """Base rates over all_flags; residual rates over error rows; gate metrics.

    error_mask aligns with all_flags (True = residual error row).
    n_scored = total scored solution-cell rows (denominator for core_sw-points).
    """
    base = compute_rate_block(all_flags)
    err_flags = [f for f, e in zip(all_flags, error_mask, strict=True) if e]
    residual = compute_rate_block(err_flags)
    n_err = len(err_flags)
    n_union_err = sum(1 for f in err_flags if f["union"])
    gate = fraction(n_union_err, n_err)
    core_pts = fraction(n_union_err, n_scored)
    return {
        "base": base,
        "residual_error": residual,
        "n_error": n_err,
        "n_union_recoverable_error": n_union_err,
        "gate_union_recoverable_frac": gate,
        "recoverable_as_core_sw_points": core_pts,
        "enrichment": {
            "naked": residual["naked"] - base["naked"],
            "hidden": residual["hidden"] - base["hidden"],
            "union": residual["union"] - base["union"],
        },
    }


def verdict_from_gate(gate_frac: float) -> str:
    """FIRES if >= 0.30; DEAD if < 0.10; MIDDLE (escalate) if 0.10-0.30."""
    if gate_frac >= VERDICT_FIRES:
        return "FIRES"
    if gate_frac < VERDICT_DEAD:
        return "DEAD"
    return "MIDDLE (escalate)"


def abort_if_residual_too_small(error_rate: float) -> str | None:
    """Return abort message if residual error rate < 0.02, else None."""
    if error_rate < ABORT_RESIDUAL:
        return f"ABORT: residual too small (error_rate={error_rate:.6f} < {ABORT_RESIDUAL})"
    return None


# ---------------------------------------------------------------------------
# Cover regeneration (expensive -- not at import time)
# ---------------------------------------------------------------------------

def run_cover_regeneration() -> tuple[Any, list, Any]:
    """Run v5.main() with a capture wrapper; return (model, rules, original_core_cover_sw)."""
    original = v5.core_cover_sw
    captured: dict[str, Any] = {}

    def capture(model, rules, ids, yv, cls, idxs, **kwargs):
        captured["model"] = model
        captured["rules"] = list(rules)
        return original(model, rules, ids, yv, cls, idxs, **kwargs)

    v5.core_cover_sw = capture
    try:
        v5.main()
    finally:
        v5.core_cover_sw = original
    if not captured:
        raise RuntimeError("core_cover_sw capture wrapper was never called")
    return captured["model"], captured["rules"], original


def raw_window_from_compact(ids_row: torch.Tensor, uv: torch.Tensor) -> list[int]:
    return [int(x) for x in uv[ids_row.detach().cpu()].tolist()]


def gold_raw_token(y_i: int, cls: torch.Tensor, uv: torch.Tensor) -> int:
    return int(uv[int(cls[int(y_i)].item())].item())


def digit_from_raw(tid: int, value_map: dict[int, int | None]) -> int | None:
    if tid not in value_map:
        return None
    d = value_map[tid]
    return d  # may be None for empty


# ---------------------------------------------------------------------------
# Main campaign
# ---------------------------------------------------------------------------

def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    t0 = time.time()
    log("=" * 72)
    log("SLICE #89 -- sudoku forced-move residual probe (measurement only)")
    log("=" * 72)
    log(DESIGN_REGISTERED)
    log(CLASSIFICATION_DEFS)
    log(f"HONESTY: {HONESTY_NOTE}")
    log()
    log("WYLY_* env used for this run:")
    env_used = {k: os.environ.get(k, "") for k in WYLY_ENV}
    for k, v in env_used.items():
        log(f"  {k}={v}")
    log(f"  (module-bound) TAG={v5.TAG} DS={v5.DS} LIB={v5.LIB} JUDGE={v5.JUDGE} "
        f"LABELS={v5.LABELS} SWCOVER={v5.SWCOVER} STATE={v5.STATE.name}")
    log()

    codec = v5.load_codec()
    puz_pat, sol_pat = header_patterns(codec)
    value_map = build_value_map(codec)
    bare = {d: int(codec.encode(str(d))[0]) for d in range(1, 10)}
    space = {d: int(codec.encode(f" {d}")[0]) for d in range(1, 10)}
    log(f"header patterns: PUZZLE={puz_pat} SOLUTION={sol_pat}")
    log(f"value_map size={len(value_map)} bare={bare} space={space}")

    # --- Round-trip on raw windows (before expensive regen) ---
    raw = torch.load(v5.DATA, map_location="cpu")
    raw_ids = raw["kept_ids"]
    log(f"\nRaw windows: {tuple(raw_ids.shape)} from {v5.DATA.name}")
    rt = run_round_trip_sample(raw_ids, n=ROUND_TRIP_N, seed=0)
    log(f"ROUND-TRIP ({ROUND_TRIP_N}-window sample): checked={rt['n_checked']} "
        f"failed_windows={rt['n_failed_windows']} mismatch_cells={rt['n_mismatch_cells']} "
        f"skipped={rt['n_skipped']} passed={rt['passed']}")
    if not rt["passed"]:
        raise AssertionError(
            f"round-trip failed: {rt['n_failed_windows']} windows, "
            f"{rt['n_mismatch_cells']} cell mismatches"
        )

    # --- Expensive regeneration ---
    log("\n--- run_cover_regeneration (v5.main) ---")
    t_reg = time.time()
    model, rules, original_cover = run_cover_regeneration()
    log(f"regeneration wall time: {time.time() - t_reg:.1f}s")
    log(f"captured rules ({len(rules)}): {[n for n, _ in rules]}")

    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    n_raw = int(raw_ids.shape[0])
    n_kept = int(ids.shape[0])
    log(f"\nload_ds: N={n_kept} (raw={n_raw}, dropped={n_raw - n_kept}) "
        f"tr={len(tr)} te={len(te)} vocab_compact={len(uv)}")
    log(f"top-V filter drop: {n_raw - n_kept} rows "
        f"({fraction(n_raw - n_kept, n_raw):.4%}); solution-cell rows expected retained "
        f"(digits/newline in top-V).")

    # Map compact rows -> raw windows. With drop=0 this is identity; still use uv.
    # Identify solution-cell rows over full dataset and te.
    log("\n--- identifying solution-cell rows ---")
    sol_rows: list[int] = []
    sol_meta: dict[int, dict[str, Any]] = {}
    # Board inventory from full solutions
    train_sol_hashes: set[str] = set()
    test_sol_hashes: set[str] = set()
    puzzle_to_sol: dict[str, str] = {}
    tr_set = set(int(x) for x in tr.detach().cpu().tolist())
    te_set = set(int(x) for x in te.detach().cpu().tolist())

    ids_cpu = ids.detach().cpu()
    y_cpu = y.detach().cpu()
    cls_cpu = cls.detach().cpu()
    uv_cpu = uv.detach().cpu() if torch.is_tensor(uv) else uv

    for i in range(n_kept):
        seq = raw_window_from_compact(ids_cpu[i], uv_cpu)
        rec = reconstruct_from_window(seq, sol_pat, puz_pat, value_map)
        if rec is not None and rec["full_solution"] and rec["solution"] is not None:
            sh = board_hash(rec["solution"])
            if rec["puzzle"] is not None:
                ph = board_hash(rec["puzzle"])
                puzzle_to_sol[ph] = sh
            if i in tr_set:
                train_sol_hashes.add(sh)
            if i in te_set:
                test_sol_hashes.add(sh)

        rc = target_solution_rc(seq, sol_pat, L=ids_cpu.shape[1])
        if rc is None:
            continue
        r, c = rc
        if rec is None or rec["puzzle"] is None:
            continue
        # gold digit from corpus label
        gold_tid = int(uv_cpu[int(cls_cpu[int(y_cpu[i])].item())].item())
        true_v = value_map.get(gold_tid, ...)
        if true_v is ... or true_v is None:
            continue
        true_v = int(true_v)
        board = board_state_before_cell(rec["puzzle"], rec["solution"], r, c)
        flags = classify_forced(board, r, c, true_v)
        ph = board_hash(rec["puzzle"])
        sh = puzzle_to_sol.get(ph)
        sol_rows.append(i)
        sol_meta[i] = {
            "r": r, "c": c, "true_v": true_v, "flags": flags,
            "puzzle_hash": ph, "solution_hash": sh,
        }

    log(f"solution-cell rows (classifiable): {len(sol_rows)}")
    te_sol = [i for i in sol_rows if i in te_set]
    tr_sol = [i for i in sol_rows if i in tr_set]
    log(f"  of which tr={len(tr_sol)} te={len(te_sol)}")
    log(f"full-solution boards: train={len(train_sol_hashes)} test={len(test_sol_hashes)} "
        f"puzzle->sol map={len(puzzle_to_sol)}")

    if not te_sol:
        raise RuntimeError("no classifiable solution-cell rows in te")

    # --- Cover predictions on te ---
    log("\n--- core_cover_sw on te (return_pred) ---")
    te_idx = te
    out_te, pred_te = original_cover(
        model, rules, ids, yv, cls, te_idx, return_pred=True
    )
    log(f"te core_sw: agree={out_te['agree']:.4f} cover={out_te['cover']:.4f} "
        f"agree_fired={out_te['agree_fired']:.4f}")

    # Map te position -> pred
    te_list = [int(x) for x in te_idx.detach().cpu().tolist()]
    te_pos = {row: p for p, row in enumerate(te_list)}
    pred_cpu = pred_te.detach().cpu()
    yv_cpu = yv.detach().cpu()

    # Solution-cell residual on te
    te_flags: list[dict[str, bool]] = []
    te_error: list[bool] = []
    te_abstain: list[bool] = []
    te_nonoverlap: list[bool] = []
    n_err = 0
    n_abs = 0
    for row in te_sol:
        p = te_pos[row]
        pr = int(pred_cpu[p].item())
        gold = int(yv_cpu[row].item())
        is_abs = pr < 0
        is_err = pr != gold  # includes abstain (MinedGates.mine residual pattern)
        meta = sol_meta[row]
        te_flags.append(meta["flags"])
        te_error.append(is_err)
        te_abstain.append(is_abs)
        sh = meta["solution_hash"]
        non_ol = sh is None or sh not in train_sol_hashes
        te_nonoverlap.append(non_ol)
        if is_err:
            n_err += 1
        if is_abs:
            n_abs += 1

    n_scored = len(te_sol)
    residual_rate = fraction(n_err, n_scored)
    abstain_rate = fraction(n_abs, n_scored)
    log(f"\nSTEP 0(a): solution-cell residual error rate = {residual_rate:.6f} "
        f"({n_err}/{n_scored}); abstain_rate={abstain_rate:.6f} ({n_abs}/{n_scored})")
    abort_msg = abort_if_residual_too_small(residual_rate)
    if abort_msg:
        log(abort_msg)
        report = {
            "aborted": True,
            "abort_message": abort_msg,
            "residual_rate": residual_rate,
            "n_scored": n_scored,
            "honesty_note": HONESTY_NOTE,
            "wyly_env": env_used,
            "round_trip": rt,
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))
        log(f"wrote {OUTPUT}")
        return 2
    log(f"STEP 0(a): residual >= {ABORT_RESIDUAL} -- continue")

    # Board overlap
    if test_sol_hashes:
        overlap_n = len(test_sol_hashes & train_sol_hashes)
        overlap_frac = fraction(overlap_n, len(test_sol_hashes))
    else:
        overlap_n = 0
        overlap_frac = 0.0
    log(f"STEP 0(b): board-level train/test overlap = {overlap_frac:.6f} "
        f"({overlap_n}/{len(test_sol_hashes)} test full-solution boards also in train)")

    # Metrics: full te solution-cell set
    metrics_all = compute_forced_metrics(te_flags, te_error, n_scored)
    gate = metrics_all["gate_union_recoverable_frac"]
    verdict = verdict_from_gate(gate)

    # Non-overlapping subset
    no_flags = [f for f, no in zip(te_flags, te_nonoverlap, strict=True) if no]
    no_err = [e for e, no in zip(te_error, te_nonoverlap, strict=True) if no]
    n_scored_no = len(no_flags)
    metrics_no = compute_forced_metrics(no_flags, no_err, n_scored_no) if n_scored_no else None
    gate_no = metrics_no["gate_union_recoverable_frac"] if metrics_no else None
    verdict_no = verdict_from_gate(gate_no) if gate_no is not None else None

    # --- Scoreboard ---
    log("\n" + "=" * 72)
    log("SCOREBOARD")
    log("=" * 72)
    log(f"Round-trip: PASSED on {rt['n_checked']} windows "
        f"(mismatch_cells={rt['n_mismatch_cells']})")
    log(f"Step-0 residual rate: {residual_rate:.6f} (abort threshold {ABORT_RESIDUAL})")
    log(f"Step-0 board overlap: {overlap_frac:.6f}")
    log(f"te solution-cell rows scored: {n_scored}")
    log(f"  errors (pred!=gold, incl abstain): {n_err}")
    log(f"  abstains: {n_abs}")
    log()
    log("BASE RATES (te solution-cell rows):")
    b = metrics_all["base"]
    log(f"  naked={b['naked']:.4f}  hidden={b['hidden']:.4f}  "
        f"union={b['union']:.4f}  non_forced={b['non_forced']:.4f}  n={b['n']}")
    log("RESIDUAL ERROR ROW FRACTIONS:")
    r_ = metrics_all["residual_error"]
    log(f"  naked={r_['naked']:.4f}  hidden={r_['hidden']:.4f}  "
        f"union={r_['union']:.4f}  non_forced={r_['non_forced']:.4f}  n={r_['n']}")
    log(f"  enrichment (residual - base): naked={metrics_all['enrichment']['naked']:+.4f} "
        f"hidden={metrics_all['enrichment']['hidden']:+.4f} "
        f"union={metrics_all['enrichment']['union']:+.4f}")
    log()
    log(f"GATE (absolute union-recoverable frac of residual ERROR rows): {gate:.6f}")
    log(f"  union-recoverable error rows: {metrics_all['n_union_recoverable_error']}")
    log(f"  recoverable-as-core_sw-points: {metrics_all['recoverable_as_core_sw_points']:.6f}")
    log(f"VERDICT: {verdict}")
    log()
    log("NAKED/HIDDEN/UNION breakdown (register-scoping guide):")
    log(f"  residual errors: n_naked={r_['n_naked']} n_hidden={r_['n_hidden']} "
        f"n_union={r_['n_union']} n_non_forced={r_['n_non_forced']}")
    log()
    if metrics_no is not None:
        log(f"NON-OVERLAPPING-BOARD SUBSET (n_scored={n_scored_no}):")
        bn = metrics_no["base"]
        rn = metrics_no["residual_error"]
        log(f"  base:    naked={bn['naked']:.4f} hidden={bn['hidden']:.4f} "
            f"union={bn['union']:.4f}")
        log(f"  residual: naked={rn['naked']:.4f} hidden={rn['hidden']:.4f} "
            f"union={rn['union']:.4f} n_err={metrics_no['n_error']}")
        log(f"  GATE: {gate_no:.6f}  core_sw-points: "
            f"{metrics_no['recoverable_as_core_sw_points']:.6f}")
        log(f"  VERDICT (non-overlap): {verdict_no}")
    else:
        log("NON-OVERLAPPING-BOARD SUBSET: empty (no rows)")
    log()
    log(f"HONESTY: {HONESTY_NOTE}")
    log(f"wall time total: {time.time() - t0:.1f}s")

    report = {
        "aborted": False,
        "honesty_note": HONESTY_NOTE,
        "wyly_env": env_used,
        "module_bound": {
            "TAG": v5.TAG, "DS": v5.DS, "LIB": v5.LIB, "JUDGE": v5.JUDGE,
            "LABELS": v5.LABELS, "STATE": str(v5.STATE.name),
        },
        "round_trip": rt,
        "load_ds": {
            "n_raw": n_raw, "n_kept": n_kept, "n_dropped": n_raw - n_kept,
            "n_tr": len(tr), "n_te": len(te),
        },
        "step0": {
            "residual_rate": residual_rate,
            "n_error": n_err,
            "n_abstain": n_abs,
            "n_scored": n_scored,
            "board_overlap_frac": overlap_frac,
            "n_train_sol_boards": len(train_sol_hashes),
            "n_test_sol_boards": len(test_sol_hashes),
            "n_overlap_boards": overlap_n,
        },
        "te_core_sw": out_te,
        "metrics": metrics_all,
        "verdict": verdict,
        "metrics_non_overlapping": metrics_no,
        "verdict_non_overlapping": verdict_no,
        "classification_defs": {
            "NAKED": "|candidates|==1, candidates = digits 1-9 not used by any filled peer",
            "HIDDEN": (
                "true value v is a candidate AND v has exactly one legal cell in at least "
                "one of the cell's three units among still-empty cells"
            ),
            "UNION": "naked OR hidden",
            "NON_FORCED": "neither",
        },
        "wall_time_s": time.time() - t0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    log(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
