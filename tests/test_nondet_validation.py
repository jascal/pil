"""Fast unit tests for nondet validation helpers (hand-built boards only).

No full corpus load, no full generation pipeline. Mirrors tests/test_beam_engine.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

import campaign_gate_b_pilot as gbp  # noqa: E402
from beam_engine import beam_decode  # noqa: E402
from sudoku_oracle import SudokuOracle  # noqa: E402

from experiments import campaign_nondet_validation as X  # noqa: E402

SEED = 0
RULE_ID = X.RULE_ID
BEAM_WIDTH = X.BEAM_WIDTH


def _uniform_count_table(
    favor: dict[tuple[int, int, int], int] | None = None,
) -> list[list[list[int]]]:
    """All cells/digits count=1, then optional favor overrides (higher = more det)."""
    table = [[[0] + [1] * 9 for _ in range(9)] for _ in range(9)]
    if favor:
        for (r, c, v), cnt in favor.items():
            table[r][c][v] = cnt
    return table


def _empty_board() -> list[list[int]]:
    return [[0] * 9 for _ in range(9)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Pre-propagated unique-ish stall board (true local fixpoint).
# First min-candidate pivot is (0, 7) with candidates {1, 3}; gold = 1.
# Wrong guess 3 survives M=1 ranking (det_rank-favored) but is pruned after
# branch + two prop steps (beam M=3): M=1 commits wrong, M>=3 commits gold
# with prune_events > 0. Refutation-depth of the pivot itself is r=1 (filled
# by the first nested guess); deeper cells on the gold path have r>=2.
_BRANCH_PRUNE_BOARD = [
    [0, 0, 4, 6, 7, 8, 9, 0, 2],
    [6, 7, 2, 1, 9, 0, 0, 0, 8],
    [0, 9, 8, 3, 0, 2, 0, 0, 0],
    [8, 0, 9, 7, 6, 0, 4, 2, 3],
    [0, 2, 6, 8, 0, 0, 7, 9, 1],
    [0, 0, 0, 9, 2, 0, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 0, 4, 1, 9, 0, 0, 0],
    [0, 4, 5, 2, 8, 6, 1, 0, 9],
]
_BRANCH_PIVOT = (0, 7)
_BRANCH_GOLD = 1
_BRANCH_WRONG = 3

# r=1-only board: one correct pivot guess + prop fully solves all remaining empties.
# Empties are a coupled 2-candidate structure; pivot (3,1) gold=5? Wait — solution grid
# below; after prop-to-fixpoint this board is already at fixpoint with 6 empties,
# all filled at depth 1 by the first successful branch.
_R1_BOARD = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 0, 9, 7, 6, 1, 4, 0, 3],
    [4, 0, 6, 8, 0, 3, 7, 9, 1],
    [7, 1, 3, 9, 0, 4, 8, 0, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]
# Canonical min-cand pivot is (3,1) with cands {2,5}. Sorted trial order tries 2
# first; that path prop-solves the remaining five empties, so r=1 for all six.


# ---------------------------------------------------------------------------
# (a) M=1 may commit wrong; M large enough to refute commits gold
# ---------------------------------------------------------------------------


def test_wrong_guess_pruned_with_lookahead():
    """Wrong det_rank-favored guess at branch pivot; lookahead+prune recovers gold.

    Reasoning: board is already a true local fixpoint (propagate_one_pass returns []).
    Pivot (0,7) has candidates {1, 3}; gold is 1. Count table heavily favors 3 so
    M=1 ranks the wrong branch first (no prop yet after the guess step). After two
    further expansion steps the wrong path hits a contradiction (prune_events > 0)
    and the surviving gold path commits 1. (Beam step budget: branch + 2 prop ≈ M=3.)
    """
    board = [row[:] for row in _BRANCH_PRUNE_BOARD]
    r0, c0 = _BRANCH_PIVOT
    gold = _BRANCH_GOLD
    wrong = _BRANCH_WRONG

    # Sanity: true fixpoint + branch cell
    forced, contra = gbp.propagate_one_pass(board)
    assert contra is False
    assert forced == []
    assert gbp.propagation_depth(board, r0, c0) is None

    table = _uniform_count_table(
        favor={(r0, c0, wrong): 1000, (r0, c0, gold): 1}
    )

    res1 = beam_decode(
        SudokuOracle(board, r0, c0, table, SEED), 1, BEAM_WIDTH, RULE_ID, SEED
    )
    # M=1 is not guaranteed correct (favored wrong commit)
    assert res1["committed_value"] == wrong
    assert res1["committed_value"] != gold

    res3 = beam_decode(
        SudokuOracle(board, r0, c0, table, SEED), 3, BEAM_WIDTH, RULE_ID, SEED
    )
    assert res3["committed_value"] == gold
    assert res3["prune_events"] > 0

    # M >= 3 continues to commit gold
    for m in (4, 5):
        res = beam_decode(
            SudokuOracle(board, r0, c0, table, SEED), m, BEAM_WIDTH, RULE_ID, SEED
        )
        assert res["committed_value"] == gold


# ---------------------------------------------------------------------------
# (b) Genuine prune (contradiction eliminates a path)
# ---------------------------------------------------------------------------


def test_beam_prune_events_positive():
    """beam_decode reports prune_events > 0 when a wrong branch contradicts.

    Same board as (a): at M=3 the wrong (0,7)=3 child is eliminated by a
    propagation contradiction (not merely beam-width truncation — beam_width=8
    is larger than the number of live children here).
    """
    board = [row[:] for row in _BRANCH_PRUNE_BOARD]
    r0, c0 = _BRANCH_PIVOT
    table = _uniform_count_table()
    res = beam_decode(
        SudokuOracle(board, r0, c0, table, SEED), 3, BEAM_WIDTH, RULE_ID, SEED
    )
    assert res["prune_events"] > 0
    assert res["anomaly_total_contradiction"] is False
    assert res["committed_value"] == _BRANCH_GOLD


# ---------------------------------------------------------------------------
# (c) Branch-cell detector via propagation_depth
# ---------------------------------------------------------------------------


def test_branch_cell_detector():
    """propagation_depth is None at a true stall cell; not None when forced.

    Branch board: naked-pair-style stall on row 0 — cells (0,0) and (0,1) both have
    candidates {1, 2}, no naked/hidden single anywhere on that partial board, so
    propagate_one_pass returns [] and propagation_depth((0,0)) is None.

    Forced board: complete valid Latin row-block with only (0,0) empty — that cell
    is a naked single (only 1 fits), so propagation_depth is 1 (not None).
    """
    # --- true local fixpoint / BRANCH cell ---
    stall = _empty_board()
    stall[0] = [0, 0, 3, 4, 5, 6, 7, 8, 9]
    # Only (0,0) and (0,1) empty in row 0; both candidates {1,2} — naked pair stall.
    forced, contra = gbp.propagate_one_pass(stall)
    assert contra is False
    assert forced == []
    assert gbp.propagation_depth(stall, 0, 0) is None
    assert gbp.propagation_depth(stall, 0, 1) is None
    # branch_cells helper agrees
    bcells = X.branch_cells(stall)
    assert (0, 0) in bcells and (0, 1) in bcells

    # --- propagation-forced cell (NOT a branch cell) ---
    forced_board = _empty_board()
    # Full standard first band except (0,0):
    forced_board[0] = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    # (0,0) is naked single = 1
    assert gbp.propagation_depth(forced_board, 0, 0) is not None
    assert gbp.propagation_depth(forced_board, 0, 0) == 1
    assert (0, 0) not in X.branch_cells(forced_board)


# ---------------------------------------------------------------------------
# (d) Refutation-depth solve_with_depths on a known tiny case
# ---------------------------------------------------------------------------


def test_refutation_depth_r1_one_guess_solves():
    """One correct pivot guess + prop fills every remaining empty ⇒ those cells have r=1.

    Board is a near-complete valid grid with six empty cells forming a coupled
    structure that pure single-cell propagation cannot break (propagation_depth
    is None for each). The canonical min-candidate pivot is (3, 1) with candidates
    {2, 5}. Sorted trial order tries 2 first; that path prop-solves the remaining
    five empties. Every branch cell is therefore filled at depth 1 (r=1).
    """
    board = [row[:] for row in _R1_BOARD]
    # All empties are branch cells
    empties = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
    assert len(empties) == 6
    for r, c in empties:
        assert gbp.propagation_depth(board, r, c) is None

    result = X.solve_with_depths(board, 0)
    assert result is not None
    solved, depths = result
    assert all(solved[r][c] != 0 for r in range(9) for c in range(9))

    for r, c in empties:
        assert depths[(r, c)] == 1, f"expected r=1 at {(r, c)}, got {depths.get((r, c))}"

    # Pivot itself is recorded at depth 1; first successful trial value is 2
    assert depths[(3, 1)] == 1
    assert solved[3][1] == 2


def test_refutation_depth_pivot_on_branch_prune_board():
    """On the branch-prune fixture, the first pivot has r=1 (filled by first guess)."""
    board = [row[:] for row in _BRANCH_PRUNE_BOARD]
    r0, c0 = _BRANCH_PIVOT
    result = X.solve_with_depths(board, 0)
    assert result is not None
    _solved, depths = result
    assert depths[(r0, c0)] == 1
    # At least one cell deeper than the first guess exists on this harder board
    assert max(depths.values()) >= 1
