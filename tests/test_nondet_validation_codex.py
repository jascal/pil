"""Small, corpus-free tests for the independent non-deterministic validation pilot."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

import campaign_sudoku_forced_move as osc  # noqa: E402
from beam_engine import beam_decode  # noqa: E402
from sudoku_oracle import SudokuOracle  # noqa: E402

from experiments import nondet_validation_codex as pilot  # noqa: E402

_SOLUTION = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 5, 6, 7, 8, 9, 1, 2, 3],
    [7, 8, 9, 1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7, 8, 9, 1],
    [5, 6, 7, 8, 9, 1, 2, 3, 4],
    [8, 9, 1, 2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8, 9, 1, 2],
    [6, 7, 8, 9, 1, 2, 3, 4, 5],
    [9, 1, 2, 3, 4, 5, 6, 7, 8],
]

# A unique 36-clue puzzle. At its propagation fixpoint the canonical first pivot is
# (0, 1), candidates [2, 4], while gold is 4: the first sorted guess is genuinely wrong.
# The wrong branch is refuted during the beam search; target (0, 4) has known r=2.
_R2_BOARD = [
    [0, 0, 0, 7, 0, 0, 8, 3, 0],
    [5, 9, 3, 1, 6, 0, 4, 2, 0],
    [0, 0, 8, 0, 0, 3, 6, 0, 0],
    [3, 7, 4, 0, 8, 0, 0, 0, 2],
    [9, 0, 0, 0, 0, 0, 0, 7, 6],
    [0, 0, 1, 0, 7, 9, 0, 4, 8],
    [0, 0, 0, 0, 0, 6, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 7, 0, 4],
    [0, 1, 0, 8, 0, 7, 9, 6, 3],
]
_R2_SOLUTION = [
    [1, 4, 6, 7, 5, 2, 8, 3, 9],
    [5, 9, 3, 1, 6, 8, 4, 2, 7],
    [7, 2, 8, 4, 9, 3, 6, 5, 1],
    [3, 7, 4, 6, 8, 1, 5, 9, 2],
    [9, 8, 2, 5, 3, 4, 1, 7, 6],
    [6, 5, 1, 2, 7, 9, 3, 4, 8],
    [4, 3, 7, 9, 1, 6, 2, 8, 5],
    [8, 6, 9, 3, 2, 5, 7, 1, 4],
    [2, 1, 5, 8, 4, 7, 9, 6, 3],
]
_TARGET = (0, 4)


def _uniform_count_table() -> list[list[list[int]]]:
    return [[[0] + [1] * 9 for _ in range(9)] for _ in range(9)]


def _run_r2(m: int) -> dict:
    r, c = _TARGET
    oracle = SudokuOracle(_R2_BOARD, r, c, _uniform_count_table(), pilot.WYLY_SEED)
    return beam_decode(oracle, m, pilot.BEAM_WIDTH, pilot.RULE_ID, pilot.WYLY_SEED)


def test_r2_beam_commits_gold_after_real_refutation() -> None:
    """The wrong first canonical guess is pruned and sufficient beam search reaches gold."""
    fixed, contradiction = pilot.propagation_fixpoint(_R2_BOARD)
    assert contradiction is False
    empties = [(r, c) for r in range(9) for c in range(9) if fixed[r][c] == 0]
    pivot = min(empties, key=lambda rc: (len(osc.cell_candidates(fixed, *rc)), rc))
    assert pivot == (0, 1)
    assert sorted(osc.cell_candidates(fixed, *pivot)) == [2, 4]
    assert _R2_SOLUTION[0][1] == 4  # sorted candidate 2 is a wrong first guess

    m1 = _run_r2(1)
    assert m1["committed_value"] == 9  # deterministic fallback is documented and wrong here
    assert m1["prune_events"] == 0

    deep = _run_r2(11)
    assert deep["committed_value"] == _R2_SOLUTION[0][4] == 5
    assert deep["prune_events"] > 0
    assert pilot.prune_bite([deep]) > 0


def test_prune_bite_is_from_a_real_sudoku_contradiction() -> None:
    """No mocks: SudokuOracle expansion refutes a live wrong-guess path by M=10."""
    result = _run_r2(10)
    assert result["anomaly_total_contradiction"] is False
    assert result["prune_events"] == 1
    assert pilot.actual_attempted_steps(result) == 10
    assert pilot.prune_bite([result]) == 0.1


def test_branch_cell_detector_distinguishes_stall_from_forced() -> None:
    r, c = _TARGET
    assert pilot.gbp.propagation_depth(_R2_BOARD, r, c) is None
    assert (r, c) in pilot.branch_cells(_R2_BOARD)

    forced_board = [row[:] for row in _SOLUTION]
    forced_board[0][0] = 0
    assert pilot.gbp.propagation_depth(forced_board, 0, 0) == 1
    assert (0, 0) not in pilot.branch_cells(forced_board)


def test_canonical_refutation_depth_is_exactly_two() -> None:
    assert pilot.count_solutions(_R2_BOARD) == 1
    result = pilot.canonical_solve_with_r(_R2_BOARD)
    assert result is not None
    solved, rgrid = result
    assert solved == _R2_SOLUTION
    r, c = _TARGET
    assert rgrid[r][c] == 2
