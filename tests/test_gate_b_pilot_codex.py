"""Fast, dataset-free tests for the independent Gate (b) implementation."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.gate_b_pilot_codex import (  # noqa: E402
    DetRank,
    PathState,
    compare_det_rank,
    decide_energy,
    path_quality_key,
    propagate_pass,
    propagation_depth,
    rank_candidates,
)

SOLVED = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def count_table(preferred: int = 1) -> dict:
    counts = [[0] * 10 for _ in range(9)]
    for c in range(9):
        for value in range(1, 10):
            counts[c][value] = 10 - value
        counts[c][preferred] = 20
    return {"counts": counts, "totals": [sum(row[1:]) for row in counts]}


def test_depth_two_board_is_forced_by_two_step_beam() -> None:
    board = [
        [5, 3, 0, 6, 0, 0, 9, 1, 0],
        [6, 7, 0, 0, 9, 5, 3, 0, 8],
        [1, 9, 8, 3, 0, 0, 5, 6, 7],
        [8, 5, 9, 7, 0, 1, 0, 0, 3],
        [4, 2, 0, 0, 5, 0, 7, 0, 0],
        [7, 1, 3, 0, 2, 4, 8, 5, 6],
        [9, 6, 1, 0, 0, 0, 0, 0, 4],
        [2, 0, 7, 4, 0, 9, 0, 3, 5],
        [3, 0, 5, 0, 8, 0, 0, 7, 9],
    ]
    cell0 = (0, 4)
    assert propagation_depth(board, cell0) == 2
    assert len(propagate_pass(board)[1][cell0]) == 2

    result = decide_energy(board, cell0, M=2, table=count_table(preferred=3))
    assert result["commit"] == SOLVED[cell0[0]][cell0[1]]
    assert result["source"] == "forced"


def test_unforced_fixpoint_does_not_claim_forced_source() -> None:
    board = [[0] * 9 for _ in range(9)]
    cell0 = (4, 4)
    assert propagation_depth(board, cell0) is None

    result = decide_energy(board, cell0, M=2, table=count_table(preferred=5))
    assert result["source"] != "forced"


def test_propagate_pass_exact_row_column_box_candidates() -> None:
    board = [[0] * 9 for _ in range(9)]
    board[4][0:4] = [1, 2, 3, 4]
    board[0][4] = 5
    board[1][4] = 6
    board[3][3] = 7
    board[5][5] = 8

    _, candidates, contradiction = propagate_pass(board)
    assert not contradiction
    assert candidates[(4, 4)] == frozenset({9})


def test_det_rank_cross_multiply_and_exact_tie_break() -> None:
    assert compare_det_rank(7, 10, 5, 8) > 0
    assert DetRank(7, 10) > DetRank(5, 8)

    counts = [[0] * 10 for _ in range(9)]
    denominators = [[1] * 10 for _ in range(9)]
    counts[2][4], denominators[2][4] = 1, 2
    counts[2][7], denominators[2][7] = 2, 4
    table = {"counts": counts, "totals": [1] * 9, "denominators": denominators}
    assert rank_candidates({4, 7}, 2, table, "branch", 0) == [7, 4]

    source = Path(__file__).parents[1] / "experiments" / "gate_b_pilot_codex.py"
    tree = ast.parse(source.read_text())
    comparator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compare_det_rank"
    )
    assert not any(isinstance(node, ast.Div) for node in ast.walk(comparator))


def test_path_quality_key_is_length_robust() -> None:
    board = [[0] * 9 for _ in range(9)]
    all_forced = PathState(board, margins=(1, 1))
    one_guess_longer = PathState(board, margins=(1, 0, 1, 1, 1))
    assert path_quality_key(all_forced) > path_quality_key(one_guess_longer)

    module_doc = ast.get_docstring(
        ast.parse((Path(__file__).parents[1] / "experiments" / "gate_b_pilot_codex.py").read_text())
    )
    assert module_doc is not None
    assert "margins=(1, 1)" in module_doc
    assert "margins=(1, 0, 1, 1, 1)" in module_doc
