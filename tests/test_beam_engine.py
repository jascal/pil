"""Engine+SudokuOracle fast unit tests vs pilot decide_energy (hand-built boards).

Mirrors tests/test_gate_b_pilot.py fixture style. No full dataset/model load.
Each case asserts (a) engine+oracle path and (b) exact agreement with gbp.decide_energy.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

import campaign_gate_b_pilot as gbp  # noqa: E402
import campaign_sudoku_forced_move as osc  # noqa: E402
from beam_engine import beam_decode  # noqa: E402
from beam_engine import stable_hash_step as engine_stable_hash_step  # noqa: E402
from sudoku_oracle import SudokuOracle  # noqa: E402

SEED = 0
RULE_ID = gbp.RULE_ID
BEAM_WIDTH = gbp.BEAM_WIDTH
COMPARE_FIELDS = (
    "committed_value",
    "anomaly_total_contradiction",
    "prune_events",
    "surviving_counts",
)


def _empty_board() -> list[list[int]]:
    return [[0] * 9 for _ in range(9)]


def _uniform_count_table(
    favor: dict[tuple[int, int, int], int] | None = None,
) -> list[list[list[int]]]:
    """All cells/digits count=1, then apply optional favor overrides (higher = more det)."""
    table = [[[0] + [1] * 9 for _ in range(9)] for _ in range(9)]
    if favor:
        for (r, c, v), cnt in favor.items():
            table[r][c][v] = cnt
    return table


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


def _board_without(*cells: tuple[int, int]) -> list[list[int]]:
    board = [row[:] for row in _SOLUTION]
    for r, c in cells:
        board[r][c] = 0
    return board


def _engine_run(board, r0, c0, M, table, seed=SEED):
    oracle = SudokuOracle(board, r0, c0, table, seed)
    return beam_decode(oracle, M, BEAM_WIDTH, RULE_ID, seed)


def _pilot_run(board, r0, c0, M, table, seed=SEED):
    return gbp.decide_energy(board, r0, c0, M, BEAM_WIDTH, table, seed, RULE_ID)


def _assert_agree(board, r0, c0, M, table, seed=SEED):
    eng = _engine_run(board, r0, c0, M, table, seed)
    pil = _pilot_run(board, r0, c0, M, table, seed)
    for k in COMPARE_FIELDS:
        assert eng[k] == pil[k], (
            f"M={M} field {k!r}: engine={eng[k]!r} pilot={pil[k]!r}\n"
            f"  engine={eng!r}\n  pilot={pil!r}"
        )
    return eng, pil


# ---------------------------------------------------------------------------
# d=2 cascade fixture (same construction as test_gate_b_pilot)
# ---------------------------------------------------------------------------


def _fixture_d2_cascade():
    """Pure 2-pass deterministic cascade at (0,0) with gold=1.

    cell0 is NOT forceable at pass 1 but IS forced by pass 2.
    Count table favors wrong digit so M=1 det_rank fallback commits wrong;
    M>=2 forces gold via pure propagation.
    """
    board = _board_without(
        (0, 0), (0, 1), (0, 3), (2, 0), (2, 3), (3, 0), (3, 3), (3, 7)
    )
    r0, c0, gold = 0, 0, 1
    table = _uniform_count_table(favor={(0, 0, 2): 1000, (0, 0, 1): 1})
    return board, r0, c0, gold, table


def test_forced_2step_cascade_engine_matches_pilot():
    board, r0, c0, gold, table = _fixture_d2_cascade()

    # Intermediate mechanism (must hold before beam asserts)
    assert osc.cell_candidates(board, r0, c0) == {1, 2}
    assert gbp.propagation_depth(board, r0, c0) == 2

    eng1, pil1 = _assert_agree(board, r0, c0, 1, table)
    assert eng1["committed_value"] != gold
    assert eng1["committed_value"] == 2
    assert pil1["committed_value"] == 2

    eng2, _ = _assert_agree(board, r0, c0, 2, table)
    assert eng2["committed_value"] == gold

    for m in (3, 4, 5):
        eng, _ = _assert_agree(board, r0, c0, m, table)
        assert eng["committed_value"] == gold


def test_m1_corner_engine_matches_pilot():
    """M=1 corner: engine reproduces pilot decide_energy exactly (Slice-1 delegate)."""
    board, r0, c0, _gold, table = _fixture_d2_cascade()
    eng, pil = _assert_agree(board, r0, c0, 1, table)
    assert eng["committed_value"] == pil["committed_value"]
    assert type(eng["committed_value"]) is int


def test_branch_then_prune_engine_matches_pilot():
    """Near-terminal contradictory board: prune path exercised; engine == pilot."""
    board = _empty_board()
    board[0] = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    board[1][1] = 1
    # target elsewhere so we observe anomaly on a non-zero-cand cell too
    assert osc.cell_candidates(board, 0, 0) == set()
    table = _uniform_count_table()
    for m in (1, 2):
        eng, pil = _assert_agree(board, 0, 0, m, table)
        assert eng["anomaly_total_contradiction"] is True
        assert eng["committed_value"] is None
        assert eng["prune_events"] >= 1
        assert pil["anomaly_total_contradiction"] is True


def test_total_contradiction_anomaly_engine_matches_pilot():
    """Board where propagate_one_pass immediately contradicts → anomaly both sides."""
    board = _empty_board()
    board[0] = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    board[1][1] = 1
    forced, contradiction = gbp.propagate_one_pass(board)
    assert contradiction is True
    assert forced is None
    table = _uniform_count_table()
    eng, pil = _assert_agree(board, 0, 0, 1, table)
    assert eng["anomaly_total_contradiction"] is True
    assert eng["committed_value"] is None
    assert pil["anomaly_total_contradiction"] is True
    assert pil["committed_value"] is None


def test_true_fixpoint_branch_engine_matches_pilot():
    """Naked-pair stall: true fixpoint → branch; engine and pilot agree on commits."""
    board = _empty_board()
    board[0] = [0, 0, 3, 4, 5, 6, 7, 8, 9]
    assert osc.cell_candidates(board, 0, 0) == {1, 2}
    assert osc.cell_candidates(board, 0, 1) == {1, 2}
    forced, contra = gbp.propagate_one_pass(board)
    assert contra is False
    assert forced == []
    table = _uniform_count_table(favor={(0, 0, 1): 100, (0, 0, 2): 1})
    for m in (1, 2, 3):
        _assert_agree(board, 0, 0, m, table)


def test_cross_repo_hash_identity():
    """Generalized *key hash is byte-identical to pilot's hardcoded r,c,v version."""
    rule_id, r, c, v, wyly_seed = "decide_energy_v1", 2, 5, 9, 0
    eng = engine_stable_hash_step(rule_id, r, c, v, wyly_seed=wyly_seed)
    pil = gbp.stable_hash_step(rule_id, r, c, v, wyly_seed)
    assert eng == pil

    # a few more tuples
    for rr, cc, vv, seed in [(0, 0, 1, 0), (8, 8, 9, 7), (3, 4, 5, 99)]:
        assert engine_stable_hash_step(
            "det_rank_fallback", rr, cc, vv, wyly_seed=seed
        ) == gbp.stable_hash_step("det_rank_fallback", rr, cc, vv, seed)
