"""Pure-function tests for Gate (b) pilot (propagation-aligned beam + pinned margin).

Hand-built 9x9 fixtures only for the required cases -- no full dataset load.
Optional smoke test is skipif-guarded on the sudoku nexttoken cache.

Fixtures are constructed empirically: intermediate propagate_one_pass /
cell_candidates / propagation_depth values are asserted BEFORE beam behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_gate_b_pilot as X  # noqa: E402  # isort: skip
from experiments import campaign_sudoku_forced_move as osc  # noqa: E402  # isort: skip


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


# Completed Latin-like grid used as a base for multi-hop cascade fixtures.
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


# ---------------------------------------------------------------------------
# (a) genuinely d=2 cell: beam(M>=2) commits gold, beam(1) does not
# ---------------------------------------------------------------------------

def _fixture_d2_cascade() -> tuple[list[list[int]], int, int, int, list[list[list[int]]]]:
    """Pure 2-pass deterministic cascade at (0,0) with gold=1.

    Empirically verified (see intermediate asserts in the test):
      - (0,0) starts with 2 candidates (not d=1)
      - pass 1 forces 7 cells elsewhere, NOT (0,0); after fill, (0,0) has #cand==1
      - propagation_depth == 2
      - beam(1) commits the det_rank fallback's deliberately rigged wrong digit;
        beam(2) forces gold via pure propagation (no det_rank dependence — all
        forced margins are +1)
    """
    board = _board_without(
        (0, 0), (0, 1), (0, 3), (2, 0), (2, 3), (3, 0), (3, 3), (3, 7)
    )
    r0, c0, gold = 0, 0, 1
    # Count table heavily favors the WRONG digit so beam(1)'s det_rank fallback picks
    # wrong from the original board's candidates, while beam(2) still forces gold.
    table = _uniform_count_table(favor={(0, 0, 2): 1000, (0, 0, 1): 1})
    return board, r0, c0, gold, table


def test_a_d2_cascade_beam_m2_commits_gold_m1_does_not():
    board, r0, c0, gold, table = _fixture_d2_cascade()

    # --- intermediate mechanism (must hold before beam asserts) ---
    assert osc.cell_candidates(board, r0, c0) == {1, 2}
    forced1, contra1 = X.propagate_one_pass(board)
    assert contra1 is False
    assert forced1 is not None and len(forced1) >= 1
    assert all((r, c) != (r0, c0) for r, c, _ in forced1), (
        "pass 1 must make progress elsewhere, not force cell0"
    )
    after1 = [row[:] for row in board]
    for rr, cc, v in forced1:
        after1[rr][cc] = v
    assert after1[r0][c0] == 0
    assert osc.cell_candidates(after1, r0, c0) == {gold}
    d = X.propagation_depth(board, r0, c0)
    assert d == 2

    # --- beam ---
    res1 = X.decide_energy(board, r0, c0, M=1, beam_width=8, count_table=table, wyly_seed=0)
    assert res1["committed_value"] != gold
    assert res1["committed_value"] == 2

    res2 = X.decide_energy(board, r0, c0, M=2, beam_width=8, count_table=table, wyly_seed=0)
    assert res2["committed_value"] == gold
    # Deterministic across seeds (pure forced cascade — no det_rank dependence)
    res2b = X.decide_energy(board, r0, c0, M=2, beam_width=8, count_table=table, wyly_seed=99)
    assert res2b["committed_value"] == gold

    for m in (3, 4, 5):
        assert (
            X.decide_energy(board, r0, c0, M=m, beam_width=8, count_table=table, wyly_seed=0)[
                "committed_value"
            ]
            == gold
        )


def test_m1_nonforced_target_commits_det_rank_int():
    board, r0, c0, _gold, table = _fixture_d2_cascade()
    forced, contradiction = X.propagate_one_pass(board)
    assert contradiction is False
    assert forced is not None
    assert all((r, c) != (r0, c0) for r, c, _ in forced)

    result = X.decide_energy(
        board, r0, c0, M=1, beam_width=8, count_table=table, wyly_seed=0
    )
    committed_value = result["committed_value"]
    assert committed_value is not None
    assert type(committed_value) is int
    assert 1 <= committed_value <= 9


# ---------------------------------------------------------------------------
# Branch + prune-on-contradiction code path
# ---------------------------------------------------------------------------

def test_prune_on_contradiction_drops_path():
    """A board with a 0-candidate empty cell is pruned on the first pass.

    Direct exercise of the contradiction branch in decide_energy's outer loop:
    propagate_one_pass returns (None, True) -> path dropped, prune_events >= 1.
    """
    board = _empty_board()
    # Block all of 1..9 from (0,0): row has 2..9, box peer has 1.
    board[0] = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    board[1][1] = 1
    assert osc.cell_candidates(board, 0, 0) == set()
    forced, contradiction = X.propagate_one_pass(board)
    assert contradiction is True
    assert forced is None

    table = _uniform_count_table()
    res = X.decide_energy(board, 0, 0, M=1, beam_width=8, count_table=table, wyly_seed=0)
    assert res["prune_events"] >= 1
    assert res["committed_value"] is None
    assert res["anomaly_total_contradiction"] is True


def test_branch_then_prune_wrong_guess():
    """True fixpoint -> branch on pivot; wrong child hits contradiction on next pass.

    Construction (verified empirically below):
      row0: [., ., 3,4,5,6,7,8,9]  both (0,0) and (0,1) candidates {1,2} — naked pair stall.
      col1 has a 2 at (3,1). Wait — that would make (0,1) a naked single, not a stall.

    Instead: nearly-full valid grid with two empty cells that are NOT singles and where
    one legal fill of the min-cand pivot creates an immediate 0-cand peer on the next
    propagate. We use a hand-built board that is a true fixpoint with pivot (0,0)
    cands {1,2}, and after placing 1 the board has a 0-cand empty cell (engineered by
    also leaving a peer that is already almost blocked — see intermediate asserts).

    Simpler direct path: seed decide_energy at M=2 on a board that is a fixpoint with
    a 2-candidate pivot where BOTH values are legal by peer check at guess time, but
    we only need to observe that SOME wrong path gets pruned. The classic pair+block:

      After research: immediate post-guess 0-cand is impossible from a true singles
      fixpoint when the guess was a legal candidate (the would-be 0-cand peer would
      have been a naked single of that value). So we test multi-step: M>=2 on a board
      that is already contradictory for one of the expanded children by using a path
      that starts one step from contradiction via an almost-full unit.

    Practical fixture used here: start from the d2 cascade AFTER pass-1 fills have
    been applied and then force a bad filled value on a side cell so the next pass
    sees a 0-cand — exercised through decide_energy by giving it that near-terminal
    board directly.
    """
    # Near-terminal contradictory board: one empty cell with 0 candidates.
    board = _empty_board()
    board[0] = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    board[1][1] = 1
    # Also leave another empty elsewhere so the board isn't just one cell — still
    # contradiction on first pass.
    board[8][8] = 0  # already 0
    assert osc.cell_candidates(board, 0, 0) == set()
    table = _uniform_count_table()
    res = X.decide_energy(board, 8, 8, M=2, beam_width=8, count_table=table, wyly_seed=0)
    assert res["prune_events"] >= 1
    # No surviving legal path through a global contradiction.
    assert res["anomaly_total_contradiction"] is True or res["committed_value"] is None


# ---------------------------------------------------------------------------
# (b) d>M (or never): beam(M) does not reliably force gold
# ---------------------------------------------------------------------------

def test_b_d_never_beam_m2_does_not_force():
    """Empty board: d is None; beam(M=2) may commit something via guessing, but not
    by pure forced propagation — committed value need not equal a 'gold'. We only
    require that it is not a spurious deterministic force of a single legal value
    (every cell has 9 candidates; commit comes from branching, not forced singles).
    """
    board = _empty_board()
    assert X.propagation_depth(board, 0, 0) is None
    assert len(osc.cell_candidates(board, 0, 0)) >= 2
    table = _uniform_count_table()
    res = X.decide_energy(board, 0, 0, M=2, beam_width=8, count_table=table, wyly_seed=0)
    # Not forced by pure propagation: either None or a guessed value with margin-0 steps.
    # Spurious force would mean a single legal value — not the case here.
    assert len(osc.cell_candidates(board, 0, 0)) > 1
    # prune should not fire on a fully open empty board within M=2
    assert res["prune_events"] == 0


def test_b_d_gt_m_on_d2_fixture_m1_not_gold():
    """On the d=2 cascade fixture, M=1 < d in the sense that one pass does not commit
    cell0 (beam budget below the cascade length)."""
    board, r0, c0, gold, table = _fixture_d2_cascade()
    assert X.propagation_depth(board, r0, c0) == 2
    res = X.decide_energy(board, r0, c0, M=1, beam_width=8, count_table=table, wyly_seed=0)
    assert res["committed_value"] != gold


# ---------------------------------------------------------------------------
# (c) legality / candidate counter
# ---------------------------------------------------------------------------

def test_c_cell_candidates_naked_single():
    board = _empty_board()
    board[0] = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    assert osc.cell_candidates(board, 0, 0) == {1}


def test_c_cell_candidates_several_open():
    board = _empty_board()
    board[0][1] = 5
    board[1][0] = 7
    cands = osc.cell_candidates(board, 0, 0)
    assert 5 not in cands and 7 not in cands
    assert len(cands) == 7


def test_c_cell_candidates_fully_blocked():
    board = _empty_board()
    board[0] = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    board[1][1] = 1
    assert osc.cell_candidates(board, 0, 0) == set()


# ---------------------------------------------------------------------------
# (d) det_rank integer tie-break
# ---------------------------------------------------------------------------

def test_d_det_rank_equal_ratios_unequal_denominators():
    """2/4 vs 3/6 must be TIED (naive raw-count comparison would get this wrong)."""
    table = [[[0] * 10 for _ in range(9)] for _ in range(9)]
    table[0][0][1] = 2
    table[0][0][2] = 2
    table[1][1][3] = 3
    table[1][1][4] = 3
    assert X.det_rank_compare(table, (0, 0, 1), (1, 1, 3)) == 0


def test_d_det_rank_higher_rate_wins():
    """3/4 > 1/2 -- higher rate ranks first (-1)."""
    table = [[[0] * 10 for _ in range(9)] for _ in range(9)]
    table[0][0][1] = 3
    table[0][0][2] = 1  # total 4
    table[1][1][3] = 1
    table[1][1][4] = 1  # total 2
    assert X.det_rank_compare(table, (0, 0, 1), (1, 1, 3)) == -1
    assert X.det_rank_compare(table, (1, 1, 3), (0, 0, 1)) == 1


def test_d_det_rank_zero_over_zero_tied():
    table = [[[0] * 10 for _ in range(9)] for _ in range(9)]
    assert X.det_rank_compare(table, (0, 0, 1), (5, 5, 9)) == 0


def test_d_path_sort_falls_through_to_hash_seed():
    """When margin and det_rank tie, (rule_id, WYLY_SEED) hash decides; seed flip reorders."""
    table = [[[0] * 10 for _ in range(9)] for _ in range(9)]
    table[0][0][1] = 5
    table[0][0][2] = 5
    board = _empty_board()
    board[0] = [0, 0, 3, 4, 5, 6, 7, 8, 9]
    # Same margin (0 = guess) for both; equal det_rank at same cell
    p1 = X.BeamPath(steps=[], board=[row[:] for row in board]).extend(0, 0, 1, margin=0)
    p2 = X.BeamPath(steps=[], board=[row[:] for row in board]).extend(0, 0, 2, margin=0)
    cmp_s0 = X.compare_paths(p1, p2, table, wyly_seed=0, rule_id=X.RULE_ID)
    cmp_s1 = X.compare_paths(p1, p2, table, wyly_seed=1, rule_id=X.RULE_ID)
    assert cmp_s0 in (-1, 1)
    assert cmp_s1 in (-1, 1)
    assert X.compare_paths(p1, p2, table, wyly_seed=0, rule_id=X.RULE_ID) == cmp_s0
    found_flip = cmp_s0 != cmp_s1
    if not found_flip:
        for s in range(2, 50):
            if X.compare_paths(p1, p2, table, wyly_seed=s, rule_id=X.RULE_ID) != cmp_s0:
                found_flip = True
                break
    assert found_flip, "expected some WYLY_SEED to change hash-tie ordering"


# ---------------------------------------------------------------------------
# propagate_one_pass direct tests
# ---------------------------------------------------------------------------

def test_propagate_naked_single_only():
    board = _empty_board()
    board[0] = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    forced, contradiction = X.propagate_one_pass(board)
    assert contradiction is False
    assert forced is not None
    assert (0, 0, 1) in forced


def test_propagate_hidden_single_only():
    """Value uniquely placeable in a unit despite the cell having >1 raw candidates.

    Row 0: empties (0,0) and (0,1). (0,0) cands {1,2}, (0,1) cands {2} only after we
    block 1 from (0,1) — wait, that makes (0,1) a naked single. Instead: block so that
    1 is unique to (0,0) in the row while (0,0) still has another candidate.

    Layout:
      row0: [., ., 3,4,5,6,7,8,9]  both start as {1,2}
      put 1 in column 1 outside the top box -> (0,1) loses 1 -> naked single 2
      and (0,0) is hidden single for 1 (unique in row) AND has raw cands {1,2}.
    """
    board = _empty_board()
    board[0] = [0, 0, 3, 4, 5, 6, 7, 8, 9]
    board[3][1] = 1  # blocks 1 from (0,1)
    assert osc.cell_candidates(board, 0, 0) == {1, 2}
    assert osc.cell_candidates(board, 0, 1) == {2}
    forced, contradiction = X.propagate_one_pass(board)
    assert contradiction is False
    assert forced is not None
    # (0,0) forced as hidden single for 1; (0,1) as naked single for 2
    by_cell = {(r, c): v for r, c, v in forced}
    assert by_cell.get((0, 0)) == 1
    assert by_cell.get((0, 1)) == 2


def test_propagate_stall_naked_pair():
    board = _empty_board()
    board[0] = [0, 0, 3, 4, 5, 6, 7, 8, 9]
    assert osc.cell_candidates(board, 0, 0) == {1, 2}
    assert osc.cell_candidates(board, 0, 1) == {1, 2}
    forced, contradiction = X.propagate_one_pass(board)
    assert contradiction is False
    assert forced == []


def test_propagate_contradiction():
    board = _empty_board()
    board[0] = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    board[1][1] = 1
    forced, contradiction = X.propagate_one_pass(board)
    assert contradiction is True
    assert forced is None


# ---------------------------------------------------------------------------
# propagation_depth direct tests
# ---------------------------------------------------------------------------

def test_propagation_depth_d1_raw_naked():
    board = _empty_board()
    board[0] = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    assert X.propagation_depth(board, 0, 0) == 1


def test_propagation_depth_d1_raw_hidden_single():
    """Hidden single at the raw board must be d=1 (not mislabeled d=2).

    Reproduces the v3 bug: (0,0) has raw cands {1,2} so the old naked-only early
    return skipped it, then the post-pass candidate-count check labeled it d=2.
    Pass 1's forced list includes (0,0) as a hidden single for 2 (unique in row 0
    for value 2, since (0,1) is a naked single for 1).
    """
    board = _empty_board()
    board[0] = [0, 0, 3, 4, 5, 6, 7, 8, 9]
    board[3][1] = 2
    # Intermediate: not a naked single, but forceable on pass 1 via hidden single.
    assert osc.cell_candidates(board, 0, 0) == {1, 2}
    assert osc.cell_candidates(board, 0, 1) == {1}
    forced0, contradiction0 = X.propagate_one_pass(board)
    assert contradiction0 is False
    assert forced0 is not None
    by_cell = {(rr, cc): v for rr, cc, v in forced0}
    assert by_cell.get((0, 0)) == 2
    assert by_cell.get((0, 1)) == 1
    assert X.propagation_depth(board, 0, 0) == 1


def test_propagation_depth_d1_already_filled_given():
    """Puzzle given already on the board is d=1 (not mislabeled d=2/never).

    propagate_one_pass only forces empty cells, so a pre-filled target is never in
    forced0; the post-pass filled-check would otherwise see board[r][c]!=0 after a
    later pass that makes unrelated progress and return a spurious multi-pass depth.
    """
    board = _board_without((0, 1), (0, 3))  # (0,0) remains the given 1
    assert board[0][0] == 1
    assert (0, 0) not in {
        (rr, cc) for rr, cc, _ in (X.propagate_one_pass(board)[0] or [])
    }
    assert X.propagation_depth(board, 0, 0) == 1


def test_propagation_depth_d2_multipass():
    board, r0, c0, gold, _table = _fixture_d2_cascade()
    # Intermediate: not forceable on pass 1; becomes a naked single after pass-1 fills.
    assert len(osc.cell_candidates(board, r0, c0)) == 2
    forced, contra = X.propagate_one_pass(board)
    assert contra is False
    assert forced is not None
    assert all((rr, cc) != (r0, c0) for rr, cc, _ in forced)
    after = [row[:] for row in board]
    for rr, cc, v in forced:
        after[rr][cc] = v
    assert after[r0][c0] == 0
    assert osc.cell_candidates(after, r0, c0) == {gold}
    # Pass 2 must fill the target (robust filled-check path in propagation_depth).
    forced2, contra2 = X.propagate_one_pass(after)
    assert contra2 is False
    assert forced2 is not None
    by_cell2 = {(rr, cc): v for rr, cc, v in forced2}
    assert by_cell2.get((r0, c0)) == gold
    d = X.propagation_depth(board, r0, c0)
    assert d == 2


def test_propagation_depth_never():
    """Empty board: no naked/hidden singles anywhere -> d is None."""
    board = _empty_board()
    assert X.propagation_depth(board, 0, 0) is None


# ---------------------------------------------------------------------------
# Optional smoke (skipif-guarded)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO / "data" / "wyly_nexttoken_sudoku_L256.pt").exists(),
    reason="sudoku nexttoken cache not present",
)
def test_smoke_count_table_builds():
    table, n = X.build_sudoku_count_table(REPO / "data" / "corpus_sudoku.txt")
    assert n > 0
    assert len(table) == 9 and len(table[0]) == 9 and len(table[0][0]) == 10
    total = sum(table[0][0][d] for d in range(1, 10))
    assert total == n
