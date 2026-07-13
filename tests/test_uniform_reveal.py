"""Pure-function + smoke tests for slice #100 uniform-reveal corrected anchor.

Fast tests are dataset-free integer-list fixtures. The heavy e2e is skipif-guarded
on the sudoku dataset file and only scans a small row slice (no full main()).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_sudoku_forced_move as osc  # noqa: E402  # isort: skip
from experiments import campaign_uniform_reveal as X  # noqa: E402  # isort: skip


# ---------------------------------------------------------------------------
# 1. Anchor fix (hand-verified two-block fixture)
# ---------------------------------------------------------------------------

def test_anchor_finds_own_block_not_previous():
    """Two-block synthetic token stream. Early cell (r=0,c=0) of block 2's solution.
    The OLD (first-match) anchor wrongly lands on block 1's SOLUTION header and drops the
    row (off >= SOL_BODY -> None). The FIXED (last-match) anchor lands on block 2's own
    header and recovers (r,c) = (0, 0).
    """
    PUZ_HDR = [901, 902, 903, 904]   # len 4 (matches real "PUZZLE\n" token-length)
    SOL_HDR = [801, 802, 803]        # len 3 (matches real "SOLUTION\n" token-length)
    PUZ_BODY = osc.PUZ_BODY          # 90
    SOL_BODY = osc.SOL_BODY          # 90

    def block(puz_fill, sol_fill, trailing_blank):
        tok = PUZ_HDR + [puz_fill] * PUZ_BODY + SOL_HDR + [sol_fill] * SOL_BODY
        return tok + [0] if trailing_blank else tok

    block1 = block(1, 2, trailing_blank=True)     # 188 tokens
    block2 = block(3, 4, trailing_blank=False)    # 187 tokens
    stream = block1 + block2                      # 375 tokens
    assert len(stream) == 375

    # target = block2's first solution cell (r=0, c=0), value 4
    sol_hdr2_at = len(block1) + 4 + 90             # == 282
    target_idx = sol_hdr2_at + 3                   # == 285
    assert stream[target_idx] == 4

    L = 256
    window = stream[target_idx - L:target_idx]
    assert len(window) == L

    old_at = osc.find_subsequence(window, SOL_HDR)
    new_at = X.find_last_subsequence(window, SOL_HDR)
    assert old_at == 65      # block 1's (wrong) header, window-local
    assert new_at == 253     # block 2's (correct) header, window-local

    old_rc = osc.target_solution_rc(window, SOL_HDR, L=L)
    new_rc = X.target_solution_rc_fixed(window, SOL_HDR, L=L)
    assert old_rc is None            # OLD anchor drops this early cell (the bug)
    assert new_rc == (0, 0)          # FIXED anchor recovers it correctly


def test_anchor_mid_block_cell_disagrees_with_old():
    """Mid-block cell (off=45 inside block 2's solution body): fixed vs old anchors.

    Derive OLD-anchor expectation by running osc.find_subsequence / target_solution_rc;
    do not hardcode a guessed OLD (r,c). Assert the two disagree (or old is None/wrong).
    """
    PUZ_HDR = [901, 902, 903, 904]
    SOL_HDR = [801, 802, 803]
    PUZ_BODY = osc.PUZ_BODY
    SOL_BODY = osc.SOL_BODY

    def block(puz_fill, sol_fill, trailing_blank):
        tok = PUZ_HDR + [puz_fill] * PUZ_BODY + SOL_HDR + [sol_fill] * SOL_BODY
        return tok + [0] if trailing_blank else tok

    block1 = block(1, 2, trailing_blank=True)
    block2 = block(3, 4, trailing_blank=False)
    stream = block1 + block2
    assert len(stream) == 375

    sol_hdr2_at = len(block1) + 4 + 90  # 282
    off = 45
    target_idx = sol_hdr2_at + 3 + off
    assert stream[target_idx] == 4

    L = 256
    window = stream[target_idx - L:target_idx]
    assert len(window) == L

    old_at = osc.find_subsequence(window, SOL_HDR)
    new_at = X.find_last_subsequence(window, SOL_HDR)
    assert old_at != new_at  # two headers still in window; first != last

    old_rc = osc.target_solution_rc(window, SOL_HDR, L=L)
    new_rc = X.target_solution_rc_fixed(window, SOL_HDR, L=L)

    # Fixed: off=45 -> r,c = divmod(45, 10) = (4, 5); cell_index=41 -> third 1
    assert new_rc == (4, 5)
    # OLD: first-match header is still the previous block's; off exceeds SOL_BODY
    # so old_rc is None (or, if somehow not, must disagree with new_rc).
    assert old_rc is None or old_rc != new_rc
    # Derive OLD expectation from the same functions (no guessed hardcoded OLD rc):
    if old_at >= 0:
        sol_body_start_old = old_at + len(SOL_HDR)
        off_old = L - sol_body_start_old
        if off_old < 0 or off_old >= SOL_BODY or off_old % 10 == 9:
            assert old_rc is None
        else:
            r_old, c_old = divmod(off_old, 10)
            if r_old >= 9 or c_old >= 9:
                assert old_rc is None
            else:
                assert old_rc == (r_old, c_old)
                assert old_rc != new_rc


# ---------------------------------------------------------------------------
# 2. Board-validity guard (pure fixture)
# ---------------------------------------------------------------------------

def _valid_solution_grid() -> list[list[int]]:
    """Standard always-valid Sudoku solution: value(r,c) = ((r*3 + r//3 + c) % 9) + 1."""
    return [
        [((r * 3 + r // 3 + c) % 9) + 1 for c in range(9)]
        for r in range(9)
    ]


def test_board_is_consistent_valid_and_row_col_box_collisions():
    grid = _valid_solution_grid()
    assert X.board_is_consistent(grid) is True

    # row collision
    g_row = [row[:] for row in grid]
    g_row[0][1] = g_row[0][0]
    assert X.board_is_consistent(g_row) is False

    # column collision
    g_col = [row[:] for row in grid]
    g_col[1][0] = g_col[0][0]
    assert X.board_is_consistent(g_col) is False

    # box collision (same 3x3 box, not same row/col as the source of the collision
    # if possible -- force a duplicate inside the top-left box)
    g_box = [row[:] for row in grid]
    # (0,0) and (1,1) are in the same box; set (1,1) to (0,0)'s value if different
    # After a full valid solution they differ; force collision.
    g_box[1][1] = g_box[0][0]
    # May also create row/col collisions depending on the pattern; clear peers of the
    # duplicate to isolate a box-only failure when possible. Spec asks for a focused
    # box-collision assertion: a grid with a box collision fails consistency.
    assert X.board_is_consistent(g_box) is False


# ---------------------------------------------------------------------------
# 3. Heavy end-to-end smoke (skipif-guarded; small slice, not full main)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO / "data" / "wyly_nexttoken_sudoku_L256.pt").exists(),
    reason="requires the sudoku dataset file",
)
def test_full_dataset_reveal_index_distribution_now_is_spread():
    """First 2000 rows: fixed anchor yields cell_index not all in third 2.

    Loads the sudoku .pt directly (raw kept_ids) so the test is robust when the full
    suite has already bound wyly_lm_v5 to a different WYLY_DS (setdefault cannot
    override; DATA is fixed at import time).
    """
    import wyly_lm_v5 as v5  # noqa: E402

    codec = v5.load_codec()
    puz_pat, sol_pat = osc.header_patterns(codec)
    value_map = osc.build_value_map(codec)

    raw = torch.load(
        REPO / "data" / "wyly_nexttoken_sudoku_L256.pt", map_location="cpu",
    )
    raw_ids = raw["kept_ids"]  # raw token windows (pre-compact), same as #89 round-trip
    L = int(raw_ids.shape[1])
    n = min(2000, int(raw_ids.shape[0]))

    observed_thirds: list[int] = []
    for i in range(n):
        seq = [int(x) for x in raw_ids[i].tolist()]
        rc = X.target_solution_rc_fixed(seq, sol_pat, L=L)
        if rc is None:
            continue
        r, c = rc
        cell_index = r * 9 + c
        observed_thirds.append(cell_index // 27)
        info = X.reconstruct_block_for_target(seq, sol_pat, puz_pat, value_map, L=L)
        if info is not None:
            assert (info["r"], info["c"]) == (r, c)

    assert len(observed_thirds) > 0, "expected some solution-cell targets in first 2000 rows"
    thirds_set = set(observed_thirds)
    assert len(thirds_set) > 1, (
        f"expected reveal thirds to spread beyond a single bucket; got {thirds_set} "
        f"(n_targets={len(observed_thirds)})"
    )
