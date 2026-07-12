"""Pure/fixture tests for slice #91's Soufflé legality certificate."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_legality_certificate as X  # noqa: E402  # isort: skip
from experiments import campaign_legality_pinning as pin  # noqa: E402  # isort: skip


VALID_GRID = [
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


@pytest.mark.skipif(
    shutil.which("souffle") is None,
    reason="host execution with souffle on PATH is required for the live certificate test",
)
def test_datalog_mirror_agrees_on_naked_and_abstain_rows():
    naked = [row[:] for row in VALID_GRID]
    naked[0][0] = 0
    abstain = [[0] * 9 for _ in range(9)]
    hidden_only = [[0] * 9 for _ in range(9)]
    for col, digit in enumerate(range(3, 10), start=2):
        hidden_only[0][col] = digit
    hidden_only[5][1] = 1
    grids = torch.tensor([naked, abstain, hidden_only], dtype=torch.long)
    rows = torch.tensor([0, 4, 0], dtype=torch.long)
    cols = torch.tensor([0, 4, 0], dtype=torch.long)
    true_v = torch.tensor([5, 5, 1], dtype=torch.long)

    tensor = X.legality_feature_with_scope(grids, rows, cols, true_v)
    datalog = X.souffle_legality(grids, rows, cols)
    expected = [
        (
            int(tensor["n_candidates"][i]),
            bool(tensor["is_naked"][i]),
            int(tensor["forced_value"][i]),
        )
        for i in range(3)
    ]
    # The shared #90 tensor reports hidden-only forced_value=1 for row 2; the frozen
    # certificate projection and Soufflé correctly represent that naked-only row as -1.
    assert int(tensor["forced_value"][2]) == 1
    expected[2] = (expected[2][0], expected[2][1], -1)
    assert datalog == {0: expected[0], 1: expected[1], 2: expected[2]}
    assert expected == [(1, True, 5), (9, False, -1), (2, False, -1)]

    agreement = X.three_way_agreement(grids, rows, cols, true_v, dataset_rows=[10, 11, 12])
    assert agreement["agreement_count"] == 3
    assert agreement["mismatches"] == 0
    assert agreement["percent"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [(0, None), *[(digit, digit) for digit in range(1, 10)]],
)
def test_val_digit_normalization(raw, normalized):
    assert X.normalize_grid_value(raw) == normalized


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (0, (0, 0, 0)),
        (8, (0, 8, 2)),
        (9, (1, 0, 0)),
        (26, (2, 8, 2)),
        (40, (4, 4, 4)),
        (80, (8, 8, 8)),
    ],
)
def test_cell_index_recovery(position, expected):
    assert X.recover_cell_index(position) == expected


def test_validity_scan_accepts_real_solution():
    result = X.scan_solution_grids([VALID_GRID])
    assert result == {
        "scanned": 1,
        "invalid": 0,
        "invalid_grids": [],
        "clean": True,
    }


def test_validity_scan_reports_invalid_unit():
    invalid = [row[:] for row in VALID_GRID]
    invalid[0][0] = invalid[0][1]
    result = X.scan_solution_grids([invalid])
    assert result["scanned"] == 1
    assert result["invalid"] == 1
    assert result["clean"] is False
    reasons = result["invalid_grids"][0]["reasons"]
    assert "row 0: not a permutation of 1..9" in reasons
    assert any(reason.startswith("column 0:") for reason in reasons)
    assert any(reason.startswith("box 0:") for reason in reasons)


@pytest.mark.parametrize("cert_ok", [False, True])
@pytest.mark.parametrize("validity_ok", [False, True])
@pytest.mark.parametrize("regressions_ok", [False, True])
def test_confidence_promotion_truth_table(cert_ok, validity_ok, regressions_ok):
    measured = 0.875
    promoted, final_conf, reason = X.confidence_promotion(
        1.0 if cert_ok else 0.99,
        validity_ok,
        0 if regressions_ok else 2,
        measured,
    )
    expected = cert_ok and validity_ok and regressions_ok
    assert promoted is expected
    assert final_conf == (1.0 if expected else measured)
    if not cert_ok:
        assert "certificate agreement" in reason
    if not validity_ok:
        assert "validity scan" in reason
    if not regressions_ok:
        assert "regressions" in reason


def test_pluggable_scope_default_is_identical_to_number_90_tensor():
    grid_a = [row[:] for row in VALID_GRID]
    grid_a[0][0] = 0
    grid_b = [[0] * 9 for _ in range(9)]
    grids = torch.tensor([grid_a, grid_b], dtype=torch.long)
    rows = torch.tensor([0, 4], dtype=torch.long)
    cols = torch.tensor([0, 4], dtype=torch.long)
    true_v = torch.tensor([5, 5], dtype=torch.long)
    direct = pin.legality_feature_tensor(grids, rows, cols, true_v)
    wrapped = X.legality_feature_with_scope(
        grids, rows, cols, true_v, X.row_col_box_scope,
    )
    assert direct.keys() == wrapped.keys()
    for key in direct:
        assert torch.equal(direct[key], wrapped[key]), key


def test_regression_computation_identifies_correct_to_wrong_flips():
    result = X.compute_regressions(
        [1, 2, -1, 4, 5],
        [1, -1, 3, 9, 5],
        [1, 2, 3, 4, 0],
        row_indices=[10, 11, 12, 13, 14],
    )
    assert result == {"count": 2, "row_indices": [11, 13]}


def test_registered_scope_and_caveat_are_exact():
    assert X.REVEAL_INDEX_NOTE == (
        "the sudoku dataset presents ONLY late-reveal solution cells (index ~59-80); "
        "early/mid cells never appear as prediction targets, so the register is measured "
        "only on the near-solved sub-task and its capability on early/mid cells is "
        "UNTESTED by this data."
    )
    assert X.FRAMING_NOTE == (
        "sudoku-only, LABELS=corpus, no natural-text blending; scope hand-authored; cert covers "
        "the concrete row∪col∪box instance, not the abstract pluggable hook."
    )
