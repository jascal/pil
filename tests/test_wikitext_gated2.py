import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_wikitext_gated2 as gated2  # noqa: E402
from experiments import campaign_wikitext_gated_rescue as rescue  # noqa: E402


def test_two_sided_claim_rule_all_quadrants_and_abstain():
    b0_pred = torch.tensor([10, 11, 12, 13, 14])
    b0_conf = torch.tensor([0.2, 0.2, 0.8, 0.8, 0.2])
    b1_pred = torch.tensor([20, 21, 22, 23, -1])
    b1_conf = torch.tensor([0.9, 0.4, 0.9, 0.4, 0.9])

    pred, conf, claim = gated2.two_sided_merge(
        b0_pred, b0_conf, b1_pred, b1_conf, tau_low=0.5, tau_high=0.7
    )

    assert claim.tolist() == [True, False, False, False, False]
    assert pred.tolist() == [20, 11, 12, 13, 14]
    assert conf.tolist() == pytest.approx([0.9, 0.2, 0.8, 0.8, 0.2])


def _row(agree, regressions, coverage, tau_low=0.1, tau_high=0.2):
    return {
        "tau_low": tau_low,
        "tau_high": tau_high,
        "val_agree": agree,
        "val_regressions": regressions,
        "gated_coverage": coverage,
    }


def test_val_selection_clear_winner_by_agree():
    frontier = [_row(0.36, 2, 0.3), _row(0.37, 5, 0.1, 0.2, 0.3)]
    assert gated2.select_operating_point(frontier, 0.35, 5) is frontier[1]


def test_val_selection_tie_broken_by_fewer_regressions():
    frontier = [_row(0.37, 4, 0.8), _row(0.37, 2, 0.1, 0.2, 0.3)]
    assert gated2.select_operating_point(frontier, 0.35, 5) is frontier[1]


def test_val_selection_double_tie_broken_by_larger_coverage():
    frontier = [_row(0.37, 2, 0.2), _row(0.37, 2, 0.7, 0.2, 0.3)]
    assert gated2.select_operating_point(frontier, 0.35, 5) is frontier[1]


def test_no_admissible_path_prints_exact_outcome_and_never_touches_test(capsys):
    frontier = [_row(0.354, 0, 0.5), _row(0.36, 3, 0.4)]
    selected = gated2.select_operating_point(frontier, 0.35, 2)
    touched = False

    def touch_test(_point):
        nonlocal touched
        touched = True
        raise AssertionError("test_te must not be touched")

    result = gated2.resolve_operating_point(selected, touch_test)
    assert capsys.readouterr().out.strip() == "no admissible operating point"
    assert selected is None
    assert not touched
    assert result == {
        "outcome": "no admissible operating point",
        "gated2_test": None,
    }


def test_flat_gate_band_is_inclusive_at_boundaries():
    assert gated2.flat_gate_passes(0.3447)
    assert gated2.flat_gate_passes(0.3547)
    assert not gated2.flat_gate_passes(0.344699)
    assert not gated2.flat_gate_passes(0.354701)


def test_defensive_b0_gate_is_imported_unchanged():
    assert gated2.campaign.b0_within_registered_band is rescue.b0_within_registered_band
    assert gated2.campaign.b0_within_registered_band(0.341)
    assert gated2.campaign.b0_within_registered_band(0.351)


def test_tau_low_grid_registered_literal_values():
    assert gated2.EXPECTED_TAU_LOW_GRID == [
        0.08503663539886475,
        0.15015951693058016,
        0.2123691976070404,
        0.30124656558036805,
        0.44849786162376404,
        0.5126050710678101,
        0.5625,
        0.7023643851280212,
        0.8238235354423527,
        0.9989785552024841,
    ]


def test_tau_high_grid_uses_shared_tau_deciles_deterministically(monkeypatch):
    calls = []
    original = rescue.tau_deciles

    def traced(conf):
        calls.append(conf.clone())
        return original(conf)

    monkeypatch.setattr(gated2.campaign, "tau_deciles", traced)
    conf = torch.tensor([0.9, 0.1, 0.7, 0.3, 0.5])
    first = gated2.campaign.tau_deciles(conf)
    second = gated2.campaign.tau_deciles(conf.clone())
    assert first == second == original(conf)
    assert len(calls) == 2
    assert torch.equal(calls[0], calls[1])
