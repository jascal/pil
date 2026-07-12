from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_frames_at_scale as campaign  # noqa: E402
from experiments import campaign_wikitext_gated_rescue as rescue  # noqa: E402


def test_registered_grid_is_fixed_and_spans_quarter_to_current():
    expected_interactions = (0.0625, 0.109375, 0.15625, 0.203125, 0.25)
    expected = tuple(
        (support, interaction)
        for support in (4, 10, 15)
        for interaction in expected_interactions
    )
    assert campaign.registered_grid() == expected
    assert len(expected) == 15
    assert expected[0] == (4, 0.25 / 4)
    assert expected[-1] == (15, 0.25)


def test_selection_objective_and_tie_break_are_deterministic():
    rows = [
        {"support_gate": 15, "interaction_gate": 0.25, "marginal": 0.004, "val_regressions": 0},
        {"support_gate": 10, "interaction_gate": 0.25, "marginal": 0.006, "val_regressions": 1},
        {"support_gate": 10, "interaction_gate": 0.20, "marginal": 0.005, "val_regressions": 0},
        {"support_gate": 4, "interaction_gate": 0.20, "marginal": 0.005, "val_regressions": 0},
    ]
    assert campaign.select_grid_point(rows) is rows[3]
    assert campaign.select_grid_point(rows) is rows[3]

    no_admission = [
        {"support_gate": 4, "interaction_gate": 0.0625, "marginal": 0.00049, "val_regressions": 0}
    ]
    assert campaign.select_grid_point(no_admission) is None


def test_exact_binomial_helper_is_reused_by_identity():
    assert campaign.exact_discordant_p is rescue.exact_discordant_p


def test_no_admission_stop_never_touches_test():
    touched = False

    def touch_test(_point):
        nonlocal touched
        touched = True
        raise AssertionError("test_te was touched")

    rows = [
        {"support_gate": 4, "interaction_gate": 0.0625, "marginal": 0.00049, "val_regressions": 0}
    ]
    result = campaign.resolve_selection(rows, touch_test)
    assert result["outcome"] == "SCALE FACT"
    assert result["test"] is None
    assert not touched


def test_admitted_but_regressive_selection_gap_also_never_touches_test():
    rows = [
        {"support_gate": 4, "interaction_gate": 0.0625, "marginal": 0.001, "val_regressions": 1}
    ]
    result = campaign.resolve_selection(
        rows, lambda _point: pytest.fail("test_te was touched")
    )
    assert result["outcome"] == "NO ZERO-REGRESSION SELECTION"
    assert result["test"] is None


@pytest.mark.parametrize("scale", ["410m", "2.8b"])
def test_band_check_is_inclusive_at_exact_boundaries(scale):
    reference = campaign.SCALES[scale]["reference"]
    assert campaign.within_registered_band(scale, reference - 0.005)
    assert campaign.within_registered_band(scale, reference + 0.005)
    assert not campaign.within_registered_band(scale, reference - 0.005000001)
    assert not campaign.within_registered_band(scale, reference + 0.005000001)


def test_tunable_singletons_match_original_at_default_thresholds(monkeypatch):
    # Twenty errors share anchor 1 and the rest share anchor 0; both have perfectly recovering
    # tables, while the 60 sampled errors keep the original >=50 error-pool guard active.
    n = 60
    ids = torch.zeros((n, 3), dtype=torch.long)
    ids[:20, -2] = 1
    ids[:, -1] = torch.arange(n) % 2
    yv = torch.where(ids[:, -2] == 1, torch.tensor(2), torch.tensor(1))
    cls = torch.arange(3)
    fit = torch.arange(n)
    model = SimpleNamespace(rules=[])

    def always_wrong(_model, _rules, _ids, _yv, _cls, idxs, return_pred=False):
        pred = torch.full((len(idxs),), 0, dtype=torch.long)
        metrics = {"agree": 0.0, "cover": 1.0, "agree_fired": 0.0}
        return (metrics, pred) if return_pred else metrics

    monkeypatch.setattr(campaign.v5, "core_cover_sw", always_wrong)
    kwargs = {"offs": [2], "pair_offs": []}
    original = campaign.v5.MinedGates(3, "cpu", **kwargs)
    tunable = campaign.TunableMinedGates(
        3, "cpu", support_gate=15, interaction_gate=0.25, **kwargs
    )
    original.mine(
        model, ids, yv, cls, fit, torch.Generator().manual_seed(0), [], 0,
        sample_n=n,
    )
    tunable.mine(
        model, ids, yv, cls, fit, torch.Generator().manual_seed(0), [], 0,
        sample_n=n,
    )

    assert tunable.diagnostics["error_pool_size"] == n
    assert original.mined1 == {(2, 0), (2, 1)}
    assert tunable.mined1 == original.mined1
    assert tunable.mined2 == original.mined2
    assert tunable.n_frames == original.n_frames
    assert torch.equal(tunable.t1.k, original.t1.k)
    assert torch.equal(tunable.t1.v, original.t1.v)
    assert torch.equal(tunable.t1.c, original.t1.c)
