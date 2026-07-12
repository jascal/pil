import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_frontier_rows as campaign  # noqa: E402
from experiments import campaign_wikitext_gated_rescue as rescue  # noqa: E402


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ],
)
def test_extract_row_sets_always_returns_cpu_indices(device):
    base = torch.tensor([True, False, False, True], device=device)
    candidate = torch.tensor([True, True, False, False], device=device)

    rows = campaign.extract_row_sets(base, candidate)

    assert {name: indices.tolist() for name, indices in rows.items()} == {
        "gains": [1],
        "regressions": [3],
        "untouched": [0, 2],
    }
    assert all(indices.device.type == "cpu" for indices in rows.values())


def test_row_set_crosscheck_relates_distinct_baseline_pairs():
    # GATED fixes rows 1,2 relative to B0 and breaks row 4. Relative to FLAT, only row 2 is c;
    # row 1 was already fixed by FLAT, while row 3 is c but was already B0-correct.
    base = torch.tensor([True, False, False, True, True, False])
    flat = torch.tensor([True, True, False, False, True, False])
    gated = torch.tensor([True, True, True, True, False, False])
    row = campaign.discordance_crosscheck(base, flat, gated)

    assert row["b0_to_gated_gains"] == 2
    assert row["b0_to_gated_regressions"] == 1
    assert row["flat_to_gated_b"] == 1
    assert row["flat_to_gated_c"] == 2
    assert row["gains_also_c"] + row["gains_not_c"] == row["b0_to_gated_gains"]
    assert row["gains_also_c"] + row["c_not_b0_gain"] == row["flat_to_gated_c"]
    assert row["regressions_also_b"] + row["regressions_not_b"] == row["b0_to_gated_regressions"]
    assert row["regressions_also_b"] + row["b_not_b0_regression"] == row["flat_to_gated_b"]


def test_stage3_pool_reports_pre_and_post_dedup_counts():
    pooled = campaign.pool_point_rows([
        {"gains": [1, 2, 3], "regressions": [7, 8]},
        {"gains": [2, 3, 4], "regressions": [8, 9]},
        {"gains": [4, 5], "regressions": [7, 10]},
    ])
    assert pooled == {
        "pre_dedup_gains": 8,
        "post_dedup_gains": 5,
        "pre_dedup_regressions": 6,
        "post_dedup_regressions": 4,
        "gains": [1, 2, 3, 4, 5],
        "regressions": [7, 8, 9, 10],
    }


@pytest.mark.parametrize(
    ("gains", "regressions", "expected"),
    [(20, 20, True), (19, 20, False), (20, 19, False), (21, 40, True)],
)
def test_universal_floor_is_inclusive(gains, regressions, expected):
    assert campaign.voting_eligible(gains, regressions) is expected


def test_auc_permutation_and_bootstrap_for_clean_separation():
    result = campaign.auc_statistics(
        list(range(8, 16)), list(range(8)), permutations=10_000, bootstraps=1_000, seed=7
    )
    assert result["auc"] == 1.0
    assert result["permutation_p_two_sided"] < 0.001
    assert result["bootstrap_95_ci"] == [1.0, 1.0]


def test_auc_permutation_and_bootstrap_for_identical_groups():
    values = [0, 1, 2, 3, 4, 5, 6, 7]
    result = campaign.auc_statistics(
        values, values, permutations=2_000, bootstraps=1_000, seed=11
    )
    assert result["auc"] == 0.5
    assert result["permutation_p_two_sided"] > 0.5
    low, high = result["bootstrap_95_ci"]
    assert low < 0.5 < high


def test_runner_up_trace_has_exact_original_prediction_and_confidence_parity(monkeypatch):
    ids = torch.tensor([[0, 1], [1, 0], [2, 1], [0, 2]])
    yv = torch.tensor([2, 1, 0, 1])
    cls = torch.tensor([0, 1, 2])
    idxs = torch.arange(4)
    model = SimpleNamespace(
        counts=torch.tensor([[0, 2, 0], [0, 0, 3], [2, 0, 0]]),
        counts_calib=None,
        rules2=[],
    )

    def high_rule(w):
        return torch.where(w[:, -1] == 1, torch.tensor(2), torch.tensor(-1))

    def low_rule(w):
        return torch.where(w[:, -1] != 2, torch.tensor(1), torch.tensor(-1))

    rules = [("low synthetic", low_rule), ("high synthetic", high_rule)]
    monkeypatch.setattr(rescue.v5, "STRATA", False)
    monkeypatch.setitem(
        rescue.v5.CONF_FNS,
        "low synthetic",
        lambda w: (low_rule(w), torch.full((len(w),), 0.6)),
    )
    monkeypatch.setitem(
        rescue.v5.CONF_FNS,
        "high synthetic",
        lambda w: (high_rule(w), torch.full((len(w),), 0.8)),
    )

    trace = campaign.assert_runner_trace_parity(model, rules, ids, yv, cls, idxs)
    _, _, prior = rescue.core_cover_sw_traced(model, rules, ids, yv, cls, idxs)
    assert torch.equal(trace["conf"], prior["conf"])
    assert trace["runner_conf"].tolist() == pytest.approx([0.6, 0.5, 0.6, -1e9])
    assert trace["gap"].tolist()[:3] == pytest.approx([0.2, 0.1, 0.2])


def test_stage3_guard_never_passes_test_te_to_scorer():
    val_te = torch.tensor([1, 3, 5])
    test_te = torch.tensor([2, 4, 6])
    seen = []

    def score(point, idxs):
        seen.append(idxs.data_ptr())
        assert idxs.data_ptr() == val_te.data_ptr()
        assert idxs.data_ptr() != test_te.data_ptr()
        return {
            "base_correct": torch.tensor([False, True, True]),
            "candidate_correct": torch.tensor([True, False, True]),
        }

    rows, pooled = campaign.guarded_stage3_rows(score, ["a", "b"], val_te, test_te)
    assert seen == [val_te.data_ptr(), val_te.data_ptr()]
    assert all(row["gains"] == [0] and row["regressions"] == [1] for row in rows)
    assert pooled["pre_dedup_gains"] == 2
    assert pooled["post_dedup_gains"] == 1


def _stat(auc, p=0.5, powered=True):
    return {
        "auc": auc,
        "permutation_p_two_sided": p,
        "powered_for_auc_0.65": powered,
    }


def _verdict_fixture(axis_rows, boundary_auc=0.7, boundary_p=0.01):
    rows = dict(axis_rows)
    rows["boundary_gap_untouched_higher_than_changed"] = _stat(
        boundary_auc, boundary_p, True
    )
    return rows


def test_verdict_confirmed_excludes_underpowered_axes():
    rows = _verdict_fixture({
        "confidence": _stat(0.52, powered=True),
        "gap": _stat(0.54, powered=True),
        "teacher_consensus": _stat(0.6, p=0.001, powered=False),
        "decile_index": _stat(0.5, powered=False),
    })
    result = campaign.resolve_verdict(rows)
    assert result["verdict"] == "CONFIRMED"
    assert result["underpowered_axes_excluded_from_confirmed"] == [
        "decile_index", "teacher_consensus"
    ]
    assert result["refuted_guarded_axes_literal"] == []
    assert result["refuted_guarded_axes_with_power_gate"] == []


def test_verdict_overlap_is_mixed_when_underpowered_axis_refutes_literally():
    rows = _verdict_fixture({
        "confidence": _stat(0.52, powered=True),
        "gap": _stat(0.54, powered=True),
        "teacher_consensus": _stat(0.9, p=0.001, powered=False),
        "decile_index": _stat(0.5, powered=False),
    })
    result = campaign.resolve_verdict(rows)
    assert result["verdict"] == "MIXED"
    assert result["refuted_guarded_axes_literal"] == ["teacher_consensus"]
    assert result["refuted_guarded_axes_with_power_gate"] == []


def test_verdict_refuted_names_guarded_axis_with_and_without_power_gate():
    rows = _verdict_fixture({
        "confidence": _stat(0.7, p=0.001, powered=True),
        "gap": _stat(0.5),
        "teacher_consensus": _stat(0.66, p=0.01, powered=False),
        "decile_index": _stat(0.5),
    }, boundary_auc=0.5, boundary_p=0.5)
    result = campaign.resolve_verdict(rows)
    assert result["verdict"] == "REFUTED"
    assert result["refuted_guarded_axes_literal"] == ["confidence", "teacher_consensus"]
    assert result["refuted_guarded_axes_with_power_gate"] == ["confidence"]


@pytest.mark.parametrize(
    "rows",
    [
        _verdict_fixture({name: _stat(0.6) for name in campaign.AXES}),
        _verdict_fixture({name: _stat(0.5) for name in campaign.AXES}, 0.64, 0.01),
        _verdict_fixture({name: _stat(0.5, powered=False) for name in campaign.AXES}),
    ],
)
def test_verdict_mixed_branches(rows):
    assert campaign.resolve_verdict(rows)["verdict"] == "MIXED"
