import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_wikitext_gated2 as gated2  # noqa: E402
from experiments import campaign_wikitext_gated_ra as gated_ra  # noqa: E402


@pytest.mark.parametrize(
    ("gains", "regressions", "expected"),
    [
        (0, 0, False),  # A zero-gain/zero-regression key is never banned.
        (1, 0, False),
        (2, 1, False),
        (1, 1, True),
        (0, 1, True),
        (2, 3, True),
    ],
)
def test_registered_ban_criterion_truth_table(gains, regressions, expected):
    assert gated_ra.ban_criterion(gains, regressions) is expected


def test_pre_retreat_wrapper_abstains_exactly_on_banned_keys_only():
    def key_fn(w):
        return w[:, 0]

    def fn(w):
        return torch.where(w[:, 1] < 0, torch.full_like(w[:, 1], -1), w[:, 1] + 10)

    def conf_fn(w):
        pred = fn(w)
        conf = torch.where(
            pred >= 0,
            torch.arange(len(w), dtype=torch.float32, device=w.device) / 10 + 0.5,
            torch.full((len(w),), gated_ra.ABSTAIN_CONF, device=w.device),
        )
        return pred, conf

    raw = ("synthetic", fn, conf_fn, key_fn)
    wrapped = gated_ra.retreat_candidate(raw, {2, 7})
    w = torch.tensor([[2, 1], [3, 2], [7, 3], [4, -1], [8, 4]])

    raw_pred, raw_conf = conf_fn(w)
    pred = wrapped[1](w)
    conf_pred, conf = wrapped[2](w)
    assert pred.tolist() == [-1, 12, -1, -1, 14]
    assert torch.equal(conf_pred, pred)
    assert conf.tolist() == pytest.approx([-1e9, raw_conf[1], -1e9, -1e9, raw_conf[4]])
    keep = torch.tensor([False, True, False, True, True])
    assert torch.equal(pred[keep], raw_pred[keep])
    assert torch.equal(conf[keep], raw_conf[keep])


def test_per_tau_ban_recomputation_is_deterministic():
    counts = {
        ("pointer", 3): {"val_gains": 1, "val_regressions": 1},
        ("pointer", 8): {"val_gains": 0, "val_regressions": 0},
        ("mined frames", 42): {"val_gains": 3, "val_regressions": 2},
        ("dstate gate", 9): {"val_gains": 0, "val_regressions": 2},
    }
    first = gated_ra.compute_ban_set(counts)
    second = gated_ra.compute_ban_set(dict(reversed(list(counts.items()))))
    assert first == second == {"dstate gate": {9}, "pointer": {3}}


def test_part_a_val_to_test_coverage_hand_fixture():
    val_counts = {
        ("pointer", 10): {"val_gains": 1, "val_regressions": 2},
        ("pointer", 11): {"val_gains": 2, "val_regressions": 1},
        ("mined frames", 99): {"val_gains": 0, "val_regressions": 1},
        ("dstate gate", 4): {"val_gains": 0, "val_regressions": 0},
    }
    bans = gated_ra.compute_ban_set(val_counts)
    test_regressions = [
        ("pointer", 10),
        ("pointer", 11),
        ("mined frames", 99),
        ("attrib-subj gate", 5),
    ]
    assert bans == {"mined frames": {99}, "pointer": {10}}
    assert gated_ra.val_to_test_coverage(bans, test_regressions) == 0.5


def test_below_50_percent_stop_path_never_enters_part_b_or_touches_test(capsys):
    touched = False

    def part_b_that_touches_test():
        nonlocal touched
        touched = True
        raise AssertionError("Part B/test_te must not be touched")

    result = gated_ra.gate_part_b(0.499, part_b_that_touches_test)
    assert result is None
    assert not touched
    assert capsys.readouterr().out.strip().splitlines() == [
        gated_ra.PARTIAL_PREDICTION,
        gated_ra.NEGATIVE_FINDING,
    ]


def test_selection_objective_and_structural_guard_are_literal_reuses():
    assert gated_ra.select_operating_point is gated2.select_operating_point
    assert gated_ra.resolve_operating_point is gated2.resolve_operating_point
