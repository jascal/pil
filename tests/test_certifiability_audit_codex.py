"""Fast unit tests for the pre-registered certifiability audit."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import certifiability_audit_codex as audit  # noqa: E402


def test_perfect_monotone_spearman_and_exhaustive_permutation_p() -> None:
    measure = [1, 2, 3, 4, 5, 6, 7]
    recovery = [1, 2, 3, 4, 5, 6, 7]

    assert audit.spearman_rho(measure, recovery) == pytest.approx(1.0)
    # Identity and reversal are the only two of 7! permutations with |rho|=1.
    assert audit.permutation_p(measure, recovery) == pytest.approx(2 / math.factorial(7))


def test_composite_zscores_and_sign_alignment_match_hand_calculation() -> None:
    rows = {
        "low": {"direct": 1.0, "inverse": 30.0},
        "mid": {"direct": 2.0, "inverse": 20.0},
        "high": {"direct": 3.0, "inverse": 10.0},
    }
    scores, aligned = audit.compute_composite(
        rows, signs={"direct": 1, "inverse": -1}
    )
    edge = math.sqrt(3 / 2)

    assert aligned["low"] == pytest.approx({"direct": -edge, "inverse": -edge})
    assert aligned["mid"] == pytest.approx({"direct": 0.0, "inverse": 0.0})
    assert aligned["high"] == pytest.approx({"direct": edge, "inverse": edge})
    assert scores == pytest.approx({"low": -edge, "mid": 0.0, "high": edge})


def test_missing_measure_is_filtered_for_rho_but_available_z_still_contributes() -> None:
    rows = {
        "a": {"m1": 1.0, "m2": 3.0},
        "b": {"m1": 2.0, "m2": None},
        "c": {"m1": 3.0, "m2": 1.0},
    }
    scores, aligned = audit.compute_composite(rows, signs={"m1": 1, "m2": -1})
    edge = math.sqrt(3 / 2)

    assert aligned["b"]["m2"] is None
    assert scores["b"] == pytest.approx(0.0)  # m1 only, not (m1 + fabricated zero) / 2
    assert scores["a"] == pytest.approx((-edge - 1.0) / 2.0)
    assert scores["c"] == pytest.approx((edge + 1.0) / 2.0)

    predictor = {domain: None for domain in audit.DOMAINS}
    predictor.update({"wikitext": 1.0, "wt103": 2.0, "code": 3.0})
    statistic = audit._statistic(predictor, bootstrap_seed=11)
    assert statistic["n_used"] == 3
    assert statistic["domains_used"] == ["wikitext", "wt103", "code"]
    assert statistic["rho"] == pytest.approx(1.0)


def test_ground_truth_parser_uses_inlined_notes_fixture() -> None:
    fixture = """
    ## Held-out result
    The served split reports certified-rule agreement: **75.1%** over 708 rows.
    This is the recovery figure used by the audit.
    """

    parsed = audit.parse_cited_figure(fixture, "certified-rule agreement")

    assert parsed == {
        "value": pytest.approx(0.751),
        "exact_figure": "75.1%",
        "unit": "%",
    }
