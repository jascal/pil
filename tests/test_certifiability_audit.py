"""Unit tests for Probe E certifiability-audit pilot (stdlib + numpy/scipy only)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

from campaign_certifiability_audit import (  # noqa: E402
    build_composite,
    cross_attribution,
    parse_core_sw_table_line,
    spearman_rho,
    spearman_with_perm_bootstrap,
    zscore,
)


def test_spearman_monotone_near_one_and_small_p() -> None:
    """(a) Hand-built monotone set: rho ~ 1.0, permutation p small."""
    measure = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ground_truth = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]  # monotone transform
    rho = spearman_rho(measure, ground_truth)
    assert rho == pytest.approx(1.0, abs=1e-9)

    stats = spearman_with_perm_bootstrap(
        measure, ground_truth, permutations=2000, bootstraps=500, seed=0
    )
    assert stats["rho"] == pytest.approx(1.0, abs=1e-9)
    assert stats["permutation_p_rho_ge_observed"] is not None
    # under random shuffles, rho=1 is extreme → small p_ge is NOT required
    # (almost all shuffles have rho < 1 so p_ge is small for "rho >= 1")
    assert stats["permutation_p_rho_ge_observed"] < 0.05
    assert stats["permutation_p_two_sided"] < 0.05
    assert stats["bootstrap_95_ci"] is not None
    lo, hi = stats["bootstrap_95_ci"]
    assert lo > 0.5  # CI should sit high for perfect monotone


def test_composite_zscore_and_sign_flip() -> None:
    """(b) lower-is-better measure is sign-flipped; z-scoring is per-measure."""
    # Domains A,B,C: register_density higher on A; effective_output_rank higher on C
    # (rank lower = more certifiable, so C should rank lower after flip)
    domain_vals = {
        "A": {"register_density": 0.9, "effective_output_rank": 1.0},
        "B": {"register_density": 0.5, "effective_output_rank": 2.0},
        "C": {"register_density": 0.1, "effective_output_rank": 3.0},
    }
    composites, backing, z_detail = build_composite(
        domain_vals,
        measure_keys=["register_density", "effective_output_rank"],
        lower_is_better={"effective_output_rank"},
    )
    # Both measures present for all → backing has both
    for d in ("A", "B", "C"):
        assert set(backing[d]) == {"register_density", "effective_output_rank"}

    # Per-measure z: mean 0, std 1 across the three domains
    for m in ("register_density", "effective_output_rank"):
        zs = [z_detail[d][m] for d in ("A", "B", "C")]
        assert abs(float(np.mean(zs))) < 1e-9
        assert abs(float(np.std(zs, ddof=0)) - 1.0) < 1e-9

    # register_density z: A high, C low
    assert z_detail["A"]["register_density"] > z_detail["C"]["register_density"]
    # effective_output_rank after negation: rank 1.0 → most certifiable → highest z
    assert z_detail["A"]["effective_output_rank"] > z_detail["C"]["effective_output_rank"]

    # Composite: A should be highest (high density + low rank), C lowest
    assert composites["A"] > composites["B"] > composites["C"]

    # zscore helper itself
    zs = zscore([1.0, 2.0, 3.0])
    assert abs(sum(zs)) < 1e-9
    assert abs(float(np.std(zs, ddof=0)) - 1.0) < 1e-9


def test_missing_measure_composite_from_subset() -> None:
    """(c) Domain missing a measure still gets a composite from available subset."""
    domain_vals = {
        "A": {
            "register_density": 0.8,
            "effective_output_rank": 1.5,
            "hard_constraint_recovery_pct": 0.5,
        },
        "B": {
            "register_density": 0.4,
            "effective_output_rank": 2.5,
            "hard_constraint_recovery_pct": None,  # missing
        },
        "C": {
            "register_density": 0.2,
            "effective_output_rank": None,  # missing
            "hard_constraint_recovery_pct": 0.1,
        },
    }
    composites, backing, _z = build_composite(
        domain_vals,
        measure_keys=[
            "register_density",
            "effective_output_rank",
            "hard_constraint_recovery_pct",
        ],
        lower_is_better={"effective_output_rank"},
    )
    # No domain dropped from composite entirely
    assert set(composites.keys()) == {"A", "B", "C"}
    assert "hard_constraint_recovery_pct" not in backing["B"]
    assert "effective_output_rank" not in backing["C"]
    assert "register_density" in backing["A"]
    assert "register_density" in backing["B"]
    assert "register_density" in backing["C"]
    # No fabrication: missing keys absent from backing
    assert all(math.isfinite(composites[d]) for d in composites)


def test_parse_core_sw_table_line_fixture() -> None:
    """(d) Ground-truth parse/cite against a fixture snippet (not live notes)."""
    fixture_lines = [
        "    corpus   gzip   gold  copy%  student  core_sw  crystal",
        "      code  0.251  0.692  91.9%    0.584    0.611   104.7%",
        "     wt103  0.400  0.316  73.1%    0.357    0.350    98.0%",
        "  wikitext  0.402  0.311  71.9%    0.343    0.342    99.5%",
        "   sudoku  0.269  0.307  99.9%    0.544    0.520    95.6%",
        "",
        "not a table row at all",
    ]
    parsed = [parse_core_sw_table_line(line) for line in fixture_lines]
    by_corpus = {p["corpus"]: p for p in parsed if p is not None}
    assert set(by_corpus) == {"code", "wt103", "wikitext", "sudoku"}
    assert by_corpus["code"]["core_sw"] == pytest.approx(0.611)
    assert by_corpus["wt103"]["core_sw"] == pytest.approx(0.350)
    assert by_corpus["wikitext"]["core_sw"] == pytest.approx(0.342)
    assert by_corpus["sudoku"]["core_sw"] == pytest.approx(0.520)
    # Citation-style fields populate
    assert "0.611" in by_corpus["code"]["source_line"]
    assert by_corpus["code"]["student"] == pytest.approx(0.584)
    # Header and junk yield None
    assert parse_core_sw_table_line(fixture_lines[0]) is None
    assert parse_core_sw_table_line("not a table row at all") is None

    # Short form (no copy% column) still works
    short = parse_core_sw_table_line(
        "    hist  0.405  0.333    0.433    0.394    91.0%"
    )
    assert short is not None
    assert short["corpus"] == "hist"
    assert short["core_sw"] == pytest.approx(0.394)


def test_cross_attribution_composite_dominant() -> None:
    """(e) Cross-attribution: composite-swap moves rho far more than GT-swap — sign-instability
    is composite-dominant, not a ground-truth-mining artifact. No live mine; pure
    arithmetic on hard-coded ranks."""
    result = cross_attribution()
    assert result["matrix"]["codex_comp_x_codex_gt"] == pytest.approx(0.6071, abs=1e-3)
    assert result["matrix"]["grok_comp_x_grok_gt"] == pytest.approx(-0.4286, abs=1e-3)
    assert result["matrix"]["codex_comp_x_grok_gt"] == pytest.approx(0.5714, abs=1e-3)
    assert result["matrix"]["grok_comp_x_codex_gt"] == pytest.approx(-0.1786, abs=1e-3)
    assert result["max_composite_swap_delta"] >= 0.78
    assert result["max_gt_swap_delta"] <= 0.25
    assert result["max_composite_swap_delta"] > result["max_gt_swap_delta"]
