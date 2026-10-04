"""Decision rules and boundary cases pinned by the signed #131 prereg."""

import math

import numpy as np

from pil.directional_calibration import bin_index, bounds, candidate, fit, verdict, wilson
from pil.early_exit import ExitRecord


def record(radius, ynorm, push):
    r = ExitRecord(sid="toy", pos=0, pred=0, s=1, s_resid=0, t=np.array([0, 0]),
                   R=np.array([radius, 0.0]), ynorm=np.array([ynorm, 1.0]),
                   suffix=np.zeros(2))
    return r, np.array([push, 0.0])


def test_bin_edge_tie_uses_lower_bin_and_linear_quantile():
    cal = [record(i, 1, i / 10) for i in range(1, 31)]
    f = fit(cal, "C1")
    expected = np.quantile(np.arange(1, 31), [1 / 3, 2 / 3], method="linear")
    assert np.allclose(f.edges[0], expected)
    assert bin_index(expected[0], f.edges[0]) == 0
    assert bin_index(expected[1], f.edges[0]) == 1


def test_small_bin_abstains_and_zero_norm_never_certifies():
    cal = [record(i, 1, 0.1) for i in range(1, 16)]
    f = fit(cal, "C2")
    assert np.isinf(f.q).all()  # fewer than 19 points per bin at alpha=.05
    assert math.isinf(bounds([record(4, 0, 0)], f)[0, 0])


def test_finite_group_quantile_and_negative_push():
    cal = [record(1, 1, -0.1) for _ in range(19)]
    f = fit(cal, "C0")
    assert f.q[0, 0] == -0.1
    assert bounds([record(1, 2, 0)], f)[0, 0] == -0.2


def metrics(coverage, violations, net, n=100):
    return {"N": n, "coverage": coverage, "violations": violations, "net_every": net}


def test_selection_excludes_zero_certifications_and_uses_fixed_tie_order():
    rows = {"C0": metrics(0.3, 0, 1.0), "C1": metrics(0.3, 0, 1.0),
            "C2": metrics(0.3, 0, 1.0)}
    assert candidate(rows) == "C1"
    rows["C1"] = metrics(0, 0, 10)
    assert candidate(rows) == "C2"
    rows["C2"] = metrics(0.3, 2, 10)
    assert candidate(rows) == "C0"
    rows["C0"] = metrics(0, 0, 10)
    assert candidate(rows) is None


def test_verdicts_are_exhaustive_and_failed_precedes_partial():
    assert verdict(metrics(0, 0, 2), True) == "FAILED"
    assert verdict(metrics(0.09, 0, 2), True) == "FAILED"
    assert verdict(metrics(0.10, 2, 2), True) == "FAILED"
    assert verdict(metrics(0.10, 1, 2), True) == "PARTIAL"
    assert verdict(metrics(0.25, 2, 1), True) == "CONFIRMED"
    assert verdict(metrics(0.25, 3, 1), True) == "FAILED"
    assert verdict(metrics(0.25, 0, 0.9), True) == "PARTIAL"
    assert verdict(metrics(0.25, 0, 1), False) == "VOID"
    assert wilson(0, 0) is None
