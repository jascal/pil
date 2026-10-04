import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from hull_ceiling import (  # noqa: E402
    cell_label,
    classify,
    classify_lifted,
    headline,
    hull_distances,
    lifted_ceiling,
    soundness,
    soundness_lifted,
)


def _ref_hull_distance(U, t):
    """Reference: SLSQP over the simplex of the rivals."""
    rivals = np.delete(U, t, axis=0)
    k = len(rivals)
    obj = lambda w: 0.5 * np.sum((U[t] - w @ rivals) ** 2)  # noqa: E731
    jac = lambda w: (w @ rivals - U[t]) @ rivals.T  # noqa: E731
    res = minimize(
        obj,
        np.full(k, 1 / k),
        jac=jac,
        method="SLSQP",
        bounds=[(0, 1)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    return float(np.sqrt(2 * res.fun))


def test_known_triangle():
    # target at the origin; rivals form a triangle whose nearest edge is x = 1
    U = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    h = hull_distances(U, np.array([0]), cap=5000)
    assert h["h_lo"][0] <= 1.0 + 1e-9 <= h["h_hi"][0] + 2e-9
    assert h["h_hi"][0] - h["h_lo"][0] <= 1e-3 * h["h_hi"][0] + 1e-12
    assert h["converged"][0]


def test_interior_target_has_zero_distance():
    U = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]])
    h = hull_distances(U, np.array([0]), cap=3000)
    assert h["h_lo"][0] == pytest.approx(0.0, abs=1e-9)
    assert h["h_hi"][0] < 1e-2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bracket_contains_reference(seed):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(30, 5))
    targets = np.array([0, 3, 7, 11])
    h = hull_distances(U, targets, cap=20000)
    for i, t in enumerate(targets):
        ref = _ref_hull_distance(U, t)
        assert h["h_lo"][i] - 1e-6 <= ref <= h["h_hi"][i] + 1e-6


@pytest.mark.parametrize("seed", [0, 1])
def test_theorem_margin_below_ceiling(seed):
    # certificate_hull_ceiling: every residual's worst-rival margin is <= |r| * h(t)
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(40, 6))
    r = rng.normal(size=(500, 6)) * 3
    L = r @ U.T
    t = L.argmax(1)
    top = L[np.arange(len(t)), t]
    L[np.arange(len(t)), t] = -np.inf
    m = top - L.max(1)
    targets = np.unique(t)
    h = hull_distances(U, targets, cap=20000)
    h_hi = h["h_hi"][np.searchsorted(targets, t)]
    soundness(m, np.sqrt((r**2).sum(1)), h_hi)  # raises on a violation


def test_soundness_raises_on_violation():
    with pytest.raises(AssertionError):
        soundness(np.array([2.0]), np.array([1.0]), np.array([1.0]))


def test_classify_and_labels():
    m = np.array([1.0, 1.0, 1.0, 1.0])
    d = np.array([0.1, 2.0, 0.2, 0.45])
    unorm = np.ones(4)
    h_lo = np.array([1.0, 1.0, 1.0, 0.8])
    h_hi = np.array([1.0, 1.0, 1.0, 1.0])
    uniform = np.array([True, False, False, False])
    lab = classify(m, d, unorm, h_lo, h_hi, uniform)
    # 2d = 0.2 / 4.0 / 0.4 / 0.9 vs |u| h in [0.8, 1.0]
    assert list(lab) == ["certified", "CEILING", "ALIGNMENT", "UNDECIDED"]
    out = cell_label(lab)
    assert (out["refused"], out["ceiling"], out["other"], out["undecided"]) == (3, 1, 1, 1)
    assert out["label"] == "solver-limited"  # 1/3 undecided > 5%
    assert cell_label(np.array(["CEILING"] * 9 + ["ALIGNMENT"]))["label"] == "CEILING-BOUND"
    assert cell_label(np.array(["CEILING"] * 7 + ["ALIGNMENT"] * 3))["label"] == "MIXED"
    assert headline(["CEILING-BOUND", "CEILING-BOUND", "MIXED", "solver-limited"]) == "CEILING-BOUND"
    assert headline(["CEILING-BOUND", "ALIGNMENT-BOUND"]) == "MIXED"


def test_classify_lifted():
    d = np.array([0.1, 2.0, 0.2, 0.45])
    # two lift scales; context 1 is excluded by the second scale only; context 3 straddles
    c_lo = np.array([[5, 5], [5, 3.0], [1.0, 1.0], [0.8, 2.0]])
    c_hi = np.array([[5, 5], [5, 3.5], [1.0, 1.0], [1.0, 2.0]])
    uniform = np.array([True, False, False, False])
    lab = classify_lifted(d, c_lo, c_hi, uniform)
    assert list(lab) == ["certified", "CEILING", "NOT-EXCLUDED", "UNDECIDED"]
    assert cell_label(np.array(["NOT-EXCLUDED"] * 9 + ["CEILING"]), alt="NOT-EXCLUDED")["label"] == (
        "NOT-EXCLUDED-BOUND"
    )
    assert headline(["NOT-EXCLUDED-BOUND"] * 3 + ["MIXED"], alt="NOT-EXCLUDED") == "NOT-EXCLUDED-BOUND"


@pytest.mark.parametrize("seed", [0, 1])
def test_biased_theorem_margin_below_lifted_ceiling(seed):
    # certificate_hull_ceiling_biased on a random frame with a large context-constant component
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(40, 6))
    c = rng.normal(size=6) * 20
    u = c + rng.normal(size=(500, 6))
    L = u @ U.T
    t = L.argmax(1)
    top = L[np.arange(len(t)), t]
    L[np.arange(len(t)), t] = -np.inf
    m = top - L.max(1)
    c_lo, c_hi, conv = lifted_ceiling(U, c, u, t, cap=20000, device="cpu", log=lambda s: None)
    assert (c_lo <= c_hi + 1e-12).all()
    soundness_lifted(m, c_hi)  # raises on a violation
    # the lift is much tighter than the bias-free ceiling when c dominates
    targets = np.unique(t)
    h = hull_distances(U, targets, cap=20000)["h_hi"][np.searchsorted(targets, t)]
    assert np.median(c_hi.min(1)) < np.median(np.sqrt((u**2).sum(1)) * h)


def test_soundness_lifted_raises():
    with pytest.raises(AssertionError):
        soundness_lifted(np.array([2.0]), np.array([[3.0, 1.0]]))
