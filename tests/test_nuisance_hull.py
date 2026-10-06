import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import cleanup_certificate as cc  # noqa: E402
import domain_exhaustion as de  # noqa: E402
from nuisance_hull import class_split, decide, hull_check, hull_verdicts  # noqa: E402


def _classes(seed=0, n_classes=40, k=8, noise=0.15):
    """Synthetic sigma-classes: k noisy variants around each code point."""
    rng = np.random.default_rng(seed)
    n_fill, n_role, d_f, d = 12, 3, 3, 40
    p = dict(
        ef=rng.normal(size=(n_fill, d_f)),
        er=rng.normal(size=(n_role, n_role)),
        W=rng.normal(size=(d, d_f * n_role)),
        b0=rng.normal(size=d),
    )
    sig = rng.integers(0, n_fill, size=(n_classes, 3))
    f = np.repeat(sig, k, axis=0)
    r = np.tile(np.arange(3), (len(f), 1))
    m = np.ones(f.shape, dtype=np.float32)
    group = np.repeat(np.arange(n_classes), k)
    x = cc.code_points(p, f, r, m)
    u = x + rng.normal(size=x.shape) * noise
    U = rng.normal(size=(30, d))
    return p, f, r, m, group, u, U, x, rng


def test_class_split_partitions_classes():
    tr, va, te = class_split(100, 92)
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(100))
    assert (len(tr), len(va), len(te)) == (60, 10, 30)


def test_hull_verdicts():
    full = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
    t = np.array([5, 5, 5, 5, 6, 5])
    group = np.array([0, 0, 0, 1, 1, 1])
    hull, const = hull_verdicts(full, t, group, np.array([0, 1]))
    assert hull.tolist() == [True, False] and const.tolist() == [True, False]


@pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")
def test_hull_check_passes_on_certified_classes_and_hulls_hold():
    p, f, r, m, group, u, U, x, rng = _classes()
    rd = cc.readout(p)
    F = cc.filler_sets(f, r, m, 3)
    t_host = (u @ U.T).argmax(1)
    res = de.evaluate_domain(p, rd, F, u, f, r, m, U, t_host, "cpu", rng)
    classes = np.arange(group.max() + 1)
    hull, _ = hull_verdicts(res["full"], t_host, group, classes)
    assert hull.any() and not hull.all()
    t_code = (x @ U.T).argmax(1)
    assert hull_check(u, f, r, m, group, classes, hull, p, rd, F, U, t_code, rng, "cpu") == 16 * hull.sum()


def test_hull_check_abort_fires():
    p, f, r, m, group, u, U, x, rng = _classes(noise=3.0)
    rd = cc.readout(p)
    F = cc.filler_sets(f, r, m, 3)
    t_code = (x @ U.T).argmax(1)
    classes = np.arange(group.max() + 1)
    with pytest.raises(AssertionError):  # claim every class is hull-certified when they are not
        hull_check(
            u, f, r, m, group, classes, np.ones(len(classes), dtype=bool), p, rd, F, U, t_code, rng, "cpu"
        )


def _cell(h_all, h_const):
    mm = lambda v: dict(mean=v, range=[v, v])  # noqa: E731
    return dict(H_all=mm(h_all), H_const=mm(h_const))


def test_decision_rules():
    v = decide({"dF32/mse": _cell(0.05, 0.1), "dF32/t6": _cell(0.12, 0.25)})
    assert v["H1"]["verdict"] == "pass" and v["H2"]["verdict"] == "pass"
    v = decide({"dF32/mse": _cell(0.05, 0.1), "dF32/t6": _cell(0.09, 0.19)})
    assert v["H1"]["verdict"] == "fail" and v["H2"]["verdict"] == "fail"
