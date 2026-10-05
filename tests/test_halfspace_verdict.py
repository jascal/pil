import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import cleanup_certificate as cc  # noqa: E402
import cleanup_radius as cr  # noqa: E402
from halfspace_verdict import (  # noqa: E402
    cleanup_slacks,
    decide,
    host_slacks,
    iff_check,
    neighbourhood_check,
    verdicts_python,
    verdicts_souffle,
)


def _setup(seed, n=300, noise=0.0):
    rng = np.random.default_rng(seed)
    n_fill, n_role, d_f, d, k = 12, 4, 3, 40, 3
    p = dict(
        ef=rng.normal(size=(n_fill, d_f)),
        er=rng.normal(size=(n_role, n_role)),
        W=rng.normal(size=(d, d_f * n_role)),
        b0=rng.normal(size=d),
    )
    f = rng.integers(0, n_fill, size=(n, k))
    r = np.stack([rng.permutation(n_role)[:k] for _ in range(n)])
    m = np.ones((n, k), dtype=np.float32)
    rd = cc.readout(p)
    F = cc.filler_sets(f, r, m, n_role)
    x = cc.code_points(p, f, r, m)
    u = x + rng.normal(size=x.shape) * noise
    U = rng.normal(size=(30, d))
    return p, rd, F, f, r, m, x, u, U, rng


@pytest.mark.parametrize("seed,noise", [(0, 0.3), (1, 1.0), (2, 3.0)])
def test_clean_verdict_is_iff_with_exact_cleanup(seed, noise):
    p, rd, F, f, r, m, x, u, U, rng = _setup(seed, noise=noise)
    maps = cr.role_maps(p, rd)
    rows, min_slack, _, _ = cleanup_slacks(p, rd, maps, F, u - x, f, r, m)
    _, exact = cc.cleanup(u, p, rd, F, f, r, m)
    t_code = (x @ U.T).argmax(1)
    hmin, _, _ = host_slacks(u, t_code, U)
    v = verdicts_python(len(u), rows, hmin)
    iff_check(v, exact, (u @ U.T).argmax(1) == t_code, min_slack, hmin)  # raises on mismatch
    assert exact.any() and not exact.all()  # the test exercises both sides


def test_slacks_match_brute_force():
    p, rd, F, f, r, m, x, u, U, rng = _setup(3, n=20, noise=1.0)
    maps = cr.role_maps(p, rd)
    rows, *_ = cleanup_slacks(p, rd, maps, F, u - x, f, r, m)
    got = {(c, s, a): v for c, s, a, v in rows}
    n = u - x
    for c in range(len(u)):
        for j in range(f.shape[1]):
            s, b = int(r[c, j]), int(f[c, j])
            for a in F[s]:
                if a == b:
                    continue
                d = p["ef"][a] - p["ef"][b]
                want = d @ d / 2 - (maps[s] @ n[c]) @ d
                assert got[(c, s, int(a))] == pytest.approx(want, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("seed", [0, 1])
def test_neighbourhood_holds_and_is_tight(seed):
    p, rd, F, f, r, m, x, u, U, rng = _setup(seed, noise=0.3)
    maps = cr.role_maps(p, rd)
    rows, min_slack, delta_c, worst_c = cleanup_slacks(p, rd, maps, F, u - x, f, r, m)
    t_code = (x @ U.T).argmax(1)
    hmin, delta_h, worst_h = host_slacks(u, t_code, U)
    v = verdicts_python(len(u), rows, hmin)
    rho_loc = np.minimum(delta_c, delta_h)
    assert v["full"].sum() > 10
    assert (
        neighbourhood_check(u, v["full"], rho_loc, worst_c, worst_h, t_code, p, rd, F, f, r, m, U, rng, "cpu")
        > 0
    )
    # just past rho_loc, along the binding worst-case direction, the certificate's conclusion fails
    idx = np.where(v["full"])[0]
    i = idx[0]
    dvec = worst_c[i] if delta_c[i] <= delta_h[i] else worst_h[i]
    up = u[i] + 1.001 * rho_loc[i] * dvec
    _, ex = cc.cleanup(up[None], p, rd, F, f[i : i + 1], r[i : i + 1], m[i : i + 1])
    assert (not ex[0]) or (up @ U.T).argmax() != t_code[i]


@pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")
def test_souffle_python_parity():
    p, rd, F, f, r, m, x, u, U, rng = _setup(4, n=80, noise=1.0)
    maps = cr.role_maps(p, rd)
    rows, *_ = cleanup_slacks(p, rd, maps, F, u - x, f, r, m)
    hmin, _, _ = host_slacks(u, (x @ U.T).argmax(1), U)
    py = verdicts_python(len(u), rows, hmin)
    so = verdicts_souffle(len(u), rows, hmin)
    for k in py:
        assert (py[k] == so[k]).all()
    assert py["full"].any() and not py["full"].all()


def test_abort_fires_on_planted_mismatch():
    v = dict(clean=np.array([True, False]), agree=np.array([True, True]), full=np.array([True, False]))
    iff_check(v, np.array([True, False]), np.array([True, True]), np.array([1.0, -1.0]), np.array([1.0, 1.0]))
    with pytest.raises(AssertionError):
        iff_check(
            v, np.array([False, False]), np.array([True, True]), np.array([1.0, -1.0]), np.array([1.0, 1.0])
        )
    # a tie is reported, not an abort
    out = iff_check(
        v, np.array([False, False]), np.array([True, True]), np.array([1e-9, -1.0]), np.array([1.0, 1.0])
    )
    assert out["ties_clean"] == 1


def test_neighbourhood_abort_fires():
    p, rd, F, f, r, m, x, u, U, rng = _setup(5, noise=0.3)
    t_code = (x @ U.T).argmax(1)
    full = np.ones(len(u), dtype=bool)
    huge = np.full(len(u), 1e6)
    maps = cr.role_maps(p, rd)
    _, _, _, worst_c = cleanup_slacks(p, rd, maps, F, u - x, f, r, m)
    _, _, worst_h = host_slacks(u, t_code, U)
    with pytest.raises(AssertionError):
        neighbourhood_check(u, full, huge, worst_c, worst_h, t_code, p, rd, F, f, r, m, U, rng, "cpu")


def _c(F, ratio):
    mm = lambda v: None if v is None else dict(mean=v, range=[v, v])  # noqa: E731
    return dict(F_cov=mm(F), median_rho_loc_over_n=mm(ratio))


def test_decision_rules():
    cells = {f"{t}/dF{d}/{o}": _c(0.0, None) for t, d in cc.SETTINGS for o in cc.OBJECTIVES}
    v = decide(cells)
    assert v["H2"]["verdict"] == "fail" and v["H3"]["verdict"] == "untestable"
    cells["SVO/dF32/mse"] = _c(0.6, 0.02)
    cells["SVO/dF32/t6"] = _c(0.1, 0.005)
    v = decide(cells)
    assert v["H2"]["verdict"] == "pass" and v["H3"]["verdict"] == "fail"
    cells["SVO/dF32/t6"] = _c(0.1, 0.011)
    assert decide(cells)["H3"]["verdict"] == "pass"
