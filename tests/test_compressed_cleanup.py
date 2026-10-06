import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import cleanup_certificate as cc  # noqa: E402
import cleanup_radius as cr  # noqa: E402
import halfspace_verdict as hv  # noqa: E402
from compressed_cleanup import (  # noqa: E402
    compression_error,
    decide,
    decoded_slacks,
    readout_ridge,
    ridge_decoder,
)


def _setup(seed=0, d=10, n_fill=12, n_role=4, d_f=3, n=300, k=3, noise=0.2):
    """A compressed TPR: d_f * n_role = 12 > d = 10, so W has no left inverse."""
    rng = np.random.default_rng(seed)
    p = dict(
        ef=rng.normal(size=(n_fill, d_f)),
        er=rng.normal(size=(n_role, n_role)),
        W=rng.normal(size=(d, d_f * n_role)),
        b0=rng.normal(size=d),
    )
    f = rng.integers(0, n_fill, size=(n, k))
    r = np.stack([rng.permutation(n_role)[:k] for _ in range(n)])
    m = np.ones((n, k), dtype=np.float32)
    x = cc.code_points(p, f, r, m)
    u = x + rng.normal(size=x.shape) * noise
    return p, f, r, m, x, u, rng


def _all(p, f, r, m, u):
    rd = readout_ridge(p)
    F = cc.filler_sets(f, r, m, p["er"].shape[0])
    T, E, eps, nt = compression_error(p, rd["P"], f, r, m)
    maps = cr.role_maps(p, rd)
    rows, ms, rdir, reps = decoded_slacks(p, rd, maps, F, u, f, r, m, E, eps, nt)
    return rd, F, T, E, eps, nt, rows, ms, rdir, reps


@pytest.mark.parametrize("seed,noise", [(0, 0.05), (1, 0.3), (2, 1.0)])
def test_decoded_slacks_are_iff_with_cleanup_through_the_ridge_decoder(seed, noise):
    p, f, r, m, x, u, rng = _setup(seed, noise=noise)
    rd, F, T, E, eps, nt, rows, ms, rdir, reps = _all(p, f, r, m, u)
    _, exact = cc.cleanup(u, p, rd, F, f, r, m)
    v = hv.verdicts_python(len(u), rows, np.ones(len(u)))
    hv.iff_check(v, exact, np.ones(len(u), dtype=bool), ms, np.ones(len(u)))
    assert (eps > 0).any()  # genuinely compressed


def test_compressed_has_eps_and_injective_ridge_tends_to_exact():
    p, f, r, m, x, u, rng = _setup(0)
    *_, eps, _ = compression_error(p, readout_ridge(p)["P"], f, r, m)
    assert eps.max() > 1e-3
    p2, f2, r2, m2, *_ = _setup(1, d=40)  # injective W
    P0 = ridge_decoder(p2["W"], lam_rel=0.0)
    assert compression_error(p2, P0, f2, r2, m2)[2].max() < 1e-8


def test_rho_eps_at_most_rho_dir_and_cleanup_exact_inside_rho_eps():
    p, f, r, m, x, u, rng = _setup(3, noise=0.0)
    rd, F, T, E, eps, nt, rows, ms, rdir, reps = _all(p, f, r, m, x)
    assert (reps <= rdir + 1e-12).all()
    ok = reps > 0
    if ok.any():
        noise = rng.normal(size=x.shape)
        noise *= (0.999 * np.clip(reps, 0, None) / np.linalg.norm(noise, axis=1))[:, None]
        _, exact = cc.cleanup(x[ok] + noise[ok], p, rd, F, f[ok], r[ok], m[ok])
        assert exact.all()


@pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")
def test_souffle_python_parity():
    p, f, r, m, x, u, rng = _setup(4, n=80, noise=0.3)
    rd, F, T, E, eps, nt, rows, ms, rdir, reps = _all(p, f, r, m, u)
    hmin = rng.normal(size=len(u))
    py = hv.verdicts_python(len(u), rows, hmin)
    so = hv.verdicts_souffle(len(u), rows, hmin)
    for k in py:
        assert (py[k] == so[k]).all()


def test_decision_rule():
    mm = lambda v: dict(F_cov=dict(mean=v, range=[v, v]))  # noqa: E731
    assert decide({"LIST/dF32/mse": mm(0.06), "SVO/dF32/mse": mm(0.0)})["H1"]["verdict"] == "pass"
    assert decide({"LIST/dF32/mse": mm(0.04), "SVO/dF32/mse": mm(0.9)})["H1"]["verdict"] == "fail"
