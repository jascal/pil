import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import certified_substitutes as cs  # noqa: E402
from cleanup_certificate import (  # noqa: E402
    beta_and_decision,
    cleanup,
    code_points,
    filler_sets,
    gammas,
    readout,
    rho,
    soundness,
    tpr_params,
    verdict_python,
    verdict_souffle,
)


def _synthetic(seed=0, n_fill=12, n_role=4, d_f=3, d=40, n=200, k=3):
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
    return p, f, r, m, rng


def test_code_points_match_tpr_forward():
    torch.manual_seed(0)
    model = cs.TPR(7, 4, 3, d=20)
    rng = np.random.default_rng(1)
    f = rng.integers(0, 7, size=(10, 3))
    r = np.stack([rng.permutation(4)[:3] for _ in range(10)])
    m = np.ones((10, 3), dtype=np.float32)
    m[0, 2] = 0
    with torch.no_grad():
        ref = model(torch.tensor(f), torch.tensor(r), torch.tensor(m)).double().numpy()
    assert np.allclose(code_points(tpr_params(model), f, r, m), ref, atol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cleanup_recovers_sigma_inside_rho(seed):
    p, f, r, m, rng = _synthetic(seed)
    rd = readout(p)
    assert not rd["no_left_inverse"] and not rd["no_dual_readout"]
    F_sets = filler_sets(f, r, m, p["er"].shape[0])
    gam = gammas(p["ef"], F_sets)
    rh = rho(r, m, gam, rd["K"], np.sqrt((rd["Wd"] ** 2).sum(0)))
    x = code_points(p, f, r, m)
    noise = rng.normal(size=x.shape)
    noise *= (0.999 * rh / np.linalg.norm(noise, axis=1))[:, None]
    sig_hat, exact = cleanup(x + noise, p, rd, F_sets, f, r, m)
    assert exact.all() and (sig_hat == f).all()


def test_cleanup_can_fail_far_outside_rho():
    p, f, r, m, rng = _synthetic(3)
    rd = readout(p)
    F_sets = filler_sets(f, r, m, p["er"].shape[0])
    x = code_points(p, f, r, m)
    _, exact = cleanup(x + rng.normal(size=x.shape) * 100, p, rd, F_sets, f, r, m)
    assert not exact.all()


@pytest.mark.parametrize("seed", [0, 1])
def test_agreement_inside_beta_and_tight(seed):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(40, 6))
    x = rng.normal(size=(300, 6)) * 3
    t, be = beta_and_decision(x, U)
    assert (be > 0).all()
    d = rng.normal(size=x.shape)
    u_in = x + d * (0.999 * be / np.linalg.norm(d, axis=1))[:, None]
    assert ((u_in @ U.T).argmax(1) == t).all()
    # just past beta, toward the nearest boundary, the decision flips
    L = x @ U.T
    diff = U[t][:, None, :] - U[None]
    dist = np.linalg.norm(diff, axis=2)
    ratio = (L[np.arange(len(t)), t][:, None] - L) / np.where(dist > 0, dist, 1)
    ratio[np.arange(len(t)), t] = np.inf
    v = ratio.argmin(1)
    dirn = (U[t] - U[v]) / np.linalg.norm(U[t] - U[v], axis=1)[:, None]
    u_out = x - dirn * (1.001 * be)[:, None]
    assert ((u_out @ U.T).argmax(1) != t).all()


@pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")
def test_souffle_python_parity():
    rng = np.random.default_rng(5)
    n = np.abs(rng.normal(size=300))
    rh = np.abs(rng.normal(size=300))
    be = rng.normal(size=300)
    rh[:10] = np.inf
    be[10:20] = -np.inf
    n[20:30] = rh[20:30]  # boundary: not certified (strict, with slack)
    py = verdict_python(n, rh, be)
    assert (py == verdict_souffle(n, rh, be)).all()
    assert not py[20:30].any()
    assert py.any() and not py.all()


def test_abort_fires_on_planted_violation():
    cert = np.array([True, True, False])
    soundness(cert, np.array([True, True, False]), np.array([1, 2, 3]), np.array([1, 2, 0]))
    with pytest.raises(AssertionError):
        soundness(cert, np.array([True, False, True]), np.array([1, 2, 3]), np.array([1, 2, 3]))
    with pytest.raises(AssertionError):
        soundness(cert, np.array([True, True, True]), np.array([1, 2, 3]), np.array([1, 9, 3]))


def test_flags_trigger():
    p, *_ = _synthetic(0)
    p_rank = dict(p, W=np.concatenate([p["W"][:, :-1], p["W"][:, :1]], axis=1))
    assert readout(p_rank)["no_left_inverse"]
    er = p["er"].copy()
    er[1] = er[0]
    assert readout(dict(p, er=er))["no_dual_readout"]
    ok = readout(p)
    assert not ok["no_left_inverse"] and not ok["no_dual_readout"]
