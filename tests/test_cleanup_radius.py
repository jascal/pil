import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import cleanup_certificate as cc  # noqa: E402
from cleanup_radius import (  # noqa: E402
    decide,
    directional_tables,
    rho_dir,
    role_maps,
    soundness,
    verdicts_python,
    verdicts_souffle,
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


def _setup(seed):
    p, f, r, m, rng = _synthetic(seed)
    rd = cc.readout(p)
    F = cc.filler_sets(f, r, m, p["er"].shape[0])
    maps = role_maps(p, rd)
    tables = directional_tables(p, maps, F)
    return p, f, r, m, rng, rd, F, maps, tables


def test_role_map_is_unbind_of_P():
    p, f, r, m, rng, rd, F, maps, _ = _setup(0)
    n = rng.normal(size=p["W"].shape[0])
    d_f, n_role = p["ef"].shape[1], p["er"].shape[0]
    Tn = (rd["P"] @ n).reshape(d_f, n_role)
    for s in range(n_role):
        assert np.allclose(maps[s] @ n, Tn @ rd["Wd"][:, s])


def test_tables_match_brute_force():
    p, f, r, m, rng, rd, F, maps, tables = _setup(1)
    s = 0
    Fs = F[s]
    for b in range(len(Fs)):
        for a in range(len(Fs)):
            if a == b:
                assert np.isinf(tables[s][b, a])
                continue
            d = p["ef"][Fs[a]] - p["ef"][Fs[b]]
            q = maps[s].T @ d
            assert tables[s][b, a] == pytest.approx((d @ d / 2) / np.linalg.norm(q), rel=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cleanup_exact_inside_rho_dir_and_dominates_rho(seed):
    p, f, r, m, rng, rd, F, maps, tables = _setup(seed)
    rdir = rho_dir(tables, F, f, r, m)
    rw = cc.rho(r, m, cc.gammas(p["ef"], F), rd["K"], np.sqrt((rd["Wd"] ** 2).sum(0)))
    assert (rdir >= rw * (1 - 1e-9)).all()
    x = cc.code_points(p, f, r, m)
    noise = rng.normal(size=x.shape)
    noise *= (0.999 * rdir / np.linalg.norm(noise, axis=1))[:, None]
    _, exact = cc.cleanup(x + noise, p, rd, F, f, r, m)
    assert exact.all()


def test_rho_dir_is_tight():
    # just past rho_dir, along q for the minimising (role, rival), clean-up of that role fails
    p, f, r, m, rng, rd, F, maps, tables = _setup(4)
    rdir = rho_dir(tables, F, f, r, m)
    x = cc.code_points(p, f, r, m)
    i = 0
    best = (np.inf, None)
    for j in range(f.shape[1]):
        s = int(r[i, j])
        pos = {int(v): k for k, v in enumerate(F[s])}
        b = pos[int(f[i, j])]
        a = int(np.argmin(tables[s][b]))
        if tables[s][b, a] < best[0]:
            best = (tables[s][b, a], (s, F[s][b], F[s][a]))
    s, fb, fa = best[1]
    assert best[0] == pytest.approx(rdir[i])
    q = maps[s].T @ (p["ef"][fa] - p["ef"][fb])
    u = x[i] + 1.001 * rdir[i] * q / np.linalg.norm(q)
    _, exact = cc.cleanup(u[None], p, rd, F, f[i : i + 1], r[i : i + 1], m[i : i + 1])
    assert not exact[0]


@pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")
def test_souffle_python_parity():
    rng = np.random.default_rng(7)
    n = np.abs(rng.normal(size=300))
    rd_ = np.abs(rng.normal(size=300)) * 2
    rw = rd_ * rng.uniform(0, 1, size=300)
    be = rng.normal(size=300)
    rd_[:10] = np.inf
    be[10:20] = -np.inf
    n[20:30] = rd_[20:30]
    py = verdicts_python(n, rd_, rw, be)
    so = verdicts_souffle(n, rd_, rw, be)
    for k in py:
        assert (py[k] == so[k]).all()
    assert not py["dir"][20:30].any()
    assert (py["full"] <= py["dir"]).all() and (py["wc"] <= py["dir"]).all()


def test_abort_fires_on_planted_violations():
    ok = dict(dir=np.array([True, False]), wc=np.array([False, False]), full=np.array([True, False]))
    exact = np.array([True, False])
    t = np.array([1, 2])
    soundness(ok, exact, t, t, np.array([2.0, 2.0]), np.array([1.0, 1.0]))
    with pytest.raises(AssertionError):
        soundness(ok, np.array([False, False]), t, t, np.array([2.0, 2.0]), np.array([1.0, 1.0]))
    with pytest.raises(AssertionError):
        soundness(ok, exact, t, np.array([9, 2]), np.array([2.0, 2.0]), np.array([1.0, 1.0]))
    with pytest.raises(AssertionError):
        soundness(ok, exact, t, t, np.array([0.5, 2.0]), np.array([1.0, 1.0]))


def test_flags_trigger():
    p, *_ = _synthetic(0)
    p_rank = dict(p, W=np.concatenate([p["W"][:, :-1], p["W"][:, :1]], axis=1))
    assert cc.readout(p_rank)["no_left_inverse"]
    er = p["er"].copy()
    er[1] = er[0]
    assert cc.readout(dict(p, er=er))["no_dual_readout"]


def _cell(E, S, gain):
    m = lambda v: None if v is None else dict(mean=v, range=[v, v])  # noqa: E731
    return dict(E=m(E), S=m(S), gain=m(gain))


def test_decision_rules():
    cells = {f"{t}/dF{d}/{o}": _cell(0.0, None, 2.0) for t, d in cc.SETTINGS for o in cc.OBJECTIVES}
    assert decide(cells)["H1"]["verdict"] == "untestable"
    assert decide(cells)["H2"]["verdict"] == "fail"
    cells["SVO/dF32/mse"] = _cell(0.8, 0.6, 20.0)
    cells["SVO/dF32/t6"] = _cell(0.8, 0.4, 20.0)
    cells["SVO/dF32/cert"] = _cell(0.3, 0.1, 20.0)
    for o in cc.OBJECTIVES:
        cells[f"SVO/dF8/{o}"] = _cell(0.0, None, 15.0)
    v = decide(cells)
    assert v["H1"]["n_eligible"] == 3 and v["H1"]["n_pass"] == 1 and v["H1"]["verdict"] == "fail"
    assert v["H2"]["n_pass"] == 2 and v["H2"]["verdict"] == "pass"
