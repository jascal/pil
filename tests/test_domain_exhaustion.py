import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import cleanup_certificate as cc  # noqa: E402
from domain_exhaustion import (  # noqa: E402
    N_OCC,
    N_VERB,
    canonical_index,
    decide,
    evaluate_domain,
    fit_metrics,
    in_vocab,
    partition,
)


def test_canonical_index_is_a_bijection_on_the_domain():
    f = np.array(
        [[s, N_OCC + v, o] for s in range(N_OCC) for o in range(N_OCC) if o != s for v in range(N_VERB)]
    )
    idx = canonical_index(f)
    assert len(f) == N_OCC * (N_OCC - 1) * N_VERB
    assert len(np.unique(idx)) == len(f)
    assert idx.min() >= 0 and idx.max() < N_OCC * N_OCC * N_VERB


def test_partition_covers_domain_without_overlap():
    tr, va, te, never = partition(1000, 400, 72)
    allp = np.concatenate([tr, va, te, never])
    assert sorted(allp.tolist()) == list(range(1000))
    assert len(tr) == 240 and len(va) == 40 and len(te) == 120 and len(never) == 600
    assert never.min() == 400


def test_out_of_vocabulary_contexts_are_uncertified():
    rng = np.random.default_rng(0)
    n_fill, n_role, d_f, d = 12, 3, 3, 40
    p = dict(
        ef=rng.normal(size=(n_fill, d_f)),
        er=rng.normal(size=(n_role, n_role)),
        W=rng.normal(size=(d, d_f * n_role)),
        b0=rng.normal(size=d),
    )
    f = rng.integers(0, n_fill, size=(200, 3))
    r = np.tile(np.arange(3), (200, 1))
    m = np.ones((200, 3), dtype=np.float32)
    tr = np.arange(100)
    F = cc.filler_sets(f[tr], r[tr], m[tr], n_role)
    iv = in_vocab(f, r, m, F)
    rd = cc.readout(p)
    x = cc.code_points(p, f, r, m)
    U = rng.normal(size=(30, d))
    u = x + rng.normal(size=x.shape) * 0.2
    t_host = (u @ U.T).argmax(1)
    res = evaluate_domain(p, rd, F, u, f, r, m, U, t_host, "cpu", rng)
    assert not res["full"][~iv].any()
    assert res["oov"] == int((~iv).sum())
    fm = fit_metrics(
        res, dict(D=np.arange(200), train=tr, test=np.arange(100, 150), never=np.arange(150, 200))
    )
    assert fm["uncertified_D"] == int((~res["full"]).sum())
    assert fm["domain_wide"] == bool(res["full"].all())


def _cell(d, test, never):
    mm = lambda v: dict(mean=v, range=[v, v])  # noqa: E731
    return dict(
        F_cov_D=mm(d), F_cov_test=mm(test), F_cov_never=mm(never), min_uncertified_D=5, any_domain_wide=False
    )


def test_decision_rules():
    cells = {"SVO/dF32/mse": _cell(0.6, 0.65, 0.62), "SVO/dF8/mse": _cell(0.01, 0.01, 0.01)}
    v = decide(cells)
    assert v["H1"]["verdict"] == "pass" and v["H2"]["verdict"] == "pass"
    cells["SVO/dF32/mse"] = _cell(0.4, 0.65, 0.5)
    v = decide(cells)
    assert v["H1"]["verdict"] == "fail" and v["H2"]["verdict"] == "fail"
    v = decide({"SVO/dF8/mse": _cell(0.01, 0.01, 0.01)})
    assert v["H2"]["verdict"] == "untestable"
