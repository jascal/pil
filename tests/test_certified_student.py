import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from certified_student import Student, certified, decide, evaluate, worst_logits  # noqa: E402

torch.manual_seed(0)


def _student(vocab=40, n_out=12, T=10):
    E = torch.randn(vocab, 32, dtype=torch.float64)
    return Student(E, n_out, T, d=16, L=2, H=2).double()


def test_ibp_sound_over_every_option():
    m = _student()
    for _ in range(4):
        ids = torch.randint(0, 40, (6, 10))
        opt = torch.randint(0, 40, (9,))
        lo, hi = m.ibp(ids, opt, 1.0)
        for o in opt:
            x = ids.clone()
            x[:, 0] = o
            z = m(x)
            assert bool(((z >= lo - 1e-9) & (z <= hi + 1e-9)).all())


def test_ibp_exact_with_one_option():
    m = _student()
    ids = torch.randint(0, 40, (5, 10))
    opt = torch.tensor([7])
    lo, hi = m.ibp(ids, opt, 1.0)
    x = ids.clone()
    x[:, 0] = 7
    assert float((hi - lo).abs().max()) == 0.0
    assert torch.allclose(lo, m(x), atol=1e-12)


def test_certified_and_worst_logits():
    lo = torch.tensor([[2.0, 0.0, 0.5], [1.0, 0.9, 0.0]])
    hi = torch.tensor([[3.0, 1.0, 1.5], [2.0, 1.5, 0.5]])
    y = torch.tensor([0, 0])
    assert certified(lo, hi, y).tolist() == [True, False]
    assert worst_logits(lo, hi, y).tolist() == [[2.0, 1.0, 1.5], [1.0, 1.5, 0.5]]


def test_evaluate_runs_and_soundness_holds():
    m = _student(vocab=40, n_out=6, T=10)
    sid = torch.randint(0, 40, (30, 5, 10))
    sid[:, :, 0] = torch.arange(30, 35)  # position 0 = the 5 "words"
    gold = torch.randint(0, 6, (30, 5))
    r = evaluate(
        m,
        sid,
        gold,
        torch.arange(30),
        torch.tensor([0, 1, 2]),
        torch.tensor([3, 4]),
        torch.arange(30, 35),
        torch.arange(30, 33),
    )
    assert (
        r["cert23"] <= r["cert16"] + 1e-12
    )  # interval bounds are inclusion-monotone: 23-box contains 16-box
    for k in ("faith_seen", "faith_held", "CF23"):
        assert 0.0 <= r[k] <= 1.0


def _c(cf, fc, hf):
    mm = lambda v: None if v is None else dict(mean=v, range=[v, v])  # noqa: E731
    return dict(ibp=dict(CF23=mm(cf), FC23=mm(fc), held_faith_among_cert=mm(hf)))


@pytest.mark.parametrize(
    "cf,fc,hf,want",
    [
        (0.6, 0.05, 0.95, ("pass", "pass", "pass")),
        (0.4, 0.2, 0.8, ("fail", "fail", "fail")),
        (0.0, None, None, ("fail", "untestable", "untestable")),
    ],
)
def test_decision_rules(cf, fc, hf, want):
    v = decide(_c(cf, fc, hf))
    assert (v["H1"]["verdict"], v["H2"]["verdict"], v["H3"]["verdict"]) == want
