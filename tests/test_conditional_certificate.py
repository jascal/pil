import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from conditional_certificate import auc, blind, choose_tau, decide, gate_scores, issued_metrics  # noqa: E402
from stable_student import Student  # noqa: E402

torch.manual_seed(0)


def test_gate_is_blind_to_position_zero():
    E = torch.randn(40, 32, dtype=torch.float64)
    gate = Student(E, 1, 9, d=16, L=2, H=2).double()
    sid = torch.randint(0, 40, (12, 1, 10)).expand(12, 5, 10).clone()
    sid[:, :, 0] = torch.arange(30, 35)  # variants differ only at position 0
    s = gate_scores(gate, sid, torch.arange(12))
    assert s.shape == (12,)
    assert blind(sid).shape[-1] == 9


def test_gate_scores_abort_if_variants_differ_beyond_position_zero():
    E = torch.randn(40, 32, dtype=torch.float64)
    gate = Student(E, 1, 9, d=16, L=2, H=2).double()
    sid = torch.randint(0, 40, (4, 1, 10)).expand(4, 3, 10).clone()
    sid[:, :, 0] = torch.arange(30, 33)
    gate_scores(gate, sid, torch.arange(4))  # differ only at position 0: fine
    sid[0, 2, 5] = (sid[0, 2, 5] + 1) % 40
    with pytest.raises(AssertionError):
        gate_scores(gate, sid, torch.arange(4))


def test_choose_tau():
    score = torch.arange(40, 0, -1).float()  # 40 classes, descending
    cert = torch.ones(40, dtype=torch.bool)
    ok = torch.tensor([True] * 25 + [False] + [True] * 4 + [False] * 10)
    # top 30 classes contain one miss: precision 29/30 = 0.967 >= 0.95; the top 31 give 29/31 < 0.95
    assert choose_tau(score, cert, ok) == 11.0
    assert (
        choose_tau(score, cert, ok, target=0.99) == 16.0
    )  # the top 25 are all ok; with the 26th, 25/26 < 0.99
    assert choose_tau(score[:10], cert[:10], ok[:10]) is None  # fewer than min_n
    cert2 = cert.clone()
    cert2[25] = False  # the miss is not certified, so it is never issued
    assert choose_tau(score, cert2, ok) == 10.0  # 29 ok candidates, then 29/30 >= 0.95 and 29/31 < 0.95


def test_choose_tau_includes_ties():
    score = torch.tensor([3.0] * 21 + [2.0] * 4)
    ok = torch.tensor([True] * 20 + [False] + [True] * 4)
    cert = torch.ones(25, dtype=torch.bool)
    tau = choose_tau(score, cert, ok)
    assert tau == 2.0  # 3.0 alone: 20/21 = 0.952 ok; 2.0: 24/25 = 0.96 ok; lowest qualifying is 2.0


def test_auc():
    assert auc(torch.tensor([3.0, 2.0, 1.0, 0.0]), torch.tensor([1, 1, 0, 0])) == 1.0
    assert auc(torch.tensor([0.0, 1.0]), torch.tensor([1, 0])) == 0.0
    assert auc(torch.tensor([1.0, 2.0]), torch.tensor([1, 1])) is None


def test_issued_metrics():
    issued = torch.tensor([True, True, False, True])
    p0 = torch.tensor([1, 2, 0, 3])
    gold = torch.tensor([[1, 1, 1], [2, 0, 2], [0, 0, 0], [3, 3, 0]])
    seen_ix, held_ix = torch.tensor([0, 1]), torch.tensor([2])
    seen_const = torch.tensor([True, False, True, True])
    held_const = torch.tensor([True, False, True, False])
    r = issued_metrics(issued, p0, gold, seen_ix, held_ix, seen_const, held_const)
    assert r["coverage"] == 0.75 and r["n_issued"] == 3
    assert r["FCi"] == pytest.approx(1 / 3)
    assert r["faith_issued"] == pytest.approx(2 / 3)
    assert r["held_const_issued"] == pytest.approx(1 / 3)
    assert r["held_faith_issued"] == pytest.approx(2 / 3)
    nothing = torch.zeros(4, dtype=torch.bool)
    none = issued_metrics(nothing, p0, gold, seen_ix, held_ix, seen_const, held_const)
    assert none["FCi"] is None and none["coverage"] == 0.0


def _cell(cov, fc, fi):
    mm = lambda v: None if v is None else dict(mean=v, range=[v, v])  # noqa: E731
    return dict(coverage=mm(cov), FCi=mm(fc), faith_issued=mm(fi))


@pytest.mark.parametrize(
    "cov,fc,fi,want",
    [
        (0.3, 0.05, 0.95, ("pass", "pass", "pass")),
        (0.1, 0.2, 0.8, ("fail", "fail", "fail")),
        (0.0, None, None, ("untestable", "fail", "untestable")),
    ],
)
def test_decision_rules(cov, fc, fi, want):
    v = decide(_cell(cov, fc, fi))
    assert (v["H1"]["verdict"], v["H2"]["verdict"], v["H3"]["verdict"]) == want
