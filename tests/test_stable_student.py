import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import certified_student as cst  # noqa: E402
from stable_student import Student, collapsed, decide, fc16, softmax_box  # noqa: E402

torch.manual_seed(0)


def _pair(vocab=40, n_out=12, T=10):
    E = torch.randn(vocab, 32, dtype=torch.float64)
    torch.manual_seed(1)
    a = cst.Student(E, n_out, T, d=16, L=2, H=2).double()
    b = Student(E, n_out, T, d=16, L=2, H=2).double()
    b.load_state_dict(a.state_dict())
    return a, b


def test_sorted_bound_equals_frozen_bound():
    a, b = _pair()
    ids, opt = torch.randint(0, 40, (6, 10)), torch.randint(0, 40, (9,))
    for x, y in zip(a.ibp(ids, opt, 1.0), b.ibp(ids, opt, 1.0), strict=True):
        assert torch.equal(x, y)


def test_ibp_sound_over_every_option():
    _, m = _pair()
    ids, opt = torch.randint(0, 40, (6, 10)), torch.randint(0, 40, (9,))
    lo, hi = m.ibp(ids, opt, 1.0)
    for o in opt:
        x = ids.clone()
        x[:, 0] = o
        z = m(x)
        assert bool(((z >= lo - 1e-9) & (z <= hi + 1e-9)).all())


def test_softmax_box_sound_and_ordered():
    T = 6
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    slo = torch.randn(50, T, T) * 3
    shi = slo + torch.rand(50, T, T) * 2
    slo, shi = slo.masked_fill(~mask, -1e9), shi.masked_fill(~mask, -1e9)
    alo, ahi = softmax_box(slo, shi, mask)
    assert bool((alo <= ahi).all())
    for _ in range(20):
        s = slo + torch.rand_like(slo) * (shi - slo)
        a = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
        assert bool(((a >= alo - 1e-6) & (a <= ahi + 1e-6)).all())


def test_softmax_box_ordered_under_extreme_float32_scores():
    T = 8
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    slo = (torch.randn(200, T, T) * 1e4).float()
    shi = slo + torch.rand(200, T, T).float() * 1e-3  # near-degenerate boxes, where rounding could cross
    slo, shi = slo.masked_fill(~mask, -1e9), shi.masked_fill(~mask, -1e9)
    alo, ahi = softmax_box(slo, shi, mask)
    assert bool((alo <= ahi).all())


def test_fc16_counts_only_non_seen_constant():
    _, m = _pair(vocab=40, n_out=6, T=10)
    sid = torch.randint(0, 40, (30, 5, 10))
    sid[:, :, 0] = torch.arange(30, 35)
    seen_ix = torch.tensor([0, 1, 2])
    dec = torch.zeros(30, 5, dtype=torch.long)
    v = fc16(m, sid, dec, torch.arange(30), seen_ix, torch.arange(30, 33))
    assert v is None or v == 0.0  # GPT-2 seen-constant everywhere: no false certificate possible
    dec[:, 1] = 1
    v = fc16(m, sid, dec, torch.arange(30), seen_ix, torch.arange(30, 33))
    assert v is None or v == 1.0


def _cells(plain, clip_faith, c16, c23):
    mm = lambda vs: dict(mean=sum(vs) / len(vs), range=[min(vs), max(vs)])  # noqa: E731
    return dict(
        plain=dict(faith_seen=mm([plain])),
        ibp_clip=dict(
            per_seed=[dict(faith_seen=f) for f in clip_faith],
            faith_seen=mm(clip_faith),
            cert16=mm([c16]),
            cert23=mm([c23]),
        ),
    )


def test_collapse_rule():
    assert collapsed(0.73, 0.84) and not collapsed(0.75, 0.84)


def test_decision_rules():
    v = decide(_cells(0.84, [0.84, 0.83, 0.85], 0.99, 0.0))
    assert [v[h]["verdict"] for h in ("H1", "H2", "H3", "H4")] == ["pass", "pass", "pass", "fail"]
    v = decide(_cells(0.84, [0.84, 0.30, 0.85], 0.85, 0.6))
    assert [v[h]["verdict"] for h in ("H1", "H2", "H3", "H4")] == ["fail", "fail", "fail", "pass"]
