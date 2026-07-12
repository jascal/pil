"""Parity: emitted khop package served by rosetta equals certified mir_khop2."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

import torch  # noqa: E402
from serve_package import decide, load_package  # noqa: E402

from experiments.wyly_selfcompile import L, emit_khop_package, gen_khop2, mir_khop2  # noqa: E402


def _window(filler, q, overrides=None):
    ctx = [filler] * L
    ctx[-1] = q
    for i, v in (overrides or {}).items():
        ctx[i] = v
    return ctx


def _assert_parity(lo, hi, ctx, idioms, ngrams, m):
    mir = int(mir_khop2(lo, hi)(torch.tensor([ctx]))[0].item())
    served = decide(ctx, idioms, ngrams, m)
    if mir == -1:
        assert served is None, (
            f"mir abstained (-1) but decide() returned {served!r} instead of the loader's "
            "abstain signal (None) -- parity broken on the abstain side")
    else:
        assert served is not None, (
            f"mir fired with answer {mir} but decide() abstained (None) -- parity broken "
            "on the fire side")
        assert served["answer"] == mir, (
            f"fired but disagree: mir={mir} served={served['answer']}")
    return mir


def _naive_no_exclusion(ctx, lo, hi):
    q = ctx[-1]
    if not (lo <= q <= hi):
        return None
    cand1 = [i for i in range(len(ctx) - 1) if ctx[i] == q]
    if not cand1:
        return None
    p1 = max(cand1)
    bridge = ctx[min(p1 + 1, len(ctx) - 1)]
    cand2 = [i for i in range(len(ctx) - 1) if ctx[i] == bridge]  # NOTE: no i != p1+1 guard
    if not cand2:
        return None
    p2 = max(cand2)
    return ctx[min(p2 + 1, len(ctx) - 1)]


@pytest.mark.parametrize("lo,hi", [(224, 383), (10, 60)])
def test_khop_parity_hand_built(tmp_path, lo, hi):
    base = lo
    filler = lo - 50
    out = tmp_path / f"khop_{lo}_{hi}"
    emit_khop_package(lo, hi, L, out)
    idioms, ngrams, m = load_package(out / "manifest.json")

    # a) plain 2-hop hit
    q = base + 0
    ctx_a = _window(filler, q, {5: q, 6: base + 6, 2: base + 6, 3: base + 76})
    mir_a = _assert_parity(lo, hi, ctx_a, idioms, ngrams, m)
    assert mir_a == base + 76

    # b) bridge-site exclusion (load-bearing)
    ctx_b = _window(filler, q, {
        5: q, 6: base + 6, 2: base + 6, 3: base + 76, 7: base + 77,
    })
    mir_b = _assert_parity(lo, hi, ctx_b, idioms, ngrams, m)
    assert mir_b == base + 76
    naive = _naive_no_exclusion(ctx_b, lo, hi)
    assert naive == base + 77
    assert naive != mir_b

    # c1) q out of [lo, hi]
    ctx_c1 = _window(filler, lo - 1)
    mir_c1 = _assert_parity(lo, hi, ctx_c1, idioms, ngrams, m)
    assert mir_c1 == -1

    # c2) no first-hop match
    ctx_c2 = _window(filler, q)
    mir_c2 = _assert_parity(lo, hi, ctx_c2, idioms, ngrams, m)
    assert mir_c2 == -1

    # c3) first hop matches, no second-hop match (bridge site excluded)
    ctx_c3 = _window(filler, q, {5: q, 6: base + 6})
    mir_c3 = _assert_parity(lo, hi, ctx_c3, idioms, ngrams, m)
    assert mir_c3 == -1


def test_khop_parity_random_batch(tmp_path):
    n = 200
    g = torch.Generator().manual_seed(0)
    x, _y, _s = gen_khop2(n, g)
    lo, hi = int(x[:, -1].min()), int(x[:, -1].max())
    emit_khop_package(lo, hi, L, tmp_path / "khop_random")
    idioms, ngrams, m = load_package(tmp_path / "khop_random" / "manifest.json")
    fired_count = 0
    for i in range(n):
        ctx = x[i].tolist()
        mir = _assert_parity(lo, hi, ctx, idioms, ngrams, m)
        if mir != -1:
            fired_count += 1
    assert fired_count > 0


def test_emit_khop_package_writes_manifest(tmp_path):
    manifest = emit_khop_package(224, 383, L, tmp_path)
    assert (tmp_path / "manifest.json").exists()
    assert manifest["rules"][0] == {
        "kind": "khop", "lo": 224, "hi": 383, "id": "khop2",
        "tier": "trusted", "basis": "certified", "confidence": 1.0,
        "citation": {"program": "dl_khop2", "cert": "souffle"},
    }
    idioms, _ngrams, _m = load_package(tmp_path / "manifest.json")
    assert len(idioms) == 1
    assert idioms[0]["kind"] == "khop"
    assert idioms[0]["lo"] == 224
    assert idioms[0]["hi"] == 383
