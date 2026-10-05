"""Soundness of the IBP primitives: sampled points inside the input box map inside the output box."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from ibp_gpt2 import attn_ibp, gelu_ibp, gelu_new, linear_ibp, ln_ibp, mlp_ibp  # noqa: E402

torch.manual_seed(0)
DT = torch.float64


def _box(*shape, w=0.5):
    c = torch.randn(*shape, dtype=DT)
    r = torch.rand(*shape, dtype=DT) * w
    return c - r, c + r


def _samples(lo, hi, n=200):
    return lo + (hi - lo) * torch.rand(n, *lo.shape, dtype=DT)


def _inside(y, lo, hi, tol=1e-9):
    return bool(((y >= lo - tol) & (y <= hi + tol)).all())


def test_gelu_ibp_sound():
    lo, hi = _box(500, w=2.0)
    olo, ohi = gelu_ibp(lo, hi)
    assert _inside(gelu_new(_samples(lo, hi)), olo, ohi)


def test_linear_ibp_sound_and_exact_on_points():
    W, b = torch.randn(8, 5, dtype=DT), torch.randn(5, dtype=DT)
    lo, hi = _box(3, 8)
    olo, ohi = linear_ibp(lo, hi, W, b)
    assert _inside(_samples(lo, hi) @ W + b, olo, ohi)
    plo, phi = linear_ibp(lo, lo, W, b)
    assert torch.allclose(plo, phi) and torch.allclose(plo, lo @ W + b)


def test_ln_ibp_sound_and_exact_on_points():
    g, b = torch.randn(16, dtype=DT), torch.randn(16, dtype=DT)
    lo, hi = _box(4, 16, w=0.2)
    olo, ohi = ln_ibp(lo, hi, g, b)
    xs = _samples(lo, hi)
    ys = torch.nn.functional.layer_norm(xs, (16,), g, b, eps=1e-5)
    assert _inside(ys, olo, ohi)
    plo, phi = ln_ibp(lo, lo, g, b)
    assert torch.allclose(plo, torch.nn.functional.layer_norm(lo, (16,), g, b, eps=1e-5), atol=1e-12)


def _block(D=16, H=2):
    def mk(i, o):
        return SimpleNamespace(weight=torch.randn(i, o, dtype=DT) * 0.3, bias=torch.randn(o, dtype=DT) * 0.1)

    return SimpleNamespace(
        attn=SimpleNamespace(c_attn=mk(D, 3 * D), c_proj=mk(D, D)),
        mlp=SimpleNamespace(c_fc=mk(D, 4 * D), c_proj=mk(4 * D, D)),
    ), H


def _attn_ref(x, blk, H):
    B, T, D = x.shape
    dh = D // H
    qkv = x @ blk.attn.c_attn.weight + blk.attn.c_attn.bias
    q, k, v = qkv.view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
    s = q @ k.transpose(-1, -2) / dh**0.5
    s = s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
    o = torch.softmax(s, -1) @ v
    return o.permute(0, 2, 1, 3).reshape(B, T, D) @ blk.attn.c_proj.weight + blk.attn.c_proj.bias


def test_attn_ibp_sound():
    blk, H = _block()
    lo, hi = _box(1, 5, 16, w=0.1)
    olo, ohi = attn_ibp(lo, hi, blk, n_head=H)
    for x in _samples(lo, hi, n=100):
        assert _inside(_attn_ref(x, blk, H), olo, ohi)


def test_mlp_ibp_sound():
    blk, _ = _block()
    lo, hi = _box(2, 3, 16, w=0.2)
    olo, ohi = mlp_ibp(lo, hi, blk)
    xs = _samples(lo, hi, n=100)
    ref = gelu_new(xs @ blk.mlp.c_fc.weight + blk.mlp.c_fc.bias) @ blk.mlp.c_proj.weight + blk.mlp.c_proj.bias
    assert _inside(ref, olo, ohi)
