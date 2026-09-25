"""Certified early exit: the certificate is sound and tight; the weight bound dominates int8 writes."""

import json
import math

import numpy as np
import pytest

from pil.early_exit import (
    Bundle,
    Readout,
    act_quant_factor,
    certify,
    conformal_quantile,
    fit_scale,
    suffix_weight_bound,
    weight_bounds,
)


def _readout(rng, V=40, d=12):
    U = rng.normal(size=(V, d)).astype(np.float32)
    return Readout(U, rng.uniform(0.5, 2.0, size=d).astype(np.float32))


def test_certificate_sound_under_random_and_worst_case_suffixes():
    rng = np.random.default_rng(0)
    certified_any = False
    for _ in range(300):
        ro = _readout(rng)
        x = rng.normal(size=ro.W.shape[1]) * rng.uniform(0.5, 4.0)
        t, R = ro.radius(ro.W @ x)
        if not math.isfinite(R) or R <= 0:
            continue
        B = 0.99 * R
        _, ok = certify(ro.W @ x, ro, B)
        assert ok
        certified_any = True
        for _ in range(20):                                   # random suffixes inside the bound never flip
            e = rng.normal(size=x.shape)
            e *= B * rng.uniform(0, 1) / np.linalg.norm(e)
            assert int(np.argmax(ro.W @ (x + e))) == t
        gaps = (ro.W[t] - ro.W) @ x                           # worst case: push straight at the closest rival
        dist = np.linalg.norm(ro.W[t] - ro.W, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dist > 0, gaps / dist, np.inf)
        ratio[t] = np.inf
        v = int(np.argmin(ratio))
        push = (ro.W[v] - ro.W[t]) / np.linalg.norm(ro.W[v] - ro.W[t])
        assert int(np.argmax(ro.W @ (x + B * push))) == t     # inside R: holds
        assert int(np.argmax(ro.W @ (x + 1.01 * R * push))) != t   # just past R: flips (tight)
    assert certified_any


def test_tie_has_zero_radius():
    ro = Readout(np.array([[1.0, 0.0], [0.0, 1.0]], np.float32), np.ones(2, np.float32))
    _, R = ro.radius(np.array([1.0, 1.0]))
    assert R == 0.0


def test_conformal_quantile_rank():
    x = np.arange(1, 20, dtype=float)                         # n = 19, alpha = .05 -> rank ceil(20*.95) = 19
    assert conformal_quantile(x, 0.05) == 19.0
    assert conformal_quantile(x[:10], 0.05) == math.inf       # rank 11 > n = 10


def test_fit_scale_exact():
    rng = np.random.default_rng(1)
    theta, e = rng.uniform(0.5, 2, 16), rng.normal(size=16)
    s, resid = fit_scale(0.37 * theta * e, e, theta)
    assert s == pytest.approx(0.37) and resid < 1e-12


def test_suffix_weight_bound():
    S = suffix_weight_bound(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]))
    assert S.tolist() == [55.0, 33.0, 0.0]


# ── the weight bound vs an emulation of fieldrun's int8 layer ──────────────────────────────────────────────


def _write_bundle(tmp_path, arrays, config, eps=1e-6):
    blob, index, off = bytearray(), [], 0
    for name, (dtype, arr) in arrays.items():
        b = arr.astype({"f16": np.float16, "i8": np.int8}[dtype]).tobytes()
        index.append({"name": name, "dtype": dtype, "offset": off, "bytes": len(b), "shape": list(arr.shape)})
        blob += b
        off += len(b)
    stem = tmp_path / "toy"
    (tmp_path / "toy.fieldrun.bin").write_bytes(bytes(blob))
    meta = {"format": "fieldrun-bundle", "version": 1, "arch": "rope", "config": config,
            "config_f": [1e6, eps], "eos": 0, "arrays": index}
    (tmp_path / "toy.fieldrun.json").write_text(json.dumps(meta))
    return stem


def _i8(rng, rows, cols):
    w = rng.normal(size=(rows, cols)) * 0.2
    scale = np.abs(w).max(axis=0) / 127.0
    return np.clip(np.round(w / scale), -127, 127), scale


def _act_quant(a, outliers=2):
    """fieldrun's I8 activation path: bulk scale from the (outliers)-th largest |a|, outliers kept exact."""
    order = np.argsort(-np.abs(a))
    bulk_max = np.abs(a)[order[outliers]]
    sa = bulk_max / 127.0 if bulk_max > 0 else 1.0
    q = np.clip(np.round(a / sa), -127, 127) * sa
    q[order[:outliers]] = a[order[:outliers]]
    return q


def _rms(x, g, eps=1e-6):
    return g * x / np.sqrt((x * x).mean() + eps)


def test_weight_bound_dominates_emulated_int8_writes(tmp_path):
    rng = np.random.default_rng(2)
    d, nh, nkv, hd, dff = 16, 4, 2, 4, 24
    arrays = {"embed": ("f16", rng.normal(size=(30, d))), "norm": ("f16", rng.uniform(0.5, 2, d))}
    q = {}
    shapes = {"self_attn.v_proj": (d, nkv * hd), "self_attn.o_proj": (d, d),
              "mlp.gate_proj": (d, dff), "mlp.up_proj": (d, dff), "mlp.down_proj": (dff, d)}
    for name, (r, c) in shapes.items():
        w, sc = _i8(rng, r, c)
        q[name] = w * sc[None, :]
        arrays[f"l0.{name}"] = ("i8", w)
        arrays[f"l0.{name}__scale"] = ("f16", sc)
    for name in ("in_ln", "post_ln"):
        arrays[f"l0.{name}"] = ("f16", rng.uniform(0.5, 3, d))
    arrays["l0.self_attn.v_proj.bias"] = ("f16", rng.normal(size=nkv * hd))
    stem = _write_bundle(tmp_path, arrays, [1, nh, nkv, hd, d, dff, 30, 1])
    b = Bundle(stem)
    A, M = weight_bounds(b)
    g_in, g_post = b.f32("l0.in_ln"), b.f32("l0.post_ln")
    wv, bv = b.f32("l0.self_attn.v_proj"), b.f32("l0.self_attn.v_proj.bias")
    names = ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    wo, wg, wu, wd = (b.f32(f"l0.{n}") for n in names)
    for _ in range(200):
        xs = rng.normal(size=(5, d)) * rng.uniform(0.1, 50)     # 5 source positions
        vals = np.stack([_act_quant(_rms(x, g_in)) @ wv + bv for x in xs])        # (5, nkv*hd)
        heads = []
        for h in range(nh):
            g = h // (nh // nkv)
            a = rng.dirichlet(np.ones(5))
            heads.append(a @ vals[:, g * hd : (g + 1) * hd])
        attn = _act_quant(np.concatenate(heads)) @ wo
        assert np.linalg.norm(attn) <= A[0] * (1 + 1e-5)
        hq = _act_quant(_rms(xs[0], g_post))
        gate, up = hq @ wg, hq @ wu
        mlp = _act_quant(gate / (1 + np.exp(-gate)) * up) @ wd
        assert np.linalg.norm(mlp) <= M[0] * (1 + 1e-5)


def test_act_quant_factor_covers_rounding():
    rng = np.random.default_rng(3)
    for n in (8, 64, 896):
        for _ in range(100):
            a = rng.normal(size=n) * rng.uniform(0.01, 100)
            assert np.linalg.norm(_act_quant(a)) <= act_quant_factor(n) * np.linalg.norm(a) * (1 + 1e-9)
