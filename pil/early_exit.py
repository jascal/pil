"""Certified early exit — the harness for ``docs/notes/certified_early_exit_prereg.md`` (SIGNED 2026-09-25).

After layer ``k`` the prefix residual ``x_k`` is exactly what the model computed. With an RMSNorm final norm
(no bias) and a tied unembedding, the decision is ``argmax_v ⟨x, w_v⟩`` with ``w_v = θ_f ⊙ U_v``. If ``B``
bounds the norm of everything layers ``> k`` still write, then by Cauchy–Schwarz the prefix argmax ``t_k`` is
final when

    ∀ v ≠ t_k:  ⟨x_k, w_t − w_v⟩ > B · ‖w_t − w_v‖.

So each (position, k) has a **certified radius** ``R_k = min_{v≠t} gap_v / ‖w_t − w_v‖`` over the full
vocabulary, and a bound certifies iff ``B < R_k``. Computing ``R_k`` once lets every bound (oracle /
weight-derived / calibrated) be compared against it.

Coordinates: fieldrun's ``--source-dump`` stores ``d̃_j = s · θ_f ⊙ d_j`` (``s = 1/rms(x_final)``). Logits need
no unfolding (``⟨d̃, U_v⟩`` is the logit). Norms are taken in ``y = d̃ / θ_f = s · d``, and ``s`` is recovered
from the embedding block, whose raw write is the embedding row (PIN C). numpy only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── fieldrun bundle (json index + one binary blob) ──────────────────────────────────────────────────────────


class Bundle:
    """Read-only view of a ``fieldrun-bundle`` (format 1): f16 arrays and per-output-column symmetric int8."""

    def __init__(self, stem: str | Path):
        stem = Path(stem)
        base = stem / stem.name if stem.is_dir() else stem
        self.index = json.loads(Path(f"{base}.fieldrun.json").read_text())
        if self.index.get("format") != "fieldrun-bundle":
            raise ValueError(f"{base}: not a fieldrun bundle")
        self.blob = np.memmap(f"{base}.fieldrun.bin", dtype=np.uint8, mode="r")
        self.arrays = {a["name"]: a for a in self.index["arrays"]}
        cfg = self.index["config"]
        (self.n_layer, self.n_head, self.n_kv, self.head_dim,
         self.d, self.d_ff, self.vocab, self.tied) = cfg[:8]
        self.eps = float(self.index["config_f"][1])

    def raw(self, name: str) -> np.ndarray:
        a = self.arrays[name]
        dt = {"f16": np.float16, "i8": np.int8, "rowi8": np.int8, "f32": np.float32}[a["dtype"]]
        buf = self.blob[a["offset"] : a["offset"] + a["bytes"]]
        return np.frombuffer(buf, dtype=dt).reshape(a["shape"])

    def f32(self, name: str) -> np.ndarray:
        """Dequantized float32. ``i8`` is stored ``(in, out)`` with ``W[i, j] = q[i, j] * scale[j]``;
        ``rowi8`` (embedding / unembedding) is row-major ``(vocab, d)``, ``W[r, :] = q[r, :] * scale[r]``."""
        a = self.arrays[name]
        if a["dtype"] == "i8":
            return self.raw(name).astype(np.float32) * self.raw(f"{name}__scale").astype(np.float32)[None, :]
        if a["dtype"] == "rowi8":
            return self.raw(name).astype(np.float32) * self.raw(f"{name}__scale").astype(np.float32)[:, None]
        return self.raw(name).astype(np.float32)

    def rows(self, name: str, idx) -> np.ndarray:
        """Dequantized float32 rows ``idx`` of a row-indexed ``(vocab, d)`` table, without loading it all."""
        a = self.arrays[name]
        idx = np.asarray(idx)
        q = self.raw(name)[idx].astype(np.float32)
        if a["dtype"] == "rowi8":
            return q * self.raw(f"{name}__scale").astype(np.float32)[idx][..., None]
        if a["dtype"] == "i8":
            raise ValueError(f"{name}: column-scaled i8 is not a row table")
        return q

    @property
    def readout_name(self) -> str:
        """The unembedding table: ``lm_head`` when the bundle has one (untied), else the tied ``embed``."""
        return "lm_head" if "lm_head" in self.arrays else "embed"


def act_quant_factor(n: int) -> float:
    """fieldrun's int8 path quantizes a width-``n`` activation with a bulk scale ``max_bulk/127`` (largest
    channels kept exact), so rounding adds at most ``scale/2`` per channel: ``‖ã‖ ≤ (1 + √n/254)·‖a‖``
    (prereg Addendum A)."""
    return 1.0 + math.sqrt(n) / 254.0


def _spec(w: np.ndarray) -> float:
    return float(np.linalg.norm(w.astype(np.float64), 2))


def weight_bounds(b: Bundle) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer sup-norm bounds ``(A_ℓ, M_ℓ)`` on the raw attention and MLP writes (PIN E.2 + Addendum A).

    - RMSNorm output: ``‖θ ⊙ x/rms(x)‖ ≤ max|θ|·√d``.
    - attention: each query head's output is a convex combination of its KV group's value vectors
      ``W_V^g ã + b^g``;
      ``o_proj`` (no bias) sees the concatenated heads, quantized again.
    - SwiGLU MLP (no biases): ``|silu(z)| ≤ |z|`` so ``‖silu(g) ⊙ u‖ ≤ ‖g‖_∞‖u‖``, with
      ``‖g‖_∞ ≤ max_i‖W_gate[:,i]‖·‖ã‖``.
    """
    d, c_d, c_ff = b.d, act_quant_factor(b.d), act_quant_factor(b.d_ff)
    group = b.n_head // b.n_kv
    A = np.zeros(b.n_layer)
    M = np.zeros(b.n_layer)
    for ell in range(b.n_layer):
        p = f"l{ell}."
        h_in = float(np.abs(b.f32(f"{p}in_ln")).max()) * math.sqrt(d) * c_d
        wv, bv = b.f32(f"{p}self_attn.v_proj"), b.f32(f"{p}self_attn.v_proj.bias")
        per_group = [
            _spec(wv[:, g * b.head_dim : (g + 1) * b.head_dim]) * h_in
            + float(np.linalg.norm(bv[g * b.head_dim : (g + 1) * b.head_dim]))
            for g in range(b.n_kv)
        ]
        heads = math.sqrt(group * sum(x * x for x in per_group))
        A[ell] = _spec(b.f32(f"{p}self_attn.o_proj")) * heads * c_d

        h_post = float(np.abs(b.f32(f"{p}post_ln")).max()) * math.sqrt(d) * c_d
        wg = b.f32(f"{p}mlp.gate_proj")
        g_inf = float(np.linalg.norm(wg, axis=0).max()) * h_post
        u_norm = _spec(b.f32(f"{p}mlp.up_proj")) * h_post
        M[ell] = _spec(b.f32(f"{p}mlp.down_proj")) * g_inf * u_norm * c_ff
    return A, M


def suffix_weight_bound(A: np.ndarray, M: np.ndarray) -> np.ndarray:
    """``S[k] = Σ_{ℓ>k} (A_ℓ + M_ℓ)`` for exit after layer ``k`` (raw units; ``S[n_layer-1] = 0``)."""
    per = A + M
    return np.concatenate([np.cumsum(per[::-1])[::-1][1:], [0.0]])


# ── the certificate ─────────────────────────────────────────────────────────────────────────────────────────


class Readout:
    """Full-vocabulary read-out ``W = θ_f ⊙ U`` with the helpers the pairwise certificate needs."""

    def __init__(self, U: np.ndarray, theta_f: np.ndarray):
        self.U = np.ascontiguousarray(U, dtype=np.float32)          # (V, d)
        self.theta = theta_f.astype(np.float32)
        self.W = self.U * self.theta[None, :]                        # (V, d)
        self.wn2 = np.einsum("vd,vd->v", self.W, self.W, dtype=np.float64)

    def radius(self, logits: np.ndarray) -> tuple[int, float]:
        """``(t, R)`` for one logit vector over the full vocabulary: ``t = argmax`` and the certified radius
        ``R = min_{v≠t} (L_t − L_v) / ‖w_t − w_v‖`` (``inf`` if every rival is at distance 0 with a positive
        gap, ``0`` on a tie). Logits must be in the same coordinates as ``W`` (``⟨x, w_v⟩``)."""
        t = int(np.argmax(logits))
        gap = logits[t].astype(np.float64) - logits.astype(np.float64)
        cross = (self.W @ self.W[t]).astype(np.float64)
        dist = np.sqrt(np.maximum(self.wn2[t] + self.wn2 - 2.0 * cross, 0.0))
        gap[t], dist[t] = np.inf, 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(dist > 0, gap / dist, np.where(gap > 0, np.inf, 0.0))
        return t, float(r.min())


def certify(logits: np.ndarray, readout: Readout, bound: float) -> tuple[int, bool]:
    """Pairwise early-exit certificate: ``(t, certified)`` — certified iff ``bound < R``."""
    t, r = readout.radius(logits)
    return t, bool(bound < r)


def conformal_quantile(x: np.ndarray, alpha: float) -> float:
    """Split-conformal ``(1−α)`` quantile: the ``⌈(n+1)(1−α)⌉``-th smallest value (``inf`` if that rank
    exceeds ``n``)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    rank = math.ceil((len(x) + 1) * (1.0 - alpha))
    return float(x[rank - 1]) if rank <= len(x) else math.inf


# ── dumps ───────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class ExitRecord:
    """Per-position quantities for exits ``k = 0..n_layer-1`` (``k = n_layer-1`` is the full model)."""

    sid: str
    pos: int
    pred: int                 # the model's decision (fieldrun, full vocabulary)
    s: float                  # recovered 1/rms(x_final)
    s_resid: float            # relative residual of the s fit (PIN C self-test)
    t: np.ndarray             # (n_layer,) prefix argmax
    R: np.ndarray             # (n_layer,) certified radius, y coordinates (vs suffix, s·B_w, B_cal)
    ynorm: np.ndarray         # (n_layer,) ‖y_k‖ = s·‖x_k‖
    suffix: np.ndarray        # (n_layer,) ‖y_final − y_k‖ = s·‖x_final − x_k‖ (the oracle)


def layer_prefix_index(blocks: list[str], n_layer: int) -> list[int]:
    """Index of each layer's last block (``L{k}.mlp``); the prefix after layer ``k`` is ``blocks[:idx+1]``."""
    idx = []
    for k in range(n_layer):
        name = f"L{k}.mlp"
        if name not in blocks:
            raise ValueError(f"block {name} missing from dump")
        idx.append(blocks.index(name))
    if blocks[0] != "embed":
        raise ValueError("first block must be the embedding")
    return idx


def fit_scale(d_embed: np.ndarray, emb_row: np.ndarray, theta_f: np.ndarray) -> tuple[float, float]:
    """PIN C: ``s = ⟨d̃_embed, θ⊙e⟩ / ‖θ⊙e‖²`` and the relative residual ``‖d̃_embed − s θ⊙e‖ / ‖d̃_embed‖``."""
    te = (theta_f * emb_row).astype(np.float64)
    de = d_embed.astype(np.float64)
    s = float(de @ te / (te @ te))
    return s, float(np.linalg.norm(de - s * te) / max(np.linalg.norm(de), 1e-30))


def radii_batch(P: np.ndarray, readout: Readout, exact_k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """``(t, R)`` for a batch of folded prefixes ``P`` (``C x d``; logits ``U @ p``), full vocabulary.

    Two float32 GEMMs against the unembedding give every gap and pairwise distance. The ``exact_k``
    rivals with the smallest ratio are then recomputed in float64 from the vectors themselves
    (``(U_t − U_v)·p`` and ``‖w_t − w_v‖``), so float32 rounding cannot create a certificate at a
    near-tie. A negative exact gap (the float32 argmax was not the true one) gives ``R < 0``, which
    never certifies."""
    P32 = np.ascontiguousarray(P, dtype=np.float32)
    L = readout.U @ P32.T                                              # (V, C)
    t = L.argmax(axis=0)
    uniq, inv = np.unique(t, return_inverse=True)
    cross = readout.W @ readout.W[uniq].T                             # (V, n_unique)
    R = np.empty(len(t))
    k = min(exact_k, L.shape[0] - 1)
    for c, tc in enumerate(t):
        gap = L[tc, c].astype(np.float64) - L[:, c].astype(np.float64)
        d2 = readout.wn2[tc] + readout.wn2 - 2.0 * cross[:, inv[c]].astype(np.float64)
        dist = np.sqrt(np.maximum(d2, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dist > 0, gap / dist, np.where(gap > 0, np.inf, 0.0))
        ratio[tc] = np.inf
        part = np.argpartition(ratio, k)
        cand, rest = part[:k], ratio[part[k]]
        pc = P[c].astype(np.float64)
        gap_e = (readout.U[tc].astype(np.float64) - readout.U[cand].astype(np.float64)) @ pc
        dist_e = np.linalg.norm(readout.W[tc].astype(np.float64) - readout.W[cand].astype(np.float64), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_e = np.where(dist_e > 0, gap_e / dist_e, np.where(gap_e > 0, np.inf, 0.0))
        R[c] = min(float(r_e.min()), float(rest))
    return t, R


def process_records(recs, readout: Readout, emb: np.ndarray, n_layer: int, chunk: int = 16, predict=None):
    """Turn ``--source-dump`` lines into :class:`ExitRecord` s, batching ``chunk`` positions per GEMM.

    ``predict`` (optional) maps the ``(n_layer, d)`` prefix residuals in ``y`` coordinates to predicted final
    residuals ``ŷ_k`` (e.g. the J-lens). Radii are then taken at ``ŷ_k`` and ``suffix`` holds the leftover
    ``‖y_final − ŷ_k‖`` instead of the full suffix; without it, ``ŷ_k = y_k`` (the plain prefix)."""
    buf = []
    theta = readout.theta[None, :].astype(np.float64)

    def flush():
        prefixes, meta = [], []
        for rec in buf:
            D = np.asarray(rec["d"], dtype=np.float32)                   # (nb, d) folded writes d̃_j
            last = layer_prefix_index(rec["blocks"], n_layer)
            s, s_resid = fit_scale(D[0], emb[rec["cur"]].astype(np.float32), readout.theta)
            pre = np.cumsum(D.astype(np.float64), axis=0)[last]          # (n_layer, d) folded prefixes
            y = pre / theta                                              # y = s·x coordinates
            yh = y if predict is None else predict(y)
            prefixes.append(pre if predict is None else yh * theta)
            meta.append((rec, s, s_resid, y, yh))
        t, R = radii_batch(np.concatenate(prefixes), readout)
        out = []
        for i, (rec, s, s_resid, y, yh) in enumerate(meta):
            sl = slice(i * n_layer, (i + 1) * n_layer)
            out.append(ExitRecord(
                sid=str(rec.get("sid", "")), pos=int(rec["pos"]), pred=int(rec["pred"]), s=s, s_resid=s_resid,
                t=t[sl], R=R[sl], ynorm=np.linalg.norm(y, axis=1),
                suffix=np.linalg.norm(y[-1][None, :] - yh, axis=1),
            ))
        buf.clear()
        return out

    for rec in recs:
        buf.append(rec)
        if len(buf) == chunk:
            yield from flush()
    if buf:
        yield from flush()


def process_record(rec: dict, readout: Readout, emb: np.ndarray, n_layer: int) -> ExitRecord:
    """One ``--source-dump`` line → :class:`ExitRecord` (see :func:`process_records`)."""
    return next(process_records([rec], readout, emb, n_layer))


def iter_dump(path: str | Path):
    """Stream a ``--source-dump`` JSON-lines file one record at a time (dumps are hundreds of MB)."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def jlens_predictor(J: np.ndarray, lam: float):
    """``y ↦ ŷ`` with ``ŷ_k = ((1−λ)I + λ J_k) y_k`` — the shrunk J-lens. ``J`` is ``[n_layer, d, d]``,
    applied as ``J_k @ y_k``; linear, so it acts the same in ``x`` and ``y = s·x`` coordinates."""
    J64 = J.astype(np.float64)

    def predict(y: np.ndarray) -> np.ndarray:
        return (1.0 - lam) * y + lam * np.einsum("kij,kj->ki", J64, y)

    return predict


def adverse_push(W: np.ndarray, t: int, suffix: np.ndarray) -> float:
    """Arm D: ``a = max_{v≠t} ⟨−suffix, (w_t − w_v)/‖w_t − w_v‖⟩`` — how far the suffix moves toward any
    rival along that rival's separating direction. ``t`` stays the argmax of ``W @ (y + suffix)`` whenever the
    certified radius of ``W @ y`` exceeds ``a`` (``gap_v + ⟨suffix, w_t − w_v⟩ ≥ ‖w_t − w_v‖ (R − a)``)."""
    diff = W[t][None, :] - W                                   # (K, d)
    dist = np.linalg.norm(diff, axis=1)
    push = -(diff @ suffix)
    ok = np.arange(len(W)) != t
    ok &= dist > 0
    return float(np.max(push[ok] / dist[ok])) if ok.any() else -np.inf
