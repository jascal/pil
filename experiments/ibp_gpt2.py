"""Exploratory: interval bound propagation (IBP) through GPT-2 small over a nuisance token set at one
position.

Not pre-registered. A feasibility measurement for model-side certificates (study 2 after pil #143).
Every interval op is sound (an outer bound); the box at the final position must contain every variant's
residual.
"""

from __future__ import annotations

import math

import torch

SQRT_2_PI = math.sqrt(2.0 / math.pi)


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(SQRT_2_PI * (x + 0.044715 * x.pow(3))))


_g = torch.linspace(-3, 0, 300001, dtype=torch.float64)
_gv = gelu_new(_g)
X_STAR = float(_g[_gv.argmin()])  # gelu_new is decreasing below X_STAR and increasing above
G_MIN = float(_gv.min()) - 1e-12  # small outward rounding


def gelu_ibp(lo, hi):
    glo, ghi = gelu_new(lo), gelu_new(hi)
    out_lo = torch.where(hi <= X_STAR, ghi, torch.where(lo >= X_STAR, glo, torch.full_like(lo, G_MIN)))
    out_hi = torch.maximum(glo, ghi)
    return out_lo, out_hi


def linear_ibp(lo, hi, W, b):
    """GPT-2 Conv1D: y = x @ W + b, W of shape (in, out)."""
    c, r = (lo + hi) / 2, (hi - lo) / 2
    yc, yr = c @ W + b, r @ W.abs()
    return yc - yr, yc + yr


def ln_ibp(lo, hi, g, b, eps=1e-5):
    n = lo.shape[-1]
    s_lo, s_hi = lo.sum(-1, keepdim=True), hi.sum(-1, keepdim=True)
    c_lo = (1 - 1 / n) * lo - (s_hi - hi) / n  # x_i - mean, exact interval of a linear function
    c_hi = (1 - 1 / n) * hi - (s_lo - lo) / n
    sq_lo = torch.where(c_lo > 0, c_lo**2, torch.where(c_hi < 0, c_hi**2, torch.zeros_like(c_lo)))
    sq_hi = torch.maximum(c_lo**2, c_hi**2)
    sd_lo = torch.sqrt(sq_lo.mean(-1, keepdim=True) + eps)
    sd_hi = torch.sqrt(sq_hi.mean(-1, keepdim=True) + eps)
    cands = torch.stack([c_lo / sd_lo, c_lo / sd_hi, c_hi / sd_lo, c_hi / sd_hi])
    y_lo, y_hi = cands.min(0).values, cands.max(0).values
    a, bb = y_lo * g + b, y_hi * g + b
    return torch.minimum(a, bb), torch.maximum(a, bb)


def mul_ibp(alo, ahi, blo, bhi):
    c = torch.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return c.min(0).values, c.max(0).values


def attn_ibp(lo, hi, blk, n_head=12):
    B, T, D = lo.shape
    dh = D // n_head
    qkv_lo, qkv_hi = linear_ibp(lo, hi, blk.attn.c_attn.weight, blk.attn.c_attn.bias)
    split = lambda t: t.view(B, T, 3, n_head, dh).permute(2, 0, 3, 1, 4)  # noqa: E731  (3, B, H, T, dh)
    qlo, klo, vlo = split(qkv_lo)
    qhi, khi, vhi = split(qkv_hi)
    # scores s_ij = q_i . k_j / sqrt(dh): sum over dh of interval products
    plo, phi = mul_ibp(qlo.unsqueeze(3), qhi.unsqueeze(3), klo.unsqueeze(2), khi.unsqueeze(2))  # (B,H,T,T,dh)
    s_lo, s_hi = plo.sum(-1) / math.sqrt(dh), phi.sum(-1) / math.sqrt(dh)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=lo.device))
    neg = torch.finfo(lo.dtype).min / 4
    s_lo, s_hi = s_lo.masked_fill(~mask, neg), s_hi.masked_fill(~mask, neg)
    m = s_hi.max(-1, keepdim=True).values
    e_lo, e_hi = torch.exp(s_lo - m) * mask, torch.exp(s_hi - m) * mask
    # a_j in [e_lo_j / (e_lo_j + sum_{l!=j} e_hi_l), e_hi_j / (e_hi_j + sum_{l!=j} e_lo_l)]
    a_lo = e_lo / (e_lo + (e_hi.sum(-1, keepdim=True) - e_hi)).clamp_min(1e-300)
    a_hi = e_hi / (e_hi + (e_lo.sum(-1, keepdim=True) - e_lo)).clamp_min(1e-300)
    a_hi = a_hi.clamp(max=1.0)
    olo, ohi = mul_ibp(
        a_lo.unsqueeze(-1), a_hi.unsqueeze(-1), vlo.unsqueeze(2), vhi.unsqueeze(2)
    )  # (B,H,T,T,dh)
    olo, ohi = olo.sum(3), ohi.sum(3)  # (B,H,T,dh)
    merge = lambda t: t.permute(0, 2, 1, 3).reshape(B, T, D)  # noqa: E731
    return linear_ibp(merge(olo), merge(ohi), blk.attn.c_proj.weight, blk.attn.c_proj.bias)


def mlp_ibp(lo, hi, blk):
    hlo, hhi = linear_ibp(lo, hi, blk.mlp.c_fc.weight, blk.mlp.c_fc.bias)
    glo, ghi = gelu_ibp(hlo, hhi)
    return linear_ibp(glo, ghi, blk.mlp.c_proj.weight, blk.mlp.c_proj.bias)


@torch.no_grad()
def ibp_forward(model, ids_options, track=None):
    """ids_options: (B, k, T) token ids; the k options of a class differ only at some positions.
    Returns the IBP box (lo, hi) of the decode input u = ln_f(h_L) at the last position, shape (B, D)."""
    tr = model.transformer
    pos = torch.arange(ids_options.shape[-1], device=ids_options.device)
    emb = tr.wte.weight[ids_options].double() + tr.wpe.weight[pos].double()  # (B, k, T, D)
    lo, hi = emb.min(1).values, emb.max(1).values
    for blk in tr.h:
        a_lo, a_hi = ln_ibp(lo, hi, blk.ln_1.weight.double(), blk.ln_1.bias.double())
        d_lo, d_hi = attn_ibp(a_lo, a_hi, _dbl(blk))
        lo, hi = lo + d_lo, hi + d_hi
        m_lo, m_hi = ln_ibp(lo, hi, blk.ln_2.weight.double(), blk.ln_2.bias.double())
        d_lo, d_hi = mlp_ibp(m_lo, m_hi, _dbl(blk))
        lo, hi = lo + d_lo, hi + d_hi
        if track is not None:
            track.append(float((hi - lo)[:, -1].mean()))
    return ln_ibp(lo[:, -1], hi[:, -1], tr.ln_f.weight.double(), tr.ln_f.bias.double())


_cache: dict = {}


def _dbl(blk):
    """A float64 view of a block's parameters (cached)."""
    key = id(blk)
    if key not in _cache:
        import copy

        _cache[key] = copy.deepcopy(blk).double()
    return _cache[key]
