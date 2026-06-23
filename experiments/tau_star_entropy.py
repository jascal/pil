"""τ* test: does the decode's effective block count track output entropy, not parameter count?

The decode-circuit sweep found effective block usage pinned near ~12 while total blocks drift
49 -> 73 across 0.5B -> 3B (6x params). The hypothesis (the "τ* law"): the binding constraint is
the *output distribution's* effective support exp(H), not capacity -- so r_eff ~ min(exp(H), nb).
exp(H) ~ 5-20 << nb, so r_eff saturates at exp(H), independent of scale.

This is FALSIFIABLE and not obviously true: a weak model (pythia-70m) has HIGH output entropy yet few
effective blocks (nb is tiny there too). We measure rather than assert. Two tests:

  CROSS-MODEL  : does model-level mean r_eff correlate with exp(mean H) better than with params/hidden/nb?
  WITHIN-MODEL : per position, does r_eff rise with exp(H)? (hundreds of points/model -> the solid test)

Per position from a `fieldrun --pil-dump` (now carrying full-vocab `ent`):
  expH   = exp(entropy)                          effective output support (τ*)
  r_eff  = participation_ratio(|c_decode|)       effective # blocks driving the decode (the "~12")
  mblk   = min greedy blocks to keep argmax       compactness / head-tail

Run:  python experiments/tau_star_entropy.py qwen-0.5B=d05.jsonl pythia-70m=p70.jsonl ...
      (label must contain a known model key for the params/hidden/layers covariates.)
"""

from __future__ import annotations

import math
import sys

import torch

from pil.fieldrun_io import load_pil_dump
from pil.geometry import participation_ratio

# params (M), hidden, layers -- the regression covariates. nb is read from the dump itself.
# Label your dump `key=path.jsonl`; `key` must contain one of these (substring match, model_key()).
MODELS = {
    # Qwen2.5 (rope) -- the wide scale axis to 72B
    "qwen-0.5B": (494, 896, 24),
    "qwen-1.5B": (1540, 1536, 28),
    "qwen-3B": (3090, 2048, 36),
    "qwen-7B": (7620, 3584, 28),
    "qwen-14B": (14770, 5120, 48),
    "qwen-32B": (32760, 5120, 64),
    "qwen-72B": (72710, 8192, 80),
    # Llama-3.x (rope) -- a second architecture to 70B (decouples depth/width from Qwen)
    "llama-1B": (1240, 2048, 16),
    "llama-3B": (3210, 3072, 28),
    "llama-8B": (8030, 4096, 32),
    "llama-70B": (70550, 8192, 80),
    # Pythia / GPT-NeoX (neox) -- the clean controlled ladder (same data/tokenizer, only scale)
    "pythia-14m": (14, 128, 6),
    "pythia-31m": (31, 256, 6),
    "pythia-70m": (70, 512, 6),
    "pythia-160m": (160, 768, 12),
    "pythia-410m": (410, 1024, 24),
    "pythia-1b": (1010, 2048, 16),
    "pythia-1.4b": (1410, 2048, 24),
    "pythia-2.8b": (2780, 2560, 32),
    "pythia-6.9b": (6900, 4096, 32),
    "pythia-12b": (11600, 5120, 36),
}


def model_key(label: str) -> str | None:
    for k in MODELS:
        if k in label:
            return k
    return None


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    d = x.norm() * y.norm()
    return float((x @ y) / d) if d > 0 else float("nan")


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    def rank(t):
        order = t.argsort()
        r = torch.empty_like(order, dtype=torch.float64)
        r[order] = torch.arange(len(t), dtype=torch.float64)
        return r
    return pearson(rank(x).float(), rank(y).float())


def min_blocks_to_argmax(contrib: torch.Tensor) -> torch.Tensor:
    """Per position: min # of greedily-added blocks (sorted by net contribution to decode-vs-runnerup)
    whose partial sum keeps the decode (cand 0) as the argmax over all candidates. The head-tail measure."""
    N, nb, K = contrib.shape
    out = torch.empty(N, dtype=torch.float64)
    for i in range(N):
        c = contrib[i]                       # (nb, K); col 0 = decode
        key = c[:, 0] - c[:, 1]              # contribution toward decode over the runner-up
        order = key.argsort(descending=True)
        acc = torch.zeros(K)
        m = nb
        for j, b in enumerate(order, start=1):
            acc += c[b]
            if int(acc.argmax()) == 0:
                m = j
                break
        out[i] = m
    return out


def analyse(label: str, path: str) -> dict | None:
    b = load_pil_dump(path)
    contrib = b.contrib                       # (N, nb, K); col 0 = decode (argmax), col 1 = runner-up
    N, nb, K = contrib.shape
    recon = (contrib.sum(dim=1).argmax(1) == 0).float().mean().item()
    if torch.isnan(b.ent).any():
        print(f"  skip {label}: dump has no `ent` field (reconvert with the τ*-entropy fieldrun)")
        return None
    expH = b.ent.double().exp()               # (N,)  effective output support per position
    # discriminative per-block contribution: centre each block across candidates so the common-mode
    # (e.g. neox LayerNorm/embedding offset, identical for every candidate, zero effect on the argmax)
    # drops out. PR over |centred decode contribution| = architecture-fair effective decode-block count.
    cc = contrib - contrib.mean(dim=2, keepdim=True)
    r_eff = participation_ratio(cc[:, :, 0], dim=1).double()        # (N,) effective discriminative blocks
    r_raw = participation_ratio(contrib[:, :, 0], dim=1).double()   # (N,) raw PR(c_decode) (continuity)
    mblk = min_blocks_to_argmax(contrib)
    key = model_key(label)
    params, hidden, layers = MODELS.get(key, (float("nan"),) * 3)
    return {
        "label": label, "N": N, "nb": nb, "recon": recon,
        "params": params, "hidden": hidden, "layers": layers,
        "H_mean": float(b.ent.double().mean()),
        "expH_geom": math.exp(float(b.ent.double().mean())),   # exp(mean H)
        "expH_mean": float(expH.mean()),                       # mean exp(H)
        "r_eff": float(r_eff.mean()), "r_eff_med": float(r_eff.median()),
        "r_raw": float(r_raw.mean()),
        "mblk_med": float(mblk.median()), "mblk_mean": float(mblk.mean()),
        # within-model per-position coupling of effective blocks to output entropy
        "wp_reff_expH": pearson(r_eff, expH), "ws_reff_expH": spearman(r_eff, expH),
        "wp_mblk_expH": pearson(mblk, expH), "ws_mblk_expH": spearman(mblk, expH),
        # entropy-quartile binned mean r_eff (monotone rise = within-model τ*)
        "_expH": expH, "_reff": r_eff,
    }


def quartile_bins(expH: torch.Tensor, reff: torch.Tensor) -> list[float]:
    q = torch.quantile(expH, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
    out = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (expH >= lo) & (expH <= hi)
        out.append(float(reff[m].mean()) if m.any() else float("nan"))
    return out


def main():
    items = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
    rows = [r for r in (analyse(lbl, p) for lbl, p in items) if r is not None]
    if not rows:
        print("usage: tau_star_entropy.py qwen-0.5B=d.jsonl pythia-70m=p.jsonl ...")
        return
    rows.sort(key=lambda r: r["params"])

    print("=" * 104)
    print("PER-MODEL  (r_eff = effective decode blocks; expH = exp(H) effective output support)")
    print("=" * 104)
    hdr = (f"{'model':<14}{'N':>5}{'nb':>4}{'recon':>6}{'params':>8}{'hid':>6}{'L':>4}"
           f"{'H̄':>7}{'e^H̄':>7}{'mean e^H':>9}{'r_eff':>7}{'r_raw':>7}{'r_eff/e^H̄':>10}{'mblk_med':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ratio = r["r_eff"] / r["expH_geom"] if r["expH_geom"] > 0 else float("nan")
        print(f"{r['label']:<14}{r['N']:>5}{r['nb']:>4}{r['recon']:>6.2f}{r['params']:>8.0f}"
              f"{r['hidden']:>6}{r['layers']:>4}{r['H_mean']:>7.2f}{r['expH_geom']:>7.1f}"
              f"{r['expH_mean']:>9.1f}{r['r_eff']:>7.1f}{r['r_raw']:>7.1f}{ratio:>10.2f}{r['mblk_med']:>9.1f}")

    # ---- cross-model: what does model-level r_eff track? ----
    if len(rows) >= 3:
        reff = torch.tensor([r["r_eff"] for r in rows], dtype=torch.float64)
        cov = {
            "exp(H̄)": torch.tensor([r["expH_geom"] for r in rows], dtype=torch.float64),
            "mean e^H": torch.tensor([r["expH_mean"] for r in rows], dtype=torch.float64),
            "log10 params": torch.tensor([math.log10(r["params"]) for r in rows], dtype=torch.float64),
            "hidden": torch.tensor([r["hidden"] for r in rows], dtype=torch.float64),
            "nb": torch.tensor([r["nb"] for r in rows], dtype=torch.float64),
            "layers": torch.tensor([r["layers"] for r in rows], dtype=torch.float64),
        }
        print("\n" + "=" * 70)
        print(f"CROSS-MODEL  corr(model-level r_eff, X)   [{len(rows)} models]")
        print("=" * 70)
        print(f"{'X':<16}{'Pearson':>10}{'Spearman':>10}")
        print("-" * 36)
        for name, v in cov.items():
            print(f"{name:<16}{pearson(reff, v):>10.3f}{spearman(reff, v):>10.3f}")
        print("\n  τ* law predicts r_eff ~ exp(H): Pearson/Spearman with exp(H̄) should DOMINATE")
        print("  params/hidden/nb. If r_eff tracks params/nb instead, it's capacity- not entropy-bound.")

    # ---- within-model: does r_eff rise with exp(H) per position? (the solid test) ----
    print("\n" + "=" * 92)
    print("WITHIN-MODEL  per-position coupling  +  entropy-quartile binned r_eff (Q1 low H -> Q4 high H)")
    print("=" * 92)
    hdr2 = (f"{'model':<14}{'r_eff~e^H (P/S)':>18}{'mblk~e^H (P/S)':>18}"
            f"{'Q1':>7}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'monotone?':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in rows:
        bins = quartile_bins(r["_expH"], r["_reff"])
        mono = all(bins[i] <= bins[i + 1] + 1e-6 for i in range(3))
        print(f"{r['label']:<14}{r['wp_reff_expH']:>8.2f}/{r['ws_reff_expH']:<9.2f}"
              f"{r['wp_mblk_expH']:>8.2f}/{r['ws_mblk_expH']:<9.2f}"
              f"{bins[0]:>7.1f}{bins[1]:>7.1f}{bins[2]:>7.1f}{bins[3]:>7.1f}{'yes' if mono else 'no':>11}")


if __name__ == "__main__":
    main()
