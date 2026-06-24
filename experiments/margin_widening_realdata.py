"""Decisive test (proposal §6): does margin-widening help on REAL fieldrun residuals, where ‖r‖ is
heterogeneous (unlike the synthetic PoC)? Consumes a `fieldrun --source-dump` and (1) reports the gate —
the heterogeneity of ‖r(x)‖ and the model's own certified-bit distribution b*(x)=−log2(γ̃) — and (2) trains
a frame U on the real residuals r_x → target two ways (raw vs widen_t, + pure-NLL), reporting the binding
normalized margin nm_p10 at matched top-1. The decode logit is ⟨r_x,U_v⟩ (the per-block d̃_b sum to r_x),
so this is a linear readout on the *real* residuals — the ‖r‖-heterogeneity the synthetic substrate lacked.

Run:  python experiments/margin_widening_realdata.py <source_dump.jsonl> [label] [--seeds 3 --steps 800]
"""

from __future__ import annotations

import argparse
import math
import statistics as st

import numpy as np
import torch

from experiments.margin_widening_loop import margin_term, normalized_margin  # noqa: E402
from pil.fieldrun_io import load_source_dump
from pil.geometry import participation_ratio
from pil.learner import PILConfig, ProjectiveIncidenceLearner


def heterogeneity(sb):
    """The gate: how heterogeneous is ‖r‖, and the model's own b* distribution (deployed certified bits)."""
    rn = np.linalg.norm(sb.r, axis=1)
    cv = float(rn.std() / (rn.mean() + 1e-9))                  # coefficient of variation of ‖r‖
    gt = sb.margin / (rn + 1e-9)                               # model's own normalized margin γ̃
    gt = np.clip(gt, 1e-6, None)
    bstar = -np.log2(gt)                                       # model's own certified-bit proxy (c absorbed)
    q = lambda a, p: float(np.quantile(a, p))                 # noqa: E731
    return {"N": sb.N, "dim": sb.dim, "rnorm_cv": cv,
            "rnorm_med": float(np.median(rn)), "rnorm_p90/p10": float(q(rn, .9) / (q(rn, .1) + 1e-9)),
            "gt_p10": q(gt, .1), "gt_p50": q(gt, .5),
            "bstar_p50": q(bstar, .5), "bstar_p90": q(bstar, .9)}


def train_eval(sb, objective, target, mw, seed, steps, split=0.25):
    dev = "cpu"
    vocab = sorted(set(sb.cands.reshape(-1).tolist()) | set(sb.target.tolist()))
    idx = {t: i for i, t in enumerate(vocab)}
    r = torch.tensor(sb.r, dtype=torch.float32)                # (N, dim)
    y = torch.tensor([idx[int(t)] for t in sb.target], dtype=torch.long)
    n = r.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    nho = max(8, int(split * n))
    ho, tr = perm[:nho], perm[nho:]
    cfg = PILConfig(dim=sb.dim, n_propositions=len(vocab), n_sources_per_step=1,
                    lr=5e-3, frame_reg=0.05, margin_weight=mw, seed=seed, device=dev)
    model = ProjectiveIncidenceLearner(cfg)
    model.normalize_U()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    src_tr = r[tr].unsqueeze(1)                                # (Ntr, 1, dim) -> logits = <r, U_v>
    for _ in range(steps):
        out = model.forward(src_tr, target=y[tr])
        loss = out["nll"] + mw * margin_term(objective, target, out["logits"], src_tr, y[tr]) \
            + cfg.frame_reg * out["frame_potential"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.normalize_U()

    @torch.no_grad()
    def ev(ii):
        L = model.forward(r[ii].unsqueeze(1), target=None)["logits"]
        top1 = (L.argmax(-1) == y[ii]).float().mean().item()
        nm, _ = normalized_margin(L, r[ii].unsqueeze(1), y[ii])
        nm = nm.sort().values
        return top1, nm[int(0.10 * (len(nm) - 1))].item(), nm.median().item()
    t1, nm10, nm50 = ev(tr)
    t1h, nm10h, _ = ev(ho)
    with torch.no_grad():
        nll = model.forward(src_tr, target=y[tr])["nll"].item()
    return {"top1": t1, "top1_ho": t1h, "nm_p10": nm10, "nm_p10_ho": nm10h, "nm_p50": nm50, "nll": nll}


def train_frame(sb, objective, target, mw, seed, steps):
    """Train a frame U on ALL real residuals (no split) — for the interpretability PR analysis."""
    vocab = sorted(set(sb.cands.reshape(-1).tolist()) | set(sb.target.tolist()))
    idx = {t: i for i, t in enumerate(vocab)}
    r = torch.tensor(sb.r, dtype=torch.float32)
    y = torch.tensor([idx[int(t)] for t in sb.target], dtype=torch.long)
    cfg = PILConfig(dim=sb.dim, n_propositions=len(vocab), n_sources_per_step=1,
                    lr=5e-3, frame_reg=0.05, margin_weight=mw, seed=seed, device="cpu")
    model = ProjectiveIncidenceLearner(cfg)
    model.normalize_U()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    src = r.unsqueeze(1)
    for _ in range(steps):
        out = model.forward(src, target=y)
        loss = out["nll"] + mw * margin_term(objective, target, out["logits"], src, y) \
            + cfg.frame_reg * out["frame_potential"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.normalize_U()
    return model, y


def interp_regimes(sb, label, knee_lam, steps, seeds):
    """retrieve / select / compose under margin-widening: does widening the SELECT margin concentrate the
    SOURCE support (block-PR: low=retrieve, high=compose)? Compares no-margin (λ=0) vs knee (λ=knee_lam)."""
    tgt = 0.10 * float(np.median(np.linalg.norm(sb.r, axis=1)))
    D = torch.tensor(sb.D, dtype=torch.float32)                  # (N, nb, dim)
    src = torch.tensor(sb.r, dtype=torch.float32).unsqueeze(1)   # (N, 1, dim)
    print(f"[interp:{label}] retrieve/select/compose under margin-widening (nb={sb.nb} blocks):")
    print(f"  {'regime':>12}{'select_nm50':>13}{'blockPR_p50':>13}{'blockPR_p10':>13}{'retr_frac':>11}")
    for name, lam in (("none λ=0", 0.0), (f"knee λ={knee_lam}", knee_lam)):
        prs, sels = [], []
        for s in range(seeds):
            model, y = train_frame(sb, "raw", tgt, lam, s, steps)
            with torch.no_grad():
                U = model.U[y]                                   # (N, dim) target frame rows
                cb = torch.einsum("nbd,nd->nb", D, U)           # block contributions to the target
                pr = participation_ratio(cb.abs(), dim=-1)       # (N,) low=retrieve, high=compose
                nm, _ = normalized_margin(model.forward(src, target=None)["logits"], src, y)
            prs.append(pr)
            sels.append(nm.median().item())
        pr = torch.cat(prs)
        # retrieve = concentrated support: PR below a quarter of the blocks
        rfrac = (pr <= 0.25 * sb.nb).float().mean().item()
        print(f"  {name:>10}{st.mean(sels):>15.3f}{pr.median().item():>13.2f}"
              f"{pr.sort().values[int(0.1*len(pr))].item():>13.2f}{rfrac:>14.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("label", nargs="?", default="model")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--gamma", type=float, default=0.15)        # widen_t knee target
    ap.add_argument("--mw", type=float, default=0.5)            # margin_weight
    ap.add_argument("--raw-target", type=float, default=2.0)
    ap.add_argument("--gate-only", action="store_true", help="skip training; report the gate + headroom only")
    ap.add_argument("--knee", action="store_true",
                    help="sweep margin_weight (raw hinge) -> the free-lunch knee on real residuals")
    ap.add_argument("--knee-frac", type=float, default=0.10)
    ap.add_argument("--lam-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--interp", action="store_true",
                    help="retrieve/select/compose: does widening the margin concentrate source block-PR?")
    ap.add_argument("--knee-lam", type=float, default=0.5)
    ap.add_argument("--match-target", action="store_true",
                    help="raw_target = gamma*median||r|| so raw vs widen_t match pressure (kills artifact)")
    a = ap.parse_args()
    sb = load_source_dump(a.dump)
    h = heterogeneity(sb)
    print(f"[real:{a.label}] N={h['N']} dim={h['dim']} | GATE ‖r‖ cv={h['rnorm_cv']:.3f} "
          f"p90/p10={h['rnorm_p90/p10']:.2f} | model-own γ̃ p10={h['gt_p10']:.3f} p50={h['gt_p50']:.3f} "
          f"| own b* p50={h['bstar_p50']:.1f} p90={h['bstar_p90']:.1f}")
    if a.gate_only:
        rn = np.linalg.norm(sb.r, axis=1)
        gate = "OPEN" if h["rnorm_cv"] > 0.35 else "CLOSED"
        print(f"SWEEP\t{a.label}\t{h['dim']}\t{h['N']}\t{h['rnorm_cv']:.4f}\t{h['bstar_p50']:.2f}\t"
              f"{h['bstar_p90']:.2f}\t{float(np.median(rn)):.1f}\tgate={gate}")
        return
    if a.interp:
        interp_regimes(sb, a.label, a.knee_lam, a.steps, a.seeds)
        return
    if a.knee:
        tgt = a.knee_frac * float(np.median(np.linalg.norm(sb.r, axis=1)))
        print(f"[real:{a.label}] KNEE sweep (raw hinge, target={a.knee_frac}*median‖r‖={tgt:.1f}): "
              f"does raw margin-widening free-lunch on real residuals?")
        print(f"  {'lambda':>7}{'nll':>8}{'top1':>7}{'nm_p10':>8}{'nm_p50':>8}")
        for lam in a.lam_grid:
            es = [train_eval(sb, "raw", tgt, lam, s, a.steps) for s in range(a.seeds)]
            ag = {k: st.mean(e[k] for e in es) for k in es[0]}
            print(f"  {lam:>7.2f}{ag['nll']:>8.3f}{ag['top1']:>7.3f}{ag['nm_p10']:>8.3f}{ag['nm_p50']:>8.3f}")
        return
    if a.match_target:
        a.raw_target = a.gamma * float(np.median(np.linalg.norm(sb.r, axis=1)))
        print(f"[real:{a.label}] matched raw_target = gamma*median‖r‖ = {a.raw_target:.2f}")

    def agg(obj, tgt):
        es = [train_eval(sb, obj, tgt, a.mw, s, a.steps) for s in range(a.seeds)]
        return {k: st.mean(e[k] for e in es) for k in es[0]}
    raw = agg("raw", a.raw_target)
    wt = agg("widen_t", a.gamma)
    ok = raw["nm_p10"] > 0 and wt["nm_p10"] > 0
    dbits = math.log2(wt["nm_p10"] / raw["nm_p10"]) if ok else float("nan")
    print(f"[real:{a.label}] raw    : top1={raw['top1']:.3f}(ho {raw['top1_ho']:.3f}) "
          f"nm_p10={raw['nm_p10']:.3f}(ho {raw['nm_p10_ho']:.3f})")
    print(f"[real:{a.label}] widen_t: top1={wt['top1']:.3f}(ho {wt['top1_ho']:.3f}) "
          f"nm_p10={wt['nm_p10']:.3f}(ho {wt['nm_p10_ho']:.3f})")
    print(f"[real:{a.label}] --> Δbits (widen_t vs raw, c-free) = {dbits:+.2f}  "
          f"[gate cv={h['rnorm_cv']:.2f}: {'OPEN' if h['rnorm_cv'] > 0.35 else 'closed'}]")
    # machine-readable summary line for the sweep aggregator
    print(f"SWEEP\t{a.label}\t{h['dim']}\t{h['N']}\t{h['rnorm_cv']:.3f}\t{h['bstar_p90']:.2f}\t"
          f"{raw['nm_p10']:.4f}\t{wt['nm_p10']:.4f}\t{dbits:+.3f}\t{raw['top1']:.3f}\t{wt['top1']:.3f}")


if __name__ == "__main__":
    main()
