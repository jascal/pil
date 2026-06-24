"""Margin-widening loop: does the decode-margin regime control how certifiably quantizable a PIL
frame is -- and what does each regime cost in fidelity, capacity, and *speed of learning*?

Downstream (fieldrun certified-quant, CERTIFIED_QUANT_PROPOSAL §10 / v1.5) the decode survives
quantization at bit-width b iff the margin certificate holds:  2*delta(x) < margin(x), with the
read-out distortion  delta(x) ~ c * 2^-b / sqrt(d) * ||r(x)||  (TurboQuant, measured tight in v1.5).
So the *minimum certified bits* at position x is

    b*(x) = log2( 2c * ||r(x)|| / (sqrt(d) * margin(x)) )   ( = const - log2(gamma_tilde(x)) )

driven entirely by the SCALE-FREE normalized margin  gamma_tilde(x) = margin(x) / ||r(x)||.
PIL's current hinge targets the RAW margin (absolute `margin_target`), gameable with ||r|| free.
This experiment sweeps THREE regimes on a fixed planted-frame problem:

  raw     : relu(target - margin)              the current hinge (absolute target)
  widen   : relu(target - margin/||r||)        proposed -- push the scale-free margin UP (more certified bits)
  narrow  : relu(margin/||r|| - target)        the OPPOSITE -- squeeze margins DOWN (fewer bits)

and reports, per regime: top-1 fidelity; the normalized-margin percentiles (binding->typical); the
realized gamma-code capacity vs the DecodeCapacity.thy packing ceiling; and t90 = steps to reach 90%
of that run's final top-1 (speed of descent). The c-free payoff of widen vs raw is
log2(gamma_tilde_widen / gamma_tilde_raw) at the binding percentile = certified bits saved.

Hypotheses: widen raises binding gamma_tilde (saves bits) but is a harder constraint (slower t90,
smaller realized code -> capacity price per the packing bound); narrow fits fast and packs more but
is quant-brittle (binding gamma_tilde -> 0, certificate fails).

Run:  python experiments/margin_widening_loop.py --steps 600 --seeds 3
"""

from __future__ import annotations

import argparse
import math
import statistics as st

import torch

from pil.geometry import (
    gamma_decodable_count,
    log10_packing_bound,
    margin_to_worst,
    min_frame_separation,
)
from pil.learner import PILConfig, ProjectiveIncidenceLearner
from pil.synthetic import create_clustered_problem


def _progress(it, desc):
    try:
        from tqdm import tqdm

        return tqdm(it, desc=desc, leave=False)
    except ImportError:
        return it


def normalized_margin(L, sources, targets):
    """gamma_tilde(x) = margin(x) / ||r(x)||, r_x = sum_j d_j (the scale-free certificate quantity).

    Note: r depends on the DATA only (sum of sources), not on U -- so raw and normalized margins
    differ by the per-position factor ||r(x)||, i.e. normalized = a ||r||-weighted margin target."""
    r = sources.sum(dim=-2)                       # (B, dim)
    rnorm = r.norm(dim=-1).clamp_min(1e-6)        # (B,)
    return margin_to_worst(L, targets) / rnorm, rnorm


def margin_term(objective, target, L, s, t):
    if objective == "raw":
        return torch.relu(target - margin_to_worst(L, t)).mean()
    nm, rnorm = normalized_margin(L, s, t)
    if objective == "widen":
        return torch.relu(target - nm).mean()                  # naive: 1/||r|| grad down-weights binding!
    if objective == "widen_t":
        m = margin_to_worst(L, t)
        return torch.relu(target * rnorm.detach() - m).mean()  # principled: target gamma*||r||, unscaled grad
    if objective == "narrow":
        return torch.relu(nm - target).mean()                       # opposite: squeeze normalized margin DOWN
    raise ValueError(objective)


@torch.no_grad()
def _snapshot(model, src, tgt):
    L = model.forward(src, target=None)["logits"]
    top1 = (L.argmax(-1) == tgt).float().mean().item()
    nm, _ = normalized_margin(L, src, tgt)
    return top1, nm.median().item()


def train_once(objective, target, cfg, src, tgt, steps, batch, margin_weight, seed, log_every=25):
    model = ProjectiveIncidenceLearner(cfg)
    model.normalize_U()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    traj = []                                      # (step, top1, nm_p50) -- speed of descent
    for step in _progress(range(steps), f"{objective}={target}"):
        idx = torch.randint(0, len(tgt), (batch,), generator=g).to(cfg.device)
        s, t = src[idx], tgt[idx]
        out = model.forward(s, target=t)
        loss = out["nll"] + margin_weight * margin_term(objective, target, out["logits"], s, t) \
            + cfg.frame_reg * out["frame_potential"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.normalize_U()
        if step % log_every == 0 or step == steps - 1:
            traj.append((step, *_snapshot(model, src, tgt)))
    # t90 = first step reaching 90% of final top1 (lower = faster descent)
    final = traj[-1][1]
    t90 = next((s for s, a, _ in traj if a >= 0.9 * final), traj[-1][0]) if final > 0 else steps
    return model, t90, final, traj


@torch.no_grad()
def evaluate(model, src, tgt, gamma_cap, t90, final_top1):
    out = model.forward(src, target=tgt)
    L = out["logits"]
    nm, _ = normalized_margin(L, src, tgt)
    nm_sorted = nm.sort().values
    p = lambda q: nm_sorted[int(q * (len(nm_sorted) - 1))].item()   # noqa: E731
    return {
        "top1": final_top1,
        "nll": out["nll"].item(),        # final loss (the fit / learning goal)
        "t90": float(t90),               # learning rate / speed of descent
        "nm_p0": p(0.0), "nm_p10": p(0.10), "nm_p50": p(0.50),
        "code": float(gamma_decodable_count(L, gamma_cap)),
        "sep": min_frame_separation(model.U, L.argmax(-1).unique()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=24)
    ap.add_argument("--V", type=int, default=192)            # 8x over-complete (capacity-stressed)
    ap.add_argument("--J", type=int, default=24)
    ap.add_argument("--clusters", type=int, default=24)      # ~8 synonyms/cluster (contested margins)
    ap.add_argument("--spread", type=float, default=0.10)    # within-cluster coherence
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-examples", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--margin-weight", type=float, default=0.5)
    ap.add_argument("--raw-grid", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--widen-grid", type=float, nargs="+", default=[0.15, 0.30, 0.50, 0.80])
    ap.add_argument("--narrow-grid", type=float, nargs="+", default=[0.20, 0.10, 0.05, 0.02])
    ap.add_argument("--gamma-cap", type=float, default=0.20)  # capacity probe threshold (normalized)
    ap.add_argument("--mode", choices=["regimes", "weights", "sizes", "both"], default="regimes")
    ap.add_argument("--weight-grid", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--size-grid", type=str, nargs="+",
                    default=["16x8", "24x4", "24x8", "24x16", "24x32", "48x8"],  # dim x overcompleteness
                    help="dimDIMxMULT pairs; V=dim*mult (over-completeness sweep -> capacity price)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    trajs = {}     # regime -> seed-0 trajectory at its representative (first) target

    def fit(objective, tv, mw, seed, dim=None, V=None, J=None, clusters=None):
        dim, V, J = dim or a.dim, V or a.V, J or a.J
        clusters = clusters or (a.clusters if V == a.V else max(2, V // 8))
        cfg = PILConfig(dim=dim, n_propositions=V, n_sources_per_step=J,
                        lr=5e-3, frame_reg=0.05, margin_weight=mw, seed=seed, device=dev)
        src, tgt, _, _ = create_clustered_problem(cfg, n_examples=a.n_examples,
                                                  n_clusters=clusters, cluster_spread=a.spread,
                                                  sparsity=0.12, noise=0.08)
        model, t90, final, traj = train_once(objective, tv, cfg, src, tgt, a.steps, a.batch, mw, seed)
        return evaluate(model, src, tgt, a.gamma_cap, t90, final), traj

    def run(objective, grid, mw):
        rows = {}
        for tv in grid:
            evs = []
            for seed in range(a.seeds):
                ev, traj = fit(objective, tv, mw, seed)
                if seed == 0 and tv == grid[0]:
                    trajs[objective] = traj
                evs.append(ev)
            rows[tv] = {k: st.mean(e[k] for e in evs) for k in evs[0]}
        return rows

    ceil = log10_packing_bound(0.3, a.gamma_cap, a.dim)
    print(f"[mwl] dim={a.dim} V={a.V} ({a.V/a.dim:.0f}x) J={a.J} steps={a.steps} seeds={a.seeds} "
          f"mode={a.mode}  packing ceiling log10 N_gamma(rho=0.3,gamma={a.gamma_cap})={ceil:.1f} "
          f"(V log10 {math.log10(a.V):.1f})")

    if a.mode == "weights":
        weight_sweep(a, fit, ceil)
        return
    if a.mode == "sizes":
        size_sweep(a, fit)
        return

    cols = ["top1", "nll", "t90", "nm_p0", "nm_p10", "nm_p50", "code", "sep"]
    hdr = "target".rjust(8) + "".join(c.rjust(9) for c in cols)

    grids = {"raw": a.raw_grid, "widen": a.widen_grid, "widen_t": a.widen_grid, "narrow": a.narrow_grid}
    res = {o: run(o, g, a.margin_weight) for o, g in grids.items()}
    for o in ("raw", "widen", "widen_t", "narrow"):
        kind = {"raw": "absolute margin", "widen": "normalized UP (naive /||r||)",
                "widen_t": "normalized UP (target gamma*||r||, detached)",
                "narrow": "normalized margin DOWN"}[o]
        print(f"\n--- objective={o} (hinge: {kind}) ---\n" + hdr)
        for k in grids[o]:
            print(f"{k:>8.2f}" + "".join(f"{res[o][k][c]:>9.3f}" for c in cols))

    # c-free payoff: at matched top1, does widen raise the BINDING (p10) normalized margin vs raw?
    def best_at_top1(rows, floor):
        ok = [v for v in rows.values() if v["top1"] >= floor]
        return max(ok, key=lambda v: v["nm_p10"]) if ok else None
    floor = 0.95 * max(v["top1"] for v in res["raw"].values())
    best = {o: best_at_top1(res[o], floor) for o in ("raw", "widen", "widen_t", "narrow")}
    br = best["raw"]
    print(f"\n[mwl] at top1 >= {floor:.3f} (95% of best raw top1) -- best binding nm_p10 per regime:")
    if br:
        for o in ("raw", "widen", "widen_t", "narrow"):
            b = best[o]
            if not b:
                print(f"  {o:<8}: (no config cleared the top1 floor)")
                continue
            ok = br["nm_p10"] > 0 and b["nm_p10"] > 0
            dbits = math.log2(b["nm_p10"] / br["nm_p10"]) if ok else float("nan")
            tag = "" if o == "raw" else f"  bits vs raw={dbits:+.2f} (c-free)"
            print(f"  {o:<8}: nm_p10={b['nm_p10']:.3f} top1={b['top1']:.3f} t90={b['t90']:.0f} "
                  f"code={b['code']:.0f}{tag}")
        print(f"  ceiling: packing log10 N_gamma={ceil:.1f} >> V={a.V} (log10 {math.log10(a.V):.1f}) "
              f"-> capacity does NOT bind at this scale; 'code' tracks realized margin, not a capacity price")
    else:
        print("  (no configuration cleared the top1 floor; widen the grid)")

    # speed of descent: top1 (and nm_p50) vs step, seed 0, representative target per regime
    steps_logged = [s for s, _, _ in trajs["raw"]]
    show = [s for s in steps_logged if s in (steps_logged[0],) or s % 100 == 0 or s == steps_logged[-1]]
    print(f"\n[mwl] speed of descent (seed 0; top1 | nm_p50 vs step; reps: "
          f"raw={a.raw_grid[0]} widen_t={a.widen_grid[0]} narrow={a.narrow_grid[0]}):")
    print("    step" + "".join(f"{o:>16}" for o in ("raw", "widen_t", "narrow")))
    tmap = {o: {s: (a1, n) for s, a1, n in trajs[o]} for o in ("raw", "widen_t", "narrow")}
    for s in show:
        cells = "".join(f"   {tmap[o][s][0]:.2f}|{tmap[o][s][1]:+.2f}" for o in ("raw", "widen_t", "narrow"))
        print(f"  {s:>6}{cells}")

    if a.mode == "both":
        weight_sweep(a, fit, ceil)


def weight_sweep(a, fit, ceil):
    """2D phase diagram of the margin-vs-fit weighting: margin_weight (lambda on the margin term) x
    target, for the principled widen_t objective. Reports the FIT goal (final nll), LEARNING RATE
    (t90 = steps to 90% of final top1), top1, and the certified-margin payoff (binding nm_p10).

    margin_weight=0 is the pure-NLL baseline (no margin pressure). 'A small but meaningful margin' is
    a modest target at the lambda where nll is still near its floor but nm_p10 has lifted -- the knee."""
    import statistics as st
    obj = "widen_t"
    tgts = a.widen_grid
    print(f"\n[mwl] WEIGHT SWEEP (objective={obj}): margin_weight x target -> "
          f"fit (nll) | learning rate (t90) | top1 | binding nm_p10   (seeds={a.seeds})")
    cols = ["nll", "t90", "top1", "nm_p10", "nm_p50", "code"]
    print("  m_weight  target" + "".join(c.rjust(9) for c in cols))
    base_nll = None
    for mw in a.weight_grid:
        row_tgts = tgts if mw > 0 else tgts[:1]   # at lambda=0 the margin term is inert -> one row
        for tv in row_tgts:
            evs = [fit(obj, tv, mw, s)[0] for s in range(a.seeds)]
            agg = {k: st.mean(e[k] for e in evs) for k in evs[0]}
            if mw == 0.0:
                base_nll = agg["nll"]
            print(f"  {mw:>8.2f}{tv:>8.2f}" + "".join(f"{agg[c]:>9.3f}" for c in cols))
    floor = f"{base_nll:.3f}" if base_nll is not None else "n/a"
    print(f"  note: margin_weight=0 is pure-NLL (nll floor={floor}); raise lambda until nm_p10 lifts "
          f"while nll stays near floor = the 'small but meaningful margin' knee.")


def size_sweep(a, fit):
    """Scale sweep over (dim, over-completeness V/dim): is the margin-widening knee scale-stable, and does
    the PROVABLE capacity price (packing bound) activate as the frame gets crowded? Compares raw (target
    2.0) vs widen_t at the knee (gamma=0.15) at each size. Watch dtop1 (widening's fidelity price) and
    code/V (gamma-code saturation) GROW with V/dim if capacity starts to bind. Architecture cannot be
    swept synthetically -- real arch dependence (cf. the spec's tau* Pythia-vs-Qwen result) is v1.5/§6."""
    import statistics as st
    print(f"\n[mwl] SIZE SWEEP: raw(t=2.0) vs widen_t(knee gamma=0.15), margin_weight={a.margin_weight}, "
          f"seeds={a.seeds}, steps={a.steps}")
    print("   size      V  ceil_log10  Vlog10 | raw:top1 nm10 code | wt:top1 nm10 code |dtop1  dbits code/V")
    for spec in a.size_grid:
        dim, mult = (int(x) for x in spec.lower().split("x"))
        V = dim * mult
        ceil = log10_packing_bound(0.3, a.gamma_cap, dim)

        def agg(obj, tv, dim=dim, V=V):
            evs = [fit(obj, tv, a.margin_weight, s, dim=dim, V=V)[0] for s in range(a.seeds)]
            return {k: st.mean(e[k] for e in evs) for k in evs[0]}
        r, w = agg("raw", 2.0), agg("widen_t", 0.15)
        ok = r["nm_p10"] > 0 and w["nm_p10"] > 0
        dbits = math.log2(w["nm_p10"] / r["nm_p10"]) if ok else float("nan")
        dtop1 = w["top1"] - r["top1"]
        print(f"  {spec:>5}{V:>7}{ceil:>11.1f}{math.log10(V):>8.1f} |"
              f" {r['top1']:>6.3f}{r['nm_p10']:>5.2f}{r['code']:>5.0f} |"
              f" {w['top1']:>6.3f}{w['nm_p10']:>5.2f}{w['code']:>5.0f} |"
              f"{dtop1:>+6.3f}{dbits:>+6.2f}{w['code'] / V:>7.2f}")
    print("  read: capacity binds where code/V -> 1 AND dtop1 turns negative (widening can no longer "
          "separate all V tokens); the packing ceiling (ceil_log10 >> Vlog10) says how far that is.")


if __name__ == "__main__":
    main()
