"""Why the uniform substitution certificate fails: hull ceiling vs alignment.

Pre-registered: docs/notes/hull_ceiling_prereg.md.

    # 1. dump (needs `transformers`):
    /path/to/venv/bin/python experiments/hull_ceiling.py dump --out runs/hull_ceiling
    # 2. run (needs souffle):
    /path/to/venv/bin/python experiments/hull_ceiling.py run --dumps runs/hull_ceiling \
        --out runs/hull_ceiling/summary.json

Theorem (i-orca#29, PIC_Binding.certificate_hull_ceiling, kernel-checked): for a bias-free decode, a
certificate that needs every margin > m can fire only if m < |u| * h(t), where
h(t) = infdist(U_t, conv{U_v : v != t}).
Addendum A (prereg): GPT-2's decode input is dominated by a context-constant component c, which makes the
bias-free ceiling vacuous. The primary ceiling treats c as a per-token bias b_v = <c, U_v> and lifts it
(certificate_hull_ceiling_biased): m < |(u - c, s)| * h_s(t), with h_s the hull distance of (U_t, b_t/s),
for every s in S_GRID. A uniform-refused context is CEILING (2 delta >= the lifted ceiling's upper bracket for
some s: certain), NOT-EXCLUDED (2 delta below its lower bracket for every s), or UNDECIDED.
The original bias-free classification is kept as a secondary.

Tasks, families and training are certified_substitutes.py's (#136), unchanged; only the seeds differ.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_substitutes as cs  # noqa: E402
from substitution_certificate import certificate_facts, certify_all_souffle  # noqa: E402

SEEDS = dict(stimulus=31, split=32, fit=0)
FW_CAP = 20000  # Frank-Wolfe iteration cap, fixed in the pre-registration
FW_REL_GAP = 1e-3  # stop when h_hi - h_lo <= FW_REL_GAP * h_hi
SLACK = 1e-9  # relative slack on every comparison (float64 rounding)
UNDECIDED_MAX = 0.05  # above this, a cell is solver-limited
LABEL_SHARE = 0.80  # CEILING-BOUND / NOT-EXCLUDED-BOUND threshold
PCA_RANK = 240  # addendum A: matches #134
S_GRID = (1, 2, 3, 5, 10, 20, 30, 100)  # addendum A: lift scales
CONTROL_MIN = 0.50
CELLS = [(b, fam, obj) for b in ("B8", "B32") for fam in ("tpr", "paircode") for obj in ("mse", "cert")]


# ---------------------------------------------------------------- hull distance


def hull_distances(
    U: np.ndarray,
    targets: np.ndarray,
    cap: int = FW_CAP,
    rel_gap: float = FW_REL_GAP,
    device: str = "cpu",
    log=print,
) -> dict:
    """Certified bracket [h_lo, h_hi] on infdist(U_t, conv{U_v : v != t}) for each t, by batched Frank-Wolfe
    with away steps on f(lam) = 1/2 |U_t - lam U|^2 over the simplex (lam_t pinned to 0).

    h_hi = |U_t - p| for the current hull point p (feasible). The Frank-Wolfe gap g >= f - f*, so
    h* = sqrt(2 f*) >= sqrt(2 (f - g)) = h_lo.
    """
    Ud = torch.tensor(U, dtype=torch.float64, device=device)
    n = Ud.shape[0]
    t = torch.tensor(targets, dtype=torch.long, device=device)
    B = len(t)
    x = Ud[t]  # (B, d)
    rows = torch.arange(B, device=device)
    # start at the nearest rival vertex
    d2 = (x * x).sum(1, keepdim=True) - 2 * x @ Ud.T + (Ud * Ud).sum(1)[None]
    d2[rows, t] = float("inf")
    s0 = d2.argmin(1)
    lam = torch.zeros(B, n, dtype=torch.float64, device=device)
    lam[rows, s0] = 1.0
    p = Ud[s0].clone()
    h_hi = torch.full((B,), float("inf"), dtype=torch.float64, device=device)
    h_lo = torch.zeros(B, dtype=torch.float64, device=device)
    iters = torch.zeros(B, dtype=torch.long, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    for it in range(cap):
        idx = active.nonzero().squeeze(1)
        if len(idx) == 0:
            break
        xa, pa, la, ta = x[idx], p[idx], lam[idx], t[idx]
        r_ = torch.arange(len(idx), device=device)
        g = (pa - xa) @ Ud.T  # grad wrt lam
        g[r_, ta] = float("inf")  # t is not a rival
        s = g.argmin(1)
        lg = (la * torch.where(torch.isinf(g), torch.zeros_like(g), g)).sum(1)
        gap = lg - g[r_, s]  # FW gap >= f - f*
        f = 0.5 * ((xa - pa) ** 2).sum(1)
        hi = torch.sqrt(2 * f)
        lo = torch.sqrt(torch.clamp(2 * (f - gap), min=0))
        h_hi[idx], h_lo[idx], iters[idx] = hi, lo, it
        done = (hi - lo) <= rel_gap * hi
        # away vertex: largest gradient among the active set
        ga = torch.where(la > 0, g, torch.full_like(g, -float("inf")))
        ga[r_, ta] = -float("inf")
        a = ga.argmax(1)
        away_gain = g[r_, a] - lg
        use_fw = gap >= away_gain
        d = torch.where(use_fw[:, None], Ud[s] - pa, pa - Ud[a])
        la_a = la[r_, a]
        gmax = torch.where(use_fw, torch.ones_like(la_a), la_a / torch.clamp(1 - la_a, min=1e-300))
        dn = (d * d).sum(1)
        gamma = torch.where(
            dn > 0, ((xa - pa) * d).sum(1) / torch.clamp(dn, min=1e-300), torch.zeros_like(dn)
        )
        gamma = torch.clamp(gamma, min=0)
        gamma = torch.minimum(gamma, gmax)
        gamma = torch.where(done, torch.zeros_like(gamma), gamma)
        fw_rows, aw_rows = use_fw & ~done, ~use_fw & ~done
        la[fw_rows] *= (1 - gamma[fw_rows])[:, None]
        la[r_[fw_rows], s[fw_rows]] += gamma[fw_rows]
        la[aw_rows] *= (1 + gamma[aw_rows])[:, None]
        la[r_[aw_rows], a[aw_rows]] -= gamma[aw_rows]
        la.clamp_(min=0)
        pa = pa + gamma[:, None] * d
        lam[idx], p[idx] = la, pa
        active[idx[done]] = False
        if it % 1000 == 0:
            log(f"    FW iter {it:6d}: {int(active.sum())}/{B} tokens open")
    return dict(
        h_lo=h_lo.cpu().numpy(),
        h_hi=h_hi.cpu().numpy(),
        iters=iters.cpu().numpy(),
        converged=(~active).cpu().numpy(),
    )


# ---------------------------------------------------------------- classification


def classify(m, d, unorm, h_lo, h_hi, uniform):
    """ORIGINAL (bias-free, vacuous on GPT-2; addendum A keeps it as secondary).
    Per context: 'certified' | 'CEILING' | 'ALIGNMENT' | 'UNDECIDED'. Comparisons carry SLACK."""
    two_d = 2 * d
    ceil_ = two_d >= unorm * h_hi * (1 + SLACK)
    align = two_d * (1 + SLACK) < unorm * h_lo
    out = np.where(ceil_, "CEILING", np.where(align, "ALIGNMENT", "UNDECIDED")).astype(object)
    out[uniform] = "certified"
    return out


def classify_lifted(d, c_lo, c_hi, uniform):
    """Addendum A. c_lo, c_hi: (contexts, len(S_GRID)) brackets on the lifted ceiling |(w,s)| h_s(t).
    CEILING if 2d >= c_hi for SOME s (certain); NOT-EXCLUDED if 2d < c_lo for EVERY s; else UNDECIDED."""
    two_d = (2 * d)[:, None]
    ceil_ = (two_d >= c_hi * (1 + SLACK)).any(1)
    notex = (two_d * (1 + SLACK) < c_lo).all(1)
    out = np.where(ceil_, "CEILING", np.where(notex, "NOT-EXCLUDED", "UNDECIDED")).astype(object)
    out[uniform] = "certified"
    return out


def soundness(m, unorm, h_hi):
    """The theorem gives m <= |u| h(t) <= |u| h_hi. A violation is a bug: abort."""
    bad = m > unorm * h_hi * (1 + SLACK) + 1e-12
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} contexts have margin above |u| h_hi: solver or pipeline bug")


def soundness_lifted(m, c_hi):
    """certificate_hull_ceiling_biased: m < |(w,s)| h_s(t) <= c_hi for every s. Violation = bug: abort."""
    bad = (m[:, None] > c_hi * (1 + SLACK) + 1e-12).any(1)
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} contexts have margin above the lifted ceiling: bug")


def cell_label(labels, alt="ALIGNMENT"):
    refused = labels[labels != "certified"]
    n_ref = len(refused)
    und = int((refused == "UNDECIDED").sum())
    dec = refused[refused != "UNDECIDED"]
    c, a = int((dec == "CEILING").sum()), int((dec == alt).sum())
    if n_ref == 0:
        label = "no refusals"
    elif und > UNDECIDED_MAX * n_ref:
        label = "solver-limited"
    elif c >= LABEL_SHARE * len(dec):
        label = "CEILING-BOUND"
    elif a >= LABEL_SHARE * len(dec):
        label = f"{alt}-BOUND"
    else:
        label = "MIXED"
    return dict(refused=n_ref, ceiling=c, other=a, undecided=und, label=label)


def headline(labels, alt="ALIGNMENT"):
    names = ("CEILING-BOUND", f"{alt}-BOUND", "MIXED")
    classified = [x for x in labels if x in names]
    for lab in names:
        if classified and sum(x == lab for x in classified) > len(classified) / 2:
            return lab
    return "MIXED"


def quartiles(a):
    return [float(v) for v in np.quantile(a, [0.25, 0.5, 0.75])] if len(a) else None


def lifted_ceiling(U, c, u_te, t_host, cap, device, log):
    """Brackets (contexts, len(S_GRID)) on |(w,s)| h_s(t) with b_v = <c, U_v>, w = u - c (addendum A)."""
    wn = np.sqrt(((u_te - c) ** 2).sum(1))
    b = U @ c
    targets = np.unique(t_host)
    pos = np.searchsorted(targets, t_host)
    c_lo, c_hi, conv = [], [], []
    for s_ in S_GRID:
        hd = hull_distances(np.hstack([U, (b / s_)[:, None]]), targets, cap=cap, device=device, log=log)
        norm = np.sqrt(wn**2 + s_**2)
        c_lo.append(norm * hd["h_lo"][pos])
        c_hi.append(norm * hd["h_hi"][pos])
        conv.append(int(hd["converged"].sum()))
        log(
            f"    lift s={s_:3d}: {conv[-1]}/{len(targets)} converged, "
            f"median ceiling {np.median(c_hi[-1]):.3f}"
        )
    return np.stack(c_lo, 1), np.stack(c_hi, 1), conv


# ---------------------------------------------------------------- dump / run


def cmd_dump(args):
    cs.SEEDS["stimulus"] = 999 if args.smoke else SEEDS["stimulus"]
    if args.smoke:
        cs.N_CONTEXTS = 600
    cs.cmd_dump(args)


def pca_substitute(u_tr, u_te, k=PCA_RANK):
    mu = u_tr.mean(0)
    _, _, vt = np.linalg.svd(u_tr - mu, full_matrices=False)
    P = vt[:k]
    return mu + (u_te - mu) @ P.T @ P


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = 500 if args.smoke else FW_CAP
    if args.smoke:
        cs.STEPS = 30
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    summary = dict(
        tag="empirical",
        prereg="docs/notes/hull_ceiling_prereg.md",
        smoke=bool(args.smoke),
        seeds=SEEDS,
        fw_cap=cap,
        tasks={},
    )
    for task in ("LIST", "SVO"):
        z = np.load(Path(args.dumps) / f"{task}.npz")
        n_fill, n_role = int(z["n_fill"]), int(z["n_role"])
        n = len(z["u"])
        order = np.random.default_rng(SEEDS["split"]).permutation(n)
        n_tr, n_va = int(0.6 * n), int(0.1 * n)
        tr, te = order[:n_tr], order[n_tr + n_va :]
        seen = {
            (int(a), int(s)) for i in tr for a, s, k in zip(z["f"][i], z["r"][i], z["m"][i], strict=True) if k
        }
        keep = [
            i
            for i in te
            if all(
                (int(a), int(s)) in seen for a, s, k in zip(z["f"][i], z["r"][i], z["m"][i], strict=True) if k
            )
        ]
        dropped = len(te) - len(keep)
        te = np.array(keep)
        U = z["U"].astype(np.float64)
        u_te = z["u"][te].astype(np.float64)
        unorm = np.sqrt((u_te**2).sum(1))
        t_host = (u_te @ U.T).argmax(1)
        targets = np.unique(t_host)
        log(f"{task}: train {len(tr)} test {len(te)} (dropped {dropped}); {len(targets)} distinct decisions")
        hd = hull_distances(U, targets, cap=cap, device=str(device), log=log)
        pos = np.searchsorted(targets, t_host)
        h_lo, h_hi = hd["h_lo"][pos], hd["h_hi"][pos]
        log(
            f"{task}: bias-free hull done; {int(hd['converged'].sum())}/{len(targets)} tokens converged; "
            f"median h_hi {np.median(hd['h_hi']):.4f}"
        )
        c_mean = z["u"][tr].astype(np.float64).mean(0)
        c_lo, c_hi, lift_conv = lifted_ceiling(U, c_mean, u_te, t_host, cap, str(device), log)

        T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
        f, r, mk = T(z["f"], torch.long), T(z["r"], torch.long), T(z["m"])
        u, Ut = T(z["u"]), T(z["U"])
        tr_t, te_t = torch.tensor(tr, device=device), torch.tensor(te, device=device)
        budgets = {
            "B8": cs.params("tpr", n_fill, n_role, 8, 0),
            "B32": cs.params("tpr", n_fill, n_role, 32, 0),
        }

        subs = {}
        for b, fam, obj in CELLS:
            size = (
                (8 if b == "B8" else 32)
                if fam == "tpr"
                else cs.largest_k(fam, budgets[b], n_fill, n_role, len(seen))
            )
            model = cs.build(fam, size, n_fill, n_role).to(device)
            cs.train(model, f[tr_t], r[tr_t], mk[tr_t], u[tr_t], Ut, obj, SEEDS["fit"])
            with torch.no_grad():
                subs[f"{b}/{fam}/{obj}"] = model(f[te_t], r[te_t], mk[te_t]).double().cpu().numpy()
        subs["control/pca240"] = pca_substitute(z["u"][tr].astype(np.float64), u_te)

        cells, out_labels, orig_labels = {}, [], []
        m_host = certificate_facts(u_te, u_te, z["U"])["m"]
        for key, uh in subs.items():
            facts = certificate_facts(u_te, uh, z["U"])
            if not (facts["t"] == t_host).all():
                raise AssertionError("host decision differs between checker and hull targets")
            uniform = certify_all_souffle(facts)["uniform"]
            m, d = facts["m"], facts["d"]
            soundness(m, unorm, h_hi)
            soundness_lifted(m, c_hi)
            labels = classify_lifted(d, c_lo, c_hi, uniform)
            ceil_ = labels == "CEILING"
            cell = dict(
                uniform_coverage=float(uniform.mean()),
                agree=float((facts["t_hat"] == facts["t"]).mean()),
                **cell_label(labels, alt="NOT-EXCLUDED"),
                shrink_factor_quartiles=quartiles(2 * d[ceil_] / c_hi[ceil_].min(1)),
                original_bias_free=cell_label(classify(m, d, unorm, h_lo, h_hi, uniform)),
            )
            cells[key] = cell
            if not key.startswith("control"):
                out_labels.append(cell["label"])
                orig_labels.append(cell["original_bias_free"]["label"])
            log(
                f"{task} {key:20s} uniform {cell['uniform_coverage']:.3f} refused {cell['refused']:5d} "
                f"C {cell['ceiling']:5d} NX {cell['other']:5d} U {cell['undecided']:5d} -> {cell['label']}  "
                f"(original: {cell['original_bias_free']['label']})"
            )
        summary["tasks"][task] = dict(
            n_train=len(tr),
            n_test=len(te),
            test_dropped_unseen_pair=dropped,
            distinct_decisions=len(targets),
            hull=dict(
                converged=int(hd["converged"].sum()),
                median_h_hi=float(np.median(hd["h_hi"])),
                max_iters=int(hd["iters"].max()),
                lifted_converged=dict(zip(map(str, S_GRID), lift_conv, strict=True)),
            ),
            norms=dict(
                u=quartiles(unorm),
                mean_u=float(np.linalg.norm(c_mean)),
                w=quartiles(np.sqrt(((u_te - c_mean) ** 2).sum(1))),
            ),
            host_margin=quartiles(m_host),
            ceiling_fraction=quartiles(m_host / c_hi.min(1)),
            control_ok=cells["control/pca240"]["uniform_coverage"] >= CONTROL_MIN,
            cells=cells,
            labels=out_labels,
            original_labels=orig_labels,
        )
    all_labels = [x for t in summary["tasks"].values() for x in t["labels"]]
    orig = [x for t in summary["tasks"].values() for x in t["original_labels"]]
    summary["verdict"] = dict(
        control_ok=all(t["control_ok"] for t in summary["tasks"].values()),
        headline=headline(all_labels, alt="NOT-EXCLUDED"),
        counts={lab: all_labels.count(lab) for lab in sorted(set(all_labels))},
        ceiling_fraction_median={k: t["ceiling_fraction"][1] for k, t in summary["tasks"].items()},
        original_bias_free_headline=headline(orig),
    )
    summary["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["verdict"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--smoke", action="store_true", help="bug check: stimulus seed 999, 600 contexts")
    rr = sub.add_parser("run")
    rr.add_argument("--dumps", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument(
        "--smoke", action="store_true", help="bug check: 30 steps, 500 FW iterations; numbers are NOT results"
    )
    args = ap.parse_args()
    {"dump": cmd_dump, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
