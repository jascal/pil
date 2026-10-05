"""The nuisance-class hull ceiling: how much could a model-side bound certify?

Pre-registered: docs/notes/nuisance_hull_prereg.md.

    # 1. dump (needs `transformers`):
    /path/to/venv/bin/python experiments/nuisance_hull.py dump --out runs/nuisance
    # 2. run (needs souffle), under a sleep inhibitor:
    systemd-inhibit --what=sleep:idle /path/to/venv/bin/python experiments/nuisance_hull.py run \
        --dumps runs/nuisance --out runs/nuisance/summary.json

A sigma-class is the 8 variants of one sigma = (S, V, O) under a nuisance time phrase P that sigma does not
encode. The clean-up and agreement conditions are strict affine inequalities in u (i-orca
hull_certified_conditions), so a class is certified on its whole convex hull iff all 8 variants are
certified: the ceiling for any sound set-level bound. Per-context certificates are
domain_exhaustion.evaluate_domain (#142, which reuses #141), unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_substitutes as cs  # noqa: E402
import cleanup_certificate as cc  # noqa: E402
import domain_exhaustion as de  # noqa: E402

SEEDS = dict(stimulus=91, split=92, fit=(0, 1, 2))
N_SIGMA, SMOKE_SIGMA = 1200, 80
PHRASES = (
    "Yesterday,",
    "On Monday,",
    "Last week,",
    "This morning,",
    "In the evening,",
    "Once again,",
    "Later that day,",
    "At noon,",
)
D_F = 32
N_COMBOS = 16


# ---------------------------------------------------------------- dump


def cmd_dump(args):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # optional dependency, dump only

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token, tok.padding_side = tok.eos_token, "right"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    occ = [w for w in cs.OCCUPATIONS if len(tok.encode(" " + w)) == 1][:40]
    verbs = [w for w in cs.VERBS if w.endswith("ed") and len(tok.encode(" " + w)) == 1][:16]
    combos = [
        (s, v, o) for s in range(len(occ)) for o in range(len(occ)) if s != o for v in range(len(verbs))
    ]
    random.Random(999 if args.smoke else SEEDS["stimulus"]).shuffle(combos)
    sigma = combos[: SMOKE_SIGMA if args.smoke else N_SIGMA]
    rows, meta = [], []
    for g, (s, v, o) in enumerate(sigma):
        for k, ph in enumerate(PHRASES):
            rows.append(f"{ph} the {occ[s]} {verbs[v]} the {occ[o]}. The {occ[o]} was {verbs[v]} by the")
            meta.append((g, s, v, o, k))
    us = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            enc = tok(rows[i : i + 256], return_tensors="pt", padding=True).to(device)
            last = enc["attention_mask"].sum(1) - 1
            h = model.transformer(**enc).last_hidden_state
            us.append(h[torch.arange(len(last), device=device), last].float().cpu())
    meta = np.array(meta)
    f = np.stack([meta[:, 1], len(occ) + meta[:, 2], meta[:, 3]], axis=1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "nuisance.npz",
        u=torch.cat(us).numpy(),
        f=f,
        r=np.tile(np.arange(3), (len(f), 1)),
        m=np.ones(f.shape, np.float32),
        group=meta[:, 0],
        phrase=meta[:, 4],
        n_fill=len(occ) + len(verbs),
        n_role=3,
        U=model.lm_head.weight.detach().float().cpu().numpy(),
    )
    print(f"wrote {out / 'nuisance.npz'}: {len(rows)} contexts in {len(sigma)} classes", flush=True)


# ---------------------------------------------------------------- classes and the hull


def class_split(n_classes: int, split_seed: int):
    order = np.random.default_rng(split_seed).permutation(n_classes)
    n_tr, n_va = int(0.6 * n_classes), int(0.1 * n_classes)
    return order[:n_tr], order[n_tr : n_tr + n_va], order[n_tr + n_va :]


def hull_verdicts(full: np.ndarray, t_host: np.ndarray, group: np.ndarray, classes: np.ndarray):
    """Per class: hull-certified (all variants full) and decision-constant (one host argmax for all
    variants)."""
    hull = np.array([full[group == c].all() for c in classes])
    const = np.array([len(np.unique(t_host[group == c])) == 1 for c in classes])
    return hull, const


def hull_check(u, f, r, m, group, classes, hull, p, rd, F_sets, U, t_code, rng, device):
    """For every hull-certified class, N_COMBOS random convex combinations of its residuals must keep exact
    clean-up and the code decision (hull_certified_conditions). Abort on any failure."""
    Ud = torch.tensor(U, dtype=torch.float64, device=device)
    checked = 0
    for c in classes[hull]:
        idx = np.where(group == c)[0]
        w = rng.dirichlet(np.ones(len(idx)), size=N_COMBOS)
        uc = w @ u[idx]
        rep = np.repeat(idx[:1], N_COMBOS)  # all variants share sigma (same f, r, m)
        _, exact = cc.cleanup(uc, p, rd, F_sets, f[rep], r[rep], m[rep])
        dec = (torch.tensor(uc, dtype=torch.float64, device=device) @ Ud.T).argmax(1).cpu().numpy()
        if (~exact).any() or (dec != t_code[idx[0]]).any():
            raise AssertionError(f"hull violated in class {c} (bug)")
        checked += N_COMBOS
    return checked


def decide(cells):
    mean = lambda c, k: c[k]["mean"] if c.get(k) else None  # noqa: E731
    h_all = {k: mean(c, "H_all") for k, c in cells.items()}
    h_const = {k: mean(c, "H_const") for k, c in cells.items()}
    return dict(
        H1=dict(H_all=h_all, verdict="pass" if any((v or 0) >= 0.10 for v in h_all.values()) else "fail"),
        H2=dict(
            H_const=h_const, verdict="pass" if any((v or 0) >= 0.20 for v in h_const.values()) else "fail"
        ),
    )


KEYS = ("H_all", "H_const", "const_share", "F_cov", "median_class_radius", "median_rho_loc")


def summarise(per_seed):
    cell = {}
    for k in KEYS:
        vals = [s[k] for s in per_seed if s.get(k) is not None]
        cell[k] = (
            dict(mean=float(np.mean(vals)), range=[float(min(vals)), float(max(vals))]) if vals else None
        )
    cell["flagged_seeds"] = sum(bool(s["flagged"]) for s in per_seed)
    return cell


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        cs.STEPS = 30
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    z = np.load(Path(args.dumps) / "nuisance.npz")
    f, r, m, group = z["f"], z["r"], z["m"], z["group"]
    n_fill, n_role = int(z["n_fill"]), int(z["n_role"])
    n_classes = int(group.max()) + 1
    ctr, cva, cte = class_split(n_classes, SEEDS["split"])
    tr = np.where(np.isin(group, ctr))[0]
    te = np.where(np.isin(group, cte))[0]
    U = z["U"].astype(np.float64)
    u = z["u"].astype(np.float64)
    t_host = np.concatenate([(u[s : s + 1024] @ U.T).argmax(1) for s in range(0, len(u), 1024)])
    F_sets = cc.filler_sets(f[tr], r[tr], m[tr], n_role)
    summary = dict(
        tag="empirical",
        prereg="docs/notes/nuisance_hull_prereg.md",
        smoke=bool(args.smoke),
        seeds=dict(SEEDS, stimulus=999 if args.smoke else SEEDS["stimulus"]),
        data=dict(
            classes=n_classes,
            train_classes=len(ctr),
            val_classes=len(cva),
            test_classes=len(cte),
            contexts=len(u),
            phrases=list(PHRASES),
        ),
        cells={},
    )
    log(f"{n_classes} classes ({len(u)} contexts): train {len(ctr)} val {len(cva)} test {len(cte)}")
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
    f_t, r_t, m_t, u_t, U_t = T(f, torch.long), T(r, torch.long), T(m), T(z["u"]), T(z["U"])
    tr_t = torch.tensor(tr, device=device)
    ute, fte, rte, mte, gte = u[te], f[te], r[te], m[te], group[te]
    for obj in cc.OBJECTIVES:
        per_seed = []
        for seed in SEEDS["fit"]:
            model = cs.TPR(n_fill, n_role, D_F).to(device)
            args_tr = (model, f_t[tr_t], r_t[tr_t], m_t[tr_t], u_t[tr_t], U_t)
            if obj == "t6":
                cc.train_t6(*args_tr, seed)
            else:
                cs.train(*args_tr, obj, seed)
            p = cc.tpr_params(model)
            rd = cc.readout(p)
            rng = np.random.default_rng(3000 + seed)
            res = de.evaluate_domain(p, rd, F_sets, ute, fte, rte, mte, U, t_host[te], str(device), rng)
            if res["flagged"]:
                per_seed.append(dict(flagged=True, seed=seed, H_all=0.0, H_const=0.0))
                continue
            hull, const = hull_verdicts(res["full"], t_host[te], gte, cte)
            x = cc.code_points(p, fte, rte, mte)
            t_code, _ = cc.beta_and_decision(x, U, str(device))
            combos = hull_check(
                ute, fte, rte, mte, gte, cte, hull, p, rd, F_sets, U, t_code, rng, str(device)
            )
            nn = np.sqrt(((ute - x) ** 2).sum(1))
            radius = [np.linalg.norm(ute[gte == c] - ute[gte == c].mean(0), axis=1).max() for c in cte]
            fu = res["full"]
            fm = dict(
                flagged=False,
                seed=seed,
                H_all=float(hull.mean()),
                H_const=float(hull[const].mean()) if const.any() else None,
                const_share=float(const.mean()),
                F_cov=float(fu.mean()),
                median_class_radius=float(np.median(radius)),
                median_rho_loc=float(np.median(res["ratio"][fu] * nn[fu])) if fu.any() else None,
                hull_classes=int(hull.sum()),
                hull_combos_checked=combos,
                ties=list(res["ties"]),
                neighbourhood_checks=res["checks"],
                oov=res["oov"],
            )
            per_seed.append(fm)
            log(
                f"dF{D_F} {obj:4s} seed {seed}: H_all {fm['H_all']:.4f} H_const {fm['H_const']} "
                f"const {fm['const_share']:.3f} F_cov {fm['F_cov']:.4f} hull classes {fm['hull_classes']} "
                f"combos {combos} ties {fm['ties']} checks {fm['neighbourhood_checks']}"
            )
        summary["cells"][f"dF{D_F}/{obj}"] = dict(per_seed=per_seed, **summarise(per_seed))
    summary["verdict"] = decide(summary["cells"])
    summary["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["verdict"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--smoke", action="store_true", help="bug check: stimulus 999, 80 classes")
    rr = sub.add_parser("run")
    rr.add_argument("--dumps", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument("--smoke", action="store_true", help="bug check: 30 steps; numbers are NOT results")
    args = ap.parse_args()
    {"dump": cmd_dump, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
