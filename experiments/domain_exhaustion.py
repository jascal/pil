"""Certifying the whole SVO template domain by exhaustion.

Pre-registered: docs/notes/domain_exhaustion_prereg.md.

    # 1. dump every context of D (needs `transformers`):
    /path/to/venv/bin/python experiments/domain_exhaustion.py dump --out runs/domain
    # 2. run (needs souffle), under a sleep inhibitor:
    systemd-inhibit --what=sleep:idle /path/to/venv/bin/python experiments/domain_exhaustion.py run \
        --dumps runs/domain --out runs/domain/summary.json

D = all 40 x 39 x 16 = 24,960 SVO contexts (certified_substitutes.build_rows, unchanged, with
N_CONTEXTS >= |D|). The first 12,000 in the seeded order are the sampled pool (train / validation / test); the
rest are never sampled.
Every context of D gets halfspace_verdict.py's per-context certificate (#141, reused unchanged).
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
import cleanup_radius as cr  # noqa: E402
import halfspace_verdict as hv  # noqa: E402

SEEDS = dict(stimulus=71, split=72, fit=(0, 1, 2))
POOL = 12000
SETTINGS = (("SVO", 8), ("SVO", 32))
N_OCC, N_VERB = 40, 16
SMOKE_D, SMOKE_POOL = 600, 300


# ---------------------------------------------------------------- dump


def cmd_dump(args):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # optional dependency, dump only

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token, tok.padding_side = tok.eos_token, "right"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    cs.N_CONTEXTS = 10**9  # >= |D|: build_rows returns every combination, in a seeded shuffled order
    seed = 999 if args.smoke else SEEDS["stimulus"]
    rows, n_fill, n_role = cs.build_rows("SVO", tok, random.Random(seed + 1000))
    if args.smoke:
        rows = rows[:SMOKE_D]
    f = np.array([[a for a, _ in pairs] for _, pairs in rows], dtype=np.int64)
    r = np.array([[s for _, s in pairs] for _, pairs in rows], dtype=np.int64)
    m = np.ones(f.shape, dtype=np.float32)
    us = []
    with torch.no_grad():
        for s in range(0, len(rows), 256):
            enc = tok([t for t, _ in rows[s : s + 256]], return_tensors="pt", padding=True).to(device)
            last = enc["attention_mask"].sum(1) - 1
            h = model.transformer(**enc).last_hidden_state
            us.append(h[torch.arange(len(last), device=device), last].float().cpu())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "SVO.npz",
        u=torch.cat(us).numpy(),
        f=f,
        r=r,
        m=m,
        n_fill=n_fill,
        n_role=n_role,
        U=model.lm_head.weight.detach().float().cpu().numpy(),
    )
    print(f"wrote {out / 'SVO.npz'}: {len(rows)} contexts, {n_fill} fillers, {n_role} roles", flush=True)


# ---------------------------------------------------------------- domain bookkeeping


def canonical_index(f: np.ndarray) -> np.ndarray:
    """(S, V, O) -> S * N_OCC * N_VERB + O * N_VERB + V over the 40 x 40 x 16 grid (S != O cells used).
    Fillers: S and O are occupations 0..39 (roles 0 and 2); V is N_OCC + verb (role 1)."""
    s, v, o = f[:, 0], f[:, 1] - N_OCC, f[:, 2]
    return s * N_OCC * N_VERB + o * N_VERB + v


def partition(n: int, pool: int, split_seed: int):
    order = np.random.default_rng(split_seed).permutation(pool)
    n_tr, n_va = int(0.6 * pool), int(0.1 * pool)
    return order[:n_tr], order[n_tr : n_tr + n_va], order[n_tr + n_va :], np.arange(pool, n)


def in_vocab(f, r, m, F_sets) -> np.ndarray:
    sets = [set(int(x) for x in Fs) for Fs in F_sets]
    return np.array(
        [all(int(f[i, j]) in sets[int(r[i, j])] for j in range(f.shape[1]) if m[i, j]) for i in range(len(f))]
    )


# ---------------------------------------------------------------- evaluation


def evaluate_domain(p, rd, F_sets, u, f, r, m, U, t_host, device, rng):
    """#141's certificate on every in-vocabulary context of D; out-of-vocabulary contexts are uncertified."""
    n_d = len(u)
    full = np.zeros(n_d, dtype=bool)
    exact = np.zeros(n_d, dtype=bool)
    agree_m = np.zeros(n_d, dtype=bool)
    ratio = np.full(n_d, np.nan)
    if rd["no_left_inverse"] or rd["no_dual_readout"]:
        return dict(
            flagged=True, full=full, exact=exact, agree=agree_m, ratio=ratio, ties=(0, 0), checks=0, oov=None
        )
    iv = in_vocab(f, r, m, F_sets)
    idx = np.where(iv)[0]
    ui, fi, ri, mi = u[idx], f[idx], r[idx], m[idx]
    x = cc.code_points(p, fi, ri, mi)
    n = ui - x
    t_code, _ = cc.beta_and_decision(x, U, device)
    maps = cr.role_maps(p, rd)
    rows, min_slack, delta_c, worst_c = hv.cleanup_slacks(p, rd, maps, F_sets, n, fi, ri, mi)
    hmin, delta_h, worst_h = hv.host_slacks(ui, t_code, U, device)
    _, ex = cc.cleanup(ui, p, rd, F_sets, fi, ri, mi)
    host_agree = t_host[idx] == t_code
    v = hv.verdicts_souffle(len(idx), rows, hmin)
    vp = hv.verdicts_python(len(idx), rows, hmin)
    if any(not (v[k] == vp[k]).all() for k in v):
        raise AssertionError("Soufflé and Python verdicts differ")
    ties = hv.iff_check(v, ex, host_agree, min_slack, hmin)
    rho_loc = np.minimum(delta_c, delta_h)
    checks = hv.neighbourhood_check(
        ui, v["full"], rho_loc, worst_c, worst_h, t_code, p, rd, F_sets, fi, ri, mi, U, rng, device
    )
    full[idx], exact[idx], agree_m[idx] = v["full"], ex, host_agree
    ratio[idx] = rho_loc / np.sqrt((n**2).sum(1))
    return dict(
        flagged=False,
        full=full,
        exact=exact,
        agree=agree_m,
        ratio=ratio,
        ties=(ties["ties_clean"], ties["ties_agree"]),
        checks=checks,
        oov=int((~iv).sum()),
    )


def fit_metrics(res, parts):
    out = dict(
        flagged=res["flagged"], oov=res["oov"], ties=list(res["ties"]), neighbourhood_checks=res["checks"]
    )
    for name, ix in parts.items():
        out[f"F_cov_{name}"] = float(res["full"][ix].mean()) if len(ix) else None
        out[f"E_{name}"] = float(res["exact"][ix].mean()) if len(ix) else None
        out[f"A_{name}"] = float(res["agree"][ix].mean()) if len(ix) else None
    cert = res["full"]
    out["uncertified_D"] = int((~cert).sum())
    out["domain_wide"] = bool(cert.all())
    out["median_rho_loc_over_n_D"] = float(np.nanmedian(res["ratio"][cert])) if cert.any() else None
    return out


def summarise(per_seed):
    keys = [k for k in per_seed[0] if k.startswith(("F_cov_", "E_", "A_")) or k == "median_rho_loc_over_n_D"]
    cell = {}
    for k in keys:
        vals = [s[k] for s in per_seed if s.get(k) is not None]
        cell[k] = (
            dict(mean=float(np.mean(vals)), range=[float(min(vals)), float(max(vals))]) if vals else None
        )
    cell["min_uncertified_D"] = min(s["uncertified_D"] for s in per_seed)
    cell["any_domain_wide"] = any(s["domain_wide"] for s in per_seed)
    cell["flagged_seeds"] = sum(bool(s["flagged"]) for s in per_seed)
    return cell


def decide(cells):
    mean = lambda c, k: c[k]["mean"] if c.get(k) else None  # noqa: E731
    fd = {k: mean(c, "F_cov_D") for k, c in cells.items()}
    elig = {k: c for k, c in cells.items() if (mean(c, "F_cov_test") or 0) >= 0.05}
    h2 = {
        k: dict(
            test=mean(c, "F_cov_test"),
            never=mean(c, "F_cov_never"),
            gap=abs(mean(c, "F_cov_never") - mean(c, "F_cov_test")),
            passes=bool(abs(mean(c, "F_cov_never") - mean(c, "F_cov_test")) <= 0.05),
        )
        for k, c in elig.items()
    }
    return dict(
        H1=dict(F_cov_D=fd, verdict="pass" if any((v or 0) >= 0.5 for v in fd.values()) else "fail"),
        H2=dict(
            cells=h2,
            verdict="untestable" if not h2 else ("pass" if all(v["passes"] for v in h2.values()) else "fail"),
        ),
        domain_wide=dict(
            any=any(c["any_domain_wide"] for c in cells.values()),
            min_uncertified={k: c["min_uncertified_D"] for k, c in cells.items()},
        ),
    )


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        cs.STEPS = 30
    pool = SMOKE_POOL if args.smoke else POOL
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    z = np.load(Path(args.dumps) / "SVO.npz")
    n_fill, n_role = int(z["n_fill"]), int(z["n_role"])
    f, r, m = z["f"], z["r"], z["m"]
    n_d = len(z["u"])
    if not args.smoke and n_d != N_OCC * (N_OCC - 1) * N_VERB:
        raise AssertionError(f"domain size {n_d} != {N_OCC * (N_OCC - 1) * N_VERB}")
    canon = canonical_index(f)
    if len(np.unique(canon)) != n_d:
        raise AssertionError("domain contains duplicate contexts")
    tr, va, te, never = partition(n_d, pool, SEEDS["split"])
    parts = dict(D=np.arange(n_d), train=tr, test=te, never=never)
    U = z["U"].astype(np.float64)
    u = z["u"].astype(np.float64)
    t_host = np.concatenate([(u[s : s + 1024] @ U.T).argmax(1) for s in range(0, n_d, 1024)])
    F_sets = cc.filler_sets(f[tr], r[tr], m[tr], n_role)
    summary = dict(
        tag="empirical",
        prereg="docs/notes/domain_exhaustion_prereg.md",
        smoke=bool(args.smoke),
        seeds=dict(SEEDS, stimulus=999 if args.smoke else SEEDS["stimulus"]),
        domain=dict(size=n_d, pool=pool, train=len(tr), validation=len(va), test=len(te), never=len(never)),
        cells={},
    )
    log(f"D {n_d}: train {len(tr)} val {len(va)} test {len(te)} never {len(never)}")
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
    f_t, r_t, m_t, u_t, U_t = T(f, torch.long), T(r, torch.long), T(m), T(z["u"]), T(z["U"])
    tr_t = torch.tensor(tr, device=device)
    masks = {}
    for _, d_f in SETTINGS:
        for obj in cc.OBJECTIVES:
            per_seed = []
            for seed in SEEDS["fit"]:
                model = cs.TPR(n_fill, n_role, d_f).to(device)
                args_tr = (model, f_t[tr_t], r_t[tr_t], m_t[tr_t], u_t[tr_t], U_t)
                if obj == "t6":
                    cc.train_t6(*args_tr, seed)
                else:
                    cs.train(*args_tr, obj, seed)
                p = cc.tpr_params(model)
                rd = cc.readout(p)
                res = evaluate_domain(
                    p, rd, F_sets, u, f, r, m, U, t_host, str(device), np.random.default_rng(2000 + seed)
                )
                fm = fit_metrics(res, parts)
                fm["seed"] = seed
                per_seed.append(fm)
                grid = np.zeros(N_OCC * N_OCC * N_VERB, dtype=bool)
                grid[canon[res["full"]]] = True
                masks[f"dF{d_f}_{obj}_seed{seed}"] = np.packbits(grid)
                log(
                    f"SVO dF{d_f} {obj:4s} seed {seed}: F_cov D {fm['F_cov_D']:.4f} "
                    f"train {fm['F_cov_train']:.4f} "
                    f"test {fm['F_cov_test']:.4f} never {fm['F_cov_never']} "
                    f"uncertified {fm['uncertified_D']} "
                    f"oov {fm['oov']} ties {fm['ties']} checks {fm['neighbourhood_checks']}"
                )
            summary["cells"][f"SVO/dF{d_f}/{obj}"] = dict(per_seed=per_seed, **summarise(per_seed))
    summary["verdict"] = decide(summary["cells"])
    summary["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    valid = np.zeros(N_OCC * N_OCC * N_VERB, dtype=bool)
    valid[canon] = True
    np.savez_compressed(out.with_name("certified.npz"), domain=np.packbits(valid), **masks)
    print(json.dumps(summary["verdict"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--smoke", action="store_true", help="bug check: first 600 of a seed-999 shuffle")
    rr = sub.add_parser("run")
    rr.add_argument("--dumps", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument("--smoke", action="store_true", help="bug check: 30 steps; numbers are NOT results")
    args = ap.parse_args()
    {"dump": cmd_dump, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
