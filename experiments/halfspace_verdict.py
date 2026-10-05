"""Per-context half-space verdicts and certified neighbourhoods around observed residuals.

Pre-registered: docs/notes/halfspace_verdict_prereg.md.

    # 1. dump (needs `transformers`):
    /path/to/venv/bin/python experiments/halfspace_verdict.py dump --out runs/halfspace
    # 2. run (needs souffle), under a sleep inhibitor:
    systemd-inhibit --what=sleep:idle /path/to/venv/bin/python experiments/halfspace_verdict.py run \
        --dumps runs/halfspace --out runs/halfspace/summary.json

Theorems (i-orca, PIC_Cleanup.thy, kernel-checked):
- nearest_iff_halfspace: role s of u = x(sigma) + n snaps to sigma(s) IFF for every rival a,
  slack(s, a) = |d_a|^2 / 2 - <M_s n, d_a> > 0
  (dual readouts, d_a = f_a - f_sigma(s), M_s n = unbind(P n, w_s));
- the pairwise iff: the host decides the code point's decision t IFF <u, U_t - U_v> > 0 for every v != t;
- cleanup_local_certified: every u + e with |e| < min(delta_c, delta_h) gets exact clean-up and the
  decision t,
  delta_c = min_{s,a} slack(s, a) / |M_s^T d_a|,  delta_h = min_v <u, U_t - U_v> / |U_t - U_v|.

Data, training, readouts and clean-up are cleanup_certificate.py's (#139); the role maps are
cleanup_radius.py's (#140). Both are reused unchanged; only the seeds differ.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_substitutes as cs  # noqa: E402
import cleanup_certificate as cc  # noqa: E402
import cleanup_radius as cr  # noqa: E402

SEEDS = dict(stimulus=61, split=62, fit=(0, 1, 2))
SCALE, TAU = cc.SCALE, cc.TAU
TIE = 1e-6  # a context whose deciding slack is within TIE of 0 is a tie, not an iff mismatch
N_RANDOM = 8  # random perturbation directions per certified context
SHRINK = 0.999

SOUFFLE = """
// clean(c): every half-space slack exceeds tau.
// agree(c): the host's minimum gap to the code decision exceeds tau.
.decl context(c:number) .input context
.decl slack(c:number, s:number, a:number, x:number) .input slack
.decl hmin(c:number, x:number) .input hmin
.decl tau(t:number) .input tau
.decl bad(c:number)
bad(c) :- slack(c, _, _, x), tau(t), x <= t.
.decl clean(c:number) .output clean
clean(c) :- context(c), !bad(c).
.decl agree(c:number) .output agree
agree(c) :- hmin(c, h), tau(t), h > t.
.decl full(c:number) .output full
full(c) :- clean(c), agree(c).
.decl seen(c:number) .output seen
seen(c) :- context(c).
"""


# ---------------------------------------------------------------- slacks


def cleanup_slacks(p, rd, maps, F_sets, n, f, r, m):
    """Per context: every half-space slack (as rows (c, s, a, slack)), the min slack, delta_c, and the worst
    (role, rival) direction q = M_s^T d_a / |M_s^T d_a| for the argmin of slack / |q|."""
    ef = p["ef"]
    rows = []
    min_slack = np.full(len(n), np.inf)
    delta_c = np.full(len(n), np.inf)
    worst = np.zeros_like(n)
    for s, Fs in enumerate(F_sets):
        if len(Fs) < 2:
            continue
        E = ef[Fs]
        G = maps[s] @ maps[s].T
        D = E[None, :, :] - E[:, None, :]  # D[b, a] = f_a - f_b
        dn2 = (D**2).sum(2)
        qn = np.sqrt(np.clip(np.einsum("bai,ij,baj->ba", D, G, D), 0, None))
        pos = {int(x): i for i, x in enumerate(Fs)}
        for j in range(f.shape[1]):
            idx = np.where(m[:, j].astype(bool) & (r[:, j] == s))[0]
            if len(idx) == 0:
                continue
            o = n[idx] @ maps[s].T  # (k, d_F) = M_s n
            b = np.array([pos[int(x)] for x in f[idx, j]])
            sl = dn2[b] / 2 - (o @ E.T - (o * E[b]).sum(1, keepdims=True))  # (k, |F_s|)
            sl[np.arange(len(idx)), b] = np.inf
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(
                    qn[b] > 0, sl / np.where(qn[b] > 0, qn[b], 1), np.where(sl > 0, np.inf, -np.inf)
                )
            ratio[np.arange(len(idx)), b] = np.inf
            for kk, c in enumerate(idx):
                for a in range(len(Fs)):
                    if a != b[kk]:
                        rows.append((int(c), s, int(Fs[a]), float(sl[kk, a])))
            amin = ratio.argmin(1)
            for kk, c in enumerate(idx):
                min_slack[c] = min(min_slack[c], sl[kk].min())
                if ratio[kk, amin[kk]] < delta_c[c]:
                    delta_c[c] = ratio[kk, amin[kk]]
                    q = maps[s].T @ (E[amin[kk]] - E[b[kk]])
                    worst[c] = q / max(np.linalg.norm(q), 1e-300)
    return rows, min_slack, delta_c, worst


def host_slacks(u, t, U, device="cpu", chunk=256):
    """hmin = min_{v != t} <u, U_t - U_v>; delta_h = min_v of that over |U_t - U_v|; the worst host direction
    -(U_t - U_v*) / |U_t - U_v*| (the one that lowers the margin fastest)."""
    Ud = torch.tensor(U, dtype=torch.float64, device=device)
    un2 = (Ud * Ud).sum(1)
    hmin, dh, worst = [], [], []
    for s in range(0, len(u), chunk):
        us = torch.tensor(u[s : s + chunk], dtype=torch.float64, device=device)
        tt = torch.tensor(t[s : s + chunk], device=device)
        rows = torch.arange(len(tt), device=device)
        L = us @ Ud.T
        gap = L[rows, tt][:, None] - L
        dist = torch.sqrt(torch.clamp(un2[tt][:, None] + un2[None] - 2 * Ud[tt] @ Ud.T, min=0))
        gap[rows, tt] = math.inf
        ratio = torch.where(
            dist > 0,
            gap / torch.where(dist > 0, dist, torch.ones_like(dist)),
            torch.where(gap > 0, torch.full_like(gap, math.inf), torch.full_like(gap, -math.inf)),
        )
        ratio[rows, tt] = math.inf
        v = ratio.argmin(1)
        hmin.append(gap.min(1).values.cpu().numpy())
        dh.append(ratio.min(1).values.cpu().numpy())
        dvec = Ud[tt] - Ud[v]
        worst.append((-dvec / torch.clamp(dvec.norm(dim=1, keepdim=True), min=1e-300)).cpu().numpy())
    return np.concatenate(hmin), np.concatenate(dh), np.concatenate(worst)


# ---------------------------------------------------------------- verdicts


def _fixed(rows, hmin, tau=TAU):
    fl = lambda x: int(math.floor(x * SCALE)) if math.isfinite(x) else (2**62 if x > 0 else -(2**62))  # noqa: E731
    return [(c, s, a, fl(x)) for c, s, a, x in rows], [fl(x) for x in hmin], int(math.ceil(tau * SCALE))


def verdicts_python(n_ctx, rows, hmin, tau=TAU):
    fr, fh, t = _fixed(rows, hmin, tau)
    bad = np.zeros(n_ctx, dtype=bool)
    for c, _, _, x in fr:
        if x <= t:
            bad[c] = True
    clean = ~bad
    agree = np.array([h > t for h in fh])
    return dict(clean=clean, agree=agree, full=clean & agree)


def verdicts_souffle(n_ctx, rows, hmin, tau=TAU):
    fr, fh, t = _fixed(rows, hmin, tau)
    with tempfile.TemporaryDirectory(prefix="halfspace-verdict-") as raw:
        d = Path(raw)
        (d / "hs.dl").write_text(SOUFFLE)
        (d / "context.facts").write_text("".join(f"{i}\n" for i in range(n_ctx)))
        with open(d / "slack.facts", "w") as fh_:
            fh_.writelines(f"{c}\t{s}\t{a}\t{x}\n" for c, s, a, x in fr)
        (d / "hmin.facts").write_text("".join(f"{i}\t{x}\n" for i, x in enumerate(fh)))
        (d / "tau.facts").write_text(f"{t}\n")
        subprocess.run(
            ["souffle", "-F", str(d), "-D", str(d), str(d / "hs.dl")],
            check=True,
            capture_output=True,
            text=True,
        )
        read = lambda nm: {int(x) for x in (d / f"{nm}.csv").read_text().split()}  # noqa: E731
        if read("seen") != set(range(n_ctx)):
            raise RuntimeError("Soufflé did not account for every context")
        return {k: np.array([i in read(k) for i in range(n_ctx)]) for k in ("clean", "agree", "full")}


def iff_check(v, exact, host_agree, min_slack, hmin):
    """H1: clean == exact and agree == host agreement, except ties (deciding slack within TIE of 0). Abort."""
    tie_c = np.abs(min_slack) <= TIE
    tie_h = np.abs(hmin) <= TIE
    mis_c = (v["clean"] != exact) & ~tie_c
    mis_h = (v["agree"] != host_agree) & ~tie_h
    if mis_c.any() or mis_h.any():
        raise AssertionError(f"iff mismatch: {int(mis_c.sum())} clean, {int(mis_h.sum())} agree (bug)")
    return dict(ties_clean=int(tie_c.sum()), ties_agree=int(tie_h.sum()))


def neighbourhood_check(u, full, rho_loc, worst_c, worst_h, t_code, p, rd, F_sets, f, r, m, U, rng, device):
    """For every full context: N_RANDOM random directions plus the two worst-case directions at
    SHRINK * rho_loc must keep exact clean-up and the code decision. Abort on any failure."""
    idx = np.where(full)[0]
    if len(idx) == 0:
        return 0
    dirs = [rng.normal(size=(len(idx), u.shape[1])) for _ in range(N_RANDOM)] + [worst_c[idx], worst_h[idx]]
    Ud = torch.tensor(U, dtype=torch.float64, device=device)
    checked = 0
    for dvec in dirs:
        dvec = dvec / np.linalg.norm(dvec, axis=1, keepdims=True)
        up = u[idx] + (SHRINK * rho_loc[idx])[:, None] * dvec
        _, exact = cc.cleanup(up, p, rd, F_sets, f[idx], r[idx], m[idx])
        dec = np.concatenate(
            [
                (torch.tensor(up[s : s + 256], dtype=torch.float64, device=device) @ Ud.T)
                .argmax(1)
                .cpu()
                .numpy()
                for s in range(0, len(up), 256)
            ]
        )
        bad = ~exact | (dec != t_code[idx])
        if bad.any():
            raise AssertionError(f"neighbourhood violated on {int(bad.sum())} certified contexts (bug)")
        checked += len(idx)
    return checked


# ---------------------------------------------------------------- evaluation and decision rules


def evaluate(p, rd, F_sets, u_te, f_te, r_te, m_te, U, t_host, device, rng):
    if rd["no_left_inverse"] or rd["no_dual_readout"]:
        return dict(flagged=True, E=None, A=None, F_cov=0.0)
    x = cc.code_points(p, f_te, r_te, m_te)
    n = u_te - x
    nn = np.sqrt((n**2).sum(1))
    t_code, _ = cc.beta_and_decision(x, U, device)
    maps = cr.role_maps(p, rd)
    rows, min_slack, delta_c, worst_c = cleanup_slacks(p, rd, maps, F_sets, n, f_te, r_te, m_te)
    hmin, delta_h, worst_h = host_slacks(u_te, t_code, U, device)
    _, exact = cc.cleanup(u_te, p, rd, F_sets, f_te, r_te, m_te)
    host_agree = t_host == t_code
    v = verdicts_souffle(len(u_te), rows, hmin)
    vp = verdicts_python(len(u_te), rows, hmin)
    if any(not (v[k] == vp[k]).all() for k in v):
        raise AssertionError("Soufflé and Python verdicts differ")
    ties = iff_check(v, exact, host_agree, min_slack, hmin)
    rho_loc = np.minimum(delta_c, delta_h)
    checked = neighbourhood_check(
        u_te, v["full"], rho_loc, worst_c, worst_h, t_code, p, rd, F_sets, f_te, r_te, m_te, U, rng, device
    )
    rdir = cr.rho_dir(cr.directional_tables(p, maps, F_sets), F_sets, f_te, r_te, m_te)
    fu = v["full"]
    med = lambda a: float(np.median(a[fu])) if fu.any() else None  # noqa: E731
    return dict(
        flagged=False,
        E=float(exact.mean()),
        A=float(host_agree.mean()),
        F_cov=float(fu.mean()),
        clean_cov=float(v["clean"].mean()),
        agree_cov=float(v["agree"].mean()),
        **ties,
        neighbourhood_checks=checked,
        median_rho_loc=med(rho_loc),
        median_delta_c=med(delta_c),
        median_delta_h=med(delta_h),
        median_rho_loc_over_n=med(rho_loc / nn),
        median_rho_loc_over_rho_dir=med(rho_loc / rdir),
        bottleneck_share=float((delta_h[fu] < delta_c[fu]).mean()) if fu.any() else None,
        median_n=float(np.median(nn)),
    )


KEYS = (
    "E",
    "A",
    "F_cov",
    "clean_cov",
    "agree_cov",
    "median_rho_loc",
    "median_delta_c",
    "median_delta_h",
    "median_rho_loc_over_n",
    "median_rho_loc_over_rho_dir",
    "bottleneck_share",
    "median_n",
)


def summarise(per_seed):
    cell = {}
    for k in KEYS:
        vals = [s[k] for s in per_seed if s.get(k) is not None]
        cell[k] = (
            dict(mean=float(np.mean(vals)), range=[float(min(vals)), float(max(vals))]) if vals else None
        )
    cell["flagged_seeds"] = sum(bool(s["flagged"]) for s in per_seed)
    cell["ties_clean"] = sum(s.get("ties_clean", 0) for s in per_seed)
    cell["ties_agree"] = sum(s.get("ties_agree", 0) for s in per_seed)
    bs = cell["bottleneck_share"]["mean"] if cell["bottleneck_share"] else None
    cell["bottleneck"] = (
        None if bs is None else "host-bound" if bs >= 0.8 else "clean-up-bound" if bs <= 0.2 else "mixed"
    )
    return cell


def decide(cells):
    mean = lambda c, k: c[k]["mean"] if c[k] else None  # noqa: E731
    h2 = {k: mean(c, "F_cov") for k, c in cells.items()}
    eligible = {k: c for k, c in cells.items() if (mean(c, "F_cov") or 0) >= 0.05}
    h3 = {
        k: dict(
            F_cov=mean(c, "F_cov"),
            median_rho_loc_over_n=mean(c, "median_rho_loc_over_n"),
            passes=bool((mean(c, "median_rho_loc_over_n") or 0) >= 0.01),
        )
        for k, c in eligible.items()
    }
    return dict(
        H1=dict(verdict="pass", note="the run aborts on any iff mismatch, so completion means no mismatch"),
        H2=dict(F_cov=h2, verdict="pass" if any((v or 0) >= 0.5 for v in h2.values()) else "fail"),
        H3=dict(
            cells=h3,
            verdict="untestable" if not h3 else ("pass" if all(v["passes"] for v in h3.values()) else "fail"),
        ),
    )


# ---------------------------------------------------------------- dump / run


def cmd_dump(args):
    cs.SEEDS["stimulus"] = 999 if args.smoke else SEEDS["stimulus"]
    if args.smoke:
        cs.N_CONTEXTS = 600
    cs.cmd_dump(args)


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        cs.STEPS = 30
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    summary = dict(
        tag="empirical",
        prereg="docs/notes/halfspace_verdict_prereg.md",
        smoke=bool(args.smoke),
        seeds=dict(SEEDS, stimulus=999 if args.smoke else SEEDS["stimulus"]),
        tasks={},
        cells={},
    )
    for task in ("LIST", "SVO"):
        z = np.load(Path(args.dumps) / f"{task}.npz")
        n_fill, n_role = int(z["n_fill"]), int(z["n_role"])
        fa, ra, ma = z["f"], z["r"], z["m"]
        for i in range(len(fa)):
            bound = ra[i][ma[i].astype(bool)]
            assert len(set(bound.tolist())) == len(bound), "a role is bound twice in one context"
        n = len(z["u"])
        order = np.random.default_rng(SEEDS["split"]).permutation(n)
        n_tr, n_va = int(0.6 * n), int(0.1 * n)
        tr, te = order[:n_tr], order[n_tr + n_va :]
        seen = {(int(a), int(s)) for i in tr for a, s, k in zip(fa[i], ra[i], ma[i], strict=True) if k}
        keep = [
            i
            for i in te
            if all((int(a), int(s)) in seen for a, s, k in zip(fa[i], ra[i], ma[i], strict=True) if k)
        ]
        dropped = len(te) - len(keep)
        te = np.array(keep)
        U = z["U"].astype(np.float64)
        u_te = z["u"][te].astype(np.float64)
        t_host = (u_te @ U.T).argmax(1)
        F_sets = cc.filler_sets(fa[tr], ra[tr], ma[tr], n_role)
        summary["tasks"][task] = dict(n_train=len(tr), n_test=len(te), test_dropped_unseen_pair=dropped)
        log(f"{task}: train {len(tr)} test {len(te)} (dropped {dropped})")
        T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
        f_t, r_t, m_t, u_t, U_t = T(fa, torch.long), T(ra, torch.long), T(ma), T(z["u"]), T(z["U"])
        tr_t = torch.tensor(tr, device=device)
        for s_task, d_f in cc.SETTINGS:
            if s_task != task:
                continue
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
                    rng = np.random.default_rng(1000 + seed)
                    res = evaluate(p, rd, F_sets, u_te, fa[te], ra[te], ma[te], U, t_host, str(device), rng)
                    res["seed"] = seed
                    per_seed.append(res)
                    log(
                        f"{task} dF{d_f} {obj:4s} seed {seed}: E {res['E']} A {res['A']} "
                        f"F_cov {res['F_cov']} "
                        f"rho_loc/n {res.get('median_rho_loc_over_n')} "
                        f"host-bound {res.get('bottleneck_share')} "
                        f"ties {res.get('ties_clean')}/{res.get('ties_agree')} "
                        f"checks {res.get('neighbourhood_checks')}"
                    )
                summary["cells"][f"{task}/dF{d_f}/{obj}"] = dict(per_seed=per_seed, **summarise(per_seed))
    summary["verdict"] = decide(summary["cells"])
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
    rr.add_argument("--smoke", action="store_true", help="bug check: 30 steps; numbers are NOT results")
    args = ap.parse_args()
    {"dump": cmd_dump, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
