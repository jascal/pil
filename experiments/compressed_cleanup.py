"""Compressed binding (T4(c)): clean-up certificates without a left inverse.

Pre-registered: docs/notes/compressed_cleanup_prereg.md.

    # 1. dump (needs `transformers`):
    /path/to/venv/bin/python experiments/compressed_cleanup.py dump --out runs/compressed
    # 2. run (needs souffle), under a sleep inhibitor:
    systemd-inhibit --what=sleep:idle /path/to/venv/bin/python experiments/compressed_cleanup.py run \
        --dumps runs/compressed --out runs/compressed/summary.json

Theorem (i-orca#36, PIC_Cleanup.compressed_cleanup_certified, kernel-checked). For a linear decoder P with P
W != I, P(u - b0) = T(sigma) + E(sigma) + P n with E(sigma) = (P W - I) T(sigma). If |E| <= eps |T|, each
half-space slack loses at most eps |T| |w_s| |d_a| (compression_offset_bound), giving rho_eps = min (|d|^2/2
- eps |T| |w_s| |d|) / |M_s^T d|. Exact per-context clean-up is decided by the slacks of the decoded tensor
(nearest_iff_halfspace), with no eps needed.
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
import cleanup_certificate as cc  # noqa: E402
import cleanup_radius as cr  # noqa: E402
import halfspace_verdict as hv  # noqa: E402

SEEDS = dict(stimulus=111, split=112, fit=(0, 1, 2))
SETTINGS = (("LIST", 32), ("SVO", 32))
LAM_REL = 1e-3
SLACK = 1e-9


# ---------------------------------------------------------------- the decoder and its error


def ridge_decoder(W: np.ndarray, lam_rel: float = LAM_REL) -> np.ndarray:
    """P = (W^T W + lam I)^-1 W^T with lam = lam_rel * sigma_max(W)^2."""
    smax = float(np.linalg.svd(W, compute_uv=False).max())
    lam = lam_rel * smax**2
    return np.linalg.solve(W.T @ W + lam * np.eye(W.shape[1]), W.T)


def readout_ridge(p: dict, lam_rel: float = LAM_REL) -> dict:
    """The rd dict cc.cleanup / cr.role_maps expect, with the ridge decoder in place of W^+."""
    er = p["er"]
    cond = float(np.linalg.cond(er))
    no_dual = bool(not np.isfinite(cond) or cond > 1e8)
    return dict(
        P=ridge_decoder(p["W"], lam_rel),
        Wd=np.linalg.inv(er) if not no_dual else None,
        cond_er=cond,
        no_dual_readout=no_dual,
    )


def compression_error(p, P, f, r, m):
    """Per context: T(sigma) flattened, E(sigma) = (P W - I) T(sigma), and eps(sigma) = |E| / |T|."""
    T = cc.tensors(p, f, r, m).reshape(len(f), -1)
    E = T @ (P @ p["W"]).T - T
    nt = np.sqrt((T**2).sum(1))
    return T, E, np.sqrt((E**2).sum(1)) / np.maximum(nt, 1e-300), nt


# ---------------------------------------------------------------- slacks, the offset bound, radii


def decoded_slacks(p, rd, maps, F_sets, u, f, r, m, E, eps, nt):
    """Exact half-space slacks of the decoded tensor, the offset-bound check (abort), and per-context radii.
    Returns rows (c, s, a, slack), min slack, rho_dir, rho_eps."""
    d_f, n_role = p["ef"].shape[1], p["er"].shape[0]
    Tt = ((u - p["b0"]) @ rd["P"].T).reshape(len(u), d_f, n_role)
    Er = E.reshape(len(u), d_f, n_role)
    wn = np.sqrt((rd["Wd"] ** 2).sum(0))
    rows, min_slack = [], np.full(len(u), np.inf)
    rho_dir, rho_eps = np.full(len(u), np.inf), np.full(len(u), np.inf)
    for s, Fs in enumerate(F_sets):
        if len(Fs) < 2:
            continue
        Ef = p["ef"][Fs]
        G = maps[s] @ maps[s].T
        D = Ef[None, :, :] - Ef[:, None, :]  # D[b, a] = f_a - f_b
        dn2 = (D**2).sum(2)
        dn = np.sqrt(dn2)
        qn = np.sqrt(np.clip(np.einsum("bai,ij,baj->ba", D, G, D), 0, None))
        pos = {int(x): i for i, x in enumerate(Fs)}
        for j in range(f.shape[1]):
            idx = np.where(m[:, j].astype(bool) & (r[:, j] == s))[0]
            if len(idx) == 0:
                continue
            b = np.array([pos[int(x)] for x in f[idx, j]])
            g = Tt[idx] @ rd["Wd"][:, s]  # (k, d_F)
            off = g - Ef[b]  # g_s - f_sigma(s)
            sl = dn2[b] / 2 - (off @ Ef.T - (off * Ef[b]).sum(1, keepdims=True))
            eo = Er[idx] @ rd["Wd"][:, s]  # unbind(E, w_s)
            proj = eo @ Ef.T - (eo * Ef[b]).sum(1, keepdims=True)  # <unbind(E, w_s), d_a>
            bound = (eps[idx] * nt[idx] * wn[s])[:, None] * dn[b]
            other = np.ones_like(sl, dtype=bool)
            other[np.arange(len(idx)), b] = False
            if (np.abs(proj[other]) > bound[other] * (1 + SLACK) + 1e-12).any():
                raise AssertionError("compression_offset_bound violated (bug)")
            with np.errstate(divide="ignore", invalid="ignore"):
                rd_ = np.where(qn[b] > 0, (dn2[b] / 2) / np.where(qn[b] > 0, qn[b], 1), np.inf)
                re_ = np.where(qn[b] > 0, (dn2[b] / 2 - bound) / np.where(qn[b] > 0, qn[b], 1), np.inf)
            rd_[~other], re_[~other], sl[~other] = np.inf, np.inf, np.inf
            for kk, c in enumerate(idx):
                for a in range(len(Fs)):
                    if a != b[kk]:
                        rows.append((int(c), s, int(Fs[a]), float(sl[kk, a])))
            min_slack[idx] = np.minimum(min_slack[idx], sl.min(1))
            rho_dir[idx] = np.minimum(rho_dir[idx], rd_.min(1))
            rho_eps[idx] = np.minimum(rho_eps[idx], re_.min(1))
    if (rho_eps > rho_dir * (1 + SLACK) + 1e-12).any():
        raise AssertionError("rho_eps exceeds rho_dir (bug)")
    return rows, min_slack, rho_dir, rho_eps


# ---------------------------------------------------------------- evaluation and decision rule


def evaluate(p, F_sets, u, f, r, m, U, t_host, device, eps_train=None):
    rd = readout_ridge(p)
    if rd["no_dual_readout"]:
        return dict(flagged=True, F_cov=0.0)
    T, E, eps, nt = compression_error(p, rd["P"], f, r, m)
    x = cc.code_points(p, f, r, m)
    n = u - x
    t_code, beta = cc.beta_and_decision(x, U, device)
    maps = cr.role_maps(p, rd)
    rows, min_slack, rho_dir, rho_eps = decoded_slacks(p, rd, maps, F_sets, u, f, r, m, E, eps, nt)
    hmin, _, _ = hv.host_slacks(u, t_code, U, device)
    _, exact = cc.cleanup(u, p, rd, F_sets, f, r, m)
    host_agree = t_host == t_code
    v = hv.verdicts_souffle(len(u), rows, hmin)
    vp = hv.verdicts_python(len(u), rows, hmin)
    if any(not (v[k] == vp[k]).all() for k in v):
        raise AssertionError("Soufflé and Python verdicts differ")
    ties = hv.iff_check(v, exact, host_agree, min_slack, hmin)
    nn = np.sqrt((n**2).sum(1))
    ball = nn < np.minimum(rho_eps, beta)
    ok = rho_dir > 0
    return dict(
        flagged=False,
        F_cov=float(v["full"].mean()),
        E=float(exact.mean()),
        A=float(host_agree.mean()),
        eps_median=float(np.median(eps)),
        eps_max=float(eps.max()),
        eps_over_train=float((eps > eps_train).mean()) if eps_train is not None else None,
        radius_shrink_median=float(np.median(np.clip(rho_eps[ok], 0, None) / rho_dir[ok])),
        rho_eps_nonpositive=float((rho_eps <= 0).mean()),
        ball_cov=float(ball.mean()),
        median_rho_dir=float(np.median(rho_dir)),
        median_rho_eps=float(np.median(rho_eps)),
        median_n=float(np.median(nn)),
        ties=[ties["ties_clean"], ties["ties_agree"]],
        cond_er=rd["cond_er"],
    )


KEYS = (
    "F_cov",
    "E",
    "A",
    "eps_train",
    "eps_median",
    "eps_max",
    "eps_over_train",
    "radius_shrink_median",
    "rho_eps_nonpositive",
    "ball_cov",
    "median_rho_dir",
    "median_rho_eps",
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
    return cell


def decide(cells):
    lst = {k: c["F_cov"]["mean"] for k, c in cells.items() if k.startswith("LIST") and c.get("F_cov")}
    return dict(H1=dict(F_cov_LIST=lst, verdict="pass" if any(v >= 0.05 for v in lst.values()) else "fail"))


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
        prereg="docs/notes/compressed_cleanup_prereg.md",
        smoke=bool(args.smoke),
        seeds=dict(SEEDS, stimulus=999 if args.smoke else SEEDS["stimulus"]),
        lam_rel=LAM_REL,
        tasks={},
        cells={},
    )
    for task in ("LIST", "SVO"):
        z = np.load(Path(args.dumps) / f"{task}.npz")
        n_fill, n_role = int(z["n_fill"]), int(z["n_role"])
        fa, ra, ma = z["f"], z["r"], z["m"]
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
        T_ = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
        f_t, r_t, m_t, u_t, U_t = T_(fa, torch.long), T_(ra, torch.long), T_(ma), T_(z["u"]), T_(z["U"])
        tr_t = torch.tensor(tr, device=device)
        for s_task, d_f in SETTINGS:
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
                    P = ridge_decoder(p["W"])
                    eps_train = float(compression_error(p, P, fa[tr], ra[tr], ma[tr])[2].max())
                    res = evaluate(p, F_sets, u_te, fa[te], ra[te], ma[te], U, t_host, str(device), eps_train)
                    res.update(seed=seed, eps_train=eps_train)
                    per_seed.append(res)
                    log(
                        f"{task} dF{d_f} {obj:4s} seed {seed}: F_cov {res['F_cov']:.4f} E {res.get('E')} "
                        f"A {res.get('A')} eps_train {eps_train:.3g} eps_med {res.get('eps_median')} "
                        f"shrink {res.get('radius_shrink_median')} ties {res.get('ties')}"
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
