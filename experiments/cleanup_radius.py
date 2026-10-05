"""How much exact clean-up does the directional radius certify?

Pre-registered: docs/notes/cleanup_radius_prereg.md.

    # 1. dump (needs `transformers`):
    /path/to/venv/bin/python experiments/cleanup_radius.py dump --out runs/cleanup_radius
    # 2. run (needs souffle):
    /path/to/venv/bin/python experiments/cleanup_radius.py run --dumps runs/cleanup_radius \
        --out runs/cleanup_radius/summary.json

Theorem (i-orca#32, PIC_Cleanup, kernel-checked). With dual role readouts, role s of a residual x(sigma) + n
snaps to sigma(s) iff <M_s n, d_a> < |d_a|^2 / 2 for every rival filler a, where d_a = f_a - f_sigma(s) and
M_s n = unbind(P n, w_s). The largest ball of n inside that region has radius
    rho_dir(sigma) = min_{s in D(sigma)} min_{a != sigma(s)} (|d_a|^2 / 2) / |M_s^T d_a|,
which is exact per half-space and never below the worst case rho = min_s gamma_s / (2 K |w_s|).

Data, training, readouts, clean-up and beta are cleanup_certificate.py's (#139), unchanged; only the seeds
differ.
"""

from __future__ import annotations

import argparse
import json
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

SEEDS = dict(stimulus=51, split=52, fit=(0, 1, 2))
SCALE, TAU, BIG = cc.SCALE, cc.TAU, cc.BIG

SOUFFLE = """
// dir: |n| < rho_dir; wc: |n| < rho; full: |n| < min(rho_dir, beta). Fixed point: |n| ceiled, radii floored.
.decl n(c:number, x:number) .input n
.decl rhod(c:number, x:number) .input rhod
.decl rhow(c:number, x:number) .input rhow
.decl beta(c:number, x:number) .input beta
.decl tau(t:number) .input tau
.decl dir(c:number) .output dir
dir(c) :- n(c, a), rhod(c, r), tau(t), a + t < r.
.decl wc(c:number) .output wc
wc(c) :- n(c, a), rhow(c, r), tau(t), a + t < r.
.decl full(c:number) .output full
full(c) :- n(c, a), rhod(c, r), beta(c, b), tau(t), a + t < r, a + t < b.
.decl context(c:number) .output context
context(c) :- n(c, _).
"""


# ---------------------------------------------------------------- the directional radius


def role_maps(p: dict, rd: dict) -> list[np.ndarray]:
    """M_s (d_F x d): n -> unbind(P n, w_s) = sum_j (w_s)_j P[:, j, :] n."""
    d_f, n_role = p["ef"].shape[1], p["er"].shape[0]
    P3 = rd["P"].reshape(d_f, n_role, -1)
    return [np.einsum("ijk,j->ik", P3, rd["Wd"][:, s]) for s in range(n_role)]


def directional_tables(p: dict, maps: list[np.ndarray], F_sets) -> list[np.ndarray]:
    """For each role s, the table r_s[b, a] = (|f_a - f_b|^2 / 2) / |M_s^T (f_a - f_b)| over positions in F_s
    (inf on the diagonal, and where M_s^T d = 0: such a rival cannot be reached by any noise)."""
    out = []
    for s, Fs in enumerate(F_sets):
        E = p["ef"][Fs]
        D = E[None, :, :] - E[:, None, :]  # D[b, a] = f_a - f_b
        G = maps[s] @ maps[s].T  # |M^T d|^2 = d^T (M M^T) d
        qn = np.sqrt(np.clip(np.einsum("bai,ij,baj->ba", D, G, D), 0, None))
        num = (D**2).sum(2) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(qn > 0, num / np.where(qn > 0, qn, 1), np.inf)
        np.fill_diagonal(r, np.inf)
        out.append(r)
    return out


def rho_dir(tables, F_sets, f, r, m) -> np.ndarray:
    """rho_dir(sigma) = min over bound roles s of min_{a != sigma(s)} r_s[sigma(s), a]."""
    rowmin = [t.min(1) for t in tables]
    pos = [{int(x): i for i, x in enumerate(Fs)} for Fs in F_sets]
    out = np.full(len(f), np.inf)
    for i in range(len(f)):
        for j in range(f.shape[1]):
            if m[i, j]:
                s = int(r[i, j])
                out[i] = min(out[i], rowmin[s][pos[s][int(f[i, j])]])
    return out


# ---------------------------------------------------------------- verdicts


def _fixed(n, rd_, rw, be, tau=TAU):
    a, r1, b, t = cc._fixed(n, rd_, be, tau)
    _, r2, _, _ = cc._fixed(n, rw, be, tau)
    return a, r1, r2, b, t


def verdicts_python(n, rd_, rw, be, tau=TAU) -> dict:
    a, r1, r2, b, t = _fixed(n, rd_, rw, be, tau)
    return {"dir": a + t < r1, "wc": a + t < r2, "full": (a + t < r1) & (a + t < b)}


def verdicts_souffle(n, rd_, rw, be, tau=TAU) -> dict:
    a, r1, r2, b, t = _fixed(n, rd_, rw, be, tau)
    with tempfile.TemporaryDirectory(prefix="cleanup-radius-") as raw:
        d = Path(raw)
        (d / "radius.dl").write_text(SOUFFLE)
        for name, vals in (("n", a), ("rhod", r1), ("rhow", r2), ("beta", b)):
            (d / f"{name}.facts").write_text("".join(f"{i}\t{v}\n" for i, v in enumerate(vals.tolist())))
        (d / "tau.facts").write_text(f"{t}\n")
        subprocess.run(
            ["souffle", "-F", str(d), "-D", str(d), str(d / "radius.dl")],
            check=True,
            capture_output=True,
            text=True,
        )
        read = lambda nm: {int(x) for x in (d / f"{nm}.csv").read_text().split()}  # noqa: E731
        if read("context") != set(range(len(a))):
            raise RuntimeError("Soufflé did not account for every context")
        return {k: np.array([i in read(k) for i in range(len(a))]) for k in ("dir", "wc", "full")}


def soundness(v: dict, exact, t_code, t_host, rd_, rw) -> None:
    """Theorems: dir/wc-certified => exact clean-up; full => exact AND host agrees; rho_dir >= rho."""
    errs = []
    for k in ("dir", "wc", "full"):
        if (v[k] & ~exact).any():
            errs.append(f"{int((v[k] & ~exact).sum())} {k}-certified contexts with inexact clean-up")
    if (v["full"] & (t_code != t_host)).any():
        errs.append(
            f"{int((v['full'] & (t_code != t_host)).sum())} full-certified contexts disagree with the host"
        )
    if (rd_ < rw * (1 - 1e-9)).any():
        errs.append(f"{int((rd_ < rw * (1 - 1e-9)).sum())} contexts with rho_dir < rho")
    if errs:
        raise AssertionError("unsound: " + "; ".join(errs))


# ---------------------------------------------------------------- evaluation and decision rules


def evaluate(p, rd, F_sets, gam, u_te, f_te, r_te, m_te, U, t_host, device):
    x = cc.code_points(p, f_te, r_te, m_te)
    n = np.sqrt(((u_te - x) ** 2).sum(1))
    flags = dict(
        no_left_inverse=rd["no_left_inverse"],
        no_dual_readout=rd["no_dual_readout"],
        sigma_ratio=rd["sigma_ratio"],
        pinv_err=rd["pinv_err"],
        cond_er=rd["cond_er"],
    )
    if rd["no_left_inverse"] or rd["no_dual_readout"]:
        return dict(flagged=True, flags=flags, C_dir=0.0, C_wc=0.0, full=0.0, E=None, S=None, gain=None)
    t_code, be = cc.beta_and_decision(x, U, device)
    rw = cc.rho(r_te, m_te, gam, rd["K"], np.sqrt((rd["Wd"] ** 2).sum(0)))
    rd_ = rho_dir(directional_tables(p, role_maps(p, rd), F_sets), F_sets, f_te, r_te, m_te)
    _, exact = cc.cleanup(u_te, p, rd, F_sets, f_te, r_te, m_te)
    v = verdicts_souffle(n, rd_, rw, be)
    vp = verdicts_python(n, rd_, rw, be)
    if any(not (v[k] == vp[k]).all() for k in v):
        raise AssertionError("Soufflé and Python verdicts differ")
    soundness(v, exact, t_code, t_host, rd_, rw)
    E = float(exact.mean())
    unc = ~v["full"]
    return dict(
        flagged=False,
        flags=flags,
        E=E,
        C_dir=float(v["dir"].mean()),
        C_wc=float(v["wc"].mean()),
        S=float(v["dir"].mean() / E) if E > 0 else None,
        gain=float(np.median(rd_ / rw)),
        full=float(v["full"].mean()),
        share_n_ge_rho_dir=float((n[unc] >= rd_[unc]).mean()) if unc.any() else None,
        share_n_ge_beta=float((n[unc] >= be[unc]).mean()) if unc.any() else None,
        median_n=float(np.median(n)),
        median_rho=float(np.median(rw)),
        median_rho_dir=float(np.median(rd_)),
        median_beta=float(np.median(be)),
    )


KEYS = (
    "E",
    "C_dir",
    "C_wc",
    "S",
    "gain",
    "full",
    "share_n_ge_rho_dir",
    "share_n_ge_beta",
    "median_n",
    "median_rho",
    "median_rho_dir",
    "median_beta",
)


def summarise(per_seed: list[dict]) -> dict:
    cell = {}
    for k in KEYS:
        vals = [s[k] for s in per_seed if s.get(k) is not None]
        cell[k] = (
            dict(mean=float(np.mean(vals)), range=[float(min(vals)), float(max(vals))]) if vals else None
        )
    cell["flagged_seeds"] = sum(bool(s["flagged"]) for s in per_seed)
    return cell


def decide(cells: dict) -> dict:
    mean = lambda c, k: c[k]["mean"] if c[k] else None  # noqa: E731
    eligible = {k: c for k, c in cells.items() if (mean(c, "E") or 0) >= 0.2}
    h1_cells = {
        k: dict(E=mean(c, "E"), S=mean(c, "S"), passes=bool((mean(c, "S") or 0) >= 0.5))
        for k, c in eligible.items()
    }
    if not h1_cells:
        h1 = dict(cells={}, verdict="untestable")
    else:
        n_pass = sum(v["passes"] for v in h1_cells.values())
        h1 = dict(
            cells=h1_cells,
            n_pass=n_pass,
            n_eligible=len(h1_cells),
            verdict="pass" if n_pass >= len(h1_cells) / 2 else "fail",
        )
    h2_settings = {}
    for task, d_f in cc.SETTINGS:
        gains = [mean(cells[f"{task}/dF{d_f}/{o}"], "gain") for o in cc.OBJECTIVES]
        gains = [g for g in gains if g is not None]
        med = float(np.median(gains)) if gains else None
        h2_settings[f"{task}/dF{d_f}"] = dict(median_gain=med, passes=bool(med is not None and med >= 10))
    n2 = sum(v["passes"] for v in h2_settings.values())
    return dict(H1=h1, H2=dict(settings=h2_settings, n_pass=n2, verdict="pass" if n2 >= 2 else "fail"))


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
        prereg="docs/notes/cleanup_radius_prereg.md",
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
                    gam = cc.gammas(p["ef"], F_sets)
                    res = evaluate(p, rd, F_sets, gam, u_te, fa[te], ra[te], ma[te], U, t_host, str(device))
                    res["seed"] = seed
                    per_seed.append(res)
                    log(
                        f"{task} dF{d_f} {obj:4s} seed {seed}: E {res['E']} C_dir {res['C_dir']:.4f} "
                        f"C_wc {res['C_wc']:.4f} S {res['S']} gain {res['gain']} full {res['full']:.4f} "
                        f"|n| {res.get('median_n')} rho_dir {res.get('median_rho_dir')} "
                        f"flagged {res['flagged']}"
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
