"""Calibrated conditional certificates: Learn-then-Test picks the gate threshold so that, with probability
>= 1 - delta over the calibration draw, the false-certificate rate among issued certificates is <= alpha.

Pre-registered: docs/notes/ltt_certificate_prereg.md. Student and gate are #148's
(experiments/conditional_certificate.py), unchanged; only the threshold rule and the split differ.

    # 1. dump GPT-2 labels on the fresh stimulus seed (needs `transformers`):
    /path/to/venv/bin/python experiments/ltt_certificate.py dump --out runs/ltt
    # 2. train, calibrate and evaluate (torch + scipy), under a sleep inhibitor, from the repo:
    systemd-inhibit --what=sleep:idle /path/to/venv/bin/python experiments/ltt_certificate.py run \
        --dumps runs/ltt --out runs/ltt/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_student as cst  # noqa: E402
import conditional_certificate as cc  # noqa: E402
from certified_student import HELD_OUT, WORDS, evaluate  # noqa: E402

SEEDS = dict(stimulus=171, split=172, fit=(40, 41, 42, 43, 44))
N_SIGMA, SMOKE_SIGMA = 6000, 300
FRAC_TRAIN, FRAC_CAL = 0.4, 0.3  # test is the rest (0.3)
ALPHA, DELTA = 0.10, 0.10
GRID_Q = tuple(round(0.95 - 0.01 * j, 2) for j in range(95))  # 0.95 .. 0.01, descending tau


def grid_from_train(train_scores):
    """Thresholds at train-score quantiles q = 0.95 .. 0.01 (descending tau); independent of calibration."""
    q = torch.tensor(GRID_Q, dtype=train_scores.dtype, device=train_scores.device)
    return torch.quantile(train_scores, q)


def p_value(n, w, alpha=ALPHA):
    """Valid p-value for H0: FC >= alpha, given n issued with w wrong (binomial lower tail); 1 if n == 0."""
    return 1.0 if n == 0 else float(binom.cdf(w, n, alpha))


def ltt_tau(grid, score, cert, ok, alpha=ALPHA, delta=DELTA):
    """Fixed-sequence Learn-then-Test down a descending grid. Returns (chosen, steps): steps lists
    (tau, n, w, p, rejected) for every grid point tested (testing stops at the first non-rejection); chosen is
    the step of the lowest rejected tau, or None if the first point is not rejected."""
    steps, chosen = [], None
    for t in grid.tolist():
        iss = (score >= t) & cert
        n, w = int(iss.sum()), int((iss & ~ok).sum())
        p = p_value(n, w, alpha)
        rej = p <= delta
        steps.append((t, n, w, p, rej))
        if not rej:
            break
        chosen = steps[-1]
    return chosen, steps


KEYS = cc.KEYS + ("coverage_emp", "FCi_emp")


def summarise(per_seed):
    cell = {}
    for k in KEYS:
        vals = [s[k] for s in per_seed if s.get(k) is not None]
        cell[k] = (
            dict(mean=float(np.mean(vals)), range=[float(min(vals)), float(max(vals))]) if vals else None
        )
    return cell


def decide(per_seed, alpha=ALPHA):
    viol = sum(1 for s in per_seed if s["FCi"] is not None and s["FCi"] > alpha)
    cov = [s["coverage"] for s in per_seed]
    mean_cov, spread = float(np.mean(cov)), float(max(cov) - min(cov))
    return dict(
        H1=dict(violations=viol, seeds=len(per_seed), verdict="pass" if viol <= 1 else "fail"),
        H2=dict(coverage=mean_cov, verdict="pass" if mean_cov >= 0.20 else "fail"),
        H3=dict(coverage_range=spread, verdict="pass" if spread <= 0.10 else "fail"),
    )


def cmd_dump(args):
    cst.SEEDS = dict(cst.SEEDS, stimulus=SEEDS["stimulus"])  # cst.cmd_dump reads these module globals
    cst.N_SIGMA, cst.SMOKE_SIGMA = N_SIGMA, SMOKE_SIGMA
    cst.cmd_dump(args)


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = cst.SMOKE_STEPS if args.smoke else cst.STEPS
    gsteps = cc.SMOKE_GATE_STEPS if args.smoke else cc.GATE_STEPS
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    z = np.load(Path(args.dumps) / "student.npz")
    sid = torch.tensor(z["sid"], device=device)
    dec = torch.tensor(z["dec"], device=device)
    n_cls, k, T = sid.shape
    seen_ix = torch.tensor([i for i, w in enumerate(WORDS) if w not in HELD_OUT], device=device)
    held_ix = torch.tensor([i for i, w in enumerate(WORDS) if w in HELD_OUT], device=device)
    word_ids = torch.tensor(z["word_ids"], device=device)
    opt_all, opt_seen = word_ids, word_ids[seen_ix]
    order = np.random.default_rng(SEEDS["split"]).permutation(n_cls)
    n_tr, n_ca = int(FRAC_TRAIN * n_cls), int(FRAC_CAL * n_cls)
    tr = torch.tensor(order[:n_tr], device=device)
    ca = torch.tensor(order[n_tr : n_tr + n_ca], device=device)
    te = torch.tensor(order[n_tr + n_ca :], device=device)
    outs = sorted(set(dec[tr][:, seen_ix].flatten().tolist()))
    omap = {t: i for i, t in enumerate(outs)}
    gold = torch.tensor([[omap.get(int(t), -1) for t in row] for row in dec.tolist()], device=device)
    seen_const = (dec[:, seen_ix] == dec[:, seen_ix[:1]]).all(1)
    all_const = (dec == dec[:, seen_ix[:1]]).all(1)
    tr_const = tr[seen_const[tr]]
    summary = dict(
        tag="empirical",
        prereg="docs/notes/ltt_certificate_prereg.md",
        smoke=bool(args.smoke),
        seeds=dict(SEEDS, stimulus=999 if args.smoke else SEEDS["stimulus"]),
        held_out=list(HELD_OUT),
        alpha=ALPHA,
        delta=DELTA,
        data=dict(
            classes=n_cls,
            train=len(tr),
            cal=len(ca),
            test=len(te),
            outputs=len(outs),
            test_seen_constant=float(seen_const[te].float().mean()),
            cal_seen_constant=float(seen_const[ca].float().mean()),
            test_all_constant=float(all_const[te].float().mean()),
        ),
    )
    log(
        f"{n_cls} classes: train {len(tr)} cal {len(ca)} test {len(te)}; outputs {len(outs)}; "
        f"test seen-constant {summary['data']['test_seen_constant']:.3f}"
    )
    E = torch.tensor(z["wte"], device=device)
    per_seed = []
    for seed in SEEDS["fit"]:
        m = cc.train_student(E, len(outs), T, sid, gold, tr, tr_const, seen_ix, opt_seen, seed, steps, device)
        r = evaluate(m, sid, gold, te, seen_ix, held_ix, opt_all, opt_seen)  # soundness aborts + #146 metrics
        gate = cc.train_gate(E, T, sid, seen_const.float(), tr, seen_ix, seed, gsteps, device)
        s_tr, s_ca, s_te = (cc.gate_scores(gate, sid, x) for x in (tr, ca, te))
        c_ca, _ = cc.cert16_and_decision(m, sid, ca, seen_ix, opt_seen)
        c_te, p_te = cc.cert16_and_decision(m, sid, te, seen_ix, opt_seen)
        grid = grid_from_train(s_tr)
        chosen, tested = ltt_tau(grid, s_ca, c_ca, seen_const[ca])
        tau = chosen[0] if chosen else None
        if any(rej and p > DELTA for _, _, _, p, rej in tested):
            raise AssertionError("a rejected grid point has p > delta (bug)")
        issued = (s_te >= tau) & c_te if tau is not None else torch.zeros_like(c_te)
        out = dict(
            seed=seed,
            tau=tau,
            rejected=sum(rej for *_, rej in tested),
            cal_n=chosen[1] if chosen else 0,
            cal_w=chosen[2] if chosen else 0,
            faith_seen=r["faith_seen"],
            cert16=r["cert16"],
            FC16_ungated=float((~seen_const[te][c_te]).float().mean()) if c_te.any() else None,
            gate_auc=cc.auc(s_te, seen_const[te]),
        )
        out.update(cc.issued_metrics(issued, p_te, gold[te], seen_ix, held_ix, seen_const[te], all_const[te]))
        tau_e = cc.choose_tau(s_ca, c_ca, seen_const[ca])  # #148's empirical rule, for comparison only
        iss_e = (s_te >= tau_e) & c_te if tau_e is not None else torch.zeros_like(c_te)
        emp = cc.issued_metrics(iss_e, p_te, gold[te], seen_ix, held_ix, seen_const[te], all_const[te])
        out["coverage_emp"], out["FCi_emp"] = emp["coverage"], emp["FCi"]
        per_seed.append(out)
        log(
            f"seed {seed}: faith {out['faith_seen']:.3f} cert16 {out['cert16']:.3f} "
            f"gate AUC {out['gate_auc']} | "
            f"LTT tau {tau} ({out['rejected']} rejected; cal n {out['cal_n']} w {out['cal_w']}) coverage "
            f"{out['coverage']:.3f} FCi {out['FCi']} faith|issued {out['faith_issued']} | empirical rule: "
            f"coverage {out['coverage_emp']:.3f} FCi {out['FCi_emp']}"
        )
    summary["cell"] = dict(per_seed=per_seed, **summarise(per_seed))
    summary["verdict"] = decide(per_seed)
    summary["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["verdict"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--smoke", action="store_true", help="bug check: stimulus 999, 300 sigma")
    rr = sub.add_parser("run")
    rr.add_argument("--dumps", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument("--smoke", action="store_true", help="bug check: 300 steps; numbers are NOT results")
    args = ap.parse_args()
    {"dump": cmd_dump, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
