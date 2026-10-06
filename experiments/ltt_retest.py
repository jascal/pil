"""LTT re-test: many independent calibration draws per seed, each scored on a large audit set.

Pre-registered: docs/notes/ltt_retest_prereg.md. Corrects #149's design (one shared split; a grid starting at
the extreme top). Student, gate and the LTT test are #148/#149's, unchanged; the grid starts at q = 0.80.

    # 1. dump GPT-2 labels on the fresh stimulus seed (needs `transformers`):
    /path/to/venv/bin/python experiments/ltt_retest.py dump --out runs/ltt_retest
    # 2. train, calibrate R times and audit (torch + scipy), under a sleep inhibitor, from the repo:
    systemd-inhibit --what=sleep:idle /path/to/venv/bin/python experiments/ltt_retest.py run \
        --dumps runs/ltt_retest --out runs/ltt_retest/summary.json
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
import certified_student as cst  # noqa: E402
import conditional_certificate as cc  # noqa: E402
from certified_student import HELD_OUT, WORDS, evaluate  # noqa: E402
from ltt_certificate import ALPHA, DELTA, ltt_tau  # noqa: E402

SEEDS = dict(stimulus=181, split=182, fit=(50, 51, 52, 53, 54))
N_SIGMA, SMOKE_SIGMA = 14400, 1440
FRAC_TRAIN, FRAC_POOL = 2400 / 14400, 6000 / 14400  # audit is the rest
DRAW_FRAC = 0.3  # each calibration draw is 30% of the pool (1,800 of 6,000)
R, SMOKE_R = 200, 20
GRID_Q = tuple(round(0.80 - 0.01 * j, 2) for j in range(80))  # 0.80 .. 0.01, descending tau


def grid_from_train(train_scores):
    """Thresholds at train-score quantiles q = 0.80 .. 0.01 (descending tau); independent of calibration."""
    q = torch.tensor(GRID_Q, dtype=train_scores.dtype, device=train_scores.device)
    return torch.quantile(train_scores, q)


def audit_draw(grid, s_pool, c_pool, ok_pool, idx, s_aud, c_aud, ok_aud):
    """One calibration draw (pool indices idx): LTT threshold, then issue on the audit set.
    Returns dict(tau, issued (bool tensor over audit), cal_n, cal_w, coverage, FCa or None)."""
    chosen, tested = ltt_tau(grid, s_pool[idx], c_pool[idx], ok_pool[idx])
    if any(rej and p > DELTA for _, _, _, p, rej in tested):
        raise AssertionError("a rejected grid point has p > delta (bug)")
    if chosen is None:
        return dict(tau=None, issued=torch.zeros_like(c_aud), cal_n=0, cal_w=0, coverage=0.0, FCa=None)
    iss = (s_aud >= chosen[0]) & c_aud
    fca = float((~ok_aud[iss]).float().mean()) if iss.any() else None
    return dict(
        tau=chosen[0],
        issued=iss,
        cal_n=chosen[1],
        cal_w=chosen[2],
        coverage=float(iss.float().mean()),
        FCa=fca,
    )


def is_violation(fca, alpha=ALPHA):
    return fca is not None and fca > alpha


def seed_stats(draws):
    n = len(draws)
    issuing = [d for d in draws if d["FCa"] is not None]
    return dict(
        draws=n,
        violation_rate=sum(is_violation(d["FCa"]) for d in draws) / n,
        coverage=float(np.mean([d["coverage"] for d in draws])),
        non_issuance=sum(d["tau"] is None for d in draws) / n,
        FCa_mean=float(np.mean([d["FCa"] for d in issuing])) if issuing else None,
        faith_issued_mean=float(np.mean([d["faith_issued"] for d in issuing])) if issuing else None,
        held_faith_issued_mean=float(np.mean([d["held_faith_issued"] for d in issuing])) if issuing else None,
    )


def decide(pooled):
    return dict(
        H1=dict(
            violation_rate=pooled["violation_rate"],
            verdict="pass" if pooled["violation_rate"] <= 0.15 else "fail",
        ),
        H2=dict(coverage=pooled["coverage"], verdict="pass" if pooled["coverage"] >= 0.25 else "fail"),
        H3=dict(
            non_issuance=pooled["non_issuance"], verdict="pass" if pooled["non_issuance"] <= 0.10 else "fail"
        ),
    )


def cmd_dump(args):
    cst.SEEDS = dict(cst.SEEDS, stimulus=SEEDS["stimulus"])  # cst.cmd_dump reads these module globals
    cst.N_SIGMA, cst.SMOKE_SIGMA = N_SIGMA, SMOKE_SIGMA
    cst.cmd_dump(args)


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = cst.SMOKE_STEPS if args.smoke else cst.STEPS
    gsteps = cc.SMOKE_GATE_STEPS if args.smoke else cc.GATE_STEPS
    n_draws = SMOKE_R if args.smoke else R
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
    n_tr, n_po = round(FRAC_TRAIN * n_cls), round(FRAC_POOL * n_cls)
    tr = torch.tensor(order[:n_tr], device=device)
    pool = torch.tensor(order[n_tr : n_tr + n_po], device=device)
    aud = torch.tensor(order[n_tr + n_po :], device=device)
    n_draw = round(DRAW_FRAC * n_po)
    outs = sorted(set(dec[tr][:, seen_ix].flatten().tolist()))
    omap = {t: i for i, t in enumerate(outs)}
    gold = torch.tensor([[omap.get(int(t), -1) for t in row] for row in dec.tolist()], device=device)
    seen_const = (dec[:, seen_ix] == dec[:, seen_ix[:1]]).all(1)
    all_const = (dec == dec[:, seen_ix[:1]]).all(1)
    tr_const = tr[seen_const[tr]]
    summary = dict(
        tag="empirical",
        prereg="docs/notes/ltt_retest_prereg.md",
        smoke=bool(args.smoke),
        seeds=dict(SEEDS, stimulus=999 if args.smoke else SEEDS["stimulus"]),
        held_out=list(HELD_OUT),
        alpha=ALPHA,
        delta=DELTA,
        draws_per_seed=n_draws,
        grid_q=[GRID_Q[0], GRID_Q[-1]],
        data=dict(
            classes=n_cls,
            train=len(tr),
            pool=len(pool),
            draw=n_draw,
            audit=len(aud),
            outputs=len(outs),
            pool_seen_constant=float(seen_const[pool].float().mean()),
            audit_seen_constant=float(seen_const[aud].float().mean()),
        ),
    )
    log(
        f"{n_cls} classes: train {len(tr)} pool {len(pool)} (draws of {n_draw}) audit {len(aud)}; "
        f"outputs {len(outs)}; audit seen-constant {summary['data']['audit_seen_constant']:.3f}"
    )
    E = torch.tensor(z["wte"], device=device)
    per_seed, all_draws = [], []
    for seed in SEEDS["fit"]:
        m = cc.train_student(E, len(outs), T, sid, gold, tr, tr_const, seen_ix, opt_seen, seed, steps, device)
        r = evaluate(
            m, sid, gold, aud, seen_ix, held_ix, opt_all, opt_seen
        )  # soundness aborts + #146 metrics
        gate = cc.train_gate(E, T, sid, seen_const.float(), tr, seen_ix, seed, gsteps, device)
        s_tr, s_po, s_au = (cc.gate_scores(gate, sid, x) for x in (tr, pool, aud))
        c_po, _ = cc.cert16_and_decision(m, sid, pool, seen_ix, opt_seen)
        c_au, p_au = cc.cert16_and_decision(m, sid, aud, seen_ix, opt_seen)
        grid = grid_from_train(s_tr)
        rng = np.random.default_rng([SEEDS["split"], seed])
        draws = []
        for _ in range(n_draws):
            idx = torch.tensor(rng.choice(len(pool), n_draw, replace=False), device=device)
            d = audit_draw(grid, s_po, c_po, seen_const[pool], idx, s_au, c_au, seen_const[aud])
            im = cc.issued_metrics(
                d.pop("issued"), p_au, gold[aud], seen_ix, held_ix, seen_const[aud], all_const[aud]
            )
            d["faith_issued"], d["held_faith_issued"] = im["faith_issued"], im["held_faith_issued"]
            draws.append(d)
        st = seed_stats(draws)
        st.update(
            seed=seed,
            faith_seen=r["faith_seen"],
            cert16=r["cert16"],
            gate_auc=cc.auc(s_au, seen_const[aud]),
            first_draw=dict(draws[0]),
        )
        per_seed.append(st)
        all_draws += draws
        log(
            f"seed {seed}: faith {st['faith_seen']:.3f} cert16 {st['cert16']:.3f} "
            f"gate AUC {st['gate_auc']:.3f} | "
            f"violations {st['violation_rate']:.3f} coverage {st['coverage']:.3f} "
            f"non-issuance {st['non_issuance']:.3f} FCa {st['FCa_mean']} "
            f"faith|issued {st['faith_issued_mean']}"
        )
    pooled = seed_stats(all_draws)
    summary["per_seed"], summary["pooled"] = per_seed, pooled
    summary["verdict"] = decide(pooled)
    summary["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["verdict"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--smoke", action="store_true", help="bug check: stimulus 999, 1,440 sigma")
    rr = sub.add_parser("run")
    rr.add_argument("--dumps", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument(
        "--smoke", action="store_true", help="bug check: 300 steps, R = 20; numbers are NOT results"
    )
    args = ap.parse_args()
    {"dump": cmd_dump, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
