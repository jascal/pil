"""Conditional certificates: issue the stable student's invariance certificate only where a word-blind gate
predicts GPT-2 is invariant.

Pre-registered: docs/notes/conditional_certificate_prereg.md. The student is #147's `ibp_clip` recipe
(experiments/stable_student.py); the gate is the same architecture over the sequence with the nuisance
position removed, so it is invariant to the nuisance word by construction.

    # 1. dump GPT-2 labels on the fresh stimulus seed (needs `transformers`):
    /path/to/venv/bin/python experiments/conditional_certificate.py dump --out runs/cond
    # 2. train and evaluate (torch only), under a sleep inhibitor, from the repo:
    systemd-inhibit --what=sleep:idle /path/to/venv/bin/python experiments/conditional_certificate.py run \
        --dumps runs/cond --out runs/cond/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_student as cst  # noqa: E402
from certified_student import HELD_OUT, WORDS, certified, evaluate, worst_logits  # noqa: E402
from stable_student import CLIP, Student  # noqa: E402

SEEDS = dict(stimulus=161, split=162, fit=(30, 31, 32, 33, 34))
STEPS, SMOKE_STEPS, BATCH, LR, IBP_W = cst.STEPS, cst.SMOKE_STEPS, cst.BATCH, cst.LR, cst.IBP_W
GATE_STEPS, SMOKE_GATE_STEPS = 3000, 300
TARGET_PREC, MIN_ISSUED = 0.95, 20
SMOKE_MIN_ISSUED = 5  # smoke only: its 20-class validation split could never reach MIN_ISSUED


def train_student(E, n_out, T, sid, gold, tr, tr_const, seen_ix, opt_seen, seed, steps, device):
    """#147's ibp_clip recipe, unchanged."""
    torch.manual_seed(seed)
    m = Student(E, n_out, T).to(device)
    o = torch.optim.Adam(m.parameters(), lr=LR)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for step in range(steps):
        c = tr[torch.randint(len(tr), (BATCH,), generator=g).to(device)]
        w = seen_ix[torch.randint(len(seen_ix), (BATCH,), generator=g).to(device)]
        yb = gold[c, w]
        keep = yb >= 0
        loss = F.cross_entropy(m(sid[c, w][keep]), yb[keep])
        if len(tr_const):
            cc = tr_const[torch.randint(len(tr_const), (BATCH,), generator=g).to(device)]
            yc = gold[cc, seen_ix[0]]
            kc = yc >= 0
            alpha = min(1.0, step / (0.5 * steps))
            lo, hi = m.ibp(sid[cc, seen_ix[0]][kc], opt_seen, alpha)
            loss = (1 - IBP_W * alpha) * loss + IBP_W * alpha * F.cross_entropy(
                worst_logits(lo, hi, yc[kc]), yc[kc]
            )
        o.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
        o.step()
    return m


def blind(sid_rows):
    """The gate's input: the sequence with the nuisance position (0) removed."""
    return sid_rows[..., 1:]


def train_gate(E, T, sid, target, tr, seen_ix, seed, steps, device):
    torch.manual_seed(seed)
    m = Student(E, 1, T - 1).to(device)
    o = torch.optim.Adam(m.parameters(), lr=LR)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        c = tr[torch.randint(len(tr), (BATCH,), generator=g).to(device)]
        loss = F.binary_cross_entropy_with_logits(m(blind(sid[c, seen_ix[0]])).squeeze(1), target[c])
        o.zero_grad()
        loss.backward()
        o.step()
    return m


@torch.no_grad()
def gate_scores(gate, sid, cls):
    """Gate logit per class. Aborts unless the gate's integer input is identical across all nuisance variants,
    which makes its output invariant to the nuisance word by construction."""
    xb = blind(sid[cls])
    if not torch.equal(xb, xb[:, :1].expand_as(xb)):
        raise AssertionError("gate input differs across nuisance variants (bug)")
    return gate(xb[:, 0]).squeeze(1)


@torch.no_grad()
def cert16_and_decision(m, sid, cls, seen_ix, opt_seen):
    x = sid[cls]
    p0 = m(x[:, seen_ix[0]]).argmax(1)
    lo, hi = m.ibp(x[:, 0], opt_seen, 1.0)
    return certified(lo, hi, p0), p0


def choose_tau(score, cert, ok, target=TARGET_PREC, min_n=MIN_ISSUED):
    """Lowest tau with precision(ok | score >= tau & cert) >= target on >= min_n classes; None if none."""
    cand = score[cert]
    okc = ok[cert].float()
    if len(cand) < min_n:
        return None
    order = torch.argsort(cand, descending=True)
    s_sorted, ok_sorted = cand[order], okc[order]
    prec = torch.cumsum(ok_sorted, 0) / torch.arange(1, len(cand) + 1, device=cand.device)
    best = None
    for i in range(min_n - 1, len(cand)):
        if i + 1 < len(cand) and s_sorted[i + 1] == s_sorted[i]:
            continue  # tau must include all ties
        if prec[i] >= target:
            best = float(s_sorted[i])
    return best


def auc(score, y):
    o = torch.argsort(score, descending=True)
    yy = y[o].float()
    P, N = yy.sum(), len(yy) - yy.sum()
    if P == 0 or N == 0:
        return None
    return float(((1 - yy) * torch.cumsum(yy, 0)).sum() / (P * N))


def issued_metrics(issued, p0, gold, seen_ix, held_ix, seen_const, held_const):
    n = len(issued)
    cov = float(issued.float().mean()) if n else 0.0
    if not issued.any():
        return dict(
            coverage=cov,
            n_issued=0,
            FCi=None,
            faith_issued=None,
            held_const_issued=None,
            held_faith_issued=None,
        )
    g = gold[issued]
    d = p0[issued][:, None]
    return dict(
        coverage=cov,
        n_issued=int(issued.sum()),
        FCi=float((~seen_const[issued]).float().mean()),
        faith_issued=float((g[:, seen_ix] == d).all(1).float().mean()),
        held_const_issued=float(held_const[issued].float().mean()),
        held_faith_issued=float((g[:, held_ix] == d).all(1).float().mean()),
    )


KEYS = (
    "coverage",
    "FCi",
    "faith_issued",
    "held_const_issued",
    "held_faith_issued",
    "FC16_ungated",
    "gate_auc",
    "faith_seen",
    "cert16",
)


def summarise(per_seed):
    cell = {}
    for k in KEYS:
        vals = [s[k] for s in per_seed if s.get(k) is not None]
        cell[k] = (
            dict(mean=float(np.mean(vals)), range=[float(min(vals)), float(max(vals))]) if vals else None
        )
    return cell


def decide(cell):
    cov = cell["coverage"]["mean"] if cell.get("coverage") else 0.0
    fc = cell["FCi"]["mean"] if cell.get("FCi") else None
    fi = cell["faith_issued"]["mean"] if cell.get("faith_issued") else None
    return dict(
        H1=dict(FCi=fc, verdict="untestable" if fc is None else ("pass" if fc <= 0.10 else "fail")),
        H2=dict(coverage=cov, verdict="pass" if cov >= 0.20 else "fail"),
        H3=dict(faith_issued=fi, verdict="untestable" if fi is None else ("pass" if fi >= 0.90 else "fail")),
    )


def cmd_dump(args):
    cst.SEEDS = dict(cst.SEEDS, stimulus=SEEDS["stimulus"])  # cst.cmd_dump reads its module's stimulus seed
    cst.cmd_dump(args)


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = SMOKE_STEPS if args.smoke else STEPS
    gsteps = SMOKE_GATE_STEPS if args.smoke else GATE_STEPS
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
    n_tr, n_va = int(0.6 * n_cls), int(0.1 * n_cls)
    tr = torch.tensor(order[:n_tr], device=device)
    va = torch.tensor(order[n_tr : n_tr + n_va], device=device)
    te = torch.tensor(order[n_tr + n_va :], device=device)
    outs = sorted(set(dec[tr][:, seen_ix].flatten().tolist()))
    omap = {t: i for i, t in enumerate(outs)}
    gold = torch.tensor([[omap.get(int(t), -1) for t in row] for row in dec.tolist()], device=device)
    seen_const = (dec[:, seen_ix] == dec[:, seen_ix[:1]]).all(1)
    held_const = (dec == dec[:, seen_ix[:1]]).all(1)  # constant over all 23 words
    tr_const = tr[seen_const[tr]]
    summary = dict(
        tag="empirical",
        prereg="docs/notes/conditional_certificate_prereg.md",
        smoke=bool(args.smoke),
        seeds=dict(SEEDS, stimulus=999 if args.smoke else SEEDS["stimulus"]),
        held_out=list(HELD_OUT),
        target_precision=TARGET_PREC,
        min_issued=SMOKE_MIN_ISSUED if args.smoke else MIN_ISSUED,
        data=dict(
            classes=n_cls,
            train=len(tr),
            val=len(va),
            test=len(te),
            outputs=len(outs),
            test_seen_constant=float(seen_const[te].float().mean()),
            test_all_constant=float(held_const[te].float().mean()),
            train_seen_constant=int(len(tr_const)),
        ),
    )
    log(
        f"{n_cls} classes: train {len(tr)} val {len(va)} test {len(te)}; outputs {len(outs)}; "
        f"test seen-constant {summary['data']['test_seen_constant']:.3f}"
    )
    E = torch.tensor(z["wte"], device=device)
    per_seed = []
    for seed in SEEDS["fit"]:
        m = train_student(E, len(outs), T, sid, gold, tr, tr_const, seen_ix, opt_seen, seed, steps, device)
        r = evaluate(m, sid, gold, te, seen_ix, held_ix, opt_all, opt_seen)  # soundness aborts + #146 metrics
        gate = train_gate(E, T, sid, seen_const.float(), tr, seen_ix, seed, gsteps, device)
        s_va, s_te = gate_scores(gate, sid, va), gate_scores(gate, sid, te)
        c_va, _ = cert16_and_decision(m, sid, va, seen_ix, opt_seen)
        c_te, p_te = cert16_and_decision(m, sid, te, seen_ix, opt_seen)
        tau = choose_tau(s_va, c_va, seen_const[va], min_n=SMOKE_MIN_ISSUED if args.smoke else MIN_ISSUED)
        issued = (s_te >= tau) & c_te if tau is not None else torch.zeros_like(c_te)
        out = dict(
            seed=seed,
            tau=tau,
            faith_seen=r["faith_seen"],
            cert16=r["cert16"],
            FC16_ungated=float((~seen_const[te][c_te]).float().mean()) if c_te.any() else None,
            gate_auc=auc(s_te, seen_const[te]),
        )
        out.update(issued_metrics(issued, p_te, gold[te], seen_ix, held_ix, seen_const[te], held_const[te]))
        per_seed.append(out)
        log(
            f"seed {seed}: faith {out['faith_seen']:.3f} cert16 {out['cert16']:.3f} FC16(ungated) "
            f"{out['FC16_ungated']} | gate AUC {out['gate_auc']} tau {tau} | coverage {out['coverage']:.3f} "
            f"FCi {out['FCi']} faith|issued {out['faith_issued']} "
            f"held-const|issued {out['held_const_issued']} "
            f"held-faith|issued {out['held_faith_issued']}"
        )
    summary["cell"] = dict(per_seed=per_seed, **summarise(per_seed))
    summary["verdict"] = decide(summary["cell"])
    summary["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["verdict"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--smoke", action="store_true", help="bug check: stimulus 999, 200 sigma")
    rr = sub.add_parser("run")
    rr.add_argument("--dumps", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument("--smoke", action="store_true", help="bug check: 300 steps; numbers are NOT results")
    args = ap.parse_args()
    {"dump": cmd_dump, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
