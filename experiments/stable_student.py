"""Stable certified training: does gradient clipping remove the IBP collapse, and what does a stable
student certify?

Pre-registered: docs/notes/stable_student_prereg.md. Follows experiments/certified_student.py (#146), whose
student, bound, certificates and metrics are reused unchanged except for one defensive edit to the bound: the
attention-softmax interval endpoints are sorted (they cannot cross except by float rounding).

    # 1. dump GPT-2 labels on the fresh stimulus seed (needs `transformers`):
    /path/to/venv/bin/python experiments/stable_student.py dump --out runs/stable
    # 2. train and evaluate (torch only), under a sleep inhibitor:
    systemd-inhibit --what=sleep:idle /path/to/venv/bin/python experiments/stable_student.py run \
        --dumps runs/stable --out runs/stable/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_student as cst  # noqa: E402
from certified_student import HELD_OUT, WORDS, _box, _mul, certified, evaluate, worst_logits  # noqa: E402

SEEDS = dict(stimulus=151, split=152, fit=(20, 21, 22, 23, 24))
STEPS, SMOKE_STEPS, BATCH, LR, IBP_W = cst.STEPS, cst.SMOKE_STEPS, cst.BATCH, cst.LR, cst.IBP_W
CLIP = 1.0
ARMS = ("plain", "ibp", "ibp_clip")
COLLAPSE_MARGIN = 0.10


def softmax_box(slo, shi, mask):
    """Sound bounds on causal softmax weights when each score lies in [slo, shi]; endpoints sorted."""
    mx = shi.max(-1, keepdim=True).values
    elo, ehi = torch.exp(slo - mx) * mask, torch.exp(shi - mx) * mask
    alo = elo / (elo + (ehi.sum(-1, keepdim=True) - ehi)).clamp_min(1e-30)
    ahi = (ehi / (ehi + (elo.sum(-1, keepdim=True) - elo)).clamp_min(1e-30)).clamp(max=1.0)
    return torch.minimum(alo, ahi), torch.maximum(alo, ahi)


class Student(cst.Student):
    def ibp(self, ids, opt_ids, alpha=1.0):
        """As certified_student.Student.ibp, with the softmax endpoints sorted (softmax_box)."""
        x = self.embed(ids) + self.pos
        Eo = self.embed(opt_ids)
        mid, half = (Eo.max(0).values + Eo.min(0).values) / 2, (Eo.max(0).values - Eo.min(0).values) / 2
        lo, hi = x.clone(), x.clone()
        lo[:, 0] = mid + self.pos[0] - alpha * half
        hi[:, 0] = mid + self.pos[0] + alpha * half
        B, T, Dm = lo.shape
        dh = Dm // self.H
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))
        for i in range(self.L):
            qlo_, qhi_ = _box(lo, hi, self.qkv[i].weight, self.qkv[i].bias)
            sp = lambda t: t.view(B, T, 3, self.H, dh).permute(2, 0, 3, 1, 4)  # noqa: E731
            qlo, klo, vlo = sp(qlo_)
            qhi, khi, vhi = sp(qhi_)
            plo, phi = _mul(qlo.unsqueeze(3), qhi.unsqueeze(3), klo.unsqueeze(2), khi.unsqueeze(2))
            slo, shi = plo.sum(-1) / math.sqrt(dh), phi.sum(-1) / math.sqrt(dh)
            slo, shi = slo.masked_fill(~mask, -1e9), shi.masked_fill(~mask, -1e9)
            alo, ahi = softmax_box(slo, shi, mask)
            olo, ohi = _mul(alo.unsqueeze(-1), ahi.unsqueeze(-1), vlo.unsqueeze(2), vhi.unsqueeze(2))
            mg = lambda t: t.sum(3).permute(0, 2, 1, 3).reshape(B, T, Dm)  # noqa: E731
            dlo, dhi = _box(mg(olo), mg(ohi), self.proj[i].weight, self.proj[i].bias)
            lo, hi = lo + dlo, hi + dhi
            hlo, hhi = _box(lo, hi, self.fc[i].weight, self.fc[i].bias)
            dlo, dhi = _box(torch.relu(hlo), torch.relu(hhi), self.out_[i].weight, self.out_[i].bias)
            lo, hi = lo + dlo, hi + dhi
        return _box(lo[:, -1], hi[:, -1], self.head.weight, self.head.bias)


@torch.no_grad()
def fc16(m, sid, dec, cls, seen_ix, opt_seen):
    """Among classes certified over the seen words, the share where GPT-2 is not seen-constant."""
    x = sid[cls]
    p0 = torch.cat([m(x[i : i + 4096, seen_ix[0]]).argmax(1) for i in range(0, len(cls), 4096)])
    lo, hi = m.ibp(x[:, 0], opt_seen, 1.0)
    c16 = certified(lo, hi, p0)
    sc = (dec[cls][:, seen_ix] == dec[cls][:, seen_ix[:1]]).all(1)
    return float((c16 & ~sc).sum() / c16.sum()) if c16.any() else None


def collapsed(faith_seen, plain_mean):
    return faith_seen < plain_mean - COLLAPSE_MARGIN


KEYS = cst.KEYS + ("FC16",)


def summarise(per_seed):
    cell = {}
    for k in KEYS:
        vals = [s[k] for s in per_seed if s.get(k) is not None]
        cell[k] = (
            dict(mean=float(np.mean(vals)), range=[float(min(vals)), float(max(vals))]) if vals else None
        )
    return cell


def decide(cells):
    pm = cells["plain"]["faith_seen"]["mean"]
    clip = cells["ibp_clip"]
    n_col = sum(collapsed(s["faith_seen"], pm) for s in clip["per_seed"])
    fs, c16, c23 = clip["faith_seen"]["mean"], clip["cert16"]["mean"], clip["cert23"]["mean"]
    return dict(
        H1=dict(collapsed=n_col, fits=len(clip["per_seed"]), verdict="pass" if n_col == 0 else "fail"),
        H2=dict(faith_seen=fs, plain=pm, verdict="pass" if fs >= pm - 0.02 else "fail"),
        H3=dict(cert16=c16, verdict="pass" if c16 >= 0.90 else "fail"),
        H4=dict(cert23=c23, verdict="pass" if c23 >= 0.50 else "fail"),
    )


def cmd_dump(args):
    cst.SEEDS = dict(cst.SEEDS, stimulus=SEEDS["stimulus"])  # cst.cmd_dump reads its module's stimulus seed
    cst.cmd_dump(args)


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = SMOKE_STEPS if args.smoke else STEPS
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
    te = torch.tensor(order[n_tr + n_va :], device=device)
    outs = sorted(set(dec[tr][:, seen_ix].flatten().tolist()))
    omap = {t: i for i, t in enumerate(outs)}
    gold = torch.tensor([[omap.get(int(t), -1) for t in row] for row in dec.tolist()], device=device)
    seen_const = (dec[:, seen_ix] == dec[:, seen_ix[:1]]).all(1)
    all_const = (dec == dec[:, :1]).all(1)
    tr_const = tr[seen_const[tr]]
    summary = dict(
        tag="empirical",
        prereg="docs/notes/stable_student_prereg.md",
        smoke=bool(args.smoke),
        seeds=dict(SEEDS, stimulus=999 if args.smoke else SEEDS["stimulus"]),
        held_out=list(HELD_OUT),
        clip=CLIP,
        data=dict(
            classes=n_cls,
            train=len(tr),
            test=len(te),
            outputs=len(outs),
            test_all_constant=float(all_const[te].float().mean()),
            test_seen_constant=float(seen_const[te].float().mean()),
            train_seen_constant=int(len(tr_const)),
        ),
        cells={},
    )
    log(
        f"{n_cls} classes: train {len(tr)} ({len(tr_const)} seen-constant) test {len(te)}; "
        f"outputs {len(outs)}; "
        f"test seen-constant {summary['data']['test_seen_constant']:.3f}"
    )
    E = torch.tensor(z["wte"], device=device)
    for arm in ARMS:
        per_seed = []
        for seed in SEEDS["fit"]:
            torch.manual_seed(seed)
            m = Student(E, len(outs), T).to(device)
            o = torch.optim.Adam(m.parameters(), lr=LR)
            g = torch.Generator(device="cpu").manual_seed(seed)
            for step in range(steps):
                c = tr[torch.randint(len(tr), (BATCH,), generator=g).to(device)]
                w = seen_ix[torch.randint(len(seen_ix), (BATCH,), generator=g).to(device)]
                yb = gold[c, w]
                keep = yb >= 0
                loss = F.cross_entropy(m(sid[c, w][keep]), yb[keep])
                if arm != "plain" and len(tr_const):
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
                if arm == "ibp_clip":
                    torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
                o.step()
            r = evaluate(m, sid, gold, te, seen_ix, held_ix, opt_all, opt_seen)
            r["FC16"] = fc16(m, sid, dec, te, seen_ix, opt_seen)
            r["seed"] = seed
            per_seed.append(r)
            log(
                f"{arm:8s} seed {seed}: faith seen {r['faith_seen']:.3f} held {r['faith_held']:.3f} | cert16 "
                f"{r['cert16']:.3f} cert23 {r['cert23']:.3f} | CF23 {r['CF23']:.3f} FC23 {r['FC23']} "
                f"FC16 {r['FC16']}"
            )
        summary["cells"][arm] = dict(per_seed=per_seed, **summarise(per_seed))
    pm = summary["cells"]["plain"]["faith_seen"]["mean"]
    for arm in ("ibp", "ibp_clip"):
        summary["cells"][arm]["collapsed"] = sum(
            collapsed(s["faith_seen"], pm) for s in summary["cells"][arm]["per_seed"]
        )
    summary["verdict"] = decide(summary["cells"])
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
