"""Certifiable by construction: a certified-trained student with the right invariances.

Pre-registered: docs/notes/certified_student_prereg.md.

    # 1. dump GPT-2 labels and frozen embedding rows (needs `transformers`):
    /path/to/venv/bin/python experiments/certified_student.py dump --out runs/student
    # 2. train and evaluate (torch only), under a sleep inhibitor:
    systemd-inhibit --what=sleep:idle /path/to/venv/bin/python experiments/certified_student.py run \
        --dumps runs/student --out runs/student/summary.json

The student is a 2-layer causal transformer without LayerNorm, over GPT-2's frozen token embeddings (a
trained projection), with a sound, differentiable interval bound (IBP) over a nuisance word set at position
0. A certificate proves the STUDENT's decision invariant over the set; faithfulness to GPT-2 is measured,
not certified.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_substitutes as cs  # noqa: E402

WORDS = [
    "Yesterday",
    "Today",
    "Tonight",
    "Recently",
    "Later",
    "Earlier",
    "Again",
    "Then",
    "Meanwhile",
    "Suddenly",
    "Finally",
    "Once",
    "Now",
    "Still",
    "Soon",
    "Eventually",
    "Instead",
    "Apparently",
    "Clearly",
    "Luckily",
    "Sadly",
    "Indeed",
    "Naturally",
]
HELD_OUT = (
    "Apparently",
    "Eventually",
    "Finally",
    "Luckily",
    "Naturally",
    "Once",
    "Today",
)  # fixed by seed 133
SEEDS = dict(stimulus=141, split=142, fit=(0, 1, 2))
N_SIGMA, SMOKE_SIGMA = 4000, 200
STEPS, SMOKE_STEPS, BATCH, LR, IBP_W = 6000, 300, 256, 1e-3, 0.5
D, LAYERS, HEADS = 128, 2, 4


# ---------------------------------------------------------------- the student and its interval bound


def _box(lo, hi, W, b):
    c, r = (lo + hi) / 2, (hi - lo) / 2
    c2, r2 = c @ W.T + b, r @ W.abs().T
    return c2 - r2, c2 + r2


def _mul(alo, ahi, blo, bhi):
    c = torch.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return c.min(0).values, c.max(0).values


class Student(nn.Module):
    def __init__(self, frozen_emb: torch.Tensor, n_out: int, T: int, d=D, L=LAYERS, H=HEADS):
        super().__init__()
        self.register_buffer("E0", frozen_emb.clone())  # (vocab, 768), frozen
        self.proj_in = nn.Linear(frozen_emb.shape[1], d)
        self.pos = nn.Parameter(torch.randn(T, d) * 0.02)
        self.L, self.H, self.d = L, H, d
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d) for _ in range(L)])
        self.proj = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        self.fc = nn.ModuleList([nn.Linear(d, 4 * d) for _ in range(L)])
        self.out_ = nn.ModuleList([nn.Linear(4 * d, d) for _ in range(L)])
        self.head = nn.Linear(d, n_out)

    def embed(self, ids):
        return self.proj_in(self.E0[ids])

    def forward(self, ids):
        x = self.embed(ids) + self.pos
        B, T, Dm = x.shape
        dh = Dm // self.H
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))
        for i in range(self.L):
            q, k, v = self.qkv[i](x).view(B, T, 3, self.H, dh).permute(2, 0, 3, 1, 4)
            s = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            a = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
            x = x + self.proj[i]((a @ v).permute(0, 2, 1, 3).reshape(B, T, Dm))
            x = x + self.out_[i](torch.relu(self.fc[i](x)))
        return self.head(x[:, -1])

    def ibp(self, ids, opt_ids, alpha=1.0):
        """Sound logit bounds when position 0 ranges over opt_ids (their embedding box, scaled by alpha)."""
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
            mx = shi.max(-1, keepdim=True).values
            elo, ehi = torch.exp(slo - mx) * mask, torch.exp(shi - mx) * mask
            alo = elo / (elo + (ehi.sum(-1, keepdim=True) - ehi)).clamp_min(1e-30)
            ahi = (ehi / (ehi + (elo.sum(-1, keepdim=True) - elo)).clamp_min(1e-30)).clamp(max=1.0)
            olo, ohi = _mul(alo.unsqueeze(-1), ahi.unsqueeze(-1), vlo.unsqueeze(2), vhi.unsqueeze(2))
            mg = lambda t: t.sum(3).permute(0, 2, 1, 3).reshape(B, T, Dm)  # noqa: E731
            dlo, dhi = _box(mg(olo), mg(ohi), self.proj[i].weight, self.proj[i].bias)
            lo, hi = lo + dlo, hi + dhi
            hlo, hhi = _box(lo, hi, self.fc[i].weight, self.fc[i].bias)
            dlo, dhi = _box(torch.relu(hlo), torch.relu(hhi), self.out_[i].weight, self.out_[i].bias)
            lo, hi = lo + dlo, hi + dhi
        return _box(lo[:, -1], hi[:, -1], self.head.weight, self.head.bias)


def worst_logits(lo, hi, y):
    w = hi.clone()
    w.scatter_(1, y[:, None], lo.gather(1, y[:, None]))
    return w


def certified(lo, hi, y):
    t = lo.gather(1, y[:, None]).squeeze(1)
    o = hi.clone()
    o.scatter_(1, y[:, None], float("-inf"))
    return t > o.max(1).values


# ---------------------------------------------------------------- metrics and decision rules


@torch.no_grad()
def evaluate(m, sid, gold, cls, seen_ix, held_ix, opt_all, opt_seen):
    """sid: (classes, 23, T) student token ids; gold: (classes, 23) GPT-2 decisions as output indices
    (-1 = outside the output set)."""
    x = sid[cls]
    n, k, T = x.shape
    pred = torch.cat([m(x.reshape(-1, T)[i : i + 4096]).argmax(1) for i in range(0, n * k, 4096)]).view(n, k)
    g = gold[cls]
    p0 = pred[:, seen_ix[0]]
    out = {}
    for name, opt, ix in (("16", opt_seen, seen_ix), ("23", opt_all, torch.arange(k, device=x.device))):
        lo, hi = m.ibp(x[:, 0], opt, 1.0)
        cert = certified(lo, hi, p0)
        if (cert[:, None] & (pred[:, ix] != p0[:, None])).any():
            raise AssertionError(f"certified over {name} words with a concrete prediction that differs (bug)")
        out[f"cert{name}"] = cert
    one_lo, one_hi = m.ibp(x[:, 0], opt_all[:1], 1.0)
    if float((one_hi - one_lo).abs().max()) > 1e-9:
        raise AssertionError("k=1 box is not exact (bug)")
    c23 = out["cert23"]
    faithful_all = (g == p0[:, None]).all(1)
    cf = c23 & faithful_all
    fc = c23 & ~faithful_all
    held_f = (g[:, held_ix] == p0[:, None]).all(1)
    return dict(
        faith_seen=float((pred[:, seen_ix] == g[:, seen_ix]).float().mean()),
        faith_held=float((pred[:, held_ix] == g[:, held_ix]).float().mean()),
        cert16=float(out["cert16"].float().mean()),
        cert23=float(c23.float().mean()),
        CF23=float(cf.float().mean()),
        FC23=float(fc.sum() / c23.sum()) if c23.any() else None,
        held_faith_among_cert=float(held_f[c23].float().mean()) if c23.any() else None,
        n_cert23=int(c23.sum()),
    )


def decide(cells):
    ib = cells.get("ibp", {})
    mean = lambda k: ib[k]["mean"] if ib.get(k) else None  # noqa: E731
    cf, fc, hf = mean("CF23"), mean("FC23"), mean("held_faith_among_cert")
    return dict(
        H1=dict(CF23=cf, verdict="pass" if cf is not None and cf >= 0.50 else "fail"),
        H2=dict(FC23=fc, verdict="untestable" if fc is None else ("pass" if fc <= 0.10 else "fail")),
        H3=dict(
            held_faith_among_cert=hf,
            verdict="untestable" if hf is None else ("pass" if hf >= 0.90 else "fail"),
        ),
    )


KEYS = ("faith_seen", "faith_held", "cert16", "cert23", "CF23", "FC23", "held_faith_among_cert")


def summarise(per_seed):
    cell = {}
    for k in KEYS:
        vals = [s[k] for s in per_seed if s.get(k) is not None]
        cell[k] = (
            dict(mean=float(np.mean(vals)), range=[float(min(vals)), float(max(vals))]) if vals else None
        )
    return cell


# ---------------------------------------------------------------- dump / run


def cmd_dump(args):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # optional dependency, dump only

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    assert all(len(tok.encode(w + ", the")) == 3 for w in WORDS), "words must be position-aligned"
    occ = [w for w in cs.OCCUPATIONS if len(tok.encode(" " + w)) == 1][:40]
    verbs = [w for w in cs.VERBS if w.endswith("ed") and len(tok.encode(" " + w)) == 1][:16]
    combos = [(s, v, o) for s in range(40) for o in range(40) if s != o for v in range(16)]
    random.Random(999 if args.smoke else SEEDS["stimulus"]).shuffle(combos)
    sig = combos[: SMOKE_SIGMA if args.smoke else N_SIGMA]
    texts = [
        f"{w}, the {occ[s]} {verbs[v]} the {occ[o]}. The {occ[o]} was {verbs[v]} by the"
        for (s, v, o) in sig
        for w in WORDS
    ]
    ids = torch.tensor([tok.encode(t) for t in texts], device=device)
    with torch.no_grad():
        dec = torch.cat([gpt(ids[i : i + 512]).logits[:, -1].argmax(-1) for i in range(0, len(ids), 512)])
    vocab = sorted(set(ids.flatten().tolist()))
    vmap = {t: i for i, t in enumerate(vocab)}
    sid = np.array([[vmap[t] for t in row] for row in ids.tolist()]).reshape(len(sig), len(WORDS), -1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "student.npz",
        sid=sid,
        dec=dec.view(len(sig), len(WORDS)).cpu().numpy(),
        vocab=np.array(vocab),
        wte=gpt.transformer.wte.weight.detach()[vocab].float().cpu().numpy(),
        word_ids=np.array([vmap[tok.encode(w)[0]] for w in WORDS]),
    )
    print(
        f"wrote {out / 'student.npz'}: {len(sig)} classes x {len(WORDS)} words, vocab {len(vocab)}",
        flush=True,
    )


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
        prereg="docs/notes/certified_student_prereg.md",
        smoke=bool(args.smoke),
        seeds=dict(SEEDS, stimulus=999 if args.smoke else SEEDS["stimulus"]),
        held_out=list(HELD_OUT),
        data=dict(
            classes=n_cls,
            train=len(tr),
            test=len(te),
            outputs=len(outs),
            test_all_constant=float(all_const[te].float().mean()),
            train_seen_constant=int(len(tr_const)),
        ),
        cells={},
    )
    log(
        f"{n_cls} classes: train {len(tr)} ({len(tr_const)} seen-constant) test {len(te)}; "
        f"outputs {len(outs)}; "
        f"test all-constant {summary['data']['test_all_constant']:.3f}"
    )
    E = torch.tensor(z["wte"], device=device)
    for arm in ("plain", "ibp"):
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
                if arm == "ibp" and len(tr_const):
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
                o.step()
            r = evaluate(m, sid, gold, te, seen_ix, held_ix, opt_all, opt_seen)
            r["seed"] = seed
            per_seed.append(r)
            log(
                f"{arm:5s} seed {seed}: faith seen {r['faith_seen']:.3f} held {r['faith_held']:.3f} | cert16 "
                f"{r['cert16']:.3f} cert23 {r['cert23']:.3f} | CF23 {r['CF23']:.3f} FC23 {r['FC23']} "
                f"held-faith|cert {r['held_faith_among_cert']}"
            )
        summary["cells"][arm] = dict(per_seed=per_seed, **summarise(per_seed))
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
