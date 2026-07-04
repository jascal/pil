"""Does the rule path add decode value on the INDUCTION/COPY subset (invisible to overall top-1)?

Move-1.5 found HYBRID (memb + rules) ~= memb-only overall (+0.001): rules add ~0 to bulk next-token accuracy.
But that metric is dominated by frequent tokens. Rules (eq_atom composition) are supposed to matter where the
next token is a COPY/INDUCTION target -- the token repeats something in context. This splits the test set:
  COPY subset     : target next-token APPEARS in the L-token context window (copy/induction case)
  NON-COPY subset : it does not
and reports top-1 per subset for linear / memb->V / rules->V / HYBRID. If HYBRID > memb on the COPY subset,
the rule path (eq_atom composition) earns decode value exactly where it should; if not, rules are purely for
interpretability and the static decode head simply cannot copy.

Run: cd pil && .venv/bin/python experiments/wyly_decode_repeat.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import wyly_decode as WD  # noqa: E402
import wyly_decode_ablation as AB  # noqa: E402

K, R, STEPS = 256, 512, 4000


def train(model, r, ids, y, tr, seed=0):
    torch.manual_seed(seed)
    model = model.to(WD.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        idx = tr[torch.randint(len(tr), (256,), generator=g)]
        out = model(r[idx], ids[idx]) if ids is not None else model(r[idx])
        opt.zero_grad()
        F.cross_entropy(out, y[idx]).backward()
        opt.step()
    return model


def top1(model, r, ids, y, idxs, use_ids):
    with torch.no_grad():
        out = model(r[idxs], ids[idxs]) if use_ids else model(r[idxs])
        return float((out.argmax(1) == y[idxs]).float().mean())


def main():
    d = torch.load(WD.DATA)
    r = d["r"].float()
    r = (r - r.mean(0)) / (r.std(0) + 1e-6)
    ids = d["kept_ids"]
    target = d["target"]
    copy = (ids == target.unsqueeze(1)).any(1)                    # target next-token appears in the window
    uniq, y = target.unique(return_inverse=True)
    r, ids = r.to(WD.DEV), ids.to(WD.DEV)
    y, copy = y.to(WD.DEV), copy.to(WD.DEV)
    nclass, n = len(uniq), r.shape[0]
    tr = torch.arange(int(0.85 * n), device=WD.DEV)
    te = torch.arange(int(0.85 * n), n, device=WD.DEV)
    te_c = te[copy[te]]
    te_n = te[~copy[te]]
    print(f"repeat-subset decode -- FULL vocab ({nclass}), {WD.DEV}, {STEPS} steps")
    pct = len(te_c) / len(te) * 100
    print(f"test {len(te)}: COPY (target in window) {len(te_c)} ({pct:.0f}%) | NON-COPY {len(te_n)}\n")

    models = {
        "linear": (WD.Linear(r.shape[1], nclass), False),
        "memb->V": (AB.MembDecode(r.shape[1], nclass), True),
        "rules->V": (WD.WylyDecode(r.shape[1], nclass, K, R), True),
        "HYBRID": (AB.HybridDecode(r.shape[1], nclass), True),
    }
    print(f"{'decode':>10}{'overall':>10}{'COPY':>9}{'non-copy':>10}")
    res = {}
    for name, (m, uid) in models.items():
        m = train(m, r, ids if uid else None, y, tr)
        ov = top1(m, r, ids, y, te, uid)
        cc = top1(m, r, ids, y, te_c, uid)
        nn = top1(m, r, ids, y, te_n, uid)
        res[name] = (ov, cc, nn)
        print(f"{name:>10}{ov:>10.3f}{cc:>9.3f}{nn:>10.3f}", flush=True)
    hyb_c, memb_c = res["HYBRID"][1], res["memb->V"][1]
    print(f"\nHYBRID vs memb on COPY subset: {hyb_c:.3f} vs {memb_c:.3f}  (delta {hyb_c - memb_c:+.3f})")
    if hyb_c - memb_c > 0.01:
        print("=> the RULE PATH earns decode value on copy/induction positions (eq_atom composition),")
        print("   though it adds ~0 overall -- rules matter for the computed tail, not the frequent bulk.")
    else:
        print("=> rules add ~0 even on the copy subset: a static vocab head cannot COPY a context token")
        print("   and soft-AND-over-concepts does not recover it -- rules stay decode-irrelevant.")


if __name__ == "__main__":
    main()
