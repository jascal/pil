"""Localize the move-1 ~16% decode gap: is the sigmoid-membership bottleneck or the soft-AND rule stage lossy?

Move-1 found Wyly's rule-decode plateaus at ~84% of the linear ceiling (form-limited, not concept-count).
This ablates the decode path at fixed K to localize the lossy stage. All decode to the FULL vocab, top-1:
  linear    : r -> V                                   (ceiling; full residual, no bottleneck)
  memb->V   : sigmoid(r@Ug)+eq_atom -> V               (Wyly CONCEPTS, NO rules)
  rules->V  : sigmoid -> soft-AND rules -> V           (full Wyly)
  mlp->V    : ReLU(r@W1) -> V                          (generic nonlinear-K control)
Reading: memb->V ~ linear => concepts preserve the decode, the SOFT-AND rules are the lossy stage.
         memb->V ~ rules->V (both ~84%) => the sigmoid-membership BOTTLENECK is the limit, rules add no loss.

Run: cd pil && .venv/bin/python experiments/wyly_decode_ablation.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import wyly_decode as WD  # noqa: E402

K, R = 256, 512


class MembDecode(torch.nn.Module):
    """Wyly concepts (sigmoid memberships) + eq_atom, decoded linearly to V -- NO soft-AND rule layer."""
    def __init__(self, d, nclass, k=K):
        super().__init__()
        self.Ug = torch.nn.Parameter(torch.randn(k, d) * (1.0 / d ** 0.5))
        self.bg = torch.nn.Parameter(torch.zeros(k))
        self.dec = torch.nn.Linear(k + 1, nclass)

    def forward(self, r, ids):
        memb = torch.sigmoid((r @ self.Ug.T - self.bg) / WD.TAU)
        teq = (ids[:, :-1] == ids[:, -1:]).any(1, keepdim=True).float()
        return self.dec(torch.cat([memb, teq], -1))


class MLPDecode(torch.nn.Module):
    """generic nonlinear K-bottleneck control: ReLU hidden layer of width K -> V (no eq_atom, no rules)."""
    def __init__(self, d, nclass, k=K):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, k), torch.nn.ReLU(), torch.nn.Linear(k, nclass))

    def forward(self, r):
        return self.net(r)


V, STEPS = 1024, 1800    # restrict to top-V frequent next-tokens (in-vocab examples only) for a fast, clean
#                          localization; the RELATIVE gaps (linear/memb/rules) are what matter, not absolutes


def main():
    d = torch.load(WD.DATA)
    r = d["r"].float()
    r = (r - r.mean(0)) / (r.std(0) + 1e-6)
    ids = d["kept_ids"]
    target = d["target"]
    topv = torch.bincount(target).argsort(descending=True)[:V]     # keep only in-top-V examples (no 'other')
    keep = torch.isin(target, topv)
    r, ids, target = r[keep], ids[keep], target[keep]
    uniq, y = target.unique(return_inverse=True)
    nclass = len(uniq)
    n = r.shape[0]
    tr = torch.arange(int(0.85 * n))
    te = torch.arange(int(0.85 * n), n)
    print(f"decode ablation @ K={K}, R={R}  -- top-{V} vocab ({nclass} classes, in-vocab only), top-1  "
          f"(train {len(tr)}/test {len(te)})\n")

    lin = WD.train_top1(WD.Linear(r.shape[1], nclass), r, None, y, tr, te, steps=STEPS)
    memb = WD.train_top1(MembDecode(r.shape[1], nclass), r, ids, y, tr, te, steps=STEPS)
    rules = WD.train_top1(WD.WylyDecode(r.shape[1], nclass, K, R), r, ids, y, tr, te, steps=STEPS)
    mlp = WD.train_top1(MLPDecode(r.shape[1], nclass), r, None, y, tr, te, steps=STEPS)

    print(f"{'decode path':>28}{'top-1':>9}{'% linear':>10}")
    rows = [("linear  r->V (ceiling)", lin), ("memb->V  (concepts, no rules)", memb),
            ("rules->V (full Wyly)", rules), ("mlp->V   (ReLU-K control)", mlp)]
    for name, a in rows:
        print(f"{name:>28}{a:>9.3f}{a / lin * 100:>9.0f}%", flush=True)
    print("\nlocalization:")
    print(f"  concept bottleneck cost (linear - memb): {lin - memb:+.3f}")
    print(f"  soft-AND rule cost     (memb - rules)  : {memb - rules:+.3f}")
    if memb - rules > lin - memb:
        print("  => SOFT-AND RULE stage is the dominant lossy step (concepts largely preserve decode).")
    else:
        print("  => SIGMOID-MEMBERSHIP bottleneck is the dominant lossy step (rules add little loss).")


if __name__ == "__main__":
    main()
