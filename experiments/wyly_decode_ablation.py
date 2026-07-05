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


class HybridDecode(torch.nn.Module):
    """THE FIX: decode reads BOTH the membership vector (bulk, linear -- beats residual) AND soft-AND
    rule firings (relational structure the linear reader cannot do -- eq_atom, composition)."""
    def __init__(self, d, nclass, k=K, rr=R):
        super().__init__()
        self.Ug = torch.nn.Parameter(torch.randn(k, d) * (1.0 / d ** 0.5))
        self.bg = torch.nn.Parameter(torch.zeros(k))
        self.req = torch.nn.Parameter(torch.randn(rr, k + 1) * 0.3 - 2.0)
        self.dec = torch.nn.Linear(k + 1 + rr, nclass)             # [memberships, eq_atom, rule firings] -> V

    def forward(self, r, ids):
        memb = torch.sigmoid((r @ self.Ug.T - self.bg) / WD.TAU)
        teq = (ids[:, :-1] == ids[:, -1:]).any(1, keepdim=True).float()
        feats = torch.cat([memb, teq], -1)
        req = torch.sigmoid(self.req)
        fire = (1 - req.unsqueeze(0) + req.unsqueeze(0) * feats.unsqueeze(1)).prod(-1)
        return self.dec(torch.cat([memb, teq, fire], -1))


def main():
    gpu = WD.DEV == "cuda"
    V, STEPS = (None, 4000) if gpu else (1024, 1800)               # full vocab on GPU; restricted on CPU
    d = torch.load(WD.DATA)
    r = d["r"].float()
    r = (r - r.mean(0)) / (r.std(0) + 1e-6)
    ids = d["kept_ids"]
    target = d["target"]
    if V is not None:                                             # CPU fallback: restrict to top-V
        topv = torch.bincount(target).argsort(descending=True)[:V]
        keep = torch.isin(target, topv)
        r, ids, target = r[keep], ids[keep], target[keep]
    uniq, y = target.unique(return_inverse=True)
    r, ids, y = r.to(WD.DEV), ids.to(WD.DEV), y.to(WD.DEV)
    nclass = len(uniq)
    n = r.shape[0]
    tr = torch.arange(int(0.85 * n), device=WD.DEV)
    te = torch.arange(int(0.85 * n), n, device=WD.DEV)
    vlab = "FULL" if V is None else f"top-{V}"
    print(f"decode ablation @ K={K}, R={R} -- {vlab} vocab ({nclass} cls), {STEPS} steps, {WD.DEV}, "
          f"top-1 (tr {len(tr)}/te {len(te)})\n")

    lin = WD.train_top1(WD.Linear(r.shape[1], nclass), r, None, y, tr, te, steps=STEPS)
    memb = WD.train_top1(MembDecode(r.shape[1], nclass), r, ids, y, tr, te, steps=STEPS)
    rules = WD.train_top1(WD.WylyDecode(r.shape[1], nclass, K, R), r, ids, y, tr, te, steps=STEPS)
    hybrid = WD.train_top1(HybridDecode(r.shape[1], nclass), r, ids, y, tr, te, steps=STEPS)
    mlp = WD.train_top1(MLPDecode(r.shape[1], nclass), r, None, y, tr, te, steps=STEPS)

    print(f"{'decode path':>32}{'top-1':>9}{'% linear':>10}")
    rows = [("linear  r->V (ceiling)", lin), ("memb->V  (concepts, no rules)", memb),
            ("rules->V (soft-AND, full Wyly)", rules), ("HYBRID->V (memb + rules)", hybrid),
            ("mlp->V   (ReLU-K control)", mlp)]
    for name, a in rows:
        print(f"{name:>32}{a:>9.3f}{a / lin * 100:>9.0f}%", flush=True)
    print("\nlocalization + fix:")
    print(f"  concept bottleneck cost (linear - memb): {lin - memb:+.3f}")
    print(f"  soft-AND rule cost      (memb - rules) : {memb - rules:+.3f}")
    print(f"  HYBRID lift over full-Wyly (hybrid - rules): {hybrid - rules:+.3f}   "
          f"over memb-only: {hybrid - memb:+.3f}")
    if hybrid >= max(memb, lin) - 0.005:
        print("  => HYBRID (linear-over-memberships + rules) CLEARS the soft-AND plateau: bulk decode off")
        print("     memberships, rules kept for the relational structure. Move-1.5 fix works.")
    else:
        print("  => hybrid did NOT clear it; the decode form needs more than concatenating memb + fire.")


if __name__ == "__main__":
    main()
