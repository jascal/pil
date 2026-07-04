"""Generality of grounding: is the linear ceiling universal across concept TYPES, or grammar-specific?

Broader-claim test. On real pythia-70m residuals, compare the linear-probe + geometric-concept ceiling for:
  GRAMMAR (syntactic): func, punct, cap        [already ~0.95 in the curriculum]
  SEMANTIC (content):  number, time, place, social  [curated word categories]
Semantic categories are RARE, so we BALANCE (all positives + equal random negatives) -> chance = 0.5, the
ceiling is comparable across concepts regardless of base rate. If semantic ceilings fall well below grammar's,
linear-grounding is grammar-specific and semantics is the messier/higher-rank regime (the computed/retrieved
boundary); if they match, "concepts are hyperplanes" is type-universal.

Run: cd pil && .venv/bin/python experiments/ground_semantic.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "semantic_pythia70m.pt"
GRAMMAR = ["func", "punct", "cap"]
SEMANTIC = ["number", "time", "place", "social"]


def linear_probe(Xtr, ytr, Xte, yte, steps=600, lr=0.05):
    W = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(Xtr @ W + b, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((((Xte @ W + b) > 0).float() == yte).float().mean())


class Geo(torch.nn.Module):
    def __init__(self, d, K=8):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(K, d) * (1.0 / d ** 0.5))
        self.b = torch.nn.Parameter(torch.zeros(K))
        self.h = torch.nn.Parameter(torch.zeros(K + 1))

    def forward(self, r, tau=0.6):
        m = torch.sigmoid((r @ self.U.T - self.b) / tau)
        return m @ self.h[:-1] + self.h[-1]


def geo_acc(r, y, tr, te, seed):
    torch.manual_seed(seed)
    gc = Geo(r.shape[1])
    opt = torch.optim.Adam(gc.parameters(), lr=0.03)
    g = torch.Generator().manual_seed(0)
    for _step in range(2000):
        idx = tr[torch.randint(len(tr), (256,), generator=g)]
        loss = F.binary_cross_entropy_with_logits(gc(r[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((gc(r[te]) > 0).float() == y[te]).float().mean())


def balanced(r, y, seed):
    """All positives + equal random negatives -> chance 0.5; return standardized r, y, train/test idx."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.where(y == 1)[0]
    neg = torch.where(y == 0)[0]
    neg = neg[torch.randperm(len(neg), generator=g)[:len(pos)]]
    keep = torch.cat([pos, neg])[torch.randperm(2 * len(pos), generator=g)]
    rk, yk = r[keep], y[keep]
    rk = (rk - rk.mean(0)) / (rk.std(0) + 1e-6)
    cut = int(0.7 * len(keep))
    return rk, yk, torch.arange(cut), torch.arange(cut, len(keep))


def main():
    d = torch.load(DATA)
    r = d["r"].float()
    names = d["names"]
    labs = d["labels"]
    print(f"grammar vs semantics linear ceiling on pythia-70m residuals  r {tuple(r.shape)}")
    print("(balanced: all positives + equal negatives, chance = 0.500)\n")
    print(f"{'concept':>9}{'n_pos':>8}{'linear':>9}{'geo':>8}   type")
    res = {"grammar": [], "semantic": []}
    for nm in GRAMMAR + SEMANTIC:
        i = names.index(nm)
        y = labs[:, i].float()
        npos = int(y.sum())
        if npos < 200:
            print(f"{nm:>9}{npos:>8}   too few positives; skipping")
            continue
        lin, geo = [], []
        for s in (0, 1, 2):
            rk, yk, tr, te = balanced(r, y, s)
            lin.append(linear_probe(rk[tr], yk[tr], rk[te], yk[te]))
            geo.append(geo_acc(rk, yk, tr, te, s))
        lm, gm = sum(lin) / 3, sum(geo) / 3
        typ = "grammar" if nm in GRAMMAR else "semantic"
        res[typ].append(lm)
        print(f"{nm:>9}{npos:>8}{lm:>9.3f}{gm:>8.3f}   {typ}")
    gav = sum(res["grammar"]) / max(len(res["grammar"]), 1)
    sav = sum(res["semantic"]) / max(len(res["semantic"]), 1)
    print(f"\nmean linear ceiling:  grammar {gav:.3f}   semantic {sav:.3f}   gap {gav - sav:+.3f}")
    print("read: if semantic << grammar, linear-grounding is grammar-specific -- semantics lives in a "
          "messier, higher-rank, superposed regime (the retrieved/computed boundary); concepts-as-"
          "is then TYPE-DEPENDENT, not universal. If ~equal, the linear-representation hypothesis is "
          "type-universal (both syntactic and semantic concepts are hyperplanes).")


if __name__ == "__main__":
    main()
