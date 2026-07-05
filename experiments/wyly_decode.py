"""Wyly move 1: can rule-firings feed a real NEXT-TOKEN decode over vocab? (the make-or-break test)

Turns Wyly from binary probes into a next-token predictor: geometric concepts (hyperplanes on the frozen
pythia-70m residual) + token eq_atom -> soft-AND rules -> a full-vocab decode head (rules = PIC decode
monomials). Trained on next-token cross-entropy over the FULL vocab (every unique next-token as a class --
no 'other' bucket, which would otherwise dominate top-1 and make the metric uninformative).

The question: does the concept+rule path preserve next-token information, or does the concept bottleneck
destroy it? Sweep K (concepts) and compare top-1 to:
  linear   -- residual -> vocab logits (the raw-linear decode ceiling ~ the backbone's own unembedding)
  unigram  -- predict the most frequent token (floor ~4.7%)
If Wyly approaches the linear ceiling as K grows, the rule-decode holds (architecture viable); if it plateaus
far below, the concept bottleneck / rule form is the limiter.

Run: cd pil && .venv/bin/python experiments/wyly_decode.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "nexttoken_pythia70m.pt"
# (full-vocab decode; no top-V cutoff)
TAU = 0.6
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS = [32, 64, 128, 256]


class WylyDecode(torch.nn.Module):
    def __init__(self, d, nclass, K, R):
        super().__init__()
        self.Ug = torch.nn.Parameter(torch.randn(K, d) * (1.0 / d ** 0.5))
        self.bg = torch.nn.Parameter(torch.zeros(K))
        self.req = torch.nn.Parameter(torch.randn(R, K + 1) * 0.3 - 2.0)     # rules over K concepts + eq_atom
        self.Wdec = torch.nn.Parameter(torch.zeros(R + 1, nclass))          # rules -> vocab decode (+bias)
        self.R = R

    def forward(self, r, ids):
        memb = torch.sigmoid((r @ self.Ug.T - self.bg) / TAU)               # (B, K)
        teq = (ids[:, :-1] == ids[:, -1:]).any(1, keepdim=True).float()     # (B, 1) eq_atom
        feats = torch.cat([memb, teq], -1)
        req = torch.sigmoid(self.req)
        fire = (1 - req.unsqueeze(0) + req.unsqueeze(0) * feats.unsqueeze(1)).prod(-1)   # (B, R)
        return fire @ self.Wdec[:self.R] + self.Wdec[self.R]


def train_top1(model, r, ids, y, tr, te, steps=4000, lr=0.01, seed=0):
    torch.manual_seed(seed)
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = tr[torch.randint(len(tr), (256,), generator=g)]
        out = model(r[idx], ids[idx]) if ids is not None else model(r[idx])
        loss = F.cross_entropy(out, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = model(r[te], ids[te]) if ids is not None else model(r[te])
        return float((out.argmax(1) == y[te]).float().mean())


class Linear(torch.nn.Module):
    def __init__(self, d, nclass):
        super().__init__()
        self.lin = torch.nn.Linear(d, nclass)

    def forward(self, r):
        return self.lin(r)


def main():
    d = torch.load(DATA)
    r = d["r"].float().to(DEV)
    r = (r - r.mean(0)) / (r.std(0) + 1e-6)
    ids = d["kept_ids"].to(DEV)
    target = d["target"]
    uniq, y = target.unique(return_inverse=True)                   # FULL vocab, dense-remapped to [0, nclass)
    y = y.to(DEV)
    nclass = len(uniq)
    n = r.shape[0]
    tr = torch.arange(int(0.85 * n), device=DEV)
    te = torch.arange(int(0.85 * n), n, device=DEV)
    unigram = float((y[te] == torch.bincount(y[tr]).argmax()).float().mean())
    print(f"Wyly next-token decode -- FULL vocab ({nclass} unique next-tokens), r {tuple(r.shape)}, "
          f"train {len(tr)}/test {len(te)}")
    print(f"floor: unigram (most-frequent) top-1 {unigram:.3f}\n")
    lin_acc = train_top1(Linear(r.shape[1], nclass), r, None, y, tr, te)
    print(f"ceiling: linear residual->vocab top-1 {lin_acc:.3f}\n")
    print(f"{'K concepts':>11}{'R rules':>9}{'Wyly top-1':>12}{'% of linear':>13}")
    for K in KS:
        R = max(128, 2 * K)
        acc = train_top1(WylyDecode(r.shape[1], nclass, K, R), r, ids, y, tr, te)
        print(f"{K:>11}{R:>9}{acc:>12.3f}{acc / lin_acc * 100:>12.0f}%", flush=True)
    print("\nread: Wyly top-1 rising toward the linear ceiling as K grows = concept+rule decode PRESERVES")
    print("next-token info (move-1 viable). Plateau far below linear = concept bottleneck / rule form is the")
    print("limiter. Gap above unigram = how much structure the rules capture at all.")


if __name__ == "__main__":
    main()
