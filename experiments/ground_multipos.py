"""Grounded RELATIONAL rule: does an eq_atom over concepts ACROSS positions capture is_repeat?

The single-residual reader caps ~0.80 on is_repeat because the earlier-position ingredient isn't in the
current residual. Here we give the concept substrate a WINDOW (last L positions' residuals) and a soft
eq_atom over concept-memberships across positions -- the geometric version of the PIC rule learner's
`x[o1]==x[o2]` Datalog join, but over hyperplane-concepts instead of raw token ids.

  A  single-residual : head over m[cur] (current position concepts only)   [the ~0.80 floor]
  B  multi-pos relational : concepts shared across positions; relational feature per position o<cur =
       match_o = <m[cur], m[o]> (concept-pattern coincidence = soft eq_atom); is_repeat reads
       max_o match_o and mean_o match_o (+ m[cur]).  This is the grounded relational rule.
If B >> A, the cross-token relation IS rule-structured across positions -- recoverable by relational rules
over grounded concepts, resolving why the single-residual reader failed.

Run: cd pil && .venv/bin/python experiments/ground_multipos.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "multipos_pythia70m.pt"


class Concepts(torch.nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(K, d) * (1.0 / d ** 0.5))
        self.b = torch.nn.Parameter(torch.zeros(K))

    def memb(self, r, tau=0.6):                                     # r (..., d) -> (..., K)
        return torch.sigmoid((r @ self.U.T - self.b) / tau)


class Single(torch.nn.Module):
    """A: head over the current position's concept memberships only."""
    def __init__(self, d, K=24):
        super().__init__()
        self.c = Concepts(d, K)
        self.head = torch.nn.Linear(K, 1)

    def forward(self, R):                                          # R (B, L, d); current = last
        return self.head(self.c.memb(R[:, -1])).squeeze(-1)


class MultiPosRelational(torch.nn.Module):
    """B: relational eq_atom over concepts across positions -- match current pattern to earlier ones."""
    def __init__(self, d, K=24):
        super().__init__()
        self.c = Concepts(d, K)
        self.head = torch.nn.Linear(K + 2, 1)                     # m[cur] (K) + [max_match, mean_match]

    def forward(self, R):                                         # R (B, L, d)
        m = self.c.memb(R)                                        # (B, L, K)
        cur = m[:, -1]                                            # (B, K)
        earlier = m[:, :-1]                                       # (B, L-1, K)
        match = (earlier * cur.unsqueeze(1)).sum(-1) / m.shape[-1]  # (B, L-1) soft eq_atom coincidence
        feat = torch.cat([cur, match.max(1).values.unsqueeze(1), match.mean(1, keepdim=True)], -1)
        return self.head(feat).squeeze(-1)


class RawMatch(torch.nn.Module):
    """C: match RAW residuals across positions (bypasses coarse concepts) -- do residuals carry identity?"""
    def __init__(self, d, K=24):
        super().__init__()
        self.head = torch.nn.Linear(2, 1)                        # [max cos, mean cos] of cur vs earlier

    def forward(self, R):                                        # R (B, L, d)
        rn = R / (R.norm(dim=-1, keepdim=True) + 1e-6)
        cur = rn[:, -1]
        cos = (rn[:, :-1] * cur.unsqueeze(1)).sum(-1)           # (B, L-1) cosine to earlier positions
        feat = torch.cat([cos.max(1).values.unsqueeze(1), cos.mean(1, keepdim=True)], -1)
        return self.head(feat).squeeze(-1)


def train_acc(model, R, y, tr, te, steps=3000, lr=0.02, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = tr[torch.randint(len(tr), (256,), generator=g)]
        loss = F.binary_cross_entropy_with_logits(model(R[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((model(R[te]) > 0).float() == y[te]).float().mean())


def balanced(R, y, seed):
    g = torch.Generator().manual_seed(seed)
    pos = torch.where(y == 1)[0]
    neg = torch.where(y == 0)[0]
    neg = neg[torch.randperm(len(neg), generator=g)[:len(pos)]]
    keep = torch.cat([pos, neg])[torch.randperm(2 * len(pos), generator=g)]
    Rk = R[keep].clone()
    mu = Rk.reshape(-1, Rk.shape[-1]).mean(0)
    sd = Rk.reshape(-1, Rk.shape[-1]).std(0) + 1e-6
    Rk = (Rk - mu) / sd                                          # per-dim standardize (shared over positions)
    cut = int(0.7 * len(keep))
    return Rk, y[keep].float(), torch.arange(cut), torch.arange(cut, len(keep))


def main():
    d = torch.load(DATA)
    R = d["r"].float()                                           # (n, L, d)
    y = d["local_repeat"]
    print(f"grounded relational rule -- is_repeat, pythia-70m  R {tuple(R.shape)} (last {d['L']} positions)")
    print(f"local_repeat frac {float(y.float().mean()):.3f}  (balanced -> chance 0.5)\n")
    print(f"{'model':>22}{'is_repeat acc':>15}")
    for name, Cls in [("A single-residual", Single), ("B multi-pos relational", MultiPosRelational),
                      ("C raw-residual match", RawMatch)]:
        accs = []
        for s in (0, 1, 2):
            Rk, yk, tr, te = balanced(R, y, s)
            accs.append(train_acc(Cls(R.shape[-1]), Rk, yk, tr, te, seed=s))
        print(f"{name:>22}{sum(accs) / len(accs):>15.3f}", flush=True)
    print("\nread: B (relational eq_atom over concepts across positions) >> A (single residual) = the cross-"
          "token relation IS rule-structured across positions -- recoverable by a grounded relational rule "
          "(the geometric x[o1]==x[o2] join), resolving why the single-residual reader capped ~0.80. B ~ A "
          "= even the cross-position match doesn't help (the concepts don't carry token identity / it's "
          "genuinely computed elsewhere).")


if __name__ == "__main__":
    main()
