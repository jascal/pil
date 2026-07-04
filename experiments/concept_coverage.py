"""Identifiability test: at fixed K, does concept RECOVERY track how many concepts the tasks EXERCISE?

concept_rules.py found: task accuracy stays ~0.99 at every K, but concept recovery falls as K grows.
Interpretation: the boundary is IDENTIFIABILITY, not optimization collapse — the co-evolving substrate
recovers exactly the concept distinctions the task family exercises, no more. Direct test: hold K=8 and
SWEEP task coverage E (# concepts the tasks reference, via per-concept first_c/contains_c tasks). Prediction:
recovery climbs from low (sparse) toward ~1.0 (full coverage E=K), while task acc stays high throughout.
Multi-seed (the concept_rules K-sweep was single-seed; make the trend a law).

Run: cd pil && .venv/bin/python experiments/concept_coverage.py
"""

from __future__ import annotations

import itertools
import random

import torch
import torch.nn.functional as F
from torch import nn

V, L, R, M, KK = 15, 5, 28, 2, 8               # tokens, seq len, #rules, supers, fixed #concepts
DEV = "cpu"


def plant(seed):
    rng = random.Random(seed)
    cat = list(range(KK)) + [rng.randrange(KK) for _ in range(V - KK)]   # every concept used >=1 token
    rng.shuffle(cat)
    return cat, [rng.randrange(M) for _ in range(KK)]


def build_tasks(cover):
    """Tasks exercising concepts 0..cover-1: 'first is c' and 'contains c' for each such c."""
    specs = []
    for c in range(cover):
        specs.append((f"first{c}", ("first", c)))
        specs.append((f"has{c}", ("contains", c)))
    return specs


def label(seq, cat, spec):
    cs = [cat[t] for t in seq]
    kind, c = spec[1]
    return int(cs[0] == c) if kind == "first" else int(c in cs)


def gen(n, cat, specs, seed):
    rng = random.Random(seed)
    X = [[rng.randrange(V) for _ in range(L)] for _ in range(n)]
    Y = [torch.tensor([label(s, cat, sp) for s in X]).float() for sp in specs]
    return torch.tensor(X), Y


class ConceptRuleLearner(nn.Module):
    def __init__(self, ntasks):
        super().__init__()
        self.E = nn.Parameter(torch.randn(V, KK) * 0.3)
        rs = torch.randn(R, L, KK + 1) * 0.3
        rs[:, :, -1] += 1.5
        self.rule_sel = nn.Parameter(rs)
        self.heads = nn.Parameter(torch.zeros(ntasks, R + 1))

    def fire(self, x, tau):
        c = F.softmax(self.E[x] / tau, -1)
        sel = F.softmax(self.rule_sel / tau, -1)
        lit = torch.einsum("blk,rlk->brl", c, sel[:, :, :-1]) + sel[:, :, -1]
        return lit.prod(dim=2)

    def logit(self, ti, x, tau):
        w = self.heads[ti]
        return self.fire(x, tau) @ w[:-1] + w[-1]

    def entropy(self):
        pe = F.softmax(self.E, -1)
        pr = F.softmax(self.rule_sel, -1)
        return -(pe * (pe + 1e-9).log()).sum(-1).mean() - (pr * (pr + 1e-9).log()).sum(-1).mean()


def recovery(model, cat):
    a = F.softmax(model.E, -1).argmax(-1).tolist()
    return max(sum(perm[a[v]] == cat[v] for v in range(V)) / V
               for perm in itertools.permutations(range(KK)))


def train(model, X, Y, steps=2500, ent_max=0.03, lr=0.04):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        tau = max(0.3, 1.5 - 1.2 * step / steps)
        ent_w = ent_max * min(1.0, step / (steps * 0.5))
        idx = torch.randint(n, (256,), generator=g)
        loss = sum(F.binary_cross_entropy_with_logits(model.logit(ti, X[idx], tau), Y[ti][idx])
                   for ti in range(len(Y))) + ent_w * model.entropy()
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def acc(model, X, Y):
    return sum(float(((model.logit(ti, X, 0.3) > 0).float() == Y[ti]).float().mean())
               for ti in range(len(Y))) / len(Y)


def main():
    seeds = [0, 1, 2]
    covers = [2, 3, 4, 6, 8]
    print(f"identifiability test: K={KK} fixed, sweep task coverage E, {len(seeds)} seeds\n")
    print(f"{'coverage E':>11}{'#tasks':>8}{'recovery(mean)':>16}{'recovery(seeds)':>22}{'task_acc':>10}")
    for cover in covers:
        specs = build_tasks(cover)
        recs, accs = [], []
        for sd in seeds:
            cat, sup = plant(sd)
            Xtr, Ytr = gen(2500, cat, specs, sd + 10)
            Xte, Yte = gen(3000, cat, specs, sd + 99)
            m = ConceptRuleLearner(len(specs))
            train(m, Xtr, Ytr)
            recs.append(recovery(m, cat))
            accs.append(acc(m, Xte, Yte))
        mean = sum(recs) / len(recs)
        print(f"{cover:>11}{len(specs):>8}{mean:>16.3f}   {[round(r, 2) for r in recs]!s:>19}"
              f"{sum(accs) / len(accs):>10.3f}", flush=True)
    print("\nread: if recovery rises with coverage toward ~1.0 at E=K while task_acc stays high, the "
          "collapse boundary IS identifiability -- the substrate recovers exactly the concepts the tasks "
          "exercise; richer tasks recover more. Flat/low recovery would instead mean optimization collapse.")


if __name__ == "__main__":
    main()
