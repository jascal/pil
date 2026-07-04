"""Co-evolving CONCEPT substrate: learnable token->concept atoms shared across a task family (gradient DS).

The idea (this session's thread): don't SEARCH for the right predicate (discrete, defeated the naive
proposer in predicate_invention.py) — GROW it. Start from random token->concept associations and a random
concept->super hierarchy, compose rules over concept-features, share the concept lattice across many tasks,
and let backprop + a crystallization pressure carve meaningful, reusable concepts out of the random init.

Planted world: V tokens each have a hidden category in {0..K-1} (the ground-truth "concepts") and the
categories group into {0..M-1} supers (a 2-level hierarchy). A family of tasks is defined purely over
categories (majority, same-ends, contains, count, super-membership) — so the category structure is the
REUSABLE abstraction shared by every task. The learner is never told the categories.

Tests: (1) do learned concepts RECOVER the planted categories (permutation-aligned purity)? (2) REUSE win —
a SHARED concept lattice vs per-task concept lattices, on data-efficiency; (3) crystallization soft->crisp;
(4) hierarchy recovery.

Run: cd pil && .venv/bin/python experiments/concept_atoms.py
"""

from __future__ import annotations

import itertools
import random
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn

V, K, M, L = 12, 3, 2, 5                      # tokens, categories, supers, sequence length
DEV = "cpu"


# ---- planted world ------------------------------------------------------------------------------
def plant(seed=0):
    rng = random.Random(seed)
    cat = [rng.randrange(K) for _ in range(V)]           # token -> category
    sup = [rng.randrange(M) for _ in range(K)]           # category -> super
    return cat, sup


def labels(seq, cat, sup):
    cs = [cat[t] for t in seq]
    most = Counter(cs).most_common(1)[0][0]
    return {
        "majority0": int(most == 0),
        "same_ends": int(cs[0] == cs[-1]),
        "contains2": int(2 in cs),
        "first0": int(cs[0] == 0),
        "count0>=2": int(cs.count(0) >= 2),
        "super_first_A": int(sup[cs[0]] == 0),
    }


TASKS = ["majority0", "same_ends", "contains2", "first0", "count0>=2", "super_first_A"]


def gen(n, cat, sup, seed):
    rng = random.Random(seed)
    X = [[rng.randrange(V) for _ in range(L)] for _ in range(n)]
    Y = {t: [] for t in TASKS}
    for s in X:
        lab = labels(s, cat, sup)
        for t in TASKS:
            Y[t].append(lab[t])
    return torch.tensor(X), {t: torch.tensor(Y[t]).float() for t in TASKS}


# ---- learner: shared (or per-task) token->concept lattice + rules over concept features ----------
FEAT = 4 * K + 1 + 2 * M          # 4K concept (mean/first/last/any) + 1 same + 2M super (mean/first)


class ConceptLearner(nn.Module):
    def __init__(self, shared=True):
        super().__init__()
        nE = 1 if shared else len(TASKS)
        self.shared = shared
        self.E = nn.Parameter(torch.randn(nE, V, K) * 0.3)     # token -> concept logits
        self.S = nn.Parameter(torch.randn(nE, K, M) * 0.3)     # concept -> super logits (hierarchy)
        self.heads = nn.Parameter(torch.zeros(len(TASKS), FEAT + 1))

    def feats(self, ti, x, tau):
        e = self.E[0 if self.shared else ti]
        s_map = F.softmax(self.S[0 if self.shared else ti] / tau, -1)
        c = F.softmax(e[x] / tau, -1)                          # (B, L, K)
        sup = c @ s_map                                        # (B, L, M)
        same = (c[:, 0] * c[:, -1]).sum(-1, keepdim=True)
        return torch.cat([c.mean(1), c[:, 0], c[:, -1], c.max(1).values, same,
                          sup.mean(1), sup[:, 0]], -1)

    def logit(self, ti, x, tau):
        w = self.heads[ti]
        return self.feats(ti, x, tau) @ w[:-1] + w[-1]

    def concept_entropy(self):
        p = F.softmax(self.E, -1)
        return -(p * (p + 1e-9).log()).sum(-1).mean()


def recovery(model, cat):
    """Best permutation-aligned purity of learned token->concept vs planted categories."""
    assign = F.softmax(model.E[0], -1).argmax(-1).tolist()
    best = 0.0
    for perm in itertools.permutations(range(K)):
        acc = sum(perm[assign[v]] == cat[v] for v in range(V)) / V
        best = max(best, acc)
    return best


def train(model, Xtr, Ytr, steps=3000, ent_max=0.05, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        tau = max(0.3, 1.5 - 1.2 * step / steps)               # anneal soft -> crisp
        ent_w = ent_max * min(1.0, step / (steps * 0.5))       # ramp crystallization pressure
        idx = torch.randint(n, (256,), generator=g)
        loss = sum(F.binary_cross_entropy_with_logits(model.logit(ti, Xtr[idx], tau), Ytr[t][idx])
                   for ti, t in enumerate(TASKS))
        loss = loss + ent_w * model.concept_entropy()
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def test_acc(model, Xte, Yte):
    accs = {}
    for ti, t in enumerate(TASKS):
        pred = (model.logit(ti, Xte, 0.3) > 0).float()
        accs[t] = float((pred == Yte[t]).float().mean())
    return sum(accs.values()) / len(accs), accs


def main():
    cat, sup = plant(0)
    print(f"planted categories {cat}  supers-of-cat {sup}  (V={V} K={K} M={M} L={L})")
    Xte, Yte = gen(4000, cat, sup, seed=99)
    base = {t: max(float(Yte[t].mean()), 1 - float(Yte[t].mean())) for t in TASKS}
    print(f"task base rates (majority-class): {[round(base[t], 2) for t in TASKS]}\n")

    print("(1,2,3) SHARED vs PER-TASK concept lattice, at increasing data per task:")
    print(f"{'N/task':>7}{'shared_acc':>12}{'shared_recov':>14}{'pertask_acc':>13}{'pertask_recov':>15}")
    for n in (150, 600, 2400):
        Xtr, Ytr = gen(n, cat, sup, seed=n)
        sh = ConceptLearner(shared=True).to(DEV)
        train(sh, Xtr, Ytr)
        pt = ConceptLearner(shared=False).to(DEV)
        train(pt, Xtr, Ytr)
        sa, _ = test_acc(sh, Xte, Yte)
        pa, _ = test_acc(pt, Xte, Yte)
        print(f"{n:>7}{sa:>12.3f}{recovery(sh, cat):>14.3f}{pa:>13.3f}{recovery(pt, cat):>15.3f}", flush=True)

    print("\n(4) hierarchy: learned concept->super vs planted (shared model, largest N):")
    sh = ConceptLearner(shared=True).to(DEV)
    Xtr, Ytr = gen(2400, cat, sup, seed=7)
    train(sh, Xtr, Ytr)
    learned_sup = F.softmax(sh.S[0], -1).argmax(-1).tolist()
    print(f"   planted super-of-category {sup}  learned {learned_sup}")
    print("\nread: shared_recov -> planted categories crystallize from RANDOM init by gradient + reuse; "
          "shared beats per-task at low N = the reuse win (concepts discovered once serve every task).")


if __name__ == "__main__":
    main()
