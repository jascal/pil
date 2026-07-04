"""Tokens belong to MANY concepts: sparse-membership vs softmax-partition on a multi-attribute world.

The concept_atoms/concept_rules learners used a SOFTMAX token->concept map = a competitive PARTITION (one
concept per token). But a token is a bundle of independent attributes ("2" is even AND small AND
single-digit), on different branches -> multi-membership, and the hierarchy is a lattice, not a tree.
The right map is independent per-concept membership (SIGMOID + sparsity), not softmax.

Planted world: V tokens, each carrying a random subset of A binary attributes. Tasks include SAME-TOKEN
attribute conjunctions ("first token has attr0 AND attr1") -- the discriminator: a partition can't represent a
token being two things unless it has a category for that combination (needs ~2^A categories). Membership does
it with A units.

Race: sparse-sigmoid MEMBERSHIP (A units) vs softmax PARTITION (K categories, sweep K). Unified rule form: a
rule may REQUIRE several attributes at a position (soft-AND of memberships). Prediction: membership solves the
conjunction tasks and recovers the A attributes; partition needs K~2^A and represents combos, not attributes.

Run: cd pil && .venv/bin/python experiments/multi_membership.py
"""

from __future__ import annotations

import itertools
import random

import torch
import torch.nn.functional as F
from torch import nn

V, A, L, R = 16, 4, 4, 24                       # tokens, attributes, seq len, #rules
DEV = "cpu"


def plant(seed):
    rng = random.Random(seed)
    while True:
        attr = [[int(rng.random() < 0.5) for _ in range(A)] for _ in range(V)]   # token -> {0,1}^A
        cols = [sum(attr[v][a] for v in range(V)) for a in range(A)]
        if all(3 <= c <= V - 3 for c in cols):                                    # each attribute used
            return attr


def has(token_attr, a):
    return token_attr[a] == 1


def task_labels(seq, attr):
    aa = [attr[t] for t in seq]
    # decisive discriminator: recover EVERY attribute of the first (and last) token -> the code must
    # preserve all A bits per token. A K-category partition carries log2(K) bits; K<2^A must lose some.
    d = {f"first_a{a}": int(has(aa[0], a)) for a in range(A)}
    d["last_a0"] = int(has(aa[-1], 0))
    d["last_a3"] = int(has(aa[-1], 3))
    d["first_a0_AND_a1"] = int(has(aa[0], 0) and has(aa[0], 1))         # a same-token conjunction (bonus)
    return d


TASKS = list(task_labels([0] * L, [[0] * A] * V))
CONJ = {"first_a0_AND_a1"}


def gen(n, attr, seed):
    rng = random.Random(seed)
    X = [[rng.randrange(V) for _ in range(L)] for _ in range(n)]
    Y = {t: [] for t in TASKS}
    for s in X:
        lab = task_labels(s, attr)
        for t in TASKS:
            Y[t].append(lab[t])
    return torch.tensor(X), {t: torch.tensor(Y[t]).float() for t in TASKS}


class RuleNet(nn.Module):
    """Feature per token = A sigmoid memberships (mode='membership') or Fdim softmax categories
    (mode='partition'). A rule REQUIRES a soft subset of features at each position (multi-attribute AND)."""

    def __init__(self, mode, fdim):
        super().__init__()
        self.mode, self.fdim = mode, fdim
        self.E = nn.Parameter(torch.randn(V, fdim) * 0.3)
        self.req = nn.Parameter(torch.randn(R, L, fdim) * 0.3 - 1.5)      # sigmoid -> mostly OFF at init
        self.heads = nn.Parameter(torch.zeros(len(TASKS), R + 1))

    def features(self, x, tau):
        z = self.E[x] / tau
        return torch.sigmoid(z) if self.mode == "membership" else F.softmax(z, -1)

    def fire(self, x, tau):
        f = self.features(x, tau)                                          # (B, L, Fdim) in [0,1]
        req = torch.sigmoid(self.req)                                      # (R, L, Fdim)
        term = 1 - req.unsqueeze(0) + req.unsqueeze(0) * f.unsqueeze(1)    # (B, R, L, Fdim)
        return term.prod(-1).prod(-1)                             # soft-AND over features & positions

    def logit(self, ti, x, tau):
        w = self.heads[ti]
        return self.fire(x, tau) @ w[:-1] + w[-1]

    def reg(self):
        r = torch.sigmoid(self.req).mean()                                # sparse rule bodies
        if self.mode == "membership":
            return r + torch.sigmoid(self.E).mean()                       # sparse memberships
        return r


def train(model, X, Y, steps=3000, lr=0.05, reg_w=0.02):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        tau = max(0.4, 1.5 - 1.1 * step / steps)
        idx = torch.randint(n, (256,), generator=g)
        loss = sum(F.binary_cross_entropy_with_logits(model.logit(ti, X[idx], tau), Y[t][idx])
                   for ti, t in enumerate(TASKS)) + reg_w * min(1.0, step / (steps * 0.5)) * model.reg()
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def task_acc(model, X, Y):
    return {t: float(((model.logit(ti, X, 0.4) > 0).float() == Y[t]).float().mean())
            for ti, t in enumerate(TASKS)}


@torch.no_grad()
def attr_recovery(model, attr):
    """Best assignment of learned membership units to planted attributes: mean per-attribute accuracy."""
    m = (torch.sigmoid(model.E) > 0.5).int().tolist()                     # (V, A) learned memberships
    planted = attr
    best = 0.0
    for perm in itertools.permutations(range(A)):
        acc = sum(m[v][perm[a]] == planted[v][a] for v in range(V) for a in range(A)) / (V * A)
        best = max(best, acc)
    return best


def main():
    seeds = [0, 1]
    print(f"multi-attribute world: V={V} tokens, A={A} independent attributes, same-token-conj tasks\n")

    print("MEMBERSHIP model (A sigmoid units), mean over seeds:")
    macc, mrec = {t: [] for t in TASKS}, []
    for sd in seeds:
        attr = plant(sd)
        Xtr, Ytr = gen(3000, attr, sd + 10)
        Xte, Yte = gen(4000, attr, sd + 99)
        m = RuleNet("membership", A)
        train(m, Xtr, Ytr)
        a = task_acc(m, Xte, Yte)
        for t in TASKS:
            macc[t].append(a[t])
        mrec.append(attr_recovery(m, attr))
    for t in TASKS:
        tag = " <- SAME-TOKEN CONJ" if t in CONJ else ""
        print(f"   {t:>18}: {sum(macc[t]) / len(seeds):.3f}{tag}")
    print(f"   attribute recovery: {sum(mrec) / len(mrec):.3f}   (units used: {A})")

    print("\nPARTITION model (K softmax categories), conjunction-task acc vs K:")
    print(f"{'K':>4}{'conj_acc':>10}{'all_acc':>9}   (2^A = {2 ** A} categories needed for all combos)")
    for kk in (A, 2 * A, 2 ** A):
        cacc, aacc = [], []
        for sd in seeds:
            attr = plant(sd)
            Xtr, Ytr = gen(3000, attr, sd + 10)
            Xte, Yte = gen(4000, attr, sd + 99)
            m = RuleNet("partition", kk)
            train(m, Xtr, Ytr)
            a = task_acc(m, Xte, Yte)
            cacc.append(sum(a[t] for t in CONJ) / len(CONJ))
            aacc.append(sum(a.values()) / len(a))
        print(f"{kk:>4}{sum(cacc) / len(cacc):>10.3f}{sum(aacc) / len(aacc):>9.3f}", flush=True)
    print("\nHONEST read (the experiment did NOT show a membership advantage): a SOFT softmax-over-K is a "
          "continuous K-dim code (a simplex point), not a hard log2(K)-bit label -- so soft partition and "
          "soft membership are equivalent-capacity here, and 16 atomic tokens are memorizable by both. The "
          "multi-membership advantage (A units cover 2^A combos) needs the HARD-crystallized regime with "
          "#distinctions >> K, or COMPOSITIONAL GENERALIZATION to novel attribute-combos via structured "
          "(non-atomic) tokens -- neither of which this setup exercises. Membership also does NOT cleanly "
          "disentangle (recovery ~0.67): per-attribute tasks leave a rotation ambiguity in the code.")


if __name__ == "__main__":
    main()
