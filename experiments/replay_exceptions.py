"""Replay curation WITH an exception tail: now edge cases carry value representative coverage misses.

replay_curation.py validated the escape (small buffer recovers continual learning) but couldn't test the
'edge cases matter' half -- its planted world was cleanly categorical, so it had no genuine exceptions for
edge-selection to rescue. Here we plant an EXCEPTION TAIL: a few IRREGULAR tokens violate the categorical
rule on specific A-tasks (rare, rule-breaking, high-impact -- the computed/exception structure variance-greedy
compression drops). Prediction: representative (k-center coverage) samples the common bulk and MISSES the rare
irregulars; edge (high learning-impact) CATCHES them; so rep+edge > representative-only, especially on the
EXCEPTION examples specifically.

Metrics: overall A-retention AND exception-token A-retention. Run: cd pil && .venv/bin/python
experiments/replay_exceptions.py
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

V, L, K, R = 18, 5, 12, 28
KT = 6
A_CATS, B_CATS = [0, 1, 2], [3, 4, 5]
N_IRREG = 3                       # this many tokens are irregular (exceptions to the rule)


def plant(seed):
    rng = random.Random(seed)
    cat = list(range(KT)) + [rng.randrange(KT) for _ in range(V - KT)]
    rng.shuffle(cat)
    irreg = set(rng.sample(range(V), N_IRREG))            # irregular tokens (rule-breakers)
    # each irregular flips the label of exactly one A "first==c" task when it's the first token
    flip = {t: rng.choice(A_CATS) for t in irreg}
    return cat, irreg, flip


def specs_for(cats):
    return [(kind, c) for c in cats for kind in ("first", "contains")]


A_SPECS, B_SPECS = specs_for(A_CATS), specs_for(B_CATS)
SPECS = A_SPECS + B_SPECS
A_IDS, B_IDS = list(range(len(A_SPECS))), list(range(len(A_SPECS), len(SPECS)))


def label(seq, cat, spec, flip):
    cs = [cat[t] for t in seq]
    if spec[0] == "first":
        base = int(cs[0] == spec[1])
        if seq[0] in flip and flip[seq[0]] == spec[1]:       # irregular first token -> flip this task
            return 1 - base
        return base
    return int(spec[1] in cs)


def gen(n, cat, flip, seed):
    rng = random.Random(seed)
    X = [[rng.randrange(V) for _ in range(L)] for _ in range(n)]
    Y = [torch.tensor([label(s, cat, sp, flip) for s in X]).float() for sp in SPECS]
    return torch.tensor(X), Y


class Net:
    def __init__(self, seed):
        torch.manual_seed(seed)
        self.E = (torch.randn(V, K) * 0.3).requires_grad_()
        rl = torch.randn(R, L, K + 1) * 0.3
        rl[:, :, -1] += 1.5
        self.rule = rl.requires_grad_()
        self.heads = torch.zeros(len(SPECS), R + 1).requires_grad_()

    def params(self):
        return [self.E, self.rule, self.heads]

    def clone(self):
        n = Net.__new__(Net)
        n.E = self.E.detach().clone().requires_grad_()
        n.rule = self.rule.detach().clone().requires_grad_()
        n.heads = self.heads.detach().clone().requires_grad_()
        return n


def fire(s, x, tau=0.5):
    c = F.softmax(s.E[x] / tau, -1)
    sel = F.softmax(s.rule / tau, -1)
    lit = torch.einsum("blk,rlk->brl", c, sel[:, :, :-1]) + sel[:, :, -1]
    return lit.prod(2)


def logits(s, x, tau=0.5):
    return fire(s, x, tau) @ s.heads[:, :-1].T + s.heads[:, -1]


def train_A(s, X, Y, steps=3000, lr=0.04):
    opt = torch.optim.Adam(s.params(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        tau = max(0.4, 1.3 - 0.9 * step / steps)
        idx = torch.randint(X.shape[0], (256,), generator=g)
        lg = logits(s, X[idx], tau)
        loss = sum(F.binary_cross_entropy_with_logits(lg[:, ti], Y[ti][idx]) for ti in A_IDS) / len(A_IDS)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_B_replay(s, X, Y, buf_idx, steps=2200, lr=0.04, alpha=1.5):
    opt = torch.optim.Adam(s.params(), lr=lr)
    g = torch.Generator().manual_seed(1)
    buf = torch.tensor(buf_idx) if len(buf_idx) else None
    for step in range(steps):
        tau = max(0.4, 1.3 - 0.9 * step / steps)
        idx = torch.randint(X.shape[0], (256,), generator=g)
        lg = logits(s, X[idx], tau)
        loss = sum(F.binary_cross_entropy_with_logits(lg[:, ti], Y[ti][idx]) for ti in B_IDS) / len(B_IDS)
        if buf is not None:
            bi = buf[torch.randint(len(buf), (min(128, len(buf)),), generator=g)]
            lgb = logits(s, X[bi], tau)
            ab = [F.binary_cross_entropy_with_logits(lgb[:, ti], Y[ti][bi]) for ti in A_IDS]
            loss = loss + alpha * sum(ab) / len(A_IDS)
        opt.zero_grad()
        loss.backward()
        opt.step()


def acc_ids(s, X, Y, ids, rows=None):
    with torch.no_grad():
        lg = logits(s, X, 0.4)
        if rows is not None:
            lg, Yr = lg[rows], [Y[ti][rows] for ti in range(len(Y))]
        else:
            Yr = Y
        return sum(float(((lg[:, ti] > 0).float() == Yr[ti]).float().mean()) for ti in ids) / len(ids)


def a_loss_per_example(s, X, Y):
    with torch.no_grad():
        lg = logits(s, X, 0.4)
        return torch.stack([F.binary_cross_entropy_with_logits(lg[:, ti], Y[ti], reduction="none")
                            for ti in A_IDS]).mean(0)


def kcenter(feats, M, seed=0):
    g = torch.Generator().manual_seed(seed)
    first = int(torch.randint(feats.shape[0], (1,), generator=g))
    picked = [first]
    d = ((feats - feats[first]) ** 2).sum(-1)
    for _ in range(M - 1):
        j = int(d.argmax())
        picked.append(j)
        d = torch.minimum(d, ((feats - feats[j]) ** 2).sum(-1))
    return picked


def select(strategy, s, X, Y, M):
    n = X.shape[0]
    if strategy == "none":
        return []
    if strategy == "full":
        return list(range(n))
    if strategy == "random":
        return torch.randperm(n, generator=torch.Generator().manual_seed(0))[:M].tolist()
    impact = a_loss_per_example(s, X, Y)
    feats = fire(s, X, 0.4)
    if strategy == "edge":
        return impact.argsort(descending=True)[:M].tolist()
    if strategy == "representative":
        return kcenter(feats, M)
    if strategy == "rep+edge":
        edge = impact.argsort(descending=True)[:M // 2].tolist()
        return list(dict.fromkeys(kcenter(feats, M - len(edge)) + edge))
    raise ValueError(strategy)


def main():
    M = 60
    seeds = [0, 1, 2]
    strategies = ["none", "random", "representative", "edge", "rep+edge", "full"]
    print(f"replay curation WITH exception tail ({N_IRREG} irregular tokens); M={M} buffer\n")
    print(f"{'strategy':>16}{'A_retained':>12}{'EXC_retained':>14}{'B_learned':>11}")
    rows = {st: [] for st in strategies}
    for sd in seeds:
        cat, irreg, flip = plant(sd)
        Xtr, Ytr = gen(4000, cat, flip, sd + 10)
        Xte, Yte = gen(4000, cat, flip, sd + 99)
        exc_rows = torch.tensor([i for i in range(Xte.shape[0]) if int(Xte[i, 0]) in irreg])   # exceptions
        base = Net(sd)
        train_A(base, Xtr, Ytr)
        for st in strategies:
            buf = select(st, base, Xtr, Ytr, M)
            s = base.clone()
            train_B_replay(s, Xtr, Ytr, buf)
            rows[st].append((acc_ids(s, Xte, Yte, A_IDS),
                             acc_ids(s, Xte, Yte, A_IDS, exc_rows),
                             acc_ids(s, Xte, Yte, B_IDS)))
    for st in strategies:
        r = rows[st]
        n = len(r)
        print(f"{st:>16}{sum(x[0] for x in r) / n:>12.3f}{sum(x[1] for x in r) / n:>14.3f}"
              f"{sum(x[2] for x in r) / n:>11.3f}", flush=True)
    print("\nread: EXC_retained = A-accuracy on the rare IRREGULAR tokens (the exception tail). If "
          "representative MISSES them (low EXC) while edge/rep+edge CATCH them (high EXC), edge cases carry "
          "value coverage can't -- confirming 'keep representative AND edge': the bulk needs coverage, the "
          "computed tail needs impact-selection. rep+edge should win overall AND on exceptions.")


if __name__ == "__main__":
    main()
