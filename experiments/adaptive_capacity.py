"""Adaptive K (concepts) and R (rules) via split/merge/prune/add under an MDL cost — capacity is EARNED.

The coverage sweep (concept_coverage.py) held K fixed and hit a recovery/accuracy plateau: fixed capacity
is a mismatched knob for the joint concept+rule problem. Here K and R are STRUCTURAL VARIABLES with
bidirectional pressure:
  SPLIT concept — the concept carrying the most attributed residual loss (doing two jobs) duplicates+perturbs.
  MERGE concept — the most-similar concept pair (redundant) fuses (its trace = forming a super-concept).
  ADD rule      — when the worst-fit task underfits, a fresh rule is born.
  PRUNE rule    — a rule with ~zero head weight / low firing is removed.
Every move is accepted only if it lowers total MDL = mean task loss + lambda*(K + R). Timescale separation:
fast gradient training between slow structural phases (avoids the moving-target instability).

Decisive test on the coverage planted world: does adaptive K CONVERGE to ~the exercised-concept-count from
BOTH an undersized (K=2) and an oversized (K=12) init? That would show capacity self-tunes to the task.

Run: cd pil && .venv/bin/python experiments/adaptive_capacity.py
"""

from __future__ import annotations

import itertools
import random

import torch
import torch.nn.functional as F

V, L = 15, 5
DEV = "cpu"
LAMBDA = 0.02                    # MDL capacity cost per concept/rule
MAX_K, MAX_R = 14, 40


# ---- planted world (same family as concept_coverage): K_true categories, tasks over concepts ----
def plant(k_true, seed):
    rng = random.Random(seed)
    cat = list(range(k_true)) + [rng.randrange(k_true) for _ in range(V - k_true)]
    rng.shuffle(cat)
    return cat


def build_tasks(cover):
    specs = []
    for c in range(cover):
        specs.append(("first", c))
        specs.append(("contains", c))
    return specs


def label(seq, cat, spec):
    cs = [cat[t] for t in seq]
    return int(cs[0] == spec[1]) if spec[0] == "first" else int(spec[1] in cs)


def gen(n, cat, specs, seed):
    rng = random.Random(seed)
    X = [[rng.randrange(V) for _ in range(L)] for _ in range(n)]
    Y = [torch.tensor([label(s, cat, sp) for s in X]).float() for sp in specs]
    return torch.tensor(X), Y


# ---- adaptive learner: E (V,K) softmax token->concept; rule (R,L,K+1) softmax (K concepts + wildcard) ----
class State:
    def __init__(self, K, R, T):
        self.K, self.R, self.T = K, R, T
        self.E = (torch.randn(V, K) * 0.3).requires_grad_()
        rl = torch.randn(R, L, K + 1) * 0.3
        rl[:, :, -1] += 1.5
        self.rule = rl.requires_grad_()
        self.heads = torch.zeros(T, R + 1).requires_grad_()

    def params(self):
        return [self.E, self.rule, self.heads]

    def clone(self):
        s = State.__new__(State)
        s.K, s.R, s.T = self.K, self.R, self.T
        s.E = self.E.detach().clone().requires_grad_()
        s.rule = self.rule.detach().clone().requires_grad_()
        s.heads = self.heads.detach().clone().requires_grad_()
        return s


def fire(s, x, tau):
    c = F.softmax(s.E[x] / tau, -1)                                   # (B,L,K)
    sel = F.softmax(s.rule / tau, -1)                                 # (R,L,K+1)
    lit = torch.einsum("blk,rlk->brl", c, sel[:, :, :-1]) + sel[:, :, -1]
    return lit.prod(2)                                                # (B,R)


def logits(s, x, tau):
    g = fire(s, x, tau)
    return g @ s.heads[:, :-1].T + s.heads[:, -1]                     # (B,T)


def task_loss(s, X, Y, tau=0.5):
    lg = logits(s, X, tau)
    return sum(F.binary_cross_entropy_with_logits(lg[:, ti], Y[ti]) for ti in range(s.T)) / s.T


def mdl(s, X, Y):
    with torch.no_grad():
        return float(task_loss(s, X, Y)) + LAMBDA * (s.K + s.R)


def train(s, X, Y, steps, lr=0.04):
    opt = torch.optim.Adam(s.params(), lr=lr)
    n = X.shape[0]
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        tau = max(0.4, 1.3 - 0.9 * step / max(steps, 1))
        idx = torch.randint(n, (256,), generator=g)
        loss = task_loss(s, X[idx], [y[idx] for y in Y], tau)
        opt.zero_grad()
        loss.backward()
        opt.step()


# ---- structural moves (operate on tensors, return a new State) ----
def split_concept(s, k):
    n = State.__new__(State)
    n.K, n.R, n.T = s.K + 1, s.R, s.T
    E = s.E.detach()
    n.E = torch.cat([E, E[:, k:k + 1] + torch.randn(V, 1) * 0.1], 1).requires_grad_()
    rl = s.rule.detach()
    concept, wild = rl[:, :, :-1], rl[:, :, -1:]
    n.rule = torch.cat([concept, concept[:, :, k:k + 1], wild], 2).requires_grad_()
    n.heads = s.heads.detach().clone().requires_grad_()
    return n


def merge_concepts(s, a, b):                                          # a < b
    n = State.__new__(State)
    n.K, n.R, n.T = s.K - 1, s.R, s.T
    E = s.E.detach()
    merged = torch.logsumexp(torch.stack([E[:, a], E[:, b]]), 0, keepdim=True).T
    keep = [j for j in range(s.K) if j != b]
    Ek = E[:, keep].clone()
    Ek[:, a if a < b else a] = merged.squeeze(1)                      # a stays in place among 'keep'
    n.E = Ek.requires_grad_()
    rl = s.rule.detach()
    concept, wild = rl[:, :, :-1], rl[:, :, -1:]
    cm = torch.logsumexp(torch.stack([concept[:, :, a], concept[:, :, b]]), 0)
    ck = concept[:, :, keep].clone()
    ck[:, :, a] = cm
    n.rule = torch.cat([ck, wild], 2).requires_grad_()
    n.heads = s.heads.detach().clone().requires_grad_()
    return n


def add_rule(s):
    n = State.__new__(State)
    n.K, n.R, n.T = s.K, s.R + 1, s.T
    rl = s.rule.detach()
    new = torch.randn(1, L, s.K + 1) * 0.3
    new[:, :, -1] += 1.5
    n.rule = torch.cat([rl, new], 0).requires_grad_()
    n.E = s.E.detach().clone().requires_grad_()
    n.heads = torch.cat([s.heads.detach()[:, :-1], torch.zeros(s.T, 1), s.heads.detach()[:, -1:]],
                        1).requires_grad_()
    return n


def prune_rule(s, r):
    n = State.__new__(State)
    n.K, n.R, n.T = s.K, s.R - 1, s.T
    keep = [j for j in range(s.R) if j != r]
    n.rule = s.rule.detach()[keep].clone().requires_grad_()
    n.E = s.E.detach().clone().requires_grad_()
    hk = torch.cat([s.heads.detach()[:, keep], s.heads.detach()[:, -1:]], 1)
    n.heads = hk.requires_grad_()
    return n


# ---- proposal signals ----
def concept_residual(s, X, Y):
    """Mean task-loss attributed to the queried-token's concept — the overloaded concept to split."""
    with torch.no_grad():
        lg = logits(s, X, 0.5)
        per = torch.stack([F.binary_cross_entropy_with_logits(lg[:, ti], Y[ti], reduction="none")
                           for ti in range(s.T)]).mean(0)             # (B,)
        c0 = F.softmax(s.E[X[:, 0]] / 0.4, -1).argmax(-1)
        res = torch.zeros(s.K)
        for k in range(s.K):
            m = c0 == k
            res[k] = per[m].mean() if m.any() else torch.tensor(0.0)
    return res


def most_similar_concepts(s):
    with torch.no_grad():
        E = F.softmax(s.E / 0.4, -1).T                                # (K,V)
        best, pair = -1.0, (0, 1)
        for a in range(s.K):
            for b in range(a + 1, s.K):
                sim = float(F.cosine_similarity(E[a], E[b], 0))
                if sim > best:
                    best, pair = sim, (a, b)
    return pair, best


def dead_rule(s, X):
    with torch.no_grad():
        fireR = fire(s, X, 0.4).mean(0)                              # (R,) mean firing
        wmag = s.heads[:, :-1].abs().sum(0)                         # (R,) head usage
        score = fireR * (wmag + 1e-3)
    return int(score.argmin())


# ---- adaptive loop ----
def adapt(X, Y, K0, R0, T, phases=7, inner=500, seed=0):
    torch.manual_seed(seed)
    s = State(K0, R0, T)
    train(s, X, Y, 1200)
    trace = [(s.K, s.R, mdl(s, X, Y))]
    moves = []
    for _ in range(phases):
        cur = mdl(s, X, Y)
        cands = []
        # concept split (overloaded concept)
        if s.K < MAX_K:
            k = int(concept_residual(s, X, Y).argmax())
            cands.append(("split", split_concept(s, k)))
        # concept merge (redundant pair)
        if s.K > 2:
            (a, b), _ = most_similar_concepts(s)
            cands.append(("merge", merge_concepts(s, a, b)))
        # rule add / prune
        if s.R < MAX_R:
            cands.append(("add_rule", add_rule(s)))
        if s.R > 2:
            cands.append(("prune_rule", prune_rule(s, dead_rule(s, X))))
        # evaluate each candidate after a short inner train; accept the best MDL improvement
        best_move, best_state, best_mdl = None, None, cur
        for name, cand in cands:
            train(cand, X, Y, inner)
            m = mdl(cand, X, Y)
            if m < best_mdl - 1e-4:
                best_move, best_state, best_mdl = name, cand, m
        if best_state is None:
            break
        s = best_state
        moves.append(best_move)
        trace.append((s.K, s.R, best_mdl))
    train(s, X, Y, 1000)
    return s, trace, moves


def recovery(s, cat):
    if s.K > 8:
        return float("nan")
    a = F.softmax(s.E, -1).argmax(-1).tolist()
    return max(sum(perm[a[v]] == cat[v] for v in range(V)) / V
               for perm in itertools.permutations(range(s.K)))


def acc(s, X, Y):
    with torch.no_grad():
        lg = logits(s, X, 0.4)
        return sum(float(((lg[:, ti] > 0).float() == Y[ti]).float().mean()) for ti in range(s.T)) / s.T


def main():
    print(f"adaptive K/R via split/merge/add/prune under MDL (lambda={LAMBDA})\n")
    for cover in (3, 6):
        cat = plant(cover, 0)
        specs = build_tasks(cover)
        Xtr, Ytr = gen(3000, cat, specs, 10)
        Xte, Yte = gen(3000, cat, specs, 99)
        print(f"== task family exercises {cover} concepts ({len(specs)} tasks) ==")
        for K0, R0, tag in ((2, 8, "undersized K=2"), (12, 24, "oversized K=12")):
            s, trace, moves = adapt(Xtr, Ytr, K0, R0, len(specs))
            path = " -> ".join(f"K{k}R{r}" for k, r, _ in trace)
            rec = recovery(s, cat)
            print(f"   {tag:>16}: final K={s.K} R={s.R}  test_acc={acc(s, Xte, Yte):.3f}  "
                  f"recovery={rec:.3f}  |  {path}")
        print()
    print("read: if final K converges to ~the exercised-concept-count from BOTH inits (undersized grows, "
          "oversized shrinks), capacity SELF-TUNES to the task -- K/R are earned structural variables, not "
          "hyperparameters. Divergent/stuck K would mean the structural search or MDL cost needs work.")


if __name__ == "__main__":
    main()
