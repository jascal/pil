"""Replay curation: a small REPRESENTATIVE+EDGE buffer (by learning-impact) beats forgetting cheaply.

The escape from targeted proposal (this session's hard problem): don't targeted-regrow the right structure --
keep a diverse standing pool + SELECT by replay. Selection is scoring (tractable); proposal is generation
(intractable). The remaining problem is which past content to keep & replay. Claim (user): attribute content
by its IMPACT ON LEARNING and keep HIGH-VALUE REPRESENTATIVE + EDGE-CASE content -- representative preserves
the bulk, edge cases preserve the tail (the computed/exception structure variance-greedy compression drops).

Experiment: learn family A -> learn NEW family B WITH a small A-replay buffer -> measure A-retention (and B).
Race buffer-curation strategies at a fixed tiny memory budget M:
  none            -> catastrophic forgetting (lower bound)
  full            -> rehearse all of A (upper bound)
  random          -> M random A examples
  representative  -> M by k-center coverage on concept-firing features (bulk)
  edge            -> M by top learning-impact (high A-loss = boundary/surprising)
  rep+edge        -> M/2 representative + M/2 edge (the proposal)
If rep+edge ~ full at a fraction of the memory and beats random/rep-only/edge-only, over-generate+select+
replay is the escape: a small impact-curated buffer, no targeted regrow.

Run: cd pil && .venv/bin/python experiments/replay_curation.py
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

V, L, K, R = 15, 5, 10, 24
KT = 6
A_CATS, B_CATS = [0, 1, 2], [3, 4, 5]


def plant(seed):
    rng = random.Random(seed)
    cat = list(range(KT)) + [rng.randrange(KT) for _ in range(V - KT)]
    rng.shuffle(cat)
    return cat


def specs_for(cats):
    return [(kind, c) for c in cats for kind in ("first", "contains")]


A_SPECS, B_SPECS = specs_for(A_CATS), specs_for(B_CATS)
SPECS = A_SPECS + B_SPECS
A_IDS, B_IDS = list(range(len(A_SPECS))), list(range(len(A_SPECS), len(SPECS)))


def label(seq, cat, spec):
    cs = [cat[t] for t in seq]
    return int(cs[0] == spec[1]) if spec[0] == "first" else int(spec[1] in cs)


def gen(n, cat, seed):
    rng = random.Random(seed)
    X = [[rng.randrange(V) for _ in range(L)] for _ in range(n)]
    Y = [torch.tensor([label(s, cat, sp) for s in X]).float() for sp in SPECS]
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
    return lit.prod(2)                                       # (B,R)


def logits(s, x, tau=0.5):
    return fire(s, x, tau) @ s.heads[:, :-1].T + s.heads[:, -1]


def train_A(s, X, Y, steps=2500, lr=0.04):
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


def train_B_replay(s, X, Y, buf_idx, steps=2000, lr=0.04, alpha=1.0):
    """Learn B; if buf_idx given, REHEARSE those A examples (through A-heads) each step."""
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


def acc(s, X, Y, ids):
    with torch.no_grad():
        lg = logits(s, X, 0.4)
        return sum(float(((lg[:, ti] > 0).float() == Y[ti]).float().mean()) for ti in ids) / len(ids)


# ---- curation: attribute A examples by learning-impact + coverage ----
def a_loss_per_example(s, X, Y):
    with torch.no_grad():
        lg = logits(s, X, 0.4)
        per = torch.stack([F.binary_cross_entropy_with_logits(lg[:, ti], Y[ti], reduction="none")
                           for ti in A_IDS]).mean(0)           # (N,) mean A-loss = learning impact
    return per


def kcenter(feats, M, seed=0):
    """Greedy farthest-point coverage on the concept-firing features (representative selection)."""
    g = torch.Generator().manual_seed(seed)
    n = feats.shape[0]
    first = int(torch.randint(n, (1,), generator=g))
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
        g = torch.Generator().manual_seed(0)
        return torch.randperm(n, generator=g)[:M].tolist()
    impact = a_loss_per_example(s, X, Y)
    feats = fire(s, X, 0.4)
    if strategy == "edge":
        return impact.argsort(descending=True)[:M].tolist()
    if strategy == "representative":
        return kcenter(feats, M)
    if strategy == "rep+edge":
        edge = impact.argsort(descending=True)[:M // 2].tolist()
        rep = kcenter(feats, M - len(edge))
        return list(dict.fromkeys(rep + edge))                # dedup, preserve
    raise ValueError(strategy)


def main():
    M = 60          # tiny buffer budget (vs 3000 training examples = 2%)
    seeds = [0, 1]
    strategies = ["none", "random", "representative", "edge", "rep+edge", "full"]
    print(f"replay curation: learn A -> learn B with an M={M}-example A-buffer (2% of 3000). retain A?\n")
    print(f"{'strategy':>16}{'A_after_A':>11}{'A_retained':>12}{'B_learned':>11}{'buffer':>8}")
    rows = {st: [] for st in strategies}
    for sd in seeds:
        cat = plant(sd)
        Xtr, Ytr = gen(3000, cat, sd + 10)
        Xte, Yte = gen(3000, cat, sd + 99)
        base = Net(sd)
        train_A(base, Xtr, Ytr)
        aA = acc(base, Xte, Yte, A_IDS)
        for st in strategies:
            buf = select(st, base, Xtr, Ytr, M)
            s = base.clone()
            train_B_replay(s, Xtr, Ytr, buf)
            rows[st].append((aA, acc(s, Xte, Yte, A_IDS), acc(s, Xte, Yte, B_IDS), len(buf)))
    for st in strategies:
        r = rows[st]
        n = len(r)
        aA = sum(x[0] for x in r) / n
        aR = sum(x[1] for x in r) / n
        bL = sum(x[2] for x in r) / n
        buf = int(sum(x[3] for x in r) / n)
        print(f"{st:>16}{aA:>11.3f}{aR:>12.3f}{bL:>11.3f}{buf:>8}", flush=True)
    print(f"\nread: 'none' = catastrophic forgetting (floor), 'full' = all-of-A rehearsal (ceiling). If "
          f"'rep+edge' at M={M} (2%) approaches 'full' and beats random/representative/edge alone, then a "
          "small IMPACT-CURATED representative+edge buffer is the escape: over-generate+select+replay "
          "recovers continual learning with NO targeted regrow -- just a good, tiny memory.")


if __name__ == "__main__":
    main()
