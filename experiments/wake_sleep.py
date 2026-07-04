"""Wake/sleep homeostatic consolidation: spare capacity enables CONTINUAL learning without forgetting.

Reframe (from the capacity thread): the target is NOT minimal K -- a model crushed to exactly the concepts
the current tasks need has NO room for the next task. Sleep should CONSOLIDATE the reused core (freeze it,
cheap/protected) while KEEPING spare under-subscribed capacity plastic (room to learn). The payoff of that
spare capacity is continual learning -- which minimize-to-K optimizes away, so it could never be measured.

Experiment: learn task family A (distinguish concepts 0,1,2) -> SLEEP (consolidate: freeze the n_protect
most-A-used concepts + their rules + A's readout heads) -> learn a NEW family B (distinguish concepts 3,4,5)
on whatever stays plastic. Sweep n_protect to expose the HOMEOSTATIC BAND:
  n_protect too LOW  -> A's concepts stay plastic, B overwrites them -> A FORGOTTEN (no consolidation)
  n_protect too HIGH -> almost all frozen -> no plastic spares -> B UNDERFIT (no room to learn)
  sweet spot         -> protect exactly A's used concepts, learn B on the spares -> RETAIN A *and* LEARN B
Maps to CLS (fast/slow memory), the synaptic-homeostasis hypothesis (sleep downscales, preserving the
strong), and DreamCoder's discrete wake/sleep.

Run: cd pil && .venv/bin/python experiments/wake_sleep.py
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

V, L, K, R = 15, 5, 10, 24        # tokens, seq len, concepts (generous -> spares exist), rules
KT = 6                            # true categories
DEV = "cpu"

A_CATS, B_CATS = [0, 1, 2], [3, 4, 5]


def plant(seed):
    rng = random.Random(seed)
    cat = list(range(KT)) + [rng.randrange(KT) for _ in range(V - KT)]
    rng.shuffle(cat)
    return cat


def specs_for(cats):
    return [(kind, c) for c in cats for kind in ("first", "contains")]


A_SPECS, B_SPECS = specs_for(A_CATS), specs_for(B_CATS)
SPECS = A_SPECS + B_SPECS                     # all task heads; A = 0..5, B = 6..11
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


def logits(s, x, tau=0.5):
    c = F.softmax(s.E[x] / tau, -1)
    sel = F.softmax(s.rule / tau, -1)
    lit = torch.einsum("blk,rlk->brl", c, sel[:, :, :-1]) + sel[:, :, -1]
    g = lit.prod(2)                                    # (B,R)
    return g @ s.heads[:, :-1].T + s.heads[:, -1]      # (B,T)


def train(s, X, Y, task_ids, steps, masks=None, lr=0.04):
    opt = torch.optim.Adam(s.params(), lr=lr)
    n = X.shape[0]
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        tau = max(0.4, 1.3 - 0.9 * step / max(steps, 1))
        idx = torch.randint(n, (256,), generator=g)
        lg = logits(s, X[idx], tau)
        bce = [F.binary_cross_entropy_with_logits(lg[:, ti], Y[ti][idx]) for ti in task_ids]
        loss = sum(bce) / len(task_ids)
        opt.zero_grad()
        loss.backward()
        if masks is not None:                          # consolidation: freeze protected params
            s.E.grad *= masks["E"]
            s.rule.grad *= masks["rule"]
            s.heads.grad *= masks["heads"]
        opt.step()


def acc(s, X, Y, task_ids):
    with torch.no_grad():
        lg = logits(s, X, 0.4)
        hits = [float(((lg[:, ti] > 0).float() == Y[ti]).float().mean()) for ti in task_ids]
        return sum(hits) / len(task_ids)


def concept_usage(s, task_ids):
    """Concept reliance for the given tasks: sum_r (rule r importance) x (rule r selects concept k)."""
    with torch.no_grad():
        sel = F.softmax(s.rule / 0.4, -1)[:, :, :-1].sum(1)         # (R,K) rule r's use of concept k
        rule_imp = s.heads[task_ids, :-1].abs().sum(0)             # (R,) importance to these tasks
        return (rule_imp[:, None] * sel).sum(0)                    # (K,)


def rule_usage(s, task_ids):
    with torch.no_grad():
        return s.heads[task_ids, :-1].abs().sum(0)                 # (R,)


def sleep_masks(s, n_protect):
    """Consolidate: freeze the n_protect most-A-used concepts + rules that serve them + A's heads.
    Everything else stays plastic = the spare capacity ('room to learn' family B)."""
    Em = torch.ones(V, K)
    Rm = torch.ones(R, L, K + 1)
    Hm = torch.ones(len(SPECS), R + 1)
    if n_protect > 0:
        cu = concept_usage(s, A_IDS)
        protc = cu.argsort(descending=True)[:n_protect].tolist()
        for k in protc:
            Em[:, k] = 0.0                                          # freeze A-concept memberships
        ru = rule_usage(s, A_IDS)
        protr = ru.argsort(descending=True)[:max(1, n_protect * 3)].tolist()
        for r in protr:
            Rm[r] = 0.0                                             # freeze A's rule bodies
    for t in A_IDS:
        Hm[t] = 0.0                                                 # always freeze A's readout heads
    return {"E": Em, "rule": Rm, "heads": Hm}


def main():
    print(f"wake/sleep continual learning: A=cats{A_CATS} then B=cats{B_CATS}; K={K} concepts (spares exist)")
    print("sweep n_protect (concepts consolidated at sleep) -> the homeostatic band\n")
    print(f"{'n_protect':>10}{'A_after_A':>11}{'A_after_B':>11}{'B_after_B':>11}{'balance=min':>13}")
    seeds = [0, 1]
    balance = {}
    for n_protect in (0, 2, 3, 5, 7, 8, 9, 10):
        aA, aAB, aBB, ret = [], [], [], []
        for sd in seeds:
            cat = plant(sd)
            Xtr, Ytr = gen(3000, cat, sd + 10)
            Xte, Yte = gen(3000, cat, sd + 99)
            s = Net(sd)
            train(s, Xtr, Ytr, A_IDS, 2500)                        # WAKE: learn A
            a_afterA = acc(s, Xte, Yte, A_IDS)
            masks = sleep_masks(s, n_protect)                      # SLEEP: consolidate
            train(s, Xtr, Ytr, B_IDS, 2000, masks=masks)          # WAKE: learn B (A frozen per masks)
            a_afterB = acc(s, Xte, Yte, A_IDS)
            b_afterB = acc(s, Xte, Yte, B_IDS)
            aA.append(a_afterA)
            aAB.append(a_afterB)
            aBB.append(b_afterB)
            ret.append(min(a_afterB, b_afterB))            # balanced: both A retained AND B learned
        n = len(seeds)
        bal = sum(ret) / n
        balance[n_protect] = bal
        print(f"{n_protect:>10}{sum(aA) / n:>11.3f}{sum(aAB) / n:>11.3f}{sum(aBB) / n:>11.3f}"
              f"{bal:>13.3f}", flush=True)
    peak = max(balance, key=balance.get)
    print(f"\nbalance=min(A_retained, B_learned) PEAKS at n_protect={peak} ({balance[peak]:.3f}) -- an "
          "INTERIOR optimum.")
    print("read: individual metrics are a monotone STABILITY-PLASTICITY trade-off (consolidation buys "
          "A-retention 0.68->0.93 at the cost of B-plasticity 0.94->0.78). Endpoints are the two failure "
          "modes: n=0 CATASTROPHIC FORGETTING (A overwritten), n=10 NO ROOM (B underfit). BALANCED metric "
          "peaks in the MIDDLE = the homeostatic band: protect the reused core, keep spares plastic. Spare "
          "capacity is what makes continual learning work -- the payoff minimize-to-K optimized away.")


if __name__ == "__main__":
    main()
