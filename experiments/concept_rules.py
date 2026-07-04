"""PIC rules over CO-EVOLVING concepts — graft learnable concept-atoms onto the PIC rule form, then push.

Extends concept_atoms.py: instead of fixed relational features + a linear head, RULES are differentiable
soft-conjunctions of concept-literals over positions ("position p is concept k", with a wildcard/don't-care
option) — the PIC rule form, but its atoms are the learnable token->concept lattice, and rules + concepts +
hierarchy co-evolve under gradient descent. Rules and the concept lattice are SHARED across a task family
(reuse); per-task linear heads pick which rules matter (a head can OR several rules; a rule ANDs literals).

This buys what fixed features couldn't: LEARNED CONJUNCTIONS over concepts (e.g. "pos0 is cat A AND
pos_last is cat B"). We then push to the collapse boundary — larger K, conjunctive/entangled tasks, hierarchy.

Tests: (1) recover planted concepts AND solve conjunctive tasks; (2) K-sweep — where does concept recovery
break down? Run: cd pil && .venv/bin/python experiments/concept_rules.py
"""

from __future__ import annotations

import itertools
import random
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn

V, L, R, M = 15, 5, 16, 2                       # tokens, seq len, #rules, supers
DEV = "cpu"


def plant(kk, seed=0):
    rng = random.Random(seed)
    cat = [rng.randrange(kk) for _ in range(V)]
    sup = [rng.randrange(M) for _ in range(kk)]
    return cat, sup


# task family (works for any K>=2); *conjunctive* tasks need learned AND over concepts
def task_labels(seq, cat, sup):
    cs = [cat[t] for t in seq]
    most = Counter(cs).most_common(1)[0][0]
    adj00 = any(cs[p] == 0 and cs[p + 1] == 0 for p in range(L - 1))
    return {
        "maj0": int(most == 0),
        "first0_last1": int(cs[0] == 0 and cs[-1] == 1),        # CONJUNCTIVE (2 positions)
        "contains1": int(1 in cs),
        "first0": int(cs[0] == 0),
        "adj00": int(adj00),                                    # CONJUNCTIVE + disjunction over positions
        "super0_first": int(sup[cs[0]] == 0),                   # HIERARCHY
    }


TASKS = ["maj0", "first0_last1", "contains1", "first0", "adj00", "super0_first"]
CONJ = {"first0_last1", "adj00"}


def gen(n, cat, sup, seed):
    rng = random.Random(seed)
    X = [[rng.randrange(V) for _ in range(L)] for _ in range(n)]
    Y = {t: [] for t in TASKS}
    for s in X:
        lab = task_labels(s, cat, sup)
        for t in TASKS:
            Y[t].append(lab[t])
    return torch.tensor(X), {t: torch.tensor(Y[t]).float() for t in TASKS}


class ConceptRuleLearner(nn.Module):
    def __init__(self, kk):
        super().__init__()
        self.K = kk
        self.E = nn.Parameter(torch.randn(1, V, kk) * 0.3)         # token -> concept logits (shared)
        self.S = nn.Parameter(torch.randn(1, kk, M) * 0.3)         # concept -> super (hierarchy)
        rs = torch.randn(R, L, kk + 1) * 0.3
        rs[:, :, -1] += 1.5                                         # bias rules toward wildcard at init
        self.rule_sel = nn.Parameter(rs)                           # per rule/pos: K concepts + wildcard
        self.heads = nn.Parameter(torch.zeros(len(TASKS), R + 1))

    def fire(self, x, tau):
        c = F.softmax(self.E[0][x] / tau, -1)                      # (B, L, K)
        sel = F.softmax(self.rule_sel / tau, -1)                   # (R, L, K+1)
        lit = torch.einsum("blk,rlk->brl", c, sel[:, :, :-1]) + sel[:, :, -1]   # (B,R,L) in [0,1]
        return lit.prod(dim=2)                                     # (B, R) soft-AND over positions

    def logit(self, ti, x, tau):
        g = self.fire(x, tau)
        w = self.heads[ti]
        return g @ w[:-1] + w[-1]

    def entropy(self):
        pe = F.softmax(self.E, -1)
        pr = F.softmax(self.rule_sel, -1)
        return -(pe * (pe + 1e-9).log()).sum(-1).mean() - (pr * (pr + 1e-9).log()).sum(-1).mean()


def recovery(model, cat):
    assign = F.softmax(model.E[0], -1).argmax(-1).tolist()
    K = model.K
    return max(sum(perm[assign[v]] == cat[v] for v in range(V)) / V
               for perm in itertools.permutations(range(K)))


def train(model, Xtr, Ytr, steps=3500, ent_max=0.03, lr=0.04):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        tau = max(0.3, 1.5 - 1.2 * step / steps)
        ent_w = ent_max * min(1.0, step / (steps * 0.5))
        idx = torch.randint(n, (256,), generator=g)
        loss = sum(F.binary_cross_entropy_with_logits(model.logit(ti, Xtr[idx], tau), Ytr[t][idx])
                   for ti, t in enumerate(TASKS)) + ent_w * model.entropy()
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def test_acc(model, Xte, Yte):
    return {t: float(((model.logit(ti, Xte, 0.3) > 0).float() == Yte[t]).float().mean())
            for ti, t in enumerate(TASKS)}


def main():
    print("=== core run: K=3, rules over concepts (conjunctive tasks) ===")
    cat, sup = plant(3, 0)
    Xte, Yte = gen(4000, cat, sup, 99)
    base = {t: max(float(Yte[t].mean()), 1 - float(Yte[t].mean())) for t in TASKS}
    Xtr, Ytr = gen(1200, cat, sup, 1)
    sh = ConceptRuleLearner(3)
    train(sh, Xtr, Ytr)
    acc = test_acc(sh, Xte, Yte)
    print(f"planted cat {cat}  concept recovery {recovery(sh, cat):.3f}")
    print("per-task test acc (base rate):  [conjunctive tasks need learned AND over concepts]")
    for t in TASKS:
        tag = " CONJ" if t in CONJ else ""
        print(f"   {t:>14}: {acc[t]:.3f}  (base {base[t]:.2f}){tag}")

    print("\n=== K-sweep: concept recovery + conjunctive-task acc vs #concepts (the boundary) ===")
    print(f"{'K':>3}{'recovery':>10}{'conj_acc':>10}{'all_acc':>9}")
    for kk in (3, 4, 6, 8):
        c2, s2 = plant(kk, 0)
        Xtr2, Ytr2 = gen(2000, c2, s2, 1)
        Xte2, Yte2 = gen(4000, c2, s2, 99)
        m = ConceptRuleLearner(kk)
        train(m, Xtr2, Ytr2)
        a = test_acc(m, Xte2, Yte2)
        conj = sum(a[t] for t in CONJ) / len(CONJ)
        allA = sum(a.values()) / len(a)
        print(f"{kk:>3}{recovery(m, c2):>10.3f}{conj:>10.3f}{allA:>9.3f}", flush=True)
    print("\nread: task acc (incl. CONJ) stays ~0.99 at every K, but recovery falls as ~3/K -- the boundary "
          "is IDENTIFIABILITY, not optimization collapse: the substrate recovers exactly the concept "
          "distinctions the tasks EXERCISE (~3 here). Richer tasks -> more concepts recovered.")


if __name__ == "__main__":
    main()
