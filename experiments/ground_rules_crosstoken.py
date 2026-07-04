"""Is cross-token comparison DISTRIBUTED, or a RULE over multiple concepts a linear probe can't read?

The frontier finding: cross-token relations (is_repeat, prev_func) are only ~0.78 LINEARLY decodable vs
lexical ~0.94. I called them "distributed/computed". But a single linear hyperplane CANNOT express a RELATION
ACROSS concepts -- "current token's concept matches an earlier one" (is_repeat) or "the PREVIOUS position was
a function-concept" (prev_func) are inherently MULTI-CONCEPT RULES (soft-conjunctions), not single directions.
So the ~0.78 might mean "it's there as a rule over concepts, and a linear probe is the wrong reader."

Test: for each concept, compare full-rank balanced ceiling of
  linear -- single hyperplane (r @ W)                         [the frontier baseline]
  rule   -- K concept-hyperplanes + soft-conjunction RULES over them (the PIC/Wyly rule layer)
  mlp    -- 1 hidden layer (nonlinear upper bound)
If for cross-token concepts rule >> linear (approaching mlp), the cross-token structure is RULE-STRUCTURED
(multi-concept, compositional) -- recoverable by the rule form, NOT irreducibly distributed. For lexical,
rule ~ linear (single-concept, no rule needed).

Run: cd pil && .venv/bin/python experiments/ground_rules_crosstoken.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import ground_relational as GR  # noqa: E402

LEXICAL = ["func", "cap"]
CROSS = ["is_repeat", "local_repeat", "prev_func"]
STRUCT = ["inside_paren"]


class RuleNet(torch.nn.Module):
    """K concept-hyperplanes -> R soft-conjunction RULES (product of selected concept-literals) -> head."""
    def __init__(self, d, K=24, R=24):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(K, d) * (1.0 / d ** 0.5))
        self.b = torch.nn.Parameter(torch.zeros(K))
        req = torch.randn(R, K) * 0.3 - 1.0
        self.req = torch.nn.Parameter(req)                  # per rule: which concepts it requires (sigmoid)
        self.head = torch.nn.Parameter(torch.zeros(R + 1))

    def forward(self, r, tau=0.6):
        m = torch.sigmoid((r @ self.U.T - self.b) / tau)    # (B,K) memberships
        req = torch.sigmoid(self.req)                       # (R,K)
        term = 1 - req.unsqueeze(0) + req.unsqueeze(0) * m.unsqueeze(1)   # (B,R,K)
        fire = term.prod(-1)                                # (B,R) soft-AND
        return fire @ self.head[:-1] + self.head[-1]


class MLP(torch.nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))

    def forward(self, r):
        return self.net(r).squeeze(-1)


def train_acc(model, r, y, tr, te, steps=2500, lr=0.02, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = tr[torch.randint(len(tr), (256,), generator=g)]
        loss = F.binary_cross_entropy_with_logits(model(r[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((model(r[te]) > 0).float() == y[te]).float().mean())


def main():
    d = torch.load(GR.DATA)
    r = d["r"].float()
    names, labs = d["names"], d["labels"]
    print(f"cross-token: linear vs RULE vs mlp, pythia-70m  r {tuple(r.shape)} (balanced, full-rank)\n")
    print(f"{'concept':>13}{'linear':>9}{'rule':>8}{'mlp':>8}{'rule-lin':>10}   type")
    for nm in LEXICAL + CROSS + STRUCT:
        y = labs[:, names.index(nm)].float()
        if int(y.sum()) < 300:
            continue
        lin, rul, mlp = [], [], []
        for s in (0, 1):
            rk, yk, tr, te = GR.balanced(r, y, s)
            lin.append(GR.probe(rk[tr], yk[tr], rk[te], yk[te]))
            rul.append(train_acc(RuleNet(rk.shape[1]), rk, yk, tr, te, seed=s))
            mlp.append(train_acc(MLP(rk.shape[1]), rk, yk, tr, te, seed=s))
        lm, rm, mm = (sum(v) / len(v) for v in (lin, rul, mlp))
        typ = "lexical" if nm in LEXICAL else ("struct" if nm in STRUCT else "cross-token")
        print(f"{nm:>13}{lm:>9.3f}{rm:>8.3f}{mm:>8.3f}{rm - lm:>+10.3f}   {typ}", flush=True)
    print("\nread: for CROSS-TOKEN, RULE >> linear (~ mlp) = a RULE OVER MULTIPLE CONCEPTS (compositional,\n"
          "recoverable by the rule form), NOT irreducibly distributed -- the linear probe was the wrong\n"
          "reader. rule ~ linear < mlp = nonlinear but not rule-structured. rule~linear~mlp = residual\n"
          "genuinely lacks it (computed on-the-fly). LEXICAL: rule ~ linear expected (single-concept).")


if __name__ == "__main__":
    main()
