"""DECISIVE normative test: does Wyly's reuse beat PackNet when B REUSES A (compositional, low budget)?

Rung-2 found Wyly ties PackNet on UNRELATED continual tasks (reuse got no chance). Here B's concepts are
explicit compositions of A's PRIMITIVES, so cross-task reuse is genuinely available -- the setting where
Wyly's design is supposed to pay off. On real pythia-70m residuals:
  A (primitives): func, punct, cap
  B (compositions of A): B0 = func AND cap ,  B1 = punct OR cap ,  B2 = cap AND NOT func   (derived labels)
We SWEEP the B training budget (few -> many examples). A learner that can REUSE clean A-concepts should hold
B accuracy at LOW budget; one that must learn B from scratch should collapse.

Learners: wyly+reuse (B reads frozen A-concepts) · wyly-isolated (fresh B concepts, no reuse) ·
packnet (prune A, B reads frozen-A transfer + released) · scratch (fresh bank on residual, no A).
Decisive: wyly+reuse > packnet at low budget = the NOVEL Wyly win; tie = interpretability-at-parity.

Run: cd pil && .venv/bin/python experiments/normative_compositional.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import normative_continual as NC  # noqa: E402

SP = NC.SP
KA, KB = 8, 8
A_IDS = [0, 1, 2]                        # func, punct, cap (label columns)
BUDGETS = [100, 300, 1000, 4000]


def make_B(labels):
    """Compositions of A-primitives (func=0, punct=1, cap=2) -> 3 derived binary B-tasks."""
    func, punct, cap = labels[:, 0], labels[:, 1], labels[:, 2]
    B = torch.stack([
        ((func == 1) & (cap == 1)).float(),        # func AND cap
        ((punct == 1) | (cap == 1)).float(),       # punct OR cap
        ((cap == 1) & (func == 0)).float(),        # cap AND NOT func
    ], 1)
    return B


def fit_budget(bank, r, Y, task_ids, steps, n_budget, mask=None, seed=0):
    """Train on only n_budget examples (subsample) -- the reuse/data-efficiency stress."""
    g = torch.Generator().manual_seed(seed)
    sub = torch.randperm(r.shape[0], generator=g)[:n_budget]
    NC.fit(bank, r[sub], [y[sub] for y in Y], task_ids, steps, mask=mask)


def run(name, r, YA, YB, nB, seed):
    torch.manual_seed(seed)
    d, K = r.shape[1], KA + KB
    ntask = len(YA) + len(YB)
    Y = YA + YB
    aids = list(range(len(YA)))
    bids = list(range(len(YA), ntask))
    tr = torch.arange(int(0.8 * r.shape[0]))
    te = torch.arange(int(0.8 * r.shape[0]), r.shape[0])
    Rtr, Rte = r[tr], r[te]
    Ytr = [y[tr] for y in Y]
    Yte = [y[te] for y in Y]

    def zeros_mask():
        return {"U": torch.zeros(K, d), "b": torch.zeros(K), "heads": torch.zeros(ntask, K + 1)}

    # learn A (full A budget) on the first KA concepts
    bank = NC.Bank(d, K, ntask)
    mA = zeros_mask()
    mA["U"][:KA] = 1
    mA["b"][:KA] = 1
    for t in aids:
        mA["heads"][t] = 1
        mA["heads"][t, KA:K] = 0
    NC.fit(bank, Rtr, Ytr, aids, 3000, mask=mA)

    if name == "packnet":                               # prune A -> B reads frozen-A + released
        imp = bank.heads.detach()[aids][:, :K].abs().sum(0)
        keep = imp.argsort(descending=True)[:KA]
        frozen = torch.zeros(K, dtype=torch.bool)
        frozen[keep] = True
        with torch.no_grad():
            for t in aids:
                bank.heads[t, :K][~frozen] = 0.0
        mFT = zeros_mask()
        for t in aids:
            mFT["heads"][t, :K] = frozen.float()
            mFT["heads"][t, -1] = 1
        NC.fit(bank, Rtr, Ytr, aids, 1500, mask=mFT)
        with torch.no_grad():
            bank.U[~frozen] = torch.randn(int((~frozen).sum()), d) * (1.0 / d ** 0.5)
            bank.b[~frozen] = 0.0
        mB = {"U": (~frozen).float()[:, None].expand(K, d).contiguous(), "b": (~frozen).float(),
              "heads": torch.zeros(ntask, K + 1)}
        for t in bids:
            mB["heads"][t] = 1
        fit_budget(bank, Rtr, Ytr, bids, 3000, nB, mask=mB, seed=seed)
        return NC.acc(bank, Rte, Yte, aids), NC.acc(bank, Rte, Yte, bids)

    # wyly+reuse / wyly-isolated: freeze A-concepts, learn B on plastic concepts at budget nB
    mB = zeros_mask()
    mB["U"][KA:K] = 1
    mB["b"][KA:K] = 1
    for t in bids:
        mB["heads"][t] = 1
        if name == "wyly-isolated":
            mB["heads"][t, :KA] = 0                     # B reads ONLY plastic B-concepts (no reuse)
    fit_budget(bank, Rtr, Ytr, bids, 3000, nB, mask=mB, seed=seed)
    return NC.acc(bank, Rte, Yte, aids), NC.acc(bank, Rte, Yte, bids)


def run_scratch(r, YB, nB, seed):
    """No A: learn B directly on a fresh K-bank from the residual (must build compositions from scratch)."""
    torch.manual_seed(seed)
    d, K = r.shape[1], KA + KB
    tr = torch.arange(int(0.8 * r.shape[0]))
    te = torch.arange(int(0.8 * r.shape[0]), r.shape[0])
    bank = NC.Bank(d, K, len(YB))
    fit_budget(bank, r[tr], [y[tr] for y in YB], list(range(len(YB))), 3000, nB, seed=seed)
    return NC.acc(bank, r[te], [y[te] for y in YB], list(range(len(YB))))


def main():
    dd = torch.load(SP / "concepts_pythia70m.pt")
    r = dd["r"].float()
    r = (r - r.mean(0)) / (r.std(0) + 1e-6)
    labels = dd["labels"]
    YA = [labels[:, i].float() for i in A_IDS]
    YB = [make_B(labels)[:, j] for j in range(3)]
    print("DECISIVE normative test: B = compositions of A-primitives (reuse available), sweep B budget")
    print("A = [func,punct,cap] -> B = [func&cap, punct|cap, cap&~func]  (real pythia-70m)\n")
    cols = ["wyly+reuse", "wyly-isol", "packnet", "scratch"]
    print(f"{'B-budget':>9}" + "".join(f"{c:>15}" for c in cols))
    for nB in BUDGETS:
        row = []
        for name in ["wyly+reuse", "wyly-isolated", "packnet"]:
            res = [run(name, r, YA, YB, nB, s) for s in (0, 1, 2)]
            row.append(sum(x[1] for x in res) / len(res))          # B accuracy
        sc = sum(run_scratch(r, YB, nB, s) for s in (0, 1, 2)) / 3
        row.append(sc)
        print(f"{nB:>9}" + "".join(f"{v:>16.3f}" for v in row), flush=True)
    print("\nread: reuse-learners (wyly+reuse, packnet) holding B at LOW budget while isolated/scratch\n"
          "collapse = REUSE pays off. wyly+reuse > packnet at low budget = the NOVEL Wyly win; ~ = at\n"
          "parity even WITH reuse ('better' = interpretability only).")


if __name__ == "__main__":
    main()
