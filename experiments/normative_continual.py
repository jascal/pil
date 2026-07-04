"""NORMATIVE head-to-head: does Wyly's consolidation beat baselines at continual concept learning?

The first test of the NORMATIVE claim (Wyly BETTER, not just descriptive). Real pythia-70m residuals +
6 concept labels (concepts_pythia70m.pt), split into task families A and B, learned SEQUENTIALLY
(A then B, never A again). All learners get MATCHED total feature capacity K = K_A + K_B. Metric after B:
retain-A + learn-B, and the balanced min.

Learners:
  wyly   -- frozen-core consolidation: K_A concept-hyperplanes for A, FROZEN at sleep; K_B fresh plastic
               concept-hyperplanes for B. A decoded via frozen A-concepts, B via plastic ones.
  naive     -- one shared K-feature bank + linear heads; train A then B (features drift -> forget A).
  ewc       -- naive + a Fisher/quadratic penalty pinning the feature bank near its post-A values during B.
  joint     -- CEILING: train the K-feature bank on A and B together.
  frozen    -- FLOOR-ish control: random frozen K features (no learning of the bank) + heads for A then B.

wyly >= ewc on the balanced metric = a genuine normative win (parity/better vs a real CL method, plus the
features are interpretable directions). wyly < ewc = an honest negative. Reported straight.

Run: cd pil && .venv/bin/python experiments/normative_continual.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "concepts_pythia70m.pt"
KA, KB = 8, 8                        # per-family concept capacity (total K = 16)
A_IDS, B_IDS = [0, 1, 2], [3, 4, 5]  # concept-label columns for family A vs B


class Bank(torch.nn.Module):
    """K concept-hyperplanes on residual r; membership = sigmoid(<r,u>-b); per-task linear heads."""
    def __init__(self, d, K, ntask):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(K, d) * (1.0 / d ** 0.5))
        self.b = torch.nn.Parameter(torch.zeros(K))
        self.heads = torch.nn.Parameter(torch.zeros(ntask, K + 1))

    def memb(self, r, tau=0.6):
        return torch.sigmoid((r @ self.U.T - self.b) / tau)

    def logit(self, ti, r):
        m = self.memb(r)
        return m @ self.heads[ti, :-1] + self.heads[ti, -1]


def fit(bank, r, Y, task_ids, steps, mask=None, ewc=None, lr=0.02):
    """Train task_ids (heads + bank). mask freezes params (grad*=mask); ewc = (theta0, fisher, lam)."""
    opt = torch.optim.Adam(bank.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    n = r.shape[0]
    for _ in range(steps):
        idx = torch.randint(n, (256,), generator=g)
        loss = sum(F.binary_cross_entropy_with_logits(bank.logit(ti, r[idx]), Y[ti][idx]) for ti in task_ids)
        loss = loss / len(task_ids)
        if ewc is not None:
            (th0, fish, lam) = ewc
            loss = loss + lam * (fish * (bank.U - th0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        if mask is not None:
            bank.U.grad *= mask["U"]
            bank.b.grad *= mask["b"]
            bank.heads.grad *= mask["heads"]
        opt.step()


def acc(bank, r, Y, task_ids):
    with torch.no_grad():
        return sum(float(((bank.logit(ti, r) > 0).float() == Y[ti]).float().mean())
                   for ti in task_ids) / len(task_ids)


def fisher_U(bank, r, Y, task_ids, n=2000):
    """Diagonal Fisher on the feature bank U from task A (for EWC)."""
    idx = torch.randperm(r.shape[0])[:n]
    g2 = torch.zeros_like(bank.U)
    for ti in task_ids:
        bank.zero_grad()
        loss = F.binary_cross_entropy_with_logits(bank.logit(ti, r[idx]), Y[ti][idx])
        loss.backward()
        g2 += bank.U.grad.detach() ** 2
    return g2 / len(task_ids)


def evaluate(name, r, Y, seed):
    torch.manual_seed(seed)
    d, K = r.shape[1], KA + KB
    tr = torch.arange(int(0.8 * r.shape[0]))
    te = torch.arange(int(0.8 * r.shape[0]), r.shape[0])
    Rtr, Rte = r[tr], r[te]
    Ytr = [Y[i][tr] for i in range(len(Y))]
    Yte = [Y[i][te] for i in range(len(Y))]

    if name == "wyly":
        bank = Bank(d, K, len(Y))
        # learn A on the FIRST KA concepts only (mask others off)
        mA = {"U": torch.zeros(K, d), "b": torch.zeros(K), "heads": torch.zeros(len(Y), K + 1)}
        mA["U"][:KA] = 1
        mA["b"][:KA] = 1
        for t in A_IDS:
            mA["heads"][t] = 1
            mA["heads"][t, KA:K] = 0     # A heads read only A-concepts
        fit(bank, Rtr, Ytr, A_IDS, 3000, mask=mA)
        # SLEEP: freeze A-concepts; learn B on the plastic KB concepts
        mB = {"U": torch.zeros(K, d), "b": torch.zeros(K), "heads": torch.zeros(len(Y), K + 1)}
        mB["U"][KA:K] = 1
        mB["b"][KA:K] = 1
        for t in B_IDS:
            mB["heads"][t] = 1
            mB["heads"][t, :KA] = 0       # B heads read plastic-only
        fit(bank, Rtr, Ytr, B_IDS, 3000, mask=mB)
        return acc(bank, Rte, Yte, A_IDS), acc(bank, Rte, Yte, B_IDS)

    if name == "wyly+reuse":
        # like wyly, but B-heads ALSO read the frozen A-concepts (forward transfer / reuse)
        bank = Bank(d, K, len(Y))
        mA = {"U": torch.zeros(K, d), "b": torch.zeros(K), "heads": torch.zeros(len(Y), K + 1)}
        mA["U"][:KA] = 1
        mA["b"][:KA] = 1
        for t in A_IDS:
            mA["heads"][t] = 1
            mA["heads"][t, KA:K] = 0
        fit(bank, Rtr, Ytr, A_IDS, 3000, mask=mA)
        mB = {"U": torch.zeros(K, d), "b": torch.zeros(K), "heads": torch.zeros(len(Y), K + 1)}
        mB["U"][KA:K] = 1
        mB["b"][KA:K] = 1
        for t in B_IDS:
            mB["heads"][t] = 1                            # B heads read ALL concepts (frozen A + plastic B)
        fit(bank, Rtr, Ytr, B_IDS, 3000, mask=mB)
        return acc(bank, Rte, Yte, A_IDS), acc(bank, Rte, Yte, B_IDS)

    if name == "packnet":
        # train ALL K on A -> prune to top-KA important concepts (frozen) -> release rest -> learn B
        bank = Bank(d, K, len(Y))
        fit(bank, Rtr, Ytr, A_IDS, 3000)
        imp = bank.heads.detach()[A_IDS][:, :K].abs().sum(0)     # concept importance for A
        keep = imp.argsort(descending=True)[:KA]
        frozen = torch.zeros(K, dtype=torch.bool)
        frozen[keep] = True
        kf = frozen.float()
        with torch.no_grad():
            for t in A_IDS:                              # A reads ONLY its kept (frozen) concepts
                bank.heads[t, :K][~frozen] = 0.0
        # PackNet fine-tune: re-tune A-heads on the KEPT (frozen) concepts to recover A after pruning
        mFT = {"U": torch.zeros(K, d), "b": torch.zeros(K), "heads": torch.zeros(len(Y), K + 1)}
        for t in A_IDS:
            mFT["heads"][t, :K] = kf                     # only kept-concept A-head weights train
            mFT["heads"][t, -1] = 1
        fit(bank, Rtr, Ytr, A_IDS, 1500, mask=mFT)
        with torch.no_grad():                            # NOW release the pruned concepts for B
            bank.U[~frozen] = torch.randn(int((~frozen).sum()), d) * (1.0 / d ** 0.5)
            bank.b[~frozen] = 0.0
        rel = (~frozen).float()
        mB = {"U": rel[:, None].expand(K, d).contiguous(), "b": rel,
              "heads": torch.zeros(len(Y), K + 1)}
        for t in B_IDS:
            mB["heads"][t] = 1                            # B heads read ALL K (frozen A + released)
        fit(bank, Rtr, Ytr, B_IDS, 3000, mask=mB)
        return acc(bank, Rte, Yte, A_IDS), acc(bank, Rte, Yte, B_IDS)

    if name == "frozen":
        bank = Bank(d, K, len(Y))
        mf = {"U": torch.zeros(K, d), "b": torch.zeros(K), "heads": torch.ones(len(Y), K + 1)}
        fit(bank, Rtr, Ytr, A_IDS, 3000, mask=mf)            # only heads move
        fit(bank, Rtr, Ytr, B_IDS, 3000, mask=mf)
        return acc(bank, Rte, Yte, A_IDS), acc(bank, Rte, Yte, B_IDS)

    if name == "joint":                                      # ceiling
        bank = Bank(d, K, len(Y))
        fit(bank, Rtr, Ytr, A_IDS + B_IDS, 6000)
        return acc(bank, Rte, Yte, A_IDS), acc(bank, Rte, Yte, B_IDS)

    # naive / ewc: shared K-bank, learn A then B
    bank = Bank(d, K, len(Y))
    fit(bank, Rtr, Ytr, A_IDS, 3000)
    ewc = None
    if name == "ewc":
        fish = fisher_U(bank, Rtr, Ytr, A_IDS)
        fish = fish / (fish.mean() + 1e-9)
        ewc = (bank.U.detach().clone(), fish, 30.0)
    fit(bank, Rtr, Ytr, B_IDS, 3000, ewc=ewc)
    return acc(bank, Rte, Yte, A_IDS), acc(bank, Rte, Yte, B_IDS)


def main():
    d = torch.load(DATA)
    r = d["r"].float()
    r = (r - r.mean(0)) / (r.std(0) + 1e-6)          # per-dim standardize (pythia residual norm ~440
    #                                                  saturates the membership sigmoid otherwise)
    names = d["names"]
    Y = [d["labels"][:, i].float() for i in range(d["labels"].shape[1])]
    print(f"normative continual learning on pythia-70m residuals  r {tuple(r.shape)}")
    print(f"task A = {[names[i] for i in A_IDS]}  ->  task B = {[names[i] for i in B_IDS]}  "
          f"(matched capacity K={KA + KB})\n")
    print(f"{'learner':>10}{'retain_A':>10}{'learn_B':>10}{'balanced':>10}")
    rows = {}
    for name in ["frozen", "naive", "ewc", "packnet", "wyly", "wyly+reuse", "joint"]:
        res = [evaluate(name, r, Y, s) for s in (0, 1, 2, 3)]
        aA = sum(x[0] for x in res) / len(res)
        aB = sum(x[1] for x in res) / len(res)
        bal = sum(min(x[0], x[1]) for x in res) / len(res)
        rows[name] = (aA, aB, bal)
        print(f"{name:>10}{aA:>10.3f}{aB:>10.3f}{bal:>10.3f}", flush=True)
    best = max(rows, key=lambda k: rows[k][2])
    cop = max(rows["wyly"][2], rows["wyly+reuse"][2])
    print(f"\nbest balanced: {best} {rows[best][2]:.3f}  |  wyly(best) {cop:.3f}  "
          f"packnet {rows['packnet'][2]:.3f}  ewc {rows['ewc'][2]:.3f}")
    print("read: PACKNET is the fair strong baseline (also parameter-isolation: full-capacity A -> prune -> "
          "release for B, forward transfer). If wyly ~ packnet, Wyly is AT PARITY with the best "
          "method (its extra value is then interpretability, not raw CL). Beating packnet would need the "
          "reuse/compositional structure to pay off -- unrelated surface concepts likely give parity. "
          "wyly+reuse lets B read frozen A-concepts (the transfer packnet also has).")


if __name__ == "__main__":
    main()
