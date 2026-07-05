"""Correlation-guided PROPOSER for sparse Wyly rules -- the crux the scaling fixes isolated.

The sparse-gather rule form is cheap (O(R*KSLOT)) but with RANDOM incidence the rules add ~nothing (random
conjunctions are rarely useful; over-generate+select is too weak). The fix is a PROPOSER that chooses WHICH
literals to conjoin -- guided by correlation with prediction error, not random:
  in sleep, on high-loss/mispredicted examples, for a frequently-mispredicted target token t:
    score each literal by DISCRIMINATIVENESS for t = mean(literal | next==t) - mean(literal | next!=t);
    conjoin the top-KSLOT discriminative literals into a NEW rule, wired to predict t (head init).
So new rules COVER the errors the current model makes. Compared to the random proposer and the dense-softmax
(learnable-but-O(LIT)) reference. Still O(R*KSLOT) sparse -> if it approaches the dense reference, we have a
rule form BOTH scalable AND useful. (Backpocket alt: an external LM proposes which concepts to conjoin.)

Run: cd pil && .venv/bin/python experiments/wyly_lm_proposer.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "layers_pythia70m.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K0, CTX, V = 192, 4, 2048
RMAX, RINIT, RGROW, KSLOT = 1536, 64, 128, 4
WAKE_STEPS, LR, EPISODES = 500, 0.5, 6


class SparseWyly(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn(vocab, K0) * 0.5)
        self.D = K0 * CTX + CTX
        self.register_buffer("idx", torch.randint(0, 2 * self.D, (RMAX, KSLOT)))
        self.dec = torch.nn.Parameter(torch.zeros(self.D + RMAX, V))
        self.bias = torch.nn.Parameter(torch.zeros(V))
        self.register_buffer("active", torch.zeros(RMAX, dtype=torch.bool))
        self.active[:RINIT] = True

    def base(self, ids):
        c = self.C[ids]
        cur = ids[:, -1:]
        cc = [c[:, -1 - i] for i in range(CTX)]
        ones = torch.ones(ids.shape[0], 1, device=ids.device)
        eqs = [(ids[:, -1 - i:-i] == cur).any(1, keepdim=True).float() if i else ones for i in range(CTX)]
        return torch.cat(cc + eqs, -1)

    def literals(self, f):
        return torch.cat([torch.sigmoid(f), 1 - torch.sigmoid(f)], -1)          # (B, 2D)

    def fire(self, lit):
        return lit[:, self.idx].prod(-1) * self.active

    def forward(self, ids):
        f = self.base(ids)
        lit = self.literals(f)
        return torch.cat([f, self.fire(lit)], -1) @ self.dec + self.bias

    def sgd(self, lr):
        for p in [self.C, self.dec, self.bias]:
            if p.grad is not None:
                p.data -= lr * p.grad


def flogits(model, ids_all, idxs, bs=256):
    return torch.cat([model(ids_all[idxs[i:i + bs]]) for i in range(0, len(idxs), bs)])


def top1(model, ids_all, y, te):
    with torch.no_grad():
        return float((flogits(model, ids_all, te).argmax(1) == y[te]).float().mean())


def grow_random(model, ids, y, s, g):
    nr = torch.where(~model.active)[0][:RGROW]
    with torch.no_grad():
        model.idx[nr] = torch.randint(0, 2 * model.D, (len(nr), KSLOT), device=DEV)
    model.active[nr] = True


def grow_correlation(model, ids, y, s, g):
    """PROPOSER: for mispredicted targets, conjoin the literals most DISCRIMINATIVE for that target."""
    with torch.no_grad():
        lit = model.literals(model.base(ids[s]))                                # (N, 2D)
        pred = flogits(model, ids, s).argmax(1)
        wrong = pred != y[s]
        if wrong.sum() < 10:
            return grow_random(model, ids, y, s, g)
        # target tokens the model gets wrong, by frequency
        wt, wc = y[s][wrong].unique(return_counts=True)
        cand_t = wt[wc.argsort(descending=True)[:RGROW]]
        free = torch.where(~model.active)[0][:len(cand_t)]
        for r, t in zip(free.tolist(), cand_t.tolist(), strict=False):
            pos = y[s] == t
            if pos.sum() < 3:
                continue
            score = lit[pos].mean(0) - lit[~pos].mean(0)
            model.idx[r] = score.topk(KSLOT).indices                            # conjoin the top-KSLOT
            model.dec.data[model.D + r, t] += 2.0                               # wire the rule to predict t
            model.active[r] = True


def train_steps(model, ids, y, idxb, g, steps):
    for _ in range(steps):
        bi = idxb[torch.randint(len(idxb), (128,), generator=g)]
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(ids[bi]), y[bi]).backward()
        model.sgd(LR)


def run(vocab, ids, y, tr, te, grow, g):
    torch.manual_seed(0)
    model = SparseWyly(vocab).to(DEV)
    for ch in torch.chunk(tr, EPISODES):
        train_steps(model, ids, y, ch, g, WAKE_STEPS)                 # SGD on the current rule set
        if int((~model.active).sum()) >= RGROW:
            s = ch[torch.randint(len(ch), (2000,), generator=g)]
            grow(model, ids, y, s, g)                                 # PROPOSE new rules for the errors
        train_steps(model, ids, y, ch, g, WAKE_STEPS // 2)            # integrate the new rules
    return top1(model, ids, y, te), int(model.active.sum())


def main():
    d = torch.load(DATA)
    ids, target = d["kept_ids"], d["target"]
    keep = torch.isin(target, torch.bincount(target).argsort(descending=True)[:V])
    ids, target = ids[keep], target[keep]
    uv, ids = ids.reshape(-1).unique(return_inverse=True)
    ids = ids.reshape(target.shape[0], -1).to(DEV)
    y = target.unique(return_inverse=True)[1].to(DEV)
    vocab, n = len(uv), ids.shape[0]
    order = torch.arange(n, device=DEV)[ids[:, -1].argsort()]
    tr, te = order[:int(0.85 * n)], order[int(0.85 * n):]
    print(f"PROPOSER test -- sparse rules O(R*KSLOT={KSLOT}), vocab {vocab}, V {V}, {DEV}")
    print("references: dense-softmax (learnable, O(LIT)) 0.121 | bigram ~0.09\n")
    print(f"{'proposer':>16}{'top-1':>9}{'active rules':>14}")
    for name, grow in [("random", grow_random), ("correlation", grow_correlation)]:
        acc, nr = run(vocab, ids, y, tr, te, grow, torch.Generator().manual_seed(0))
        print(f"{name:>16}{acc:>9.3f}{nr:>14}", flush=True)
    print("\nread: correlation >> random(~0.068) approaching dense-softmax 0.121 = a PROPOSER picking")
    print("discriminative literals makes SPARSE O(R*KSLOT) rules useful -> scalable AND useful.")
    print("random = the discriminative proposer isn't finding the right conjunctions yet.")


if __name__ == "__main__":
    main()
