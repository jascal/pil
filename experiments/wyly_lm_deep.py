"""Deep Wyly stack: fixed layers, each owning its rules, each locally managed by its OWN wake/sleep.

Design (user-specified): a STACK of layers. Each layer = k-slot sparse-conjunction rules over its input
concepts; layer L's rule-firings ARE read as layer L+1's input concepts (homoiconic BY WIRING -- nothing is
promoted/migrated up or down the stack; every rule stays in its layer). Each layer separately runs its own
wake (over-subscribe/grow when saturated) + sleep (consolidate reused, prune dead) on ITS rules. End-to-end
SGD updates params; the STRUCTURAL grow/prune/protect is per-layer and local. Head decodes from the raw
features + the top layer's firings. This tests whether DEPTH (stacked self-managed conjunction layers) beats
the single-layer form -- the across-layer scaling fix, composed with the k-slot within-layer rule form.

Run: cd pil && .venv/bin/python experiments/wyly_lm_deep.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

DATA = Path(os.environ.get(  # canonical dataset -- see experiments/extract_nexttoken.py
    "WYLY_DATA", Path(__file__).resolve().parent.parent / "data" / "wyly_nexttoken_pythia70m.pt"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K0, CTX, V = 192, 4, 2048                         # token concepts, positions read, output vocab
RMAX, RINIT, RGROW = 384, 128, 64                 # per-layer rule pool
KSLOT, SEL_TAU = 4, 0.7
WAKE_STEPS, LR, EPISODES = 500, 0.5, 6


class Layer(torch.nn.Module):
    """one self-managed conjunction layer: concepts (B,Cin) -> k-slot rule firings (B,R); rules stay here."""
    def __init__(self, cin):
        super().__init__()
        self.cin, self.lit = cin, 2 * cin
        self.A = torch.nn.Parameter(torch.randn(RMAX, KSLOT, self.lit) * 0.3)
        self.register_buffer("active", torch.zeros(RMAX, dtype=torch.bool))
        self.register_buffer("prot", torch.zeros(RMAX, dtype=torch.bool))
        self.register_buffer("use", torch.zeros(RMAX))
        self.active[:RINIT] = True

    def forward(self, h):
        lit = torch.cat([h, 1 - h], -1)
        a = torch.softmax(self.A / SEL_TAU, -1)
        slots = torch.einsum("bl,rkl->brk", lit, a)
        return slots.prod(-1) * self.active                     # (B, R) firings = next layer's concepts

    def grow(self):
        nr = torch.where(~self.active)[0][:RGROW]
        with torch.no_grad():
            self.A[nr] = torch.randn(len(nr), KSLOT, self.lit, device=self.A.device) * 0.3
        self.active[nr] = True
        return len(nr) > 0

    def sgd(self, lr):
        m = (self.active & ~self.prot).float().view(-1, 1, 1)
        self.A.data -= lr * self.A.grad * m

    def sleep_update(self, h_in):
        with torch.no_grad():
            fire = self.forward(h_in).mean(0)
            self.use = torch.maximum(self.use, fire)
            self.prot |= (self.use > 0.15) & self.active         # protect the reused core (in place)
            self.active[(self.use < 0.02) & self.active & ~self.prot] = False   # prune dead (homeostasis)


class Stack(torch.nn.Module):
    def __init__(self, vocab, depth):
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn(vocab, K0) * 0.5)
        self.feat = K0 * CTX + CTX
        self.layers = torch.nn.ModuleList([Layer(self.feat if i == 0 else RMAX) for i in range(depth)])
        top = self.feat + (RMAX if depth else 0)
        self.dec = torch.nn.Parameter(torch.zeros(top, V))
        self.bias = torch.nn.Parameter(torch.zeros(V))

    def base(self, ids):
        m = torch.sigmoid(self.C[ids])
        cur = ids[:, -1:]
        cc = [m[:, -1 - i] for i in range(CTX)]
        ones = torch.ones_like(cur, dtype=m.dtype)
        eqs = [(ids[:, -1 - i:-i] == cur).any(1, keepdim=True).float() if i else ones for i in range(CTX)]
        return torch.cat(cc + eqs, -1)

    def forward(self, ids):
        f = self.base(ids)
        h = f
        for lyr in self.layers:
            h = lyr(h)                                           # each layer's firings feed the next
        top = torch.cat([f, h], -1) if self.layers else f
        return top @ self.dec + self.bias

    def stack_inputs(self, ids):                                # per-layer input concepts (for sleep usage)
        f = self.base(ids)
        outs, h = [f], f
        for lyr in self.layers:
            h = lyr(h)
            outs.append(h)
        return outs                                             # [base, h1, h2, ...]

    def sgd(self, lr):
        for p in [self.C, self.dec, self.bias]:
            if p.grad is not None:
                p.data -= lr * p.grad
        for lyr in self.layers:
            lyr.sgd(lr)


def flogits(model, ids_all, idxs, bs=256):
    return torch.cat([model(ids_all[idxs[i:i + bs]]) for i in range(0, len(idxs), bs)])


def top1(model, ids_all, y, idxs):
    with torch.no_grad():
        return float((flogits(model, ids_all, idxs).argmax(1) == y[idxs]).float().mean())


def wake(model, ids, y, idxb, g):
    for _ in range(4):
        for _ in range(WAKE_STEPS):
            bi = idxb[torch.randint(len(idxb), (128,), generator=g)]
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(ids[bi]), y[bi]).backward()
            model.sgd(LR)
        with torch.no_grad():
            bi = idxb[torch.randint(len(idxb), (512,), generator=g)]
            acc = float((model(ids[bi]).argmax(1) == y[bi]).float().mean())
        # each layer over-subscribes independently: grow a layer only if its active rules are ~all used
        grew = False
        for lyr in model.layers:
            if float(((lyr.use > 0.1) & lyr.active).float().sum()) > 0.8 * float(lyr.active.sum()):
                grew = lyr.grow() or grew
        if acc > 0.20 or not grew:
            break


def sleep(model, ids, y, seen, g):
    with torch.no_grad():
        s = seen[torch.randperm(len(seen), generator=g)[:1500]]
        acc = [[] for _ in range(len(model.layers) + 1)]         # per-layer inputs collected in batches
        for i in range(0, len(s), 256):
            for li, h in enumerate(model.stack_inputs(ids[s[i:i + 256]])):
                acc[li].append(h)
        outs = [torch.cat(a) for a in acc]
        for li, lyr in enumerate(model.layers):
            lyr.sleep_update(outs[li])                           # each layer consolidates on ITS own input


def main():
    d = torch.load(DATA)
    ids, target = d["kept_ids"], d["target"]
    keep = torch.isin(target, torch.bincount(target).argsort(descending=True)[:V])
    ids, target = ids[keep], target[keep]
    uv, ids = ids.reshape(-1).unique(return_inverse=True)
    ids = ids.reshape(target.shape[0], -1).to(DEV)
    y = target.unique(return_inverse=True)[1].to(DEV)
    vocab, n = len(uv), ids.shape[0]
    g = torch.Generator().manual_seed(0)
    order = torch.arange(n, device=DEV)[ids[:, -1].argsort()]    # non-stationary: order by current token
    tr, te = order[:int(0.85 * n)], order[int(0.85 * n):]
    print(f"DEEP Wyly stack (per-layer wake/sleep, no migration) -- vocab {vocab}, V {V}, {DEV}, {n} windows")
    print("single-layer ref 0.121 | bigram ~0.09\n")
    print(f"{'depth':>6}{'final top-1':>13}{'active rules/layer':>22}")
    for depth in [1, 2, 3]:
        torch.manual_seed(0)
        model = Stack(vocab, depth).to(DEV)
        seen = []
        chunks = torch.chunk(tr, EPISODES)
        for ch in chunks:
            wake(model, ids, y, ch, g)
            seen.append(ch)
            sleep(model, ids, y, torch.cat(seen), g)
        acc = top1(model, ids, y, te)
        rules = "/".join(str(int(lyr.active.sum())) for lyr in model.layers)
        print(f"{depth:>6}{acc:>13.3f}{rules:>22}", flush=True)
    print("\nread: top-1 rising with DEPTH = stacked per-layer conjunction layers compose (across-layer")
    print("scaling fix); flat = depth not helping (deeper layers not learning, or 1 layer already")
    print("saturates this tiny 12-token-context task).")


if __name__ == "__main__":
    main()
