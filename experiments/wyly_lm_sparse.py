"""Wyly-LM with GENUINELY SPARSE rules + a RESIDUAL concept-stream (the two scaling fixes, composed).

(1) SPARSE COMPUTE: each rule holds KSLOT integer literal-indices and GATHERS those literals then products
    them -> fire = prod_k lit[idx[r,k]]. Cost O(R*KSLOT), INDEPENDENT of the literal count (the dense
    softmax-over-all-literals form was O(R*KSLOT*LIT)). 'Which literals' is learned STRUCTURALLY: rules are
    random sparse conjunctions (over-generate); the head weights + sleep prune/grow SELECT the useful ones
    (the design's proposer = over-generation + selection), not a dense-softmax gradient.
(2) RESIDUAL STREAM (was MISSING -- likely why the plain stack's depth collapsed): each layer READS a shared
    real-valued concept-stream s, computes rules over its memberships sigmoid(s), and WRITES the firings back
    ADDITIVELY: s = s + fire @ Wproj (the transformer residual pattern -- preserves signal + gradient).
Per-layer wake/sleep, no migration. SGD (not Adam). Depth sweep vs the dense single-layer 0.121.

Run: cd pil && .venv/bin/python experiments/wyly_lm_sparse.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "layers_pythia70m.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K0, CTX, V = 192, 4, 2048
RMAX, RINIT, RGROW, KSLOT = 512, 128, 96, 4
WAKE_STEPS, LR, EPISODES = 500, 0.5, 6


class SparseLayer(torch.nn.Module):
    """residual: rules GATHER KSLOT literals of sigmoid(stream), write firings back. O(R*KSLOT)."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.register_buffer("idx", torch.randint(0, 2 * d, (RMAX, KSLOT)))
        self.Wproj = torch.nn.Parameter(torch.zeros(RMAX, d))
        self.register_buffer("active", torch.zeros(RMAX, dtype=torch.bool))
        self.register_buffer("prot", torch.zeros(RMAX, dtype=torch.bool))
        self.register_buffer("use", torch.zeros(RMAX))
        self.active[:RINIT] = True

    def fire(self, s):
        m = torch.sigmoid(s)
        lit = torch.cat([m, 1 - m], -1)                                         # (B, 2D) literals + negations
        gathered = lit[:, self.idx]                                            # (B, R, KSLOT) SPARSE gather
        return gathered.prod(-1) * self.active                                 # (B, R) O(R*KSLOT) conjunction

    def forward(self, s):
        return s + self.fire(s) @ self.Wproj                                   # RESIDUAL write

    def grow(self):
        nr = torch.where(~self.active)[0][:RGROW]
        with torch.no_grad():
            self.idx[nr] = torch.randint(0, 2 * self.d, (len(nr), KSLOT), device=self.idx.device)
            self.Wproj[nr] = 0.0
        self.active[nr] = True
        return len(nr) > 0

    def sgd(self, lr):
        m = (self.active & ~self.prot).float().unsqueeze(1)
        self.Wproj.data -= lr * self.Wproj.grad * m

    def sleep_update(self, s):
        with torch.no_grad():
            self.use = torch.maximum(self.use, self.fire(s).mean(0))
            w = self.Wproj.norm(dim=1)
            self.prot |= (w > 0.05) & self.active
            dead = (self.use < 0.02) & self.active & ~self.prot
            self.active[dead | ((w < 0.005) & self.active & ~self.prot)] = False


class Stack(torch.nn.Module):
    def __init__(self, vocab, depth):
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn(vocab, K0) * 0.5)
        self.D = K0 * CTX + CTX
        self.layers = torch.nn.ModuleList([SparseLayer(self.D) for _ in range(depth)])
        self.dec = torch.nn.Parameter(torch.zeros(self.D, V))
        self.bias = torch.nn.Parameter(torch.zeros(V))

    def base(self, ids):
        c = self.C[ids]                                                        # (B, L, K0) token codes
        cur = ids[:, -1:]
        cc = [c[:, -1 - i] for i in range(CTX)]
        ones = torch.ones(ids.shape[0], 1, device=ids.device)
        eqs = [(ids[:, -1 - i:-i] == cur).any(1, keepdim=True).float() if i else ones for i in range(CTX)]
        return torch.cat(cc + eqs, -1)

    def forward(self, ids):
        s = self.base(ids)
        for lyr in self.layers:
            s = lyr(s)                                                         # residual read/compute/write
        return s @ self.dec + self.bias

    def stream_at(self, ids):
        s, outs = self.base(ids), []
        for lyr in self.layers:
            outs.append(s)
            s = lyr(s)
        outs.append(s)
        return outs

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
        grew = False
        for lyr in model.layers:
            frac = float(((lyr.use > 0.1) & lyr.active).float().sum()) / max(float(lyr.active.sum()), 1)
            if frac > 0.7:
                grew = lyr.grow() or grew
        if acc > 0.22 or not grew:
            break


def sleep(model, ids, y, seen, g):
    with torch.no_grad():
        s = seen[torch.randperm(len(seen), generator=g)[:1500]]
        acc = [[] for _ in range(len(model.layers) + 1)]
        for i in range(0, len(s), 256):
            for li, h in enumerate(model.stream_at(ids[s[i:i + 256]])):
                acc[li].append(h)
        outs = [torch.cat(a) for a in acc]
        for li, lyr in enumerate(model.layers):
            lyr.sleep_update(outs[li])


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
    order = torch.arange(n, device=DEV)[ids[:, -1].argsort()]
    tr, te = order[:int(0.85 * n)], order[int(0.85 * n):]
    print(f"SPARSE+RESIDUAL Wyly-LM -- vocab {vocab}, V {V}, {DEV}, {n}; dense single-layer ref 0.121")
    print(f"rules: O(R*KSLOT={KSLOT}) sparse GATHER (not dense over 2*D literals); residual stream\n")
    print(f"{'depth':>6}{'final top-1':>13}{'active rules/layer':>22}")
    for depth in [0, 1, 2, 3]:
        torch.manual_seed(0)
        model = Stack(vocab, depth).to(DEV)
        seen = []
        for ch in torch.chunk(tr, EPISODES):
            wake(model, ids, y, ch, g)
            seen.append(ch)
            if depth:
                sleep(model, ids, y, torch.cat(seen), g)
        acc = top1(model, ids, y, te)
        rules = "/".join(str(int(lyr.active.sum())) for lyr in model.layers) or "(linear)"
        print(f"{depth:>6}{acc:>13.3f}{rules:>22}", flush=True)
    print("\nread: depth-0 = linear-over-base. top-1 rising with depth = the residual lets stacked sparse-")
    print("conjunction layers compose without collapsing (residual + sparse = the scaling fixes);")
    print("O(R*KSLOT) regardless of literal count -> reachable at long context / large vocab.")


if __name__ == "__main__":
    main()
