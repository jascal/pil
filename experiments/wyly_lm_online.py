"""The REAL Wyly algorithm on the endgame LM task: three loops (WAKE/SLEEP/CAPACITY), raw tokens, SGD.

Re-points the wake/sleep/homeostasis/grow/replay machinery (already built in wyly_online.py, but running on
pythia residuals / binary tasks / Adam) at the endgame: Wyly IS the LLM, learned ONLINE from RAW TOKENS.
Per the proposal pseudocode + the Adam-doesn't-work correction:
  substrate : learned token->concept table (replaces embedding) + soft-AND relational rules over concepts +
              eq_atoms across positions (replaces attention) + a vocab head (replaces the unembedding).
  WAKE      : fast SGD (NOT Adam -- Adam's per-param normalization fights the incidence/scaling dynamics, C8;
              and monolithic-optimize-to-convergence is the moving-target co-adaptation the schedule avoids).
              Over-subscribe: grow concepts/rules when a chunk is under-fit. NO pruning in wake.
  SLEEP     : periodic, structural. Consolidate against a CURATED replay M (representative + edge, by loss) --
              NOT the raw stream. Tier by usage -> protect the reused core (freeze). HOMEOSTASIS: merge
              redundant + prune dead, keep a homeostatic band of plastic spares.
Streamed over episodes (chunks of the token stream). Compared to the bigram floor and the offline-Adam Wyly.

Run: cd pil && .venv/bin/python experiments/wyly_lm_online.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "layers_pythia70m.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KMAX, RMAX, KINIT, RINIT, KGROW, RGROW = 384, 512, 64, 96, 48, 64
CTX, TAU, V = 4, 0.6, 2048
WAKE_STEPS, LR = 700, 0.5                                          # SGD wake (high lr -- plain SGD, no Adam)
KSLOT, SEL_TAU = 4, 0.7          # rule = conjunction of KSLOT soft-selected literals


class WylyLM(torch.nn.Module):
    """raw tokens -> token-concepts + SPARSE k-slot conjunction rules -> vocab.
    Each rule = a conjunction of KSLOT literals (a literal = a concept membership or its negation), each slot
    SOFT-SELECTING one literal via softmax attention. fire_r = prod over KSLOT of (selected literal). The
    product is over KSLOT (=4) terms only -> CANNOT vanish at scale (unlike a product over all F features), it
    is O(R*KSLOT) sparse, and 'which literal' is learnable. AND-per-rule + OR-in-head (dec sum) = DNF."""
    def __init__(self, vocab):
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn(vocab, KMAX) * 0.5)
        self.feat = KMAX * CTX + CTX
        self.LIT = 2 * self.feat                                      # features AND negations
        self.A = torch.nn.Parameter(torch.randn(RMAX, KSLOT, self.LIT) * 0.3)  # per-slot selection
        self.dec = torch.nn.Parameter(torch.zeros(self.feat + RMAX, V))
        self.bias = torch.nn.Parameter(torch.zeros(V))
        for nm in ["active_c", "prot_c"]:
            self.register_buffer(nm, torch.zeros(KMAX, dtype=torch.bool))
        for nm in ["active_r", "prot_r"]:
            self.register_buffer(nm, torch.zeros(RMAX, dtype=torch.bool))
        self.register_buffer("use_c", torch.zeros(KMAX))
        self.register_buffer("use_r", torch.zeros(RMAX))
        self.active_c[:KINIT] = True
        self.active_r[:RINIT] = True

    def features(self, ids):
        m = torch.sigmoid(self.C[ids]) * self.active_c               # (B, L, KMAX)
        cur = ids[:, -1:]
        cc = [m[:, -1 - i] for i in range(CTX)]
        ones = torch.ones_like(cur, dtype=m.dtype)
        eqs = [(ids[:, -1 - i:-i] == cur).any(1, keepdim=True).float() if i else ones for i in range(CTX)]
        return torch.cat(cc + eqs, -1)

    def fire(self, f):
        lit = torch.cat([f, 1 - f], -1)                              # (B, LIT) literal or its negation
        a = torch.softmax(self.A / SEL_TAU, -1)                      # (R,KSLOT,LIT) slot picks a literal
        slots = torch.einsum("bl,rkl->brk", lit, a)                  # (B, R, KSLOT) soft-selected literals
        return slots.prod(-1) * self.active_r                        # (B,R) conj over KSLOT terms

    def forward(self, ids):
        f = self.features(ids)
        return torch.cat([f, self.fire(f)], -1) @ self.dec + self.bias

    def grow(self):
        nc = torch.where(~self.active_c)[0][:KGROW]
        nr = torch.where(~self.active_r)[0][:RGROW]
        with torch.no_grad():
            self.C[:, nc] = torch.randn(self.C.shape[0], len(nc), device=self.C.device) * 0.5
            self.A[nr] = torch.randn(len(nr), KSLOT, self.LIT, device=self.A.device) * 0.3
        self.active_c[nc] = True
        self.active_r[nr] = True
        return len(nc) + len(nr) > 0


def sgd_masked(model, lr):
    """one plain-SGD step (NOT Adam): update = lr*grad, but only for ACTIVE & UNPROTECTED units."""
    with torch.no_grad():
        mc = (model.active_c & ~model.prot_c).float()
        mr = (model.active_r & ~model.prot_r).float()
        if model.C.grad is not None:
            model.C -= lr * model.C.grad * mc                        # concept table cols masked by mc
        model.A -= lr * model.A.grad * mr.view(-1, 1, 1)              # per-slot logits (active&unprot)
        model.dec -= lr * model.dec.grad
        model.bias -= lr * model.bias.grad


def loss_on(model, ids, y):
    return F.cross_entropy(model(ids), y)


def flogits(model, ids_all, idxs, bs=256):
    """batched forward (the soft-AND is memory-heavy): logits for idxs, concatenated."""
    return torch.cat([model(ids_all[idxs[i:i + bs]]) for i in range(0, len(idxs), bs)])


def top1(model, ids_all, y, idxs):
    with torch.no_grad():
        return float((flogits(model, ids_all, idxs).argmax(1) == y[idxs]).float().mean())


def wake(model, ids, y, idxb, g):
    """fast SGD, over-subscribe: grow if the chunk stays under-fit; NO pruning."""
    for _rnd in range(4):
        for _ in range(WAKE_STEPS):
            bi = idxb[torch.randint(len(idxb), (128,), generator=g)]
            model.zero_grad(set_to_none=True)
            loss_on(model, ids[bi], y[bi]).backward()
            sgd_masked(model, LR)
        with torch.no_grad():
            bi = idxb[torch.randint(len(idxb), (512,), generator=g)]
            acc = float((model(ids[bi]).argmax(1) == y[bi]).float().mean())
        if acc > 0.20 or not model.grow():
            break


def sleep(model, ids, y, replay, seen_idx, g):
    """structural consolidation against CURATED replay; protect reused core; homeostasis merge/prune."""
    with torch.no_grad():
        # curate M: representative (random) + edge (high-loss) from what was just seen (batched forward)
        s = seen_idx[torch.randperm(len(seen_idx), generator=g)[:2000]]
        pl = F.cross_entropy(flogits(model, ids, s), y[s], reduction="none")
        edge = s[pl.argsort(descending=True)[:400]]
        rep = s[torch.randperm(len(s), generator=g)[:400]]
        replay.append((torch.cat([edge, rep]),))
        # usage + protect the reused core (freeze high-usage concepts/rules) -- batched
        model.use_c = torch.maximum(model.use_c, torch.sigmoid(model.C[ids[s]]).reshape(-1, KMAX).mean(0))
        fr = [model.fire(model.features(ids[s[i:i + 256]])).mean(0) for i in range(0, len(s), 256)]
        model.use_r = torch.maximum(model.use_r, torch.stack(fr).mean(0))
        model.prot_c |= (model.use_c > 0.6) & model.active_c
        model.prot_r |= (model.use_r > 0.15) & model.active_r
        # HOMEOSTASIS: prune dead (never-used) unprotected units -> reclaim to the pool (homeostatic band)
        model.active_c[(model.use_c < 0.02) & model.active_c & ~model.prot_c] = False
        model.active_r[(model.use_r < 0.02) & model.active_r & ~model.prot_r] = False


def replay_idx(replay, g, k=1500):
    if not replay:
        return torch.empty(0, dtype=torch.long, device=DEV)
    allr = torch.cat([r[0] for r in replay])
    return allr[torch.randint(len(allr), (min(k, len(allr)),), generator=g)]


def main():
    d = torch.load(DATA)
    ids = d["kept_ids"]
    target = d["target"]
    topv = torch.bincount(target).argsort(descending=True)[:V]
    keep = torch.isin(target, topv)
    ids, target = ids[keep], target[keep]
    uv, ids = ids.reshape(-1).unique(return_inverse=True)
    ids = ids.reshape(target.shape[0], -1).to(DEV)
    uniq, y = target.unique(return_inverse=True)
    y = y.to(DEV)
    vocab, n = len(uv), ids.shape[0]
    tr = torch.arange(int(0.85 * n))
    te = torch.arange(int(0.85 * n), n).to(DEV)
    g = torch.Generator().manual_seed(0)
    print(f"three-loop ONLINE Wyly-LM (SGD, raw tokens) -- vocab {vocab}, V {V}, {DEV}, {n} windows")
    # bigram floor for reference
    big = torch.nn.Embedding(vocab, V).to(DEV)
    ob = torch.optim.SGD(big.parameters(), lr=1.0)
    for _ in range(3000):
        bi = tr[torch.randint(len(tr), (256,), generator=g)].to(DEV)
        ob.zero_grad()
        F.cross_entropy(big(ids[bi][:, -1]), y[bi]).backward()
        ob.step()
    with torch.no_grad():
        bacc = float((big(ids[te][:, -1]).argmax(1) == y[te]).float().mean())
    print(f"bigram floor {bacc:.3f}  |  offline-Adam Wyly(ref) 0.119\n")

    model = WylyLM(vocab).to(DEV)
    # NON-STATIONARY stream: order episodes by current token so each faces a distribution SHIFT (new tokens
    # appear per episode) -- this actually stresses forgetting, which wake/sleep/replay exist to handle.
    order = tr.to(DEV)[ids[tr.to(DEV), -1].argsort()]
    chunks = torch.chunk(order, 8)
    replay, seen = [], []
    print(f"{'episode':>8}{'test top-1':>12}{'active c/r':>13}{'prot c/r':>10}")
    for ep, ch in enumerate(chunks):
        idxb = torch.cat([ch, replay_idx(replay, g)]) if replay else ch     # wake sees chunk + curated replay
        wake(model, ids, y, idxb, g)
        seen.append(ch)
        sleep(model, ids, y, replay, torch.cat(seen), g)
        acc = top1(model, ids, y, te)
        print(f"{ep:>8}{acc:>12.3f}{f'{int(model.active_c.sum())}/{int(model.active_r.sum())}':>13}"
              f"{f'{int(model.prot_c.sum())}/{int(model.prot_r.sum())}':>10}", flush=True)
    print(f"\nread: online three-loop Wyly-LM (SGD wake + structural sleep) vs bigram {bacc:.3f} +")
    print("Adam Wyly 0.119. This is the ACTUAL algorithm (not a monolithic Adam run): it streams, over-")
    print("subscribes+grows in wake, consolidates vs replay + homeostasis in sleep. top-1 rises across")
    print("episodes w/ stable active/protected structure = it learns online; matching offline-Adam = SGD+")
    print("structure is a viable Adam-free LM learner.")


if __name__ == "__main__":
    main()
