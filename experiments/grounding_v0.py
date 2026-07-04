"""Grounding bridge v0: concepts as HYPERPLANES in a real tiny model's residual space.

The load-bearing open problem (Coppice §5.5, confirmed by Grok): the concept substrate has been over token
IDs (a lookup), not grounded in a model's geometry. Grounding = concept is a hyperplane (direction u_c +
bias b_c) in RESIDUAL space; membership is geometric incidence m_c(r) = sigmoid(<r,u_c> - b_c). Then the
concept lattice = the face lattice of a hyperplane arrangement = tropic's cells; rules = polyhedral regions.

Minimal honest test on a model we fully control (no fieldrun plumbing yet): train a tiny transformer on a
(color,shape) task family over TRAIN combos; extract its residuals; then learn GEOMETRIC concepts
(hyperplanes on the residual) + rules. Two questions:
  (1) RECOVERY -- do the learned concept-directions recover the planted color/shape structure from the
      residual (i.e., does the model represent attributes as linear directions grounding can find)?
  (2) DOES COMPOSITIONAL GENERALIZATION SURVIVE GROUNDING -- on HELD-OUT (color,shape) combos, does the
      geometric-concept model generalize (like the token-ID compositional result did)?
Baseline: a direct linear probe on the residual (upper bound for what's linearly present).

Run: cd pil && .venv/bin/python experiments/grounding_v0.py
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import nn

NC, NS, L, D = 4, 4, 4, 48                      # colors, shapes, seq len, model dim
V = NC * NS
HELD = {(c, c) for c in range(min(NC, NS))}     # held-out combos (diagonal)
HELD_IDS = {c * NS + s for c, s in HELD}
TRAIN_IDS = [t for t in range(V) if t not in HELD_IDS]


def color(t):
    return t // NS


def shape(t):
    return t % NS


def task_labels(seq):
    c0, s0 = color(seq[0]), shape(seq[0])
    d = {f"col{c}": int(c0 == c) for c in range(NC)}
    d.update({f"shp{s}": int(s0 == s) for s in range(NS)})
    d.update({f"has_col{c}": int(any(color(t) == c for t in seq)) for c in range(NC)})
    return d


TASKS = list(task_labels([0] * L))


def gen(n, pool, seed, held_first=False):
    rng = random.Random(seed)
    X = []
    for _ in range(n):
        s = [rng.choice(pool) for _ in range(L)]
        if held_first:
            s[0] = rng.choice(list(HELD_IDS))
        X.append(s)
    Y = {t: torch.tensor([task_labels(s)[t] for s in X]).float() for t in TASKS}
    return torch.tensor(X), Y


# ---- tiny transformer: produces a residual at the last position that must encode the structure ----
class TinyTransformer(nn.Module):
    def __init__(self, d=D, nlayer=2, nhead=4):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(L, d)
        self.blocks = nn.ModuleList(_Block(d, nhead) for _ in range(nlayer))
        self.ln = nn.LayerNorm(d)
        self.heads = nn.Linear(d, len(TASKS))          # trained task readout (defines the residual geometry)

    def residual(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        t = x.shape[1]
        bias = torch.zeros(t, t, device=x.device).masked_fill(
            ~torch.ones(t, t, device=x.device).tril().bool(), float("-inf"))
        for blk in self.blocks:
            h = blk(h, bias)
        return self.ln(h)[:, -1]                        # (B, d) last-position residual

    def forward(self, x):
        return self.heads(self.residual(x))


class _Block(nn.Module):
    def __init__(self, d, nhead):
        super().__init__()
        self.nh, self.hd = nhead, d // nhead
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bias):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q, k, v = (z.view(b, t, self.nh, self.hd).transpose(1, 2) for z in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(o.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


def train_model(m, X, Y, steps=3000, lr=2e-3):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(X.shape[0], (256,), generator=g)
        lg = m(X[idx])
        loss = sum(F.binary_cross_entropy_with_logits(lg[:, ti], Y[t][idx]) for ti, t in enumerate(TASKS))
        opt.zero_grad()
        loss.backward()
        opt.step()


# ---- geometric concept learner: concepts = hyperplanes in residual space ----
class GeoConcepts(nn.Module):
    def __init__(self, d=D, K=12, Rn=24):
        super().__init__()
        self.U = nn.Parameter(torch.randn(K, d) * 0.3)         # concept directions
        self.b = nn.Parameter(torch.zeros(K))                  # concept biases (hyperplane offset)
        rl = torch.randn(Rn, K + 1) * 0.3
        rl[:, -1] += 1.5
        self.req = nn.Parameter(rl)                            # per rule: require a soft subset of concepts
        self.heads = nn.Parameter(torch.zeros(len(TASKS), Rn + 1))
        self.K = K

    def membership(self, r, tau=1.0):
        return torch.sigmoid((r @ self.U.T - self.b) / tau)    # (B,K) geometric incidence

    def fire(self, r, tau=1.0):
        m = self.membership(r, tau)                            # (B,K)
        req = torch.sigmoid(self.req)                          # (Rn,K+1)
        term = 1 - req[:, :-1].unsqueeze(0) + req[:, :-1].unsqueeze(0) * m.unsqueeze(1)
        return term.prod(-1) * 1.0 + req[:, -1].unsqueeze(0) * 0.0   # (B,Rn) soft-AND (wildcard ~ always-on)

    def logit(self, ti, r, tau=1.0):
        g = self.fire(r, tau)
        return g @ self.heads[ti, :-1] + self.heads[ti, -1]


def train_concepts(gc, Rtr, Ytr, tr_idx, steps=3000, lr=0.03):
    opt = torch.optim.Adam(gc.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    n = len(tr_idx)
    for step in range(steps):
        tau = max(0.4, 1.3 - 0.9 * step / steps)
        idx = tr_idx[torch.randint(n, (256,), generator=g)]
        loss = sum(F.binary_cross_entropy_with_logits(gc.logit(ti, Rtr[idx], tau), Ytr[t][idx])
                   for ti, t in enumerate(TASKS)) / len(TASKS)
        opt.zero_grad()
        loss.backward()
        opt.step()


def concept_recovery(gc, R, X):
    """Best match of a concept's membership pattern to a planted attribute (first-token color / shape)."""
    with torch.no_grad():
        m = (gc.membership(R, 0.4) > 0.5).float()              # (N,K)
    firstcol = torch.tensor([color(int(x[0])) for x in X])
    firstshp = torch.tensor([shape(int(x[0])) for x in X])
    best_col = best_shp = 0.0
    for k in range(gc.K):
        mk = m[:, k]
        for c in range(NC):
            tgt = (firstcol == c).float()
            best_col = max(best_col, float((mk == tgt).float().mean()), float((mk != tgt).float().mean()))
        for s in range(NS):
            tgt = (firstshp == s).float()
            best_shp = max(best_shp, float((mk == tgt).float().mean()), float((mk != tgt).float().mean()))
    return best_col, best_shp


def acc(gc, R, Y, idx):
    with torch.no_grad():
        return sum(float(((gc.logit(ti, R[idx], 0.4) > 0).float() == Y[t][idx]).float().mean())
                   for ti, t in enumerate(TASKS)) / len(TASKS)


def probe_heldout(Rtr, Ytr, tr_idx, Rho, Yho):
    """Linear-probe upper bound: what fraction of tasks are LINEARLY decodable on held-out combos."""
    accs = []
    for t in TASKS:
        w = torch.zeros(D, requires_grad=True)
        bb = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, bb], lr=0.05)
        for _ in range(300):
            lg = Rtr[tr_idx] @ w + bb
            loss = F.binary_cross_entropy_with_logits(lg, Ytr[t][tr_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            accs.append(float(((Rho @ w + bb > 0).float() == Yho[t]).float().mean()))
    return sum(accs) / len(accs)


def main():
    print(f"grounding v0: concepts = hyperplanes in a tiny transformer's residual space (d={D})")
    print(f"{V} (color,shape) tokens; held-out combos {sorted(HELD)}; 3 seeds\n")
    print(f"{'seed':>5}{'probe_held':>12}{'recov_col':>11}{'recov_shp':>11}{'geo_train':>11}{'geo_held':>10}")
    agg = {"probe": [], "col": [], "shp": [], "tr": [], "ho": []}
    for sd in (0, 1, 2):
        torch.manual_seed(sd)
        Xtr, Ytr = gen(4000, TRAIN_IDS, sd + 10)
        Xho, Yho = gen(2000, TRAIN_IDS, sd + 90, held_first=True)
        m = TinyTransformer()
        train_model(m, Xtr, Ytr)
        with torch.no_grad():
            Rtr, Rho = m.residual(Xtr), m.residual(Xho)
        tr_idx = torch.arange(Xtr.shape[0])
        probe = probe_heldout(Rtr, Ytr, tr_idx, Rho, Yho)
        gc = GeoConcepts()
        train_concepts(gc, Rtr, Ytr, tr_idx)
        col, shp = concept_recovery(gc, Rtr, Xtr)
        tr = acc(gc, Rtr, Ytr, tr_idx)
        ho = acc(gc, Rho, Yho, torch.arange(Xho.shape[0]))
        for k, v in zip(agg, (probe, col, shp, tr, ho), strict=True):
            agg[k].append(v)
        print(f"{sd:>5}{probe:>12.3f}{col:>11.3f}{shp:>11.3f}{tr:>11.3f}{ho:>10.3f}", flush=True)
    mean = {k: sum(v) / len(v) for k, v in agg.items()}
    print(f"{'mean':>5}{mean['probe']:>12.3f}{mean['col']:>11.3f}{mean['shp']:>11.3f}"
          f"{mean['tr']:>11.3f}{mean['ho']:>10.3f}")
    print("\nread: recovery (col/shp) high => concept-hyperplanes align with planted attributes = grounding "
          "finds structure in real residuals. geo_held ~ probe_held => grounding extracts the linearly-"
          "available structure and doesn't collapse below a linear probe (composition survives grounding to "
          "the extent the MODEL is linearly compositional). The ceiling is set by the model's representation "
          "(probe_held), not by the grounding mechanism -- forward pointer: do threx/Pythia carry MORE "
          "linearly-compositional structure (linear representation hypothesis at scale)?")


if __name__ == "__main__":
    main()
