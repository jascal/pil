"""Warm-start concept-hyperplanes from the model's OWN geometry vs random init.

grounding_v0 showed concepts-as-hyperplanes recover structure from real residuals (random init worked
surprisingly well). Grok + we both flagged the natural next test: seed the concept pool from the model's
own directions (its readout/writer geometry) rather than random. Hypothesis: model-native priors should
(a) raise recovery, (b) accelerate discovery (higher accuracy earlier), (c) tighten the identifiability
gauge (concepts stay aligned to the model's directions instead of finding rotated equivalents).

Init strategies for the K concept directions u_c:
  random   -- randn (grounding_v0 baseline)
  svd      -- top-K right singular vectors of the (centered) residual matrix (high-variance directions)
  readout  -- the model's task-head rows (the directions it actually decodes along) -- the LLM analogue is
              the unembedding/writer subspace
  hybrid   -- half readout + half random (model prior + diversity for over-generation)

Metrics (3 seeds): recovery (color/shape), held-out compositional @ early step (convergence) and final,
and gauge alignment (mean max-cosine of final u_c to the model's readout directions).

Run: cd pil && .venv/bin/python experiments/warm_start.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import grounding_v0 as G  # noqa: E402

D, K = G.D, 12
SCALE = (D ** 0.5) * 0.3          # match the norm of the randn*0.3 baseline init


def model_directions(Rtr, W):
    """SVD directions of the residual + the model's readout (task-head) directions, unit-normalized."""
    Rc = Rtr - Rtr.mean(0)
    _, _, Vh = torch.linalg.svd(Rc, full_matrices=False)
    svd = Vh[:K]                                              # (K, d)
    readout = W / (W.norm(dim=1, keepdim=True) + 1e-8)        # (n_tasks, d)
    return svd, readout


def make_gc(init, svd, readout):
    gc = G.GeoConcepts(d=D, K=K)
    with torch.no_grad():
        if init == "svd":
            gc.U.copy_(svd[:K] * SCALE)
        elif init == "readout":
            r = readout[torch.arange(K) % readout.shape[0]]
            gc.U.copy_(r * SCALE)
        elif init == "hybrid":
            gc.U[: K // 2].copy_(readout[: K // 2] * SCALE)   # rest stays random
    return gc


def train_checkpointed(gc, Rtr, Ytr, Rho, Yho, early=500, total=2500, lr=0.03):
    opt = torch.optim.Adam(gc.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    n = Rtr.shape[0]
    tr_idx = torch.arange(n)
    ho_idx = torch.arange(Rho.shape[0])
    early_acc = None
    for step in range(total):
        tau = max(0.4, 1.3 - 0.9 * step / total)
        idx = torch.randint(n, (256,), generator=g)
        loss = sum(F.binary_cross_entropy_with_logits(gc.logit(ti, Rtr[idx], tau), Ytr[t][idx])
                   for ti, t in enumerate(G.TASKS)) / len(G.TASKS)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == early:
            early_acc = G.acc(gc, Rho, Yho, ho_idx)
    return early_acc, G.acc(gc, Rtr, Ytr, tr_idx), G.acc(gc, Rho, Yho, ho_idx)


def gauge_alignment(gc, readout):
    """Mean over concepts of the max cosine to any model-readout direction (higher = more pinned)."""
    with torch.no_grad():
        u = gc.U / (gc.U.norm(dim=1, keepdim=True) + 1e-8)
        cos = (u @ readout.T).abs()                           # (K, n_tasks)
        return float(cos.max(1).values.mean())


def run(inits, n_ct, steps):
    """Compare inits at a given CONCEPT-training-data budget (does a good init help when starved?)."""
    agg = {i: {"recov": [], "early": [], "held": [], "align": []} for i in inits}
    for sd in (0, 1, 2):
        torch.manual_seed(sd)
        Xtr, Ytr = G.gen(n_ct, G.TRAIN_IDS, sd + 10)
        Xho, Yho = G.gen(2000, G.TRAIN_IDS, sd + 90, held_first=True)
        m = G.TinyTransformer()
        G.train_model(m, *G.gen(4000, G.TRAIN_IDS, sd + 10))    # model well-trained; only concepts starved
        with torch.no_grad():
            Rtr, Rho = m.residual(Xtr), m.residual(Xho)
        svd, readout = model_directions(Rtr, m.heads.weight.detach())
        ecp = min(steps // 3, 500)
        for init in inits:
            gc = make_gc(init, svd, readout)
            early, _tr, held = train_checkpointed(gc, Rtr, Ytr, Rho, Yho, early=ecp, total=steps)
            col, shp = G.concept_recovery(gc, Rtr, Xtr)
            agg[init]["recov"].append((col + shp) / 2)
            agg[init]["early"].append(early)
            agg[init]["held"].append(held)
            agg[init]["align"].append(gauge_alignment(gc, readout))
    return agg


def main():
    inits = ["random", "svd", "readout", "hybrid"]
    print(f"warm-start concepts from model geometry vs random (K={K}, d={D}), 3 seeds")
    print("testing the 'toy too easy' explanation: sweep the CONCEPT-training-data budget\n")
    for n_ct, steps in ((4000, 2500), (400, 1500), (120, 1200)):
        print(f"== concept-training examples = {n_ct} ==")
        agg = run(inits, n_ct, steps)
        print(f"{'init':>9}{'recovery':>10}{'held_early':>12}{'held_final':>12}{'gauge_align':>13}")
        for init in inits:
            a = agg[init]
            n = len(a["recov"])
            print(f"{init:>9}{sum(a['recov']) / n:>10.3f}{sum(a['early']) / n:>12.3f}"
                  f"{sum(a['held']) / n:>12.3f}{sum(a['align']) / n:>13.3f}", flush=True)
        print()
    print("read: if warm-start (readout/svd) beats random MORE as concept-data shrinks, the abundant-data "
          "'toy too easy' explanation holds -- model-native priors buy SAMPLE efficiency + gauge alignment, "
          "mattering when data is scarce. If it stays ~equal even when starved, warm-start gives little "
          "beyond gauge alignment here (a stronger negative). Ceiling stays model-set throughout.")


if __name__ == "__main__":
    main()
