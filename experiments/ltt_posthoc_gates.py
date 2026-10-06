"""POST-HOC: retrain the study's transformer gates (seeds 41-44, same recipe) and repeat the split diagnostic.
cert16 taken as all-true (study students: 0.998-1.000)."""

import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import conditional_certificate as cc
from certified_student import HELD_OUT, WORDS
from ltt_certificate import SEEDS, grid_from_train, ltt_tau

dev = torch.device("cuda")
z = np.load(sys.argv[1])
sid = torch.tensor(z["sid"], device=dev)
dec = torch.tensor(z["dec"], device=dev)
n, k, T = sid.shape
seen = torch.tensor([i for i, w in enumerate(WORDS) if w not in HELD_OUT], device=dev)
order = np.random.default_rng(SEEDS["split"]).permutation(n)
tr, ca, te = (torch.tensor(a, device=dev) for a in (order[:2400], order[2400:4200], order[4200:]))
sc = (dec[:, seen] == dec[:, seen[:1]]).all(1)
E = torch.tensor(z["wte"], device=dev)
ones = torch.ones(1800, dtype=torch.bool, device=dev)
allc = torch.arange(n, device=dev)
for seed in (41, 42, 43, 44):
    g = cc.train_gate(E, T, sid, sc.float(), tr, seen, seed, cc.GATE_STEPS, dev)
    s = cc.gate_scores(g, sid, allc)
    grid = grid_from_train(s[tr])

    def run(a, b, grid=grid, s=s):
        ch, _ = ltt_tau(grid, s[a], ones, sc[a])
        if ch is None:
            return None
        iss = s[b] >= ch[0]
        return ch[2] / ch[1], float((~sc[b][iss]).float().mean()), float(iss.float().mean())

    r0 = run(ca, te)
    rs = run(te, ca)
    rng = np.random.default_rng(1)
    hold = order[2400:]
    R = []
    for _ in range(200):
        p = torch.tensor(rng.permutation(hold), device=dev)
        x = run(p[:1800], p[1800:])
        if x:
            R.append(x)
    R = np.array(R)
    fmt = lambda x: "none issued" if x is None else f"cal {x[0]:.3f} test {x[1]:.3f} cov {x[2]:.3f}"  # noqa: E731
    print(
        f"seed {seed}: prereg split {fmt(r0)} | swapped {fmt(rs)} | 200 re-splits ({len(R)} issued): "
        f"mean cal {R[:, 0].mean():.3f} mean test {R[:, 1].mean():.3f} "
        f"test>0.10 {np.mean(R[:, 1] > 0.10):.3f}",
        flush=True,
    )
