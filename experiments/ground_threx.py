"""Grounding bridge rung 1 — concepts as hyperplanes in THREX's real residual space (from fieldrun).

threx is a tiny rope transformer (fieldrun-decompiled, rosetta-mined) with KNOWN idiom structure. We dumped
its per-position residual r = Σ_b d̃_b (decode frame, ⟨r,U_v⟩ = logit_v) via `fieldrun --source-dump`. threx
has three known idioms, identified by the current token:
  cur==9 (na)  -> RETRIEVED  (fixed: na -> dø(16))
  cur==6 (wø)  -> SELECTED   (place-gated: PLACE_OBJ[place], a 3-way lookup)
  cur==7 (gɪ)  -> COMPOSED   (additive THINGS[strength(Bi)+strength(Bj)] -- threx's COMPUTED slice)

Tests: (1) do concept-hyperplanes on r RECOVER which idiom is firing? (2) linear-probe ceiling for idiom-type
(is threx's residual more linearly structured than the toy's ~0.64?); (3) the retrieved/computed GEOMETRY --
rank-truncated decodability of the idiom ANSWER (target) per idiom: does COMPOSED (computed/additive) need
MORE rank than RETRIEVED/SELECTED (retrieved lookups)? -- grounding the retrieved/computed thesis in a real
model's geometry.

Run: cd pil && .venv/bin/python experiments/ground_threx.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DUMP = SP / "threx_source.jsonl"
BUNDLE = Path("/home/allans/code/fieldrun/sim/data/threx")
IDIOM = {9: "RETRIEVED", 6: "SELECTED", 7: "COMPOSED"}


def load_dump():
    r, cur, tgt, pred = [], [], [], []
    for line in open(DUMP):
        d = json.loads(line)
        r.append(np.sum(np.array(d["d"], dtype=np.float32), axis=0))     # r = Σ_b d̃_b  (32,)
        cur.append(d["cur"])
        tgt.append(d["target"])
        pred.append(d["pred"])
    return (torch.tensor(np.stack(r)), torch.tensor(cur), torch.tensor(tgt), torch.tensor(pred))


def load_U():
    h = json.load(open(str(BUNDLE) + ".fieldrun.json"))
    blob = open(str(BUNDLE) + ".fieldrun.bin", "rb").read()
    a = next(x for x in h["arrays"] if x["name"] == "embed")
    U = np.frombuffer(blob, "<f4", count=int(np.prod(a["shape"])), offset=a["offset"]).reshape(a["shape"])
    return torch.tensor(U.copy())                                        # (31, 32) tied unembed


def linear_probe(Xtr, ytr, Xte, yte, nclass, steps=400, lr=0.05):
    d = Xtr.shape[1]
    W = torch.zeros(d, nclass, requires_grad=True)
    b = torch.zeros(nclass, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(Xtr @ W + b, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((( Xte @ W + b).argmax(1) == yte).float().mean())


class GeoConcepts(torch.nn.Module):
    def __init__(self, d, K, nclass):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(K, d) * 0.3)
        self.b = torch.nn.Parameter(torch.zeros(K))
        self.head = torch.nn.Linear(K, nclass)
        self.K = K

    def membership(self, r, tau=1.0):
        return torch.sigmoid((r @ self.U.T - self.b) / tau)

    def forward(self, r, tau=1.0):
        return self.head(self.membership(r, tau))


def main():
    r, cur, tgt, pred = load_dump()
    U = load_U()
    recon = float((( r.float() @ U.T).argmax(1) == pred).float().mean())
    print(f"threx residuals: r {tuple(r.shape)}  recon(r@U.T argmax == pred) = {recon:.3f}  "
          "(1.0 confirms r is the faithful decode residual)\n")

    # label idiom-firing positions
    lab = torch.tensor([{9: 0, 6: 1, 7: 2}.get(int(c), -1) for c in cur])
    mask = lab >= 0
    ridi, yidi = r[mask].float(), lab[mask]
    from collections import Counter
    counts = {IDIOM[k]: v for k, v in Counter(int(c) for c in cur.tolist() if int(c) in IDIOM).items()}
    print(f"idiom-firing positions: {counts}  (total {int(mask.sum())})")

    # split
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(ridi.shape[0], generator=g)
    cut = int(0.7 * len(perm))
    tr, te = perm[:cut], perm[cut:]

    # (2) linear-probe ceiling for idiom-type
    ceil = linear_probe(ridi[tr], yidi[tr], ridi[te], yidi[te], 3)
    print(f"\n(2) idiom-type linear-probe ceiling: {ceil:.3f}  (toy was ~0.64; threx residual structure)")

    # (1) geometric concepts recover idiom-type
    gc = GeoConcepts(d=r.shape[1], K=12, nclass=3)
    opt = torch.optim.Adam(gc.parameters(), lr=0.03)
    for step in range(2000):
        tau = max(0.4, 1.3 - 0.9 * step / 2000)
        loss = F.cross_entropy(gc(ridi[tr], tau), yidi[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        gacc = float((gc(ridi[te], 0.4).argmax(1) == yidi[te]).float().mean())
    print(f"(1) geometric-concept idiom-type accuracy: {gacc:.3f}  (concept-hyperplanes recover the idiom)")

    # (3) retrieved/computed GEOMETRY: rank-truncated decodability of the ANSWER (target), per idiom
    print("\n(3) rank-truncated ANSWER (target) decodability per idiom "
          "(computed=COMPOSED should need MORE rank):")
    print(f"{'rank':>6}{'RETRIEVED':>11}{'SELECTED':>10}{'COMPOSED':>10}")
    rc = ridi - ridi.mean(0)
    _, _, Vh = torch.linalg.svd(rc, full_matrices=False)
    for k in (1, 2, 4, 8, 16, 32):
        Pk = Vh[:k]                                                      # top-k principal directions
        row = []
        for idi in range(3):
            m = yidi == idi
            idx = torch.where(m)[0]
            if len(idx) < 20:
                row.append(float("nan"))
                continue
            pr = torch.randperm(len(idx), generator=torch.Generator().manual_seed(idi))
            c2 = int(0.7 * len(idx))
            itr, ite = idx[pr[:c2]], idx[pr[c2:]]
            Xtr = (ridi[itr] @ Pk.T)
            Xte = (ridi[ite] @ Pk.T)
            # remap answers (target) to dense class ids within this idiom
            ans = tgt[mask]
            uniq = ans[m].unique()
            remap = {int(v): j for j, v in enumerate(uniq)}
            ytr = torch.tensor([remap[int(v)] for v in ans[itr]])
            yte = torch.tensor([remap[int(v)] for v in ans[ite]])
            row.append(linear_probe(Xtr, ytr, Xte, yte, len(uniq)))
        print(f"{k:>6}{row[0]:>11.3f}{row[1]:>10.3f}{row[2]:>10.3f}", flush=True)
    print("\nread: (1)+(2) grounding WORKS on a REAL fieldrun model -- concept-hyperplanes recover threx's "
          "known idioms (1.00) and idiom-type is near-perfectly linear (0.99, vs the toy's 0.64 = ceiling "
          "RISES with model quality). (3) INCONCLUSIVE: data-starved per idiom + answer-ENTROPY confounds "
          "rank (RETRIEVED constant->rank-1; SELECTED/COMPOSED both low/noisy) -- does not cleanly show the "
          "retrieved/computed split; needs more data + an entropy-controlled design.")


if __name__ == "__main__":
    main()
