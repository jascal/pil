"""Grounding bridge rung 4 — concepts as hyperplanes in PYTHIA residuals (the real-LLM, info-barred end).

The final curriculum rung. pythia at increasing scale (70m -> 410m -> 1b) on real text (wikitext), residuals
extracted by tropic/extract_stories.py (last_hidden_state -> logits via embed_out, same decode frame as the
other rungs). Gradeable structure = function-vs-content word class, probed with the CLEAN deterministic label
(the CURRENT token's class -- the fix from the stories-rung entropy confound), plus the stochastic
next-token label for contrast.

The load-bearing question the whole curriculum was built to reach: does the linear-compositional ceiling STAY
high / keep RISING as we scale to a real LLM (linear-representation hypothesis holds), or does it plateau/fall
in the genuinely data-rich real-LLM regime?

Run: cd pil && .venv/bin/python experiments/ground_pythia.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
LADDERS = {
    "pythia": [("pythia-70m", "pythia70m_stories.pt"), ("pythia-410m", "pythia410m_stories.pt"),
               ("pythia-1b", "pythia1b_stories.pt")],
    "qwen2.5": [("qwen-0.5b", "qwen05b_stories.pt"), ("qwen-1.5b", "qwen15b_stories.pt"),
                ("qwen-3b", "qwen3b_stories.pt"), ("qwen-7b", "qwen7b_stories.pt")],
}


def linear_probe(Xtr, ytr, Xte, yte, nclass, steps=600, lr=0.05):
    W = torch.zeros(Xtr.shape[1], nclass, requires_grad=True)
    b = torch.zeros(nclass, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(Xtr @ W + b, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((Xte @ W + b).argmax(1) == yte).float().mean())


class GeoConcepts(torch.nn.Module):
    def __init__(self, d, K, nclass):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(K, d) * 0.3)
        self.b = torch.nn.Parameter(torch.zeros(K))
        self.head = torch.nn.Linear(K, nclass)

    def forward(self, r, tau=1.0):
        return self.head(torch.sigmoid((r @ self.U.T - self.b) / tau))


def run(name, path):
    f = SP / path
    if not f.exists():
        print(f"[{name}] {path} not found; skipping")
        return None
    d = torch.load(f)
    r = d["r"].float()
    fc, fn = d["is_func_cur"], d["is_func_nxt"]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(r.shape[0], generator=g)
    cut = int(0.7 * r.shape[0])
    tr, te = perm[:cut], perm[cut:]

    def ceil(y):
        base = max(float(y.float().mean()), 1 - float(y.float().mean()))
        return linear_probe(r[tr], y[tr], r[te], y[te], 2), base

    c_cur, b_cur = ceil(fc)
    c_tgt, b_tgt = ceil(fn)
    # geometric concepts on the clean (deterministic) label
    gc = GeoConcepts(d=r.shape[1], K=16, nclass=2)
    opt = torch.optim.Adam(gc.parameters(), lr=0.03)
    for step in range(2500):
        tau = max(0.4, 1.3 - 0.9 * step / 2500)
        idx = tr[torch.randint(len(tr), (256,), generator=g)]
        loss = F.cross_entropy(gc(r[idx], tau), fc[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        gacc = float((gc(r[te], 0.4).argmax(1) == fc[te]).float().mean())
    print(f"[{name:>12}] d={r.shape[1]:>4} n={r.shape[0]} || CUR ceiling {c_cur:.3f} (base {b_cur:.3f}) "
          f"geo {gacc:.3f}  [clean] || TGT ceiling {c_tgt:.3f} (base {b_tgt:.3f})  [entropy]")
    return c_cur


def main():
    import sys
    fams = sys.argv[1:] or list(LADDERS)
    print("LLM grounding scale sweep — function/content, CLEAN deterministic CUR label")
    print("baseline ladder: toy 0.64 < stories 0.84-0.88 < threx/lisp 0.99. is ~0.95 universal across "
          "family & scale?\n")
    for fam in fams:
        ladder = []
        for name, path in LADDERS[fam]:
            c = run(name, path)
            if c is not None:
                ladder.append((name, c))
        if len(ladder) >= 2:
            trend = " -> ".join(f"{n.split('-')[-1]}:{c:.3f}" for n, c in ladder)
            rising = ladder[-1][1] > ladder[0][1] + 0.02
            print(f"  [{fam}] CUR-ceiling trend: {trend}  "
                  f"({'RISES' if rising else 'plateaus/flat'} with scale)\n")
    print("read: if the clean (CUR) ceiling stays ~0.95 across BOTH families and up to 7B, the "
          "linear-representation grounding of grammar is a UNIVERSAL property of real LLMs (not "
          "pythia-specific). A family/scale-dependent fall would localize where it breaks. "
          "(TGT stays entropy-confounded -- ignore its absolute level.)")


if __name__ == "__main__":
    main()
