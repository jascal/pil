"""Grounding bridge rung 3 — concepts as hyperplanes in a TinyStories model's residual (REAL LANGUAGE).

Rung 3 = real language, near-masterable. stories15M (Karpathy TinyStories, a fieldrun rope bundle in
rosetta/models). Residuals r = Σ_b d̃_b via `fieldrun --source-dump` on the story corpus (recon 1.00). The
gradeable structure is the FUNCTION-vs-CONTENT word class of the target token (closed-class function words =
grammar; open-class content = retrieved lexis -- the forge-tax split), decoded from the bundle lexicon.

The load-bearing question: does the linear-compositional ceiling STAY high on REAL language (like threx/lisp
at 0.99) or FALL (real language is messier than clean synthetic structure)? Plus: does grounding recover the
grammar/lexis distinction, and does the function-word (grammar) subspace differ in effective rank from
content?

Run: cd pil && .venv/bin/python experiments/ground_stories.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
MODELS = {
    "stories260K": (SP / "stories260k_source.jsonl", "stories260K"),
    "stories15M": (SP / "stories_source.jsonl", "stories15M"),
}

_FW = """a an the this that these those and or but nor so yet for of to in on at by with from into over
under between among through during before after above below up down out off about against along around
i you he she it we they me him her us them my your his its our their mine yours hers ours theirs myself
is am are was were be been being have has had do does did will would shall should can could may might must
not no as if then than when while where who whom whose which what how why because although though since
unless until whether both either neither each every all any some few many much more most other another such"""
FUNCTION = set(_FW.split())


def load_dump(dump):
    r, cur, tgt, pred = [], [], [], []
    for line in open(dump):
        d = json.loads(line)
        r.append(np.sum(np.array(d["d"], dtype=np.float32), axis=0))
        cur.append(d["cur"])
        tgt.append(d["target"])
        pred.append(d["pred"])
    return torch.tensor(np.stack(r)), torch.tensor(cur), torch.tensor(tgt), torch.tensor(pred)


def load_U(model):
    stem = f"/home/allans/code/rosetta/models/{model}/bundle"
    h = json.load(open(stem + ".fieldrun.json"))
    blob = open(stem + ".fieldrun.bin", "rb").read()
    a = next(x for x in h["arrays"] if x["name"] == "embed")
    U = np.frombuffer(blob, "<f4", count=int(np.prod(a["shape"])), offset=a["offset"]).reshape(a["shape"])
    return torch.tensor(U.copy())


def lexicon(model):
    return json.load(open(f"/home/allans/code/rosetta/models/{model}/lexicon.json"))["tokens"]


def word_of(tokens, i):
    g = tokens[int(i)][0] if 0 <= int(i) < len(tokens) else ""
    return g.replace("▁", " ").strip().lower()          # strip SentencePiece ▁


def linear_probe(Xtr, ytr, Xte, yte, nclass, steps=500, lr=0.05):
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


def effective_rank(X):
    Xc = X - X.mean(0)
    ev = torch.linalg.svdvals(Xc) ** 2
    ev = ev / ev.sum()
    return float(1.0 / (ev ** 2).sum())


class GeoConcepts(torch.nn.Module):
    def __init__(self, d, K, nclass):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(K, d) * 0.3)
        self.b = torch.nn.Parameter(torch.zeros(K))
        self.head = torch.nn.Linear(K, nclass)

    def forward(self, r, tau=1.0):
        return self.head(torch.sigmoid((r @ self.U.T - self.b) / tau))


def run_model(model, dump):
    if not dump.exists():
        print(f"[{model}] dump not found ({dump.name}); skipping")
        return None
    r, cur, tgt, pred = load_dump(dump)
    U = load_U(model)
    tokens = lexicon(model)
    recon = float(((r @ U.T).argmax(1) == pred).float().mean())
    rf = r.float()
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(r.shape[0], generator=g)
    cut = int(0.7 * r.shape[0])
    tr, te = perm[:cut], perm[cut:]

    def probe_class(toks):
        y = torch.tensor([int(word_of(tokens, t) in FUNCTION) for t in toks])
        base = max(float(y.float().mean()), 1 - float(y.float().mean()))
        ceil = linear_probe(rf[tr], y[tr], rf[te], y[te], 2)
        return ceil, base

    # DETERMINISTIC label (current token is known) vs STOCHASTIC label (actual next token = language entropy)
    c_cur, b_cur = probe_class(cur)
    c_tgt, b_tgt = probe_class(tgt)
    print(f"[{model}] r {tuple(r.shape)} recon {recon:.3f} || "
          f"CUR-token func/content ceiling {c_cur:.3f} (base {b_cur:.3f})  [deterministic] || "
          f"TGT-token ceiling {c_tgt:.3f} (base {b_tgt:.3f})  [stochastic/entropy]")
    return c_cur


def main():
    print("STORIES rung (TinyStories, real language): function/content-word grounding, scale comparison")
    print("baselines: toy 0.64 / threx 0.99 / lisp 0.99 -- does the ceiling STAY high on REAL language,")
    print("and RISE from 260K -> 15M (controlled: same corpus & task, only model size)?\n")
    ceils = {}
    for model, (dump, _) in MODELS.items():
        c = run_model(model, dump)
        if c is not None:
            ceils[model] = c
    if len(ceils) == 2:
        lo, hi = ceils["stories260K"], ceils["stories15M"]
        print(f"\nCUR-token (deterministic) ceiling scale: 260K {lo:.3f} -> 15M {hi:.3f}  "
              f"({'RISES' if hi > lo + 0.02 else 'flat/falls'} with model size)")
    print("\nread: the CUR-token (deterministic) ceiling is the clean 'is grammar linearly present' measure; "
          "the TGT-token (stochastic) ceiling is confounded by real-language next-token ENTROPY (unlike "
          "threx/lisp whose targets are deterministic). If CUR stays high on real language, the "
          "linear-representation hypothesis holds; the drop in TGT is language entropy, not weak structure.")


if __name__ == "__main__":
    main()
