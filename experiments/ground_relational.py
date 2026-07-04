"""Retrieved/computed frontier: does RELATIONAL/compositional structure need higher RANK than LEXICAL?

At the final residual, accuracy is ~linear for anything the model computes -- so the retrieved/computed
signature is RANK, not accuracy (cf lisp arithmetic = 2x). For each concept we balance (chance 0.5) and
linear-probe at TRUNCATED ranks (top-k PCA of the residual), then report the rank needed to reach ~ceiling.
LEXICAL (func, cap) should saturate at LOW rank (retrieved = a direction). RELATIONAL/compositional
(is_repeat/induction, prev_func, inside_paren = counting/matching) should need HIGHER rank if they are
genuinely computed.

Run: cd pil && .venv/bin/python experiments/ground_relational.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "relational_pythia70m.pt"
LEXICAL = ["func", "cap"]
RELATIONAL = ["is_repeat", "local_repeat", "prev_func", "inside_paren"]
RANKS = [1, 2, 4, 8, 16, 32, 64, 128]


def probe(Xtr, ytr, Xte, yte, steps=500, lr=0.05):
    W = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(Xtr @ W + b, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((((Xte @ W + b) > 0).float() == yte).float().mean())


def balanced(r, y, seed):
    g = torch.Generator().manual_seed(seed)
    pos = torch.where(y == 1)[0]
    neg = torch.where(y == 0)[0]
    neg = neg[torch.randperm(len(neg), generator=g)[:len(pos)]]
    keep = torch.cat([pos, neg])[torch.randperm(2 * len(pos), generator=g)]
    rk = r[keep]
    rk = (rk - rk.mean(0)) / (rk.std(0) + 1e-6)
    cut = int(0.7 * len(keep))
    return rk, y[keep], torch.arange(cut), torch.arange(cut, len(keep))


def rank_curve(r, y, seed):
    rk, yk, tr, te = balanced(r, y, seed)
    Rc = rk[tr] - rk[tr].mean(0)
    _, _, Vh = torch.linalg.svd(Rc, full_matrices=False)
    accs = {}
    for k in RANKS:
        Pk = Vh[:k]
        accs[k] = probe(rk[tr] @ Pk.T, yk[tr], rk[te] @ Pk.T, yk[te])
    accs["full"] = probe(rk[tr], yk[tr], rk[te], yk[te])
    return accs


def rank_to(accs, frac=0.95):
    """smallest rank reaching frac * full-rank accuracy (above 0.5 chance)."""
    ceil = accs["full"]
    target = 0.5 + frac * (ceil - 0.5)
    for k in RANKS:
        if accs[k] >= target:
            return k
    return RANKS[-1]


def main():
    d = torch.load(DATA)
    r = d["r"].float()
    names = d["names"]
    labs = d["labels"]
    print(f"retrieved/computed frontier: rank-to-decode, pythia-70m  r {tuple(r.shape)} (balanced)\n")
    hdr = "".join(f"{k:>6}" for k in RANKS) + f"{'full':>7}{'rank95':>8}"
    print(f"{'concept':>13}{hdr}   type")
    summ = {"lexical": [], "relational": []}
    for nm in LEXICAL + RELATIONAL:
        i = names.index(nm)
        y = labs[:, i].float()
        if int(y.sum()) < 300:
            print(f"{nm:>13}  too few positives ({int(y.sum())}); skipping")
            continue
        curves = [rank_curve(r, y, s) for s in (0, 1)]
        acc = {k: sum(c[k] for c in curves) / 2 for k in RANKS + ["full"]}
        r95 = sum(rank_to(c) for c in curves) / 2
        typ = "lexical" if nm in LEXICAL else "relational"
        summ[typ].append(r95)
        row = "".join(f"{acc[k]:>6.2f}" for k in RANKS) + f"{acc['full']:>7.2f}{r95:>8.0f}"
        print(f"{nm:>13}{row}   {typ}", flush=True)
    lx = sum(summ["lexical"]) / max(len(summ["lexical"]), 1)
    rl = sum(summ["relational"]) / max(len(summ["relational"]), 1)
    ratio = rl / max(lx, 1e-9)
    print(f"\nmean rank-to-95%-ceiling:  lexical {lx:.1f}  relational {rl:.1f}  ratio {ratio:.1f}x")
    print("read: relational/compositional needing HIGHER rank than lexical = the retrieved->computed\n"
          "frontier IS compositional (grounds the forge-tax rank signature; both decode at FULL rank, but\n"
          "computed needs more dims). ~equal rank = the computed core is narrower than composition.")


if __name__ == "__main__":
    main()
