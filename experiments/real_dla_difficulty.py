"""Is the rank-slack / packing-looseness a CORPUS-DIFFICULTY artifact? (the difficulty dial)

Hypothesis (j.allan.scott): on predictable text most next-tokens are retrieval/induction -- a few
blocks, loose packing -- so the observed slack may be an easy-corpus artifact, not a model property.
Test it WITHIN a corpus: split positions by the model's own decode margin (the difficulty proxy
fieldrun dumps; low margin = hard/contested/computed, high margin = easy/confident/retrieved) into
terciles, and recompute the rank (source-PR over blocks) and the routing-feature coherence vs the
Welch floor per tercile.

Prediction if the hypothesis holds: hard (low-margin) positions engage MORE blocks (higher PR) and
pack TIGHTER (μ closer to Welch). Flat across terciles => the slack is intrinsic, not difficulty-driven.

Run:  python experiments/real_dla_difficulty.py /tmp/d_instruct_prose.jsonl
"""

from __future__ import annotations

import argparse

import torch

from pil.fieldrun_io import load_pil_dump
from pil.geometry import participation_ratio


def bin_stats(contrib, nb):
    """PR/nb and routing-feature coherence μ vs Welch floor for a subset of positions."""
    n = contrib.shape[0]
    c0, c1 = contrib[:, :, 0], contrib[:, :, 1]
    pr = participation_ratio(c0, dim=1).mean().item()
    f = c0 - c1
    fn = f / (f.norm(dim=1, keepdim=True) + 1e-8)
    G = (fn @ fn.T).abs()
    mu = ((G.sum() - n) / (n * (n - 1))).item() if n > 1 else float("nan")
    welch = ((n - nb) / (nb * (n - 1))) ** 0.5 if n > nb else 0.0
    return n, pr, mu, welch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--bins", type=int, default=3)
    a = p.parse_args()

    b = load_pil_dump(a.path)
    contrib = b.contrib
    N, nb, _ = contrib.shape
    logits = contrib.sum(dim=1)
    margin = logits[:, 0] - logits[:, 1]

    order = torch.argsort(margin)            # ascending: hardest (lowest margin) first
    edges = [int(round(i * N / a.bins)) for i in range(a.bins + 1)]
    names = (["hard", "mid", "easy"] if a.bins == 3 else [f"q{i+1}" for i in range(a.bins)])

    print(f"[difficulty] {b.meta['path']}: N={N}, nb={nb}; split by decode margin into {a.bins} bins")
    print(f"\n{'bin':<7}{'n':>5}{'margin-range':>16}{'PR/nb':>10}{'mu':>8}{'welch':>8}{'mu/welch':>10}")
    print("-" * 64)
    for i in range(a.bins):
        idx = order[edges[i]:edges[i + 1]]
        sub = contrib[idx]
        m = margin[idx]
        n, pr, mu, welch = bin_stats(sub, nb)
        ratio = mu / welch if welch > 0 else float("nan")
        rng = f"[{m.min():.2f},{m.max():.2f}]"
        print(f"{names[i]:<7}{n:>5}{rng:>16}{pr:>5.0f}/{nb:<4}{mu:>8.3f}{welch:>8.3f}{ratio:>10.2f}")

    # verdict
    hard = contrib[order[:edges[1]]]
    easy = contrib[order[edges[-2]:]]
    pr_hard = participation_ratio(hard[:, :, 0], dim=1).mean().item()
    pr_easy = participation_ratio(easy[:, :, 0], dim=1).mean().item()
    verdict = "HARD uses more blocks (difficulty-driven)" if pr_hard > pr_easy + 1 else "flat/inverse (intrinsic)"
    print(f"\n  rank vs difficulty: hard PR {pr_hard:.0f} vs easy PR {pr_easy:.0f} / {nb}  -> {verdict}")


if __name__ == "__main__":
    main()
