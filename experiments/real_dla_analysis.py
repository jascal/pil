"""Do the synthetic routing findings transfer to a real model? (fieldrun --pil-dump seam)

Loads a `fieldrun --pil-dump` of per-block DLA incidences from a real LM (e.g. Qwen2.5-0.5B) and
runs the real-model analogues of the synthetic analyses:

  - RANK    : source-PR over blocks toward the target, (Σ_b c_b)²/Σ_b c_b² -- the effective number of
              DLA blocks the decode actually uses, vs nb total (RoutingRank: adjustments live in <=nb dims).
  - COHERENCE: treat each position's block-contribution-toward-target-vs-competitor as a routing feature
              f_i in R^nb; measure mean |cosine| among them vs the Welch floor for (N positions, nb blocks)
              -- the §5f question, on real data: does the model pack near the Welch-optimal coherence?
  - MARGIN  : recomputed from the incidences (sanity vs the dumped margin) + its relation to PR/coherence.

Tests whether the synthetic story (M-dominated, near-Welch packing, coherence decoupled from margin) holds
on a real LM where the routing features are the model's own circuits, not planted XOR.

Run:  python experiments/real_dla_analysis.py /tmp/pil_real.jsonl
"""

from __future__ import annotations

import argparse

import torch

from pil.fieldrun_io import load_pil_dump
from pil.geometry import participation_ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--gamma", type=float, default=1.0)
    a = p.parse_args()

    b = load_pil_dump(a.path)
    contrib = b.contrib                       # (N, nb, K); cands sorted by logit (idx 0 = decode, 1 = runner-up)
    N, nb, K = contrib.shape
    print(f"[real-dla] {b.meta['path']}: N={N} positions, nb={nb} DLA blocks, K={K} candidates")

    # Analyse the model's OWN DECODE (argmax vs runner-up), not the ground-truth target -- the PIL
    # mechanism is about what the model computes, not whether it is correct. cands[:,0] is the argmax.
    logits = contrib.sum(dim=1)               # (N, K) reconstructed candidate logits
    recon_ok = (logits.argmax(1) == 0).float().mean().item()   # idx 0 should be the argmax (sanity)
    acc = (b.tgt_idx == 0).float().mean().item()               # how often the decode == ground truth
    margin = logits[:, 0] - logits[:, 1]      # decode margin (matches dumped top1-top2)

    c0 = contrib[:, :, 0]                      # (N, nb) block contributions to the decode
    c1 = contrib[:, :, 1]                      # to the runner-up
    pr = participation_ratio(c0, dim=1)       # effective # blocks driving the decode (RoutingRank side)

    f = c0 - c1                                # (N, nb) routing feature: decode vs runner-up in block space
    fn = f / (f.norm(dim=1, keepdim=True) + 1e-8)
    G = (fn @ fn.T).abs()
    mu = ((G.sum() - N) / (N * (N - 1))).item()
    welch = ((N - nb) / (nb * (N - 1))) ** 0.5 if N > nb else 0.0

    print(f"\n  sanity: incidences reconstruct the decode argmax {recon_ok:.2f} of the time; "
          f"\n          decode==ground-truth {acc:.2f} (the model's own next-token accuracy)")
    print(f"  margin recompute vs dump: mean {margin.mean():.2f} vs {b.margins.mean():.2f}")
    print("\n  RANK (RoutingRank side -- adjustments live in <= nb dims):")
    print(f"    source-PR over blocks toward the decode : mean {pr.mean():.1f} / {nb}  "
          f"(median {pr.median():.1f})  -> effective blocks used; large slack = nb - PR")
    print("\n  COHERENCE (§5f / RoutingWelch side):")
    print(f"    N={N} routing features in nb={nb} block-dims  (overpacked {N/nb:.1f}x)")
    if welch > 0:
        print(f"    realized mean |cos| mu = {mu:.3f}   vs Welch floor {welch:.3f}   (ratio {mu/welch:.2f}x)")
    else:
        print(f"    realized mu = {mu:.3f}  (N <= nb, Welch floor 0)")
    print("\n  MARGIN:")
    print(f"    mean {margin.mean():.2f}, median {margin.median():.2f}, "
          f"10th-pct {torch.quantile(margin, 0.1).item():.2f}  (logit units)")

    print("\n  transfer read (descriptive):")
    print(f"    - rank: the decode effectively uses ~{pr.mean():.0f} of {nb} blocks "
          f"({'concentrated -> rank slack, matches synthetic' if pr.mean() < nb / 2 else 'diffuse'}).")
    if welch > 0:
        near = mu < 2 * welch
        verdict = "near-optimal (§5f)" if near else "LOOSER than synthetic (real-model diff)"
        tag = "NEAR" if near else "ABOVE"
        print(f"    - packing: realized coherence {tag} the Welch floor -> {verdict}.")


if __name__ == "__main__":
    main()
