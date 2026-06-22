"""How few blocks reproduce the decode? -- the compactness / head-tail test on real circuits.

Sharpens the decode-circuit finding from ATTRIBUTION (where the decode mass is) to COMPACTNESS (the
smallest block set whose partial-sum argmax IS the decode). This is the empirical instance of the
kernel-proved head/tail theorem (i-orca HeadTail.thy): the compact HEAD reproduces the decode exactly
when it dominates the TAIL. For the global top-k blocks S_k (ranked by mean |contribution| to the
decode), we ask what fraction of positions keep the model's decode as the argmax over the candidate
set using ONLY S_k -- and the minimal k for 90/95/99%.

Caveat (honest): the reconstruction is over the dump's top-K logit candidates (the decode's strongest
rivals), not the full vocab -- so this is "the k-block circuit keeps the decode above its K-1 nearest
competitors", the hardest local test, not a full-vocab guarantee.

Run:  python experiments/decode_compactness.py /tmp/d_instruct_prose.jsonl
"""

from __future__ import annotations

import argparse

import torch

from pil.fieldrun_io import load_pil_dump


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    a = p.parse_args()

    b = load_pil_dump(a.path)
    contrib = b.contrib                       # (N, nb, K); idx 0 = the model's decode (cands logit-sorted)
    N, nb, _ = contrib.shape
    full = contrib.sum(dim=1)                 # (N, K) full logits over candidates
    assert (full.argmax(1) == 0).float().mean() > 0.99, "decode not at idx 0 (seam mismatch)"

    # GLOBAL block ranking by mean |contribution to the decode|
    order = torch.argsort(contrib[:, :, 0].abs().mean(0), descending=True)   # (nb,)

    print(f"[compactness] {b.meta['path']}: N={N}, nb={nb}")
    print(f"\n  global top-k blocks reproduce the decode (argmax over {contrib.shape[2]} candidates):")
    print(f"  {'k':>4}{'agree%':>9}{'dom-margin':>12}")
    agree_at = {}
    prev = -1
    for k in range(1, nb + 1):
        S = order[:k]
        partial = contrib[:, S, :].sum(dim=1)                                 # (N, K)
        agree = (partial.argmax(1) == 0).float().mean().item()
        # head/tail domination margin: decode logit - best competitor, using S
        dom = (partial[:, 0] - partial.scatter(1, torch.zeros(N, 1, dtype=torch.long),
                                               float("-inf")).max(1).values).mean().item()
        for thr in (0.90, 0.95, 0.99):
            if thr not in agree_at and agree >= thr:
                agree_at[thr] = k
        if k <= 6 or agree != prev or k == nb:                               # print early ks + changes
            print(f"  {k:>4}{agree*100:>8.1f}%{dom:>12.2f}")
        prev = agree

    # PER-POSITION: each position's own minimal block count to reproduce its decode
    per_pos_k = []
    for i in range(N):
        ci = contrib[i]                                                       # (nb, K)
        oi = torch.argsort(ci[:, 0].abs(), descending=True)
        for k in range(1, nb + 1):
            if ci[oi[:k]].sum(0).argmax().item() == 0:
                per_pos_k.append(k)
                break
        else:
            per_pos_k.append(nb)
    ppk = torch.tensor(per_pos_k, dtype=torch.float)

    print("\n  minimal GLOBAL k (one fixed circuit for all positions):")
    for thr in (0.90, 0.95, 0.99):
        kk = agree_at.get(thr, nb)
        print(f"    {int(thr*100)}% of positions reproduced by the top-{kk}/{nb} blocks "
              f"({kk/nb*100:.0f}% of blocks)")
    print(f"\n  minimal PER-POSITION k: median {ppk.median():.0f}, mean {ppk.mean():.1f}, "
          f"90th-pct {torch.quantile(ppk, 0.9):.0f} / {nb}")


if __name__ == "__main__":
    main()
