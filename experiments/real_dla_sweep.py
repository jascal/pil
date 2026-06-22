"""Tabulate the real-DLA analysis across models / corpora (the §5i transfer sweep).

Pass labelled fieldrun --pil-dump files as `label=path` pairs; prints one row per condition with the
rank (source-PR over blocks toward the decode), the routing-feature coherence μ vs the Welch floor,
and the decode margin. Tests whether the two synthetic findings — rank-slack and (non-)near-Welch
packing — hold across scale (0.5B→1.5B→3B) and corpus (prose vs code).

Run:  python experiments/real_dla_sweep.py 0.5B-prose=d0.jsonl 1.5B-prose=d1.jsonl 3B-prose=d2.jsonl
"""

from __future__ import annotations

import sys

from pil.fieldrun_io import load_pil_dump
from pil.geometry import participation_ratio


def analyse(path):
    b = load_pil_dump(path)
    contrib = b.contrib                       # (N, nb, K); idx 0 = decode, 1 = runner-up
    N, nb, K = contrib.shape
    logits = contrib.sum(dim=1)
    recon = (logits.argmax(1) == 0).float().mean().item()
    margin = (logits[:, 0] - logits[:, 1])
    c0, c1 = contrib[:, :, 0], contrib[:, :, 1]
    pr = participation_ratio(c0, dim=1).mean().item()
    f = c0 - c1
    fn = f / (f.norm(dim=1, keepdim=True) + 1e-8)
    G = (fn @ fn.T).abs()
    mu = ((G.sum() - N) / (N * (N - 1))).item()
    welch = ((N - nb) / (nb * (N - 1))) ** 0.5 if N > nb else 0.0
    return {
        "N": N, "nb": nb, "recon": recon, "pr": pr, "pr_frac": pr / nb,
        "mu": mu, "welch": welch, "ratio": (mu / welch if welch > 0 else float("nan")),
        "margin": margin.mean().item(),
    }


def main():
    items = []
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        label, path = arg.split("=", 1)
        try:
            items.append((label, analyse(path)))
        except (FileNotFoundError, ValueError) as e:
            print(f"  skip {label}: {e}")

    if not items:
        print("usage: real_dla_sweep.py label1=dump1.jsonl label2=dump2.jsonl ...")
        return

    hdr = (f"{'condition':<22}{'N':>5}{'nb':>4}{'recon':>7}{'PR/nb':>9}"
           f"{'mu':>8}{'welch':>8}{'mu/welch':>10}{'margin':>8}")
    print(hdr)
    print("-" * len(hdr))
    for label, r in items:
        print(f"{label:<22}{r['N']:>5}{r['nb']:>4}{r['recon']:>7.2f}"
              f"{r['pr']:>5.0f}/{r['nb']:<3}{r['mu']:>8.3f}{r['welch']:>8.3f}"
              f"{r['ratio']:>10.2f}{r['margin']:>8.2f}")
    print("\nrank: PR/nb « 1 = the decode is rank-concentrated (RoutingRank slack).")
    print("packing: mu/welch ~1 = near-Welch-optimal (synthetic); »1 = looser (real-model slack).")


if __name__ == "__main__":
    main()
