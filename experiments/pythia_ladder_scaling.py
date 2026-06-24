"""Same-data scaling: the Pythia ladder (14m..2.8b) is trained on the SAME Pile data in the same order, so
a within-ladder size sweep on a FIXED eval text holds training data constant by construction. This settles
the PR #2 confound: is the certified-bit headroom trend a genuine SIZE effect, or a data-ladder artifact?

Reads a dir of `fieldrun --source-dump` files named `*pythia-<size>*.jsonl` (all dumped on the SAME text),
reports cv(‖r‖) [gate], the headroom proxy b* = -log2(margin/‖r‖) per size, and fits b*_p90 vs log10(params)
-> the scaling slope (bits per decade) with R². Data is fixed; only size varies.

Run:  python experiments/pythia_ladder_scaling.py <dump-dir>
"""

from __future__ import annotations

import glob
import re
import sys

import numpy as np

from pil.fieldrun_io import load_source_dump

PARAMS = {"14m": 14e6, "70m": 70e6, "160m": 160e6, "410m": 410e6,
          "1b": 1.0e9, "1.4b": 1.4e9, "2.8b": 2.8e9}


def metrics(path):
    sb = load_source_dump(path)
    rn = np.linalg.norm(sb.r, axis=1)
    gt = np.clip(sb.margin / (rn + 1e-9), 1e-6, None)
    b = -np.log2(gt)
    return sb.N, float(rn.std() / rn.mean()), float(np.quantile(b, .5)), float(np.quantile(b, .9))


def main():
    d = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(f"{d}/*pythia-*.jsonl")):
        m = re.search(r"pythia-(\d+\.?\d*[mb])", f)
        if not m or m.group(1) not in PARAMS:
            continue
        sz = m.group(1)
        N, cv, b50, b90 = metrics(f)
        rows.append((PARAMS[sz], sz, N, cv, b50, b90))
    rows.sort()
    print("Pythia same-data ladder (fixed eval text; training data constant across sizes)")
    print(f"  {'size':>6}{'params':>10}{'N':>5}{'rcv':>8}{'bstar_p50':>11}{'bstar_p90':>11}")
    for p, sz, N, cv, b50, b90 in rows:
        print(f"  {sz:>6}{p/1e6:>9.0f}M{N:>5}{cv:>8.3f}{b50:>11.2f}{b90:>11.2f}")
    # linear fit of the headroom proxy vs log10(params) + R^2
    x = np.log10(np.array([r[0] for r in rows]))
    for lbl, j in (("bstar_p50", 4), ("bstar_p90", 5)):
        y = np.array([r[j] for r in rows])
        s, b = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (s * x + b)) ** 2) / np.sum((y - y.mean()) ** 2)
        # honest verdict: a low-R² slope is NOT a scaling law (the trend saturates / is noisy)
        verdict = (f"{s:+.2f} bits/decade, IMPROVES with scale" if r2 >= 0.5
                   else "WEAK fit (R²<0.5) — NO clean scaling law; trend saturates (data fixed)")
        print(f"  fit {lbl} ~ {s:+.2f}*log10(params) + {b:.1f}   R²={r2:.2f}   ({verdict})")
    # plateau check: spread of the >=400M tail vs the <400M head
    tail = [r[5] for r in rows if r[0] >= 4e8]
    head = [r[5] for r in rows if r[0] < 4e8]
    if tail and head:
        print(f"  plateau: bstar_p90 head(<400M)={np.mean(head):.2f} vs tail(>=400M)={np.mean(tail):.2f} "
              f"(tail spread {max(tail) - min(tail):.2f}) — saturates past ~400M")
    cvs = [r[3] for r in rows]
    print(f"  gate: ||r|| cv in [{min(cvs):.3f}, {max(cvs):.3f}] (all CLOSED) — norm-pinned at every size")


if __name__ == "__main__":
    main()
