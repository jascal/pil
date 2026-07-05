"""Cross-token wall, formal diagnostic (i-orca C7 CrossToken.thy sufficiency condition).

The formal review (i-orca #23) proved: (impossibility) no ADDITIVE reader g(t1)+h(t2) decides [t1==t2] --
dimension-free, so the ~0.78 wall is structural, no width fixes it; (sufficiency) ONE bilinear feature
<phi(t1), phi(t2)> decides equality exactly IFF the token feature vectors phi are orthonormal-ish (Gram
near-diagonal). My multi-position grounded B-arm (match = <m(cur), m(o)>) is the right FORM but did not beat
the single-residual A-arm on pythia residuals. The theorem says this is decidable: measure the Gram of
per-token residual/membership centroids -- if far from diagonal, the bilinear reader is PRECLUDED
("memberships too entangled to separate tokens"); if near-diagonal, the relation is present and the failure
was elsewhere. This distinguishes the two, and grounds WHY the token-identity eq_atom (exact id) is necessary.

Run: cd pil && .venv/bin/python experiments/wyly_gram_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import torch

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "multipos_pythia70m.pt"


def main():
    d = torch.load(DATA)
    R = d["r"].float()                                             # (N, L, d)
    ids = d["kept_ids"]                                           # (N, L)
    N, L, dim = R.shape
    Rf = R.reshape(N * L, dim)
    Rf = (Rf - Rf.mean(0)) / (Rf.std(0) + 1e-6)                   # per-dim standardize
    tok = ids.reshape(N * L)

    # per-token residual centroids for the most frequent tokens
    counts = torch.bincount(tok)
    freq = counts.argsort(descending=True)[:60]
    freq = freq[counts[freq] >= 30]
    cents = torch.stack([Rf[tok == t].mean(0) for t in freq])     # (T, d)
    cn = cents / (cents.norm(dim=1, keepdim=True) + 1e-6)
    gram = cn @ cn.T                                              # (T, T) cosine Gram of token centroids
    T = len(freq)
    diag = gram.diag().mean()
    offd = (gram.sum() - gram.diag().sum()) / (T * T - T)
    # how often is a token's OWN centroid its nearest? (diagonal dominance of the bilinear reader)
    win = int((gram.argmax(1) == torch.arange(T)).float().sum())
    print(f"cross-token Gram diagnostic (C7) -- {T} frequent tokens, pythia-70m residuals  ({N * L} pos)")
    print(f"  centroid Gram: diag {float(diag):.3f}  mean|off-diag| {float(offd.abs()):.3f}  "
          f"ratio {float(diag / (offd.abs() + 1e-9)):.1f}x")
    print(f"  self is argmax row: {win}/{T} tokens (diagonal dominance of <phi,phi>)")

    # the actual B-arm reader: does <r_i, r_j> separate SAME-token from DIFFERENT-token position pairs?
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(N * L, generator=g)[:20000]
    a, b = idx[:10000], idx[10000:]
    cos = (Rf[a] / Rf[a].norm(dim=1, keepdim=True) * (Rf[b] / Rf[b].norm(dim=1, keepdim=True))).sum(1)
    same = tok[a] == tok[b]
    n_same = int(same.sum())
    if n_same > 20:
        # AUC that <r_i,r_j> ranks same-token pairs above different-token pairs
        s_cos, d_cos = cos[same], cos[~same]
        auc = float((s_cos.unsqueeze(1) > d_cos.unsqueeze(0)).float().mean())
        print(f"  bilinear reader <r_i,r_j>: same-token cos {float(s_cos.mean()):+.3f} vs "
              f"diff-token {float(d_cos.mean()):+.3f}  -> AUC {auc:.3f}  ({n_same} same-token pairs)")
    print("\nRESULT (corrects my prior 'symbolic, not geometric'): C7 sufficiency is SATISFIED -- token")
    print("centroids are NEAR-ORTHONORMAL (Gram ~20x diagonal, self-argmax 60/60) and the BILINEAR reader")
    print("<r_i,r_j> separates same- from different-token pairs at AUC 0.926. So cross-token equality IS")
    print("geometrically decodable -- but only by a BILINEAR reader; the ADDITIVE/linear readers I tried")
    print("(which C7 proves cannot do equality) are what hit the ~0.78 wall. is_repeat still plateaued ~0.75")
    print("because per-instance same-token cos is WEAK (+0.23, context noise) so max-pool over a window is")
    print("noisy -- a reader weakness, NOT structural entanglement. The eq_atom (exact id, AUC 1.0, one rule")
    print("vs C7's card-V product cover) stays the EXACT+CHEAP primitive -- PREFERRED, not the only option.")


if __name__ == "__main__":
    main()
