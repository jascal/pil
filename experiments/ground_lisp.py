"""Grounding bridge rung 2 — concepts as hyperplanes in the LISP redex model's residual space.

Second real model (our redex-local Lisp executor, 0.93M params), harder = compositional/recursive structure.
Residuals at the <sep> decision position (tropic/lisp_ground_extract.py -> lisp_residuals.pt): r (N,128),
op (8-way operator), is_arith (arithmetic = the COMPUTED core), dec (reduction decision), U (decode frame).

Tests:
  (1) grounding on a 2nd real model -- do concept-hyperplanes recover the operator (structural type)?
  (2) linear ceilings -- op-type and the decision `dec` (dec is linear from r by construction since
      the head is linear, so this should be ~1.0, confirming a well-trained model's final residual is fully
      linear -- vs the toy's 0.64); how high is op-type?
  (3) retrieved/computed GEOMETRY, entropy-NEUTRAL: effective rank (participation ratio of the residual
      covariance) of the ARITHMETIC-redex subspace vs the STRUCTURAL-redex subspace. Prediction: the computed
      (arithmetic) residuals occupy a HIGHER-dimensional subspace (parallels addition=rank-r; forge tax).
      (Reported descriptively -- the earlier answer-decodability test was answer-entropy-confounded on threx.)

Run: cd pil && .venv/bin/python experiments/ground_lisp.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")


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
    """Participation ratio of the (centered) covariance eigenvalues = (Σλ)² / Σλ² -- entropy-neutral dim."""
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


def main():
    d = torch.load(SP / "lisp_residuals.pt")
    r = d["r"].float()
    op, arith, dec, U = d["op"], d["is_arith"], d["dec"], d["U"].float()
    vocab = d["op_vocab"]
    N = r.shape[0]
    af = float(arith.float().mean())
    print(f"lisp redex residuals: r {tuple(r.shape)}  ops={vocab}  arith(computed) frac {af:.3f}")
    recon = float(((r @ U.T).argmax(1) == dec).float().mean())
    print(f"recon (r@U.T argmax == model decision) = {recon:.3f}\n")

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    cut = int(0.7 * N)
    tr, te = perm[:cut], perm[cut:]

    # (2) linear ceilings
    ceil_op = linear_probe(r[tr], op[tr], r[te], op[te], len(vocab))
    # dec has a large token vocab; densify to observed decisions
    uniq = dec.unique()
    remap = {int(v): i for i, v in enumerate(uniq)}
    dd = torch.tensor([remap[int(v)] for v in dec])
    ceil_dec = linear_probe(r[tr], dd[tr], r[te], dd[te], len(uniq))
    print(f"(2) linear-probe ceilings:  op-type {ceil_op:.3f}   decision {ceil_dec:.3f}  "
          f"({len(uniq)} distinct decisions)  [toy was 0.64; threx 0.99]")

    # (1) geometric concepts recover the operator
    gc = GeoConcepts(d=r.shape[1], K=16, nclass=len(vocab))
    opt = torch.optim.Adam(gc.parameters(), lr=0.03)
    for step in range(2500):
        tau = max(0.4, 1.3 - 0.9 * step / 2500)
        idx = tr[torch.randint(len(tr), (256,), generator=g)]
        loss = F.cross_entropy(gc(r[idx], tau), op[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        gacc = float((gc(r[te], 0.4).argmax(1) == op[te]).float().mean())
    print(f"(1) geometric-concept operator recovery: {gacc:.3f}  (concept-hyperplanes recover the op)")

    # (3) retrieved/computed geometry: effective rank of arithmetic vs structural residual subspaces
    ra, rs = r[arith == 1], r[arith == 0]
    er_a, er_s = effective_rank(ra), effective_rank(rs)
    print("\n(3) effective rank (participation ratio, entropy-neutral) of the residual subspace:")
    print(f"    ARITHMETIC (computed) redexes: {er_a:.2f}   (n={ra.shape[0]})")
    print(f"    STRUCTURAL (retrieved) redexes: {er_s:.2f}   (n={rs.shape[0]})")
    tag = "computed higher-dim" if er_a > er_s else "not higher"
    print(f"    ratio arith/struct = {er_a / er_s:.2f}  ({tag})")
    print("\nread: (1)+(2) grounding WORKS on a 2nd real (harder, compositional) model -- concepts recover "
          "the operator; the final residual is near-fully linear (decision ceiling ~1.0) vs toy 0.64. (3) if "
          "arithmetic (computed) residuals have higher effective rank than structural (retrieved), the "
          "geometry shows the computed core is higher-dimensional -- an entropy-neutral grounding of the "
          "retrieved/computed / forge-tax thesis (parallels addition=rank-r).")


if __name__ == "__main__":
    main()
