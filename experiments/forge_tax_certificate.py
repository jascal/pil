"""Forge-tax certificate (LP): is a decode position single-source-realizable under SOME decoder frame?

Reads a `fieldrun --source-dump` (the raw per-block contribution vectors d̃_b, with r_x = Σ_b d̃_b). For a
position x0 it asks the certificate LP of pic/spec/forge_tax_certificate.md §3:

    ∃ a decoder frame U' that  (a) faithfully reproduces EVERY decode in the held-out set
                               (b) lets a SINGLE block j alone decode x0 ?

Free frame U', fixed measured d̃_b ⟹ faithfulness + single-source are LINEAR in U' ⟹ an LP. We project U'
onto span{r_x, d̃_j(x0)} (dim ≤ N+nb ≪ d), losslessly. If the LP is feasible for some block j, x0 is
REDUCIBLE; if infeasible for ALL blocks, x0 is 1-IRREDUCIBLE (certified computed).

FINDING (see results/forge_tax_certificate.txt): empirically this free-frame version is VACUOUS — every
position is reducible via its dominant block (‖d̃_dom‖≈‖r_x‖), so 0/40 are 1-irreducible. The harness is
correct (it returns infeasible for the small blocks, and the faithfulness-only LP is feasible); the free
frame is just too permissive. A non-trivial certificate needs the shared-write-direction (fix U, free A)
structure at neuron granularity — future work. Kept as the reusable seam; the objective is what to
strengthen.

Run:  python experiments/forge_tax_certificate.py dump.source.jsonl [--x0 IDX] [--sweep --limit L] [--gamma G]
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from pil.fieldrun_io import load_source_dump


def basis_of(vectors: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Orthonormal basis (dim x q) of the row span of `vectors` (m x dim), via SVD."""
    _, s, vt = np.linalg.svd(vectors, full_matrices=False)
    q = int((s > tol * s[0]).sum()) if s.size and s[0] > 0 else 0
    return vt[:q].T  # (dim, q)


def _add_margin(rows, cols, data, rhs, ghat, ti, vi, q, gamma):
    """Append one constraint  ĝ·(û_t − û_v) ≥ γ  ⟺  −ĝ·û_t + ĝ·û_v ≤ −γ  (sparse-LP triples)."""
    ci = len(rhs)
    for c in range(q):
        rows.append(ci)
        cols.append(ti * q + c)
        data.append(-ghat[c])
        rows.append(ci)
        cols.append(vi * q + c)
        data.append(ghat[c])
    rhs.append(-gamma)


def faithfulness_rows(b, B, tok_index, gamma):
    """The shared faithfulness constraints (every position decodes its token over its competitors)."""
    q = B.shape[1]
    rows, cols, data, rhs = [], [], [], []
    rhat = b.r @ B  # (N, q)
    for p in range(b.N):
        ti = tok_index[int(b.cands[p, 0])]
        for vtok in b.cands[p, 1:]:
            _add_margin(rows, cols, data, rhs, rhat[p], ti, tok_index[int(vtok)], q, gamma)
    return rows, cols, data, rhs


def feasible(base, b, B, tok_index, x0, j, gamma):
    """Append "block j alone decodes x0" to the shared faithfulness base; test LP feasibility."""
    q = B.shape[1]
    nvar = len(tok_index) * q
    rows, cols, data, rhs = (list(base[0]), list(base[1]), list(base[2]), list(base[3]))
    ghat = B.T @ b.D[x0, j]
    ti = tok_index[int(b.cands[x0, 0])]
    for vtok in b.cands[x0, 1:]:
        _add_margin(rows, cols, data, rhs, ghat, ti, tok_index[int(vtok)], q, gamma)
    a_ub = csr_matrix((data, (rows, cols)), shape=(len(rhs), nvar))
    res = linprog(np.zeros(nvar), A_ub=a_ub, b_ub=np.asarray(rhs), bounds=(None, None), method="highs")
    return res.status == 0  # 0 = feasible, 2 = infeasible


def certify(b, x0, gamma, verbose=False):
    """Test all blocks for position x0. Returns (irreducible: bool, reducible_blocks: list[int])."""
    toks = sorted({int(t) for t in b.cands.reshape(-1)})
    tok_index = {t: i for i, t in enumerate(toks)}
    # basis must span the faithfulness residuals AND x0's block vectors (for an exact single-source test)
    B = basis_of(np.concatenate([b.r, b.D[x0]], axis=0))
    base = faithfulness_rows(b, B, tok_index, gamma)
    # dominant block (≈ r_x) is the likeliest single-source witness → order by norm desc, early-exit
    order = np.argsort(-np.linalg.norm(b.D[x0], axis=1))
    reducible = []
    for j in order:
        if feasible(base, b, B, tok_index, x0, int(j), gamma):
            reducible.append(int(j))
            if verbose:
                print(f"    reducible via {b.blocks[j]} (||d̃||={np.linalg.norm(b.D[x0, j]):.2f})")
            else:
                break
    return (len(reducible) == 0), reducible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--x0", type=int, default=None, help="certify a single position (detailed)")
    ap.add_argument("--sweep", action="store_true", help="classify every position up to --limit")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()

    b = load_source_dump(args.dump)
    print(f"loaded {args.dump}: N={b.N} positions · nb={b.nb} blocks · dim={b.dim}")

    if args.x0 is not None:
        x0 = args.x0
        print(f"\n=== position {x0}  (decoded {int(b.cands[x0, 0])}, margin {b.margin[x0]:.3f}) ===")
        irr, red = certify(b, x0, args.gamma, verbose=True)
        if irr:
            print("  >>> 1-IRREDUCIBLE — certified computed (no frame makes any single block decode it)")
        else:
            print(f"  reducible: {len(red)} block(s) can be made single-source under some frame")
        return

    n = min(args.limit, b.N)
    print(f"\n=== sweep: certifying {n} positions (1-irreducible = no frame makes it single-source) ===")
    irr_positions = []
    for x0 in range(n):
        irr, red = certify(b, x0, args.gamma)
        tag = "IRREDUCIBLE" if irr else f"reducible(>={len(red)})"
        print(f"  pos {x0:3d}  tok {int(b.cands[x0, 0]):6d}  margin {b.margin[x0]:6.3f}  -> {tag}")
        if irr:
            irr_positions.append(x0)
    frac = len(irr_positions) / n if n else 0.0
    print(f"\n1-irreducible (certified computed): {len(irr_positions)}/{n} = {frac:.2%}")
    print(f"  positions: {irr_positions}")
    print("(representation-invariant forge-tax measure: tokens no faithful frame can make single-source)")


if __name__ == "__main__":
    main()
