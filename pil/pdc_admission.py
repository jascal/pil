"""Affine free-frame forge-tax LP and the separate PDC margin admission guard.

Measured source writes are data. Only the common decoder frame and bias move.
Numerical primal/dual checks are evidence, not Isabelle proofs about the data.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack


def margin_admission(sources, frame, bias, target, coalition, candidates):
    """Check the same strict margin and centered-tail premises as PDC_Admission."""
    sources = np.asarray(sources, dtype=np.float64)
    frame, bias = np.asarray(frame, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    candidates = list(candidates)
    if target not in candidates or len(candidates) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("target and at least one distinct rival must be covered")
    if not coalition or len(set(coalition)) != len(coalition):
        raise ValueError("nonempty coalition with distinct source indices required")
    if any(j < 0 or j >= len(sources) for j in coalition):
        raise ValueError("source index outside measured table")
    if not all(np.isfinite(x).all() for x in [sources, frame, bias]):
        raise ValueError("nonfinite source/frame data")
    partial = sources[coalition].sum(axis=0)
    omitted = sources.sum(axis=0) - partial
    scores = frame[candidates] @ partial + bias[candidates]
    tail = frame[candidates] @ omitted
    ti = candidates.index(target)
    margin = float(scores[ti] - np.max(np.delete(scores, ti)))
    kappa = float((tail.max() + tail.min()) / 2)
    delta = float((tail.max() - tail.min()) / 2)
    gamma = (margin + 2 * delta) / 2
    admitted = margin > gamma > 2 * delta and delta >= 0
    full = frame[candidates] @ sources.sum(axis=0) + bias[candidates]
    return {"admitted": bool(admitted), "target": int(target), "coalition": list(coalition),
        "candidates": candidates, "margin": margin, "gamma": gamma, "delta": delta, "kappa": kappa,
        "coalition_scores": scores.tolist(), "tail_scores": tail.tolist(),
        "full_scores": full.tolist(), "substitution_error": 0.0,
        "full_strict_target": bool(np.all(full[ti] > np.delete(full, ti)))}


def affine_basis(residuals, writes):
    vectors = np.concatenate([residuals, writes], axis=0)
    augmented = np.column_stack([vectors, np.ones(len(vectors))])
    if augmented.shape[1] <= 5:
        return np.eye(augmented.shape[1])
    _, singular, vt = np.linalg.svd(augmented, full_matrices=False)
    basis = vt[singular > max(augmented.shape) * np.finfo(float).eps * singular[0]].T
    error = np.max(np.abs(augmented - (augmented @ basis) @ basis.T))
    if error > 1e-7:
        raise ValueError(f"basis projection loses measured vectors: {error}")
    return basis


def constraints(residuals, candidates, writes, x0, coalition, gamma, basis):
    """Return A,b for A theta <= b, including affine bias exactly once per row."""
    tokens = sorted(set(int(t) for row in candidates for t in row))
    index = {t: i for i, t in enumerate(tokens)}
    q = basis.shape[1]
    rows, cols, values, rhs = [], [], [], []

    def add(vector, target, rivals):
        projected = np.append(vector, 1.0) @ basis
        for rival in rivals:
            if rival == target:
                continue
            row = len(rhs)
            nonzero = np.flatnonzero(projected)
            for token, sign in [(target, -1), (rival, 1)]:
                rows.extend([row] * len(nonzero))
                cols.extend((index[int(token)] * q + nonzero).tolist())
                values.extend((sign * projected[nonzero]).tolist())
            rhs.append(-gamma)

    for residual, cands in zip(residuals, candidates, strict=True):
        add(residual, cands[0], cands[1:])
    add(writes[coalition].sum(axis=0), candidates[x0][0], candidates[x0][1:])
    return csr_matrix((values, (rows, cols)), shape=(len(rhs), len(tokens) * q)), np.array(rhs), tokens


def solve_affine(A, rhs, tolerance=1e-7):
    result = linprog(np.zeros(A.shape[1]), A_ub=A, b_ub=rhs, bounds=(None, None), method="highs")
    if result.status == 0:
        violation = float(np.max(A @ result.x - rhs))
        if not np.isfinite(result.x).all() or violation > tolerance:
            return {"status": "unknown", "reason": "primal failed independent check", "violation": violation}
        return {"status": "feasible", "coefficients": result.x, "primal_max_violation": violation}
    if result.status != 2:
        return {"status": "unknown", "solver_status": int(result.status), "reason": result.message}
    # Normalize sum(y)=1. For uniform negative RHS, any exact null nonnegative y
    # yields y^T b=-gamma<0: a checkable Farkas infeasibility witness.
    equalities = vstack([A.T, csr_matrix(np.ones((1, A.shape[0])))], format="csr")
    dual = linprog(np.zeros(A.shape[0]), A_eq=equalities,
        b_eq=np.r_[np.zeros(A.shape[1]), 1.0], bounds=(0, None), method="highs")
    if dual.status != 0:
        return {"status": "unknown", "reason": "infeasible status without checked dual"}
    y = dual.x
    error = float(np.max(np.abs(A.T @ y)))
    contradiction = float(rhs @ y)
    if y.min() < -tolerance or error > tolerance or contradiction >= -tolerance:
        return {"status": "unknown", "reason": "dual failed independent check"}
    witness = {"status": "infeasible", "dual": y.tolist(),
        "dual_max_residual": error, "dual_rhs": contradiction, "dual_min": float(y.min())}
    if A.shape[1] < 100 and np.all(A.data == np.round(A.data)):
        rational = [Fraction(float(v)).limit_denominator(1000000) for v in y]
        sums = [Fraction(0) for _ in range(A.shape[1])]
        for row in range(A.shape[0]):
            for col, value in zip(
                A.indices[A.indptr[row]:A.indptr[row+1]], A.data[A.indptr[row]:A.indptr[row+1]], strict=True
            ):
                sums[col] += rational[row] * Fraction(float(value))
        exact = all(v == 0 for v in sums) and all(v >= 0 for v in rational)
        exact = exact and sum(rational[i] * Fraction(float(rhs[i])) for i in range(len(rhs))) < 0
        witness.update({"dual_exact_rational_check": exact, "dual_rational": [str(v) for v in rational]})
    return witness
