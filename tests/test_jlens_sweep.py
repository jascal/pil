"""Smoke test for the J-lens correction λ-sweep harness (experiments/jlens_correction_sweep.py).

Hermetic: a synthetic SourceBundle + J + U built so the lam=0 read reconstructs the decode. Pins the harness
contract (lam=0 recon ~ 1.0, a row per lam, resolve a valid fraction-of-depth) without a real fieldrun
--source-dump / --jlens pair -- that end-to-end validation is the harness's actual, still-open job.
"""

import importlib.util
from pathlib import Path

import numpy as np

from pil.fieldrun_io import SourceBundle

_PATH = Path(__file__).resolve().parent.parent / "experiments" / "jlens_correction_sweep.py"
_SPEC = importlib.util.spec_from_file_location("jlens_correction_sweep", _PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
sweep = _MOD.sweep

BLOCKS = ["embed", "L0.attn", "L0.mlp", "L1.attn", "L1.mlp"]


def _synthetic(dim=4, vocab=10, k=4, n=5, n_layer=2, seed=1):
    """A SourceBundle + (J, fitted) + U with cands[:,0] == the decode, so λ=0 recon == 1.0 by construction."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((vocab, dim)).astype(np.float32)
    D = rng.standard_normal((n, len(BLOCKS), dim)).astype(np.float32)
    r = D.sum(axis=1)
    cands = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        ids = rng.choice(vocab, size=k, replace=False)
        order = np.argsort(-(r[i] @ U[ids].T))               # sort candidates by model logit → argmax first
        cands[i] = ids[order]
    sb = SourceBundle(D=D, r=r, cands=cands, target=cands[:, 0].copy(),
                      margin=np.ones(n, dtype=np.float32), blocks=BLOCKS)
    J = rng.standard_normal((n_layer, dim, dim)).astype(np.float32)
    J[n_layer - 1] = np.eye(dim, dtype=np.float32)
    fitted = np.array([True, False], dtype=bool)
    return sb, J, fitted, U


def test_sweep_lambda_zero_recon_and_shape():
    sb, J, fitted, U = _synthetic()
    rows = sweep(sb, J, fitted, U, lams=[0.0, 0.5, 1.0])
    assert [r["lam"] for r in rows] == [0.0, 0.5, 1.0]
    assert abs(rows[0]["recon"] - 1.0) < 1e-6                 # λ=0 = the model read → reconstructs the decode
    for r in rows:
        assert 0.0 <= r["resolve"] <= 1.0                     # a valid fraction-of-depth
        assert np.isfinite(r["margin"])


def test_sweep_gamma_is_noop_at_ones():
    sb, J, fitted, U = _synthetic()
    plain = sweep(sb, J, fitted, U, lams=[1.0])[0]
    ones = sweep(sb, J, fitted, U, lams=[1.0], gamma=np.ones(sb.dim, np.float32))[0]
    assert abs(plain["margin"] - ones["margin"]) < 1e-4      # γ=1 conjugation ≡ direct application
