"""The fieldrun→pil J-lens seam: consuming ``fieldrun --jlens-export`` (PR #124) to J-correct DLA incidences.

Where the plain DLA/logit-lens incidence ``⟨d_b, U_v⟩`` assumes block b reaches the output unchanged, the
J-lens routes it through the layer's averaged causal Jacobian ``J[l]`` first, scoring b's TOTAL effect. These
tests are hermetic (synthetic J + labels, no fieldrun binary): they pin the block→layer map fixed by the
export's ``capture_point``, the ``fitted`` gate, the λ-shrinkage baseline (λ=0 ≡ the plain logit-lens), the
final-norm-fold ``γ``-conjugation, and the ``.npz``/``.meta.json`` loader contract. Tag: ``empirical``.
"""

import json

import numpy as np
import pytest
import torch

from pil.fieldrun_io import jcorrect_sources, load_jlens
from pil.geometry import incidences

# SourceBundle.blocks layout for the fixtures below (layer 0 fitted, layer 1 unfit)
BLOCKS = ["embed", "L0.attn", "L0.mlp", "L1.attn", "L1.mlp"]


def _synthetic(dim=4, nb=5, n=3, n_layer=2, seed=0):
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((n, nb, dim)).astype(np.float32)
    J = rng.standard_normal((n_layer, dim, dim)).astype(np.float32)
    J[n_layer - 1] = np.eye(dim, dtype=np.float32)                 # last layer is identity by construction
    fitted = np.array([True, False], dtype=bool)                    # layer 0 fit, layer 1 unfit → identity
    return D, J, fitted


def test_lambda_zero_is_the_logit_lens_baseline():
    """λ=0 reproduces the plain logit-lens incidences EXACTLY — the baseline any correction must beat."""
    D, J, fitted = _synthetic()
    out = jcorrect_sources(D, BLOCKS, J, fitted, lam=0.0)
    assert np.array_equal(out, D.astype(np.float32))               # identity, bit-for-bit at λ=0
    # and it composes with geometry.incidences unchanged
    U = torch.randn(6, D.shape[-1])
    base = incidences(torch.from_numpy(D), U)
    corr = incidences(torch.from_numpy(out), U)
    assert torch.allclose(base, corr, atol=1e-6)


def test_block_to_layer_mapping_and_fitted_gate():
    """L{l}.mlp/attn route through J[l] (per capture_point); embed and UNFIT layers stay identity."""
    D, J, fitted = _synthetic()
    out = jcorrect_sources(D, BLOCKS, J, fitted, lam=1.0)
    idx = {b: i for i, b in enumerate(BLOCKS)}
    # embed → identity (no captured J before layer 0)
    assert np.array_equal(out[:, idx["embed"], :], D[:, idx["embed"], :])
    # layer 0 is fitted → attn & mlp both routed through J[0]  (d @ J.T, the fieldrun convention)
    for blk in ("L0.attn", "L0.mlp"):
        assert np.allclose(out[:, idx[blk], :], D[:, idx[blk], :] @ J[0].T, atol=1e-5)
        assert not np.allclose(out[:, idx[blk], :], D[:, idx[blk], :])   # genuinely changed
    # layer 1 is NOT fitted → identity (graceful degradation to the logit-lens), never scrambled
    for blk in ("L1.attn", "L1.mlp"):
        assert np.array_equal(out[:, idx[blk], :], D[:, idx[blk], :])


def test_gamma_conjugation_exact_folded_basis():
    """With the final-norm gain γ, the operator is diag(γ) J diag(1/γ) (the exact folded-basis form)."""
    D, J, fitted = _synthetic()
    g = np.array([2.0, 0.5, 1.0, 4.0], dtype=np.float32)
    out = jcorrect_sources(D, BLOCKS, J, fitted, lam=1.0, gamma=g)
    idx = BLOCKS.index("L0.mlp")
    jl_eff = (g[:, None] * J[0]) * (1.0 / g)[None, :]
    assert np.allclose(out[:, idx, :], D[:, idx, :] @ jl_eff.T, atol=1e-5)
    # γ = ones is a no-op vs the direct application
    out_ones = jcorrect_sources(D, BLOCKS, J, fitted, lam=1.0, gamma=np.ones(4, np.float32))
    assert np.allclose(out_ones, jcorrect_sources(D, BLOCKS, J, fitted, lam=1.0), atol=1e-6)


def test_shape_guards():
    """A wrong-shape J / dim / blocks mismatch fails loudly (a contract mismatch, not a silent bug)."""
    D, J, fitted = _synthetic()
    with pytest.raises(ValueError, match="n_layer, dim, dim"):
        jcorrect_sources(D, BLOCKS, J[0], fitted)                  # 2-D J
    with pytest.raises(ValueError, match="D dim"):
        jcorrect_sources(D[:, :, :3], BLOCKS, J, fitted)           # dim mismatch
    with pytest.raises(ValueError, match="blocks"):
        jcorrect_sources(D, BLOCKS[:-1], J, fitted)                # nb mismatch


def test_load_jlens_roundtrips_npz_and_meta(tmp_path):
    """load_jlens reads the --jlens-export .npz (J float32, fitted→bool) + the .meta.json sidecar."""
    D, J, fitted = _synthetic()
    npz = tmp_path / "model.npz"
    np.savez(npz, J=J, fitted=fitted.astype(np.int32))             # fieldrun writes fitted as int32
    (tmp_path / "model.meta.json").write_text(json.dumps({
        "capture_point": "h_l = the POST-block residual of layer l (after the attn+MLP add, PRE final-norm)",
        "apply": "J[l] @ r (numpy: r @ J[l].T)", "n_layer": 2, "d": 4, "fitted_layers": [0],
    }))
    Jl, fit, meta = load_jlens(str(npz))
    assert Jl.shape == (2, 4, 4) and Jl.dtype == np.float32
    assert fit.dtype == bool and fit.tolist() == [True, False]
    assert "POST-block residual" in meta["capture_point"]
    # and it drives the correction end-to-end
    out = jcorrect_sources(D, BLOCKS, Jl, fit, lam=0.5)
    assert out.shape == D.shape

    bad = tmp_path / "bad.npz"
    np.savez(bad, foo=J)
    with pytest.raises(KeyError, match="missing 'J'/'fitted'"):
        load_jlens(str(bad))
