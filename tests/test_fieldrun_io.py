"""The fieldrun --pil-dump seam: contract + faithfulness on a committed real-model fixture.

M0 of the picard program (PICARD_GOAL.md): PIL's generative step is seeded from the
*real* per-block DLA incidences a transformer produces, via ``fieldrun --pil-dump``.
The fixture ``data/pythia160m_sample.pil.jsonl`` was emitted by
``fieldrun --bundle bundles/pythia-160m --recursion-explain --pil-dump ... --strict``
(12 positions, top-8 candidates) so this test needs no fieldrun binary at pytest time.

Guards two things: (1) ``load_pil_dump`` parses the emitter's schema into a
well-formed IncidenceBundle; (2) the incidences are FAITHFUL — summing each position's
per-block contributions reconstructs the model's own decision (``Σ_b ⟨d_b, U_k⟩``
argmaxes to ``pred``). A silently non-faithful dump is exactly the failure the emitter's
``--strict`` recon check and this test exist to prevent.
"""

from pathlib import Path

import torch

from pil.fieldrun_io import load_pil_dump

FIXTURE = Path(__file__).parent / "data" / "pythia160m_sample.pil.jsonl"


def test_load_pil_dump_contract():
    b = load_pil_dump(FIXTURE)
    n, nb, k = b.contrib.shape
    assert n == 12 and nb > 0 and k == 8
    assert b.cands.shape == (n, k)
    assert b.tgt_idx.shape == (n,) and b.margins.shape == (n,) and b.ent.shape == (n,)
    assert (b.tgt_idx >= 0).all(), "every retained record has its target in the candidate set"
    assert b.n_blocks == nb
    # pred (cands[0], the model argmax) is a valid token id; margins non-negative (top1 - top2)
    assert (b.margins >= 0).all()
    assert torch.isfinite(b.ent).all() and (b.ent >= 0).all()


def test_incidences_reconstruct_the_decision():
    """Faithfulness: Σ_b contrib[b][k] = ⟨r, U_cand_k⟩, so its argmax must be the model's pred = cands[0]."""
    b = load_pil_dump(FIXTURE)
    blocksum = b.contrib.sum(dim=1)              # (N, K) bias-free per-candidate logit
    recon = blocksum.argmax(dim=1)
    bad = int((recon != 0).sum())
    assert bad == 0, f"incidences do not reconstruct the decision at {bad} / {len(recon)} positions"


def test_decision_subspace_quantity_extracts():
    """The picard-P1 signal: per-block margin contribution D_b = ⟨d_b, U_t - U_v*⟩.

    Its block-sum equals the target-vs-best-competitor logit gap, which is >= the
    negative margin and exactly +margin whenever the model's prediction IS the target.
    """
    b = load_pil_dump(FIXTURE)
    n = b.contrib.shape[0]
    rows = torch.arange(n)
    blocksum = b.contrib.sum(dim=1)
    # best competitor = highest-logit candidate that is not the target
    masked = blocksum.clone()
    masked[rows, b.tgt_idx] = float("-inf")
    comp = masked.argmax(dim=1)
    Db = b.contrib[rows, :, b.tgt_idx] - b.contrib[rows, :, comp]   # (N, nb)
    gap = Db.sum(dim=1)                                             # ⟨r, U_t - U_v*⟩
    assert Db.shape == (n, b.n_blocks)
    # where the model predicted the target (tgt_idx == 0), the gap is exactly the +margin
    pred_is_target = b.tgt_idx == 0
    if pred_is_target.any():
        assert torch.allclose(gap[pred_is_target], b.margins[pred_is_target], atol=1e-2)
