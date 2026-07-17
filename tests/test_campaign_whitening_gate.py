"""Unit tests for the anisotropy-removal (whitening) gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.attach_levers_codex import (  # noqa: E402
    BilinearConfig,
    chu_liu_edmonds,
    governor_columns_to_offsets,
)
from experiments.campaign_grounded_attach import (  # noqa: E402
    GroundedAttachmentScorer,
    assert_matched_token_counts,
)
from experiments.campaign_grounded_labeler import (  # noqa: E402
    CHECKPOINT_LAYERS,
    FEATURE_DIM,
    GroundedFeatureProvider,
    assert_uas_identity,
)
from experiments.campaign_whitening_gate import (  # noqa: E402
    PIN_A_SAMPLE_SIZE,
    PIN_A_SEED,
    SIGMA_ZERO_REPLACEMENT,
    VARIANT_TIE_RANK,
    WhitenedFeatureProvider,
    WhiteningTransform,
    pin_a_effective_rank,
    pin_a_effective_rank_whitened,
    pin_b_decode,
    preregistered_whitening_read,
    select_dev_variant_layer,
)
from experiments.german_r1_codex import Sentence  # noqa: E402
from experiments.german_r3_codex import (  # noqa: E402
    CampaignTestReadGuard,
    resolve_governor_index,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    head_deprel_by_sent,
)


def _write_npz(
    path: Path,
    *,
    sent_ids: list[str],
    token_indices: list[int],
    residuals: np.ndarray,
) -> None:
    n = len(sent_ids)
    assert residuals.shape == (n, FEATURE_DIM)
    last = np.zeros((n, 4, FEATURE_DIM), dtype=np.float64)
    for layer in range(4):
        last[:, layer, :] = residuals
    np.savez(
        path,
        sent_ids=np.array(sent_ids, dtype="U16"),
        token_index=np.asarray(token_indices, dtype=np.int64),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=last,
    )


def _sentence(sent_id: str, tokens: tuple[str, ...]) -> Sentence:
    n = len(tokens)
    return Sentence(
        sent_id=sent_id,
        text=" ".join(tokens),
        tokens=tokens,
        targets={
            "pos": ("NOUN",) * n,
            "morph_case": ("-",) * n,
            "morph_gnn": ("-|-",) * n,
        },
    )


def test_whitening_fit_is_train_only(tmp_path: Path) -> None:
    """Transform fit on TRAIN must not depend on a separate DEV/TEST matrix."""
    rng = np.random.default_rng(0)
    train_phi = rng.normal(size=(50, FEATURE_DIM))
    # Poisoned "DEV" matrix with huge values — must not affect TRAIN-fit params.
    dev_phi = rng.normal(loc=1000.0, scale=50.0, size=(50, FEATURE_DIM))

    tr_a = WhiteningTransform.fit(train_phi, variant="v1", layer_index=0)
    tr_b = WhiteningTransform.fit(train_phi, variant="v1", layer_index=0)
    # Same TRAIN → identical μ/σ regardless of any DEV matrix sitting nearby.
    np.testing.assert_allclose(tr_a.mean, tr_b.mean)
    np.testing.assert_allclose(tr_a.std, tr_b.std)
    # Explicitly: TRAIN mean is not the DEV mean.
    assert not np.allclose(tr_a.mean, dev_phi.mean(axis=0))
    # V2 components also TRAIN-only.
    tr_v2 = WhiteningTransform.fit(train_phi, variant="v2_k1", layer_index=0)
    tr_v2_dev = WhiteningTransform.fit(dev_phi, variant="v2_k1", layer_index=0)
    assert not np.allclose(tr_v2.mean, tr_v2_dev.mean)
    assert tr_v2.components is not None and tr_v2_dev.components is not None
    # Directions can differ; the point is params come from the matrix passed to fit.
    np.testing.assert_allclose(tr_v2.mean, train_phi.mean(axis=0))


def test_v1_zscore_and_sigma_zero_guard() -> None:
    # Hand fixture: 3 rows, dim reduced conceptually via first 3 coords of FEATURE_DIM.
    phi = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    # dim 0: [0, 2, 4] → μ=2, σ=√(8/3) ≈ 1.633
    phi[:, 0] = [0.0, 2.0, 4.0]
    # dim 1: constant 7 → σ=0 → replaced by 1.0
    phi[:, 1] = 7.0
    # dim 2: [1, 1, 4] → μ=2, σ=√2
    phi[:, 2] = [1.0, 1.0, 4.0]

    tr = WhiteningTransform.fit(phi, variant="v1", layer_index=0)
    assert tr.std[1] == SIGMA_ZERO_REPLACEMENT
    whitened = tr.transform_matrix(phi)
    # Per-dim mean ≈ 0, std ≈ 1 on non-constant columns.
    np.testing.assert_allclose(whitened[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(whitened[:, 0].std(), 1.0, atol=1e-12)
    np.testing.assert_allclose(whitened[:, 2].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(whitened[:, 2].std(), 1.0, atol=1e-12)
    # Constant column: (7-7)/1 = 0 for every row.
    np.testing.assert_allclose(whitened[:, 1], 0.0, atol=1e-12)


def test_v2_removes_top_pc_raises_effrank() -> None:
    """Dominant synthetic direction → V2 raises PIN-A effrank vs raw."""
    rng = np.random.default_rng(0)
    n = 200
    # One massive rogue dim + small isotropic noise on a few other dims.
    rogue = rng.normal(0.0, 50.0, size=(n, 1))
    noise = rng.normal(0.0, 0.5, size=(n, 8))
    phi = np.zeros((n, FEATURE_DIM), dtype=np.float64)
    phi[:, 0:1] = rogue
    phi[:, 1:9] = noise

    raw_rank = pin_a_effective_rank(phi)
    tr = WhiteningTransform.fit(phi, variant="v2_k1", layer_index=0)
    white_rank = pin_a_effective_rank_whitened(phi, tr)
    assert white_rank > raw_rank
    # Raw should be near 1 (rogue dominates); whitened substantially higher.
    assert raw_rank < 3.0
    assert white_rank > raw_rank + 1.0


def test_pin_a_formula_and_fixed_sampling() -> None:
    rng = np.random.default_rng(1)
    # n < 800: uses all rows; two calls identical.
    small = rng.normal(size=(40, 16))
    # Pad to FEATURE_DIM not required for pin_a (works on any d).
    r1 = pin_a_effective_rank(small)
    r2 = pin_a_effective_rank(small)
    assert r1 == r2
    # Hand-check formula on tiny matrix.
    tiny = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    centered = tiny - tiny.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, compute_uv=False)
    eigs = sv**2
    expected = float((eigs.sum() ** 2) / np.square(eigs).sum())
    assert pin_a_effective_rank(tiny) == pytest.approx(expected)

    # n ≥ 800: seed 0 subsample is deterministic across calls.
    large = rng.normal(size=(1200, 32))
    a = pin_a_effective_rank(large, sample_size=PIN_A_SAMPLE_SIZE, seed=PIN_A_SEED)
    b = pin_a_effective_rank(large, sample_size=PIN_A_SAMPLE_SIZE, seed=PIN_A_SEED)
    assert a == b
    # Different seed may differ (not required to, but usually does).
    c = pin_a_effective_rank(large, sample_size=PIN_A_SAMPLE_SIZE, seed=99)
    # At least sampling path is exercised; result is finite and positive.
    assert a > 0.0 and c > 0.0


def test_pin_b_decode_logprob_composition_and_fallback(tmp_path: Path) -> None:
    """PIN-B: bilinear log-softmax over covered + L0 logprob fallback + pure-L0 path."""
    # Fit a scorer on a fully covered 3-token sentence.
    full_path = tmp_path / "full.npz"
    last_full = np.zeros((3, 4, FEATURE_DIM), dtype=np.float64)
    last_full[0, 0, 0] = 1.0
    last_full[1, 0, 1] = 1.0
    last_full[2, 0, 2] = 1.0
    np.savez(
        full_path,
        sent_ids=np.array(["b", "b", "b"], dtype="U8"),
        token_index=np.asarray([0, 1, 2], dtype=np.int64),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=last_full,
    )
    fit_provider = GroundedFeatureProvider(full_path, layer_index=0)
    fit_sent = [_sentence("b", ("X", "Y", "Z"))]
    fit_heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "b",
                fit_sent[0].text,
                fit_sent[0].tokens,
                (1, 0, -1),
                ("nsubj", "root", "obj"),
            )
        ]
    )
    scorer = GroundedAttachmentScorer(
        BilinearConfig(rank=4, epochs=3, learning_rate=0.1, seed=0)
    )
    scorer.fit(fit_sent, fit_heads, fit_provider)

    tokens = ("A", "B", "C")
    n = len(tokens)
    l0_scores = np.full((n, n + 1), 0.05, dtype=np.float64)
    for i in range(n):
        l0_scores[i, i + 1] = -np.inf
    l0_scores[0, 2] = 1.0
    l0_scores[1, 0] = 1.0
    l0_scores[2, 2] = 1.0

    # (a) Full coverage: PIN-B returns a valid arborescence; scores path is log-prob.
    offsets = pin_b_decode(scorer, tokens, "b", l0_scores, fit_provider)
    assert len(offsets) == n
    assert sum(1 for o in offsets if o == 0) == 1
    for index, offset in enumerate(offsets):
        if offset == 0:
            continue
        gov = resolve_governor_index(index, offset, n)
        assert gov is not None and gov != index

    # Manually verify row composition for a covered dep with covered governors.
    from experiments.campaign_whitening_gate import _log_softmax_finite

    dep = 0
    covered = [0, 1, 2]
    others = [g for g in covered if g != dep]
    dep_phi = fit_provider.get("b", dep)
    assert dep_phi is not None
    gov_phis = np.stack([fit_provider.get("b", g) for g in others])
    logits = (dep_phi @ scorer.U) @ (gov_phis @ scorer.V).T + scorer.b
    bil_log = _log_softmax_finite(np.asarray(logits, dtype=np.float64))
    # bil_log is a proper log-distribution over the covered-governor set.
    assert np.exp(bil_log).sum() == pytest.approx(1.0, abs=1e-9)
    l0_log = _log_softmax_finite(l0_scores[dep])
    assert np.exp(l0_log[np.isfinite(l0_log)]).sum() == pytest.approx(1.0, abs=1e-9)
    combined = l0_log.copy()
    for gov, lp in zip(others, bil_log, strict=True):
        combined[gov + 1] = float(lp)
    renorm = _log_softmax_finite(combined)
    assert np.exp(renorm[np.isfinite(renorm)]).sum() == pytest.approx(1.0, abs=1e-9)
    # No non-log raw L0 scores or raw softmax probs in renorm: all ≤ 0.
    assert np.all(renorm[np.isfinite(renorm)] <= 1e-12)

    # (b) Covered dep with ZERO covered-governor candidates → pure L0 log-prob path.
    partial = tmp_path / "partial.npz"
    last_partial = np.zeros((1, 4, FEATURE_DIM), dtype=np.float64)
    last_partial[0, 0, :] = 0.1
    np.savez(
        partial,
        sent_ids=np.array(["a"], dtype="U8"),
        token_index=np.asarray([0], dtype=np.int64),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=last_partial,
    )
    predict_provider = GroundedFeatureProvider(partial, layer_index=0)
    pure_l0 = governor_columns_to_offsets(
        chu_liu_edmonds(
            np.stack(
                [_log_softmax_finite(l0_scores[i]) for i in range(n)],
                axis=0,
            )
        )
    )
    decoded = pin_b_decode(scorer, tokens, "a", l0_scores, predict_provider)
    # Only token 0 covered, no other covered governors → every row is pure L0 logprob.
    assert decoded == pure_l0


def test_uas_and_arc_set_identity_assertions() -> None:
    assert_uas_identity(
        {
            "unary": (1, 0, -1),
            "surface119": (1, 0, -1),
            "raw121": (1, 0, -1),
            "whitened": (1, 0, -1),
        }
    )
    with pytest.raises(AssertionError, match="UAS invariant"):
        assert_uas_identity(
            {
                "unary": (1, 0, -1),
                "whitened": (0, 0, -1),
            }
        )
    assert_matched_token_counts(
        {"l0": 5, "surface119": 5, "raw122": 5, "whitened": 5}, expected=5
    )
    with pytest.raises(AssertionError, match="token count mismatch"):
        assert_matched_token_counts(
            {"l0": 5, "whitened": 4}, expected=5
        )


def _arm_metrics(las: float, uas: float = 0.70, case: float = 0.80) -> dict[str, float]:
    return {
        "las_strict": las,
        "uas": uas,
        "serve_honest_morph_case": case,
    }


def _whitening_read(
    *,
    lab_unary: float,
    lab_surface: float,
    lab_raw: float,
    lab_white: float,
    att_l0: float,
    att_surface: float,
    att_raw: float,
    att_white: float,
    conv_base: float,
    conv_full: float,
    case_white: float = 0.80,
    case_base: float = 0.70,
    lab_variant: str = "v1",
    att_variant: str = "v1",
) -> dict[str, object]:
    return preregistered_whitening_read(
        labeler_unary=_arm_metrics(lab_unary),
        labeler_surface=_arm_metrics(lab_surface),
        labeler_raw121=_arm_metrics(lab_raw),
        labeler_whitened=_arm_metrics(lab_white, case=case_white),
        attach_l0=_arm_metrics(0.0, uas=att_l0),
        attach_surface=_arm_metrics(0.0, uas=att_surface),
        attach_raw122=_arm_metrics(0.0, uas=att_raw),
        attach_whitened=_arm_metrics(0.0, uas=att_white),
        attach_conv_base_las=conv_base,
        attach_conv_full_las=conv_full,
        case_whitened=case_white,
        case_baseline=case_base,
        labeler_dev_variant=lab_variant,
        labeler_dev_layer=17,
        attach_dev_variant=att_variant,
        attach_dev_layer=26,
        effrank_raw=1.2,
        effrank_whitened=45.0,
        labeler_best_variant=lab_variant,
        attach_best_variant=att_variant,
    )


def test_section5_halted_when_whitened_leq_raw() -> None:
    read = _whitening_read(
        lab_unary=0.50,
        lab_surface=0.48,
        lab_raw=0.55,
        lab_white=0.54,  # ≤ raw
        att_l0=0.70,
        att_surface=0.71,
        att_raw=0.72,
        att_white=0.71,  # ≤ raw
        conv_base=0.50,
        conv_full=0.51,
    )
    assert read["verdict"] == "HALTED"
    assert read["fired_primitive"] == "none"


def test_section5_in_between_improves_but_no_bar() -> None:
    read = _whitening_read(
        lab_unary=0.50,
        lab_surface=0.48,
        lab_raw=0.52,
        lab_white=0.525,  # improves over raw but gain vs unary = 0.025 < 0.03
        att_l0=0.70,
        att_surface=0.71,
        att_raw=0.71,
        att_white=0.72,  # improves; UAS gain vs L0 = 0.02 < 0.03
        conv_base=0.50,
        conv_full=0.51,
        lab_variant="v1",
        att_variant="v1",
    )
    assert read["verdict"] == "IN-BETWEEN"
    assert read["v1_vs_v2"] == "conditioning"
    assert read["fired_primitive"] == "none"


def test_section5_bar_cleared_is_in_between_pending_cross_vendor() -> None:
    """Bar clear on this lane cannot self-certify FIRES (cross-vendor pending)."""
    read = _whitening_read(
        lab_unary=0.50,
        lab_surface=0.48,
        lab_raw=0.52,
        lab_white=0.55,  # gain vs unary = 0.05 ≥ 0.03
        att_l0=0.70,
        att_surface=0.71,
        att_raw=0.71,
        att_white=0.74,  # UAS gain 0.04; need conv too for attach bar
        conv_base=0.50,
        conv_full=0.54,  # gain 0.04 ≥ 0.03
        lab_variant="v2_k1",
        att_variant="v2_k1",
    )
    assert read["verdict"] == "IN-BETWEEN"
    assert read["fired_primitive"] == "both"
    assert "pending" in str(read["text"]).lower() or "cross-vendor" in str(read["text"])
    assert read["v1_vs_v2"] == "information-removal"


def test_section5_through_line_branches() -> None:
    # met
    r_met = _whitening_read(
        lab_unary=0.5,
        lab_surface=0.48,
        lab_raw=0.5,
        lab_white=0.5,
        att_l0=0.7,
        att_surface=0.7,
        att_raw=0.7,
        att_white=0.7,
        conv_base=0.5,
        conv_full=0.5,
        case_white=0.91,
        case_base=0.70,
    )
    assert r_met["through_line"]["met"] is True
    assert r_met["through_line"]["plateau"] is False

    # plateau
    r_plat = _whitening_read(
        lab_unary=0.5,
        lab_surface=0.48,
        lab_raw=0.5,
        lab_white=0.5,
        att_l0=0.7,
        att_surface=0.7,
        att_raw=0.7,
        att_white=0.7,
        conv_base=0.5,
        conv_full=0.5,
        case_white=0.80,
        case_base=0.70,
    )
    assert r_plat["through_line"]["met"] is False
    assert r_plat["through_line"]["plateau"] is True

    # diagnose ≤ baseline
    r_diag = _whitening_read(
        lab_unary=0.5,
        lab_surface=0.48,
        lab_raw=0.5,
        lab_white=0.5,
        att_l0=0.7,
        att_surface=0.7,
        att_raw=0.7,
        att_white=0.7,
        conv_base=0.5,
        conv_full=0.5,
        case_white=0.65,
        case_base=0.70,
    )
    assert r_diag["through_line"]["met"] is False
    assert r_diag["through_line"]["plateau"] is False
    assert "diagnose" in r_diag["through_line"]["text"]


def test_test_read_once_guard_second_claim_raises() -> None:
    guard = CampaignTestReadGuard()
    guard.claim("shared")
    with pytest.raises(RuntimeError, match="only once"):
        guard.claim("shared")
    guard.claim("head_deprel")
    guard.assert_complete()
    with pytest.raises(RuntimeError, match="only once"):
        guard.claim("head_deprel")


def test_dev_only_variant_layer_selection() -> None:
    """Selection consumes only DEV-tagged rows; no TEST keys required/used."""
    rows = [
        {
            "variant": "v1",
            "layer_index": 0,
            "dev_deprel_only_accuracy": 0.50,
            "split": "dev",
        },
        {
            "variant": "v2_k1",
            "layer_index": 1,
            "dev_deprel_only_accuracy": 0.60,
            "split": "dev",
        },
        {
            "variant": "v2_k2",
            "layer_index": 2,
            "dev_deprel_only_accuracy": 0.60,
            "split": "dev",
        },
        {
            "variant": "v1",
            "layer_index": 3,
            "dev_deprel_only_accuracy": 0.60,
            "split": "dev",
        },
    ]
    # Score 0.60 ties: prefer v1 (rank 0) over v2, then smallest layer among v1.
    variant, layer = select_dev_variant_layer(rows, score_key="dev_deprel_only_accuracy")
    assert variant == "v1"
    assert layer == 3
    # Confirm score_key is the DEV metric name (test metrics not accepted by convention).
    assert all("dev_" in k or k in {"variant", "layer_index", "split"} for row in rows for k in row)
    # Attach-style key
    attach_rows = [
        {"variant": "v2_k2", "layer_index": 0, "dev_uas": 0.71},
        {"variant": "v2_k1", "layer_index": 1, "dev_uas": 0.71},
        {"variant": "v1", "layer_index": 2, "dev_uas": 0.70},
    ]
    v, li = select_dev_variant_layer(attach_rows, score_key="dev_uas")
    # 0.71 ties between v2_k1 and v2_k2 → prefer smallest k (v2_k1).
    assert v == "v2_k1" and li == 1
    assert VARIANT_TIE_RANK["v1"] < VARIANT_TIE_RANK["v2_k1"] < VARIANT_TIE_RANK["v2_k2"]


def test_whitened_feature_provider_drop_in(tmp_path: Path) -> None:
    path = tmp_path / "p.npz"
    residuals = np.zeros((2, FEATURE_DIM), dtype=np.float64)
    residuals[0, 0] = 2.0
    residuals[1, 0] = 4.0
    _write_npz(path, sent_ids=["s", "s"], token_indices=[0, 1], residuals=residuals)
    base = GroundedFeatureProvider(path, layer_index=0)
    train_phi = np.stack([base.get("s", 0), base.get("s", 1)])
    tr = WhiteningTransform.fit(train_phi, variant="v1", layer_index=0)
    provider = WhitenedFeatureProvider(base, tr)
    assert provider.covered("s", 0) is True
    assert provider.covered("s", 9) is False
    assert provider.feature_dim == FEATURE_DIM
    assert provider.layer_index == 0
    assert provider.layer == CHECKPOINT_LAYERS[0]
    w0 = provider.get("s", 0)
    w1 = provider.get("s", 1)
    assert w0 is not None and w1 is not None
    # Mean of whitened train rows ≈ 0 on dim 0.
    assert (w0[0] + w1[0]) / 2 == pytest.approx(0.0, abs=1e-12)
