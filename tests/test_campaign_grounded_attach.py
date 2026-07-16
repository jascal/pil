"""Unit tests for the grounded-φ bilinear ATTACHMENT scorer gate."""

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
    _effective_rank,
    _mean_pairwise_cosine,
    assert_matched_token_counts,
    attach_doubly_covered_fraction,
    collect_covered_dependent_keys,
    covered_dependent_fraction,
    preregistered_attach_read,
    score_conversion_arm,
    score_uas_on_keys,
    select_dev_attach_layer,
)
from experiments.campaign_grounded_labeler import (  # noqa: E402
    CHECKPOINT_LAYERS,
    FEATURE_DIM,
    GroundedFeatureProvider,
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


def _write_fit_npz(path: Path) -> None:
    """Two sentences with ≥2 covered words and covered gold arcs for fit."""
    # s1: tokens 0,1,2 covered; s2: tokens 0,1 covered
    last_residual = np.zeros((5, 4, FEATURE_DIM), dtype=np.float16)
    for row, col in ((0, 0), (1, 1), (2, 2), (3, 0), (4, 1)):
        last_residual[row, 0, col] = float(row + 1)
        last_residual[row, 1, col] = float((row + 1) * 10)
    # make s1 features somewhat separable
    last_residual[0, 0, 0] = 1.0
    last_residual[1, 0, 1] = 1.0
    last_residual[2, 0, 2] = 1.0
    last_residual[3, 0, 0] = 0.5
    last_residual[4, 0, 1] = 0.5
    np.savez(
        path,
        sent_ids=np.array(["s1", "s1", "s1", "s2", "s2"], dtype="U8"),
        token_index=np.asarray([0, 1, 2, 0, 1], dtype=np.int64),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=last_residual,
    )


def _sentence(sent_id: str, tokens: tuple[str, ...], *, cases: tuple[str, ...] | None = None) -> Sentence:
    n = len(tokens)
    morph_case = cases if cases is not None else ("-",) * n
    return Sentence(
        sent_id=sent_id,
        text=" ".join(tokens),
        tokens=tokens,
        targets={
            "pos": ("NOUN",) * n,
            "morph_case": morph_case,
            "morph_gnn": ("-|-",) * n,
        },
    )


def test_grounded_attachment_scorer_fit_and_decode(tmp_path: Path) -> None:
    npz_path = tmp_path / "fit.npz"
    _write_fit_npz(npz_path)
    provider = GroundedFeatureProvider(npz_path, layer_index=0)
    sentences = [
        _sentence("s1", ("A", "B", "C")),
        _sentence("s2", ("D", "E")),
    ]
    heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "s1",
                sentences[0].text,
                sentences[0].tokens,
                (1, 0, -1),
                ("nsubj", "root", "obj"),
            ),
            HeadDeprelRecord(
                "s2",
                sentences[1].text,
                sentences[1].tokens,
                (1, 0),
                ("nsubj", "root"),
            ),
        ]
    )
    config = BilinearConfig(rank=4, epochs=2, learning_rate=0.1, seed=0)
    scorer = GroundedAttachmentScorer(config)
    scorer.fit(sentences, heads, provider)
    assert len(scorer.epoch_losses) == config.epochs
    assert all(isinstance(loss, float) for loss in scorer.epoch_losses)

    # Complete finite L0-style matrix (self-loops -inf).
    n = 3
    l0_scores = np.full((n, n + 1), 0.1, dtype=np.float64)
    for i in range(n):
        l0_scores[i, i + 1] = -np.inf
    l0_scores[1, 0] = 1.0  # prefer ROOT for token 1
    offsets = scorer.decode(sentences[0].tokens, "s1", l0_scores, provider)
    assert len(offsets) == n
    for index, offset in enumerate(offsets):
        if offset == 0:
            continue
        gov = resolve_governor_index(index, offset, n)
        assert gov is not None
        assert gov != index


def test_decode_fallback_when_only_dependent_covered(tmp_path: Path) -> None:
    """Dependent covered but no other covered word → pure L0 CLE path."""
    path = tmp_path / "partial.npz"
    last_residual = np.zeros((1, 4, FEATURE_DIM), dtype=np.float16)
    last_residual[0, 0, :] = 0.1
    np.savez(
        path,
        sent_ids=np.array(["a"], dtype="U8"),
        token_index=np.asarray([0], dtype=np.int64),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=last_residual,
    )
    # Fit on a fully covered second dump so U/V are real parameters.
    full_path = tmp_path / "full.npz"
    last_full = np.zeros((3, 4, FEATURE_DIM), dtype=np.float16)
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
        BilinearConfig(rank=4, epochs=2, learning_rate=0.1, seed=0)
    )
    scorer.fit(fit_sent, fit_heads, fit_provider)

    predict_provider = GroundedFeatureProvider(path, layer_index=0)
    tokens = ("Kind", "sieht", "Hund")
    n = len(tokens)
    l0_scores = np.full((n, n + 1), 0.05, dtype=np.float64)
    for i in range(n):
        l0_scores[i, i + 1] = -np.inf
    # Distinct L0 preferences so CLE is deterministic.
    l0_scores[0, 2] = 1.0  # token 0 → token 1
    l0_scores[1, 0] = 1.0  # token 1 → ROOT
    l0_scores[2, 2] = 1.0  # token 2 → token 1 (col 2 = token 1)

    pure_l0 = governor_columns_to_offsets(chu_liu_edmonds(l0_scores))
    decoded = scorer.decode(tokens, "a", l0_scores, predict_provider)
    # Only token 0 is covered → no bilinear overwrite → identical to pure L0 CLE.
    assert decoded == pure_l0
    assert len(decoded) == n
    # Valid single-root arborescence: exactly one ROOT, no self-loops.
    root_count = sum(1 for offset in decoded if offset == 0)
    assert root_count == 1
    for index, offset in enumerate(decoded):
        if offset == 0:
            continue
        gov = resolve_governor_index(index, offset, n)
        assert gov is not None and gov != index


def test_coverage_fractions_hand_computed(tmp_path: Path) -> None:
    npz_path = tmp_path / "tiny.npz"
    _write_fit_npz(npz_path)
    provider = GroundedFeatureProvider(npz_path, layer_index=0)
    sentences = [
        _sentence("s1", ("A", "B", "C", "D")),  # 3/4 covered (0,1,2)
        _sentence("s2", ("E", "F")),  # 2/2 covered
    ]
    # total tokens = 6; covered = 3+2 = 5 → 5/6
    assert covered_dependent_fraction(sentences, provider) == pytest.approx(5 / 6)

    heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "s1",
                sentences[0].text,
                sentences[0].tokens,
                (1, 0, -1, -2),  # 0→1 both cov; 1 ROOT; 2→1 both; 3→1 dep uncovered
                ("nsubj", "root", "obj", "amod"),
            ),
            HeadDeprelRecord(
                "s2",
                sentences[1].text,
                sentences[1].tokens,
                (1, 0),  # 0→1 both; 1 ROOT
                ("nsubj", "root"),
            ),
        ]
    )
    # s1: True, True, True, False; s2: True, True → 5/6
    assert attach_doubly_covered_fraction(sentences, heads, provider) == pytest.approx(5 / 6)


def test_uas_on_matched_set_and_count_invariant() -> None:
    keys = [("s", 0), ("s", 1), ("s", 2)]
    gold = {("s", 0): 1, ("s", 1): 0, ("s", 2): -1}
    # covered mask conceptually True,True,False — we pass only covered keys
    covered_keys = keys[:2]
    l0 = {("s", 0): 1, ("s", 1): 0, ("s", 2): 0}
    surface = {("s", 0): 0, ("s", 1): 0, ("s", 2): -1}
    grounded = {("s", 0): 1, ("s", 1): 1, ("s", 2): -1}

    assert score_uas_on_keys(l0, gold, covered_keys) == pytest.approx(1.0)  # both match
    assert score_uas_on_keys(surface, gold, covered_keys) == pytest.approx(0.5)
    assert score_uas_on_keys(grounded, gold, covered_keys) == pytest.approx(0.5)

    assert_matched_token_counts(
        {"l0": 2, "surface119": 2, "grounded": 2}, expected=2
    )
    with pytest.raises(AssertionError, match="token count mismatch"):
        assert_matched_token_counts(
            {"l0": 2, "surface119": 3, "grounded": 2}, expected=2
        )


def test_conversion_arms_plumb_through(tmp_path: Path) -> None:
    npz_path = tmp_path / "conv.npz"
    # Cover tokens 0,1,2 for sentence t1
    last_residual = np.zeros((3, 4, FEATURE_DIM), dtype=np.float16)
    last_residual[:, 0, 0] = 1.0
    np.savez(
        npz_path,
        sent_ids=np.array(["t1", "t1", "t1"], dtype="U8"),
        token_index=np.asarray([0, 1, 2], dtype=np.int64),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=last_residual,
    )
    provider = GroundedFeatureProvider(npz_path, layer_index=0)
    sentences = [
        _sentence("t1", ("A", "B", "C"), cases=("Nom", "Acc", "-")),
    ]
    gold_heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "t1",
                sentences[0].text,
                sentences[0].tokens,
                (1, 0, -1),
                ("nsubj", "root", "obj"),
            )
        ]
    )
    l0_heads = {"t1": (1, 0, -1)}
    grounded_heads = {"t1": (2, 0, -2)}  # different → mask may differ
    unary_deprels = {"t1": ("nsubj", "root", "obj")}
    grounded_deprels = {"t1": ("nsubj", "root", "obl")}
    case_base = {"t1": ("Nom", "Acc", "-")}
    case_better = {"t1": ("Nom", "Dat", "-")}

    baseline = score_conversion_arm(
        sentences, gold_heads, l0_heads, unary_deprels, case_base, provider
    )
    arm_121 = score_conversion_arm(
        sentences, gold_heads, l0_heads, grounded_deprels, case_base, provider
    )
    full = score_conversion_arm(
        sentences, gold_heads, grounded_heads, grounded_deprels, case_better, provider
    )

    for metrics in (baseline, arm_121, full):
        assert "las_strict" in metrics
        assert "las" in metrics
        assert "case" in metrics
        assert "tokens" in metrics
        assert isinstance(metrics["las_strict"], float)
        assert isinstance(metrics["case"], float)
        assert isinstance(metrics["tokens"], int)

    # baseline vs #121 share L0 heads → identical token counts
    assert baseline["tokens"] == arm_121["tokens"]
    # full uses different heads; tokens may differ (here both endpoints still covered)
    assert isinstance(full["tokens"], int)
    # distinct LAS: grounded deprels differ on token 2
    assert baseline["las_strict"] != arm_121["las_strict"] or baseline is arm_121


def test_c7_geometry_helpers_known_values() -> None:
    # Orthonormal rows (standard basis, padded): pairwise cosine ≈ 0.
    # After column-centering, n basis vectors span an (n-1)-dim flat →
    # participation-ratio effective rank ≈ n-1.
    eye = np.eye(4, FEATURE_DIM, dtype=np.float64)
    cos = _mean_pairwise_cosine(eye)
    assert cos == pytest.approx(0.0, abs=1e-9)
    rank = _effective_rank(eye)
    assert rank == pytest.approx(3.0, abs=1e-6)

    # All rows parallel (same direction, different scale) → cosine ≈ 1;
    # all variance lives in one coordinate after centering → effective rank ≈ 1.
    parallel = np.zeros((5, FEATURE_DIM), dtype=np.float64)
    parallel[:, 0] = np.arange(1, 6, dtype=np.float64)
    cos_rep = _mean_pairwise_cosine(parallel)
    assert cos_rep == pytest.approx(1.0, abs=1e-9)
    rank_rep = _effective_rank(parallel)
    assert rank_rep == pytest.approx(1.0, abs=1e-6)


def _attach_read(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "uas_l0": 0.50,
        "uas_surface119": 0.48,
        "uas_grounded": 0.55,
        "las_base": 0.40,
        "las_121": 0.42,
        "las_full": 0.45,
        "case_base": 0.50,
        "case_121": 0.52,
        "case_full": 0.55,
        "dev_layer": 17,
        "covered_dependent_fraction": {"dev": 0.70, "test": 0.68},
        "doubly_covered_fraction": {"dev": 0.60, "test": 0.58},
        "c7_cosine": 0.05,
        "c7_effrank": 120.0,
    }
    base.update(overrides)
    return preregistered_attach_read(**base)  # type: ignore[arg-type]


def test_preregistered_attach_read_fires() -> None:
    read = _attach_read()
    assert read["verdict"] == "FIRES"
    assert read["uas_gain_vs_l0"] == pytest.approx(0.05)
    assert read["uas_gain_vs_surface119"] == pytest.approx(0.07)
    assert read["las_full_gain_vs_base"] == pytest.approx(0.05)


def test_preregistered_attach_read_halts_nonpositive_uas_gain() -> None:
    read = _attach_read(uas_grounded=0.49)
    assert read["verdict"] == "HALTED"
    assert "does not beat L0" in read["text"]


@pytest.mark.parametrize(
    ("overrides", "expected_fragment"),
    [
        ({"uas_grounded": 0.52}, "UAS gain vs L0 is below +0.03"),
        (
            {"uas_grounded": 0.55, "uas_surface119": 0.56},
            "grounded UAS does not beat surface-#119",
        ),
        (
            {"uas_grounded": 0.55, "las_full": 0.41},
            "conversion strict-LAS gain vs baseline is below +0.03",
        ),
    ],
    ids=("small-uas-gain", "fails-surface119", "small-las-gain"),
)
def test_preregistered_attach_read_in_between_for_each_condition(
    overrides: dict[str, object],
    expected_fragment: str,
) -> None:
    read = _attach_read(**overrides)
    assert read["verdict"] == "IN-BETWEEN"
    assert expected_fragment in read["text"]


def test_preregistered_attach_read_through_line_branches() -> None:
    met = _attach_read(case_full=0.91)
    assert met["through_line"]["met"] is True
    assert "met" in met["through_line"]["text"]

    plateau = _attach_read(case_full=0.80, case_base=0.50)
    assert plateau["through_line"]["plateau"] is True
    assert "plateau" in plateau["through_line"]["text"]

    diagnose = _attach_read(case_full=0.40, case_base=0.50)
    assert diagnose["through_line"]["met"] is False
    assert "diagnose" in diagnose["through_line"]["text"]

    below = _attach_read(case_full=0.60, case_base=0.50)
    # 0.60 > case_base 0.50 but ≤ CASE_BASELINE 0.76
    assert below["through_line"]["met"] is False
    assert below["through_line"]["plateau"] is False
    assert "below plateau" in below["through_line"]["text"]


def test_campaign_test_read_guard_second_claim_raises() -> None:
    guard = CampaignTestReadGuard()
    guard.claim("shared")
    with pytest.raises(RuntimeError, match="only once"):
        guard.claim("shared")
    guard.claim("head_deprel")
    with pytest.raises(RuntimeError, match="only once"):
        guard.claim("head_deprel")


def test_select_dev_attach_layer_max_uas_smallest_index_tiebreak() -> None:
    rows = [
        {
            "layer_index": 0,
            "layer": 8,
            "dev_uas": 0.50,
            "dev_covered_dependent_fraction": 0.7,
            "dev_doubly_covered_fraction": 0.4,
        },
        {
            "layer_index": 1,
            "layer": 17,
            "dev_uas": 0.55,
            "dev_covered_dependent_fraction": 0.7,
            "dev_doubly_covered_fraction": 0.4,
        },
        {
            "layer_index": 2,
            "layer": 26,
            "dev_uas": 0.55,
            "dev_covered_dependent_fraction": 0.7,
            "dev_doubly_covered_fraction": 0.4,
        },
        {
            "layer_index": 3,
            "layer": 35,
            "dev_uas": 0.40,
            "dev_covered_dependent_fraction": 0.7,
            "dev_doubly_covered_fraction": 0.4,
        },
    ]
    # max UAS 0.55 shared by indices 1 and 2 → pick smallest index 1
    assert select_dev_attach_layer(rows) == 1
    rows[2]["dev_uas"] = 0.60
    assert select_dev_attach_layer(rows) == 2


def test_collect_covered_dependent_keys_order(tmp_path: Path) -> None:
    npz_path = tmp_path / "keys.npz"
    _write_fit_npz(npz_path)
    provider = GroundedFeatureProvider(npz_path, layer_index=0)
    sentences = [
        _sentence("s1", ("A", "B", "C")),
        _sentence("s2", ("D", "E")),
    ]
    keys = collect_covered_dependent_keys(sentences, provider)
    assert keys == [("s1", 0), ("s1", 1), ("s1", 2), ("s2", 0), ("s2", 1)]
