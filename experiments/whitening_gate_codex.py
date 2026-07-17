"""TRAIN-fit anisotropy-removal gate for grounded label and attachment bilinears.

The signed gate is in ``docs/notes/whitening_gate_prereg.md``.  Whitening is
fit on the richer MWT TRAIN residual dump only.  Label and attachment choose
their own whitening variant/checkpoint on DEV, then share one guarded TEST
load.  Attachment reconciliation implements the preregistered PIN-B
log-probability decode for both the raw and whitened grounded lanes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.attach_levers_codex import (  # noqa: E402
    BilinearAttachmentScorer,
    BilinearConfig,
    chu_liu_edmonds,
    governor_columns_to_offsets,
    l0_edge_scores,
    predict_deprels,
    score_prediction_sequences,
)
from experiments.biaffine_labeler_codex import (  # noqa: E402
    BilinearRelationLabeler,
    RelationLabelerConfig,
)
from experiments.german_r1_codex import (  # noqa: E402
    DATA_ROOT,
    GermanR1Student,
    RegisterLayer,
    Sentence,
    accuracy,
    load_split,
)
from experiments.german_r3_codex import (  # noqa: E402
    CampaignTestReadGuard,
    GermanR3DependencyStudent,
    _predicted_pos_by_sentence,
    choose_window,
    load_head_deprel_file,
    load_head_test_once,
    load_shared_test_once,
    run_predicted_case_sentence,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    aligned_head_record,
    head_deprel_by_sent,
)
from experiments.grounded_attach_codex import (  # noqa: E402
    GroundedAttachmentScorer,
    covered_dependent_fraction,
    covered_dependent_indices,
    doubly_covered_fraction,
)
from experiments.grounded_labeler_codex import (  # noqa: E402
    CHECKPOINT_LAYERS,
    FEATURE_DIM,
    LAYER_INDICES,
    GroundedRelationLabeler,
    GroundedResidualProvider,
    evaluate_dev_layer,
    evaluate_matched_labelers,
    select_dev_layer,
)

SEED = 0
VARIANTS = ("v1", "v2_k1", "v2_k2")
VARIANT_COMPLEXITY = {"v1": 0, "v2_k1": 1, "v2_k2": 2}
GROUND_ROOT = Path(__file__).resolve().parents[1] / "data" / "grounded"
GROUND_PATHS = {
    "train": GROUND_ROOT / "qwen3b_mwt_gsd_train_n600.npz",
    "dev": GROUND_ROOT / "qwen3b_mwt_gsd_dev_n799.npz",
    "test": GROUND_ROOT / "qwen3b_mwt_gsd_test_n977.npz",
}
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "whitening_gate_codex.json"
LABEL_ARMS = ("unary", "surface119", "raw121", "whitened")
ATTACH_ARMS = ("l0", "surface119", "raw122", "whitened")
CONVERSION_ARMS = ("baseline", "#121_labels", "raw122_full", "full_grounded_labels")


@dataclass(frozen=True)
class WhiteningTransform:
    """One fixed TRAIN-fit map for one whitening variant and checkpoint."""

    variant: str
    layer: int
    input_mean: np.ndarray
    remainder_mean: np.ndarray
    sigma: np.ndarray
    components: np.ndarray
    fit_split: str = "train"

    def transform(self, raw_phi_vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(raw_phi_vector, dtype=np.float64)
        if vector.shape != self.input_mean.shape:
            raise ValueError(
                f"raw phi shape {vector.shape} does not match fit shape {self.input_mean.shape}"
            )
        remainder = vector - self.input_mean
        if self.components.size:
            remainder = remainder - (remainder @ self.components.T) @ self.components
        centered = remainder - self.remainder_mean
        return np.divide(
            centered,
            self.sigma,
            out=np.zeros_like(centered),
            where=self.sigma > 0.0,
        )


class WhitenedResidualProvider:
    """Drop-in provider adapter that applies one transform at the phi seam."""

    def __init__(
        self,
        provider: GroundedResidualProvider,
        transform: WhiteningTransform,
    ) -> None:
        self.provider = provider
        self.transform = transform
        self.path = provider.path
        self.checkpoint_layers = provider.checkpoint_layers
        self.sentence_ids = provider.sentence_ids
        self._row_by_key = provider._row_by_key

    def _validate_layer(self, layer: int) -> None:
        self.provider._validate_layer(layer)
        if layer != self.transform.layer:
            raise ValueError(
                f"whitening transform is fit for layer {self.transform.layer}, got {layer}"
            )

    def covered(self, sent_id: str, token_index: int) -> bool:
        return self.provider.covered(sent_id, token_index)

    def residual(self, sent_id: str, token_index: int, layer: int) -> np.ndarray:
        self._validate_layer(layer)
        return self.transform.transform(
            self.provider.residual(sent_id, token_index, layer)
        )


def _top_principal_components(centered: np.ndarray, count: int = 2) -> np.ndarray:
    """Return deterministic leading right singular vectors of centered TRAIN phi."""
    if centered.ndim != 2 or not centered.shape[0] or not centered.shape[1]:
        raise ValueError("PCA input must be a nonempty matrix")
    rank_bound = min(centered.shape)
    wanted = min(count, rank_bound)
    if wanted < count:
        raise ValueError(f"need matrix rank bound >= {count}, got {rank_bound}")
    if centered.shape[0] <= 512 or rank_bound <= count + 1:
        _u, _singular, right = np.linalg.svd(centered, full_matrices=False)
        return np.asarray(right[:count], dtype=np.float64)

    # ARPACK performs the requested SVD without materializing a 2048x2048
    # covariance matrix.  The start vector fixes its otherwise arbitrary RNG.
    from scipy.sparse.linalg import svds

    _u, singular, right = svds(
        centered,
        k=count,
        which="LM",
        v0=np.random.default_rng(SEED).normal(size=rank_bound),
        return_singular_vectors=True,
    )
    order = np.argsort(singular)[::-1]
    return np.asarray(right[order], dtype=np.float64)


def fit_layer_whitening_transforms(
    train_provider: GroundedResidualProvider,
    layer: int,
) -> dict[str, WhiteningTransform]:
    """Fit V1 and both V2 maps from every row of one TRAIN checkpoint."""
    train_provider._validate_layer(layer)
    matrix = np.asarray(train_provider.last_residual[:, layer, :], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM or not len(matrix):
        raise ValueError("TRAIN residual matrix has an unexpected shape")
    input_mean = matrix.mean(axis=0)
    centered = matrix - input_mean
    components = _top_principal_components(centered, count=2)
    transforms: dict[str, WhiteningTransform] = {}
    for variant in VARIANTS:
        component_count = {"v1": 0, "v2_k1": 1, "v2_k2": 2}[variant]
        selected = components[:component_count].copy()
        remainder = centered.copy()
        if component_count:
            remainder -= (remainder @ selected.T) @ selected
        remainder_mean = remainder.mean(axis=0)
        sigma = (remainder - remainder_mean).std(axis=0)
        transforms[variant] = WhiteningTransform(
            variant=variant,
            layer=layer,
            input_mean=input_mean.copy(),
            remainder_mean=remainder_mean,
            sigma=sigma,
            components=selected,
        )
    return transforms


def pin_a_row_keys(
    provider: GroundedResidualProvider,
    sample_size: int = 800,
    seed: int = SEED,
) -> tuple[tuple[str, int], ...]:
    """Draw the fixed PIN-A row-key sample once for apples-to-apples geometry."""
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    keys = sorted(provider._row_by_key)
    if len(keys) < sample_size:
        raise ValueError(f"PIN A requires {sample_size} rows, provider has {len(keys)}")
    chosen = np.sort(
        np.random.default_rng(seed).choice(len(keys), size=sample_size, replace=False)
    )
    return tuple(keys[int(index)] for index in chosen)


def centered_participation_ratio(matrix: np.ndarray) -> float:
    """Compute ``(sum(sigma^2))^2 / sum(sigma^4)`` on a centered matrix."""
    sample = np.asarray(matrix, dtype=np.float64)
    if sample.ndim != 2 or not len(sample):
        raise ValueError("effective-rank input must be a nonempty matrix")
    centered = sample - sample.mean(axis=0)
    squared_sum = float(np.square(centered).sum())
    gram = centered @ centered.T
    fourth_sum = float(np.square(gram).sum())
    return squared_sum**2 / fourth_sum if fourth_sum else 1.0


def pin_a_effrank(
    provider: GroundedResidualProvider | WhitenedResidualProvider,
    layer: int,
    row_keys: Sequence[tuple[str, int]],
) -> dict[str, float | int]:
    """Measure centered participation-ratio effective rank on fixed row keys."""
    provider._validate_layer(layer)
    if not row_keys:
        raise ValueError("PIN-A row-key sample must not be empty")
    sample = np.stack(
        [provider.residual(sent_id, index, layer) for sent_id, index in row_keys]
    )
    return {
        "effrank": centered_participation_ratio(sample),
        "sample_size": len(row_keys),
        "seed": SEED,
    }


def _log_softmax_finite(values: np.ndarray) -> np.ndarray:
    """Normalize finite entries and preserve invalid entries as negative infinity."""
    vector = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vector)
    if not finite.any():
        raise ValueError("cannot normalize a row with no finite entries")
    maximum = float(np.max(vector[finite]))
    log_normalizer = maximum + float(np.log(np.exp(vector[finite] - maximum).sum()))
    result = np.full_like(vector, -np.inf)
    result[finite] = vector[finite] - log_normalizer
    return result


def edge_log_probs(raw_scores: np.ndarray) -> np.ndarray:
    """Convert every row of an arc-score matrix to a proper log-distribution."""
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("edge scores must be a matrix")
    return np.stack([_log_softmax_finite(row) for row in scores])


def pin_b_edge_log_probs(
    bilinear_scorer: GroundedAttachmentScorer,
    l0_student: GermanR3DependencyStudent,
    provider: GroundedResidualProvider | WhitenedResidualProvider,
    sent_id: str,
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    layer: int,
) -> np.ndarray:
    """Build the preregistered PIN-B row-normalized arc log-probabilities."""
    provider._validate_layer(layer)
    if len(tokens) != len(predicted_pos):
        raise ValueError("tokens and predicted POS must align")
    l0_log_probs = edge_log_probs(l0_edge_scores(l0_student, tokens, predicted_pos))
    result = l0_log_probs.copy()
    covered = tuple(
        index for index in range(len(tokens)) if provider.covered(sent_id, index)
    )
    covered_set = set(covered)
    for dependent in range(len(tokens)):
        if dependent not in covered_set:
            continue
        candidates = tuple(governor for governor in covered if governor != dependent)
        if not candidates:
            continue
        bilinear_scores = np.asarray(
            [
                bilinear_scorer.score_edge(
                    provider, sent_id, dependent, governor, layer
                )
                for governor in candidates
            ],
            dtype=np.float64,
        )
        bilinear_log_probs = _log_softmax_finite(bilinear_scores)
        for governor, log_probability in zip(
            candidates, bilinear_log_probs, strict=True
        ):
            result[dependent, governor + 1] = log_probability
        result[dependent] = _log_softmax_finite(result[dependent])
    for dependent in range(len(tokens)):
        result[dependent, dependent + 1] = -np.inf
        finite = np.isfinite(result[dependent])
        if not np.isclose(np.exp(result[dependent, finite]).sum(), 1.0, atol=1e-12):
            raise AssertionError("PIN-B row is not a proper log-distribution")
    return result


def select_variant_layer(
    dev_scores: Mapping[tuple[str, int], float],
) -> tuple[str, int]:
    """DEV argmax; ties prefer V1, then V2-k1, then V2-k2, then lower layer."""
    expected = {(variant, layer) for variant in VARIANTS for layer in LAYER_INDICES}
    if set(dev_scores) != expected:
        raise ValueError("DEV whitening sweep must contain every variant/layer pair")
    return max(
        expected,
        key=lambda key: (
            float(dev_scores[key]),
            -VARIANT_COMPLEXITY[key[0]],
            -key[1],
        ),
    )


def assert_matched_labeler_uas(metrics: Mapping[str, Mapping[str, Any]]) -> None:
    """Assert that all four label arms score one identical head sequence."""
    expected = float(metrics["unary"]["uas"])
    for arm in LABEL_ARMS[1:]:
        actual = float(metrics[arm]["uas"])
        if actual != expected:
            raise AssertionError(
                f"matched-labeler UAS invariant failed for {arm}: {actual} != {expected}"
            )


def assert_identical_arc_sets(
    arc_sets: Mapping[str, Sequence[tuple[str, int]]],
) -> None:
    """Assert exact covered-dependent identity across attachment arms."""
    normalized = {arm: tuple(indices) for arm, indices in arc_sets.items()}
    if set(normalized) != set(ATTACH_ARMS):
        raise AssertionError("attachment arc-set assertion requires all four arms")
    expected = normalized["l0"]
    for arm in ATTACH_ARMS[1:]:
        if normalized[arm] != expected:
            raise AssertionError(f"attachment covered arc set differs for {arm}")


def _score_dependency_arm(
    predicted_heads: Sequence[int],
    predicted_deprels: Sequence[str],
    gold_heads: Sequence[int],
    gold_deprels: Sequence[str],
) -> dict[str, float | int]:
    metric = score_prediction_sequences(
        predicted_heads, predicted_deprels, gold_heads, gold_deprels
    )
    return {
        "uas": metric["uas"],
        "las_strict": metric["las"],
        "deprel_only_accuracy": metric["deprel_only_accuracy"],
        "tokens": metric["tokens"],
    }


def evaluate_attach_dev_candidate(
    dev: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    scorer: GroundedAttachmentScorer,
    provider: GroundedResidualProvider | WhitenedResidualProvider,
    layer: int,
) -> tuple[float, int]:
    """Score one raw/whitened attachment candidate on covered DEV dependents."""
    predicted: list[int] = []
    gold: list[int] = []
    for sentence in dev:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        offsets = governor_columns_to_offsets(
            chu_liu_edmonds(
                pin_b_edge_log_probs(
                    scorer,
                    l0_student,
                    provider,
                    sentence.sent_id,
                    sentence.tokens,
                    pos,
                    layer,
                )
            )
        )
        selected = covered_dependent_indices(
            sentence.sent_id, len(sentence.tokens), provider, layer
        )
        predicted.extend(offsets[index] for index in selected)
        gold.extend(head.head_offset[index] for index in selected)
    if not gold:
        raise ValueError(f"DEV has no covered dependents at layer {layer}")
    return accuracy(predicted, gold), len(gold)


def _combine_labeler_evaluations(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    surface_labeler: BilinearRelationLabeler,
    raw_labeler: GroundedRelationLabeler,
    raw_provider: GroundedResidualProvider,
    raw_layer: int,
    whitened_labeler: GroundedRelationLabeler,
    whitened_provider: WhitenedResidualProvider,
    whitened_layer: int,
    case_student: GermanR1Student,
) -> tuple[dict[str, dict[str, Any]], float]:
    raw_metrics, raw_fraction = evaluate_matched_labelers(
        test,
        heads,
        predicted_pos,
        l0_student,
        surface_labeler,
        raw_labeler,
        raw_provider,
        raw_layer,
        case_student,
    )
    white_metrics, white_fraction = evaluate_matched_labelers(
        test,
        heads,
        predicted_pos,
        l0_student,
        surface_labeler,
        whitened_labeler,
        whitened_provider,
        whitened_layer,
        case_student,
    )
    if raw_fraction != white_fraction:
        raise AssertionError("raw and whitened labeler arc fractions differ")
    for arm in ("unary", "surface119"):
        if raw_metrics[arm] != white_metrics[arm]:
            raise AssertionError(f"raw/whitened matched baseline drifted for {arm}")
    combined = {
        "unary": raw_metrics["unary"],
        "surface119": raw_metrics["surface119"],
        "raw121": raw_metrics["grounded"],
        "whitened": white_metrics["grounded"],
    }
    assert_matched_labeler_uas(combined)
    return combined, raw_fraction


def evaluate_attachment_test(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    surface_scorer: BilinearAttachmentScorer,
    raw_scorer: GroundedAttachmentScorer,
    raw_provider: GroundedResidualProvider,
    raw_layer: int,
    whitened_scorer: GroundedAttachmentScorer,
    whitened_provider: WhitenedResidualProvider,
    whitened_layer: int,
    raw_labeler: GroundedRelationLabeler,
    raw_label_provider: GroundedResidualProvider,
    raw_label_layer: int,
    whitened_labeler: GroundedRelationLabeler,
    whitened_label_provider: WhitenedResidualProvider,
    whitened_label_layer: int,
    case_student: GermanR1Student,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Evaluate four head arms plus raw/whitened conversion on one TEST load."""
    covered_values = {arm: {"head": [], "deprel": []} for arm in ATTACH_ARMS}
    full_values = {arm: {"head": [], "deprel": []} for arm in ATTACH_ARMS}
    conversions = {
        arm: {"head": [], "deprel": [], "case": []} for arm in CONVERSION_ARMS
    }
    covered_gold_head: list[int] = []
    covered_gold_deprel: list[str] = []
    full_gold_head: list[int] = []
    full_gold_deprel: list[str] = []
    gold_case: list[str] = []
    arc_sets = {arm: [] for arm in ATTACH_ARMS}

    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        l0_heads = governor_columns_to_offsets(
            chu_liu_edmonds(edge_log_probs(l0_edge_scores(l0_student, sentence.tokens, pos)))
        )
        surface_heads = governor_columns_to_offsets(
            chu_liu_edmonds(edge_log_probs(surface_scorer.score_matrix(sentence.tokens, pos)))
        )
        raw_heads = governor_columns_to_offsets(
            chu_liu_edmonds(
                pin_b_edge_log_probs(
                    raw_scorer,
                    l0_student,
                    raw_provider,
                    sentence.sent_id,
                    sentence.tokens,
                    pos,
                    raw_layer,
                )
            )
        )
        whitened_heads = governor_columns_to_offsets(
            chu_liu_edmonds(
                pin_b_edge_log_probs(
                    whitened_scorer,
                    l0_student,
                    whitened_provider,
                    sentence.sent_id,
                    sentence.tokens,
                    pos,
                    whitened_layer,
                )
            )
        )
        head_predictions = {
            "l0": l0_heads,
            "surface119": surface_heads,
            "raw122": raw_heads,
            "whitened": whitened_heads,
        }
        head_labels = {
            arm: predict_deprels(l0_student, sentence.tokens, pos, offsets)
            for arm, offsets in head_predictions.items()
        }
        selected = covered_dependent_indices(
            sentence.sent_id, len(sentence.tokens), raw_provider, raw_layer
        )
        selected_white = covered_dependent_indices(
            sentence.sent_id,
            len(sentence.tokens),
            whitened_provider,
            whitened_layer,
        )
        if selected != selected_white:
            raise AssertionError("raw and whitened coverage differs")
        for arm in ATTACH_ARMS:
            arc_sets[arm].extend((sentence.sent_id, index) for index in selected)
            covered_values[arm]["head"].extend(
                head_predictions[arm][index] for index in selected
            )
            covered_values[arm]["deprel"].extend(
                head_labels[arm][index] for index in selected
            )
            full_values[arm]["head"].extend(head_predictions[arm])
            full_values[arm]["deprel"].extend(head_labels[arm])

        label_121 = whitened_labeler.predict(
            sentence.sent_id,
            sentence.tokens,
            pos,
            l0_heads,
            whitened_label_provider,
            whitened_label_layer,
            l0_student,
        )
        raw_full_labels = raw_labeler.predict(
            sentence.sent_id,
            sentence.tokens,
            pos,
            raw_heads,
            raw_label_provider,
            raw_label_layer,
            l0_student,
        )
        whitened_full_labels = whitened_labeler.predict(
            sentence.sent_id,
            sentence.tokens,
            pos,
            whitened_heads,
            whitened_label_provider,
            whitened_label_layer,
            l0_student,
        )
        conversion_predictions = {
            "baseline": (l0_heads, head_labels["l0"]),
            "#121_labels": (l0_heads, label_121),
            "raw122_full": (raw_heads, raw_full_labels),
            "full_grounded_labels": (whitened_heads, whitened_full_labels),
        }
        case_baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        case_eligible = [
            index
            for index, label in enumerate(sentence.targets["morph_case"])
            if label != "-"
        ]
        for arm, (offsets, labels) in conversion_predictions.items():
            case_result = run_predicted_case_sentence(
                case_student,
                sentence.tokens,
                case_baseline,
                pos,
                offsets,
                labels,
                case_eligible,
            )
            conversions[arm]["head"].extend(offsets[index] for index in selected)
            conversions[arm]["deprel"].extend(labels[index] for index in selected)
            conversions[arm]["case"].extend(
                case_result.predictions[index] for index in case_eligible
            )

        covered_gold_head.extend(head.head_offset[index] for index in selected)
        covered_gold_deprel.extend(head.deprel[index] for index in selected)
        full_gold_head.extend(head.head_offset)
        full_gold_deprel.extend(head.deprel)
        gold_case.extend(sentence.targets["morph_case"][index] for index in case_eligible)

    if not covered_gold_head:
        raise ValueError("TEST has no covered-dependent arcs")
    assert_identical_arc_sets(arc_sets)
    attachment: dict[str, dict[str, Any]] = {}
    for arm in ATTACH_ARMS:
        attachment[arm] = {
            "covered": _score_dependency_arm(
                covered_values[arm]["head"],
                covered_values[arm]["deprel"],
                covered_gold_head,
                covered_gold_deprel,
            ),
            "full": _score_dependency_arm(
                full_values[arm]["head"],
                full_values[arm]["deprel"],
                full_gold_head,
                full_gold_deprel,
            ),
        }
    if {int(attachment[arm]["covered"]["tokens"]) for arm in ATTACH_ARMS} != {
        len(covered_gold_head)
    }:
        raise AssertionError("covered-dependent scorer denominators differ")

    conversion_metrics: dict[str, dict[str, Any]] = {}
    for arm in CONVERSION_ARMS:
        dependency = _score_dependency_arm(
            conversions[arm]["head"],
            conversions[arm]["deprel"],
            covered_gold_head,
            covered_gold_deprel,
        )
        conversion_metrics[arm] = {
            **dependency,
            "serve_honest_morph_case": accuracy(conversions[arm]["case"], gold_case),
            "serve_honest_morph_case_n": len(gold_case),
        }
    return attachment, conversion_metrics


def section5_read(
    labeler: Mapping[str, Mapping[str, Any]],
    attachment: Mapping[str, Mapping[str, Any]],
    conversion: Mapping[str, Mapping[str, Any]],
    dev_selection: Mapping[str, Mapping[str, Any]],
    effrank: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    v1_vs_v2: Mapping[str, Any],
    *,
    cross_vendor_robust: bool,
) -> dict[str, Any]:
    """Apply the signed Section 5 decision rules exactly once."""
    labeler_gain = float(labeler["whitened"]["las_strict"]) - float(
        labeler["unary"]["las_strict"]
    )
    labeler_delta = float(labeler["whitened"]["las_strict"]) - float(
        labeler["raw121"]["las_strict"]
    )
    attach_l0 = float(attachment["l0"]["covered"]["uas"])
    attach_white = float(attachment["whitened"]["covered"]["uas"])
    attach_raw = float(attachment["raw122"]["covered"]["uas"])
    attach_uas_gain = attach_white - attach_l0
    attach_uas_delta = attach_white - attach_raw
    baseline_las = float(conversion["baseline"]["las_strict"])
    whitened_las = float(conversion["full_grounded_labels"]["las_strict"])
    raw_las = float(conversion["raw122_full"]["las_strict"])
    attach_conv_las_gain = whitened_las - baseline_las
    attach_conv_delta = whitened_las - raw_las
    labeler_clears = labeler_gain >= 0.03
    attach_clears = attach_uas_gain >= 0.03 and attach_conv_las_gain >= 0.03
    cleared = []
    if labeler_clears:
        cleared.append("labeler")
    if attach_clears:
        cleared.append("attach")
    any_improvement = (
        labeler_delta > 0.0 or attach_uas_delta > 0.0 or attach_conv_delta > 0.0
    )
    if cleared and cross_vendor_robust:
        verdict = "FIRES"
        fired_primitive = "+".join(cleared)
        reason = "the preregistered bar is cleared on the pinned decode and cross-vendor robust"
    elif any_improvement or cleared:
        verdict = "IN-BETWEEN"
        fired_primitive = None
        reason = (
            "a whitening gain is present but no bar is robustly cleared"
            if not cleared
            else "a bar clears locally but cross-vendor robustness is not established"
        )
    else:
        verdict = "HALTED"
        fired_primitive = None
        reason = (
            "whitening is no better than raw phi on both primitives; mechanism is "
            "optimization conditioning / deeper -> rung 3 (R4 LLM-hybrid)"
        )

    case_baseline = float(conversion["baseline"]["serve_honest_morph_case"])
    case_whitened = float(
        conversion["full_grounded_labels"]["serve_honest_morph_case"]
    )
    if case_whitened <= case_baseline:
        through_line = "diagnose (at or below baseline)"
    elif case_whitened >= 0.90:
        through_line = "met"
    elif case_whitened > 0.76:
        through_line = "plateau language"
    else:
        through_line = "diagnose"

    label_variant = str(dev_selection["labeler"]["variant"])
    label_layer = int(dev_selection["labeler"]["layer"])
    attach_variant = str(dev_selection["attach"]["variant"])
    attach_layer = int(dev_selection["attach"]["layer"])
    effrank_raw = {
        "labeler": float(effrank[str(label_layer)]["raw"]["effrank"]),
        "attach": float(effrank[str(attach_layer)]["raw"]["effrank"]),
    }
    effrank_whitened = {
        "labeler": float(effrank[str(label_layer)][label_variant]["effrank"]),
        "attach": float(effrank[str(attach_layer)][attach_variant]["effrank"]),
    }
    text = (
        f"{verdict}: labeler whitened-unary strict LAS={labeler_gain:+.6f}, "
        f"whitened-raw121={labeler_delta:+.6f}; attach whitened-L0 UAS="
        f"{attach_uas_gain:+.6f}, whitened-raw122={attach_uas_delta:+.6f}; "
        f"conversion whitened-baseline LAS={attach_conv_las_gain:+.6f}, "
        f"whitened-raw122={attach_conv_delta:+.6f}; through-line={through_line}. "
        f"PIN A effrank raw->whitened labeler {effrank_raw['labeler']:.3f}->"
        f"{effrank_whitened['labeler']:.3f}, attach {effrank_raw['attach']:.3f}->"
        f"{effrank_whitened['attach']:.3f}. {reason}."
    )
    return {
        "verdict": verdict,
        "fired_primitive": fired_primitive,
        "labeler_unary": dict(labeler["unary"]),
        "labeler_surface": dict(labeler["surface119"]),
        "labeler_raw121": dict(labeler["raw121"]),
        "labeler_whitened": dict(labeler["whitened"]),
        "labeler_gain_vs_unary": labeler_gain,
        "labeler_delta": labeler_delta,
        "attach_l0": dict(attachment["l0"]),
        "attach_surface": dict(attachment["surface119"]),
        "attach_raw122": dict(attachment["raw122"]),
        "attach_whitened": dict(attachment["whitened"]),
        "attach_uas_gain": attach_uas_gain,
        "attach_uas_delta_vs_raw122": attach_uas_delta,
        "attach_conv_las_gain": attach_conv_las_gain,
        "attach_conv_las_delta_vs_raw122": attach_conv_delta,
        "dev_variant": {primitive: row["variant"] for primitive, row in dev_selection.items()},
        "dev_layer": {primitive: row["layer"] for primitive, row in dev_selection.items()},
        "effrank_raw": effrank_raw,
        "effrank_whitened": effrank_whitened,
        "v1_vs_v2": dict(v1_vs_v2),
        "case_whitened": case_whitened,
        "case_baseline": case_baseline,
        "through_line": through_line,
        "cross_vendor_robust": cross_vendor_robust,
        "text": text,
    }


def _fit_transforms_and_geometry(
    train_provider: GroundedResidualProvider,
) -> tuple[
    dict[tuple[str, int], WhiteningTransform],
    dict[str, dict[str, dict[str, float | int]]],
]:
    transforms: dict[tuple[str, int], WhiteningTransform] = {}
    geometry: dict[str, dict[str, dict[str, float | int]]] = {}
    row_keys = pin_a_row_keys(train_provider, sample_size=800, seed=SEED)
    for layer in LAYER_INDICES:
        print(
            f"WHITEN FIT: layer={layer} checkpoint={CHECKPOINT_LAYERS[layer]} "
            "on all TRAIN rows.",
            flush=True,
        )
        fitted = fit_layer_whitening_transforms(train_provider, layer)
        geometry[str(layer)] = {
            "raw": pin_a_effrank(train_provider, layer, row_keys)
        }
        for variant, transform in fitted.items():
            transforms[(variant, layer)] = transform
            adapter = WhitenedResidualProvider(train_provider, transform)
            geometry[str(layer)][variant] = pin_a_effrank(adapter, layer, row_keys)
    return transforms, geometry


def _labeler_sweeps(
    grounded_train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
    train_provider: GroundedResidualProvider,
    transforms: Mapping[tuple[str, int], WhiteningTransform],
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    dev_provider: GroundedResidualProvider,
    l0_student: GermanR3DependencyStudent,
    config: RelationLabelerConfig,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], tuple[str, int]]:
    raw_curve: list[dict[str, Any]] = []
    for layer in LAYER_INDICES:
        labeler = GroundedRelationLabeler(config)
        labeler.fit(grounded_train, train_provider, train_heads, layer)
        dev_accuracy, fraction, count = evaluate_dev_layer(
            dev, dev_heads, dev_pos, l0_student, labeler, dev_provider, layer
        )
        raw_curve.append(
            {
                "layer": layer,
                "checkpoint_layer": int(CHECKPOINT_LAYERS[layer]),
                "deprel_only_accuracy": dev_accuracy,
                "doubly_covered_fraction": fraction,
                "doubly_covered_n": count,
            }
        )
        print(
            f"DEV LABEL RAW: layer={layer} checkpoint={CHECKPOINT_LAYERS[layer]} "
            f"deprel-only={dev_accuracy:.6f} n={count}",
            flush=True,
        )
    raw_layer = select_dev_layer(
        {row["layer"]: row["deprel_only_accuracy"] for row in raw_curve}
    )

    white_curve: list[dict[str, Any]] = []
    scores: dict[tuple[str, int], float] = {}
    for variant in VARIANTS:
        for layer in LAYER_INDICES:
            train_white = WhitenedResidualProvider(
                train_provider, transforms[(variant, layer)]
            )
            dev_white = WhitenedResidualProvider(
                dev_provider, transforms[(variant, layer)]
            )
            labeler = GroundedRelationLabeler(config)
            labeler.fit(grounded_train, train_white, train_heads, layer)
            dev_accuracy, fraction, count = evaluate_dev_layer(
                dev, dev_heads, dev_pos, l0_student, labeler, dev_white, layer
            )
            scores[(variant, layer)] = dev_accuracy
            white_curve.append(
                {
                    "variant": variant,
                    "layer": layer,
                    "checkpoint_layer": int(CHECKPOINT_LAYERS[layer]),
                    "deprel_only_accuracy": dev_accuracy,
                    "doubly_covered_fraction": fraction,
                    "doubly_covered_n": count,
                }
            )
            print(
                f"DEV LABEL WHITE: variant={variant} layer={layer} "
                f"checkpoint={CHECKPOINT_LAYERS[layer]} deprel-only={dev_accuracy:.6f} "
                f"n={count}",
                flush=True,
            )
    return raw_curve, raw_layer, white_curve, select_variant_layer(scores)


def _attachment_sweeps(
    grounded_train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
    train_provider: GroundedResidualProvider,
    transforms: Mapping[tuple[str, int], WhiteningTransform],
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    dev_provider: GroundedResidualProvider,
    l0_student: GermanR3DependencyStudent,
    config: BilinearConfig,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], tuple[str, int]]:
    raw_curve: list[dict[str, Any]] = []
    for layer in LAYER_INDICES:
        scorer = GroundedAttachmentScorer(config)
        scorer.fit(grounded_train, train_provider, train_heads, layer)
        dev_uas, count = evaluate_attach_dev_candidate(
            dev, dev_heads, dev_pos, l0_student, scorer, dev_provider, layer
        )
        raw_curve.append(
            {
                "layer": layer,
                "checkpoint_layer": int(CHECKPOINT_LAYERS[layer]),
                "covered_dependent_uas": dev_uas,
                "covered_dependent_n": count,
            }
        )
        print(
            f"DEV ATTACH RAW: layer={layer} checkpoint={CHECKPOINT_LAYERS[layer]} "
            f"covered-UAS={dev_uas:.6f} n={count}",
            flush=True,
        )
    raw_layer = select_dev_layer(
        {row["layer"]: row["covered_dependent_uas"] for row in raw_curve}
    )

    white_curve: list[dict[str, Any]] = []
    scores: dict[tuple[str, int], float] = {}
    for variant in VARIANTS:
        for layer in LAYER_INDICES:
            train_white = WhitenedResidualProvider(
                train_provider, transforms[(variant, layer)]
            )
            dev_white = WhitenedResidualProvider(
                dev_provider, transforms[(variant, layer)]
            )
            scorer = GroundedAttachmentScorer(config)
            scorer.fit(grounded_train, train_white, train_heads, layer)
            dev_uas, count = evaluate_attach_dev_candidate(
                dev, dev_heads, dev_pos, l0_student, scorer, dev_white, layer
            )
            scores[(variant, layer)] = dev_uas
            white_curve.append(
                {
                    "variant": variant,
                    "layer": layer,
                    "checkpoint_layer": int(CHECKPOINT_LAYERS[layer]),
                    "covered_dependent_uas": dev_uas,
                    "covered_dependent_n": count,
                }
            )
            print(
                f"DEV ATTACH WHITE: variant={variant} layer={layer} "
                f"checkpoint={CHECKPOINT_LAYERS[layer]} covered-UAS={dev_uas:.6f} "
                f"n={count}",
                flush=True,
            )
    return raw_curve, raw_layer, white_curve, select_variant_layer(scores)


def _variant_summary(
    label_curve: Sequence[Mapping[str, Any]],
    attach_curve: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "labeler_best_dev_by_variant": {
            variant: max(
                float(row["deprel_only_accuracy"])
                for row in label_curve
                if row["variant"] == variant
            )
            for variant in VARIANTS
        },
        "attach_best_dev_by_variant": {
            variant: max(
                float(row["covered_dependent_uas"])
                for row in attach_curve
                if row["variant"] == variant
            )
            for variant in VARIANTS
        },
        "interpretation": (
            "V1 gain is conditioning; V2 gain is information-removal-helps, per signed risk 7"
        ),
    }


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Run TRAIN fits and DEV sweeps before one shared guarded TEST read."""
    hasher = hashlib.sha256()
    print("SEED: 0 (single deterministic seed).", flush=True)
    train_provider = GroundedResidualProvider(GROUND_PATHS["train"])
    dev_provider = GroundedResidualProvider(GROUND_PATHS["dev"])
    transforms, geometry = _fit_transforms_and_geometry(train_provider)

    train = load_split(DATA_ROOT, "train", hasher)
    train_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "train.jsonl", hasher
    )
    train_heads = head_deprel_by_sent(train_head_records)
    grounded_train = [
        sentence for sentence in train if sentence.sent_id in train_provider.sentence_ids
    ]
    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    case_student = GermanR1Student(registers)
    case_student.fit(train)
    train_pos = _predicted_pos_by_sentence(case_student, train)

    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    dev_heads = head_deprel_by_sent(dev_head_records)
    dev_pos = _predicted_pos_by_sentence(case_student, dev)
    window, dev_coverage, coverage_curve = choose_window(dev_head_records)
    l0_student = GermanR3DependencyStudent(window)
    l0_student.fit(train, train_heads, train_pos)

    label_config = RelationLabelerConfig(seed=SEED)
    attach_config = BilinearConfig(seed=SEED)
    surface_labeler = BilinearRelationLabeler(label_config)
    surface_labeler.fit(train, train_heads, train_pos)
    surface_scorer = BilinearAttachmentScorer(attach_config)
    surface_scorer.fit(train, train_heads, train_pos)

    raw_label_curve, raw_label_layer, label_curve, label_selected = _labeler_sweeps(
        grounded_train,
        train_heads,
        train_provider,
        transforms,
        dev,
        dev_heads,
        dev_pos,
        dev_provider,
        l0_student,
        label_config,
    )
    label_variant, label_layer = label_selected
    print(
        f"DEV LABEL SELECTED: variant={label_variant} layer={label_layer} "
        f"checkpoint={CHECKPOINT_LAYERS[label_layer]}; tie-break=complexity then layer.",
        flush=True,
    )
    raw_attach_curve, raw_attach_layer, attach_curve, attach_selected = (
        _attachment_sweeps(
            grounded_train,
            train_heads,
            train_provider,
            transforms,
            dev,
            dev_heads,
            dev_pos,
            dev_provider,
            l0_student,
            attach_config,
        )
    )
    attach_variant, attach_layer = attach_selected
    print(
        f"DEV ATTACH SELECTED: variant={attach_variant} layer={attach_layer} "
        f"checkpoint={CHECKPOINT_LAYERS[attach_layer]}; tie-break=complexity then layer.",
        flush=True,
    )

    raw_labeler = GroundedRelationLabeler(label_config)
    raw_labeler.fit(grounded_train, train_provider, train_heads, raw_label_layer)
    white_train_label = WhitenedResidualProvider(
        train_provider, transforms[(label_variant, label_layer)]
    )
    whitened_labeler = GroundedRelationLabeler(label_config)
    whitened_labeler.fit(
        grounded_train, white_train_label, train_heads, label_layer
    )
    raw_scorer = GroundedAttachmentScorer(attach_config)
    raw_scorer.fit(grounded_train, train_provider, train_heads, raw_attach_layer)
    white_train_attach = WhitenedResidualProvider(
        train_provider, transforms[(attach_variant, attach_layer)]
    )
    whitened_scorer = GroundedAttachmentScorer(attach_config)
    whitened_scorer.fit(
        grounded_train, white_train_attach, train_heads, attach_layer
    )

    # TEST residuals and both guarded label sources remain unopened until all
    # DEV selection and fresh final fitting are complete.
    test_provider = GroundedResidualProvider(GROUND_PATHS["test"])
    white_test_label = WhitenedResidualProvider(
        test_provider, transforms[(label_variant, label_layer)]
    )
    white_test_attach = WhitenedResidualProvider(
        test_provider, transforms[(attach_variant, attach_layer)]
    )
    guard = CampaignTestReadGuard()
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_head_records = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads = head_deprel_by_sent(test_head_records)
    test_pos = _predicted_pos_by_sentence(case_student, test)

    label_metrics, label_test_fraction = _combine_labeler_evaluations(
        test,
        test_heads,
        test_pos,
        l0_student,
        surface_labeler,
        raw_labeler,
        test_provider,
        raw_label_layer,
        whitened_labeler,
        white_test_label,
        label_layer,
        case_student,
    )
    attachment_metrics, conversion_metrics = evaluate_attachment_test(
        test,
        test_heads,
        test_pos,
        l0_student,
        surface_scorer,
        raw_scorer,
        test_provider,
        raw_attach_layer,
        whitened_scorer,
        white_test_attach,
        attach_layer,
        raw_labeler,
        test_provider,
        raw_label_layer,
        whitened_labeler,
        white_test_label,
        label_layer,
        case_student,
    )
    guard.assert_complete()

    dev_selection = {
        "labeler": {
            "variant": label_variant,
            "layer": label_layer,
            "checkpoint_layer": int(CHECKPOINT_LAYERS[label_layer]),
        },
        "attach": {
            "variant": attach_variant,
            "layer": attach_layer,
            "checkpoint_layer": int(CHECKPOINT_LAYERS[attach_layer]),
        },
    }
    v1_vs_v2 = _variant_summary(label_curve, attach_curve)
    read = section5_read(
        label_metrics,
        attachment_metrics,
        conversion_metrics,
        dev_selection,
        geometry,
        v1_vs_v2,
        cross_vendor_robust=False,
    )
    coverage = {
        "labeler_test_doubly_covered_fraction": label_test_fraction,
        "attach_dev_covered_dependent_fraction": covered_dependent_fraction(
            dev, dev_provider, attach_layer
        ),
        "attach_dev_doubly_covered_fraction": doubly_covered_fraction(
            dev, dev_heads, dev_provider, attach_layer
        ),
        "attach_test_covered_dependent_fraction": covered_dependent_fraction(
            test, test_provider, attach_layer
        ),
        "attach_test_doubly_covered_fraction": doubly_covered_fraction(
            test, test_heads, test_provider, attach_layer
        ),
    }
    scoreboard: dict[str, Any] = {
        "run_tag": "empirical",
        "scope": "TRAIN-fit whitening gate on richer qwen3b_mwt grounded phi",
        "serve_honest": True,
        "seed": SEED,
        "grounded_paths": {split: str(path) for split, path in GROUND_PATHS.items()},
        "checkpoint_layers": CHECKPOINT_LAYERS.tolist(),
        "whitening": {
            "fit_split": "train",
            "variants": list(VARIANTS),
            "zero_sigma_policy": "centered dimension maps to zero",
            "pca": "top-two right singular vectors of centered TRAIN phi",
        },
        "pin_a": {
            "definition": "centered participation ratio (sum sigma^2)^2/sum sigma^4",
            "sample_size": 800,
            "seed": SEED,
            "geometry": geometry,
        },
        "pin_b": (
            "L0 row softmax; covered-governor bilinear partial log-softmax; "
            "whole-row log renormalization; CLE"
        ),
        "fixed_hyperparameters": {
            "labeler": vars(label_config),
            "attach": vars(attach_config),
        },
        "l0_window": {
            "k": window,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "raw_label_dev_curve": raw_label_curve,
        "raw_label_selected_layer": raw_label_layer,
        "label_whitening_dev_curve": label_curve,
        "raw_attach_dev_curve": raw_attach_curve,
        "raw_attach_selected_layer": raw_attach_layer,
        "attach_whitening_dev_curve": attach_curve,
        "dev_selection": dev_selection,
        "coverage": coverage,
        "labeler_arms": label_metrics,
        "attachment_arms": attachment_metrics,
        "conversion": conversion_metrics,
        "read": read,
        "test_read_counts": dict(guard.counts),
        "test_read_once": True,
        "data_hash_sha256": hasher.hexdigest(),
    }
    return scoreboard, guard


def _print_label_row(name: str, metric: Mapping[str, Any]) -> None:
    pairwise = metric["partition"]["pairwise_sensitive"]
    local = metric["partition"]["local"]
    print(
        f"{name:<12} {metric['uas']:.6f} {metric['las_strict']:.6f} "
        f"{metric['las_coarse']:.6f} {metric['deprel_only_accuracy']:.6f} "
        f"{metric['case_bearing_deprel_accuracy']:.6f} "
        f"{metric['serve_honest_morph_case']:.6f} "
        f"{pairwise['accuracy']:.6f} {local['accuracy']:.6f}"
    )


def print_scoreboard(scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard) -> None:
    """Print all preregistered curves, tables, geometry, and the one read."""
    print("\nPIN A — CENTERED PARTICIPATION-RATIO EFFRANK (n=800, seed=0)")
    print("layer checkpoint raw       v1        v2_k1     v2_k2")
    geometry = scoreboard["pin_a"]["geometry"]
    for layer in LAYER_INDICES:
        row = geometry[str(layer)]
        print(
            f"{layer:<5} {CHECKPOINT_LAYERS[layer]:<10} "
            f"{row['raw']['effrank']:<9.3f} {row['v1']['effrank']:<9.3f} "
            f"{row['v2_k1']['effrank']:<9.3f} {row['v2_k2']['effrank']:<9.3f}"
        )

    print("\nDEV LABEL WHITENING SWEEP — deprel-only accuracy")
    for row in scoreboard["label_whitening_dev_curve"]:
        print(
            f"variant={row['variant']} layer={row['layer']} "
            f"checkpoint={row['checkpoint_layer']} accuracy="
            f"{row['deprel_only_accuracy']:.6f} n={row['doubly_covered_n']}"
        )
    label_selection = scoreboard["dev_selection"]["labeler"]
    print(
        f"selected labeler: variant={label_selection['variant']} "
        f"layer={label_selection['layer']} checkpoint={label_selection['checkpoint_layer']}"
    )

    print("\nDEV ATTACH WHITENING SWEEP — covered-dependent UAS")
    for row in scoreboard["attach_whitening_dev_curve"]:
        print(
            f"variant={row['variant']} layer={row['layer']} "
            f"checkpoint={row['checkpoint_layer']} UAS="
            f"{row['covered_dependent_uas']:.6f} n={row['covered_dependent_n']}"
        )
    attach_selection = scoreboard["dev_selection"]["attach"]
    print(
        f"selected attach: variant={attach_selection['variant']} "
        f"layer={attach_selection['layer']} checkpoint={attach_selection['checkpoint_layer']}"
    )

    print("\nGSD TEST — LABELER MATCHED ARMS")
    print(
        "arm          UAS      LAS      coarse   deprel   case-rel case     pairwise local"
    )
    for arm in LABEL_ARMS:
        _print_label_row(arm, scoreboard["labeler_arms"][arm])

    print("\nGSD TEST — ATTACH MATCHED ARMS")
    print("arm          covered-UAS covered-n full-UAS  full-n")
    for arm in ATTACH_ARMS:
        metric = scoreboard["attachment_arms"][arm]
        print(
            f"{arm:<12} {metric['covered']['uas']:.6f} "
            f"{metric['covered']['tokens']:<9} {metric['full']['uas']:.6f} "
            f"{metric['full']['tokens']}"
        )

    print("\nCONVERSION — LAS on covered dependents; case on full eligible sentences")
    print("arm                    LAS      case     case-n")
    for arm in CONVERSION_ARMS:
        metric = scoreboard["conversion"][arm]
        print(
            f"{arm:<22} {metric['las_strict']:.6f} "
            f"{metric['serve_honest_morph_case']:.6f} "
            f"{metric['serve_honest_morph_case_n']}"
        )

    print("\nTHE PRE-REGISTERED SECTION 5 READ")
    print(scoreboard["read"]["text"])
    print(json.dumps(scoreboard["read"], sort_keys=True))
    guard.assert_complete()
    print("TEST READ ONCE: CONFIRMED")


def main() -> None:
    scoreboard, guard = build_scoreboard()
    OUTPUT_PATH.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print_scoreboard(scoreboard, guard)


if __name__ == "__main__":
    main()
