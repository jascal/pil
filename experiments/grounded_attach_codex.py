"""Grounded-residual bilinear attachment gate with a guarded TEST read.

Attachment and relation-label checkpoint layers are selected independently on
DEV.  Dependency conversion LAS is scored on the arm-independent
covered-dependent set: this is the only numerator consistent with the signed
attachment gate's matched-arc discipline.  Case retains #121's full-sentence
eligible-case convention.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
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
from experiments.biaffine_labeler_codex import RelationLabelerConfig  # noqa: E402
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
    resolve_governor_index,
    run_predicted_case_sentence,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    aligned_head_record,
    head_deprel_by_sent,
)
from experiments.grounded_labeler_codex import (  # noqa: E402
    CHECKPOINT_LAYERS,
    FEATURE_DIM,
    GROUND_PATHS,
    LAYER_INDICES,
    GroundedRelationLabeler,
    GroundedResidualProvider,
    evaluate_dev_layer,
    select_dev_layer,
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "grounded_attach_codex.json"
ARM_NAMES = ("l0", "surface119", "grounded")
CONVERSION_NAMES = ("baseline", "#121", "full_grounded")


class GroundedAttachmentScorer:
    """Low-rank ``phi(i)^T (U @ V.T) phi(j) + b`` grounded head scorer."""

    feature_dim = FEATURE_DIM

    def __init__(self, config: BilinearConfig | None = None) -> None:
        self.config = config or BilinearConfig()
        if self.config.rank < 1:
            raise ValueError("rank must be positive")
        if self.config.epochs < 1:
            raise ValueError("epochs must be positive")
        self._rng = np.random.default_rng(self.config.seed)
        scale = 1.0 / np.sqrt(self.feature_dim)
        self.U = self._rng.normal(
            0.0, scale, (self.feature_dim, self.config.rank)
        )
        self.V = self._rng.normal(
            0.0, scale, (self.feature_dim, self.config.rank)
        )
        self.b = 0.0
        self.epoch_losses: list[float] = []
        self.training_sentence_count = 0
        self.training_arc_count = 0

    @property
    def W(self) -> np.ndarray:
        return self.U @ self.V.T

    @staticmethod
    def _eligible_dependents(
        sentence: Sentence,
        head: HeadDeprelRecord,
        provider: GroundedResidualProvider,
    ) -> tuple[tuple[int, int], ...]:
        arcs: list[tuple[int, int]] = []
        size = len(sentence.tokens)
        for index, offset in enumerate(head.head_offset):
            if offset == 0:
                continue
            governor = resolve_governor_index(index, offset, size)
            if (
                governor is not None
                and governor != index
                and provider.covered(sentence.sent_id, index)
                and provider.covered(sentence.sent_id, governor)
            ):
                arcs.append((index, governor))
        return tuple(arcs)

    def score_edge(
        self,
        provider: GroundedResidualProvider,
        sent_id: str,
        dependent: int,
        governor: int,
        layer: int,
    ) -> float:
        dependent_feature = provider.residual(sent_id, dependent, layer)
        governor_feature = provider.residual(sent_id, governor, layer)
        return float((dependent_feature @ self.U) @ (governor_feature @ self.V) + self.b)

    def _train_sentence(
        self,
        sent_id: str,
        arcs: Sequence[tuple[int, int]],
        covered: Sequence[int],
        provider: GroundedResidualProvider,
        layer: int,
    ) -> float:
        dependent_indices = [dependent for dependent, _governor in arcs]
        dependent = np.stack(
            [provider.residual(sent_id, index, layer) for index in dependent_indices]
        )
        governor = np.stack(
            [provider.residual(sent_id, index, layer) for index in covered]
        )
        covered_column = {token_index: column for column, token_index in enumerate(covered)}
        targets = np.asarray(
            [covered_column[governor_index] for _dependent, governor_index in arcs],
            dtype=np.int64,
        )
        dependent_projection = dependent @ self.U
        governor_projection = governor @ self.V
        scores = dependent_projection @ governor_projection.T + self.b
        for row, dependent_index in enumerate(dependent_indices):
            scores[row, covered_column[dependent_index]] = -np.inf

        maxima = np.max(scores, axis=1, keepdims=True)
        exponentiated = np.exp(scores - maxima)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        example_count = len(arcs)
        loss = float(
            np.mean(
                maxima[:, 0]
                + np.log(exponentiated.sum(axis=1))
                - scores[np.arange(example_count), targets]
            )
        )

        score_gradient = probabilities
        score_gradient[np.arange(example_count), targets] -= 1.0
        score_gradient /= example_count
        grad_u = dependent.T @ (score_gradient @ governor_projection)
        grad_v = governor.T @ (score_gradient.T @ dependent_projection)
        grad_u += self.config.l2_penalty * self.U
        grad_v += self.config.l2_penalty * self.V
        grad_norm = np.sqrt(np.square(grad_u).sum() + np.square(grad_v).sum())
        if grad_norm > 5.0:
            grad_u *= 5.0 / grad_norm
            grad_v *= 5.0 / grad_norm
        self.U -= self.config.learning_rate * grad_u
        self.V -= self.config.learning_rate * grad_v
        self.b -= self.config.learning_rate * float(score_gradient.sum())
        return loss

    def fit(
        self,
        train_sentences: Sequence[Sentence],
        provider: GroundedResidualProvider,
        train_heads: Mapping[str, HeadDeprelRecord],
        layer: int,
    ) -> None:
        """Fit gold non-root arcs whose endpoints both have grounded rows."""
        provider._validate_layer(layer)
        if not train_sentences:
            raise ValueError("cannot fit on empty grounded training data")
        eligible: list[
            tuple[Sentence, tuple[tuple[int, int], ...], tuple[int, ...]]
        ] = []
        for sentence in train_sentences:
            head = aligned_head_record(sentence, train_heads)
            if not isinstance(head, HeadDeprelRecord):
                raise TypeError("labeled dependency record required")
            covered = tuple(
                index
                for index in range(len(sentence.tokens))
                if provider.covered(sentence.sent_id, index)
            )
            arcs = self._eligible_dependents(sentence, head, provider)
            if arcs:
                eligible.append((sentence, arcs, covered))
        self.training_sentence_count = len(eligible)
        self.training_arc_count = sum(len(arcs) for _sentence, arcs, _covered in eligible)
        if not self.training_arc_count:
            raise ValueError(
                f"no train sentence yielded a covered non-root arc at layer {layer}"
            )

        self.epoch_losses = []
        for _epoch in range(self.config.epochs):
            losses: list[float] = []
            for sentence_index in self._rng.permutation(len(eligible)):
                sentence, arcs, covered = eligible[int(sentence_index)]
                losses.append(
                    self._train_sentence(
                        sentence.sent_id, arcs, covered, provider, layer
                    )
                )
            self.epoch_losses.append(float(np.mean(losses)))


def mixed_edge_scores(
    bilinear: GroundedAttachmentScorer,
    l0_student: GermanR3DependencyStudent,
    provider: GroundedResidualProvider,
    sent_id: str,
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    layer: int,
) -> np.ndarray:
    """Overwrite only doubly-grounded token-token cells of the L0 score matrix."""
    provider._validate_layer(layer)
    scores = l0_edge_scores(l0_student, tokens, predicted_pos)
    covered = tuple(
        index for index in range(len(tokens)) if provider.covered(sent_id, index)
    )
    for dependent in covered:
        for governor in covered:
            if dependent != governor:
                scores[dependent, governor + 1] = bilinear.score_edge(
                    provider, sent_id, dependent, governor, layer
                )
    for index in range(len(tokens)):
        scores[index, index + 1] = -np.inf
    return scores


def covered_dependent_indices(
    sent_id: str,
    token_count: int,
    provider: GroundedResidualProvider,
    layer: int,
) -> tuple[int, ...]:
    """Return all and only grounded dependent-token indices."""
    provider._validate_layer(layer)
    return tuple(
        index for index in range(token_count) if provider.covered(sent_id, index)
    )


def covered_dependent_fraction(
    sentences: Sequence[Sentence],
    provider: GroundedResidualProvider,
    layer: int,
) -> float:
    covered = sum(
        len(covered_dependent_indices(sentence.sent_id, len(sentence.tokens), provider, layer))
        for sentence in sentences
    )
    total = sum(len(sentence.tokens) for sentence in sentences)
    if not total:
        raise ValueError("cannot compute covered-dependent fraction on an empty split")
    return covered / total


def doubly_covered_fraction(
    sentences: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    provider: GroundedResidualProvider,
    layer: int,
) -> float:
    """Return trainable doubly-covered gold arcs / all non-root gold arcs."""
    provider._validate_layer(layer)
    covered = total = 0
    for sentence in sentences:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        for index, offset in enumerate(head.head_offset):
            if offset == 0:
                continue
            total += 1
            governor = resolve_governor_index(index, offset, len(sentence.tokens))
            covered += bool(
                governor is not None
                and provider.covered(sentence.sent_id, index)
                and provider.covered(sentence.sent_id, governor)
            )
    if not total:
        raise ValueError("split has no non-root gold arcs")
    return covered / total


def grounded_geometry(
    provider: GroundedResidualProvider,
    layer: int,
    sample_size: int = 500,
    seed: int = 0,
) -> dict[str, float | int]:
    """Measure pairwise cosine and centered-spectrum participation ratio."""
    provider._validate_layer(layer)
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    keys = sorted(provider._row_by_key)
    if not keys:
        raise ValueError("grounded provider has no rows")
    count = min(sample_size, len(keys))
    rng = np.random.default_rng(seed)
    chosen = (
        np.arange(len(keys), dtype=np.int64)
        if count == len(keys)
        else np.sort(rng.choice(len(keys), size=count, replace=False))
    )
    sample = np.stack(
        [provider.residual(*keys[int(index)], layer) for index in chosen]
    )
    norms = np.linalg.norm(sample, axis=1)
    normalized = np.divide(
        sample,
        norms[:, None],
        out=np.zeros_like(sample),
        where=norms[:, None] > 0.0,
    )
    if count > 1:
        cosine_matrix = normalized @ normalized.T
        cosine = float(cosine_matrix[np.triu_indices(count, k=1)].mean())
    else:
        cosine = 0.0
    singular_values = np.linalg.svd(sample - sample.mean(axis=0), compute_uv=False)
    squared = np.square(singular_values)
    denominator = float(np.square(squared).sum())
    effrank = float(np.square(squared.sum()) / denominator) if denominator else 1.0
    return {"cosine": cosine, "effrank": effrank, "sample_size": count}


def evaluate_attach_dev_layer(
    dev: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    scorer: GroundedAttachmentScorer,
    provider: GroundedResidualProvider,
    layer: int,
) -> tuple[float, int]:
    predicted: list[int] = []
    gold: list[int] = []
    for sentence in dev:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        offsets = governor_columns_to_offsets(
            chu_liu_edmonds(
                mixed_edge_scores(
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
        raise ValueError(f"dev has no covered dependents at layer {layer}")
    return sum(left == right for left, right in zip(predicted, gold, strict=True)) / len(gold), len(gold)


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


def evaluate_test_arms(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    surface_scorer: BilinearAttachmentScorer,
    grounded_scorer: GroundedAttachmentScorer,
    provider: GroundedResidualProvider,
    attach_layer: int,
    grounded_labeler: GroundedRelationLabeler,
    label_layer: int,
    case_student: GermanR1Student,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Evaluate three head arms and the three-point LAS/case conversion."""
    covered_values = {
        arm: {"head": [], "deprel": []} for arm in ARM_NAMES
    }
    full_values = {arm: {"head": [], "deprel": []} for arm in ARM_NAMES}
    conversion_values = {
        arm: {"head": [], "deprel": [], "case": []}
        for arm in CONVERSION_NAMES
    }
    covered_gold_head: list[int] = []
    covered_gold_deprel: list[str] = []
    full_gold_head: list[int] = []
    full_gold_deprel: list[str] = []
    gold_case: list[str] = []

    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        l0_heads = l0_student.predict_sentence(sentence.tokens, pos).head_offset
        surface_heads = governor_columns_to_offsets(
            chu_liu_edmonds(surface_scorer.score_matrix(sentence.tokens, pos))
        )
        grounded_heads = governor_columns_to_offsets(
            chu_liu_edmonds(
                mixed_edge_scores(
                    grounded_scorer,
                    l0_student,
                    provider,
                    sentence.sent_id,
                    sentence.tokens,
                    pos,
                    attach_layer,
                )
            )
        )
        head_predictions = {
            "l0": l0_heads,
            "surface119": surface_heads,
            "grounded": grounded_heads,
        }
        head_labels = {
            arm: predict_deprels(l0_student, sentence.tokens, pos, offsets)
            for arm, offsets in head_predictions.items()
        }
        selected = covered_dependent_indices(
            sentence.sent_id, len(sentence.tokens), provider, attach_layer
        )
        selected_by_arm = {arm: selected for arm in ARM_NAMES}
        assert len(set(selected_by_arm.values())) == 1, (
            "attachment arms do not share an identical covered-dependent index set"
        )
        for arm in ARM_NAMES:
            covered_values[arm]["head"].extend(
                head_predictions[arm][index] for index in selected_by_arm[arm]
            )
            covered_values[arm]["deprel"].extend(
                head_labels[arm][index] for index in selected_by_arm[arm]
            )
            full_values[arm]["head"].extend(head_predictions[arm])
            full_values[arm]["deprel"].extend(head_labels[arm])

        baseline_labels = head_labels["l0"]
        labels_121 = grounded_labeler.predict(
            sentence.sent_id,
            sentence.tokens,
            pos,
            l0_heads,
            provider,
            label_layer,
            l0_student,
        )
        full_labels = grounded_labeler.predict(
            sentence.sent_id,
            sentence.tokens,
            pos,
            grounded_heads,
            provider,
            label_layer,
            l0_student,
        )
        conversions = {
            "baseline": (l0_heads, baseline_labels),
            "#121": (l0_heads, labels_121),
            "full_grounded": (grounded_heads, full_labels),
        }
        case_baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        case_eligible = [
            index
            for index, label in enumerate(sentence.targets["morph_case"])
            if label != "-"
        ]
        for arm, (offsets, labels) in conversions.items():
            case_result = run_predicted_case_sentence(
                case_student,
                sentence.tokens,
                case_baseline,
                pos,
                offsets,
                labels,
                case_eligible,
            )
            conversion_values[arm]["head"].extend(offsets[index] for index in selected)
            conversion_values[arm]["deprel"].extend(labels[index] for index in selected)
            conversion_values[arm]["case"].extend(
                case_result.predictions[index] for index in case_eligible
            )

        covered_gold_head.extend(head.head_offset[index] for index in selected)
        covered_gold_deprel.extend(head.deprel[index] for index in selected)
        full_gold_head.extend(head.head_offset)
        full_gold_deprel.extend(head.deprel)
        gold_case.extend(sentence.targets["morph_case"][index] for index in case_eligible)

    if not covered_gold_head:
        raise ValueError("test split has no covered-dependent arcs")
    attach_metrics: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        attach_metrics[arm] = {
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
    assert {
        int(attach_metrics[arm]["covered"]["tokens"]) for arm in ARM_NAMES
    } == {len(covered_gold_head)}, "covered-dependent scorer denominators differ"

    conversion_metrics: dict[str, dict[str, Any]] = {}
    for arm in CONVERSION_NAMES:
        dependency = _score_dependency_arm(
            conversion_values[arm]["head"],
            conversion_values[arm]["deprel"],
            covered_gold_head,
            covered_gold_deprel,
        )
        conversion_metrics[arm] = {
            **dependency,
            "serve_honest_morph_case": accuracy(
                conversion_values[arm]["case"], gold_case
            ),
            "serve_honest_morph_case_n": len(gold_case),
        }
    return attach_metrics, conversion_metrics


def preregistered_read(
    uas_l0: float,
    uas_surface119: float,
    uas_grounded: float,
    las_base: float,
    las_121: float,
    las_full: float,
    case_base: float,
    case_121: float,
    case_full: float,
    dev_layer: int,
    covered_dependent_fraction: float,
    doubly_covered_fraction: float,
    c7_cosine: float,
    c7_effrank: float,
) -> dict[str, Any]:
    """Apply signed section 5 once, including its stated case failure mode."""
    uas_gain_l0 = uas_grounded - uas_l0
    uas_gain_surface = uas_grounded - uas_surface119
    las_gain = las_full - las_base
    case_lift = case_full - case_base
    if uas_grounded <= uas_l0 and uas_grounded <= uas_surface119:
        verdict = "HALTED"
        reason = "grounded attachment does not exceed L0 or surface-phi #119"
    elif (
        uas_gain_l0 >= 0.03
        and uas_grounded > uas_surface119
        and las_gain >= 0.03
        and case_full > case_base
    ):
        verdict = "FIRES"
        reason = "all preregistered attachment and conversion conditions are met"
    else:
        verdict = "IN-BETWEEN"
        failed: list[str] = []
        if 0.0 < uas_gain_l0 < 0.03:
            failed.append("UAS lift over L0 is below +0.03")
        if uas_grounded > uas_l0 and uas_grounded <= uas_surface119:
            failed.append("grounded does not beat surface-phi #119")
        if uas_gain_l0 > 0.0 and las_gain < 0.03:
            failed.append("conversion LAS gain is below +0.03")
        if case_full <= case_base:
            failed.append("case does not improve over baseline")
        reason = "; ".join(failed) or "the FIRES conjunction is not met"

    if case_full >= 0.90:
        through_line = "met"
    elif case_full > 0.76:
        through_line = "plateau language"
    else:
        through_line = "diagnose"
    text = (
        f"{verdict}: grounded UAS={uas_grounded:.6f}, L0={uas_l0:.6f} "
        f"(gain={uas_gain_l0:+.6f}), surface-phi #119={uas_surface119:.6f} "
        f"(gain={uas_gain_surface:+.6f}); covered-set conversion LAS "
        f"baseline={las_base:.6f}, #121={las_121:.6f}, full={las_full:.6f} "
        f"(full-baseline={las_gain:+.6f}); case baseline={case_base:.6f}, "
        f"#121={case_121:.6f}, full={case_full:.6f} (lift={case_lift:+.6f}); "
        f"DEV attach layer={dev_layer}; covered-dependent fraction="
        f"{covered_dependent_fraction:.6f}, doubly-covered fraction="
        f"{doubly_covered_fraction:.6f}; C7 cosine={c7_cosine:.6f}, "
        f"effrank={c7_effrank:.3f}; through-line={through_line}. {reason}."
    )
    return {
        "verdict": verdict,
        "uas_l0": uas_l0,
        "uas_surface119": uas_surface119,
        "uas_grounded": uas_grounded,
        "uas_gain_vs_l0": uas_gain_l0,
        "uas_gain_vs_surface119": uas_gain_surface,
        "las_base": las_base,
        "las_121": las_121,
        "las_full": las_full,
        "las_full_gain_vs_base": las_gain,
        "case_base": case_base,
        "case_121": case_121,
        "case_full": case_full,
        "case_lift": case_lift,
        "dev_layer": dev_layer,
        "covered_dependent_fraction": covered_dependent_fraction,
        "doubly_covered_fraction": doubly_covered_fraction,
        "c7_cosine": c7_cosine,
        "c7_effrank": c7_effrank,
        "through_line": through_line,
        "text": text,
    }


def _sweep_attach_layers(
    grounded_train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
    train_provider: GroundedResidualProvider,
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    dev_provider: GroundedResidualProvider,
    l0_student: GermanR3DependencyStudent,
    config: BilinearConfig,
) -> tuple[list[dict[str, Any]], int]:
    curve: list[dict[str, Any]] = []
    for layer in LAYER_INDICES:
        scorer = GroundedAttachmentScorer(config)
        scorer.fit(grounded_train, train_provider, train_heads, layer)
        dev_uas, dev_n = evaluate_attach_dev_layer(
            dev, dev_heads, dev_pos, l0_student, scorer, dev_provider, layer
        )
        row = {
            "layer": layer,
            "checkpoint_layer": int(CHECKPOINT_LAYERS[layer]),
            "dev_uas": dev_uas,
            "covered_dependent_fraction": covered_dependent_fraction(
                dev, dev_provider, layer
            ),
            "doubly_covered_fraction": doubly_covered_fraction(
                dev, dev_heads, dev_provider, layer
            ),
            "covered_dependent_n": dev_n,
            "training_sentences": scorer.training_sentence_count,
            "training_arcs": scorer.training_arc_count,
            "epoch_losses": scorer.epoch_losses,
        }
        curve.append(row)
        print(
            f"DEV ATTACH LAYER: L={layer} checkpoint={CHECKPOINT_LAYERS[layer]} "
            f"UAS={dev_uas:.6f} covered={row['covered_dependent_fraction']:.6f} "
            f"doubly={row['doubly_covered_fraction']:.6f} n={dev_n}",
            flush=True,
        )
    selected = select_dev_layer({row["layer"]: row["dev_uas"] for row in curve})
    return curve, selected


def _sweep_label_layers(
    grounded_train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
    train_provider: GroundedResidualProvider,
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    dev_provider: GroundedResidualProvider,
    l0_student: GermanR3DependencyStudent,
    config: RelationLabelerConfig,
) -> tuple[list[dict[str, Any]], int]:
    curve: list[dict[str, Any]] = []
    for layer in LAYER_INDICES:
        labeler = GroundedRelationLabeler(config)
        labeler.fit(grounded_train, train_provider, train_heads, layer)
        dev_accuracy, dev_fraction, dev_n = evaluate_dev_layer(
            dev,
            dev_heads,
            dev_pos,
            l0_student,
            labeler,
            dev_provider,
            layer,
        )
        curve.append(
            {
                "layer": layer,
                "checkpoint_layer": int(CHECKPOINT_LAYERS[layer]),
                "deprel_only_accuracy": dev_accuracy,
                "doubly_covered_fraction": dev_fraction,
                "doubly_covered_n": dev_n,
                "training_sentences": labeler.training_sentence_count,
                "training_arcs": labeler.training_arc_count,
                "epoch_losses": labeler.epoch_losses,
            }
        )
        print(
            f"DEV LABEL LAYER: L={layer} checkpoint={CHECKPOINT_LAYERS[layer]} "
            f"deprel-only={dev_accuracy:.6f} doubly={dev_fraction:.6f} n={dev_n}",
            flush=True,
        )
    selected = select_dev_layer(
        {row["layer"]: row["deprel_only_accuracy"] for row in curve}
    )
    return curve, selected


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Run both DEV sweeps before one shared in-memory guarded TEST load."""
    hasher = hashlib.sha256()
    train_provider = GroundedResidualProvider(GROUND_PATHS["train"])
    dev_provider = GroundedResidualProvider(GROUND_PATHS["dev"])

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

    attach_config = BilinearConfig(seed=0)
    label_config = RelationLabelerConfig(seed=0)
    surface_scorer = BilinearAttachmentScorer(attach_config)
    print("FIT: surface-phi #119 attachment scorer.", flush=True)
    surface_scorer.fit(train, train_heads, train_pos)

    print("DEV: independent grounded attachment layer sweep.", flush=True)
    attach_curve, attach_layer = _sweep_attach_layers(
        grounded_train,
        train_heads,
        train_provider,
        dev,
        dev_heads,
        dev_pos,
        dev_provider,
        l0_student,
        attach_config,
    )
    print(
        f"DEV ATTACH SELECTED: L*={attach_layer} "
        f"checkpoint={CHECKPOINT_LAYERS[attach_layer]}",
        flush=True,
    )

    print("DEV: independent #121 grounded-label layer sweep.", flush=True)
    label_curve, label_layer = _sweep_label_layers(
        grounded_train,
        train_heads,
        train_provider,
        dev,
        dev_heads,
        dev_pos,
        dev_provider,
        l0_student,
        label_config,
    )
    print(
        f"DEV LABEL SELECTED: L*={label_layer} "
        f"checkpoint={CHECKPOINT_LAYERS[label_layer]}",
        flush=True,
    )

    grounded_scorer = GroundedAttachmentScorer(attach_config)
    grounded_scorer.fit(
        grounded_train, train_provider, train_heads, attach_layer
    )
    grounded_labeler = GroundedRelationLabeler(label_config)
    grounded_labeler.fit(
        grounded_train, train_provider, train_heads, label_layer
    )
    geometry = {
        str(layer): grounded_geometry(train_provider, layer)
        for layer in LAYER_INDICES
    }

    # TEST residuals are text-derived, but are still not opened until all DEV
    # selections and fresh final fits are complete.
    test_provider = GroundedResidualProvider(GROUND_PATHS["test"])
    guard = CampaignTestReadGuard()
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_head_records = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads = head_deprel_by_sent(test_head_records)
    test_pos = _predicted_pos_by_sentence(case_student, test)
    attach_metrics, conversion_metrics = evaluate_test_arms(
        test,
        test_heads,
        test_pos,
        l0_student,
        surface_scorer,
        grounded_scorer,
        test_provider,
        attach_layer,
        grounded_labeler,
        label_layer,
        case_student,
    )
    guard.assert_complete()

    coverage = {
        "dev": {
            "covered_dependent_fraction": covered_dependent_fraction(
                dev, dev_provider, attach_layer
            ),
            "doubly_covered_fraction": doubly_covered_fraction(
                dev, dev_heads, dev_provider, attach_layer
            ),
        },
        "test": {
            "covered_dependent_fraction": covered_dependent_fraction(
                test, test_provider, attach_layer
            ),
            "doubly_covered_fraction": doubly_covered_fraction(
                test, test_heads, test_provider, attach_layer
            ),
        },
    }
    read = preregistered_read(
        float(attach_metrics["l0"]["covered"]["uas"]),
        float(attach_metrics["surface119"]["covered"]["uas"]),
        float(attach_metrics["grounded"]["covered"]["uas"]),
        float(conversion_metrics["baseline"]["las_strict"]),
        float(conversion_metrics["#121"]["las_strict"]),
        float(conversion_metrics["full_grounded"]["las_strict"]),
        float(conversion_metrics["baseline"]["serve_honest_morph_case"]),
        float(conversion_metrics["#121"]["serve_honest_morph_case"]),
        float(conversion_metrics["full_grounded"]["serve_honest_morph_case"]),
        attach_layer,
        coverage["test"]["covered_dependent_fraction"],
        coverage["test"]["doubly_covered_fraction"],
        float(geometry[str(attach_layer)]["cosine"]),
        float(geometry[str(attach_layer)]["effrank"]),
    )
    scoreboard: dict[str, Any] = {
        "run_tag": "empirical",
        "scope": "grounded-phi bilinear attachment gate on covered dependents",
        "serve_honest": True,
        "measurement": (
            "matched dependency metrics use covered dependents; diluted metrics use all "
            "tokens; case uses full-sentence eligible case indices"
        ),
        "gold_usage": (
            "train gold heads are fit targets; DEV gold selects two independent layers; "
            "TEST gold is scoring-only"
        ),
        "data_hash_sha256": hasher.hexdigest(),
        "grounded_paths": {split: str(path) for split, path in GROUND_PATHS.items()},
        "checkpoint_layers": CHECKPOINT_LAYERS.tolist(),
        "seed": 0,
        "fixed_hyperparameters": {
            "rank": attach_config.rank,
            "epochs": attach_config.epochs,
            "learning_rate": attach_config.learning_rate,
            "l2_penalty": attach_config.l2_penalty,
            "seed": attach_config.seed,
            "feature_dim": FEATURE_DIM,
            "gradient_norm_clip": 5.0,
        },
        "l0_window": {
            "k": window,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "attach_dev_layer_curve": attach_curve,
        "selected_attach_layer": attach_layer,
        "selected_attach_checkpoint_layer": int(CHECKPOINT_LAYERS[attach_layer]),
        "label_dev_layer_curve": label_curve,
        "selected_label_layer": label_layer,
        "selected_label_checkpoint_layer": int(CHECKPOINT_LAYERS[label_layer]),
        "final_grounded_attach_training": {
            "training_sentences": grounded_scorer.training_sentence_count,
            "training_arcs": grounded_scorer.training_arc_count,
            "epoch_losses": grounded_scorer.epoch_losses,
        },
        "final_grounded_label_training": {
            "training_sentences": grounded_labeler.training_sentence_count,
            "training_arcs": grounded_labeler.training_arc_count,
            "epoch_losses": grounded_labeler.epoch_losses,
        },
        "coverage": coverage,
        "c7_geometry": geometry,
        "attachment_arms": attach_metrics,
        "conversion": conversion_metrics,
        "read": read,
        "test_read_counts": dict(guard.counts),
        "test_read_once": True,
    }
    return scoreboard, guard


def print_scoreboard(
    scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard
) -> None:
    """Print the complete DEV curves, TEST tables, and signed read."""
    config = scoreboard["fixed_hyperparameters"]
    print("\nGROUNDED-PHI ATTACHMENT GATE")
    print(
        f"config: rank={config['rank']} epochs={config['epochs']} "
        f"lr={config['learning_rate']} l2={config['l2_penalty']} "
        f"seed={config['seed']} feature_dim={config['feature_dim']}"
    )
    print("DEV ATTACH LAYER SWEEP — covered-dependent UAS")
    for row in scoreboard["attach_dev_layer_curve"]:
        print(
            f"L={row['layer']} checkpoint={row['checkpoint_layer']} "
            f"UAS={row['dev_uas']:.6f} "
            f"covered={row['covered_dependent_fraction']:.6f} "
            f"doubly={row['doubly_covered_fraction']:.6f} "
            f"train_sentences={row['training_sentences']} "
            f"train_arcs={row['training_arcs']}"
        )
    print(
        f"selected attach L*={scoreboard['selected_attach_layer']} "
        f"checkpoint={scoreboard['selected_attach_checkpoint_layer']}"
    )
    print("DEV LABEL LAYER SWEEP — independent #121 criterion")
    for row in scoreboard["label_dev_layer_curve"]:
        print(
            f"L={row['layer']} checkpoint={row['checkpoint_layer']} "
            f"deprel-only={row['deprel_only_accuracy']:.6f} "
            f"doubly={row['doubly_covered_fraction']:.6f}"
        )
    print(
        f"selected label L*={scoreboard['selected_label_layer']} "
        f"checkpoint={scoreboard['selected_label_checkpoint_layer']}"
    )
    for split in ("dev", "test"):
        coverage = scoreboard["coverage"][split]
        print(
            f"{split} coverage: covered-dependent="
            f"{coverage['covered_dependent_fraction']:.6f} "
            f"doubly-covered={coverage['doubly_covered_fraction']:.6f}"
        )

    print("\nC7 GROUNDED GEOMETRY — TRAIN fixed sample")
    print("layer checkpoint cosine      effrank     sample")
    for layer in LAYER_INDICES:
        row = scoreboard["c7_geometry"][str(layer)]
        print(
            f"{layer:<5} {CHECKPOINT_LAYERS[layer]:<10} "
            f"{row['cosine']:.6f}  {row['effrank']:.3f}  {row['sample_size']}"
        )

    print("\nGSD TEST — MATCHED THREE-ARM ATTACHMENT")
    print("arm          covered-UAS covered-n  diluted-full-UAS full-n")
    for arm in ARM_NAMES:
        metric = scoreboard["attachment_arms"][arm]
        print(
            f"{arm:<12} {metric['covered']['uas']:.6f}    "
            f"{metric['covered']['tokens']:<10} "
            f"{metric['full']['uas']:.6f}         {metric['full']['tokens']}"
        )

    print("\nCONVERSION — LAS on covered dependents; case on full eligible sentences")
    print("arm             LAS-strict case       case-n")
    for arm in CONVERSION_NAMES:
        metric = scoreboard["conversion"][arm]
        print(
            f"{arm:<15} {metric['las_strict']:.6f}  "
            f"{metric['serve_honest_morph_case']:.6f}  "
            f"{metric['serve_honest_morph_case_n']}"
        )
    read = scoreboard["read"]
    print(f"case cascade lift full-baseline: {read['case_lift']:+.6f}")
    print("\nTHE PRE-REGISTERED SECTION 5 READ")
    print(read["text"])
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
