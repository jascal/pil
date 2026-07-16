"""Grounded-φ biaffine relation-labeler gate (feature-ladder rung 2, slice 2b).

Reuses the shipped low-rank biaffine label form from ``BilinearRelationLabeler``
but substitutes per-word Qwen2.5-3B grounded residuals for surface hashed
features.  Three arms share the same L0-predicted heads and are scored only on
the doubly-covered arc set.  Layer is chosen on DEV; TEST is read once under
``CampaignTestReadGuard``.  Pre-registered decision rules from
``docs/notes/grounded_labeler_prereg.md`` §5 are applied verbatim.
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
    predict_deprels,
    score_case_bearing_deprels,
    score_prediction_sequences,
)
from experiments.biaffine_labeler_codex import (  # noqa: E402
    BilinearRelationLabeler,
    RelationLabelerConfig,
    coarse_deprel,
    score_relation_partition,
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
    resolve_governor_index,
    run_predicted_case_sentence,
    score_dependencies,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    aligned_head_record,
    head_deprel_by_sent,
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "campaign_grounded_labeler.json"
GROUNDED_ROOT = Path(__file__).resolve().parents[1] / "data" / "grounded"
TRAIN_NPZ = GROUNDED_ROOT / "qwen3b_gsd_train_n600.npz"
DEV_NPZ = GROUNDED_ROOT / "qwen3b_gsd_dev_n799.npz"
TEST_NPZ = GROUNDED_ROOT / "qwen3b_gsd_test_n977.npz"

RUN_TAG = "empirical"
SCOPE = (
    "axis-B grounded-φ biaffine relation-labeler gate on doubly-covered arcs; "
    "matched unary + surface-#119 baselines"
)
CHECKPOINT_LAYERS = (8, 17, 26, 35)
FEATURE_DIM = 2048
LAYER_INDICES = (0, 1, 2, 3)
GRAD_CLIP = 5.0
CASE_BASELINE = 0.76
CASE_DELIVERABLE = 0.90
LAS_FIRES_THRESHOLD = 0.03


class GroundedFeatureProvider:
    """Read-only npz-backed lookup of per-word grounded residual vectors."""

    def __init__(self, npz_path: Path, layer_index: int) -> None:
        if layer_index not in LAYER_INDICES:
            raise ValueError(f"layer_index must be in {LAYER_INDICES}, got {layer_index}")
        data = np.load(npz_path, allow_pickle=False)
        checkpoint_layers = tuple(int(value) for value in data["checkpoint_layers"])
        if checkpoint_layers != CHECKPOINT_LAYERS:
            raise ValueError(
                f"unexpected checkpoint_layers={checkpoint_layers}; "
                f"expected {CHECKPOINT_LAYERS}"
            )
        residuals = np.asarray(data["last_residual"][:, layer_index, :], dtype=np.float64)
        sent_ids = data["sent_ids"]
        token_indices = data["token_index"]
        if residuals.ndim != 2 or residuals.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"last_residual layer slice must be (N, {FEATURE_DIM}), got {residuals.shape}"
            )
        if len(sent_ids) != len(token_indices) or len(sent_ids) != residuals.shape[0]:
            raise ValueError("sent_ids, token_index, and last_residual rows must align")
        table: dict[tuple[str, int], np.ndarray] = {}
        for row, sent_id, token_index in zip(
            residuals, sent_ids, token_indices, strict=True
        ):
            key = (str(sent_id), int(token_index))
            if key in table:
                raise ValueError(f"duplicate grounded row for {key}")
            table[key] = row
        self._table = table
        self.feature_dim = FEATURE_DIM
        self.layer_index = layer_index
        self.layer = CHECKPOINT_LAYERS[layer_index]
        self.n_rows = len(table)
        self.n_sentences = len({sent_id for sent_id, _token_index in table})

    def get(self, sent_id: str, token_index: int) -> np.ndarray | None:
        return self._table.get((sent_id, token_index))

    def covered(self, sent_id: str, token_index: int) -> bool:
        return (sent_id, token_index) in self._table


def is_doubly_covered(
    sent_id: str,
    index: int,
    predicted_offset: int,
    size: int,
    provider: GroundedFeatureProvider,
) -> bool:
    """ROOT arcs are always covered; non-root needs both endpoint φ vectors."""
    if predicted_offset == 0:
        return True
    governor = resolve_governor_index(index, predicted_offset, size)
    if governor is None:
        return False
    return provider.covered(sent_id, index) and provider.covered(sent_id, governor)


class GroundedRelationLabeler:
    """Low-rank biaffine labeler over grounded residual endpoint features.

    ROOT arcs are never scored: fit skips gold ROOT, predict emits ``"root"``
    deterministically.  Uncovered or unresolvable non-root arcs fall back to
    the caller-supplied unary labels.
    """

    def __init__(self, config: RelationLabelerConfig | None = None) -> None:
        self.config = config or RelationLabelerConfig()
        if self.config.rank < 1:
            raise ValueError("rank must be positive")
        if self.config.epochs < 1:
            raise ValueError("epochs must be positive")
        self.feature_dim = FEATURE_DIM
        self._rng = np.random.default_rng(self.config.seed)
        self.relations: tuple[str, ...] = ()
        self._relation_index: dict[str, int] = {}
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.C: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.epoch_losses: list[float] = []

    def _initialize_parameters(self, relations: Sequence[str]) -> None:
        self.relations = tuple(sorted(set(relations)))
        if not self.relations:
            raise ValueError("cannot fit without relation targets")
        self._relation_index = {
            relation: index for index, relation in enumerate(self.relations)
        }
        relation_count = len(self.relations)
        feature_scale = 1.0 / np.sqrt(self.feature_dim)
        rank_scale = 1.0 / np.sqrt(self.config.rank)
        self.U = self._rng.normal(0.0, feature_scale, (self.feature_dim, self.config.rank))
        self.V = self._rng.normal(0.0, feature_scale, (self.feature_dim, self.config.rank))
        self.C = self._rng.normal(0.0, rank_scale, (relation_count, self.config.rank))
        self.P = np.zeros((relation_count, self.feature_dim), dtype=np.float64)
        self.Q = np.zeros((relation_count, self.feature_dim), dtype=np.float64)
        self.b = np.zeros(relation_count, dtype=np.float64)

    def _parameters(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if any(value is None for value in (self.U, self.V, self.C, self.P, self.Q, self.b)):
            raise RuntimeError("relation labeler used before fit")
        assert self.U is not None
        assert self.V is not None
        assert self.C is not None
        assert self.P is not None
        assert self.Q is not None
        assert self.b is not None
        return self.U, self.V, self.C, self.P, self.Q, self.b

    @staticmethod
    def _logits(
        dependent_features: np.ndarray,
        governor_features: np.ndarray,
        parameters: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u, v, core, dependent_affine, governor_affine, bias = parameters
        dependent_projection = dependent_features @ u
        governor_projection = governor_features @ v
        interaction = dependent_projection * governor_projection
        logits = (
            interaction @ core.T
            + dependent_features @ dependent_affine.T
            + governor_features @ governor_affine.T
            + bias
        )
        return logits, dependent_projection, governor_projection, interaction

    def _train_arc_batch(
        self,
        dependent_features: np.ndarray,
        governor_features: np.ndarray,
        gold_deprels: Sequence[str],
    ) -> float:
        batch_size = dependent_features.shape[0]
        if batch_size == 0:
            raise ValueError("cannot train on an empty arc batch")
        if governor_features.shape[0] != batch_size or len(gold_deprels) != batch_size:
            raise ValueError("arc-batch fields must align")
        parameters = self._parameters()
        u, v, core, dependent_affine, governor_affine, _bias = parameters
        logits, dependent_projection, governor_projection, interaction = self._logits(
            dependent_features, governor_features, parameters
        )
        targets = np.asarray(
            [self._relation_index[label] for label in gold_deprels], dtype=np.int64
        )
        maxima = np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(logits - maxima)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        loss = float(
            np.mean(
                maxima[:, 0]
                + np.log(exponentiated.sum(axis=1))
                - logits[np.arange(batch_size), targets]
            )
        )

        score_gradient = probabilities
        score_gradient[np.arange(batch_size), targets] -= 1.0
        score_gradient /= batch_size
        interaction_gradient = score_gradient @ core
        grad_u = dependent_features.T @ (interaction_gradient * governor_projection)
        grad_v = governor_features.T @ (interaction_gradient * dependent_projection)
        grad_core = score_gradient.T @ interaction
        grad_p = score_gradient.T @ dependent_features
        grad_q = score_gradient.T @ governor_features
        grad_b = score_gradient.sum(axis=0)

        penalty = self.config.l2_penalty
        grad_u += penalty * u
        grad_v += penalty * v
        grad_core += penalty * core
        grad_p += penalty * dependent_affine
        grad_q += penalty * governor_affine
        gradients = (grad_u, grad_v, grad_core, grad_p, grad_q, grad_b)
        grad_norm = np.sqrt(sum(float(np.square(gradient).sum()) for gradient in gradients))
        if grad_norm > GRAD_CLIP:
            scale = GRAD_CLIP / grad_norm
            gradients = tuple(gradient * scale for gradient in gradients)

        learning_rate = self.config.learning_rate
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter -= learning_rate * gradient
        return loss

    @staticmethod
    def _kept_arcs_for_sentence(
        sent_id: str,
        gold_offsets: Sequence[int],
        gold_deprels: Sequence[str],
        provider: GroundedFeatureProvider,
    ) -> list[tuple[int, int, str]]:
        size = len(gold_offsets)
        if len(gold_deprels) != size:
            raise ValueError("gold heads and deprels must align")
        kept: list[tuple[int, int, str]] = []
        for index, offset in enumerate(gold_offsets):
            if offset == 0:
                continue
            governor = resolve_governor_index(index, offset, size)
            if governor is None or governor == index:
                raise ValueError(f"invalid training head at token {index}: {offset}")
            if provider.covered(sent_id, index) and provider.covered(sent_id, governor):
                kept.append((index, governor, gold_deprels[index]))
        return kept

    def fit(
        self,
        train_sentences: Sequence[Sentence],
        train_heads: Mapping[str, HeadDeprelRecord],
        train_predicted_pos: Mapping[str, Sequence[str]],
        provider: GroundedFeatureProvider,
    ) -> None:
        """Fit on gold non-ROOT arcs whose dependent and governor are both covered."""
        del train_predicted_pos  # grounded φ does not use POS; kept for API symmetry
        if not train_sentences:
            raise ValueError("cannot fit on empty training data")
        sentence_arcs: list[list[tuple[int, int, str]]] = []
        relations: list[str] = []
        for sentence in train_sentences:
            head = aligned_head_record(sentence, train_heads)
            if not isinstance(head, HeadDeprelRecord):
                raise TypeError("labeled dependency record required")
            kept = self._kept_arcs_for_sentence(
                sentence.sent_id, head.head_offset, head.deprel, provider
            )
            sentence_arcs.append(kept)
            relations.extend(label for _index, _governor, label in kept)
        if not relations:
            raise ValueError("kept arc set is empty; cannot fit grounded labeler")
        self._initialize_parameters(relations)
        self.epoch_losses = []
        for _epoch in range(self.config.epochs):
            order = self._rng.permutation(len(train_sentences))
            losses: list[float] = []
            for sentence_index in order:
                sentence = train_sentences[int(sentence_index)]
                kept = sentence_arcs[int(sentence_index)]
                if not kept:
                    continue
                dependent = np.stack(
                    [provider.get(sentence.sent_id, index) for index, _g, _label in kept]
                )
                governor = np.stack(
                    [provider.get(sentence.sent_id, g) for _index, g, _label in kept]
                )
                labels = [label for _index, _g, label in kept]
                losses.append(self._train_arc_batch(dependent, governor, labels))
            if not losses:
                raise ValueError("no contributing batches in epoch")
            self.epoch_losses.append(float(np.mean(losses)))

    def predict(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
        sent_id: str,
        provider: GroundedFeatureProvider,
        unary_labels: Sequence[str],
    ) -> tuple[str, ...]:
        """Label supplied arcs; uncovered/invalid non-ROOT arcs use unary fallback."""
        del predicted_pos  # grounded φ does not use POS; kept for API symmetry
        if not (len(tokens) == len(head_offsets) == len(unary_labels)):
            raise ValueError("tokens, heads, and unary labels must align")
        if not tokens:
            return ()
        parameters = self._parameters()
        size = len(tokens)
        predictions: list[str] = []
        for index, offset in enumerate(head_offsets):
            if offset == 0:
                predictions.append("root")
                continue
            governor_index = resolve_governor_index(index, offset, size)
            if governor_index is None or not (
                provider.covered(sent_id, index) and provider.covered(sent_id, governor_index)
            ):
                predictions.append(unary_labels[index])
                continue
            dependent = provider.get(sent_id, index)
            governor = provider.get(sent_id, governor_index)
            assert dependent is not None and governor is not None
            logits, _dep_p, _gov_p, _interaction = self._logits(
                dependent[None, :], governor[None, :], parameters
            )
            predictions.append(self.relations[int(np.argmax(logits[0]))])
        return tuple(predictions)


def select_dev_layer(sweep_rows: Sequence[Mapping[str, Any]]) -> int:
    """Pick layer_index with highest DEV deprel-only accuracy; smallest index wins ties."""
    if not sweep_rows:
        raise ValueError("cannot select a layer from an empty sweep")
    best = min(
        sweep_rows,
        key=lambda row: (-float(row["dev_deprel_only_accuracy"]), int(row["layer_index"])),
    )
    return int(best["layer_index"])


def assert_uas_identity(heads_by_arm: Mapping[str, Sequence[int]]) -> None:
    """Fail loudly if matched arms disagree on the shared head sequence."""
    if not heads_by_arm:
        raise ValueError("no arms provided for UAS identity check")
    items = list(heads_by_arm.items())
    reference_name, reference = items[0]
    reference_list = list(reference)
    for name, heads in items[1:]:
        if list(heads) != reference_list:
            raise AssertionError(
                f"matched-labeler UAS invariant failed: {reference_name} vs {name}"
            )


def _empty_accumulator() -> dict[str, list[Any]]:
    return {
        "head": [],
        "deprel": [],
        "case": [],
        "gold_case": [],
    }


def _finalize_arm_metrics(
    predicted_heads: Sequence[int],
    predicted_deprels: Sequence[str],
    gold_heads: Sequence[int],
    gold_deprels: Sequence[str],
    predicted_cases: Sequence[str],
    gold_cases: Sequence[str],
) -> dict[str, Any]:
    strict = score_prediction_sequences(
        predicted_heads, predicted_deprels, gold_heads, gold_deprels
    )
    coarse_gold = tuple(coarse_deprel(label) for label in gold_deprels)
    coarse = score_dependencies(
        predicted_heads,
        tuple(coarse_deprel(label) for label in predicted_deprels),
        gold_heads,
        coarse_gold,
    )
    case_bearing_accuracy, case_bearing_n = score_case_bearing_deprels(
        predicted_deprels, gold_deprels
    )
    return {
        "uas": strict["uas"],
        "las_strict": strict["las"],
        "las_coarse": coarse.las,
        "deprel_only_accuracy": strict["deprel_only_accuracy"],
        "case_bearing_deprel_accuracy": case_bearing_accuracy,
        "case_bearing_deprel_n": case_bearing_n,
        "serve_honest_morph_case": accuracy(predicted_cases, gold_cases)
        if gold_cases
        else 0.0,
        "serve_honest_morph_case_n": len(gold_cases),
        "tokens": strict["tokens"],
        "partition": score_relation_partition(predicted_deprels, gold_deprels),
    }


def preregistered_read(
    unary: Mapping[str, Any],
    surface119: Mapping[str, Any],
    grounded: Mapping[str, Any],
    *,
    dev_layer: int,
    dev_layer_index: int,
    doubly_covered_fraction: Mapping[str, float],
) -> dict[str, Any]:
    """Apply signed grounded-labeler preregistration section 5 verbatim."""
    gain = float(grounded["las_strict"]) - float(unary["las_strict"])
    gain_vs_surface = float(grounded["las_strict"]) - float(surface119["las_strict"])
    partition_gains = {
        bucket: float(grounded["partition"][bucket]["accuracy"])
        - float(unary["partition"][bucket]["accuracy"])
        for bucket in ("pairwise_sensitive", "local")
    }
    case_unary = float(unary["serve_honest_morph_case"])
    case_surface119 = float(surface119["serve_honest_morph_case"])
    case_grounded = float(grounded["serve_honest_morph_case"])
    case_improved = case_grounded > case_unary
    concentrated = partition_gains["pairwise_sensitive"] > partition_gains["local"]
    beats_surface = float(grounded["las_strict"]) > float(surface119["las_strict"])

    if gain <= 0.0:
        verdict = "HALTED"
        reason = "grounded strict LAS gain vs unary is non-positive"
    elif (
        gain < LAS_FIRES_THRESHOLD
        or not concentrated
        or not case_improved
        or not beats_surface
    ):
        verdict = "IN-BETWEEN"
        failed: list[str] = []
        if gain < LAS_FIRES_THRESHOLD:
            failed.append(f"strict LAS gain is below +{LAS_FIRES_THRESHOLD}")
        if not concentrated:
            failed.append("pairwise-sensitive gain is not greater than local gain")
        if not case_improved:
            failed.append("serve-honest case does not improve vs unary")
        if not beats_surface:
            failed.append("grounded does not beat surface-#119 on strict LAS")
        reason = "; ".join(failed)
    else:
        verdict = "FIRES"
        reason = "all preregistered capability conditions are met"

    if case_grounded >= CASE_DELIVERABLE:
        through_line = {
            "met": True,
            "plateau": False,
            "text": (
                f"through-line met: serve-honest case={case_grounded:.4f} ≥ {CASE_DELIVERABLE}"
            ),
        }
    elif case_grounded > CASE_BASELINE:
        through_line = {
            "met": False,
            "plateau": True,
            "text": (
                f"through-line plateau: serve-honest case={case_grounded:.4f} "
                f"< {CASE_DELIVERABLE} but > ~{CASE_BASELINE}"
            ),
        }
    elif case_grounded <= case_unary:
        through_line = {
            "met": False,
            "plateau": False,
            "text": (
                f"through-line diagnose: serve-honest case={case_grounded:.4f} "
                f"≤ unary baseline {case_unary:.4f}"
            ),
        }
    else:
        through_line = {
            "met": False,
            "plateau": False,
            "text": (
                f"through-line below plateau: serve-honest case={case_grounded:.4f} "
                f"above baseline but ≤ ~{CASE_BASELINE}"
            ),
        }

    text = (
        f"{verdict}: grounded strict LAS gain vs unary={gain:+.4f}; "
        f"gain vs surface-#119={gain_vs_surface:+.4f}; pairwise-sensitive deprel gain="
        f"{partition_gains['pairwise_sensitive']:+.4f} versus local="
        f"{partition_gains['local']:+.4f}; serve-honest case "
        f"unary={case_unary:.4f} surface119={case_surface119:.4f} "
        f"grounded={case_grounded:.4f}; dev layer={dev_layer} "
        f"(index={dev_layer_index}). {reason}."
    )
    return {
        "verdict": verdict,
        "deprel_las_gain_vs_unary": gain,
        "gain_vs_surface119": gain_vs_surface,
        "partition_gains": partition_gains,
        "case_unary": case_unary,
        "case_surface119": case_surface119,
        "case_grounded": case_grounded,
        "dev_layer": dev_layer,
        "dev_layer_index": dev_layer_index,
        "doubly_covered_fraction": {
            "dev": float(doubly_covered_fraction["dev"]),
            "test": float(doubly_covered_fraction["test"]),
        },
        "through_line": through_line,
        "text": text,
    }


def _deprel_only_on_mask(
    predicted: Sequence[str],
    gold: Sequence[str],
    mask: Sequence[bool],
) -> float:
    selected = [
        pred == gold_label
        for pred, gold_label, keep in zip(predicted, gold, mask, strict=True)
        if keep
    ]
    if not selected:
        raise ValueError("no doubly-covered tokens to score")
    return sum(selected) / len(selected)


def _doubly_covered_fraction(mask: Sequence[bool]) -> float:
    if not mask:
        return 0.0
    return sum(1 for keep in mask if keep) / len(mask)


def evaluate_dev_layer(
    *,
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    grounded_labeler: GroundedRelationLabeler,
    provider: GroundedFeatureProvider,
) -> dict[str, Any]:
    """Score grounded deprel-only accuracy on DEV doubly-covered arcs only."""
    covered_flags: list[bool] = []
    predicted_labels: list[str] = []
    gold_labels: list[str] = []
    for sentence in dev:
        head = aligned_head_record(sentence, dev_heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = dev_pos[sentence.sent_id]
        common_heads = l0_student.predict_sentence(sentence.tokens, pos).head_offset
        unary_labels = predict_deprels(l0_student, sentence.tokens, pos, common_heads)
        grounded_labels = grounded_labeler.predict(
            sentence.tokens,
            pos,
            common_heads,
            sentence.sent_id,
            provider,
            unary_labels,
        )
        size = len(sentence.tokens)
        for index, offset in enumerate(common_heads):
            keep = is_doubly_covered(sentence.sent_id, index, offset, size, provider)
            covered_flags.append(keep)
            predicted_labels.append(grounded_labels[index])
            gold_labels.append(head.deprel[index])
    return {
        "layer_index": provider.layer_index,
        "layer": provider.layer,
        "dev_deprel_only_accuracy": _deprel_only_on_mask(
            predicted_labels, gold_labels, covered_flags
        ),
        "dev_doubly_covered_fraction": _doubly_covered_fraction(covered_flags),
    }


def evaluate_matched_three_arms(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    surface_labeler: BilinearRelationLabeler,
    grounded_labeler: GroundedRelationLabeler,
    provider: GroundedFeatureProvider,
    case_student: GermanR1Student,
) -> tuple[dict[str, dict[str, Any]], float]:
    """Evaluate unary / surface-#119 / grounded on the same doubly-covered arcs."""
    arm_names = ("unary", "surface119", "grounded")
    accumulated = {arm: _empty_accumulator() for arm in arm_names}
    gold_head: list[int] = []
    gold_deprel: list[str] = []
    covered_total = 0
    token_total = 0

    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        common_heads = l0_student.predict_sentence(sentence.tokens, pos).head_offset
        size = len(sentence.tokens)
        covered_mask = [
            is_doubly_covered(sentence.sent_id, index, offset, size, provider)
            for index, offset in enumerate(common_heads)
        ]
        covered_total += sum(1 for keep in covered_mask if keep)
        token_total += size

        unary_labels = predict_deprels(l0_student, sentence.tokens, pos, common_heads)
        surface_labels = surface_labeler.predict(sentence.tokens, pos, common_heads)
        grounded_labels = grounded_labeler.predict(
            sentence.tokens,
            pos,
            common_heads,
            sentence.sent_id,
            provider,
            unary_labels,
        )
        labels_by_arm = {
            "unary": unary_labels,
            "surface119": surface_labels,
            "grounded": grounded_labels,
        }
        assert_uas_identity(
            {
                "unary": common_heads,
                "surface119": common_heads,
                "grounded": common_heads,
            }
        )

        baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        eligible = [
            index
            for index, gold_label in enumerate(sentence.targets["morph_case"])
            if gold_label != "-"
        ]
        case_score_indices = [index for index in eligible if covered_mask[index]]

        for arm, deprels in labels_by_arm.items():
            case_result = run_predicted_case_sentence(
                case_student,
                sentence.tokens,
                baseline,
                pos,
                common_heads,
                deprels,
                eligible,
            )
            for index, keep in enumerate(covered_mask):
                if not keep:
                    continue
                accumulated[arm]["head"].append(common_heads[index])
                accumulated[arm]["deprel"].append(deprels[index])
            for index in case_score_indices:
                accumulated[arm]["case"].append(case_result.predictions[index])
                accumulated[arm]["gold_case"].append(sentence.targets["morph_case"][index])

        for index, keep in enumerate(covered_mask):
            if not keep:
                continue
            gold_head.append(head.head_offset[index])
            gold_deprel.append(head.deprel[index])

    # Restrict-subset UAS identity across arms (same heads by construction).
    assert_uas_identity({arm: accumulated[arm]["head"] for arm in arm_names})

    metrics: dict[str, dict[str, Any]] = {}
    for arm in arm_names:
        metrics[arm] = _finalize_arm_metrics(
            accumulated[arm]["head"],
            accumulated[arm]["deprel"],
            gold_head,
            gold_deprel,
            accumulated[arm]["case"],
            accumulated[arm]["gold_case"],
        )

    unary_uas = float(metrics["unary"]["uas"])
    for arm in ("surface119", "grounded"):
        arm_uas = float(metrics[arm]["uas"])
        if arm_uas != unary_uas:
            raise AssertionError(
                f"matched-labeler UAS invariant failed for {arm}: "
                f"unary={unary_uas}, {arm}={arm_uas}"
            )

    fraction = covered_total / token_total if token_total else 0.0
    return metrics, fraction


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit on train, sweep grounded layers on DEV, then read TEST once."""
    hasher = hashlib.sha256()
    print("FIT: loading GSD train shared tasks and head_deprel once.", flush=True)
    train = load_split(DATA_ROOT, "train", hasher)
    train_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "train.jsonl", hasher
    )
    train_heads = head_deprel_by_sent(train_head_records)

    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    r1_student = GermanR1Student(registers)
    r1_student.fit(train)
    train_pos = _predicted_pos_by_sentence(r1_student, train)

    print("DEV: loading GSD dev once for L0 window + grounded layer sweep.", flush=True)
    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_heads_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    window, dev_coverage, coverage_curve = choose_window(dev_heads_records)
    dev_heads = head_deprel_by_sent(dev_heads_records)
    dev_pos = _predicted_pos_by_sentence(r1_student, dev)

    l0_student = GermanR3DependencyStudent(window)
    l0_student.fit(train, train_heads, train_pos)

    surface_config = RelationLabelerConfig(seed=0)
    surface_labeler = BilinearRelationLabeler(surface_config)
    print(
        f"FIT: surface-#119 biaffine rank={surface_config.rank} "
        f"epochs={surface_config.epochs} lr={surface_config.learning_rate} seed=0.",
        flush=True,
    )
    surface_labeler.fit(train, train_heads, train_pos)
    print(
        "FIT: surface epoch losses="
        + ", ".join(f"{loss:.6f}" for loss in surface_labeler.epoch_losses),
        flush=True,
    )

    grounded_config = RelationLabelerConfig(seed=0)
    sweep_rows: list[dict[str, Any]] = []
    grounded_by_layer: dict[int, GroundedRelationLabeler] = {}
    for layer_index in LAYER_INDICES:
        print(
            f"FIT: grounded layer_index={layer_index} "
            f"layer={CHECKPOINT_LAYERS[layer_index]}.",
            flush=True,
        )
        train_provider = GroundedFeatureProvider(TRAIN_NPZ, layer_index)
        grounded = GroundedRelationLabeler(grounded_config)
        grounded.fit(train, train_heads, train_pos, train_provider)
        print(
            f"FIT: grounded layer={CHECKPOINT_LAYERS[layer_index]} epoch losses="
            + ", ".join(f"{loss:.6f}" for loss in grounded.epoch_losses),
            flush=True,
        )
        dev_provider = GroundedFeatureProvider(DEV_NPZ, layer_index)
        row = evaluate_dev_layer(
            dev=dev,
            dev_heads=dev_heads,
            dev_pos=dev_pos,
            l0_student=l0_student,
            grounded_labeler=grounded,
            provider=dev_provider,
        )
        sweep_rows.append(row)
        grounded_by_layer[layer_index] = grounded
        print(
            f"DEV sweep: layer={row['layer']} deprel-only={row['dev_deprel_only_accuracy']:.6f} "
            f"doubly-covered={row['dev_doubly_covered_fraction']:.6f}",
            flush=True,
        )

    selected_layer_index = select_dev_layer(sweep_rows)
    selected_layer = CHECKPOINT_LAYERS[selected_layer_index]
    selected_grounded = grounded_by_layer[selected_layer_index]
    selected_dev_row = next(
        row for row in sweep_rows if int(row["layer_index"]) == selected_layer_index
    )
    print(
        f"DEV selected layer_index={selected_layer_index} layer={selected_layer} "
        f"deprel-only={selected_dev_row['dev_deprel_only_accuracy']:.6f}",
        flush=True,
    )

    guard = CampaignTestReadGuard()
    print(
        "FINAL EVAL: reading shared test tasks once and head_deprel test once.",
        flush=True,
    )
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_head_records = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads = head_deprel_by_sent(test_head_records)
    test_pos = _predicted_pos_by_sentence(r1_student, test)
    test_provider = GroundedFeatureProvider(TEST_NPZ, selected_layer_index)

    arm_metrics, test_covered_fraction = evaluate_matched_three_arms(
        test,
        test_heads,
        test_pos,
        l0_student,
        surface_labeler,
        selected_grounded,
        test_provider,
        r1_student,
    )
    guard.assert_complete()

    doubly_covered_fraction = {
        "dev": float(selected_dev_row["dev_doubly_covered_fraction"]),
        "test": float(test_covered_fraction),
    }
    read = preregistered_read(
        arm_metrics["unary"],
        arm_metrics["surface119"],
        arm_metrics["grounded"],
        dev_layer=selected_layer,
        dev_layer_index=selected_layer_index,
        doubly_covered_fraction=doubly_covered_fraction,
    )

    scoreboard: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "scope": SCOPE,
        "measurement": (
            "serve-honest three-arm labeler comparison on doubly-covered arcs only; "
            "same raw L0 predicted heads and same case cascade input in every arm"
        ),
        "serve_honest_features": [
            "R1-predicted POS (unary/surface arms; unused by grounded φ)",
            "hashed surface/shape/position (surface-#119 only)",
            "Qwen2.5-3B last_residual grounded φ at DEV-selected layer (grounded arm)",
            "unary count-table fallback on uncovered/invalid non-ROOT arcs (grounded arm)",
        ],
        "gold_usage": (
            "GSD train gold head offsets choose fit-time governors and train gold deprels "
            "are cross-entropy targets; test gold heads/deprels/case are final scoring "
            "oracles only; layer selected on DEV only"
        ),
        "data_hash_sha256": hasher.hexdigest(),
        "test_read_counts": dict(guard.counts),
        "test_read_once": True,
        "l0_window": {
            "k": window,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "fixed_hyperparameters": {
            "rank": grounded_config.rank,
            "epochs": grounded_config.epochs,
            "learning_rate": grounded_config.learning_rate,
            "l2_penalty": grounded_config.l2_penalty,
            "seed": 0,
            "feature_dim": FEATURE_DIM,
            "gradient_norm_clip": GRAD_CLIP,
            "checkpoint_layers": list(CHECKPOINT_LAYERS),
        },
        "dev_layer_sweep": sweep_rows,
        "selected_layer_index": selected_layer_index,
        "selected_layer": selected_layer,
        "doubly_covered_fraction": doubly_covered_fraction,
        "relation_vocabulary_size": len(selected_grounded.relations),
        "relation_vocabulary": list(selected_grounded.relations),
        "arms": arm_metrics,
        "read": read,
    }
    return scoreboard, guard


def _print_metric_row(name: str, metric: Mapping[str, Any]) -> None:
    pairwise = metric["partition"]["pairwise_sensitive"]
    local = metric["partition"]["local"]
    print(
        f"{name:<15} {metric['uas']:.6f}  {metric['las_strict']:.6f}  "
        f"{metric['las_coarse']:.6f}  {metric['deprel_only_accuracy']:.6f}  "
        f"{metric['case_bearing_deprel_accuracy']:.6f} ({metric['case_bearing_deprel_n']})  "
        f"{metric['serve_honest_morph_case']:.6f} ({metric['serve_honest_morph_case_n']})  "
        f"{pairwise['accuracy']:.6f} ({pairwise['n']})  "
        f"{local['accuracy']:.6f} ({local['n']})"
    )


def print_scoreboard(scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard) -> None:
    """Print DEV sweep, 3-arm TEST table, and the §5 preregistered read."""
    print("\nGROUNDED-φ RELATION-LABELER GATE — GSD")
    print(f"run_tag: {scoreboard['run_tag']}")
    print(f"scope: {scoreboard['scope']}")
    config = scoreboard["fixed_hyperparameters"]
    print(
        f"config: rank={config['rank']} epochs={config['epochs']} "
        f"lr={config['learning_rate']} l2={config['l2_penalty']} seed={config['seed']} "
        f"feature_dim={config['feature_dim']} "
        f"relations={scoreboard['relation_vocabulary_size']}"
    )
    print(
        "matched control: every arm labels identical raw L0 predicted heads; "
        "scored on doubly-covered arcs only"
    )
    print(
        "serve honesty: grounded φ from sentence text dump; R1-predicted POS; "
        "no gold dependency fields at predict time"
    )

    print("\nDEV LAYER SWEEP (deprel-only on doubly-covered arcs)")
    print(f"{'layer_idx':<10} {'layer':<8} {'deprel_only':<14} {'doubly_covered':<14}")
    for row in scoreboard["dev_layer_sweep"]:
        print(
            f"{row['layer_index']:<10} {row['layer']:<8} "
            f"{row['dev_deprel_only_accuracy']:<14.6f} "
            f"{row['dev_doubly_covered_fraction']:<14.6f}"
        )
    print(
        f"selected: layer_index={scoreboard['selected_layer_index']} "
        f"layer={scoreboard['selected_layer']}"
    )
    fractions = scoreboard["doubly_covered_fraction"]
    print(
        f"\ndoubly-covered fraction: dev={fractions['dev']:.6f} "
        f"test={fractions['test']:.6f}"
    )

    print(
        "\narm             UAS       LAS-strict LAS-coarse deprel-only case-bearing (n)  "
        "case (n)         pairwise (n)     local (n)"
    )
    arms = scoreboard["arms"]
    _print_metric_row("unary", arms["unary"])
    _print_metric_row("surface-#119", arms["surface119"])
    _print_metric_row("grounded", arms["grounded"])

    print("\nTHE PRE-REGISTERED READ")
    print(scoreboard["read"]["text"])
    print(f"through-line: {scoreboard['read']['through_line']['text']}")
    guard.assert_complete()
    print("TEST READ ONCE: CONFIRMED")


def main() -> None:
    scoreboard, guard = build_scoreboard()
    OUTPUT_PATH.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"structured_output: {OUTPUT_PATH}")
    print_scoreboard(scoreboard, guard)


if __name__ == "__main__":
    main()
