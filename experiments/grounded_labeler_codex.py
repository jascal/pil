"""Grounded-residual biaffine relation-labeler gate on matched L0 heads.

The grounded arm replaces #119's surface feature vector with a Qwen2.5-3B
per-word residual.  Layer selection is performed on dev; the guarded test read
is used once for a matched unary/surface/grounded comparison restricted to arcs
whose predicted dependent and governor both have grounded rows.
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
    DependencyPrediction,
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

GROUND_ROOT = Path(__file__).resolve().parents[1] / "data" / "grounded"
GROUND_PATHS = {
    "train": GROUND_ROOT / "qwen3b_gsd_train_n600.npz",
    "dev": GROUND_ROOT / "qwen3b_gsd_dev_n799.npz",
    "test": GROUND_ROOT / "qwen3b_gsd_test_n977.npz",
}
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "grounded_labeler_codex.json"
CHECKPOINT_LAYERS = np.asarray([8, 17, 26, 35], dtype=np.int64)
LAYER_INDICES = (0, 1, 2, 3)
FEATURE_DIM = 2048
ARM_NAMES = ("unary", "surface119", "grounded")


class GroundedResidualProvider:
    """Index one grounded npz split by its exact ``(sent_id, token_index)`` key."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            self.checkpoint_layers = np.asarray(
                archive["checkpoint_layers"], dtype=np.int64
            )
            sent_ids = np.asarray(archive["sent_ids"])
            token_indices = np.asarray(archive["token_index"], dtype=np.int64)
            self.last_residual = np.asarray(archive["last_residual"])
        if not np.array_equal(self.checkpoint_layers, CHECKPOINT_LAYERS):
            raise AssertionError(
                f"{self.path}: checkpoint layers drifted: "
                f"{self.checkpoint_layers.tolist()} != {CHECKPOINT_LAYERS.tolist()}"
            )
        if sent_ids.ndim != 1 or token_indices.ndim != 1:
            raise ValueError(f"{self.path}: sent_ids and token_index must be vectors")
        if len(sent_ids) != len(token_indices) or len(sent_ids) != len(self.last_residual):
            raise ValueError(f"{self.path}: grounded row arrays do not align")
        if self.last_residual.shape != (len(sent_ids), len(LAYER_INDICES), FEATURE_DIM):
            raise ValueError(
                f"{self.path}: last_residual has unexpected shape "
                f"{self.last_residual.shape}"
            )
        self._row_by_key: dict[tuple[str, int], int] = {}
        for row, (sent_id, token_index) in enumerate(
            zip(sent_ids, token_indices, strict=True)
        ):
            key = (str(sent_id), int(token_index))
            if key in self._row_by_key:
                raise ValueError(f"{self.path}: duplicate grounded key {key}")
            self._row_by_key[key] = row
        self.sentence_ids = frozenset(sent_id for sent_id, _index in self._row_by_key)

    @staticmethod
    def _validate_layer(layer: int) -> None:
        if layer not in LAYER_INDICES:
            raise ValueError(f"grounded layer index must be one of {LAYER_INDICES}: {layer}")

    def covered(self, sent_id: str, token_index: int) -> bool:
        return (sent_id, int(token_index)) in self._row_by_key

    def residual(self, sent_id: str, token_index: int, layer: int) -> np.ndarray:
        """Return one last-residual checkpoint as a float64 feature vector."""
        self._validate_layer(layer)
        key = (sent_id, int(token_index))
        row = self._row_by_key[key]
        return np.asarray(self.last_residual[row, layer, :], dtype=np.float64)


class GroundedRelationLabeler:
    """The #119 low-rank biaffine label form over 2048-d grounded residuals."""

    feature_dim = FEATURE_DIM

    def __init__(self, config: RelationLabelerConfig | None = None) -> None:
        self.config = config or RelationLabelerConfig()
        if self.config.rank < 1:
            raise ValueError("rank must be positive")
        if self.config.epochs < 1:
            raise ValueError("epochs must be positive")
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
        self.training_arc_count = 0
        self.training_sentence_count = 0

    def _initialize_parameters(self, relations: Sequence[str]) -> None:
        self.relations = tuple(sorted(set(relations)))
        if not self.relations:
            raise ValueError("no covered non-root train arcs at grounded layer")
        self._relation_index = {
            relation: index for index, relation in enumerate(self.relations)
        }
        relation_count = len(self.relations)
        feature_scale = 1.0 / np.sqrt(self.feature_dim)
        rank_scale = 1.0 / np.sqrt(self.config.rank)
        self.U = self._rng.normal(
            0.0, feature_scale, (self.feature_dim, self.config.rank)
        )
        self.V = self._rng.normal(
            0.0, feature_scale, (self.feature_dim, self.config.rank)
        )
        self.C = self._rng.normal(
            0.0, rank_scale, (relation_count, self.config.rank)
        )
        self.P = np.zeros((relation_count, self.feature_dim), dtype=np.float64)
        self.Q = np.zeros((relation_count, self.feature_dim), dtype=np.float64)
        self.b = np.zeros(relation_count, dtype=np.float64)

    def _parameters(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if any(
            value is None for value in (self.U, self.V, self.C, self.P, self.Q, self.b)
        ):
            raise RuntimeError("grounded relation labeler used before fit")
        assert self.U is not None
        assert self.V is not None
        assert self.C is not None
        assert self.P is not None
        assert self.Q is not None
        assert self.b is not None
        return self.U, self.V, self.C, self.P, self.Q, self.b

    @staticmethod
    def _eligible_arcs(
        sentence: Sentence,
        head: HeadDeprelRecord,
        provider: GroundedResidualProvider,
    ) -> tuple[tuple[int, int, str], ...]:
        arcs: list[tuple[int, int, str]] = []
        size = len(sentence.tokens)
        for index, (offset, relation) in enumerate(
            zip(head.head_offset, head.deprel, strict=True)
        ):
            if offset == 0:
                continue
            governor = resolve_governor_index(index, offset, size)
            if (
                governor is not None
                and provider.covered(sentence.sent_id, index)
                and provider.covered(sentence.sent_id, governor)
            ):
                arcs.append((index, governor, relation))
        return tuple(arcs)

    def _train_features(
        self,
        sent_id: str,
        arcs: Sequence[tuple[int, int, str]],
        provider: GroundedResidualProvider,
        layer: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        dependent = np.stack(
            [provider.residual(sent_id, index, layer) for index, _g, _r in arcs]
        )
        governor = np.stack(
            [provider.residual(sent_id, governor, layer) for _i, governor, _r in arcs]
        )
        return dependent, governor, tuple(relation for _i, _g, relation in arcs)

    def _train_sentence(
        self,
        dependent: np.ndarray,
        governor: np.ndarray,
        gold_deprels: Sequence[str],
    ) -> float:
        parameters = self._parameters()
        u, v, core, dependent_affine, governor_affine, _bias = parameters
        logits, dependent_projection, governor_projection, interaction = (
            BilinearRelationLabeler._logits(dependent, governor, parameters)
        )
        targets = np.asarray(
            [self._relation_index[label] for label in gold_deprels], dtype=np.int64
        )
        example_count = len(targets)
        maxima = np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(logits - maxima)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        loss = float(
            np.mean(
                maxima[:, 0]
                + np.log(exponentiated.sum(axis=1))
                - logits[np.arange(example_count), targets]
            )
        )

        score_gradient = probabilities
        score_gradient[np.arange(example_count), targets] -= 1.0
        score_gradient /= example_count
        interaction_gradient = score_gradient @ core
        grad_u = dependent.T @ (interaction_gradient * governor_projection)
        grad_v = governor.T @ (interaction_gradient * dependent_projection)
        grad_core = score_gradient.T @ interaction
        grad_p = score_gradient.T @ dependent
        grad_q = score_gradient.T @ governor
        grad_b = score_gradient.sum(axis=0)

        penalty = self.config.l2_penalty
        grad_u += penalty * u
        grad_v += penalty * v
        grad_core += penalty * core
        grad_p += penalty * dependent_affine
        grad_q += penalty * governor_affine
        gradients = (grad_u, grad_v, grad_core, grad_p, grad_q, grad_b)
        grad_norm = np.sqrt(
            sum(float(np.square(gradient).sum()) for gradient in gradients)
        )
        if grad_norm > 5.0:
            gradients = tuple(gradient * (5.0 / grad_norm) for gradient in gradients)
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter -= self.config.learning_rate * gradient
        return loss

    def fit(
        self,
        train_sentences: Sequence[Sentence],
        provider: GroundedResidualProvider,
        train_heads: Mapping[str, HeadDeprelRecord],
        layer: int,
    ) -> None:
        """Fit only gold non-root arcs with both grounded endpoints."""
        provider._validate_layer(layer)
        if not train_sentences:
            raise ValueError("cannot fit on empty grounded training data")
        eligible_by_sentence: list[tuple[Sentence, tuple[tuple[int, int, str], ...]]] = []
        relations: list[str] = []
        for sentence in train_sentences:
            head = aligned_head_record(sentence, train_heads)
            if not isinstance(head, HeadDeprelRecord):
                raise TypeError("labeled dependency record required")
            arcs = self._eligible_arcs(sentence, head, provider)
            if arcs:
                eligible_by_sentence.append((sentence, arcs))
                relations.extend(relation for _i, _g, relation in arcs)
        self.training_sentence_count = len(eligible_by_sentence)
        self.training_arc_count = sum(len(arcs) for _sentence, arcs in eligible_by_sentence)
        self._initialize_parameters(relations)

        self.epoch_losses = []
        for _epoch in range(self.config.epochs):
            losses: list[float] = []
            order = self._rng.permutation(len(eligible_by_sentence))
            for sentence_index in order:
                sentence, arcs = eligible_by_sentence[int(sentence_index)]
                dependent, governor, labels = self._train_features(
                    sentence.sent_id, arcs, provider, layer
                )
                losses.append(self._train_sentence(dependent, governor, labels))
            if not losses:
                raise ValueError(
                    f"no train sentence yielded a covered non-root arc at layer {layer}"
                )
            self.epoch_losses.append(float(np.mean(losses)))

    def predict(
        self,
        sent_id: str,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        predicted_head_offsets: Sequence[int],
        provider: GroundedResidualProvider,
        layer: int,
        l0_student: GermanR3DependencyStudent,
    ) -> tuple[str, ...]:
        """Override unary only for doubly-covered, real-governor predicted arcs."""
        provider._validate_layer(layer)
        if not (
            len(tokens) == len(predicted_pos) == len(predicted_head_offsets)
        ):
            raise ValueError("tokens, predicted POS, and supplied heads must align")
        unary = predict_deprels(
            l0_student, tokens, predicted_pos, predicted_head_offsets
        )
        parameters = self._parameters()
        predictions = list(unary)
        size = len(tokens)
        for index, offset in enumerate(predicted_head_offsets):
            if offset == 0:
                predictions[index] = "root"
                continue
            governor = resolve_governor_index(index, offset, size)
            if (
                governor is None
                or not provider.covered(sent_id, index)
                or not provider.covered(sent_id, governor)
            ):
                continue
            dependent_feature = provider.residual(sent_id, index, layer)[None, :]
            governor_feature = provider.residual(sent_id, governor, layer)[None, :]
            logits, _dep, _gov, _interaction = BilinearRelationLabeler._logits(
                dependent_feature, governor_feature, parameters
            )
            predictions[index] = self.relations[int(np.argmax(logits[0]))]
        return tuple(predictions)


def doubly_covered_indices(
    sent_id: str,
    token_count: int,
    predicted_head_offsets: Sequence[int],
    provider: GroundedResidualProvider,
    layer: int,
) -> tuple[int, ...]:
    """Return predicted non-root arcs whose two real endpoints are grounded."""
    provider._validate_layer(layer)
    if len(predicted_head_offsets) != token_count:
        raise ValueError("token count and predicted heads must align")
    covered: list[int] = []
    for index, offset in enumerate(predicted_head_offsets):
        if offset == 0:
            continue
        governor = resolve_governor_index(index, offset, token_count)
        if (
            governor is not None
            and provider.covered(sent_id, index)
            and provider.covered(sent_id, governor)
        ):
            covered.append(index)
    return tuple(covered)


def select_dev_layer(layer_accuracies: Mapping[int, float]) -> int:
    """Select dev argmax, breaking ties toward the smallest layer index."""
    if set(layer_accuracies) != set(LAYER_INDICES):
        raise ValueError(f"dev sweep must contain exactly layers {LAYER_INDICES}")
    return max(LAYER_INDICES, key=lambda layer: (float(layer_accuracies[layer]), -layer))


def assert_matched_uas(metrics: Mapping[str, Mapping[str, Any]]) -> None:
    """Enforce identical UAS for arms that label one shared head sequence."""
    unary_uas = float(metrics["unary"]["uas"])
    for arm in ("surface119", "grounded"):
        arm_uas = float(metrics[arm]["uas"])
        assert arm_uas == unary_uas, (
            f"matched-labeler UAS invariant failed for {arm}: "
            f"unary={unary_uas}, {arm}={arm_uas}"
        )


def _score_arm(
    predicted_heads: Sequence[int],
    predicted_deprels: Sequence[str],
    gold_heads: Sequence[int],
    gold_deprels: Sequence[str],
    predicted_case: Sequence[str],
    gold_case: Sequence[str],
) -> dict[str, Any]:
    strict = score_prediction_sequences(
        predicted_heads, predicted_deprels, gold_heads, gold_deprels
    )
    coarse = score_dependencies(
        predicted_heads,
        tuple(coarse_deprel(label) for label in predicted_deprels),
        gold_heads,
        tuple(coarse_deprel(label) for label in gold_deprels),
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
        "serve_honest_morph_case": accuracy(predicted_case, gold_case),
        "serve_honest_morph_case_n": len(gold_case),
        "tokens": strict["tokens"],
        "partition": score_relation_partition(predicted_deprels, gold_deprels),
    }


def evaluate_matched_labelers(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    surface_labeler: BilinearRelationLabeler,
    grounded_labeler: GroundedRelationLabeler,
    provider: GroundedResidualProvider,
    layer: int,
    case_student: GermanR1Student,
) -> tuple[dict[str, dict[str, Any]], float]:
    """Score three full predictions on one doubly-covered dependency subset."""
    accumulated: dict[str, dict[str, list[Any]]] = {
        arm: {"head": [], "deprel": [], "case": []} for arm in ARM_NAMES
    }
    gold_head: list[int] = []
    gold_deprel: list[str] = []
    gold_case: list[str] = []
    covered_n = all_arc_n = 0

    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        common_heads = l0_student.predict_sentence(sentence.tokens, pos).head_offset
        predictions = {
            "unary": DependencyPrediction(
                common_heads,
                predict_deprels(l0_student, sentence.tokens, pos, common_heads),
            ),
            "surface119": DependencyPrediction(
                common_heads,
                surface_labeler.predict(sentence.tokens, pos, common_heads),
            ),
            "grounded": DependencyPrediction(
                common_heads,
                grounded_labeler.predict(
                    sentence.sent_id,
                    sentence.tokens,
                    pos,
                    common_heads,
                    provider,
                    layer,
                    l0_student,
                ),
            ),
        }
        selected = doubly_covered_indices(
            sentence.sent_id, len(sentence.tokens), common_heads, provider, layer
        )
        covered_n += len(selected)
        all_arc_n += len(sentence.tokens)
        case_baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        case_eligible = [
            index
            for index, label in enumerate(sentence.targets["morph_case"])
            if label != "-"
        ]
        for arm, dependency in predictions.items():
            case_result = run_predicted_case_sentence(
                case_student,
                sentence.tokens,
                case_baseline,
                pos,
                dependency.head_offset,
                dependency.deprel,
                case_eligible,
            )
            accumulated[arm]["head"].extend(common_heads[index] for index in selected)
            accumulated[arm]["deprel"].extend(
                dependency.deprel[index] for index in selected
            )
            accumulated[arm]["case"].extend(
                case_result.predictions[index] for index in case_eligible
            )
        gold_head.extend(head.head_offset[index] for index in selected)
        gold_deprel.extend(head.deprel[index] for index in selected)
        gold_case.extend(
            sentence.targets["morph_case"][index] for index in case_eligible
        )

    if not all_arc_n or not covered_n:
        raise ValueError("test split has no doubly-covered arcs")
    metrics = {
        arm: _score_arm(
            accumulated[arm]["head"],
            accumulated[arm]["deprel"],
            gold_head,
            gold_deprel,
            accumulated[arm]["case"],
            gold_case,
        )
        for arm in ARM_NAMES
    }
    assert_matched_uas(metrics)
    return metrics, covered_n / all_arc_n


def evaluate_dev_layer(
    dev: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    labeler: GroundedRelationLabeler,
    provider: GroundedResidualProvider,
    layer: int,
) -> tuple[float, float, int]:
    predicted: list[str] = []
    gold: list[str] = []
    covered_n = all_arc_n = 0
    for sentence in dev:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        common_heads = l0_student.predict_sentence(sentence.tokens, pos).head_offset
        relations = labeler.predict(
            sentence.sent_id,
            sentence.tokens,
            pos,
            common_heads,
            provider,
            layer,
            l0_student,
        )
        selected = doubly_covered_indices(
            sentence.sent_id, len(sentence.tokens), common_heads, provider, layer
        )
        predicted.extend(relations[index] for index in selected)
        gold.extend(head.deprel[index] for index in selected)
        covered_n += len(selected)
        all_arc_n += len(sentence.tokens)
    if not predicted or not all_arc_n:
        raise ValueError(f"dev has no doubly-covered arcs at layer {layer}")
    return accuracy(predicted, gold), covered_n / all_arc_n, covered_n


def preregistered_read(
    unary: Mapping[str, Any],
    surface119: Mapping[str, Any],
    grounded: Mapping[str, Any],
    dev_layer: int,
    doubly_covered_fraction: Mapping[str, float],
) -> dict[str, Any]:
    """Apply signed section 5 with HALTED precedence and no reinterpretation."""
    las_gain = float(grounded["las_strict"]) - float(unary["las_strict"])
    gain_vs_surface = float(grounded["las_strict"]) - float(
        surface119["las_strict"]
    )
    partition_gains = {
        bucket: float(grounded["partition"][bucket]["accuracy"])
        - float(unary["partition"][bucket]["accuracy"])
        for bucket in ("pairwise_sensitive", "local")
    }
    case_unary = float(unary["serve_honest_morph_case"])
    case_surface = float(surface119["serve_honest_morph_case"])
    case_grounded = float(grounded["serve_honest_morph_case"])
    concentrated = partition_gains["pairwise_sensitive"] > partition_gains["local"]
    case_improved = case_grounded > case_unary
    beats_surface = gain_vs_surface > 0.0

    if las_gain <= 0.0:
        verdict = "HALTED → rung 3 (R4 hybrid)"
        reason = "grounded strict LAS does not exceed unary"
    elif las_gain < 0.03 or not concentrated or not case_improved or not beats_surface:
        verdict = "IN-BETWEEN"
        failed: list[str] = []
        if las_gain < 0.03:
            failed.append("strict LAS gain is positive but below +0.03")
        if not concentrated:
            failed.append("partition gain is not concentrated on pairwise-sensitive")
        if not case_improved:
            failed.append("serve-honest case does not improve over unary")
        if not beats_surface:
            failed.append("grounded does not beat surface-phi #119")
        reason = "; ".join(failed)
    else:
        verdict = "FIRES"
        reason = "all preregistered grounded-labeler conditions are met"

    if case_grounded >= 0.90:
        through_line = "met"
    elif case_grounded > 0.76:
        through_line = "plateau language"
    else:
        through_line = "diagnose"
    text = (
        f"{verdict}: grounded-unary strict LAS gain={las_gain:+.6f}; "
        f"grounded-surface119 strict LAS gain={gain_vs_surface:+.6f}; "
        f"partition gains pairwise-sensitive="
        f"{partition_gains['pairwise_sensitive']:+.6f} versus "
        f"local={partition_gains['local']:+.6f}; serve-honest case "
        f"unary={case_unary:.6f}, surface119={case_surface:.6f}, "
        f"grounded={case_grounded:.6f}; DEV selected layer={dev_layer}; "
        f"doubly-covered fractions dev={doubly_covered_fraction['dev']:.6f}, "
        f"test={doubly_covered_fraction['test']:.6f}; through-line={through_line}. "
        f"{reason}."
    )
    return {
        "verdict": verdict,
        "deprel_las_gain_vs_unary": las_gain,
        "gain_vs_surface119": gain_vs_surface,
        "partition_gains": partition_gains,
        "case_unary": case_unary,
        "case_surface119": case_surface,
        "case_grounded": case_grounded,
        "dev_layer": dev_layer,
        "doubly_covered_fraction": dict(doubly_covered_fraction),
        "through_line": through_line,
        "text": text,
    }


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit fixed models, sweep grounded layers on dev, and read test once."""
    hasher = hashlib.sha256()
    print("GROUNDED: loading train/dev/test residual providers once each.", flush=True)
    providers = {
        split: GroundedResidualProvider(path) for split, path in GROUND_PATHS.items()
    }

    print("FIT: loading full GSD train shared tasks and head_deprel once.", flush=True)
    train = load_split(DATA_ROOT, "train", hasher)
    train_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "train.jsonl", hasher
    )
    train_heads = head_deprel_by_sent(train_head_records)
    grounded_train = [
        sentence
        for sentence in train
        if sentence.sent_id in providers["train"].sentence_ids
    ]
    for sentence in grounded_train:
        aligned_head_record(sentence, train_heads)

    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    r1_student = GermanR1Student(registers)
    r1_student.fit(train)
    train_pos = _predicted_pos_by_sentence(r1_student, train)

    print("DEV: loading full GSD dev for L0 window choice and layer sweep.", flush=True)
    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    dev_heads = head_deprel_by_sent(dev_head_records)
    window, dev_coverage, coverage_curve = choose_window(dev_head_records)
    dev_pos = _predicted_pos_by_sentence(r1_student, dev)

    l0_student = GermanR3DependencyStudent(window)
    l0_student.fit(train, train_heads, train_pos)
    config = RelationLabelerConfig(seed=0)
    surface_labeler = BilinearRelationLabeler(config)
    print(
        f"FIT: surface-phi #119 rank={config.rank} epochs={config.epochs} "
        f"lr={config.learning_rate} seed={config.seed}.",
        flush=True,
    )
    surface_labeler.fit(train, train_heads, train_pos)
    print(
        "FIT: surface-phi epoch losses="
        + ", ".join(f"{loss:.6f}" for loss in surface_labeler.epoch_losses),
        flush=True,
    )

    dev_curve: list[dict[str, Any]] = []
    for layer in LAYER_INDICES:
        labeler = GroundedRelationLabeler(config)
        print(
            f"FIT: grounded-phi DEV candidate layer={layer} "
            f"checkpoint={CHECKPOINT_LAYERS[layer]}.",
            flush=True,
        )
        labeler.fit(grounded_train, providers["train"], train_heads, layer)
        dev_accuracy, dev_fraction, dev_n = evaluate_dev_layer(
            dev,
            dev_heads,
            dev_pos,
            l0_student,
            labeler,
            providers["dev"],
            layer,
        )
        dev_curve.append(
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
            f"DEV LAYER: L={layer} checkpoint={CHECKPOINT_LAYERS[layer]} "
            f"deprel-only={dev_accuracy:.6f} doubly-covered={dev_fraction:.6f} "
            f"({dev_n} arcs)",
            flush=True,
        )
    selected_layer = select_dev_layer(
        {row["layer"]: row["deprel_only_accuracy"] for row in dev_curve}
    )
    print(
        f"DEV SELECTED: L*={selected_layer} "
        f"checkpoint={CHECKPOINT_LAYERS[selected_layer]}",
        flush=True,
    )
    selected_dev_fraction = float(dev_curve[selected_layer]["doubly_covered_fraction"])

    grounded_labeler = GroundedRelationLabeler(config)
    print(f"FIT: fresh grounded-phi final model at DEV-selected L*={selected_layer}.", flush=True)
    grounded_labeler.fit(
        grounded_train, providers["train"], train_heads, selected_layer
    )
    print(
        "FIT: final grounded-phi epoch losses="
        + ", ".join(f"{loss:.6f}" for loss in grounded_labeler.epoch_losses),
        flush=True,
    )

    guard = CampaignTestReadGuard()
    print("FINAL EVAL: reading shared test and head_deprel test once each.", flush=True)
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_head_records = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads = head_deprel_by_sent(test_head_records)
    test_pos = _predicted_pos_by_sentence(r1_student, test)
    metrics, test_fraction = evaluate_matched_labelers(
        test,
        test_heads,
        test_pos,
        l0_student,
        surface_labeler,
        grounded_labeler,
        providers["test"],
        selected_layer,
        r1_student,
    )
    guard.assert_complete()
    fractions = {"dev": selected_dev_fraction, "test": test_fraction}
    read = preregistered_read(
        metrics["unary"],
        metrics["surface119"],
        metrics["grounded"],
        selected_layer,
        fractions,
    )
    scoreboard: dict[str, Any] = {
        "run_tag": "empirical",
        "scope": "grounded-phi relation-labeler gate on predicted-head doubly-covered arcs",
        "measurement": (
            "three matched relation-labeler arms share raw L0 predicted heads; dependency "
            "metrics use only doubly-covered non-root arcs; case uses full sentences"
        ),
        "serve_honest": True,
        "gold_usage": (
            "train gold heads select covered non-root fit arcs and train gold deprels are "
            "cross-entropy targets; dev gold deprels select only the residual checkpoint; "
            "test gold fields are scoring oracles only"
        ),
        "data_hash_sha256": hasher.hexdigest(),
        "test_read_counts": dict(guard.counts),
        "test_read_once": True,
        "grounded_paths": {split: str(path) for split, path in GROUND_PATHS.items()},
        "checkpoint_layers": CHECKPOINT_LAYERS.tolist(),
        "dev_layer_curve": dev_curve,
        "selected_dev_layer": selected_layer,
        "selected_checkpoint_layer": int(CHECKPOINT_LAYERS[selected_layer]),
        "doubly_covered_fraction": fractions,
        "l0_window": {
            "k": window,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "fixed_hyperparameters": {
            "rank": config.rank,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "l2_penalty": config.l2_penalty,
            "seed": config.seed,
            "surface_buckets": config.surface_buckets,
            "position_buckets": config.position_buckets,
            "gradient_norm_clip": 5.0,
            "grounded_feature_dim": FEATURE_DIM,
        },
        "grounded_training": {
            "sentences_with_any_dump_row": len(grounded_train),
            "sentences_with_eligible_arc": grounded_labeler.training_sentence_count,
            "eligible_arcs": grounded_labeler.training_arc_count,
            "relation_vocabulary_size": len(grounded_labeler.relations),
            "relation_vocabulary": list(grounded_labeler.relations),
            "final_epoch_losses": grounded_labeler.epoch_losses,
        },
        "arms": metrics,
        "read": read,
    }
    return scoreboard, guard


def _print_metric_row(name: str, metric: Mapping[str, Any]) -> None:
    pairwise = metric["partition"]["pairwise_sensitive"]
    local = metric["partition"]["local"]
    print(
        f"{name:<12} {metric['uas']:.6f}  {metric['las_strict']:.6f}  "
        f"{metric['las_coarse']:.6f}  {metric['deprel_only_accuracy']:.6f}  "
        f"{metric['case_bearing_deprel_accuracy']:.6f} "
        f"({metric['case_bearing_deprel_n']})  "
        f"{metric['serve_honest_morph_case']:.6f} "
        f"({metric['serve_honest_morph_case_n']})  "
        f"{pairwise['accuracy']:.6f} ({pairwise['n']})  "
        f"{local['accuracy']:.6f} ({local['n']})"
    )


def print_scoreboard(
    scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard
) -> None:
    """Print the dev curve, matched test table, and signed decision read."""
    print("\nGROUNDED-PHI RELATION-LABELER GATE")
    config = scoreboard["fixed_hyperparameters"]
    print(
        f"config: rank={config['rank']} epochs={config['epochs']} "
        f"lr={config['learning_rate']} l2={config['l2_penalty']} "
        f"seed={config['seed']} feature_dim={config['grounded_feature_dim']}"
    )
    print("DEV LAYER SWEEP — doubly-covered deprel-only accuracy")
    for row in scoreboard["dev_layer_curve"]:
        print(
            f"L={row['layer']} checkpoint={row['checkpoint_layer']} "
            f"accuracy={row['deprel_only_accuracy']:.6f} "
            f"coverage={row['doubly_covered_fraction']:.6f} "
            f"n={row['doubly_covered_n']}"
        )
    print(
        f"selected L*={scoreboard['selected_dev_layer']} "
        f"checkpoint={scoreboard['selected_checkpoint_layer']}"
    )
    fractions = scoreboard["doubly_covered_fraction"]
    print(
        f"doubly-covered fraction: dev={fractions['dev']:.6f} "
        f"test={fractions['test']:.6f}"
    )
    print("\nGSD TEST — dependency metrics on identical doubly-covered arcs; case on full sentences")
    print(
        "arm          UAS       LAS-strict LAS-coarse deprel-only case-bearing (n)  "
        "case (n)         pairwise (n)     local (n)"
    )
    for arm in ARM_NAMES:
        _print_metric_row(arm, scoreboard["arms"][arm])
    print("\nTHE PRE-REGISTERED READ")
    print(scoreboard["read"]["text"])
    guard.assert_complete()
    print("TEST READ ONCE: CONFIRMED")


def main() -> None:
    scoreboard, guard = build_scoreboard()
    OUTPUT_PATH.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print_scoreboard(scoreboard, guard)
    print(f"structured_output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
