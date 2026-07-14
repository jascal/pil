"""Matched unary-versus-biaffine German dependency relation labeling probe.

Both arms label the same raw L0 predicted heads.  The deciding inputs are
R1-predicted POS plus the shipped hashed surface/shape/position feature vector;
gold dependency fields appear only as train targets and final evaluation labels.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.attach_levers_codex import (  # noqa: E402
    BilinearAttachmentScorer,
    BilinearConfig,
    predict_deprels,
    score_case_bearing_deprels,
    score_prediction_sequences,
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

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "biaffine_labeler_codex.json"
RUN_TAG = "empirical"
SCOPE = "axis-B matched relation-labeler comparison on raw L0 predicted heads"
SEEDS = (0, 1, 2)
PAIRWISE_SENSITIVE_DEPRELS = frozenset({"nsubj", "obj", "iobj", "obl", "nmod", "conj"})
LOCAL_DEPRELS = frozenset({"det", "amod", "case", "punct", "aux", "cop"})


@dataclass(frozen=True)
class RelationLabelerConfig:
    """Fixed-a-priori relation-labeler configuration; no test tuning is performed."""

    rank: int = 16
    epochs: int = 5
    learning_rate: float = 0.04
    l2_penalty: float = 1e-5
    seed: int = 0
    surface_buckets: int = 32
    position_buckets: int = 5


class BilinearRelationLabeler:
    """Low-rank biaffine label scorer fitted by per-sentence SGD.

    The scorer shares dependent and governor projections ``U,V`` of shape
    ``(feature_dim, rank)``.  Relation ``r`` has a diagonal bilinear core
    ``c_r``, so its interaction matrix is ``U @ diag(c_r) @ V.T``.  Its logit
    is ``(phi(i) @ U * phi(g) @ V) @ c_r + phi(i) @ p_r + phi(g) @ q_r + b_r``.
    This uses ``2*d*rank + |R|*(rank + 2*d + 1)`` parameters and remains
    bilinear in the two endpoint feature vectors.

    At prediction time, offset zero selects the shipped ROOT feature.  Any
    nonzero supplied offset for which ``resolve_governor_index`` returns
    ``None`` has no governor feature, so that position returns ``"dep"``
    directly without scoring.  This invalid/out-of-sentence head is the sole
    OOV fallback trigger; labels scored on valid arcs are always drawn from the
    sorted train relation vocabulary.
    """

    def __init__(self, config: RelationLabelerConfig | None = None) -> None:
        self.config = config or RelationLabelerConfig()
        if self.config.rank < 1:
            raise ValueError("rank must be positive")
        if self.config.epochs < 1:
            raise ValueError("epochs must be positive")
        attachment_config = BilinearConfig(
            rank=self.config.rank,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            l2_penalty=self.config.l2_penalty,
            seed=self.config.seed,
            surface_buckets=self.config.surface_buckets,
            position_buckets=self.config.position_buckets,
        )
        self._feature_builder = BilinearAttachmentScorer(attachment_config)
        self.feature_dim = self._feature_builder.feature_dim
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

    def phi(
        self,
        token: str,
        predicted_pos: str,
        index: int,
        sentence_size: int,
        *,
        root: bool = False,
    ) -> np.ndarray:
        """Return the exact shipped serve-honest attachment feature vector."""
        return self._feature_builder.phi(
            token,
            predicted_pos,
            index,
            sentence_size,
            root=root,
        )

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
        if any(value is None for value in (self.U, self.V, self.C, self.P, self.Q, self.b)):
            raise RuntimeError("relation labeler used before fit")
        assert self.U is not None
        assert self.V is not None
        assert self.C is not None
        assert self.P is not None
        assert self.Q is not None
        assert self.b is not None
        return self.U, self.V, self.C, self.P, self.Q, self.b

    def _dependent_features(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
    ) -> np.ndarray:
        dependent, _governors = self._feature_builder._feature_matrices(
            tokens, predicted_pos
        )
        return dependent

    def _training_governors(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        gold_offsets: Sequence[int],
    ) -> np.ndarray:
        if len(tokens) != len(gold_offsets):
            raise ValueError("tokens and gold heads must align")
        size = len(tokens)
        features: list[np.ndarray] = []
        for index, offset in enumerate(gold_offsets):
            if offset == 0:
                features.append(self.phi("<ROOT>", "<UNK>", 0, size, root=True))
                continue
            governor = resolve_governor_index(index, offset, size)
            if governor is None or governor == index:
                raise ValueError(f"invalid training head at token {index}: {offset}")
            features.append(
                self.phi(
                    tokens[governor],
                    predicted_pos[governor],
                    governor,
                    size,
                )
            )
        return np.stack(features)

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

    def _train_sentence(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        gold_offsets: Sequence[int],
        gold_deprels: Sequence[str],
    ) -> float:
        if not (
            len(tokens)
            == len(predicted_pos)
            == len(gold_offsets)
            == len(gold_deprels)
        ):
            raise ValueError("relation-label training fields must align")
        dependent = self._dependent_features(tokens, predicted_pos)
        governor = self._training_governors(tokens, predicted_pos, gold_offsets)
        parameters = self._parameters()
        u, v, core, dependent_affine, governor_affine, _bias = parameters
        logits, dependent_projection, governor_projection, interaction = self._logits(
            dependent, governor, parameters
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
                - logits[np.arange(len(tokens)), targets]
            )
        )

        score_gradient = probabilities
        score_gradient[np.arange(len(tokens)), targets] -= 1.0
        score_gradient /= len(tokens)
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
        grad_norm = np.sqrt(sum(float(np.square(gradient).sum()) for gradient in gradients))
        if grad_norm > 5.0:
            scale = 5.0 / grad_norm
            gradients = tuple(gradient * scale for gradient in gradients)

        learning_rate = self.config.learning_rate
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter -= learning_rate * gradient
        return loss

    def fit(
        self,
        train_sentences: Sequence[Sentence],
        train_heads: Mapping[str, HeadDeprelRecord],
        train_predicted_pos: Mapping[str, Sequence[str]],
    ) -> None:
        """Fit relation cross-entropy on gold train arcs and R1-predicted POS."""
        if not train_sentences:
            raise ValueError("cannot fit on empty training data")
        relations: list[str] = []
        for sentence in train_sentences:
            head = aligned_head_record(sentence, train_heads)
            if not isinstance(head, HeadDeprelRecord):
                raise TypeError("labeled dependency record required")
            relations.extend(head.deprel)
        self._initialize_parameters(relations)
        self.epoch_losses = []
        for _epoch in range(self.config.epochs):
            order = self._rng.permutation(len(train_sentences))
            losses: list[float] = []
            for sentence_index in order:
                sentence = train_sentences[int(sentence_index)]
                head = aligned_head_record(sentence, train_heads)
                if not isinstance(head, HeadDeprelRecord):
                    raise TypeError("labeled dependency record required")
                losses.append(
                    self._train_sentence(
                        sentence.tokens,
                        train_predicted_pos[sentence.sent_id],
                        head.head_offset,
                        head.deprel,
                    )
                )
            self.epoch_losses.append(float(np.mean(losses)))

    def fit_on_supplied_governors(
        self,
        train_sentences: Sequence[Sentence],
        governor_offsets: Mapping[str, Sequence[int]],
        deprel_targets: Mapping[str, Sequence[str]],
        predicted_pos: Mapping[str, Sequence[str]],
    ) -> None:
        """Fit on caller-supplied governors while retaining gold relation targets."""
        if not train_sentences:
            raise ValueError("cannot fit on empty training data")
        relations = [
            label
            for sentence in train_sentences
            for label in deprel_targets[sentence.sent_id]
        ]
        self._initialize_parameters(relations)
        self.epoch_losses = []
        for _epoch in range(self.config.epochs):
            order = self._rng.permutation(len(train_sentences))
            losses: list[float] = []
            for sentence_index in order:
                sentence = train_sentences[int(sentence_index)]
                losses.append(
                    self._train_sentence(
                        sentence.tokens,
                        predicted_pos[sentence.sent_id],
                        governor_offsets[sentence.sent_id],
                        deprel_targets[sentence.sent_id],
                    )
                )
            self.epoch_losses.append(float(np.mean(losses)))

    def predict(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[str, ...]:
        """Label only the supplied arcs; no gold dependency field is accepted or read."""
        if not (len(tokens) == len(predicted_pos) == len(head_offsets)):
            raise ValueError("tokens, predicted POS, and supplied heads must align")
        if not tokens:
            return ()
        parameters = self._parameters()
        dependent = self._dependent_features(tokens, predicted_pos)
        size = len(tokens)
        predictions: list[str] = []
        for index, offset in enumerate(head_offsets):
            if offset == 0:
                governor = self.phi("<ROOT>", "<UNK>", 0, size, root=True)
            else:
                governor_index = resolve_governor_index(index, offset, size)
                if governor_index is None:
                    predictions.append("dep")
                    continue
                governor = self.phi(
                    tokens[governor_index],
                    predicted_pos[governor_index],
                    governor_index,
                    size,
                )
            logits, _dep_projection, _gov_projection, _interaction = self._logits(
                dependent[index : index + 1], governor[None, :], parameters
            )
            predictions.append(self.relations[int(np.argmax(logits[0]))])
        return tuple(predictions)


def coarse_deprel(label: str) -> str:
    """Coarsen a relation by dropping the first colon and everything after it."""
    return label.split(":", 1)[0]


def score_relation_partition(
    predicted_deprel: Sequence[str],
    gold_deprel: Sequence[str],
) -> dict[str, Any]:
    """Score exact labels in two gold-coarse buckets; exclude all other relations."""
    if len(predicted_deprel) != len(gold_deprel):
        raise ValueError("predicted and gold deprels must align")
    selected: dict[str, tuple[list[str], list[str]]] = {
        "pairwise_sensitive": ([], []),
        "local": ([], []),
    }
    excluded = 0
    for predicted, gold in zip(predicted_deprel, gold_deprel, strict=True):
        coarse_gold = coarse_deprel(gold)
        if coarse_gold in PAIRWISE_SENSITIVE_DEPRELS:
            bucket = "pairwise_sensitive"
        elif coarse_gold in LOCAL_DEPRELS:
            bucket = "local"
        else:
            excluded += 1
            continue
        selected[bucket][0].append(predicted)
        selected[bucket][1].append(gold)
    result: dict[str, Any] = {"excluded_n": excluded}
    for bucket, (predicted, gold) in selected.items():
        if not gold:
            raise ValueError(f"no relations in partition bucket {bucket}")
        result[bucket] = {"accuracy": accuracy(predicted, gold), "n": len(gold)}
    return result


CaseRunner = Callable[..., Any]


def evaluate_matched_labelers(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    biaffine_by_seed: Mapping[int, BilinearRelationLabeler],
    case_student: GermanR1Student,
    case_runner: CaseRunner = run_predicted_case_sentence,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Evaluate unary and all biaffine seeds on identical in-memory L0 heads."""
    arm_names = ("unary", *(f"biaffine_{seed}" for seed in biaffine_by_seed))
    accumulated: dict[str, dict[str, list[Any]]] = {
        arm: {"head": [], "deprel": [], "case": []} for arm in arm_names
    }
    gold_head: list[int] = []
    gold_deprel: list[str] = []
    gold_case: list[str] = []

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
            **{
                f"biaffine_{seed}": DependencyPrediction(
                    common_heads,
                    labeler.predict(sentence.tokens, pos, common_heads),
                )
                for seed, labeler in biaffine_by_seed.items()
            },
        }
        baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        eligible = [
            index
            for index, gold_label in enumerate(sentence.targets["morph_case"])
            if gold_label != "-"
        ]
        for arm, dependency in predictions.items():
            case_result = case_runner(
                case_student,
                sentence.tokens,
                baseline,
                pos,
                dependency.head_offset,
                dependency.deprel,
                eligible,
            )
            accumulated[arm]["head"].extend(dependency.head_offset)
            accumulated[arm]["deprel"].extend(dependency.deprel)
            accumulated[arm]["case"].extend(
                case_result.predictions[index] for index in eligible
            )
        gold_head.extend(head.head_offset)
        gold_deprel.extend(head.deprel)
        gold_case.extend(sentence.targets["morph_case"][index] for index in eligible)

    metrics: dict[str, dict[str, Any]] = {}
    coarse_gold = tuple(coarse_deprel(label) for label in gold_deprel)
    for arm in arm_names:
        predicted_heads = accumulated[arm]["head"]
        predicted_relations = accumulated[arm]["deprel"]
        strict = score_prediction_sequences(
            predicted_heads,
            predicted_relations,
            gold_head,
            gold_deprel,
        )
        coarse = score_dependencies(
            predicted_heads,
            tuple(coarse_deprel(label) for label in predicted_relations),
            gold_head,
            coarse_gold,
        )
        case_bearing_accuracy, case_bearing_n = score_case_bearing_deprels(
            predicted_relations, gold_deprel
        )
        metrics[arm] = {
            "uas": strict["uas"],
            "las_strict": strict["las"],
            "las_coarse": coarse.las,
            "deprel_only_accuracy": strict["deprel_only_accuracy"],
            "case_bearing_deprel_accuracy": case_bearing_accuracy,
            "case_bearing_deprel_n": case_bearing_n,
            "serve_honest_morph_case": accuracy(accumulated[arm]["case"], gold_case),
            "serve_honest_morph_case_n": len(gold_case),
            "tokens": strict["tokens"],
            "partition": score_relation_partition(predicted_relations, gold_deprel),
        }

    unary_uas = float(metrics["unary"]["uas"])
    for seed in biaffine_by_seed:
        seed_uas = float(metrics[f"biaffine_{seed}"]["uas"])
        assert seed_uas == unary_uas, (
            f"matched-labeler UAS invariant failed for seed {seed}: "
            f"unary={unary_uas}, biaffine={seed_uas}"
        )
    return metrics["unary"], {
        seed: metrics[f"biaffine_{seed}"] for seed in biaffine_by_seed
    }


def _metric_gains(
    biaffine: Mapping[str, Any], unary: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "deprel_only_accuracy": float(biaffine["deprel_only_accuracy"])
        - float(unary["deprel_only_accuracy"]),
        "las_strict": float(biaffine["las_strict"]) - float(unary["las_strict"]),
        "las_coarse": float(biaffine["las_coarse"]) - float(unary["las_coarse"]),
        "serve_honest_morph_case": float(biaffine["serve_honest_morph_case"])
        - float(unary["serve_honest_morph_case"]),
        "pairwise_sensitive": float(
            biaffine["partition"]["pairwise_sensitive"]["accuracy"]
        )
        - float(unary["partition"]["pairwise_sensitive"]["accuracy"]),
        "local": float(biaffine["partition"]["local"]["accuracy"])
        - float(unary["partition"]["local"]["accuracy"]),
    }


def mean_biaffine_metrics(
    biaffine_by_seed: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Average seed-varying scores while retaining shared integer denominators."""
    if not biaffine_by_seed:
        raise ValueError("cannot average zero seeds")
    metrics = list(biaffine_by_seed.values())
    first = metrics[0]

    def mean(key: str) -> float:
        return float(np.mean([float(metric[key]) for metric in metrics]))

    for key in ("tokens", "case_bearing_deprel_n", "serve_honest_morph_case_n"):
        if any(metric[key] != first[key] for metric in metrics[1:]):
            raise AssertionError(f"seed denominators differ for {key}")
    partition: dict[str, Any] = {"excluded_n": first["partition"]["excluded_n"]}
    for bucket in ("pairwise_sensitive", "local"):
        if any(
            metric["partition"][bucket]["n"] != first["partition"][bucket]["n"]
            for metric in metrics[1:]
        ):
            raise AssertionError(f"seed partition denominators differ for {bucket}")
        partition[bucket] = {
            "accuracy": float(
                np.mean(
                    [metric["partition"][bucket]["accuracy"] for metric in metrics]
                )
            ),
            "n": first["partition"][bucket]["n"],
        }
    return {
        "uas": mean("uas"),
        "las_strict": mean("las_strict"),
        "las_coarse": mean("las_coarse"),
        "deprel_only_accuracy": mean("deprel_only_accuracy"),
        "case_bearing_deprel_accuracy": mean("case_bearing_deprel_accuracy"),
        "case_bearing_deprel_n": first["case_bearing_deprel_n"],
        "serve_honest_morph_case": mean("serve_honest_morph_case"),
        "serve_honest_morph_case_n": first["serve_honest_morph_case_n"],
        "tokens": first["tokens"],
        "partition": partition,
    }


def preregistered_read(
    unary: Mapping[str, Any],
    biaffine_by_seed: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply signed preregistration section 5, including its stated precedence."""
    if not biaffine_by_seed:
        raise ValueError("the preregistered read requires at least one seed")
    mean_metrics = mean_biaffine_metrics(biaffine_by_seed)
    las_gains = [
        float(metric["las_strict"]) - float(unary["las_strict"])
        for metric in biaffine_by_seed.values()
    ]
    las_gain_mean = float(np.mean(las_gains))
    partition_gains = {
        bucket: float(mean_metrics["partition"][bucket]["accuracy"])
        - float(unary["partition"][bucket]["accuracy"])
        for bucket in ("pairwise_sensitive", "local")
    }
    case_unary = float(unary["serve_honest_morph_case"])
    case_biaffine_mean = float(mean_metrics["serve_honest_morph_case"])
    all_seeds_positive = all(gain > 0.0 for gain in las_gains)
    case_improved = case_biaffine_mean > case_unary
    concentrated = partition_gains["pairwise_sensitive"] > partition_gains["local"]

    if las_gain_mean <= 0.0:
        verdict = "HALTED"
        reason = "mean strict LAS gain is non-positive"
    elif (
        las_gain_mean < 0.03
        or not all_seeds_positive
        or not case_improved
        or not concentrated
    ):
        verdict = "IN-BETWEEN"
        failed: list[str] = []
        if las_gain_mean < 0.03:
            failed.append("mean strict LAS gain is below +0.03")
        if not all_seeds_positive:
            failed.append("strict LAS gain is not positive at every seed")
        if not case_improved:
            failed.append("mean serve-honest case does not improve")
        if not concentrated:
            failed.append("pairwise-sensitive gain is not greater than local gain")
        reason = "; ".join(failed)
    else:
        verdict = "FIRES"
        reason = "all preregistered capability conditions are met"

    per_seed_text = ", ".join(
        f"seed {seed}={gain:+.4f}"
        for seed, gain in zip(biaffine_by_seed, las_gains, strict=True)
    )
    text = (
        f"{verdict}: mean strict LAS gain={las_gain_mean:+.4f} "
        f"({per_seed_text}); pairwise-sensitive deprel gain="
        f"{partition_gains['pairwise_sensitive']:+.4f} versus local="
        f"{partition_gains['local']:+.4f}; mean serve-honest case "
        f"moves from {case_unary:.4f} to {case_biaffine_mean:.4f}. {reason}."
    )
    return {
        "verdict": verdict,
        "las_gain_mean": las_gain_mean,
        "las_gain_per_seed": {
            str(seed): gain
            for seed, gain in zip(biaffine_by_seed, las_gains, strict=True)
        },
        "partition_gains": partition_gains,
        "case_unary": case_unary,
        "case_biaffine_mean": case_biaffine_mean,
        "text": text,
    }


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit on train, use dev only for L0's window, then read test once."""
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

    print("DEV: loading GSD dev once for the shipped L0 window choice only.", flush=True)
    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_heads = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    window, dev_coverage, coverage_curve = choose_window(dev_heads)
    del dev

    l0_student = GermanR3DependencyStudent(window)
    l0_student.fit(train, train_heads, train_pos)
    base_config = RelationLabelerConfig()
    biaffine_by_seed: dict[int, BilinearRelationLabeler] = {}
    for seed in SEEDS:
        config = replace(base_config, seed=seed)
        labeler = BilinearRelationLabeler(config)
        print(
            f"FIT: biaffine-label seed={seed} rank={config.rank} "
            f"epochs={config.epochs} lr={config.learning_rate}.",
            flush=True,
        )
        labeler.fit(train, train_heads, train_pos)
        print(
            f"FIT: seed={seed} epoch losses="
            + ", ".join(f"{loss:.6f}" for loss in labeler.epoch_losses),
            flush=True,
        )
        biaffine_by_seed[seed] = labeler

    guard = CampaignTestReadGuard()
    print("FINAL EVAL: reading shared test tasks once and head_deprel test once.", flush=True)
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_head_records = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads = head_deprel_by_sent(test_head_records)
    test_pos = _predicted_pos_by_sentence(r1_student, test)
    unary, biaffine_metrics = evaluate_matched_labelers(
        test,
        test_heads,
        test_pos,
        l0_student,
        biaffine_by_seed,
        r1_student,
    )
    guard.assert_complete()

    mean_metrics = mean_biaffine_metrics(biaffine_metrics)
    per_seed = {
        str(seed): {
            "metrics": biaffine_metrics[seed],
            "gains_vs_unary": _metric_gains(biaffine_metrics[seed], unary),
            "epoch_losses": biaffine_by_seed[seed].epoch_losses,
        }
        for seed in SEEDS
    }
    mean_gains = _metric_gains(mean_metrics, unary)
    read = preregistered_read(unary, biaffine_metrics)
    first_labeler = biaffine_by_seed[SEEDS[0]]
    if any(labeler.relations != first_labeler.relations for labeler in biaffine_by_seed.values()):
        raise AssertionError("relation vocabulary differs across seeds")
    scoreboard: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "scope": SCOPE,
        "measurement": (
            "serve-honest two-arm labeler comparison; same raw L0 predicted heads and "
            "same case cascade in every arm; no tree decode"
        ),
        "serve_honest_features": [
            "R1-predicted POS one-hot",
            "hashed lowercase surface form",
            "surface title/uppercase/digit/punctuation shape flags",
            "normalized sentence-position one-hot",
            "ROOT flag",
        ],
        "gold_usage": (
            "GSD train gold head offsets choose fit-time governors and train gold deprels are "
            "cross-entropy targets; test gold heads/deprels/case are final scoring oracles only; "
            "no half-oracle fields are supplied"
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
            "rank": base_config.rank,
            "epochs": base_config.epochs,
            "learning_rate": base_config.learning_rate,
            "l2_penalty": base_config.l2_penalty,
            "seeds": list(SEEDS),
            "surface_buckets": base_config.surface_buckets,
            "position_buckets": base_config.position_buckets,
            "gradient_norm_clip": 5.0,
        },
        "parameterization": (
            "shared U,V endpoint projections with relation-specific diagonal rank core, "
            "dependent/governor affine terms, and bias"
        ),
        "feature_dim": first_labeler.feature_dim,
        "relation_vocabulary_size": len(first_labeler.relations),
        "relation_vocabulary": list(first_labeler.relations),
        "las_coarse_rule": "split each predicted and gold deprel at its first colon",
        "partition_rule": (
            "bucket by gold coarse deprel; exact-label accuracy within each bucket; "
            "relations in neither preregistered set are excluded"
        ),
        "invalid_head_fallback": (
            "nonzero supplied offset resolving outside the sentence emits dep without scoring"
        ),
        "arms": {
            "unary": unary,
            "biaffine_by_seed": per_seed,
            "biaffine_mean": {
                "metrics": mean_metrics,
                "gains_vs_unary": mean_gains,
            },
        },
        "partition_gains": read["partition_gains"],
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
    """Print all preregistered metrics, gains, and the computed section-5 read."""
    print("\nBIAFFINE RELATION-LABELER — GSD TEST")
    print(f"run_tag: {scoreboard['run_tag']}")
    print(f"scope: {scoreboard['scope']}")
    config = scoreboard["fixed_hyperparameters"]
    print(
        f"config: rank={config['rank']} epochs={config['epochs']} "
        f"lr={config['learning_rate']} l2={config['l2_penalty']} "
        f"seeds={config['seeds']} feature_dim={scoreboard['feature_dim']} "
        f"relations={scoreboard['relation_vocabulary_size']}"
    )
    print("matched control: every arm labels identical raw L0 predicted heads; no CLE/tree decode")
    print("serve honesty: R1-predicted POS + shipped surface/shape/position/ROOT phi only")
    print("LAS coarse: predicted and gold labels split at the first ':'")
    print("partition: bucket by gold coarse label; labels in neither set are excluded")
    print(
        "\narm             UAS       LAS-strict LAS-coarse deprel-only case-bearing (n)  "
        "case (n)         pairwise (n)     local (n)"
    )
    arms = scoreboard["arms"]
    _print_metric_row("unary", arms["unary"])
    for seed in SEEDS:
        _print_metric_row(
            f"biaffine s{seed}", arms["biaffine_by_seed"][str(seed)]["metrics"]
        )
    _print_metric_row("biaffine mean", arms["biaffine_mean"]["metrics"])

    print(
        "\nseed gains vs unary   deprel-only LAS-strict LAS-coarse case      "
        "pairwise  local"
    )
    for seed in SEEDS:
        gains = arms["biaffine_by_seed"][str(seed)]["gains_vs_unary"]
        print(
            f"{seed:<21} {gains['deprel_only_accuracy']:+.6f}   "
            f"{gains['las_strict']:+.6f}   {gains['las_coarse']:+.6f}   "
            f"{gains['serve_honest_morph_case']:+.6f}  "
            f"{gains['pairwise_sensitive']:+.6f}  {gains['local']:+.6f}"
        )
    gains = arms["biaffine_mean"]["gains_vs_unary"]
    print(
        f"mean                  {gains['deprel_only_accuracy']:+.6f}   "
        f"{gains['las_strict']:+.6f}   {gains['las_coarse']:+.6f}   "
        f"{gains['serve_honest_morph_case']:+.6f}  "
        f"{gains['pairwise_sensitive']:+.6f}  {gains['local']:+.6f}"
    )
    print(
        f"partition excluded n: "
        f"{arms['unary']['partition']['excluded_n']} "
        "(not counted in either bucket)"
    )
    print("\nTHE PRE-REGISTERED READ")
    print(scoreboard["read"]["text"])
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
