"""Empirical soft-ceiling probe for cheap German attachment levers.

The four rungs deliberately share the shipped R1 POS model and R3 relation
classifier.  L0 is the shipped greedy R3 dependency student; L1 adds a
single-root non-projective Chu-Liu-Edmonds decode; L2 replaces only the head
model with a small low-rank bilinear scorer; and L3 adds the same tree decode
to L2.  Gold fields are training targets, evaluation labels, or explicit
half-oracle diagnostic inputs only.
"""

from __future__ import annotations

import hashlib
import json
import math
import string
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    _relation_key_levels,
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
    full_oracle_case_sentence,
    head_deprel_by_sent,
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "attach_levers_codex.json"
RUN_TAG = "empirical"
SCOPE = "soft-ceiling"
SEED = 0
RANK = 16
EPOCHS = 5
LEARNING_RATE = 0.04
L2_PENALTY = 1e-5
POSITION_BUCKETS = 5
SURFACE_BUCKETS = 32
POS_TAGS = (
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "<UNK>",
)
RUNG_NAMES = ("L0", "L1", "L2", "L3")
CASE_BEARING_DEPRELS = frozenset(
    {"nsubj", "nsubj:pass", "obj", "iobj", "obl:arg", "nmod", "case", "cop"}
)


@dataclass(frozen=True)
class BilinearConfig:
    """Fixed-a-priori soft attachment configuration; no test tuning is performed."""

    rank: int = RANK
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    l2_penalty: float = L2_PENALTY
    seed: int = SEED
    surface_buckets: int = SURFACE_BUCKETS
    position_buckets: int = POSITION_BUCKETS


def _find_cycle(parents: np.ndarray, root: int) -> list[int] | None:
    """Return one directed parent cycle, excluding the parentless root."""
    size = len(parents)
    globally_done: set[int] = set()
    for start in range(size):
        if start == root or start in globally_done:
            continue
        order: dict[int, int] = {}
        path: list[int] = []
        node = start
        while node != root and node not in globally_done and node not in order:
            order[node] = len(path)
            path.append(node)
            node = int(parents[node])
        if node in order:
            return path[order[node] :]
        globally_done.update(path)
    return None


def _chu_liu_edmonds_square(scores: np.ndarray, root: int) -> np.ndarray:
    """Decode one maximum arborescence from ``scores[dependent, governor]``."""
    size = scores.shape[0]
    parents = np.full(size, -1, dtype=np.int64)
    for dependent in range(size):
        if dependent == root:
            continue
        governor = int(np.argmax(scores[dependent]))
        if not np.isfinite(scores[dependent, governor]):
            raise ValueError(f"dependent {dependent} has no finite governor")
        parents[dependent] = governor

    cycle = _find_cycle(parents, root)
    if cycle is None:
        return parents

    cycle_set = set(cycle)
    outside = [node for node in range(size) if node not in cycle_set]
    old_to_new = {node: index for index, node in enumerate(outside)}
    supernode = len(outside)
    contracted_size = supernode + 1
    contracted = np.full((contracted_size, contracted_size), -np.inf, dtype=np.float64)

    for dependent in outside:
        for governor in outside:
            contracted[old_to_new[dependent], old_to_new[governor]] = scores[
                dependent, governor
            ]

    outgoing_choice: dict[int, int] = {}
    for dependent in outside:
        best_governor = max(cycle, key=lambda governor: scores[dependent, governor])
        outgoing_choice[dependent] = best_governor
        contracted[old_to_new[dependent], supernode] = scores[dependent, best_governor]

    incoming_choice: dict[int, int] = {}
    for governor in outside:
        best_dependent = max(
            cycle,
            key=lambda dependent: (
                scores[dependent, governor] - scores[dependent, parents[dependent]]
            ),
        )
        incoming_choice[governor] = best_dependent
        contracted[supernode, old_to_new[governor]] = (
            scores[best_dependent, governor] - scores[best_dependent, parents[best_dependent]]
        )

    np.fill_diagonal(contracted, -np.inf)
    contracted_root = old_to_new[root]
    contracted_parents = _chu_liu_edmonds_square(contracted, contracted_root)

    expanded = parents.copy()
    for dependent in outside:
        if dependent == root:
            continue
        contracted_governor = int(contracted_parents[old_to_new[dependent]])
        expanded[dependent] = (
            outgoing_choice[dependent]
            if contracted_governor == supernode
            else outside[contracted_governor]
        )

    entering_governor_index = int(contracted_parents[supernode])
    if entering_governor_index < 0 or entering_governor_index >= len(outside):
        raise RuntimeError("contracted cycle did not receive an external governor")
    entering_governor = outside[entering_governor_index]
    expanded[incoming_choice[entering_governor]] = entering_governor
    return expanded


def chu_liu_edmonds(edge_scores: np.ndarray) -> tuple[int, ...]:
    """Return one single-root non-projective maximum arborescence.

    ``edge_scores`` has one row per token and columns ``ROOT, token_0, ...``.
    The returned governor columns use that same convention.  Standard CLE
    permits multiple ROOT children, so this wrapper enumerates the unique ROOT
    child, forces that edge, and chooses the best resulting arborescence.  This
    is exact for the requested single-root constraint.
    """
    scores = np.asarray(edge_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != scores.shape[0] + 1:
        raise ValueError("edge scores must have shape n x (n+1)")
    token_count = scores.shape[0]
    if not token_count:
        raise ValueError("cannot decode an empty sentence")

    square = np.full((token_count + 1, token_count + 1), -np.inf, dtype=np.float64)
    square[1:, :] = scores
    for dependent in range(1, token_count + 1):
        square[dependent, dependent] = -np.inf

    best_parents: np.ndarray | None = None
    best_score = -np.inf
    for root_child in range(1, token_count + 1):
        if not np.isfinite(square[root_child, 0]):
            continue
        constrained = square.copy()
        constrained[1:, 0] = -np.inf
        forced_root_score = square[root_child, 0]
        constrained[root_child, :] = -np.inf
        constrained[root_child, 0] = forced_root_score
        try:
            parents = _chu_liu_edmonds_square(constrained, root=0)
        except ValueError:
            continue
        total = sum(square[dependent, parents[dependent]] for dependent in range(1, token_count + 1))
        if total > best_score:
            best_score = float(total)
            best_parents = parents

    if best_parents is None:
        raise ValueError("no finite single-root arborescence exists")
    return tuple(int(parent) for parent in best_parents[1:])


def governor_columns_to_offsets(governors: Sequence[int]) -> tuple[int, ...]:
    """Convert ROOT/token governor columns to the shipped relative-offset convention."""
    size = len(governors)
    offsets: list[int] = []
    for index, governor_column in enumerate(governors):
        if governor_column < 0 or governor_column > size:
            raise ValueError(f"invalid governor column: {governor_column}")
        offsets.append(0 if governor_column == 0 else governor_column - 1 - index)
    return tuple(offsets)


def greedy_head_offsets(edge_scores: np.ndarray) -> tuple[int, ...]:
    """Independently select each token's highest-scoring valid governor."""
    scores = np.asarray(edge_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != scores.shape[0] + 1:
        raise ValueError("edge scores must have shape n x (n+1)")
    return governor_columns_to_offsets(tuple(int(value) for value in np.argmax(scores, axis=1)))


def l0_edge_scores(
    student: GermanR3DependencyStudent,
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
) -> np.ndarray:
    """Derive serve-honest CLE scores from L0's hard count-table decision.

    The shipped student exposes a hard offset rather than candidate logits.  We
    therefore give its exact predicted offset score 1.0 and score every other
    valid candidate by ``0.10 / (1 + |candidate - prediction|)`` plus a small
    distance-to-dependent preference.  The signal uses only L0's fitted count
    tables, predicted POS, sentence length, and candidate distance; no gold
    head or relation participates.
    """
    predicted_offsets = student.predict_heads(tokens, predicted_pos)
    size = len(tokens)
    scores = np.full((size, size + 1), -np.inf, dtype=np.float64)
    for index, predicted in enumerate(predicted_offsets):
        candidates = [(0, 0), *((governor + 1, governor - index) for governor in range(size))]
        for column, offset in candidates:
            if column == index + 1:
                continue
            scores[index, column] = (
                1.0
                if offset == predicted
                else 0.10 / (1.0 + abs(offset - predicted)) + 0.01 / (1.0 + abs(offset))
            )
    return scores


class BilinearAttachmentScorer:
    """Small serve-honest low-rank bilinear head scorer trained by sentence SGD.

    Token ``phi`` contains fixed predicted-POS, hashed lowercase form, surface
    shape, and normalized sentence-position one-hots.  ROOT has its own flag.
    Consequently ``s(i,j) = phi(i)^T (U @ V.T) phi(j) + b`` has rank at most
    ``config.rank`` and learns positional interactions through the two position
    buckets.  Five epochs, rank 16, learning rate 0.04, and seed 0 are fixed
    a priori for this probe rather than selected on test.
    """

    def __init__(self, config: BilinearConfig | None = None) -> None:
        self.config = config or BilinearConfig()
        if self.config.rank < 1:
            raise ValueError("rank must be positive")
        self._pos_index = {tag: index for index, tag in enumerate(POS_TAGS)}
        self.feature_dim = (
            3
            + len(POS_TAGS)
            + self.config.surface_buckets
            + 4
            + self.config.position_buckets
        )
        self._rng = np.random.default_rng(self.config.seed)
        scale = 1.0 / np.sqrt(self.feature_dim)
        self.U = self._rng.normal(0.0, scale, (self.feature_dim, self.config.rank))
        self.V = self._rng.normal(0.0, scale, (self.feature_dim, self.config.rank))
        self.b = 0.0
        self.epoch_losses: list[float] = []

    @property
    def W(self) -> np.ndarray:
        return self.U @ self.V.T

    def phi(
        self,
        token: str,
        predicted_pos: str,
        index: int,
        sentence_size: int,
        *,
        root: bool = False,
    ) -> np.ndarray:
        """Build one dense, small feature vector without consulting gold fields."""
        if sentence_size < 1:
            raise ValueError("sentence size must be positive")
        vector = np.zeros(self.feature_dim, dtype=np.float64)
        vector[0] = 1.0
        vector[2 if root else 1] = 1.0
        if root:
            return vector

        cursor = 3
        pos_index = self._pos_index.get(predicted_pos, self._pos_index["<UNK>"])
        vector[cursor + pos_index] = 1.0
        cursor += len(POS_TAGS)

        digest = hashlib.blake2b(token.lower().encode("utf-8"), digest_size=8).digest()
        surface_index = int.from_bytes(digest, "little") % self.config.surface_buckets
        vector[cursor + surface_index] = 1.0
        cursor += self.config.surface_buckets

        vector[cursor : cursor + 4] = (
            float(token.istitle()),
            float(token.isupper()),
            float(token.isdigit()),
            float(bool(token) and all(character in string.punctuation for character in token)),
        )
        cursor += 4
        position_bucket = min(
            self.config.position_buckets - 1,
            (self.config.position_buckets * index) // sentence_size,
        )
        vector[cursor + position_bucket] = 1.0
        return vector

    def _feature_matrices(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(tokens) != len(predicted_pos):
            raise ValueError("tokens and predicted POS must align")
        if not tokens:
            raise ValueError("cannot score an empty sentence")
        size = len(tokens)
        token_features = np.stack(
            [
                self.phi(token, pos, index, size)
                for index, (token, pos) in enumerate(
                    zip(tokens, predicted_pos, strict=True)
                )
            ]
        )
        root_feature = self.phi("<ROOT>", "<UNK>", 0, size, root=True)
        governor_features = np.vstack((root_feature, token_features))
        return token_features, governor_features

    def score_matrix(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
    ) -> np.ndarray:
        dependent_features, governor_features = self._feature_matrices(tokens, predicted_pos)
        scores = (dependent_features @ self.U) @ (governor_features @ self.V).T + self.b
        for index in range(len(tokens)):
            scores[index, index + 1] = -np.inf
        return scores

    @staticmethod
    def _gold_columns(offsets: Sequence[int]) -> np.ndarray:
        columns: list[int] = []
        size = len(offsets)
        for index, offset in enumerate(offsets):
            governor = resolve_governor_index(index, offset, size)
            if offset == 0:
                columns.append(0)
            elif governor is None or governor == index:
                raise ValueError(f"invalid training head at token {index}: {offset}")
            else:
                columns.append(governor + 1)
        return np.asarray(columns, dtype=np.int64)

    def sentence_loss(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        gold_offsets: Sequence[int],
    ) -> float:
        scores = self.score_matrix(tokens, predicted_pos)
        gold_columns = self._gold_columns(gold_offsets)
        maxima = np.max(scores, axis=1, keepdims=True)
        log_normalizers = maxima[:, 0] + np.log(np.exp(scores - maxima).sum(axis=1))
        return float(np.mean(log_normalizers - scores[np.arange(len(tokens)), gold_columns]))

    def mean_loss(
        self,
        sentences: Sequence[Sentence],
        heads: Mapping[str, HeadDeprelRecord],
        predicted_pos: Mapping[str, Sequence[str]],
    ) -> float:
        losses = []
        for sentence in sentences:
            head = aligned_head_record(sentence, heads)
            if not isinstance(head, HeadDeprelRecord):
                raise TypeError("labeled dependency record required")
            losses.append(
                self.sentence_loss(sentence.tokens, predicted_pos[sentence.sent_id], head.head_offset)
            )
        if not losses:
            raise ValueError("cannot score loss on empty training data")
        return float(np.mean(losses))

    def _train_sentence(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        gold_offsets: Sequence[int],
    ) -> float:
        dependent_features, governor_features = self._feature_matrices(tokens, predicted_pos)
        dependent_projection = dependent_features @ self.U
        governor_projection = governor_features @ self.V
        scores = dependent_projection @ governor_projection.T + self.b
        for index in range(len(tokens)):
            scores[index, index + 1] = -np.inf
        gold_columns = self._gold_columns(gold_offsets)
        maxima = np.max(scores, axis=1, keepdims=True)
        exponentiated = np.exp(scores - maxima)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        loss = float(
            np.mean(
                maxima[:, 0]
                + np.log(exponentiated.sum(axis=1))
                - scores[np.arange(len(tokens)), gold_columns]
            )
        )

        score_gradient = probabilities
        score_gradient[np.arange(len(tokens)), gold_columns] -= 1.0
        score_gradient /= len(tokens)
        grad_u = dependent_features.T @ (score_gradient @ governor_projection)
        grad_v = governor_features.T @ (score_gradient.T @ dependent_projection)
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
        sentences: Sequence[Sentence],
        heads: Mapping[str, HeadDeprelRecord],
        predicted_pos: Mapping[str, Sequence[str]],
    ) -> None:
        if not sentences:
            raise ValueError("cannot fit on empty training data")
        self.epoch_losses = []
        for _epoch in range(self.config.epochs):
            order = self._rng.permutation(len(sentences))
            losses: list[float] = []
            for sentence_index in order:
                sentence = sentences[int(sentence_index)]
                head = aligned_head_record(sentence, heads)
                if not isinstance(head, HeadDeprelRecord):
                    raise TypeError("labeled dependency record required")
                losses.append(
                    self._train_sentence(
                        sentence.tokens,
                        predicted_pos[sentence.sent_id],
                        head.head_offset,
                    )
                )
            self.epoch_losses.append(float(np.mean(losses)))


def predict_deprels(
    student: GermanR3DependencyStudent,
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    head_offsets: Sequence[int],
) -> tuple[str, ...]:
    """Reuse the shipped L0 relation table with a supplied predicted head sequence."""
    relation_table = student.deprel_table
    if relation_table is None:
        raise RuntimeError("deprel table used before fit")
    return tuple(
        relation_table.predict(_relation_key_levels(tokens, predicted_pos, index, offset))
        for index, offset in enumerate(head_offsets)
    )


def predict_ladder_sentence(
    sentence: Sentence,
    predicted_pos: Sequence[str],
    l0_student: GermanR3DependencyStudent,
    bilinear: BilinearAttachmentScorer,
) -> dict[str, DependencyPrediction]:
    """Run all four attachment rungs for one in-memory sentence."""
    l0 = l0_student.predict_sentence(sentence.tokens, predicted_pos)
    l0_scores = l0_edge_scores(l0_student, sentence.tokens, predicted_pos)
    l1_offsets = governor_columns_to_offsets(chu_liu_edmonds(l0_scores))
    bilinear_scores = bilinear.score_matrix(sentence.tokens, predicted_pos)
    l2_offsets = greedy_head_offsets(bilinear_scores)
    l3_offsets = governor_columns_to_offsets(chu_liu_edmonds(bilinear_scores))
    return {
        "L0": l0,
        "L1": DependencyPrediction(
            l1_offsets,
            predict_deprels(l0_student, sentence.tokens, predicted_pos, l1_offsets),
        ),
        "L2": DependencyPrediction(
            l2_offsets,
            predict_deprels(l0_student, sentence.tokens, predicted_pos, l2_offsets),
        ),
        "L3": DependencyPrediction(
            l3_offsets,
            predict_deprels(l0_student, sentence.tokens, predicted_pos, l3_offsets),
        ),
    }


def score_prediction_sequences(
    predicted_head_offset: Sequence[int],
    predicted_deprel: Sequence[str],
    gold_head_offset: Sequence[int],
    gold_deprel: Sequence[str],
) -> dict[str, float | int]:
    """Apply one shared UAS/LAS scorer and add relation-only accuracy."""
    dependencies = score_dependencies(
        predicted_head_offset,
        predicted_deprel,
        gold_head_offset,
        gold_deprel,
    )
    return {
        "uas": dependencies.uas,
        "las": dependencies.las,
        "deprel_only_accuracy": accuracy(predicted_deprel, gold_deprel),
        "tokens": dependencies.tokens,
    }


def score_case_bearing_deprels(
    predicted_deprel: Sequence[str],
    gold_deprel: Sequence[str],
) -> tuple[float, int]:
    """Score relations only where the gold label routes the case cascade."""
    if len(predicted_deprel) != len(gold_deprel):
        raise ValueError("predicted and gold deprels must align")
    selected = [
        predicted == gold
        for predicted, gold in zip(predicted_deprel, gold_deprel, strict=True)
        if gold in CASE_BEARING_DEPRELS
    ]
    if not selected:
        raise ValueError("no gold case-bearing deprels to score")
    return sum(selected) / len(selected), len(selected)


CaseRunner = Callable[..., Any]


def evaluate_ladder(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    bilinear: BilinearAttachmentScorer,
    case_student: GermanR1Student,
    case_runner: CaseRunner = run_predicted_case_sentence,
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, DependencyPrediction]]]:
    """Evaluate all rungs from the same already-loaded test sentence objects."""
    accumulated: dict[str, dict[str, list[Any]]] = {
        rung: {"head": [], "deprel": [], "case": []} for rung in RUNG_NAMES
    }
    gold_head: list[int] = []
    gold_deprel: list[str] = []
    gold_case: list[str] = []
    by_sentence: dict[str, dict[str, DependencyPrediction]] = {}

    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        predictions = predict_ladder_sentence(sentence, pos, l0_student, bilinear)
        by_sentence[sentence.sent_id] = predictions
        case_baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        eligible = [
            index
            for index, label in enumerate(sentence.targets["morph_case"])
            if label != "-"
        ]
        for rung, dependency in predictions.items():
            case_result = case_runner(
                case_student,
                sentence.tokens,
                case_baseline,
                pos,
                dependency.head_offset,
                dependency.deprel,
                eligible,
            )
            accumulated[rung]["head"].extend(dependency.head_offset)
            accumulated[rung]["deprel"].extend(dependency.deprel)
            accumulated[rung]["case"].extend(case_result.predictions)
        gold_head.extend(head.head_offset)
        gold_deprel.extend(head.deprel)
        gold_case.extend(sentence.targets["morph_case"])

    metrics: dict[str, dict[str, float | int]] = {}
    for rung in RUNG_NAMES:
        metric = score_prediction_sequences(
            accumulated[rung]["head"],
            accumulated[rung]["deprel"],
            gold_head,
            gold_deprel,
        )
        metric["serve_honest_morph_case"] = accuracy(accumulated[rung]["case"], gold_case)
        metrics[rung] = metric
    return metrics, by_sentence


def add_case_bearing_deprel_metrics(
    metrics: dict[str, dict[str, float | int]],
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    by_sentence: Mapping[str, Mapping[str, DependencyPrediction]],
) -> None:
    """Add the skeptic's narrow relation score using held in-memory predictions."""
    gold_deprel: list[str] = []
    predicted_deprel: dict[str, list[str]] = {rung: [] for rung in RUNG_NAMES}
    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        gold_deprel.extend(head.deprel)
        for rung in RUNG_NAMES:
            predicted_deprel[rung].extend(by_sentence[sentence.sent_id][rung].deprel)

    for rung in RUNG_NAMES:
        relation_accuracy, relation_n = score_case_bearing_deprels(
            predicted_deprel[rung], gold_deprel
        )
        metrics[rung]["case_bearing_deprel_accuracy"] = relation_accuracy
        metrics[rung]["case_bearing_deprel_n"] = relation_n


def half_oracle_dependency_pairs(
    predicted: DependencyPrediction,
    oracle: HeadDeprelRecord,
) -> dict[str, DependencyPrediction]:
    """Construct the four predicted/oracle dependency field combinations."""
    if len(predicted.head_offset) != len(oracle.head_offset):
        raise ValueError("predicted and oracle heads must align")
    if len(predicted.deprel) != len(oracle.deprel):
        raise ValueError("predicted and oracle deprels must align")
    return {
        "predicted_heads_predicted_deprels": DependencyPrediction(
            tuple(predicted.head_offset), tuple(predicted.deprel)
        ),
        "oracle_heads_oracle_deprels": DependencyPrediction(
            tuple(oracle.head_offset), tuple(oracle.deprel)
        ),
        "oracle_heads_predicted_deprels": DependencyPrediction(
            tuple(oracle.head_offset), tuple(predicted.deprel)
        ),
        "predicted_heads_oracle_deprels": DependencyPrediction(
            tuple(predicted.head_offset), tuple(oracle.deprel)
        ),
    }


def _factorization_read(accuracies: Mapping[str, float]) -> dict[str, Any]:
    both_predicted = accuracies["predicted_heads_predicted_deprels"]
    oracle_ceiling = accuracies["oracle_heads_oracle_deprels"]
    oracle_heads = accuracies["oracle_heads_predicted_deprels"]
    oracle_deprels = accuracies["predicted_heads_oracle_deprels"]
    head_distance_to_predicted = abs(oracle_heads - both_predicted)
    head_distance_to_ceiling = abs(oracle_heads - oracle_ceiling)
    deprel_distance_to_predicted = abs(oracle_deprels - both_predicted)
    deprel_distance_to_ceiling = abs(oracle_deprels - oracle_ceiling)
    head_lift = oracle_heads - both_predicted
    deprel_lift = oracle_deprels - both_predicted
    positive_lift = max(0.0, head_lift) + max(0.0, deprel_lift)
    if positive_lift:
        head_share = max(0.0, head_lift) / positive_lift
        deprel_share = max(0.0, deprel_lift) / positive_lift
    else:
        head_share = deprel_share = 0.5

    if (
        deprel_distance_to_ceiling <= deprel_distance_to_predicted
        and head_distance_to_predicted < head_distance_to_ceiling
    ):
        verdict = "DEPREL-LABEL-bound"
    elif (
        head_distance_to_ceiling <= head_distance_to_predicted
        and deprel_distance_to_predicted < deprel_distance_to_ceiling
    ):
        verdict = "HEAD-bound"
    else:
        verdict = "both factors contribute"
    text = (
        f"{verdict}: oracle-head swap lift={head_lift:+.4f}, "
        f"oracle-deprel swap lift={deprel_lift:+.4f}; positive-lift split "
        f"head={head_share:.1%}, deprel={deprel_share:.1%}."
    )
    return {
        "verdict": verdict,
        "text": text,
        "oracle_head_swap_lift": head_lift,
        "oracle_deprel_swap_lift": deprel_lift,
        "positive_lift_split": {"head": head_share, "deprel": deprel_share},
    }


def evaluate_half_oracle_factorization(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    by_sentence: Mapping[str, Mapping[str, DependencyPrediction]],
    metrics: Mapping[str, Mapping[str, float | int]],
    case_student: GermanR1Student,
    case_runner: CaseRunner = full_oracle_case_sentence,
) -> dict[str, Any]:
    """Measure head/deprel swaps without performing any additional data read."""
    l2_las = float(metrics["L2"]["las"])
    l3_las = float(metrics["L3"]["las"])
    best_rung = max(("L2", "L3"), key=lambda rung: float(metrics[rung]["las"]))
    other_rung = "L3" if best_rung == "L2" else "L2"
    comparison = "=" if l2_las == l3_las else ">"
    selection_reason = (
        f"{best_rung} (LAS {float(metrics[best_rung]['las']):.4f} {comparison} "
        f"{other_rung} {float(metrics[other_rung]['las']):.4f})"
    )
    condition_predictions: dict[str, list[str]] = {
        condition: []
        for condition in (
            "predicted_heads_predicted_deprels",
            "oracle_heads_oracle_deprels",
            "oracle_heads_predicted_deprels",
            "predicted_heads_oracle_deprels",
        )
    }
    gold_case: list[str] = []
    for sentence in test:
        oracle = aligned_head_record(sentence, heads)
        if not isinstance(oracle, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        eligible = [
            index
            for index, label in enumerate(sentence.targets["morph_case"])
            if label != "-"
        ]
        pairs = half_oracle_dependency_pairs(
            by_sentence[sentence.sent_id][best_rung], oracle
        )
        for condition, dependency in pairs.items():
            gold_pos = (
                sentence.targets["pos"]
                if condition == "oracle_heads_oracle_deprels"
                else pos
            )
            result = case_runner(
                case_student,
                sentence.tokens,
                baseline,
                pos,
                gold_pos,
                dependency.head_offset,
                dependency.deprel,
                eligible,
            )
            condition_predictions[condition].extend(result.predictions)
        gold_case.extend(sentence.targets["morph_case"])

    accuracies = {
        condition: accuracy(predictions, gold_case)
        for condition, predictions in condition_predictions.items()
    }
    expected = float(metrics[best_rung]["serve_honest_morph_case"])
    observed = accuracies["predicted_heads_predicted_deprels"]
    consistency_matches = math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
    if not consistency_matches:
        raise AssertionError(
            "both-predicted half-oracle arm does not reproduce the best-rung case score"
        )
    predicted_pos_note = "predicted POS in pos_full and gold_pos"
    conditions = {
        "predicted_heads_predicted_deprels": {
            "accuracy": accuracies["predicted_heads_predicted_deprels"],
            "pos_treatment": predicted_pos_note,
        },
        "oracle_heads_oracle_deprels": {
            "accuracy": accuracies["oracle_heads_oracle_deprels"],
            "pos_treatment": "predicted POS in pos_full; true gold POS in gold_pos (#116 ceiling)",
        },
        "oracle_heads_predicted_deprels": {
            "accuracy": accuracies["oracle_heads_predicted_deprels"],
            "pos_treatment": predicted_pos_note,
        },
        "predicted_heads_oracle_deprels": {
            "accuracy": accuracies["predicted_heads_oracle_deprels"],
            "pos_treatment": predicted_pos_note,
        },
    }
    return {
        "best_rung": best_rung,
        "best_rung_las": float(metrics[best_rung]["las"]),
        "selection_reason": selection_reason,
        "conditions": conditions,
        "internal_consistency": {
            "matches": consistency_matches,
            "observed_both_predicted": observed,
            "expected_serve_honest_morph_case": expected,
            "absolute_difference": abs(observed - expected),
        },
        "read": _factorization_read(accuracies),
    }


def _read_verdict(metrics: Mapping[str, Mapping[str, float | int]]) -> dict[str, Any]:
    l0 = metrics["L0"]
    l1 = metrics["L1"]
    l2 = metrics["L2"]
    l3 = metrics["L3"]
    bilinear_lift = float(l2["las"]) - float(l0["las"])
    total_lift = float(l3["las"]) - float(l0["las"])
    justified = (
        float(l3["serve_honest_morph_case"]) >= 0.87
        and total_lift > 0.0
        and bilinear_lift > total_lift / 2.0
    )
    if justified:
        return {
            "verdict": "JUSTIFIED",
            "bottleneck": None,
            "text": (
                "JUSTIFIED: L3 clears the ~0.87 serve-honest case gate and the bilinear-only "
                "LAS lift is a majority of the total L0->L3 lift."
            ),
        }

    l3_uas = float(l3["uas"])
    l3_relation = float(l3["deprel_only_accuracy"])
    tree_uas_lift = (float(l1["uas"]) - float(l0["uas"])) + (
        float(l3["uas"]) - float(l2["uas"])
    )
    head_uas_lift = float(l2["uas"]) - float(l0["uas"])
    if l3_relation + 0.05 < l3_uas:
        bottleneck = "deprel-labels"
    elif tree_uas_lift > max(0.01, head_uas_lift):
        bottleneck = "tree"
    else:
        bottleneck = "head-model"
    return {
        "verdict": "plateau",
        "bottleneck": bottleneck,
        "text": (
            f"plateau: the primitive gate is not met; bottleneck={bottleneck}. "
            f"At L3 UAS={l3_uas:.3f} versus deprel-only={l3_relation:.3f}; "
            f"tree UAS lift={tree_uas_lift:+.3f}, bilinear UAS lift={head_uas_lift:+.3f}."
        ),
    }


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit on train, use dev only for L0's shipped window, then read test once."""
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
    bilinear = BilinearAttachmentScorer()
    print(
        f"FIT: bilinear rank={RANK} epochs={EPOCHS} lr={LEARNING_RATE} seed={SEED}.",
        flush=True,
    )
    bilinear.fit(train, train_heads, train_pos)
    print(
        "FIT: bilinear epoch losses="
        + ", ".join(f"{loss:.6f}" for loss in bilinear.epoch_losses),
        flush=True,
    )

    guard = CampaignTestReadGuard()
    print("FINAL EVAL: reading shared test tasks once and head_deprel test once.", flush=True)
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_head_records = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads = head_deprel_by_sent(test_head_records)
    test_pos = _predicted_pos_by_sentence(r1_student, test)
    metrics, by_sentence = evaluate_ladder(
        test,
        test_heads,
        test_pos,
        l0_student,
        bilinear,
        r1_student,
    )
    add_case_bearing_deprel_metrics(metrics, test, test_heads, by_sentence)
    half_oracle_factorization = evaluate_half_oracle_factorization(
        test,
        test_heads,
        test_pos,
        by_sentence,
        metrics,
        r1_student,
    )
    guard.assert_complete()

    lifts = {
        "L1_minus_L0_tree_only": float(metrics["L1"]["las"]) - float(metrics["L0"]["las"]),
        "L2_minus_L0_bilinear_only": float(metrics["L2"]["las"]) - float(metrics["L0"]["las"]),
        "L3_minus_L2_tree_on_bilinear": float(metrics["L3"]["las"]) - float(metrics["L2"]["las"]),
        "L3_minus_L0_total": float(metrics["L3"]["las"]) - float(metrics["L0"]["las"]),
    }
    scoreboard: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "scope": SCOPE,
        "measurement": "attachment lever-probe; not the certified build",
        "serve_honest_features": [
            "R1-predicted POS",
            "hashed lowercase surface form",
            "surface shape",
            "normalized sentence position",
        ],
        "gold_usage": (
            "GSD train dependency targets, final scoring, and explicit half-oracle swaps only; "
            "true gold POS appears only in the oracle-heads + oracle-deprels #116 sanity arm"
        ),
        "data_hash_sha256": hasher.hexdigest(),
        "test_read_counts": dict(guard.counts),
        "test_read_once": True,
        "l0_window": {
            "k": window,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "bilinear": {
            "rank": bilinear.config.rank,
            "epochs": bilinear.config.epochs,
            "learning_rate": bilinear.config.learning_rate,
            "l2_penalty": bilinear.config.l2_penalty,
            "seed": bilinear.config.seed,
            "feature_dim": bilinear.feature_dim,
            "epoch_losses": bilinear.epoch_losses,
        },
        "tree_decode": "single-root non-projective Chu-Liu-Edmonds",
        "case_cascade": full_oracle_case_sentence.__name__,
        "rungs": metrics,
        "half_oracle_factorization": half_oracle_factorization,
        "las_lifts": lifts,
        "read": _read_verdict(metrics),
    }
    return scoreboard, guard


def print_scoreboard(scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard) -> None:
    """Print the complete probe ladder and preregistered read."""
    print("\nATTACHMENT LEVER-PROBE — GSD TEST")
    print(f"run_tag: {scoreboard['run_tag']}")
    print(f"scope: {scoreboard['scope']} (NOT the certified build)")
    print("serve_honesty: R1-predicted POS + surface/shape/position only; no gold model features")
    config = scoreboard["bilinear"]
    print(
        f"bilinear: rank={config['rank']} epochs={config['epochs']} "
        f"lr={config['learning_rate']} seed={config['seed']} feature_dim={config['feature_dim']}"
    )
    guard.assert_complete()
    print(
        "TEST READ ONCE: CONFIRMED "
        "(shared pos/morph_case/morph_gnn test load once; head_deprel test load once; "
        "all four rungs reuse the same in-memory sentences)"
    )
    print(
        "\nrung  UAS       LAS       deprel-only  case-bearing-deprel (n)  "
        "serve-honest-case  tokens"
    )
    for rung in RUNG_NAMES:
        metric = scoreboard["rungs"][rung]
        print(
            f"{rung:<4}  {metric['uas']:.6f}  {metric['las']:.6f}  "
            f"{metric['deprel_only_accuracy']:.6f}     "
            f"{metric['case_bearing_deprel_accuracy']:.6f} "
            f"({metric['case_bearing_deprel_n']})          "
            f"{metric['serve_honest_morph_case']:.6f}           {metric['tokens']}"
        )
    factorization = scoreboard["half_oracle_factorization"]
    print("\nHALF-ORACLE FACTORIZATION")
    print(f"best predicted parse: {factorization['selection_reason']}")
    condition_labels = {
        "predicted_heads_predicted_deprels": "predicted-heads + predicted-deprels",
        "oracle_heads_oracle_deprels": "oracle-heads + oracle-deprels",
        "oracle_heads_predicted_deprels": "oracle-heads + predicted-deprels",
        "predicted_heads_oracle_deprels": "predicted-heads + oracle-deprels",
    }
    for key, label in condition_labels.items():
        condition = factorization["conditions"][key]
        print(
            f"{label:<38} {condition['accuracy']:.6f}  "
            f"[{condition['pos_treatment']}]"
        )
    consistency = factorization["internal_consistency"]
    print(
        "internal consistency: PASS; both-predicted="
        f"{consistency['observed_both_predicted']:.6f}, best-rung serve-honest="
        f"{consistency['expected_serve_honest_morph_case']:.6f}, "
        f"abs_diff={consistency['absolute_difference']:.3g}"
    )
    print(f"read: {factorization['read']['text']}")
    lifts = scoreboard["las_lifts"]
    print("\nPER-LEVER LAS LIFT")
    print(f"L1-L0 tree-only:          {lifts['L1_minus_L0_tree_only']:+.6f}")
    print(f"L2-L0 bilinear-only:      {lifts['L2_minus_L0_bilinear_only']:+.6f}")
    print(f"L3-L2 tree-on-bilinear:   {lifts['L3_minus_L2_tree_on_bilinear']:+.6f}")
    print(f"L3-L0 total:              {lifts['L3_minus_L0_total']:+.6f}")
    print("\nTHE READ")
    print(scoreboard["read"]["text"])


def main() -> None:
    scoreboard, guard = build_scoreboard()
    OUTPUT_PATH.write_text(json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_scoreboard(scoreboard, guard)
    print(f"structured_output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
