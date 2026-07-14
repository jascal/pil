"""Axis-B biaffine relation-labeler campaign (grok lane).

Gradient-fit low-rank biaffine deprel scorer evaluated serve-honest against the
shipped unary count-table labeler on the SAME predicted heads, wired through the
existing case cascade.  Pre-registered decision rules from
``docs/notes/biaffine_labeler_prereg.md`` §5 are applied verbatim.

Independent of the parallel codex lane; do not import codex biaffine modules.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
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

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "campaign_biaffine_labeler.json"
DIAGNOSTIC_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "campaign_biaffine_labeler_diagnostic.json"
)
RUN_TAG = "empirical"
SCOPE = "axis-B biaffine relation-labeler"

# Fixed a-priori hyperparameters (no test tuning).
RANK = 16
EPOCHS = 5
LEARNING_RATE = 0.04
L2_PENALTY = 1e-5
SEEDS = (0, 1, 2)
PRIMARY_SEED = 0
GRAD_CLIP = 5.0
CASE_BASELINE = 0.76
CASE_DELIVERABLE = 0.90
LAS_FIRES_THRESHOLD = 0.03

PAIRWISE_SENSITIVE = frozenset({"nsubj", "obj", "iobj", "obl", "nmod", "conj"})
LOCAL_DEPRELS = frozenset({"det", "amod", "case", "punct", "aux", "cop"})
GENERIC_FALLBACK = "dep"


@dataclass(frozen=True)
class BilinearLabelConfig:
    """Fixed-a-priori biaffine labeler configuration; no test tuning."""

    rank: int = RANK
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    l2_penalty: float = L2_PENALTY
    seed: int = PRIMARY_SEED
    surface_buckets: int = 32
    position_buckets: int = 5


def _coarsen(deprel: str) -> str:
    """Strip a UD subtype after the first colon (``nsubj:pass`` → ``nsubj``)."""
    return deprel.split(":", 1)[0]


def relation_vocab_from_heads(heads: Mapping[str, HeadDeprelRecord]) -> tuple[str, ...]:
    """Sorted deprel vocabulary observed in gold train head records."""
    labels: set[str] = set()
    for record in heads.values():
        labels.update(record.deprel)
    if not labels:
        raise ValueError("cannot build relation vocabulary from empty heads")
    return tuple(sorted(labels))


class BilinearRelationLabeler:
    """Low-rank Dozat-style biaffine deprel scorer over (dependent, governor).

    Parameterization (independent design choice)::

        h_i = U^T φ(i),   h_g = V^T φ(g)
        logits_r = h_i^T C_r h_g + φ(i)^T p_r + φ(g)^T q_r + b_r

    Shared projections ``U, V ∈ R^{d×rank}`` keep the bilinear map low-rank;
    each relation has its own ``rank×rank`` core ``C_r`` and linear terms.
    Features are the shipped serve-honest ``BilinearAttachmentScorer.phi``.

    At predict time the argmax is always over the fitted train vocabulary ``R``.
    Labels outside ``R`` are never emitted; if ``dep`` is in ``R`` it is the
    most-generic UD sink available in the vocabulary, but prediction is pure
    argmax (no special-case remapping after the softmax).
    """

    def __init__(
        self,
        config: BilinearLabelConfig | None = None,
        relation_vocab: Sequence[str] | None = None,
    ) -> None:
        self.config = config or BilinearLabelConfig()
        if self.config.rank < 1:
            raise ValueError("rank must be positive")
        self._feature_scorer = BilinearAttachmentScorer(
            BilinearConfig(
                rank=self.config.rank,
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                l2_penalty=self.config.l2_penalty,
                seed=self.config.seed,
                surface_buckets=self.config.surface_buckets,
                position_buckets=self.config.position_buckets,
            )
        )
        self.feature_dim = self._feature_scorer.feature_dim
        self.relation_vocab: tuple[str, ...] = tuple(relation_vocab) if relation_vocab else ()
        self._rel_index: dict[str, int] = {
            label: index for index, label in enumerate(self.relation_vocab)
        }
        self._rng = np.random.default_rng(self.config.seed)
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.C: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.epoch_losses: list[float] = []
        if self.relation_vocab:
            self._init_parameters(len(self.relation_vocab))

    def _init_parameters(self, relation_count: int) -> None:
        rank = self.config.rank
        dim = self.feature_dim
        scale = 1.0 / math.sqrt(dim)
        self.U = self._rng.normal(0.0, scale, (dim, rank))
        self.V = self._rng.normal(0.0, scale, (dim, rank))
        # Near-identity cores so early logits are roughly bilinear-dot + linear.
        self.C = np.zeros((relation_count, rank, rank), dtype=np.float64)
        for relation in range(relation_count):
            self.C[relation] = np.eye(rank) * scale
            self.C[relation] += self._rng.normal(0.0, scale * 0.1, (rank, rank))
        self.P = self._rng.normal(0.0, scale, (relation_count, dim))
        self.Q = self._rng.normal(0.0, scale, (relation_count, dim))
        self.b = np.zeros(relation_count, dtype=np.float64)

    def set_relation_vocab(self, vocab: Sequence[str]) -> None:
        """Install vocabulary and (re)initialize parameters for that label set."""
        labels = tuple(vocab)
        if not labels:
            raise ValueError("relation vocabulary must be non-empty")
        self.relation_vocab = labels
        self._rel_index = {label: index for index, label in enumerate(labels)}
        self._rng = np.random.default_rng(self.config.seed)
        self._init_parameters(len(labels))

    def phi(
        self,
        token: str,
        predicted_pos: str,
        index: int,
        sentence_size: int,
        *,
        root: bool = False,
    ) -> np.ndarray:
        """Delegate to the shipped serve-honest attachment ``phi`` (no reimplementation)."""
        return self._feature_scorer.phi(
            token, predicted_pos, index, sentence_size, root=root
        )

    def _dependent_and_governor_features(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (len(tokens) == len(predicted_pos) == len(head_offsets)):
            raise ValueError("tokens, predicted POS, and head offsets must align")
        if not tokens:
            raise ValueError("cannot score an empty sentence")
        size = len(tokens)
        dependent = np.stack(
            [
                self.phi(token, pos, index, size)
                for index, (token, pos) in enumerate(zip(tokens, predicted_pos, strict=True))
            ]
        )
        root_feature = self.phi("<ROOT>", "<UNK>", 0, size, root=True)
        governors = np.empty_like(dependent)
        for index, offset in enumerate(head_offsets):
            if int(offset) == 0:
                governors[index] = root_feature
                continue
            governor = resolve_governor_index(index, int(offset), size)
            if governor is None or governor == index:
                governors[index] = root_feature
            else:
                governors[index] = dependent[governor]
        return dependent, governors

    def logits(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> np.ndarray:
        """Return per-token logits over ``R`` for the supplied head offsets."""
        if self.U is None or self.V is None or self.C is None:
            raise RuntimeError("labeler used before fit / vocab init")
        if not self.relation_vocab:
            raise RuntimeError("relation vocabulary is empty")
        dependent, governors = self._dependent_and_governor_features(
            tokens, predicted_pos, head_offsets
        )
        return self._logits_from_features(dependent, governors)

    def _logits_from_features(
        self,
        dependent: np.ndarray,
        governors: np.ndarray,
    ) -> np.ndarray:
        assert self.U is not None and self.V is not None and self.C is not None
        assert self.P is not None and self.Q is not None and self.b is not None
        dep_h = dependent @ self.U
        gov_h = governors @ self.V
        bilinear = np.einsum("nj,rjk,nk->nr", dep_h, self.C, gov_h)
        linear = dependent @ self.P.T + governors @ self.Q.T + self.b
        return bilinear + linear

    def predict(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[str, ...]:
        """Argmax deprels over train ``R``; never consults gold heads or deprels.

        Signature is intentionally (tokens, predicted_pos, head_offsets) only —
        the supplied heads may be predicted or (diagnostic) gold, but gold
        deprels are never a parameter.
        """
        if not tokens:
            return ()
        if not self.relation_vocab:
            # Documented last-resort: empty vocab should not occur after fit.
            return tuple(GENERIC_FALLBACK for _ in tokens)
        scores = self.logits(tokens, predicted_pos, head_offsets)
        indices = np.argmax(scores, axis=1)
        return tuple(self.relation_vocab[int(index)] for index in indices)

    def _train_sentence(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
        gold_deprels: Sequence[str],
    ) -> float:
        assert self.U is not None and self.V is not None and self.C is not None
        assert self.P is not None and self.Q is not None and self.b is not None
        if not (len(tokens) == len(gold_deprels) == len(head_offsets)):
            raise ValueError("training sentence fields must align")
        targets: list[int] = []
        keep: list[int] = []
        for index, label in enumerate(gold_deprels):
            relation_index = self._rel_index.get(label)
            if relation_index is None:
                # Gold label outside R cannot contribute CE; skip (should not
                # happen when R is built from the same train gold).
                continue
            keep.append(index)
            targets.append(relation_index)
        if not targets:
            return 0.0

        dependent, governors = self._dependent_and_governor_features(
            tokens, predicted_pos, head_offsets
        )
        dependent = dependent[keep]
        governors = governors[keep]
        target_array = np.asarray(targets, dtype=np.int64)
        token_count = len(targets)

        dep_h = dependent @ self.U
        gov_h = governors @ self.V
        logits = np.einsum("nj,rjk,nk->nr", dep_h, self.C, gov_h)
        logits = logits + dependent @ self.P.T + governors @ self.Q.T + self.b

        maxima = np.max(logits, axis=1, keepdims=True)
        shifted = logits - maxima
        exp = np.exp(shifted)
        normalizer = exp.sum(axis=1, keepdims=True)
        probabilities = exp / normalizer
        log_probs = shifted - np.log(normalizer)
        loss = float(-np.mean(log_probs[np.arange(token_count), target_array]))

        grad_logits = probabilities
        grad_logits[np.arange(token_count), target_array] -= 1.0
        grad_logits /= token_count

        grad_c = np.einsum("nr,nj,nk->rjk", grad_logits, dep_h, gov_h)
        grad_dep_h = np.einsum("nr,rjk,nk->nj", grad_logits, self.C, gov_h)
        grad_gov_h = np.einsum("nr,rjk,nj->nk", grad_logits, self.C, dep_h)
        grad_u = dependent.T @ grad_dep_h + self.config.l2_penalty * self.U
        grad_v = governors.T @ grad_gov_h + self.config.l2_penalty * self.V
        grad_p = grad_logits.T @ dependent + self.config.l2_penalty * self.P
        grad_q = grad_logits.T @ governors + self.config.l2_penalty * self.Q
        grad_b = grad_logits.sum(axis=0)
        grad_c = grad_c + self.config.l2_penalty * self.C

        grad_norm = float(
            np.sqrt(
                np.square(grad_u).sum()
                + np.square(grad_v).sum()
                + np.square(grad_c).sum()
                + np.square(grad_p).sum()
                + np.square(grad_q).sum()
                + np.square(grad_b).sum()
            )
        )
        if grad_norm > GRAD_CLIP:
            scale = GRAD_CLIP / grad_norm
            grad_u *= scale
            grad_v *= scale
            grad_c *= scale
            grad_p *= scale
            grad_q *= scale
            grad_b *= scale

        lr = self.config.learning_rate
        self.U -= lr * grad_u
        self.V -= lr * grad_v
        self.C -= lr * grad_c
        self.P -= lr * grad_p
        self.Q -= lr * grad_q
        self.b -= lr * grad_b
        return loss

    def fit(
        self,
        train_sentences: Sequence[Sentence],
        train_heads: Mapping[str, HeadDeprelRecord],
        train_predicted_pos: Mapping[str, Sequence[str]],
        *,
        governor_head_offsets: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        """Cross-entropy on gold deprels; governors from gold (or supplied) heads.

        ``governor_head_offsets`` defaults to gold ``head_offset`` per sentence.
        The noise-matched diagnostic arm passes L0-predicted train heads here.
        """
        if not train_sentences:
            raise ValueError("cannot fit on empty training data")
        if not self.relation_vocab:
            self.set_relation_vocab(relation_vocab_from_heads(train_heads))
        elif self.U is None:
            self._init_parameters(len(self.relation_vocab))

        self.epoch_losses = []
        for _epoch in range(self.config.epochs):
            order = self._rng.permutation(len(train_sentences))
            losses: list[float] = []
            for sentence_index in order:
                sentence = train_sentences[int(sentence_index)]
                head = aligned_head_record(sentence, train_heads)
                if not isinstance(head, HeadDeprelRecord):
                    raise TypeError("labeled dependency record required")
                if governor_head_offsets is None:
                    offsets = head.head_offset
                else:
                    offsets = governor_head_offsets[sentence.sent_id]
                losses.append(
                    self._train_sentence(
                        sentence.tokens,
                        train_predicted_pos[sentence.sent_id],
                        offsets,
                        head.deprel,
                    )
                )
            self.epoch_losses.append(float(np.mean(losses)))


def partition_deprel_accuracy(
    predicted_deprel: Sequence[str],
    gold_deprel: Sequence[str],
) -> dict[str, float | int]:
    """Deprel-only accuracy on pairwise-sensitive vs local gold coarse partitions."""
    if len(predicted_deprel) != len(gold_deprel):
        raise ValueError("predicted and gold deprels must align")
    pairwise_hits = pairwise_n = 0
    local_hits = local_n = 0
    for predicted, gold in zip(predicted_deprel, gold_deprel, strict=True):
        # Set membership uses coarse UD relation; accuracy itself is exact deprel-only.
        coarse = _coarsen(gold)
        if coarse in PAIRWISE_SENSITIVE:
            pairwise_n += 1
            pairwise_hits += int(predicted == gold)
        elif coarse in LOCAL_DEPRELS:
            local_n += 1
            local_hits += int(predicted == gold)
    return {
        "pairwise_sensitive_accuracy": (
            pairwise_hits / pairwise_n if pairwise_n else float("nan")
        ),
        "pairwise_sensitive_n": pairwise_n,
        "local_accuracy": local_hits / local_n if local_n else float("nan"),
        "local_n": local_n,
    }


def coarse_las(
    predicted_head_offset: Sequence[int],
    predicted_deprel: Sequence[str],
    gold_head_offset: Sequence[int],
    gold_deprel: Sequence[str],
) -> float:
    """LAS with deprels compared after stripping UD subtypes."""
    size = len(gold_head_offset)
    if not size:
        raise ValueError("cannot score empty sequences")
    if not (
        len(predicted_head_offset)
        == len(predicted_deprel)
        == size
        == len(gold_deprel)
    ):
        raise ValueError("prediction/gold lengths differ")
    correct = 0
    for index in range(size):
        if predicted_head_offset[index] != gold_head_offset[index]:
            continue
        if _coarsen(predicted_deprel[index]) == _coarsen(gold_deprel[index]):
            correct += 1
    return correct / size


def score_arm_sequences(
    predicted_head_offset: Sequence[int],
    predicted_deprel: Sequence[str],
    gold_head_offset: Sequence[int],
    gold_deprel: Sequence[str],
    predicted_case: Sequence[str],
    gold_case: Sequence[str],
) -> dict[str, float | int]:
    """Full metric bundle for one labeler arm on concatenated sequences."""
    base = score_prediction_sequences(
        predicted_head_offset,
        predicted_deprel,
        gold_head_offset,
        gold_deprel,
    )
    case_bearing_accuracy, case_bearing_n = score_case_bearing_deprels(
        predicted_deprel, gold_deprel
    )
    partitions = partition_deprel_accuracy(predicted_deprel, gold_deprel)
    return {
        "uas": base["uas"],
        "las": base["las"],
        "las_coarse": coarse_las(
            predicted_head_offset,
            predicted_deprel,
            gold_head_offset,
            gold_deprel,
        ),
        "deprel_only_accuracy": base["deprel_only_accuracy"],
        "case_bearing_deprel_accuracy": case_bearing_accuracy,
        "case_bearing_deprel_n": case_bearing_n,
        "serve_honest_morph_case": accuracy(predicted_case, gold_case),
        "tokens": base["tokens"],
        **partitions,
    }


def evaluate_labeler_arm(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    common_heads_by_sent: Mapping[str, Sequence[int]],
    deprel_fn,
    case_student: GermanR1Student,
    *,
    serve_honest: bool = True,
) -> dict[str, float | int]:
    """Run one labeler arm on supplied heads through the case cascade."""
    pred_head: list[int] = []
    pred_deprel: list[str] = []
    pred_case: list[str] = []
    gold_head: list[int] = []
    gold_deprel: list[str] = []
    gold_case: list[str] = []

    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        offsets = tuple(common_heads_by_sent[sentence.sent_id])
        deprels = tuple(deprel_fn(sentence.tokens, pos, offsets))
        case_baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        eligible = [
            index
            for index, label in enumerate(sentence.targets["morph_case"])
            if label != "-"
        ]
        case_result = run_predicted_case_sentence(
            case_student,
            sentence.tokens,
            case_baseline,
            pos,
            offsets,
            deprels,
            eligible,
        )
        pred_head.extend(offsets)
        pred_deprel.extend(deprels)
        pred_case.extend(case_result.predictions)
        gold_head.extend(head.head_offset)
        gold_deprel.extend(head.deprel)
        gold_case.extend(sentence.targets["morph_case"])

    metrics = score_arm_sequences(
        pred_head, pred_deprel, gold_head, gold_deprel, pred_case, gold_case
    )
    metrics["serve_honest"] = serve_honest
    return metrics


def deprel_only_on_heads(
    sentences: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    head_offsets_by_sent: Mapping[str, Sequence[int]],
    deprel_fn,
) -> float:
    """Deprel-only accuracy with an explicit governor-head source."""
    predicted: list[str] = []
    gold: list[str] = []
    for sentence in sentences:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        offsets = tuple(head_offsets_by_sent[sentence.sent_id])
        predicted.extend(deprel_fn(sentence.tokens, pos, offsets))
        gold.extend(head.deprel)
    return accuracy(predicted, gold)


def apply_preregistered_read(
    unary_las: float,
    unary_case: float,
    per_seed: Sequence[Mapping[str, Any]],
    partition_gains: Mapping[str, float],
) -> dict[str, Any]:
    """Apply ``biaffine_labeler_prereg.md`` §5 decision rules verbatim.

    FIRES: mean biaffine−unary LAS (strict) gain ≥ +0.03 absolute AND positive at
    every seed, AND deprel-only gain on the pairwise-sensitive partition is
    STRICTLY GREATER than on the local partition, AND case-cascade improves.

    IN-BETWEEN: gain > 0 but < +0.03, OR not positive at every seed, OR no
    case-cascade improvement, OR partition gain is diffuse.

    HALTED: mean LAS gain ≤ 0.
    """
    las_gains = [float(row["las_gain"]) for row in per_seed]
    case_values = [float(row["case"]) for row in per_seed]
    las_gain_mean = float(np.mean(las_gains))
    case_biaffine_mean = float(np.mean(case_values))
    positive_every_seed = all(gain > 0.0 for gain in las_gains)
    pairwise_gain = float(partition_gains["pairwise_sensitive"])
    local_gain = float(partition_gains["local"])
    partition_concentrated = pairwise_gain > local_gain
    case_improves = case_biaffine_mean > unary_case

    if las_gain_mean <= 0.0:
        verdict = "HALTED"
        text = (
            "HALTED: mean biaffine−unary LAS (strict) gain "
            f"({las_gain_mean:+.4f}) ≤ 0 — the bilinear labeler does not realize "
            "a labeling advantage over the count-table."
        )
    elif (
        las_gain_mean >= LAS_FIRES_THRESHOLD
        and positive_every_seed
        and partition_concentrated
        and case_improves
    ):
        verdict = "FIRES"
        text = (
            "FIRES: mean biaffine−unary LAS (strict) gain "
            f"{las_gain_mean:+.4f} ≥ +{LAS_FIRES_THRESHOLD:.2f} absolute and positive "
            f"at every seed; pairwise-sensitive deprel-only gain ({pairwise_gain:+.4f}) "
            f"> local gain ({local_gain:+.4f}); case-cascade improves "
            f"({case_biaffine_mean:.4f} > unary {unary_case:.4f})."
        )
    else:
        verdict = "IN-BETWEEN"
        reasons: list[str] = []
        if las_gain_mean > 0.0 and las_gain_mean < LAS_FIRES_THRESHOLD:
            reasons.append(
                f"gain {las_gain_mean:+.4f} > 0 but < +{LAS_FIRES_THRESHOLD:.2f}"
            )
        if not positive_every_seed:
            reasons.append("LAS gain not positive at every seed")
        if not case_improves:
            reasons.append(
                "no case-cascade improvement "
                f"(biaffine mean case {case_biaffine_mean:.4f} ≤ unary {unary_case:.4f})"
            )
        if not partition_concentrated:
            reasons.append(
                "partition gain diffuse "
                f"(pairwise-sensitive {pairwise_gain:+.4f} ≤ local {local_gain:+.4f})"
            )
        if las_gain_mean >= LAS_FIRES_THRESHOLD and positive_every_seed and not reasons:
            reasons.append("FIRES sub-conditions incomplete")
        text = "IN-BETWEEN: " + "; ".join(reasons) + "."

    return {
        "verdict": verdict,
        "las_gain_mean": las_gain_mean,
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "las": float(row["las"]),
                "las_gain": float(row["las_gain"]),
                "deprel_only_accuracy": float(row["deprel_only_accuracy"]),
                "deprel_only_gain": float(row["deprel_only_gain"]),
                "case": float(row["case"]),
                "case_gain": float(row["case_gain"]),
            }
            for row in per_seed
        ],
        "partition_gains": {
            "pairwise_sensitive": pairwise_gain,
            "local": local_gain,
        },
        "case_unary": float(unary_case),
        "case_biaffine_mean": case_biaffine_mean,
        "text": text,
        "unary_las": float(unary_las),
    }


def apply_through_line_case_read(case_accuracy: float) -> dict[str, Any]:
    """Through-line germanapp case read (plateau language; never irreducible)."""
    if case_accuracy >= CASE_DELIVERABLE:
        verdict = "deliverable_met"
        text = (
            f"THROUGH-LINE: serve-honest case {case_accuracy:.4f} ≥ {CASE_DELIVERABLE:.2f} "
            "— deliverable met (empirical; certify next)."
        )
    elif case_accuracy > CASE_BASELINE:
        gap_closed = case_accuracy - CASE_BASELINE
        gap_total = CASE_DELIVERABLE - CASE_BASELINE
        fraction = gap_closed / gap_total if gap_total > 0 else 0.0
        verdict = "plateau"
        text = (
            f"THROUGH-LINE: serve-honest case {case_accuracy:.4f} < {CASE_DELIVERABLE:.2f} "
            f"but > ~{CASE_BASELINE:.2f} baseline — the recipe closes "
            f"{fraction:.1%} of the ~{CASE_BASELINE:.2f}→{CASE_DELIVERABLE:.2f} gap; "
            "achievability of 0.90 open (plateau language)."
        )
    else:
        verdict = "diagnose"
        text = (
            f"THROUGH-LINE: serve-honest case {case_accuracy:.4f} ≤ ~{CASE_BASELINE:.2f} "
            "baseline — the labeler gain did not survive the cascade; diagnose "
            "(R3 cascade-corruption pattern)."
        )
    return {"verdict": verdict, "case": float(case_accuracy), "text": text}


def _predict_heads_by_sentence(
    student: GermanR3DependencyStudent,
    sentences: Sequence[Sentence],
    predicted_pos: Mapping[str, Sequence[str]],
) -> dict[str, tuple[int, ...]]:
    return {
        sentence.sent_id: student.predict_heads(
            sentence.tokens, predicted_pos[sentence.sent_id]
        )
        for sentence in sentences
    }


def _gold_heads_by_sentence(
    sentences: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for sentence in sentences:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        result[sentence.sent_id] = tuple(head.head_offset)
    return result


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit on train, read test once, multi-seed biaffine vs unary comparison."""
    hasher = hashlib.sha256()
    print("FIT: loading GSD train shared tasks and head_deprel once.", flush=True)
    train = load_split(DATA_ROOT, "train", hasher)
    train_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "train.jsonl", hasher
    )
    train_heads = head_deprel_by_sent(train_head_records)
    relation_vocab = relation_vocab_from_heads(train_heads)
    print(f"FIT: relation vocabulary |R|={len(relation_vocab)}", flush=True)

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
    print(f"FIT: L0 dependency student window={window}.", flush=True)

    train_predicted_heads = _predict_heads_by_sentence(l0_student, train, train_pos)

    guard = CampaignTestReadGuard()
    print(
        "FINAL EVAL: reading shared test tasks once and head_deprel test once.",
        flush=True,
    )
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_head_records = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads = head_deprel_by_sent(test_head_records)
    test_pos = _predicted_pos_by_sentence(r1_student, test)
    test_predicted_heads = _predict_heads_by_sentence(l0_student, test, test_pos)
    test_gold_heads = _gold_heads_by_sentence(test, test_heads)

    def unary_deprel(
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[str, ...]:
        return predict_deprels(l0_student, tokens, predicted_pos, head_offsets)

    print("EVAL: unary arm on predicted heads (seed-independent).", flush=True)
    unary_metrics = evaluate_labeler_arm(
        test,
        test_heads,
        test_pos,
        test_predicted_heads,
        unary_deprel,
        r1_student,
        serve_honest=True,
    )

    per_seed_rows: list[dict[str, Any]] = []
    seed_metrics: dict[str, dict[str, Any]] = {}
    primary_labeler: BilinearRelationLabeler | None = None

    for seed in SEEDS:
        config = BilinearLabelConfig(seed=seed)
        labeler = BilinearRelationLabeler(config, relation_vocab=relation_vocab)
        print(
            f"FIT: biaffine seed={seed} rank={config.rank} epochs={config.epochs} "
            f"lr={config.learning_rate} l2={config.l2_penalty}.",
            flush=True,
        )
        labeler.fit(train, train_heads, train_pos)
        print(
            "FIT: biaffine epoch losses="
            + ", ".join(f"{loss:.6f}" for loss in labeler.epoch_losses),
            flush=True,
        )

        def biaffine_deprel(
            tokens: Sequence[str],
            predicted_pos: Sequence[str],
            head_offsets: Sequence[int],
            _labeler: BilinearRelationLabeler = labeler,
        ) -> tuple[str, ...]:
            return _labeler.predict(tokens, predicted_pos, head_offsets)

        metrics = evaluate_labeler_arm(
            test,
            test_heads,
            test_pos,
            test_predicted_heads,
            biaffine_deprel,
            r1_student,
            serve_honest=True,
        )
        if not math.isclose(
            float(metrics["uas"]), float(unary_metrics["uas"]), rel_tol=0.0, abs_tol=1e-15
        ):
            raise AssertionError(
                f"UAS mismatch seed={seed}: biaffine {metrics['uas']} vs unary "
                f"{unary_metrics['uas']} (common heads must be bit-identical)"
            )

        row = {
            "seed": seed,
            "las": float(metrics["las"]),
            "las_gain": float(metrics["las"]) - float(unary_metrics["las"]),
            "deprel_only_accuracy": float(metrics["deprel_only_accuracy"]),
            "deprel_only_gain": (
                float(metrics["deprel_only_accuracy"])
                - float(unary_metrics["deprel_only_accuracy"])
            ),
            "case": float(metrics["serve_honest_morph_case"]),
            "case_gain": (
                float(metrics["serve_honest_morph_case"])
                - float(unary_metrics["serve_honest_morph_case"])
            ),
            "metrics": metrics,
            "epoch_losses": list(labeler.epoch_losses),
            "config": {
                "rank": config.rank,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "l2_penalty": config.l2_penalty,
                "seed": config.seed,
                "feature_dim": labeler.feature_dim,
            },
        }
        per_seed_rows.append(row)
        seed_metrics[str(seed)] = row
        if seed == PRIMARY_SEED:
            primary_labeler = labeler
        print(
            f"SEED {seed}: LAS={metrics['las']:.6f} "
            f"(gain {row['las_gain']:+.6f}) deprel-only="
            f"{metrics['deprel_only_accuracy']:.6f} case="
            f"{metrics['serve_honest_morph_case']:.6f}",
            flush=True,
        )

    if primary_labeler is None:
        raise RuntimeError("primary seed labeler was not fitted")

    # Partition gains: mean across seeds of (biaffine_part - unary_part).
    pairwise_gains = [
        float(row["metrics"]["pairwise_sensitive_accuracy"])
        - float(unary_metrics["pairwise_sensitive_accuracy"])
        for row in per_seed_rows
    ]
    local_gains = [
        float(row["metrics"]["local_accuracy"]) - float(unary_metrics["local_accuracy"])
        for row in per_seed_rows
    ]
    partition_gains = {
        "pairwise_sensitive": float(np.mean(pairwise_gains)),
        "local": float(np.mean(local_gains)),
    }

    primary_read = apply_preregistered_read(
        float(unary_metrics["las"]),
        float(unary_metrics["serve_honest_morph_case"]),
        per_seed_rows,
        partition_gains,
    )
    case_mean = float(primary_read["case_biaffine_mean"])
    through_line = apply_through_line_case_read(case_mean)

    # --- Diagnostic (additive; does not alter primary read) ---
    print("DIAGNOSTIC: 2x2 form×heads + noise-matched arm + train/test gap.", flush=True)

    def primary_biaffine_deprel(
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[str, ...]:
        return primary_labeler.predict(tokens, predicted_pos, head_offsets)

    unary_gold_heads = evaluate_labeler_arm(
        test,
        test_heads,
        test_pos,
        test_gold_heads,
        unary_deprel,
        r1_student,
        serve_honest=False,
    )
    biaffine_gold_heads = evaluate_labeler_arm(
        test,
        test_heads,
        test_pos,
        test_gold_heads,
        primary_biaffine_deprel,
        r1_student,
        serve_honest=False,
    )
    primary_metrics = per_seed_rows[0]["metrics"]
    # primary seed is SEEDS[0] == 0; find explicitly
    for row in per_seed_rows:
        if row["seed"] == PRIMARY_SEED:
            primary_metrics = row["metrics"]
            break

    grid_2x2 = {
        "unary_x_predicted_heads": {
            **{k: v for k, v in unary_metrics.items()},
            "serve_honest": True,
            "label": "UNARY × predicted heads (primary arm)",
        },
        "biaffine_x_predicted_heads": {
            **{k: v for k, v in primary_metrics.items()},
            "serve_honest": True,
            "label": "BIAFFINE × predicted heads (primary arm, seed "
            f"{PRIMARY_SEED})",
        },
        "unary_x_gold_heads": {
            **{k: v for k, v in unary_gold_heads.items()},
            "serve_honest": False,
            "label": "UNARY × GOLD heads (diagnostic upper bound)",
        },
        "biaffine_x_gold_heads": {
            **{k: v for k, v in biaffine_gold_heads.items()},
            "serve_honest": False,
            "label": "BIAFFINE × GOLD heads (diagnostic upper bound)",
        },
    }

    noise_config = BilinearLabelConfig(seed=PRIMARY_SEED)
    noise_labeler = BilinearRelationLabeler(noise_config, relation_vocab=relation_vocab)
    print(
        "FIT: noise-matched biaffine (train governors = L0 predicted heads).",
        flush=True,
    )
    noise_labeler.fit(
        train,
        train_heads,
        train_pos,
        governor_head_offsets=train_predicted_heads,
    )
    print(
        "FIT: noise-matched epoch losses="
        + ", ".join(f"{loss:.6f}" for loss in noise_labeler.epoch_losses),
        flush=True,
    )

    def noise_deprel(
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[str, ...]:
        return noise_labeler.predict(tokens, predicted_pos, head_offsets)

    noise_metrics = evaluate_labeler_arm(
        test,
        test_heads,
        test_pos,
        test_predicted_heads,
        noise_deprel,
        r1_student,
        serve_honest=True,
    )

    train_gold_heads = _gold_heads_by_sentence(train, train_heads)
    train_deprel_only = deprel_only_on_heads(
        train,
        train_heads,
        train_pos,
        train_gold_heads,
        primary_biaffine_deprel,
    )
    test_deprel_only = float(primary_metrics["deprel_only_accuracy"])
    train_test_gap = train_deprel_only - test_deprel_only

    diagnostic = {
        "note": (
            "Additive form-vs-heads diagnostic; does not alter the primary read. "
            "Gold-heads cells are intentionally not serve-honest upper bounds. "
            "Also written to campaign_biaffine_labeler_diagnostic.json."
        ),
        "grid_2x2": grid_2x2,
        "noise_matched_biaffine": {
            "description": (
                "Separate BilinearRelationLabeler trained with governor = L0 "
                "predicted head at TRAIN time, gold deprel targets; served on "
                "predicted heads at test (isolates train/test head-source mismatch)."
            ),
            "metrics": noise_metrics,
            "epoch_losses": list(noise_labeler.epoch_losses),
            "config": {
                "rank": noise_config.rank,
                "epochs": noise_config.epochs,
                "learning_rate": noise_config.learning_rate,
                "l2_penalty": noise_config.l2_penalty,
                "seed": noise_config.seed,
            },
        },
        "train_test_deprel_gap": {
            "description": (
                "Primary biaffine (gold-head-governor training) scored on TRAIN "
                "with gold heads vs TEST deprel-only (predicted heads)."
            ),
            "train_deprel_only_accuracy": train_deprel_only,
            "test_deprel_only_accuracy": test_deprel_only,
            "gap_train_minus_test": train_test_gap,
        },
    }

    guard.assert_complete()

    scoreboard: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "scope": SCOPE,
        "measurement": (
            "matched unary vs biaffine relation-labeler on shared L0 predicted heads; "
            "not the certified build"
        ),
        "serve_honest_features": [
            "R1-predicted POS",
            "hashed lowercase surface form",
            "surface shape",
            "normalized sentence position",
        ],
        "gold_usage": (
            "Gold used only for TRAIN fit targets (deprel labels and, for the "
            "primary arm, gold head governors), final TEST scoring, and the "
            "explicit gold-heads diagnostic cells; no gold at predict time for "
            "deciding arms."
        ),
        "parameterization": (
            "Dozat-style low-rank biaffine: logits_r = (φ_i^T U) C_r (φ_g^T V)^T "
            "+ φ_i^T p_r + φ_g^T q_r + b_r; shared U,V rank projections; per-relation "
            "rank×rank core C_r. Features delegated to BilinearAttachmentScorer.phi."
        ),
        "oov_fallback": (
            f"Predict always argmaxes over train vocabulary R; never emits labels "
            f"outside R. GENERIC_FALLBACK={GENERIC_FALLBACK!r} is only used if R is "
            "empty (should not occur after fit)."
        ),
        "data_hash_sha256": hasher.hexdigest(),
        "test_read_counts": dict(guard.counts),
        "test_read_once": True,
        "seeds": list(SEEDS),
        "hyperparameters": {
            "rank": RANK,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "l2_penalty": L2_PENALTY,
            "seeds": list(SEEDS),
            "grad_clip": GRAD_CLIP,
            "relation_vocab_size": len(relation_vocab),
        },
        "l0_window": {
            "k": window,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "unary": unary_metrics,
        "biaffine_per_seed": {
            str(row["seed"]): {
                "seed": row["seed"],
                "las": row["las"],
                "las_gain": row["las_gain"],
                "deprel_only_accuracy": row["deprel_only_accuracy"],
                "deprel_only_gain": row["deprel_only_gain"],
                "case": row["case"],
                "case_gain": row["case_gain"],
                "metrics": row["metrics"],
                "epoch_losses": row["epoch_losses"],
                "config": row["config"],
            }
            for row in per_seed_rows
        },
        "partition_gains_mean": partition_gains,
        "read": primary_read,
        "through_line_case": through_line,
        "diagnostic": diagnostic,
        "predict_signature": str(inspect.signature(BilinearRelationLabeler.predict)),
    }
    return scoreboard, guard


def _fmt_metric_row(name: str, metric: Mapping[str, Any]) -> str:
    return (
        f"{name:<28}  "
        f"UAS={float(metric['uas']):.6f}  "
        f"LAS={float(metric['las']):.6f}  "
        f"LAS_coarse={float(metric['las_coarse']):.6f}  "
        f"deprel={float(metric['deprel_only_accuracy']):.6f}  "
        f"case_bearing={float(metric['case_bearing_deprel_accuracy']):.6f} "
        f"(n={metric['case_bearing_deprel_n']})  "
        f"case={float(metric['serve_honest_morph_case']):.6f}  "
        f"pair={float(metric['pairwise_sensitive_accuracy']):.6f}  "
        f"local={float(metric['local_accuracy']):.6f}"
    )


def print_scoreboard(scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard) -> None:
    """Print the full comparison, diagnostic table, and preregistered read."""
    print("\nBIAFFINE RELATION-LABELER — GSD TEST (grok lane)")
    print(f"run_tag: {scoreboard['run_tag']}")
    print(f"scope: {scoreboard['scope']}")
    print(
        "serve_honesty: R1-predicted POS + surface/shape/position only; "
        "no gold model features on deciding arms"
    )
    hp = scoreboard["hyperparameters"]
    print(
        f"biaffine hyperparameters: rank={hp['rank']} epochs={hp['epochs']} "
        f"lr={hp['learning_rate']} l2={hp['l2_penalty']} seeds={hp['seeds']} "
        f"|R|={hp['relation_vocab_size']}"
    )
    guard.assert_complete()
    print("TEST READ ONCE: CONFIRMED")

    unary = scoreboard["unary"]
    print("\n--- MATCHED COMPARISON (same L0 predicted heads) ---")
    print(_fmt_metric_row("UNARY", unary))
    for seed in scoreboard["seeds"]:
        row = scoreboard["biaffine_per_seed"][str(seed)]
        print(
            _fmt_metric_row(f"BIAFFINE seed={seed}", row["metrics"])
            + f"  LAS_gain={row['las_gain']:+.6f}  "
            f"deprel_gain={row['deprel_only_gain']:+.6f}  "
            f"case_gain={row['case_gain']:+.6f}"
        )

    print("\n--- PARTITION (mean biaffine−unary deprel-only gains) ---")
    gains = scoreboard["partition_gains_mean"]
    print(
        f"pairwise-sensitive {{nsubj,obj,iobj,obl,nmod,conj}}: "
        f"{gains['pairwise_sensitive']:+.6f}"
    )
    print(
        f"local {{det,amod,case,punct,aux,cop}}: "
        f"{gains['local']:+.6f}"
    )

    print("\n--- THE READ (prereg §5) ---")
    print(scoreboard["read"]["text"])
    print(scoreboard["through_line_case"]["text"])

    diagnostic = scoreboard["diagnostic"]
    print("\n--- DIAGNOSTIC 2×2 (form × heads) ---")
    for key in (
        "unary_x_predicted_heads",
        "biaffine_x_predicted_heads",
        "unary_x_gold_heads",
        "biaffine_x_gold_heads",
    ):
        cell = diagnostic["grid_2x2"][key]
        honest = "serve_honest=true" if cell["serve_honest"] else "serve_honest=false"
        print(f"{cell['label']}  [{honest}]")
        print("  " + _fmt_metric_row("", cell).strip())

    noise = diagnostic["noise_matched_biaffine"]
    print("\n--- NOISE-MATCHED BIAFFINE (train on predicted heads) ---")
    print(_fmt_metric_row("noise-matched", noise["metrics"]))

    gap = diagnostic["train_test_deprel_gap"]
    print("\n--- TRAIN−TEST DEPREL GAP (primary biaffine) ---")
    print(
        f"train_deprel_only={gap['train_deprel_only_accuracy']:.6f}  "
        f"test_deprel_only={gap['test_deprel_only_accuracy']:.6f}  "
        f"gap={gap['gap_train_minus_test']:+.6f}"
    )


def main() -> None:
    scoreboard, guard = build_scoreboard()
    OUTPUT_PATH.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    DIAGNOSTIC_OUTPUT_PATH.write_text(
        json.dumps(scoreboard["diagnostic"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_scoreboard(scoreboard, guard)
    print(f"structured_output: {OUTPUT_PATH}")
    print(f"diagnostic_output: {DIAGNOSTIC_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
