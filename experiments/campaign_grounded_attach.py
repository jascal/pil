"""Grounded-φ bilinear ATTACHMENT scorer gate (feature-ladder rung 2, attach).

Reuses the shipped low-rank bilinear head-scorer form from
``BilinearAttachmentScorer`` but substitutes per-word Qwen2.5-3B grounded
residuals for surface hashed features.  Three head arms (L0, surface-#119,
grounded) are scored on the covered-dependent arc set.  Attach and label layers
are chosen on DEV; TEST is read once under ``CampaignTestReadGuard``.
Pre-registered decision rules from ``docs/notes/grounded_attach_prereg.md`` §5
are applied verbatim.
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
from experiments.campaign_grounded_labeler import (  # noqa: E402
    CASE_BASELINE,
    CASE_DELIVERABLE,
    CHECKPOINT_LAYERS,
    DEV_NPZ,
    FEATURE_DIM,
    LAYER_INDICES,
    TEST_NPZ,
    TRAIN_NPZ,
    GroundedFeatureProvider,
    GroundedRelationLabeler,
    evaluate_dev_layer,
    is_doubly_covered,
    select_dev_layer,
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

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "campaign_grounded_attach.json"

RUN_TAG = "empirical"
SCOPE = (
    "axis-B grounded-φ bilinear ATTACHMENT scorer gate on covered-dependent arcs; "
    "matched L0 + surface-#119 baselines; conversion via grounded labels"
)
GRAD_CLIP = 5.0
SEED = 0
C7_SAMPLE_SIZE = 300
UAS_FIRES_THRESHOLD = 0.03
LAS_FIRES_THRESHOLD = 0.03


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis of a 1-d vector."""
    shifted = logits - np.max(logits)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


class GroundedAttachmentScorer:
    """Low-rank bilinear HEAD scorer over grounded residual word features.

    ROOT is never scored here: it has no grounded φ.  Decode always inherits the
    ROOT column (and any uncovered-governor column) from the caller-supplied L0
    fallback score matrix.
    """

    def __init__(self, config: BilinearConfig | None = None) -> None:
        self.config = config or BilinearConfig()
        if self.config.rank < 1:
            raise ValueError("rank must be positive")
        if self.config.epochs < 1:
            raise ValueError("epochs must be positive")
        self.feature_dim = FEATURE_DIM
        self._rng = np.random.default_rng(self.config.seed)
        scale = 1.0 / np.sqrt(self.feature_dim)
        self.U = self._rng.normal(0.0, scale, (self.feature_dim, self.config.rank))
        self.V = self._rng.normal(0.0, scale, (self.feature_dim, self.config.rank))
        self.b = 0.0
        self.epoch_losses: list[float] = []

    def _train_sentence(
        self,
        sent_id: str,
        gold_offsets: Sequence[int],
        provider: GroundedFeatureProvider,
    ) -> float | None:
        size = len(gold_offsets)
        covered_words = [i for i in range(size) if provider.covered(sent_id, i)]
        if len(covered_words) < 2:
            return None
        covered_set = set(covered_words)
        kept: list[tuple[int, int]] = []
        for index, offset in enumerate(gold_offsets):
            if offset == 0 or index not in covered_set:
                continue
            governor = resolve_governor_index(index, offset, size)
            if governor is None or governor == index:
                raise ValueError(f"invalid training head at token {index}: {offset}")
            if governor in covered_set:
                kept.append((index, governor))
        if not kept:
            return None

        word_features = np.stack([provider.get(sent_id, i) for i in covered_words])
        position = {idx: pos for pos, idx in enumerate(covered_words)}
        dependent_features = word_features[[position[d] for d, _g in kept]]
        dependent_projection = dependent_features @ self.U
        governor_projection = word_features @ self.V
        scores = dependent_projection @ governor_projection.T + self.b
        for row, (dep, _gov) in enumerate(kept):
            scores[row, position[dep]] = -np.inf
        target_positions = np.asarray(
            [position[g] for _d, g in kept], dtype=np.int64
        )
        batch_size = len(kept)
        maxima = np.max(scores, axis=1, keepdims=True)
        exponentiated = np.exp(scores - maxima)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        loss = float(
            np.mean(
                maxima[:, 0]
                + np.log(exponentiated.sum(axis=1))
                - scores[np.arange(batch_size), target_positions]
            )
        )

        score_gradient = probabilities
        score_gradient[np.arange(batch_size), target_positions] -= 1.0
        score_gradient /= batch_size
        grad_u = dependent_features.T @ (score_gradient @ governor_projection)
        grad_v = word_features.T @ (score_gradient.T @ dependent_projection)
        grad_u += self.config.l2_penalty * self.U
        grad_v += self.config.l2_penalty * self.V
        grad_norm = float(np.sqrt(np.square(grad_u).sum() + np.square(grad_v).sum()))
        if grad_norm > GRAD_CLIP:
            scale = GRAD_CLIP / grad_norm
            grad_u *= scale
            grad_v *= scale
        self.U -= self.config.learning_rate * grad_u
        self.V -= self.config.learning_rate * grad_v
        self.b -= self.config.learning_rate * float(score_gradient.sum())
        return loss

    def fit(
        self,
        sentences: Sequence[Sentence],
        heads: Mapping[str, HeadDeprelRecord],
        provider: GroundedFeatureProvider,
    ) -> None:
        """Fit on gold non-ROOT arcs whose dependent and governor are both covered."""
        if not sentences:
            raise ValueError("cannot fit on empty training data")
        self.epoch_losses = []
        any_kept = False
        for _epoch in range(self.config.epochs):
            order = self._rng.permutation(len(sentences))
            losses: list[float] = []
            for sentence_index in order:
                sentence = sentences[int(sentence_index)]
                head = aligned_head_record(sentence, heads)
                if not isinstance(head, HeadDeprelRecord):
                    raise TypeError("labeled dependency record required")
                loss = self._train_sentence(
                    sentence.sent_id, head.head_offset, provider
                )
                if loss is not None:
                    losses.append(loss)
                    any_kept = True
            if not losses:
                raise ValueError("no contributing batches in epoch")
            self.epoch_losses.append(float(np.mean(losses)))
        if not any_kept:
            raise ValueError("cannot fit on empty training data")

    def decode(
        self,
        tokens: Sequence[str],
        sent_id: str,
        l0_scores: np.ndarray,
        provider: GroundedFeatureProvider,
    ) -> tuple[int, ...]:
        """CLE-decode a tree, overwriting covered-dependent bilinear candidates."""
        matrix = np.asarray(l0_scores, dtype=np.float64).copy()
        size = len(tokens)
        if matrix.shape != (size, size + 1):
            raise ValueError(
                f"l0_scores must have shape ({size}, {size + 1}), got {matrix.shape}"
            )
        covered_words = [i for i in range(size) if provider.covered(sent_id, i)]
        for dep in covered_words:
            others = [g for g in covered_words if g != dep]
            if not others:
                continue
            dep_phi = provider.get(sent_id, dep)
            assert dep_phi is not None
            gov_phis = np.stack([provider.get(sent_id, g) for g in others])
            logits = (dep_phi @ self.U) @ (gov_phis @ self.V).T + self.b
            probabilities = _softmax(np.asarray(logits, dtype=np.float64))
            for governor, probability in zip(others, probabilities, strict=True):
                matrix[dep, governor + 1] = float(probability)
        return governor_columns_to_offsets(chu_liu_edmonds(matrix))


def covered_dependent_fraction(
    sentences: Sequence[Sentence],
    provider: GroundedFeatureProvider,
) -> float:
    """Fraction of all tokens whose dependent endpoint is grounded-covered."""
    total = 0
    covered = 0
    for sentence in sentences:
        for index in range(len(sentence.tokens)):
            total += 1
            if provider.covered(sentence.sent_id, index):
                covered += 1
    if total == 0:
        return 0.0
    return covered / total


def attach_doubly_covered_fraction(
    sentences: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    provider: GroundedFeatureProvider,
) -> float:
    """Fraction of tokens whose GOLD arc is doubly covered under ``provider``."""
    total = 0
    covered = 0
    for sentence in sentences:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        size = len(sentence.tokens)
        for index, offset in enumerate(head.head_offset):
            total += 1
            if is_doubly_covered(sentence.sent_id, index, offset, size, provider):
                covered += 1
    if total == 0:
        return 0.0
    return covered / total


def select_dev_attach_layer(sweep_rows: Sequence[Mapping[str, Any]]) -> int:
    """Pick layer_index with highest DEV UAS; smallest index wins ties."""
    if not sweep_rows:
        raise ValueError("cannot select a layer from an empty sweep")
    best = min(
        sweep_rows,
        key=lambda row: (-float(row["dev_uas"]), int(row["layer_index"])),
    )
    return int(best["layer_index"])


def _mean_pairwise_cosine(vectors: np.ndarray) -> float:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    gram = normalized @ normalized.T
    n = gram.shape[0]
    pair_count = n * (n - 1)
    if pair_count == 0:
        return 0.0
    return float((gram.sum() - np.trace(gram)) / pair_count)


def _effective_rank(vectors: np.ndarray) -> float:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values**2
    total = float(eigenvalues.sum())
    if total <= 0:
        return 0.0
    return float((total**2) / np.square(eigenvalues).sum())


def compute_c7_geometry(
    npz_path: Path = DEV_NPZ,
    *,
    seed: int = SEED,
    sample_size: int = C7_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """DEV-only near-orthogonality premise check over grounded residual rows."""
    data = np.load(npz_path, allow_pickle=False)
    residuals = np.asarray(data["last_residual"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for layer_index in LAYER_INDICES:
        layer_vectors = residuals[:, layer_index, :]
        n_rows = layer_vectors.shape[0]
        take = min(sample_size, n_rows)
        rng = np.random.default_rng(seed)
        if take == n_rows:
            sample = layer_vectors
        else:
            indices = rng.choice(n_rows, size=take, replace=False)
            sample = layer_vectors[indices]
        rows.append(
            {
                "layer_index": layer_index,
                "layer": CHECKPOINT_LAYERS[layer_index],
                "n_sampled": take,
                "mean_pairwise_cosine": _mean_pairwise_cosine(sample),
                "effective_rank": _effective_rank(sample),
            }
        )
    return rows


def collect_covered_dependent_keys(
    sentences: Sequence[Sentence],
    provider: GroundedFeatureProvider,
) -> list[tuple[str, int]]:
    """Ordered (sent_id, token_index) list for the covered-dependent mask."""
    keys: list[tuple[str, int]] = []
    for sentence in sentences:
        for index in range(len(sentence.tokens)):
            if provider.covered(sentence.sent_id, index):
                keys.append((sentence.sent_id, index))
    return keys


def score_uas_on_keys(
    predicted_by_key: Mapping[tuple[str, int], int],
    gold_by_key: Mapping[tuple[str, int], int],
    keys: Sequence[tuple[str, int]],
) -> float:
    """UAS over an explicit ordered key list (matched-set scoring)."""
    if not keys:
        raise ValueError("no covered-dependent tokens to score")
    matches = 0
    for key in keys:
        if predicted_by_key[key] == gold_by_key[key]:
            matches += 1
    return matches / len(keys)


def assert_matched_token_counts(
    arm_counts: Mapping[str, int],
    expected: int,
) -> None:
    """Belt-and-braces: every arm must score exactly ``expected`` tokens."""
    for name, count in arm_counts.items():
        if count != expected:
            raise AssertionError(
                f"matched-set token count mismatch for {name}: "
                f"got {count}, expected {expected}"
            )


def evaluate_dev_attach_layer(
    *,
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    scorer: GroundedAttachmentScorer,
    provider: GroundedFeatureProvider,
) -> dict[str, Any]:
    """Score grounded-attach UAS on DEV covered-dependent tokens only."""
    keys = collect_covered_dependent_keys(dev, provider)
    gold_by_key: dict[tuple[str, int], int] = {}
    predicted_by_key: dict[tuple[str, int], int] = {}
    for sentence in dev:
        head = aligned_head_record(sentence, dev_heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = dev_pos[sentence.sent_id]
        l0_scores = l0_edge_scores(l0_student, sentence.tokens, pos)
        offsets = scorer.decode(
            sentence.tokens, sentence.sent_id, l0_scores, provider
        )
        for index, offset in enumerate(offsets):
            key = (sentence.sent_id, index)
            if provider.covered(sentence.sent_id, index):
                predicted_by_key[key] = int(offset)
                gold_by_key[key] = int(head.head_offset[index])
    return {
        "layer_index": provider.layer_index,
        "layer": provider.layer,
        "dev_uas": score_uas_on_keys(predicted_by_key, gold_by_key, keys),
        "dev_covered_dependent_fraction": covered_dependent_fraction(dev, provider),
        "dev_doubly_covered_fraction": attach_doubly_covered_fraction(
            dev, dev_heads, provider
        ),
    }


def evaluate_matched_three_arm_uas(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    surface_bilinear: BilinearAttachmentScorer,
    grounded_scorer: GroundedAttachmentScorer,
    attach_provider: GroundedFeatureProvider,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
]:
    """Matched 3-arm UAS on the covered-dependent set; return L0 and grounded heads."""
    keys = collect_covered_dependent_keys(test, attach_provider)
    expected = len(keys)
    gold_by_key: dict[tuple[str, int], int] = {}
    heads_by_arm: dict[str, dict[str, tuple[int, ...]]] = {
        "l0": {},
        "surface119": {},
        "grounded": {},
    }

    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        sent_id = sentence.sent_id
        tokens = sentence.tokens

        l0_offsets = l0_student.predict_sentence(tokens, pos).head_offset
        surface_offsets = governor_columns_to_offsets(
            chu_liu_edmonds(surface_bilinear.score_matrix(tokens, pos))
        )
        grounded_offsets = grounded_scorer.decode(
            tokens,
            sent_id,
            l0_edge_scores(l0_student, tokens, pos),
            attach_provider,
        )
        heads_by_arm["l0"][sent_id] = tuple(int(v) for v in l0_offsets)
        heads_by_arm["surface119"][sent_id] = tuple(int(v) for v in surface_offsets)
        heads_by_arm["grounded"][sent_id] = tuple(int(v) for v in grounded_offsets)

        for index, gold_offset in enumerate(head.head_offset):
            gold_by_key[(sent_id, index)] = int(gold_offset)

    # Build each arm's predictions by iterating the IDENTICAL ordered key list.
    predicted: dict[str, dict[tuple[str, int], int]] = {
        arm: {} for arm in heads_by_arm
    }
    for arm, by_sent in heads_by_arm.items():
        for sent_id, index in keys:
            predicted[arm][(sent_id, index)] = int(by_sent[sent_id][index])

    arm_counts = {arm: len(predicted[arm]) for arm in predicted}
    assert_matched_token_counts(arm_counts, expected)

    metrics: dict[str, dict[str, Any]] = {}
    for arm in ("l0", "surface119", "grounded"):
        uas = score_uas_on_keys(predicted[arm], gold_by_key, keys)
        metrics[arm] = {
            "uas": uas,
            "tokens": expected,
        }
    return metrics, heads_by_arm["l0"], heads_by_arm["grounded"]


def score_conversion_arm(
    sentences: Sequence[Sentence],
    gold_heads: Mapping[str, HeadDeprelRecord],
    heads_by_sentence: Mapping[str, Sequence[int]],
    deprels_by_sentence: Mapping[str, Sequence[str]],
    case_preds_by_sentence: Mapping[str, Sequence[str]],
    label_provider: GroundedFeatureProvider,
) -> dict[str, Any]:
    """Accumulate LAS/case on doubly-covered arcs under THIS arm's own heads.

    baseline and #121 both use L0 heads → identical covered-token counts by
    construction; full uses grounded-attach heads and may differ.
    """
    pred_heads: list[int] = []
    pred_deprels: list[str] = []
    gold_head_list: list[int] = []
    gold_deprel_list: list[str] = []
    pred_cases: list[str] = []
    gold_cases: list[str] = []

    for sentence in sentences:
        head = aligned_head_record(sentence, gold_heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        sent_id = sentence.sent_id
        head_offsets = heads_by_sentence[sent_id]
        deprels = deprels_by_sentence[sent_id]
        case_preds = case_preds_by_sentence[sent_id]
        size = len(sentence.tokens)
        if not (len(head_offsets) == len(deprels) == len(case_preds) == size):
            raise ValueError("per-sentence conversion fields must align")

        keep_mask = [
            is_doubly_covered(sent_id, index, head_offsets[index], size, label_provider)
            for index in range(size)
        ]
        for index, keep in enumerate(keep_mask):
            if not keep:
                continue
            pred_heads.append(int(head_offsets[index]))
            pred_deprels.append(deprels[index])
            gold_head_list.append(int(head.head_offset[index]))
            gold_deprel_list.append(head.deprel[index])

        eligible = {
            index
            for index, gold_label in enumerate(sentence.targets["morph_case"])
            if gold_label != "-"
        }
        for index, keep in enumerate(keep_mask):
            if not keep or index not in eligible:
                continue
            pred_cases.append(case_preds[index])
            gold_cases.append(sentence.targets["morph_case"][index])

    if not pred_heads:
        raise ValueError("no doubly-covered tokens to score for conversion")
    strict = score_prediction_sequences(
        pred_heads, pred_deprels, gold_head_list, gold_deprel_list
    )
    case_acc = accuracy(pred_cases, gold_cases) if gold_cases else 0.0
    return {
        "uas": strict["uas"],
        "las_strict": strict["las"],
        "las": strict["las"],
        "deprel_only_accuracy": strict["deprel_only_accuracy"],
        "case": case_acc,
        "case_n": len(gold_cases),
        "tokens": strict["tokens"],
    }


def evaluate_conversion_point(
    sentences: Sequence[Sentence],
    gold_heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    heads_by_sentence: Mapping[str, Sequence[int]],
    *,
    use_grounded_labels: bool,
    l0_student: GermanR3DependencyStudent,
    case_student: GermanR1Student,
    grounded_labeler: GroundedRelationLabeler | None,
    label_provider: GroundedFeatureProvider,
) -> dict[str, Any]:
    """Serve-honest conversion: given heads (+ optional grounded labels) → LAS/case."""
    deprels_by_sentence: dict[str, tuple[str, ...]] = {}
    case_preds_by_sentence: dict[str, tuple[str, ...]] = {}

    for sentence in sentences:
        sent_id = sentence.sent_id
        pos = predicted_pos[sent_id]
        head_offsets = heads_by_sentence[sent_id]
        unary_labels = predict_deprels(l0_student, sentence.tokens, pos, head_offsets)
        if use_grounded_labels:
            if grounded_labeler is None:
                raise ValueError("grounded_labeler required when use_grounded_labels=True")
            deprels = grounded_labeler.predict(
                sentence.tokens,
                pos,
                head_offsets,
                sent_id,
                label_provider,
                unary_labels,
            )
        else:
            deprels = unary_labels
        deprels_by_sentence[sent_id] = tuple(deprels)

        baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        eligible = [
            index
            for index, gold_label in enumerate(sentence.targets["morph_case"])
            if gold_label != "-"
        ]
        case_result = run_predicted_case_sentence(
            case_student,
            sentence.tokens,
            baseline,
            pos,
            head_offsets,
            deprels,
            eligible,
        )
        case_preds_by_sentence[sent_id] = tuple(case_result.predictions)

    return score_conversion_arm(
        sentences,
        gold_heads,
        heads_by_sentence,
        deprels_by_sentence,
        case_preds_by_sentence,
        label_provider,
    )


def preregistered_attach_read(
    *,
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
    covered_dependent_fraction: Mapping[str, float],
    doubly_covered_fraction: Mapping[str, float],
    c7_cosine: float,
    c7_effrank: float,
) -> dict[str, Any]:
    """Apply signed grounded-attach preregistration section 5 verbatim."""
    uas_gain_vs_l0 = float(uas_grounded) - float(uas_l0)
    uas_gain_vs_surface119 = float(uas_grounded) - float(uas_surface119)
    las_full_gain_vs_base = float(las_full) - float(las_base)

    if uas_gain_vs_l0 <= 0.0:
        verdict = "HALTED"
        reason = "grounded-attach UAS does not beat L0"
    elif (
        uas_gain_vs_l0 < UAS_FIRES_THRESHOLD
        or uas_grounded <= uas_surface119
        or las_full_gain_vs_base < LAS_FIRES_THRESHOLD
    ):
        verdict = "IN-BETWEEN"
        failed: list[str] = []
        if uas_gain_vs_l0 < UAS_FIRES_THRESHOLD:
            failed.append("UAS gain vs L0 is below +0.03")
        if uas_grounded <= uas_surface119:
            failed.append("grounded UAS does not beat surface-#119")
        if las_full_gain_vs_base < LAS_FIRES_THRESHOLD:
            failed.append("conversion strict-LAS gain vs baseline is below +0.03")
        reason = "; ".join(failed)
    else:
        verdict = "FIRES"
        reason = "all preregistered capability conditions are met"

    if case_full >= CASE_DELIVERABLE:
        through_line = {
            "met": True,
            "plateau": False,
            "text": (
                f"through-line met: serve-honest case={case_full:.4f} ≥ {CASE_DELIVERABLE}"
            ),
        }
    elif case_full > CASE_BASELINE:
        through_line = {
            "met": False,
            "plateau": True,
            "text": (
                f"through-line plateau: serve-honest case={case_full:.4f} "
                f"< {CASE_DELIVERABLE} but > ~{CASE_BASELINE}"
            ),
        }
    elif case_full <= case_base:
        through_line = {
            "met": False,
            "plateau": False,
            "text": (
                f"through-line diagnose: serve-honest case={case_full:.4f} "
                f"≤ baseline {case_base:.4f}"
            ),
        }
    else:
        through_line = {
            "met": False,
            "plateau": False,
            "text": (
                f"through-line below plateau: serve-honest case={case_full:.4f} "
                f"above baseline but ≤ ~{CASE_BASELINE}"
            ),
        }

    text = (
        f"{verdict}: grounded-attach UAS l0={uas_l0:.4f} surface119={uas_surface119:.4f} "
        f"grounded={uas_grounded:.4f} (ΔL0={uas_gain_vs_l0:+.4f}, "
        f"Δsurface={uas_gain_vs_surface119:+.4f}); conversion LAS "
        f"base={las_base:.4f} #121={las_121:.4f} full={las_full:.4f} "
        f"(Δfull-base={las_full_gain_vs_base:+.4f}); case base={case_base:.4f} "
        f"#121={case_121:.4f} full={case_full:.4f}; dev attach layer={dev_layer}; "
        f"c7 cosine={c7_cosine:.4f} effrank={c7_effrank:.2f}. {reason}."
    )
    return {
        "verdict": verdict,
        "uas_l0": float(uas_l0),
        "uas_surface119": float(uas_surface119),
        "uas_grounded": float(uas_grounded),
        "uas_gain_vs_l0": uas_gain_vs_l0,
        "uas_gain_vs_surface119": uas_gain_vs_surface119,
        "las_base": float(las_base),
        "las_121": float(las_121),
        "las_full": float(las_full),
        "las_full_gain_vs_base": las_full_gain_vs_base,
        "case_base": float(case_base),
        "case_121": float(case_121),
        "case_full": float(case_full),
        "dev_layer": int(dev_layer),
        "covered_dependent_fraction": {
            key: float(value) for key, value in covered_dependent_fraction.items()
        },
        "doubly_covered_fraction": {
            key: float(value) for key, value in doubly_covered_fraction.items()
        },
        "c7_cosine": float(c7_cosine),
        "c7_effrank": float(c7_effrank),
        "through_line": through_line,
        "text": text,
    }


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit on train, sweep attach+label layers on DEV, then read TEST once."""
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

    print("DEV: loading GSD dev once for L0 window + attach/label layer sweeps.", flush=True)
    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_heads_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    window, dev_coverage, coverage_curve = choose_window(dev_heads_records)
    dev_heads = head_deprel_by_sent(dev_heads_records)
    dev_pos = _predicted_pos_by_sentence(r1_student, dev)

    l0_student = GermanR3DependencyStudent(window)
    l0_student.fit(train, train_heads, train_pos)

    attach_config = BilinearConfig()
    print(
        f"FIT: surface-#119 bilinear attach rank={attach_config.rank} "
        f"epochs={attach_config.epochs} lr={attach_config.learning_rate} seed=0.",
        flush=True,
    )
    surface_bilinear = BilinearAttachmentScorer(attach_config)
    surface_bilinear.fit(train, train_heads, train_pos)
    print(
        "FIT: surface attach epoch losses="
        + ", ".join(f"{loss:.6f}" for loss in surface_bilinear.epoch_losses),
        flush=True,
    )

    print("C7: DEV-only near-orthogonality premise check over grounded residuals.", flush=True)
    c7_geometry = compute_c7_geometry(DEV_NPZ, seed=SEED, sample_size=C7_SAMPLE_SIZE)
    for row in c7_geometry:
        print(
            f"C7 layer={row['layer']}: n={row['n_sampled']} "
            f"mean_pairwise_cosine={row['mean_pairwise_cosine']:.6f} "
            f"effective_rank={row['effective_rank']:.2f}",
            flush=True,
        )

    # --- DEV attach-layer sweep ---
    attach_sweep_rows: list[dict[str, Any]] = []
    attach_by_layer: dict[int, GroundedAttachmentScorer] = {}
    for layer_index in LAYER_INDICES:
        print(
            f"FIT: grounded attach layer_index={layer_index} "
            f"layer={CHECKPOINT_LAYERS[layer_index]}.",
            flush=True,
        )
        train_provider = GroundedFeatureProvider(TRAIN_NPZ, layer_index)
        scorer = GroundedAttachmentScorer(BilinearConfig())
        scorer.fit(train, train_heads, train_provider)
        print(
            f"FIT: grounded attach layer={CHECKPOINT_LAYERS[layer_index]} epoch losses="
            + ", ".join(f"{loss:.6f}" for loss in scorer.epoch_losses),
            flush=True,
        )
        dev_provider = GroundedFeatureProvider(DEV_NPZ, layer_index)
        row = evaluate_dev_attach_layer(
            dev=dev,
            dev_heads=dev_heads,
            dev_pos=dev_pos,
            l0_student=l0_student,
            scorer=scorer,
            provider=dev_provider,
        )
        attach_sweep_rows.append(row)
        attach_by_layer[layer_index] = scorer
        print(
            f"DEV attach sweep: layer={row['layer']} uas={row['dev_uas']:.6f} "
            f"covered_dep={row['dev_covered_dependent_fraction']:.6f} "
            f"doubly={row['dev_doubly_covered_fraction']:.6f}",
            flush=True,
        )

    selected_attach_layer_index = select_dev_attach_layer(attach_sweep_rows)
    selected_attach_layer = CHECKPOINT_LAYERS[selected_attach_layer_index]
    selected_attach_scorer = attach_by_layer[selected_attach_layer_index]
    selected_attach_dev_row = next(
        row
        for row in attach_sweep_rows
        if int(row["layer_index"]) == selected_attach_layer_index
    )
    print(
        f"DEV selected attach layer_index={selected_attach_layer_index} "
        f"layer={selected_attach_layer} uas={selected_attach_dev_row['dev_uas']:.6f}",
        flush=True,
    )

    # --- DEV label-layer sweep (identical to #121; L0 heads only) ---
    label_config = RelationLabelerConfig(seed=0)
    label_sweep_rows: list[dict[str, Any]] = []
    labeler_by_layer: dict[int, GroundedRelationLabeler] = {}
    for layer_index in LAYER_INDICES:
        print(
            f"FIT: grounded labeler layer_index={layer_index} "
            f"layer={CHECKPOINT_LAYERS[layer_index]}.",
            flush=True,
        )
        train_provider = GroundedFeatureProvider(TRAIN_NPZ, layer_index)
        labeler = GroundedRelationLabeler(label_config)
        labeler.fit(train, train_heads, train_pos, train_provider)
        print(
            f"FIT: grounded labeler layer={CHECKPOINT_LAYERS[layer_index]} epoch losses="
            + ", ".join(f"{loss:.6f}" for loss in labeler.epoch_losses),
            flush=True,
        )
        dev_provider = GroundedFeatureProvider(DEV_NPZ, layer_index)
        row = evaluate_dev_layer(
            dev=dev,
            dev_heads=dev_heads,
            dev_pos=dev_pos,
            l0_student=l0_student,
            grounded_labeler=labeler,
            provider=dev_provider,
        )
        label_sweep_rows.append(row)
        labeler_by_layer[layer_index] = labeler
        print(
            f"DEV label sweep: layer={row['layer']} "
            f"deprel-only={row['dev_deprel_only_accuracy']:.6f} "
            f"doubly-covered={row['dev_doubly_covered_fraction']:.6f}",
            flush=True,
        )

    selected_label_layer_index = select_dev_layer(label_sweep_rows)
    selected_label_layer = CHECKPOINT_LAYERS[selected_label_layer_index]
    selected_labeler = labeler_by_layer[selected_label_layer_index]
    selected_label_dev_row = next(
        row
        for row in label_sweep_rows
        if int(row["layer_index"]) == selected_label_layer_index
    )
    print(
        f"DEV selected label layer_index={selected_label_layer_index} "
        f"layer={selected_label_layer} "
        f"deprel-only={selected_label_dev_row['dev_deprel_only_accuracy']:.6f}",
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

    test_attach_provider = GroundedFeatureProvider(
        TEST_NPZ, selected_attach_layer_index
    )
    test_label_provider = GroundedFeatureProvider(TEST_NPZ, selected_label_layer_index)

    arm_metrics, l0_heads_by_sent, grounded_heads_by_sent = evaluate_matched_three_arm_uas(
        test,
        test_heads,
        test_pos,
        l0_student,
        surface_bilinear,
        selected_attach_scorer,
        test_attach_provider,
    )

    # Attach coverage (selected attach layer) — arm-invariant.
    attach_coverage = {
        "dev_covered_dependent": float(
            selected_attach_dev_row["dev_covered_dependent_fraction"]
        ),
        "dev_doubly_covered": float(selected_attach_dev_row["dev_doubly_covered_fraction"]),
        "test_covered_dependent": covered_dependent_fraction(test, test_attach_provider),
        "test_doubly_covered": attach_doubly_covered_fraction(
            test, test_heads, test_attach_provider
        ),
    }
    # Label coverage from #121-style L0-head doubly-covered fraction on DEV,
    # plus TEST measured under the selected label provider against L0 heads.
    label_test_covered = 0
    label_test_total = 0
    for sentence in test:
        l0_offsets = l0_heads_by_sent[sentence.sent_id]
        size = len(sentence.tokens)
        for index, offset in enumerate(l0_offsets):
            label_test_total += 1
            if is_doubly_covered(
                sentence.sent_id, index, offset, size, test_label_provider
            ):
                label_test_covered += 1
    label_coverage = {
        "dev_doubly_covered": float(selected_label_dev_row["dev_doubly_covered_fraction"]),
        "test_doubly_covered": (
            label_test_covered / label_test_total if label_test_total else 0.0
        ),
    }

    print("CONVERSION: baseline (L0 heads + unary labels).", flush=True)
    conversion_base = evaluate_conversion_point(
        test,
        test_heads,
        test_pos,
        l0_heads_by_sent,
        use_grounded_labels=False,
        l0_student=l0_student,
        case_student=r1_student,
        grounded_labeler=None,
        label_provider=test_label_provider,
    )
    print("CONVERSION: #121 (L0 heads + grounded labels).", flush=True)
    conversion_121 = evaluate_conversion_point(
        test,
        test_heads,
        test_pos,
        l0_heads_by_sent,
        use_grounded_labels=True,
        l0_student=l0_student,
        case_student=r1_student,
        grounded_labeler=selected_labeler,
        label_provider=test_label_provider,
    )
    print("CONVERSION: full (grounded heads + grounded labels).", flush=True)
    conversion_full = evaluate_conversion_point(
        test,
        test_heads,
        test_pos,
        grounded_heads_by_sent,
        use_grounded_labels=True,
        l0_student=l0_student,
        case_student=r1_student,
        grounded_labeler=selected_labeler,
        label_provider=test_label_provider,
    )
    # baseline vs #121 share L0 heads → identical token counts.
    if conversion_base["tokens"] != conversion_121["tokens"]:
        raise AssertionError(
            "baseline and #121 conversion must share the same doubly-covered n "
            f"(L0 heads): base={conversion_base['tokens']} "
            f"121={conversion_121['tokens']}"
        )

    guard.assert_complete()

    c7_selected = next(
        row
        for row in c7_geometry
        if int(row["layer_index"]) == selected_attach_layer_index
    )
    covered_dependent_fraction_map = {
        "dev": attach_coverage["dev_covered_dependent"],
        "test": attach_coverage["test_covered_dependent"],
    }
    doubly_covered_fraction_map = {
        "dev": attach_coverage["dev_doubly_covered"],
        "test": attach_coverage["test_doubly_covered"],
    }
    read = preregistered_attach_read(
        uas_l0=float(arm_metrics["l0"]["uas"]),
        uas_surface119=float(arm_metrics["surface119"]["uas"]),
        uas_grounded=float(arm_metrics["grounded"]["uas"]),
        las_base=float(conversion_base["las_strict"]),
        las_121=float(conversion_121["las_strict"]),
        las_full=float(conversion_full["las_strict"]),
        case_base=float(conversion_base["case"]),
        case_121=float(conversion_121["case"]),
        case_full=float(conversion_full["case"]),
        dev_layer=selected_attach_layer,
        covered_dependent_fraction=covered_dependent_fraction_map,
        doubly_covered_fraction=doubly_covered_fraction_map,
        c7_cosine=float(c7_selected["mean_pairwise_cosine"]),
        c7_effrank=float(c7_selected["effective_rank"]),
    )

    scoreboard: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "scope": SCOPE,
        "measurement": (
            "serve-honest three-arm attachment comparison on covered-dependent arcs; "
            "conversion = grounded heads + grounded labels → LAS-strict + case cascade"
        ),
        "serve_honest_features": [
            "R1-predicted POS (L0 / surface arms; unused by grounded φ)",
            "hashed surface/shape/position (surface-#119 only)",
            "Qwen2.5-3B last_residual grounded φ at DEV-selected attach layer",
            "L0 edge-score fallback for ROOT / uncovered governors (grounded arm)",
            "Qwen2.5-3B grounded φ at DEV-selected label layer (conversion labels)",
        ],
        "gold_usage": (
            "GSD train gold head offsets are fit-time attachment targets where both "
            "endpoints are covered; test gold heads/deprels/case are final scoring "
            "oracles only; attach and label layers selected on DEV only"
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
            "rank": attach_config.rank,
            "epochs": attach_config.epochs,
            "learning_rate": attach_config.learning_rate,
            "l2_penalty": attach_config.l2_penalty,
            "seed": SEED,
            "feature_dim": FEATURE_DIM,
            "gradient_norm_clip": GRAD_CLIP,
            "checkpoint_layers": list(CHECKPOINT_LAYERS),
            "attach_config": "BilinearConfig() defaults (same as #119)",
            "label_config": "RelationLabelerConfig(seed=0) defaults (same as #121)",
        },
        "dev_attach_layer_sweep": attach_sweep_rows,
        "selected_attach_layer_index": selected_attach_layer_index,
        "selected_attach_layer": selected_attach_layer,
        "dev_label_layer_sweep": label_sweep_rows,
        "selected_label_layer_index": selected_label_layer_index,
        "selected_label_layer": selected_label_layer,
        "attach_coverage": attach_coverage,
        "label_coverage": label_coverage,
        "c7_geometry": c7_geometry,
        "arms": arm_metrics,
        "conversion": {
            "baseline": conversion_base,
            "121": conversion_121,
            "full": conversion_full,
        },
        "read": read,
    }
    return scoreboard, guard


def print_scoreboard(scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard) -> None:
    """Print DEV sweeps, C7, 3-arm UAS, conversion, and the §5 preregistered read."""
    print("\nGROUNDED-φ BILINEAR ATTACHMENT SCORER GATE — GSD")
    print(f"run_tag: {scoreboard['run_tag']}")
    print(f"scope: {scoreboard['scope']}")
    config = scoreboard["fixed_hyperparameters"]
    print(
        f"config: rank={config['rank']} epochs={config['epochs']} "
        f"lr={config['learning_rate']} l2={config['l2_penalty']} seed={config['seed']} "
        f"feature_dim={config['feature_dim']} (single seed 0 throughout)"
    )
    print(
        "matched control: every arm scored on the identical covered-dependent token set; "
        "surface-#119 and L0 use their own head decisions"
    )
    print(
        "serve honesty: grounded φ from sentence text dump; R1-predicted POS; "
        "no gold dependency fields at predict time; L0 fallback for ROOT/uncovered"
    )

    print("\nDEV ATTACH LAYER SWEEP (UAS on covered-dependent tokens)")
    print(
        f"{'layer_idx':<10} {'layer':<8} {'dev_uas':<12} "
        f"{'covered_dep':<14} {'doubly_cov':<12}"
    )
    for row in scoreboard["dev_attach_layer_sweep"]:
        print(
            f"{row['layer_index']:<10} {row['layer']:<8} "
            f"{row['dev_uas']:<12.6f} "
            f"{row['dev_covered_dependent_fraction']:<14.6f} "
            f"{row['dev_doubly_covered_fraction']:<12.6f}"
        )
    print(
        f"selected attach: layer_index={scoreboard['selected_attach_layer_index']} "
        f"layer={scoreboard['selected_attach_layer']}"
    )

    print("\nDEV LABEL LAYER SWEEP (deprel-only on doubly-covered arcs, L0 heads)")
    print(f"{'layer_idx':<10} {'layer':<8} {'deprel_only':<14} {'doubly_covered':<14}")
    for row in scoreboard["dev_label_layer_sweep"]:
        print(
            f"{row['layer_index']:<10} {row['layer']:<8} "
            f"{row['dev_deprel_only_accuracy']:<14.6f} "
            f"{row['dev_doubly_covered_fraction']:<14.6f}"
        )
    print(
        f"selected label: layer_index={scoreboard['selected_label_layer_index']} "
        f"layer={scoreboard['selected_label_layer']}"
    )

    ac = scoreboard["attach_coverage"]
    lc = scoreboard["label_coverage"]
    print(
        f"\nattach coverage: covered-dependent dev={ac['dev_covered_dependent']:.6f} "
        f"test={ac['test_covered_dependent']:.6f}; "
        f"doubly-covered (gold arcs) dev={ac['dev_doubly_covered']:.6f} "
        f"test={ac['test_doubly_covered']:.6f}"
    )
    print(
        f"label coverage: doubly-covered (L0 heads) dev={lc['dev_doubly_covered']:.6f} "
        f"test={lc['test_doubly_covered']:.6f}"
    )

    print("\nC7 NEAR-ORTHOGONALITY PREMISE (DEV residual sample)")
    print(
        f"{'layer_idx':<10} {'layer':<8} {'n_sampled':<12} "
        f"{'mean_cos':<12} {'eff_rank':<12}"
    )
    for row in scoreboard["c7_geometry"]:
        print(
            f"{row['layer_index']:<10} {row['layer']:<8} "
            f"{row['n_sampled']:<12} "
            f"{row['mean_pairwise_cosine']:<12.6f} "
            f"{row['effective_rank']:<12.2f}"
        )
    selected_c7 = next(
        row
        for row in scoreboard["c7_geometry"]
        if int(row["layer_index"]) == int(scoreboard["selected_attach_layer_index"])
    )
    cos = float(selected_c7["mean_pairwise_cosine"])
    eff = float(selected_c7["effective_rank"])
    n_samp = int(selected_c7["n_sampled"])
    if cos < 0.15 and eff > 0.25 * min(n_samp, FEATURE_DIM):
        c7_statement = (
            f"substrate at selected attach layer looks near-orthogonal "
            f"(mean pairwise cosine={cos:.4f} near 0; effective rank={eff:.1f} "
            f"is a substantial fraction of n_sampled={n_samp} / dim={FEATURE_DIM})"
        )
    else:
        c7_statement = (
            f"substrate at selected attach layer is NOT clearly near-orthogonal "
            f"(mean pairwise cosine={cos:.4f}; effective rank={eff:.1f} of "
            f"n_sampled={n_samp} / dim={FEATURE_DIM}) — read the attach result "
            f"against this premise"
        )
    print(f"C7 statement: {c7_statement}")

    print("\n3-ARM UAS (covered-dependent tokens, matched set)")
    print(f"{'arm':<15} {'UAS':<12} {'tokens':<10}")
    arms = scoreboard["arms"]
    for name, key in (
        ("L0", "l0"),
        ("surface-#119", "surface119"),
        ("grounded", "grounded"),
    ):
        m = arms[key]
        print(f"{name:<15} {m['uas']:<12.6f} {m['tokens']:<10}")

    print("\nCONVERSION (doubly-covered under each point's own heads + label provider)")
    print(f"{'point':<12} {'LAS-strict':<12} {'case':<12} {'tokens':<10}")
    conv = scoreboard["conversion"]
    for name, key in (
        ("baseline", "baseline"),
        ("#121", "121"),
        ("full", "full"),
    ):
        m = conv[key]
        print(
            f"{name:<12} {m['las_strict']:<12.6f} "
            f"{m['case']:<12.6f} {m['tokens']:<10}"
        )

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
