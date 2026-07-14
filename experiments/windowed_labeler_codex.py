"""Windowed count-table deprel labeler on matched serve-honest L0 heads.

The only new model primitive in this campaign is a deterministic context-key
front end for the shipped unary deprel count table.  Window keys use predicted
POS and coarse surface shape; unsupported keys fall through inline to the
shipped per-position unary key hierarchy.
"""

from __future__ import annotations

import hashlib
import json
import string
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.attach_levers_codex import (  # noqa: E402
    predict_deprels,
    score_case_bearing_deprels,
)
from experiments.german_r1_codex import (  # noqa: E402
    DATA_ROOT,
    CountTable,
    GermanR1Student,
    RegisterLayer,
    Sentence,
    accuracy,
    fit_count_table,
    load_split,
)
from experiments.german_r3_codex import (  # noqa: E402
    CampaignTestReadGuard,
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
    head_deprel_by_sent,
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "windowed_labeler_codex.json"
RUN_TAG = "empirical"
WINDOW_RADII = (0, 1, 2)
MIN_SUPPORTS = (5, 10, 20, 50)
PAIRWISE_SENSITIVE_DEPRELS = frozenset({"nsubj", "obj", "iobj", "obl", "nmod", "conj"})
LOCAL_DEPRELS = frozenset({"det", "amod", "case", "punct", "aux", "cop"})
CASE_BASELINE = 0.76
CASE_DELIVERABLE = 0.90


@dataclass(frozen=True)
class WindowPrediction:
    """Labels plus whether and where a supported window key fired."""

    labels: tuple[str, ...]
    windowed_fired: tuple[bool, ...]
    fired_radius: tuple[int | None, ...]


def coarse_deprel(label: str) -> str:
    """Return the base UD relation, stripping any colon-delimited subtype."""
    return label.split(":", 1)[0]


def coarse_surface_shape(token: str) -> str:
    """Map a surface token to a compact form-free shape category."""
    if not token:
        return "EMPTY"
    if all(character in string.punctuation for character in token):
        return "PUNCT"
    if token.isdigit():
        return "DIGIT"
    if token.isupper():
        return "UPPER"
    if token.istitle():
        return "TITLE"
    if token.islower():
        return "LOWER"
    if any(character.isdigit() for character in token):
        return "ALNUM"
    return "MIXED"


def _neighbor_context(
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    center: int | None,
    radius: int,
) -> tuple[tuple[int, str, str], ...]:
    if radius < 0:
        raise ValueError("context radius must be non-negative")
    if center is None:
        return tuple(
            (relative, "<INVALID>", "<INVALID>")
            for relative in range(-radius, radius + 1)
            if relative
        )
    context: list[tuple[int, str, str]] = []
    for relative in range(-radius, radius + 1):
        if not relative:
            continue
        neighbor = center + relative
        if neighbor < 0:
            context.append((relative, "<BOUNDARY>", "<BOS>"))
        elif neighbor >= len(tokens):
            context.append((relative, "<BOUNDARY>", "<EOS>"))
        else:
            context.append(
                (
                    relative,
                    predicted_pos[neighbor],
                    coarse_surface_shape(tokens[neighbor]),
                )
            )
    return tuple(context)


def windowed_relation_key_levels(
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    index: int,
    head_offsets: Sequence[int],
    radius: int,
) -> tuple[Any, ...]:
    """Build ``(*windowed_levels, *unary_levels)`` for one true sentence position."""
    if not (len(tokens) == len(predicted_pos) == len(head_offsets)):
        raise ValueError("tokens, predicted POS, and head offsets must align")
    if index < 0 or index >= len(tokens):
        raise IndexError(f"token index outside sentence: {index}")
    if radius < 0:
        raise ValueError("window radius must be non-negative")
    offset = int(head_offsets[index])
    unary_levels = _relation_key_levels(tokens, predicted_pos, index, offset)
    governor = resolve_governor_index(index, offset, len(tokens))
    governor_context = _neighbor_context(tokens, predicted_pos, governor, 1)
    # unary_levels[2] is the shipped form-free POS/position/current-shape plus
    # predicted-head structure level.  It keeps the new context key estimable;
    # every shipped level, including lexical levels, remains inline below it.
    window_anchor = unary_levels[2]
    windowed_levels = tuple(
        (
            "WINDOW",
            current_radius,
            window_anchor,
            _neighbor_context(tokens, predicted_pos, index, current_radius),
            governor_context,
        )
        for current_radius in range(radius, -1, -1)
    )
    return (*windowed_levels, *unary_levels)


TrainingExample = tuple[
    Sequence[str],
    Sequence[str],
    Sequence[int],
    Sequence[str],
]


class WindowedDeprelLabeler:
    """Supported window tables followed by the shipped unary backoff inline."""

    def __init__(
        self,
        unary_student: GermanR3DependencyStudent,
        *,
        max_radius: int = max(WINDOW_RADII),
        radius: int | None = None,
        min_support: int = min(MIN_SUPPORTS),
    ) -> None:
        if max_radius < 0:
            raise ValueError("maximum radius must be non-negative")
        self.unary_student = unary_student
        self.max_radius = max_radius
        self.radius = max_radius if radius is None else radius
        self.min_support = min_support
        self.window_tables: dict[int, CountTable] = {}
        self.configure(self.radius, self.min_support)

    def configure(self, radius: int, min_support: int) -> None:
        """Select a fitted radius/support configuration without refitting counts."""
        if radius < 0 or radius > self.max_radius:
            raise ValueError(f"radius must be between 0 and {self.max_radius}")
        if min_support < 1:
            raise ValueError("minimum support must be positive")
        self.radius = radius
        self.min_support = min_support

    @staticmethod
    def _key_for_radius(
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        index: int,
        head_offsets: Sequence[int],
        radius: int,
    ) -> Any:
        return windowed_relation_key_levels(
            tokens, predicted_pos, index, head_offsets, radius
        )[0]

    def fit_examples(self, examples: Iterable[TrainingExample]) -> None:
        """Fit every radius from materialized train-only examples."""
        materialized = tuple(examples)
        if not materialized:
            raise ValueError("cannot fit an empty windowed labeler")
        for radius in range(self.max_radius + 1):
            rows = (
                (
                    self._key_for_radius(tokens, pos, index, offsets, radius),
                    label,
                )
                for tokens, pos, offsets, labels in materialized
                for index, label in enumerate(labels)
            )
            self.window_tables[radius] = fit_count_table(rows, minsupp=1, mindet=0.0)

    def fit(
        self,
        train: Sequence[Sentence],
        heads: Mapping[str, HeadDeprelRecord],
        predicted_pos: Mapping[str, Sequence[str]],
        predicted_heads: Mapping[str, Sequence[int]],
    ) -> None:
        """Fit window counts on TRAIN labels with predicted POS/heads as key features."""
        examples: list[TrainingExample] = []
        for sentence in train:
            head = aligned_head_record(sentence, heads)
            if not isinstance(head, HeadDeprelRecord):
                raise TypeError("labeled dependency record required")
            examples.append(
                (
                    sentence.tokens,
                    predicted_pos[sentence.sent_id],
                    predicted_heads[sentence.sent_id],
                    head.deprel,
                )
            )
        self.fit_examples(examples)

    def _unary_table(self) -> Any:
        relation_table = self.unary_student.deprel_table
        if relation_table is None:
            raise RuntimeError("shipped unary deprel table used before fit")
        return relation_table

    def predict_with_diagnostics(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
        *,
        radius: int | None = None,
        min_support: int | None = None,
    ) -> WindowPrediction:
        """Predict from supported windows, then per-position shipped unary keys."""
        selected_radius = self.radius if radius is None else radius
        threshold = self.min_support if min_support is None else min_support
        if selected_radius < 0 or selected_radius > self.max_radius:
            raise ValueError(f"radius must be between 0 and {self.max_radius}")
        if threshold < 1:
            raise ValueError("minimum support must be positive")
        if len(self.window_tables) != self.max_radius + 1:
            raise RuntimeError("windowed labeler used before fit")
        unary_table = self._unary_table()
        labels: list[str] = []
        fired: list[bool] = []
        fired_radius: list[int | None] = []
        for index in range(len(tokens)):
            levels = windowed_relation_key_levels(
                tokens, predicted_pos, index, head_offsets, selected_radius
            )
            window_level_count = selected_radius + 1
            chosen: str | None = None
            chosen_radius: int | None = None
            for table_radius, key in zip(
                range(selected_radius, -1, -1),
                levels[:window_level_count],
                strict=True,
            ):
                entry = self.window_tables[table_radius].lookup(key)
                if entry is not None and entry.total >= threshold:
                    chosen = entry.label
                    chosen_radius = table_radius
                    break
            if chosen is None:
                # Crucially, these unary keys were built for this index from the
                # complete head-offset sequence.  No length-one slice is used.
                chosen = unary_table.predict(levels[window_level_count:])
                fired.append(False)
                fired_radius.append(None)
            else:
                fired.append(True)
                fired_radius.append(chosen_radius)
            labels.append(chosen)
        return WindowPrediction(tuple(labels), tuple(fired), tuple(fired_radius))

    def predict(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
        *,
        radius: int | None = None,
        min_support: int | None = None,
    ) -> tuple[str, ...]:
        """Return labels only; diagnostics remain available from the companion method."""
        return self.predict_with_diagnostics(
            tokens,
            predicted_pos,
            head_offsets,
            radius=radius,
            min_support=min_support,
        ).labels


def predict_heads_by_sentence(
    student: GermanR3DependencyStudent,
    sentences: Sequence[Sentence],
    predicted_pos: Mapping[str, Sequence[str]],
) -> dict[str, tuple[int, ...]]:
    """Compute the common serve-honest L0 heads once per in-memory sentence."""
    return {
        sentence.sent_id: student.predict_heads(
            sentence.tokens, predicted_pos[sentence.sent_id]
        )
        for sentence in sentences
    }


def _gold_deprels(
    sentences: Sequence[Sentence], heads: Mapping[str, HeadDeprelRecord]
) -> tuple[str, ...]:
    labels: list[str] = []
    for sentence in sentences:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        labels.extend(head.deprel)
    return tuple(labels)


def sweep_dev_configurations(
    labeler: WindowedDeprelLabeler,
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    common_heads: Mapping[str, Sequence[int]],
    radii: Sequence[int] = WINDOW_RADII,
    thresholds: Sequence[int] = MIN_SUPPORTS,
) -> dict[str, Any]:
    """Measure every (M,T) on DEV only and choose accuracy/T/-M deterministically."""
    gold = _gold_deprels(dev, dev_heads)
    unary_predictions: list[str] = []
    for sentence in dev:
        pos = predicted_pos[sentence.sent_id]
        offsets = common_heads[sentence.sent_id]
        unary_predictions.extend(
            predict_deprels(labeler.unary_student, sentence.tokens, pos, offsets)
        )
    unary_accuracy = accuracy(unary_predictions, gold)
    rows: list[dict[str, float | int]] = []
    for radius in radii:
        for threshold in thresholds:
            predicted: list[str] = []
            fired = 0
            token_count = 0
            for sentence in dev:
                result = labeler.predict_with_diagnostics(
                    sentence.tokens,
                    predicted_pos[sentence.sent_id],
                    common_heads[sentence.sent_id],
                    radius=radius,
                    min_support=threshold,
                )
                predicted.extend(result.labels)
                fired += sum(result.windowed_fired)
                token_count += len(result.labels)
            rows.append(
                {
                    "M": int(radius),
                    "T": int(threshold),
                    "deprel_only_accuracy": accuracy(predicted, gold),
                    "windowed_hit_rate": fired / token_count,
                }
            )
    selected = max(
        rows,
        key=lambda row: (
            float(row["deprel_only_accuracy"]),
            int(row["T"]),
            -int(row["M"]),
        ),
    )
    by_m = {
        str(radius): [row for row in rows if int(row["M"]) == radius]
        for radius in radii
    }
    trend = {
        str(radius): {
            "distance_to_unary_nonincreasing": all(
                abs(float(after["deprel_only_accuracy"]) - unary_accuracy)
                <= abs(float(before["deprel_only_accuracy"]) - unary_accuracy) + 1e-15
                for before, after in zip(
                    by_m[str(radius)], by_m[str(radius)][1:], strict=False
                )
            ),
            "accuracy_falls_as_T_rises": any(
                float(after["deprel_only_accuracy"])
                < float(before["deprel_only_accuracy"]) - 1e-15
                for before, after in zip(
                    by_m[str(radius)], by_m[str(radius)][1:], strict=False
                )
            ),
        }
        for radius in radii
    }
    maximum_support = max(
        entry.total
        for table in labeler.window_tables.values()
        for entry in table.entries.values()
    )
    convergence_threshold = maximum_support + 1
    fallback_integrity: dict[str, Any] = {
        "threshold_above_all_window_support": convergence_threshold,
        "maximum_fitted_window_support": maximum_support,
        "by_M": {},
    }
    for radius in radii:
        conservative: list[str] = []
        for sentence in dev:
            conservative.extend(
                labeler.predict(
                    sentence.tokens,
                    predicted_pos[sentence.sent_id],
                    common_heads[sentence.sent_id],
                    radius=radius,
                    min_support=convergence_threshold,
                )
            )
        mismatch_n = sum(
            predicted != unary
            for predicted, unary in zip(conservative, unary_predictions, strict=True)
        )
        assert mismatch_n == 0, (
            f"forced unary fallback integrity failed for M={radius}: "
            f"{mismatch_n} DEV predictions differ"
        )
        fallback_integrity["by_M"][str(radius)] = {
            "matches_unary_exactly": True,
            "mismatch_n": 0,
            "tokens": len(conservative),
        }
    return {
        "grid": {"M": list(radii), "T": list(thresholds)},
        "unary_deprel_only_accuracy": unary_accuracy,
        "curve_by_M": by_m,
        "trend_checks": trend,
        "forced_unary_fallback_integrity": fallback_integrity,
        "selected": {
            "M": int(selected["M"]),
            "T": int(selected["T"]),
            "deprel_only_accuracy": float(selected["deprel_only_accuracy"]),
            "tie_break": "max DEV accuracy, then larger T, then smaller M",
        },
    }


def load_guarded_test_after_selection(
    data_root: Path,
    hasher: Any,
    guard: CampaignTestReadGuard,
    selected: Mapping[str, Any],
) -> tuple[list[Sentence], list[HeadDeprelRecord]]:
    """Read TEST only after a concrete DEV-selected M and T already exist."""
    if "M" not in selected or "T" not in selected:
        raise ValueError("DEV selection must fix M and T before TEST is read")
    test = load_shared_test_once(data_root, hasher, guard)
    test_heads = load_head_test_once(data_root, hasher, guard)
    return test, test_heads


def score_relation_partition(
    predicted_deprel: Sequence[str], gold_deprel: Sequence[str]
) -> dict[str, Any]:
    """Score coarse-deprel matches in the two preregistered gold partitions."""
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
        selected[bucket][0].append(coarse_deprel(predicted))
        selected[bucket][1].append(coarse_gold)
    result: dict[str, Any] = {"excluded_n": excluded}
    for bucket, (predicted, gold) in selected.items():
        if not gold:
            raise ValueError(f"no relations in partition bucket {bucket}")
        result[bucket] = {"accuracy": accuracy(predicted, gold), "n": len(gold)}
    return result


def _score_arm(
    predicted_heads: Sequence[int],
    predicted_deprel: Sequence[str],
    predicted_case: Sequence[str],
    gold_heads: Sequence[int],
    gold_deprel: Sequence[str],
    gold_case: Sequence[str],
) -> dict[str, Any]:
    strict = score_dependencies(
        predicted_heads, predicted_deprel, gold_heads, gold_deprel
    )
    coarse = score_dependencies(
        predicted_heads,
        tuple(coarse_deprel(label) for label in predicted_deprel),
        gold_heads,
        tuple(coarse_deprel(label) for label in gold_deprel),
    )
    case_bearing_accuracy, case_bearing_n = score_case_bearing_deprels(
        predicted_deprel, gold_deprel
    )
    return {
        "deprel_only_accuracy": accuracy(predicted_deprel, gold_deprel),
        "las_strict": strict.las,
        "las_coarse": coarse.las,
        "uas": strict.uas,
        "case_bearing_deprel_accuracy": case_bearing_accuracy,
        "case_bearing_deprel_n": case_bearing_n,
        "serve_honest_case_accuracy": accuracy(predicted_case, gold_case),
        "tokens": strict.tokens,
        "partition": score_relation_partition(predicted_deprel, gold_deprel),
    }


def evaluate_matched_labelers(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    common_heads: Mapping[str, Sequence[int]],
    unary_student: GermanR3DependencyStudent,
    windowed: WindowedDeprelLabeler,
    case_student: GermanR1Student,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score both labelers on identical in-memory L0 heads and case inputs."""
    accumulated: dict[str, dict[str, list[Any]]] = {
        arm: {"head": [], "deprel": [], "case": []}
        for arm in ("unary", "windowed")
    }
    gold_head: list[int] = []
    gold_deprel: list[str] = []
    gold_case: list[str] = []
    fired: list[bool] = []
    fired_windowed_labels: list[str] = []
    fired_unary_labels: list[str] = []
    fired_gold_labels: list[str] = []

    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        offsets = tuple(common_heads[sentence.sent_id])
        unary_labels = predict_deprels(
            unary_student, sentence.tokens, pos, offsets
        )
        window_result = windowed.predict_with_diagnostics(
            sentence.tokens, pos, offsets
        )
        baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        eligible = [
            index
            for index, label in enumerate(sentence.targets["morph_case"])
            if label != "-"
        ]
        for arm, labels in (
            ("unary", unary_labels),
            ("windowed", window_result.labels),
        ):
            case_result = run_predicted_case_sentence(
                case_student,
                sentence.tokens,
                baseline,
                pos,
                offsets,
                labels,
                eligible,
            )
            accumulated[arm]["head"].extend(offsets)
            accumulated[arm]["deprel"].extend(labels)
            accumulated[arm]["case"].extend(case_result.predictions)
        fired.extend(window_result.windowed_fired)
        for index, did_fire in enumerate(window_result.windowed_fired):
            if did_fire:
                fired_windowed_labels.append(window_result.labels[index])
                fired_unary_labels.append(unary_labels[index])
                fired_gold_labels.append(head.deprel[index])
        gold_head.extend(head.head_offset)
        gold_deprel.extend(head.deprel)
        gold_case.extend(sentence.targets["morph_case"])

    metrics = {
        arm: _score_arm(
            accumulated[arm]["head"],
            accumulated[arm]["deprel"],
            accumulated[arm]["case"],
            gold_head,
            gold_deprel,
            gold_case,
        )
        for arm in ("unary", "windowed")
    }
    assert metrics["unary"]["uas"] == metrics["windowed"]["uas"], (
        "matched-labeler UAS invariant failed: "
        f"unary={metrics['unary']['uas']}, windowed={metrics['windowed']['uas']}"
    )
    fired_n = sum(fired)
    diagnostics = {
        "windowed_hit_rate": fired_n / len(fired),
        "windowed_hit_n": fired_n,
        "unary_fallback_rate": (len(fired) - fired_n) / len(fired),
        "unary_fallback_n": len(fired) - fired_n,
        "tokens": len(fired),
        "where_windowed_fires": {
            "n": fired_n,
            "windowed_deprel_only_accuracy": (
                accuracy(fired_windowed_labels, fired_gold_labels) if fired_n else None
            ),
            "unary_deprel_only_accuracy_same_tokens": (
                accuracy(fired_unary_labels, fired_gold_labels) if fired_n else None
            ),
        },
    }
    return metrics, diagnostics


def metric_gains(windowed: Mapping[str, Any], unary: Mapping[str, Any]) -> dict[str, float]:
    """Return the matched windowed-minus-unary gains used in the read."""
    return {
        "deprel_only_accuracy": float(windowed["deprel_only_accuracy"])
        - float(unary["deprel_only_accuracy"]),
        "las_strict": float(windowed["las_strict"]) - float(unary["las_strict"]),
        "las_coarse": float(windowed["las_coarse"]) - float(unary["las_coarse"]),
        "case_bearing_deprel_accuracy": float(
            windowed["case_bearing_deprel_accuracy"]
        )
        - float(unary["case_bearing_deprel_accuracy"]),
        "serve_honest_case_accuracy": float(windowed["serve_honest_case_accuracy"])
        - float(unary["serve_honest_case_accuracy"]),
        "pairwise_sensitive": float(
            windowed["partition"]["pairwise_sensitive"]["accuracy"]
        )
        - float(unary["partition"]["pairwise_sensitive"]["accuracy"]),
        "local": float(windowed["partition"]["local"]["accuracy"])
        - float(unary["partition"]["local"]["accuracy"]),
    }


def apply_preregistered_read(
    unary: Mapping[str, Any],
    windowed: Mapping[str, Any],
    dev_m: int,
    dev_t: int,
) -> dict[str, Any]:
    """Apply signed windowed-labeler preregistration section 5 verbatim."""
    gain = float(windowed["las_strict"]) - float(unary["las_strict"])
    partition_gains = {
        bucket: float(windowed["partition"][bucket]["accuracy"])
        - float(unary["partition"][bucket]["accuracy"])
        for bucket in ("pairwise_sensitive", "local")
    }
    case_unary = float(unary["serve_honest_case_accuracy"])
    case_windowed = float(windowed["serve_honest_case_accuracy"])
    concentrated = partition_gains["pairwise_sensitive"] > partition_gains["local"]
    case_improved = case_windowed > case_unary

    if gain <= 0.0:
        verdict = "HALTED"
        reason = "windowed ≤ unary; report only and do not build rung 2"
    elif gain >= 0.03 and concentrated and case_improved:
        verdict = "FIRES"
        reason = (
            "deprel-LAS gain ≥ +0.03 absolute, pairwise-sensitive gain > local gain, "
            "and the case cascade improves"
        )
    else:
        verdict = "IN-BETWEEN"
        reasons: list[str] = []
        if gain < 0.03:
            reasons.append("gain > 0 but < +0.03")
        if not concentrated:
            reasons.append("diffuse: pairwise gain ≤ local gain")
        if not case_improved:
            reasons.append("no case-cascade improvement")
        reason = "; ".join(reasons)

    if case_windowed >= CASE_DELIVERABLE:
        through_line = (
            f"serve-honest case {case_windowed:.4f} ≥ 0.90: deliverable met"
        )
    elif case_windowed > CASE_BASELINE:
        closed = (case_windowed - CASE_BASELINE) / (CASE_DELIVERABLE - CASE_BASELINE)
        through_line = (
            f"serve-honest case {case_windowed:.4f} < 0.90 but > ~0.76: closes "
            f"{closed:.1%} of the ~0.76→0.90 gap; achievability open (plateau language)"
        )
    else:
        through_line = (
            f"serve-honest case {case_windowed:.4f} ≤ baseline ~0.76: "
            "cascade-corruption / diagnose"
        )
    text = (
        f"{verdict}: windowed−unary serve-honest deprel-LAS gain={gain:+.4f}; "
        f"pairwise-sensitive gain={partition_gains['pairwise_sensitive']:+.4f} versus "
        f"local={partition_gains['local']:+.4f}; case moves from {case_unary:.4f} "
        f"to {case_windowed:.4f}. {reason}. THROUGH-LINE: {through_line}."
    )
    return {
        "verdict": verdict,
        "deprel_las_gain": gain,
        "partition_gains": partition_gains,
        "case_unary": case_unary,
        "case_windowed": case_windowed,
        "dev_M": int(dev_m),
        "dev_T": int(dev_t),
        "text": text,
    }


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit train, tune M/T on DEV, then perform one matched guarded TEST read."""
    hasher = hashlib.sha256()
    print("FIT: loading GSD TRAIN shared tasks and head_deprel.", flush=True)
    train = load_split(DATA_ROOT, "train", hasher)
    train_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "train.jsonl", hasher
    )
    train_heads = head_deprel_by_sent(train_head_records)

    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    r1_student = GermanR1Student(registers)
    r1_student.fit(train)
    train_pos = _predicted_pos_by_sentence(r1_student, train)

    print("DEV: loading DEV only for shipped L0 k and windowed (M,T) selection.", flush=True)
    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    dev_heads = head_deprel_by_sent(dev_head_records)
    l0_window, dev_coverage, coverage_curve = choose_window(dev_head_records)

    unary_student = GermanR3DependencyStudent(l0_window)
    unary_student.fit(train, train_heads, train_pos)
    train_predicted_heads = predict_heads_by_sentence(unary_student, train, train_pos)
    windowed = WindowedDeprelLabeler(unary_student)
    windowed.fit(train, train_heads, train_pos, train_predicted_heads)

    dev_pos = _predicted_pos_by_sentence(r1_student, dev)
    dev_predicted_heads = predict_heads_by_sentence(unary_student, dev, dev_pos)
    dev_sweep = sweep_dev_configurations(
        windowed, dev, dev_heads, dev_pos, dev_predicted_heads
    )
    selected = dev_sweep["selected"]
    windowed.configure(int(selected["M"]), int(selected["T"]))
    print(
        f"DEV SELECTED: M={selected['M']} T={selected['T']} "
        f"deprel-only={selected['deprel_only_accuracy']:.6f}",
        flush=True,
    )
    del dev, dev_heads, dev_head_records, dev_pos, dev_predicted_heads
    del train_predicted_heads, train_head_records, train_heads, train_pos

    guard = CampaignTestReadGuard()
    print("FINAL EVAL: reading TEST shared tasks once and head_deprel once.", flush=True)
    test, test_head_records = load_guarded_test_after_selection(
        DATA_ROOT, hasher, guard, selected
    )
    test_heads = head_deprel_by_sent(test_head_records)
    test_pos = _predicted_pos_by_sentence(r1_student, test)
    common_heads = predict_heads_by_sentence(unary_student, test, test_pos)
    metrics, test_diagnostics = evaluate_matched_labelers(
        test,
        test_heads,
        test_pos,
        common_heads,
        unary_student,
        windowed,
        r1_student,
    )
    guard.assert_complete()
    gains = metric_gains(metrics["windowed"], metrics["unary"])
    read = apply_preregistered_read(
        metrics["unary"],
        metrics["windowed"],
        int(selected["M"]),
        int(selected["T"]),
    )
    scoreboard: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "scope": "feature-ladder rung 1 matched windowed deprel labeler",
        "serve_honest_features": [
            "shipped unary lowercase surface/POS/shape/position keys",
            "shipped predicted head direction/distance/governor POS structure",
            "R1-predicted POS",
            "L0-predicted head offsets",
            "dependent ±M neighbor predicted POS + coarse surface shape",
            "governor ±1 neighbor predicted POS + coarse surface shape",
        ],
        "gold_usage": (
            "Gold TRAIN deprels are count-table targets only; DEV deprels select M/T; TEST "
            "heads/deprels are scoring labels only. No gold heads, deprels, or POS are model "
            "features: deciding keys use R1-predicted POS and L0-predicted heads only."
        ),
        "data_hash_sha256": hasher.hexdigest(),
        "test_read_counts": dict(guard.counts),
        "test_read_once": True,
        "l0_window": {
            "k": l0_window,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "dev_sweep": dev_sweep,
        "selected": {"M": int(selected["M"]), "T": int(selected["T"])},
        "arms": metrics,
        "gains_windowed_minus_unary": gains,
        "diagnostics": {
            **test_diagnostics,
            "dev_accuracy_vs_T": {
                "unary_deprel_only_accuracy": dev_sweep[
                    "unary_deprel_only_accuracy"
                ],
                "curve_by_M": dev_sweep["curve_by_M"],
                "trend_checks": dev_sweep["trend_checks"],
                "forced_unary_fallback_integrity": dev_sweep[
                    "forced_unary_fallback_integrity"
                ],
            },
            "halt_diagnosis": (
                "windowed keys are unhelpful where supported; coverage/backoff protects the "
                "remaining tokens"
                if gains["las_strict"] <= 0.0
                and test_diagnostics["where_windowed_fires"][
                    "windowed_deprel_only_accuracy"
                ]
                <= test_diagnostics["where_windowed_fires"][
                    "unary_deprel_only_accuracy_same_tokens"
                ]
                else "coverage/backoff is the primary limitation"
            ),
        },
        "read": read,
    }
    return scoreboard, guard


def print_scoreboard(
    scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard
) -> None:
    """Print the complete matched scoreboard, diagnostics, and preregistered read."""
    print("\nWINDOWED DEPREL LABELER — GSD TEST")
    print(f"run_tag: {scoreboard['run_tag']}")
    selected = scoreboard["selected"]
    print(f"DEV-selected: M={selected['M']} T={selected['T']}")
    print("serve_honesty: R1-predicted POS + L0-predicted heads; no gold key features")
    print(f"data_hash_sha256: {scoreboard['data_hash_sha256']}")
    guard.assert_complete()
    print(
        "TEST READ ONCE: CONFIRMED "
        "(shared pos/morph_case/morph_gnn test once; head_deprel test once)"
    )
    print(
        "\narm       deprel-only  LAS-strict  LAS-coarse  UAS       "
        "case-bearing  serve-case  pairwise  local     tokens"
    )
    for arm in ("unary", "windowed"):
        metric = scoreboard["arms"][arm]
        print(
            f"{arm:<10} {metric['deprel_only_accuracy']:.6f}     "
            f"{metric['las_strict']:.6f}    {metric['las_coarse']:.6f}    "
            f"{metric['uas']:.6f}  {metric['case_bearing_deprel_accuracy']:.6f} "
            f"({metric['case_bearing_deprel_n']})  "
            f"{metric['serve_honest_case_accuracy']:.6f}    "
            f"{metric['partition']['pairwise_sensitive']['accuracy']:.6f} "
            f"({metric['partition']['pairwise_sensitive']['n']})  "
            f"{metric['partition']['local']['accuracy']:.6f} "
            f"({metric['partition']['local']['n']})  {metric['tokens']}"
        )

    dev = scoreboard["diagnostics"]["dev_accuracy_vs_T"]
    print(f"\nDEV unary deprel-only baseline: {dev['unary_deprel_only_accuracy']:.6f}")
    print("DEV ACCURACY VS T")
    for radius, rows in dev["curve_by_M"].items():
        values = "  ".join(
            f"T={row['T']}:{row['deprel_only_accuracy']:.6f}"
            f"(hit={row['windowed_hit_rate']:.3f})"
            for row in rows
        )
        trend = dev["trend_checks"][radius]
        print(
            f"M={radius}  {values}  "
            f"toward_unary={trend['distance_to_unary_nonincreasing']} "
            f"falls={trend['accuracy_falls_as_T_rises']}"
        )
    integrity = dev["forced_unary_fallback_integrity"]
    print(
        "forced fallback integrity: PASS; T="
        f"{integrity['threshold_above_all_window_support']} > max_support="
        f"{integrity['maximum_fitted_window_support']}; every M exactly matches unary"
    )
    diagnostics = scoreboard["diagnostics"]
    print("\nREQUIRED DIAGNOSTICS")
    print(
        "windowed hit-rate: "
        f"{diagnostics['windowed_hit_rate']:.6f} "
        f"({diagnostics['windowed_hit_n']}/{diagnostics['tokens']}); "
        f"unary fallback-rate={diagnostics['unary_fallback_rate']:.6f} "
        f"({diagnostics['unary_fallback_n']}/{diagnostics['tokens']})"
    )
    where = diagnostics["where_windowed_fires"]
    print(
        "where windowed fires: "
        f"windowed={where['windowed_deprel_only_accuracy']:.6f} "
        f"unary-same-tokens={where['unary_deprel_only_accuracy_same_tokens']:.6f} "
        f"n={where['n']}"
    )
    print(f"HALT diagnosis: {diagnostics['halt_diagnosis']}")
    print("\nTHE READ")
    print(scoreboard["read"]["text"])


def main() -> None:
    scoreboard, guard = build_scoreboard()
    OUTPUT_PATH.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print_scoreboard(scoreboard, guard)
    print(f"structured_output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
