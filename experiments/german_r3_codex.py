"""German R3 proper: deterministic labeled dependencies and serve-honest case.

This empirical campaign is SOFT=0: it fits only deterministic count tables and
uses no gradient learner or LLM.  The dependency student predicts each token
greedily (there is no tree/MST constraint).  Its head backoff chain is:

1. route the token as ROOT, LOCAL (within ``+/-k``), or LONG with progressively
   coarser surface/predicted-POS count tables;
2. predict an exact offset from the corresponding LOCAL or LONG table chain;
3. if that offset leaves the sentence, take the most frequent valid train offset.

Deprel is predicted second from the same surface features plus the student's
own predicted direction, distance bucket, and predicted governor POS.  Gold POS,
head offsets, and relations are never prediction features.  The only selected
hyperparameter is ``k``, chosen on dev by a fixed 90% coverage target.
"""

from __future__ import annotations

import hashlib
import json
import string
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from experiments.german_r3min_codex import (  # noqa: E402
    FullCaseSentenceResult,
    HeadDeprelRecord,
    aligned_head_record,
    full_oracle_case_sentence,
    head_deprel_by_sent,
    parse_head_deprel_record,
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "german_r3_codex.json"
HAIKU_PATH = DATA_ROOT / "teacher" / "ud_served.jsonl"
WINDOW_TARGET_COVERAGE = 0.90
ORACLE_CASE_CEILING = 0.92
R1_CASE_BASELINE_REFERENCE = 0.79
HASH_ORDER = tuple(
    f"{head}/{split}"
    for split in ("train", "dev", "test")
    for head in ("pos", "morph_case", "morph_gnn", "head_deprel")
)


@dataclass(frozen=True)
class DependencyScores:
    uas: float
    las: float
    tokens: int


@dataclass(frozen=True)
class DependencyPrediction:
    head_offset: tuple[int, ...]
    deprel: tuple[str, ...]


@dataclass(frozen=True)
class HaikuToken:
    position: int
    dep: str
    head_position: int


class CampaignTestReadGuard:
    """Hard-fail on a second shared-task or head-deprel test read."""

    def __init__(self) -> None:
        self._counts = {"shared": 0, "head_deprel": 0}

    def claim(self, source: str) -> None:
        if source not in self._counts:
            raise ValueError(f"unknown guarded test source: {source}")
        if self._counts[source]:
            raise RuntimeError(f"GSD {source} test data may be read only once")
        self._counts[source] += 1

    @property
    def counts(self) -> Mapping[str, int]:
        return dict(self._counts)

    def assert_complete(self) -> None:
        if self._counts != {"shared": 1, "head_deprel": 1}:
            raise RuntimeError(f"incomplete test-read claims: {self._counts}")


class BackoffTables:
    """First-present deterministic count-table backoff."""

    def __init__(self, tables: Sequence[CountTable], fallback: str) -> None:
        self.tables = tuple(tables)
        self.fallback = fallback

    def predict(self, keys: Sequence[Any]) -> str:
        if len(keys) != len(self.tables):
            raise ValueError("backoff key/table count mismatch")
        for table, key in zip(self.tables, keys, strict=True):
            entry = table.lookup(key)
            if entry is not None:
                return entry.label
        return self.fallback


def _mode(labels: Iterable[str]) -> str:
    counts = Counter(labels)
    if not counts:
        raise ValueError("cannot take mode of empty labels")
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _fit_backoff(
    rows: Sequence[tuple[tuple[Any, ...], str]],
    fallback: str | None = None,
) -> BackoffTables:
    if not rows:
        raise ValueError("cannot fit empty backoff tables")
    width = len(rows[0][0])
    if any(len(keys) != width for keys, _label in rows):
        raise ValueError("inconsistent backoff key width")
    tables = [
        fit_count_table(
            ((keys[level], label) for keys, label in rows),
            minsupp=1,
            mindet=0.0,
        )
        for level in range(width)
    ]
    return BackoffTables(tables, fallback or _mode(label for _keys, label in rows))


def load_head_deprel_file(path: Path, hasher: Any | None = None) -> list[HeadDeprelRecord]:
    """Load and validate one head/deprel JSONL file, hashing every raw byte."""
    raw = path.read_bytes()
    if hasher is not None:
        hasher.update(raw)
    return [parse_head_deprel_record(line) for line in raw.splitlines() if line.strip()]


def load_head_test_once(
    data_root: Path,
    hasher: Any,
    guard: CampaignTestReadGuard,
) -> list[HeadDeprelRecord]:
    guard.claim("head_deprel")
    return load_head_deprel_file(data_root / "tasks" / "head_deprel" / "test.jsonl", hasher)


def load_shared_test_once(
    data_root: Path,
    hasher: Any,
    guard: CampaignTestReadGuard,
) -> list[Sentence]:
    guard.claim("shared")
    return load_split(data_root, "test", hasher)


def resolve_governor_index(index: int, offset: int, token_count: int) -> int | None:
    """Resolve ``index + offset``; roots remain self-loops and invalid heads are None."""
    if index < 0 or index >= token_count:
        raise IndexError(f"token index outside sentence: {index}")
    governor = index + int(offset)
    return governor if 0 <= governor < token_count else None


def score_dependencies(
    predicted_head_offset: Sequence[int],
    predicted_deprel: Sequence[str],
    gold_head_offset: Sequence[int],
    gold_deprel: Sequence[str],
) -> DependencyScores:
    """Score exact governor UAS and joint exact-governor/exact-label LAS."""
    size = len(gold_head_offset)
    if not (
        len(predicted_head_offset)
        == len(predicted_deprel)
        == size
        == len(gold_deprel)
    ):
        raise ValueError("dependency prediction/gold lengths differ")
    if not size:
        raise ValueError("cannot score an empty dependency sequence")
    uas_correct = las_correct = 0
    for index in range(size):
        # For the same dependent index, equal relative offsets are exactly
        # equivalent to equal governor indices.  Comparing offsets also lets
        # callers concatenate sentences without inventing cross-boundary heads.
        head_correct = predicted_head_offset[index] == gold_head_offset[index]
        uas_correct += head_correct
        las_correct += head_correct and predicted_deprel[index] == gold_deprel[index]
    return DependencyScores(uas_correct / size, las_correct / size, size)


def choose_window(head_records: Sequence[HeadDeprelRecord]) -> tuple[int, float, list[dict[str, float]]]:
    """Choose the smallest k whose dev absolute-offset coverage reaches 90%."""
    offsets = [abs(offset) for record in head_records for offset in record.head_offset]
    if not offsets:
        raise ValueError("cannot choose a window from empty dev data")
    curve: list[dict[str, float]] = []
    chosen = max(offsets)
    chosen_coverage = 1.0
    for k in range(1, max(offsets) + 1):
        coverage = sum(offset <= k for offset in offsets) / len(offsets)
        curve.append({"k": k, "coverage": coverage})
        if coverage >= WINDOW_TARGET_COVERAGE:
            chosen = k
            chosen_coverage = coverage
            break
    return chosen, chosen_coverage, curve


def _punctuation(token: str) -> bool:
    return bool(token) and all(character in string.punctuation for character in token)


def _position_bucket(index: int, size: int) -> int:
    return min(4, (5 * index) // max(size, 1))


def _surface_key_levels(
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    index: int,
) -> tuple[Any, ...]:
    if len(tokens) != len(predicted_pos):
        raise ValueError("tokens and predicted POS must align")
    token = tokens[index]
    form = token.lower()
    previous_form = tokens[index - 1].lower() if index else "<BOS>"
    next_form = tokens[index + 1].lower() if index + 1 < len(tokens) else "<EOS>"
    previous_pos = predicted_pos[index - 1] if index else "<BOS>"
    next_pos = predicted_pos[index + 1] if index + 1 < len(tokens) else "<EOS>"
    position = _position_bucket(index, len(tokens))
    shape = (token.istitle(), token.isupper(), _punctuation(token))
    return (
        (
            form,
            predicted_pos[index],
            position,
            previous_form,
            previous_pos,
            next_form,
            next_pos,
            shape,
        ),
        (form, predicted_pos[index], position, previous_pos, next_pos, shape),
        (predicted_pos[index], position, previous_pos, next_pos, shape),
        (predicted_pos[index], previous_pos, next_pos),
        predicted_pos[index],
    )


def _relation_key_levels(
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    index: int,
    predicted_offset: int,
) -> tuple[Any, ...]:
    surface = _surface_key_levels(tokens, predicted_pos, index)
    governor = resolve_governor_index(index, predicted_offset, len(tokens))
    if predicted_offset == 0:
        direction = "SELF"
    elif predicted_offset < 0:
        direction = "LEFT"
    else:
        direction = "RIGHT"
    distance = abs(predicted_offset)
    distance_bucket: int | str
    if distance <= 2:
        distance_bucket = distance
    elif distance <= 5:
        distance_bucket = "3-5"
    else:
        distance_bucket = "6+"
    governor_pos = predicted_pos[governor] if governor is not None else "<INVALID>"
    structure = (direction, distance_bucket, governor_pos)
    return (
        (surface[0], structure),
        (surface[1], structure),
        (surface[2], structure),
        (predicted_pos[index], surface[3], structure),
        (predicted_pos[index], structure),
        (predicted_pos[index], direction),
    )


class GermanR3DependencyStudent:
    """SOFT=0 greedy labeled-dependency student using predicted POS only."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("dependency window must be positive")
        self.window = window
        self.route_table: BackoffTables | None = None
        self.local_offset_table: BackoffTables | None = None
        self.long_offset_table: BackoffTables | None = None
        self.deprel_table: BackoffTables | None = None
        self._local_offset_order: tuple[int, ...] = ()
        self._long_offset_order: tuple[int, ...] = ()

    def fit(
        self,
        train: Sequence[Sentence],
        heads: Mapping[str, HeadDeprelRecord],
        predicted_pos: Mapping[str, Sequence[str]],
    ) -> None:
        route_rows: list[tuple[tuple[Any, ...], str]] = []
        local_rows: list[tuple[tuple[Any, ...], str]] = []
        long_rows: list[tuple[tuple[Any, ...], str]] = []
        local_counts: Counter[int] = Counter()
        long_counts: Counter[int] = Counter()
        for sentence in train:
            head = aligned_head_record(sentence, heads)
            if not isinstance(head, HeadDeprelRecord):
                raise TypeError("labeled dependency record required")
            pos = predicted_pos[sentence.sent_id]
            for index, offset in enumerate(head.head_offset):
                keys = _surface_key_levels(sentence.tokens, pos, index)
                route = "ROOT" if offset == 0 else ("LOCAL" if abs(offset) <= self.window else "LONG")
                route_rows.append((keys, route))
                if route == "LONG":
                    long_rows.append((keys, str(offset)))
                    long_counts[offset] += 1
                elif route == "LOCAL":
                    local_rows.append((keys, str(offset)))
                    local_counts[offset] += 1
        self.route_table = _fit_backoff(route_rows)
        self.local_offset_table = _fit_backoff(local_rows)
        self.long_offset_table = _fit_backoff(long_rows)
        self._local_offset_order = tuple(
            offset for offset, _count in sorted(local_counts.items(), key=lambda item: (-item[1], item[0]))
        )
        self._long_offset_order = tuple(
            offset for offset, _count in sorted(long_counts.items(), key=lambda item: (-item[1], item[0]))
        )

        # Deprel sees the student's own fitted head predictions, never gold heads.
        relation_rows: list[tuple[tuple[Any, ...], str]] = []
        for sentence in train:
            head = aligned_head_record(sentence, heads)
            if not isinstance(head, HeadDeprelRecord):
                raise TypeError("labeled dependency record required")
            pos = predicted_pos[sentence.sent_id]
            predicted_offsets = self.predict_heads(sentence.tokens, pos)
            for index, label in enumerate(head.deprel):
                keys = _relation_key_levels(sentence.tokens, pos, index, predicted_offsets[index])
                relation_rows.append((keys, label))
        self.deprel_table = _fit_backoff(relation_rows)

    @staticmethod
    def _require(table: BackoffTables | None, name: str) -> BackoffTables:
        if table is None:
            raise RuntimeError(f"{name} used before fit")
        return table

    @staticmethod
    def _valid_offset(index: int, offset: int, size: int) -> bool:
        return resolve_governor_index(index, offset, size) is not None

    def _valid_train_fallback(
        self,
        index: int,
        size: int,
        offsets: Sequence[int],
    ) -> int:
        for offset in offsets:
            if self._valid_offset(index, offset, size):
                return offset
        return 0

    def predict_heads(self, tokens: Sequence[str], predicted_pos: Sequence[str]) -> tuple[int, ...]:
        route_table = self._require(self.route_table, "route table")
        local_table = self._require(self.local_offset_table, "local offset table")
        long_table = self._require(self.long_offset_table, "long offset table")
        predictions: list[int] = []
        for index in range(len(tokens)):
            keys = _surface_key_levels(tokens, predicted_pos, index)
            route = route_table.predict(keys)
            if route == "ROOT":
                predictions.append(0)
                continue
            if route == "LONG":
                offset = int(long_table.predict(keys))
                order = self._long_offset_order
            else:
                offset = int(local_table.predict(keys))
                order = self._local_offset_order
            if not self._valid_offset(index, offset, len(tokens)):
                offset = self._valid_train_fallback(index, len(tokens), order)
            predictions.append(offset)
        return tuple(predictions)

    def predict_sentence(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
    ) -> DependencyPrediction:
        offsets = self.predict_heads(tokens, predicted_pos)
        relation_table = self._require(self.deprel_table, "deprel table")
        labels = tuple(
            relation_table.predict(_relation_key_levels(tokens, predicted_pos, index, offset))
            for index, offset in enumerate(offsets)
        )
        return DependencyPrediction(offsets, labels)


class MajorityOffsetBaseline:
    """Predicted-POS majority offset (global backoff) plus global majority deprel."""

    def __init__(self) -> None:
        self.offset_by_pos: dict[str, int] = {}
        self.global_offset = 0
        self.majority_deprel = "root"

    @staticmethod
    def _int_mode(counts: Counter[int]) -> int:
        if not counts:
            raise ValueError("cannot fit an empty offset mode")
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def fit(
        self,
        train: Sequence[Sentence],
        heads: Mapping[str, HeadDeprelRecord],
        predicted_pos: Mapping[str, Sequence[str]],
    ) -> None:
        by_pos: dict[str, Counter[int]] = defaultdict(Counter)
        global_offsets: Counter[int] = Counter()
        relations: Counter[str] = Counter()
        for sentence in train:
            head = aligned_head_record(sentence, heads)
            if not isinstance(head, HeadDeprelRecord):
                raise TypeError("labeled dependency record required")
            pos = predicted_pos[sentence.sent_id]
            for predicted_tag, offset, relation in zip(
                pos, head.head_offset, head.deprel, strict=True
            ):
                by_pos[predicted_tag][offset] += 1
                global_offsets[offset] += 1
                relations[relation] += 1
        self.offset_by_pos = {tag: self._int_mode(counts) for tag, counts in by_pos.items()}
        self.global_offset = self._int_mode(global_offsets)
        self.majority_deprel = _mode(relations.elements())

    def predict(self, predicted_pos: Sequence[str]) -> DependencyPrediction:
        offsets = tuple(self.offset_by_pos.get(tag, self.global_offset) for tag in predicted_pos)
        return DependencyPrediction(offsets, (self.majority_deprel,) * len(predicted_pos))


def flatten_haiku_tree(tree: Mapping[str, Any]) -> tuple[HaikuToken, ...]:
    """Flatten a nested Haiku tree to sorted 1-indexed dependency records."""
    flattened: list[HaikuToken] = []

    def visit(node: Mapping[str, Any], parent_position: int | None) -> None:
        position = int(node["position"])
        head_position = position if parent_position is None else parent_position
        flattened.append(HaikuToken(position, str(node["dep"]), head_position))
        children = node.get("children") or []
        if not isinstance(children, list):
            raise ValueError("Haiku children must be a list")
        for child in children:
            if not isinstance(child, Mapping):
                raise ValueError("Haiku child must be an object")
            visit(child, position)

    visit(tree, None)
    flattened.sort(key=lambda token: token.position)
    positions = [token.position for token in flattened]
    if len(set(positions)) != len(positions):
        raise ValueError("duplicate Haiku token positions")
    return tuple(flattened)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def load_haiku_candidates(path: Path) -> tuple[dict[str, list[tuple[HaikuToken, ...]]], int]:
    """Read valid parsed trees and index them by normalized sentence text."""
    candidates: dict[str, list[tuple[HaikuToken, ...]]] = defaultdict(list)
    raw_records = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw_records += 1
            obj = json.loads(line)
            parsed = obj.get("parsed")
            if not isinstance(parsed, Mapping):
                continue
            tree = parsed.get("tree")
            if not isinstance(tree, Mapping):
                continue
            try:
                flattened = flatten_haiku_tree(tree)
            except (KeyError, TypeError, ValueError):
                continue
            candidates[_normalize_whitespace(str(obj.get("sentence", "")))].append(flattened)
    return dict(candidates), raw_records


def score_haiku(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    path: Path = HAIKU_PATH,
) -> dict[str, Any]:
    """Score one equal-token-count teacher tree per whitespace-normalized GSD text."""
    candidates, raw_records = load_haiku_candidates(path)
    predicted_offsets: list[int] = []
    predicted_relations: list[str] = []
    gold_offsets: list[int] = []
    gold_relations: list[str] = []
    text_matches = token_mismatch_excluded = overlap_sentences = 0
    for sentence in test:
        sentence_candidates = candidates.get(_normalize_whitespace(sentence.text), [])
        if not sentence_candidates:
            continue
        text_matches += 1
        flattened = next(
            (candidate for candidate in sentence_candidates if len(candidate) == len(sentence.tokens)),
            None,
        )
        if flattened is None:
            token_mismatch_excluded += 1
            continue
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        overlap_sentences += 1
        predicted_offsets.extend(token.head_position - token.position for token in flattened)
        predicted_relations.extend(token.dep for token in flattened)
        gold_offsets.extend(head.head_offset)
        gold_relations.extend(head.deprel)
    if not gold_offsets:
        return {
            "uas": 0.0,
            "las": 0.0,
            "overlap_sentences": 0,
            "test_sentences": len(test),
            "overlap_sentence_fraction": 0.0,
            "overlap_tokens": 0,
            "raw_records": raw_records,
            "text_matches": text_matches,
            "token_mismatch_excluded": token_mismatch_excluded,
        }
    scores = score_dependencies(
        predicted_offsets,
        predicted_relations,
        gold_offsets,
        gold_relations,
    )
    return {
        "uas": scores.uas,
        "las": scores.las,
        "overlap_sentences": overlap_sentences,
        "test_sentences": len(test),
        "overlap_sentence_fraction": overlap_sentences / len(test),
        "overlap_tokens": scores.tokens,
        "raw_records": raw_records,
        "text_matches": text_matches,
        "token_mismatch_excluded": token_mismatch_excluded,
    }


def run_predicted_case_sentence(
    student: GermanR1Student,
    tokens: Sequence[str],
    baseline: Sequence[str],
    predicted_pos: Sequence[str],
    predicted_head_offset: Sequence[int],
    predicted_deprel: Sequence[str],
    eligible_indices: Sequence[int],
) -> FullCaseSentenceResult:
    """Call R3-min's cascade with predicted values in all three formerly-gold slots."""
    return full_oracle_case_sentence(
        student,
        tokens,
        baseline,
        predicted_pos,
        predicted_pos,  # closes rule 5's shipped governor-UPOS oracle gap
        predicted_head_offset,
        predicted_deprel,
        eligible_indices,
    )


def _predicted_pos_by_sentence(
    student: GermanR1Student,
    sentences: Sequence[Sentence],
) -> dict[str, tuple[str, ...]]:
    return {
        sentence.sent_id: tuple(student.predict_sentence(sentence)["full"]["pos"])
        for sentence in sentences
    }


def _evaluate_dependency_models(
    dependency_student: GermanR3DependencyStudent,
    baseline: MajorityOffsetBaseline,
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any], dict[str, DependencyPrediction]]:
    student_head: list[int] = []
    student_rel: list[str] = []
    baseline_head: list[int] = []
    baseline_rel: list[str] = []
    gold_head: list[int] = []
    gold_rel: list[str] = []
    sentence_predictions: dict[str, DependencyPrediction] = {}
    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        student_prediction = dependency_student.predict_sentence(sentence.tokens, pos)
        baseline_prediction = baseline.predict(pos)
        sentence_predictions[sentence.sent_id] = student_prediction
        student_head.extend(student_prediction.head_offset)
        student_rel.extend(student_prediction.deprel)
        baseline_head.extend(baseline_prediction.head_offset)
        baseline_rel.extend(baseline_prediction.deprel)
        gold_head.extend(head.head_offset)
        gold_rel.extend(head.deprel)
    student_scores = score_dependencies(student_head, student_rel, gold_head, gold_rel)
    baseline_scores = score_dependencies(baseline_head, baseline_rel, gold_head, gold_rel)
    return (
        {
            "student": {
                "uas": student_scores.uas,
                "las": student_scores.las,
                "tokens": student_scores.tokens,
            },
            "majority_offset_baseline": {
                "uas": baseline_scores.uas,
                "las": baseline_scores.las,
                "tokens": baseline_scores.tokens,
                "majority_deprel": baseline.majority_deprel,
            },
        },
        sentence_predictions,
    )


def _evaluate_predicted_case(
    r1_student: GermanR1Student,
    test: Sequence[Sentence],
    predicted_pos: Mapping[str, Sequence[str]],
    dependencies: Mapping[str, DependencyPrediction],
) -> dict[str, Any]:
    predictions: list[str] = []
    baseline_predictions: list[str] = []
    gold: list[str] = []
    rule_counts = Counter[int]()
    rule5_attempts = rule5_hits = rule5_correct_hits = case_bearing = 0
    for sentence in test:
        r1_result = r1_student.predict_sentence(sentence)
        baseline = r1_result["full"]["morph_case"]
        pos = predicted_pos[sentence.sent_id]
        dependency = dependencies[sentence.sent_id]
        eligible = [
            index
            for index, label in enumerate(sentence.targets["morph_case"])
            if label != "-"
        ]
        result = run_predicted_case_sentence(
            r1_student,
            sentence.tokens,
            baseline,
            pos,
            dependency.head_offset,
            dependency.deprel,
            eligible,
        )
        case_bearing += len(eligible)
        rule5_attempts += result.rule5_attempts
        rule5_hits += result.rule5_hits
        for index in eligible:
            rule = result.fired_rules[index]
            if rule is None:
                raise RuntimeError(f"{sentence.sent_id}:{index}: predicted case rule missing")
            rule_counts[rule] += 1
            if rule == 5:
                rule5_correct_hits += result.predictions[index] == sentence.targets["morph_case"][index]
        predictions.extend(result.predictions)
        baseline_predictions.extend(baseline)
        gold.extend(sentence.targets["morph_case"])
    predicted_accuracy = accuracy(predictions, gold)
    baseline_accuracy = accuracy(baseline_predictions, gold)
    return {
        "accuracy": predicted_accuracy,
        "r1_baseline_accuracy_same_run": baseline_accuracy,
        "delta_vs_r1_same_run": predicted_accuracy - baseline_accuracy,
        "oracle_ceiling_reference": ORACLE_CASE_CEILING,
        "gap_to_oracle_ceiling": predicted_accuracy - ORACLE_CASE_CEILING,
        "r1_baseline_reference": R1_CASE_BASELINE_REFERENCE,
        "delta_vs_r1_reference": predicted_accuracy - R1_CASE_BASELINE_REFERENCE,
        "case_bearing_n": case_bearing,
        "rule_counts": {str(rule): rule_counts[rule] for rule in range(1, 10)},
        "rule5": {
            "attempts": rule5_attempts,
            "hits": rule5_hits,
            "empty": rule5_attempts - rule5_hits,
            "hit_rate": rule5_hits / rule5_attempts if rule5_attempts else 0.0,
            "correct_hits": rule5_correct_hits,
            "precision": rule5_correct_hits / rule5_hits if rule5_hits else 0.0,
        },
        "gold_fields_supplied": [],
        "predicted_fields_supplied": ["head_offset", "deprel", "governor_upos"],
        "verb_government_gap_closed": True,
    }


def _r4_implication(case_accuracy: float) -> str:
    if case_accuracy >= ORACLE_CASE_CEILING:
        return (
            "R4 implication: the serve-honest cascade reaches the #116 oracle reference; "
            "R4 can focus on residual validation rather than parser recovery."
        )
    return (
        "R4 implication: the serve-honest cascade remains below the #116 oracle ceiling; "
        "R4 should target labeled-attachment errors and verb_government precision as residuals."
    )


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit on train, select k on dev, then perform one guarded final test read."""
    hasher = hashlib.sha256()
    print("Loading GSD train in hash order: pos, morph_case, morph_gnn, head_deprel.", flush=True)
    train = load_split(DATA_ROOT, "train", hasher)
    train_heads_list = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "train.jsonl", hasher
    )
    train_heads = head_deprel_by_sent(train_heads_list)

    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    r1_student = GermanR1Student(registers)
    r1_student.fit(train)
    train_pos = _predicted_pos_by_sentence(r1_student, train)

    print("Loading GSD dev only to choose dependency window k.", flush=True)
    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_heads = load_head_deprel_file(DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher)
    k, dev_coverage, coverage_curve = choose_window(dev_heads)
    print(f"DEV WINDOW: k={k} coverage={dev_coverage:.6f} target={WINDOW_TARGET_COVERAGE:.2f}", flush=True)
    del dev

    dependency_student = GermanR3DependencyStudent(k)
    dependency_student.fit(train, train_heads, train_pos)
    majority_baseline = MajorityOffsetBaseline()
    majority_baseline.fit(train, train_heads, train_pos)
    del train_heads_list, train_heads, train_pos

    guard = CampaignTestReadGuard()
    print("FINAL EVAL: reading each GSD test task file once.", flush=True)
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_heads_list = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads = head_deprel_by_sent(test_heads_list)
    test_pos = _predicted_pos_by_sentence(r1_student, test)

    dependency_metrics, dependency_predictions = _evaluate_dependency_models(
        dependency_student,
        majority_baseline,
        test,
        test_heads,
        test_pos,
    )
    haiku = score_haiku(test, test_heads)
    within_five = dependency_metrics["student"]["las"] >= haiku["las"] - 0.05
    dependency_metrics["haiku"] = haiku
    dependency_metrics["within_5_points_of_haiku_las"] = within_five
    dependency_metrics["verdict"] = "FIRES" if within_five else "miss"

    case = _evaluate_predicted_case(
        r1_student,
        test,
        test_pos,
        dependency_predictions,
    )
    implication = _r4_implication(case["accuracy"])
    scoreboard: dict[str, Any] = {
        "run_tag": "empirical",
        "soft": 0,
        "scope_limit": "greedy per-token labeled dependencies; no tree/MST constraint",
        "supervision": "GSD train gold only; k selected on GSD dev only",
        "hash_order": list(HASH_ORDER),
        "data_hash_sha256": hasher.hexdigest(),
        "test_read_counts": dict(guard.counts),
        "test_read_once": True,
        "window": {
            "k": k,
            "target_coverage": WINDOW_TARGET_COVERAGE,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "labeled_dependencies": dependency_metrics,
        "morph_case_under_predicted_dependencies": case,
        "r4_implication": implication,
        "haiku_matching_rule": (
            "collapse whitespace and strip; match whole sentence text; score only equal flattened/GSD "
            "token counts; compare by 1-indexed position"
        ),
    }
    guard.assert_complete()
    return scoreboard, guard


def print_scoreboard(scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard) -> None:
    dependencies = scoreboard["labeled_dependencies"]
    student = dependencies["student"]
    baseline = dependencies["majority_offset_baseline"]
    haiku = dependencies["haiku"]
    case = scoreboard["morph_case_under_predicted_dependencies"]
    rule5 = case["rule5"]

    print("\nGERMAN R3 PROPER CODEX FINAL SCOREBOARD")
    print("run_tag: empirical")
    print("SOFT=0; supervision: GSD train gold ONLY; k selected on GSD dev ONLY")
    print(f"scope_limit: {scoreboard['scope_limit']}")
    print("hash_order: " + ", ".join(scoreboard["hash_order"]))
    print(f"data_hash_sha256: {scoreboard['data_hash_sha256']}")
    guard.assert_complete()
    print(
        "TEST READ ONCE: CONFIRMED "
        "(shared pos/morph_case/morph_gnn test load once; head_deprel test load once)"
    )
    window = scoreboard["window"]
    print(
        f"dev window: k={window['k']} coverage={window['dev_coverage']:.6f} "
        f"(smallest k reaching target {window['target_coverage']:.2f})"
    )

    print("\nLABELED DEPENDENCIES — GSD test")
    print(
        f"student:                    UAS={student['uas']:.6f} "
        f"LAS={student['las']:.6f} n={student['tokens']}"
    )
    print(
        "majority-offset baseline:    "
        f"UAS={baseline['uas']:.6f} LAS={baseline['las']:.6f} n={baseline['tokens']}"
    )
    print(
        f"Haiku overlap only:          UAS={haiku['uas']:.6f} LAS={haiku['las']:.6f} "
        f"sentences={haiku['overlap_sentences']} tokens={haiku['overlap_tokens']}"
    )
    print(
        "Haiku matching: whole-sentence text after whitespace collapse/strip; equal flattened "
        f"token count only. text_matches={haiku['text_matches']} "
        f"token_mismatch_excluded={haiku['token_mismatch_excluded']} raw_records={haiku['raw_records']}"
    )
    if haiku["overlap_sentence_fraction"] < 0.5:
        print(
            "Haiku overlap is small/limited relative to the full GSD test: "
            f"{haiku['overlap_sentences']}/{haiku['test_sentences']} sentences "
            f"({haiku['overlap_sentence_fraction']:.6f}); interpret overlap-only scores cautiously."
        )
    print(
        "student LAS within 5 points of Haiku's LAS? "
        f"{dependencies['verdict']} "
        f"(student={student['las']:.6f}, Haiku={haiku['las']:.6f})"
    )

    print("\nMORPH_CASE — R3-min cascade under predicted dependencies/POS")
    print(f"serve-honest morph_case accuracy:   {case['accuracy']:.6f}")
    print(
        f"#116 oracle ceiling reference:      {case['oracle_ceiling_reference']:.6f} "
        f"(delta={case['gap_to_oracle_ceiling']:+.6f})"
    )
    print(
        f"R1 parse-free baseline same run:    {case['r1_baseline_accuracy_same_run']:.6f} "
        f"(delta={case['delta_vs_r1_same_run']:+.6f}; prereg reference ~0.79)"
    )
    print(
        "serve_honesty: predicted head_offset + predicted deprel + predicted governor UPOS; "
        "no gold cascade fields supplied. This closes the shipped rule-5 governor-UPOS residual gap."
    )
    print(
        "rule-5 verb_government under prediction: "
        f"attempts={rule5['attempts']} hits={rule5['hits']} empty={rule5['empty']} "
        f"hit_rate={rule5['hit_rate']:.6f} precision={rule5['precision']:.6f} "
        f"({rule5['correct_hits']}/{rule5['hits']})"
    )
    print(scoreboard["r4_implication"])


def main() -> None:
    scoreboard, guard = build_scoreboard()
    OUTPUT_PATH.write_text(json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_scoreboard(scoreboard, guard)
    print(f"structured_output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
