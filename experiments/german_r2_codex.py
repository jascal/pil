"""German R2: inspectable, dev-judged ambiguity rules over indexed tokens only.

Serve-honesty contract
----------------------
The student uses only the target token's lower-cased form; original-case and
punctuation of surface neighbours; literal neighbour forms; neighbour affix
shapes; and a train-only surface-form POS predictor.  The latter is fit from
GSD POS *train* and never consumes a gold neighbour annotation at prediction
time.  No teacher path is accessed.  The DET/PRON and AUX/VERB form priors,
context trees, confidences, and rule evidence are fit from their GSD train
splits.  Dev only judges admission.  Test is loaded once per task after fit.

This is the small-data, pure-Python analogue of ``best_per_key`` plus
``core_cover_sw``: count tables provide Laplace confidence and the proposal
with the highest confidence wins, with deterministic ties.  SOFT=0.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_ROOT = Path("/home/allans/code/germandata")
TASKS = ("det_pron", "aux_verb")
LABELS = {"det_pron": ("DET", "PRON"), "aux_verb": ("AUX", "VERB")}
ALPHA = 2.0
FORM_MINSUPP = 1
FORM_MINDET = 0.0

# Fixed before test: a conditional rule is judged as one inspectable unit.
DEV_MIN_SUPPORT = 5
DEV_MIN_ACCURACY = 0.70
DEV_MIN_MARGIN = 0.01
TREE_MAX_DEPTH = 6
TREE_MIN_LEAF = 8
TREE_MIN_GAIN = 0.001
BAR = 0.97

HARD_BOUNDARIES = frozenset({".", "!", "?", ";", ":", "--", "–", "—", "``", "''"})
CLAUSE_BOUNDARIES = HARD_BOUNDARIES | frozenset(
    {
        ",",
        "aber",
        "oder",
        "denn",
        "sondern",
        "doch",
        "weil",
        "dass",
        "daß",
        "wenn",
        "falls",
        "ob",
        "als",
        "bis",
        "damit",
        "wo",
    }
)
LOCATIVE_FORMS = frozenset(
    {
        "in",
        "an",
        "auf",
        "bei",
        "unter",
        "über",
        "vor",
        "hinter",
        "zwischen",
        "aus",
        "von",
        "zu",
        "gegen",
        "für",
        "mit",
        "nach",
        "hier",
        "da",
        "dort",
        "dabei",
        "dagegen",
        "dafür",
        "drin",
        "weg",
        "zurück",
    }
)
HABEN_FORMS = frozenset(
    {"hat", "habe", "haben", "hast", "hab", "hatte", "hatten", "hätte", "hätten", "gehabt"}
)
SEIN_FORMS = frozenset(
    {"ist", "sind", "war", "waren", "bin", "bist", "sei", "seien", "wäre", "wären", "sein", "gewesen"}
)
WERDEN_FORMS = frozenset(
    {"wird", "werden", "wurde", "wurden", "werde", "würde", "würden", "worden", "geworden"}
)


@dataclass(frozen=True)
class TargetRecord:
    sent_id: str
    text: str
    tokens: tuple[str, ...]
    indices: tuple[int, ...]
    labels: tuple[str, ...]

    def targets(self) -> Iterable[tuple[int, str, str]]:
        for index, label in zip(self.indices, self.labels, strict=True):
            yield index, self.tokens[index], label


@dataclass(frozen=True)
class TableEntry:
    label: str
    confidence: float
    count: int
    total: int


@dataclass(frozen=True)
class Candidate:
    label: str
    confidence: float
    tier: str


@dataclass(frozen=True)
class RuleEvidence:
    name: str
    description: str
    predicted_label: str
    dev_support: int
    dev_accuracy: float
    dev_memorizer_accuracy: float
    margin: float
    admitted: bool


@dataclass(frozen=True)
class CatalogEvidence:
    status: str
    support: int
    branch_accuracy: float
    naive_accuracy: float
    full_rule_accuracy: float


@dataclass(frozen=True)
class TaskScores:
    full: float
    memorizer: float
    total: int


class CountTable:
    def __init__(self, entries: Mapping[Any, TableEntry] | None = None) -> None:
        self.entries = dict(entries or {})

    def lookup(self, key: Any) -> TableEntry | None:
        return self.entries.get(key)


class TestReadGuard:
    """Claim each registered task test path once and reject a second attempt."""

    def __init__(self) -> None:
        self.claimed: set[str] = set()

    def claim(self, task: str) -> None:
        if task not in TASKS:
            raise ValueError(f"unknown task: {task}")
        if task in self.claimed:
            raise RuntimeError(f"GSD {task} test split may be read only once")
        self.claimed.add(task)

    @property
    def confirmed(self) -> bool:
        return self.claimed == set(TASKS)


def parse_record(raw: str | bytes | Mapping[str, Any]) -> TargetRecord:
    """Parse one sparse-target JSONL record and validate index/label alignment."""
    obj = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    tokens = tuple(str(token) for token in obj["tokens"])
    indices = tuple(int(index) for index in obj["targets"]["indices"])
    labels = tuple(str(label) for label in obj["targets"]["label"])
    if len(indices) != len(labels):
        raise ValueError("target indices and labels have different lengths")
    if any(index < 0 or index >= len(tokens) for index in indices):
        raise ValueError("target index is outside token sequence")
    return TargetRecord(str(obj["sent_id"]), str(obj["text"]), tokens, indices, labels)


def load_records(path: Path, hasher: Any | None = None) -> list[TargetRecord]:
    raw = path.read_bytes()
    if hasher is not None:
        hasher.update(raw)
    return [parse_record(line) for line in raw.splitlines() if line.strip()]


def load_task_split(
    data_root: Path,
    task: str,
    split: str,
    hasher: Any | None = None,
    guard: TestReadGuard | None = None,
) -> list[TargetRecord]:
    if split == "test":
        if guard is None:
            raise RuntimeError("test loading requires a TestReadGuard")
        guard.claim(task)
    return load_records(data_root / "tasks" / task / f"{split}.jsonl", hasher)


def fit_count_table(
    examples: Iterable[tuple[Any, str]],
    minsupp: int = FORM_MINSUPP,
    mindet: float = FORM_MINDET,
    alpha: float = ALPHA,
) -> CountTable:
    counts: dict[Any, Counter[str]] = defaultdict(Counter)
    for key, label in examples:
        counts[key][label] += 1
    entries: dict[Any, TableEntry] = {}
    for key, by_label in counts.items():
        total = sum(by_label.values())
        label, count = max(by_label.items(), key=lambda item: (item[1], item[0]))
        if count >= minsupp and count / total >= mindet:
            entries[key] = TableEntry(label, count / (total + alpha), count, total)
    return CountTable(entries)


def majority_label(labels: Iterable[str]) -> str:
    counts = Counter(labels)
    if not counts:
        raise ValueError("cannot take majority of no labels")
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def choose_candidate(candidates: Iterable[Candidate], fallback: str) -> str:
    tier_rank = {"form": 0, "rule": 1}
    proposals = list(candidates)
    if not proposals:
        return fallback
    return max(
        proposals,
        key=lambda proposal: (
            proposal.confidence,
            tier_rank[proposal.tier],
            proposal.label,
        ),
    ).label


class SurfacePosPredictor:
    """Train-only surface-form POS pass; no gold tag is accepted by predict()."""

    def __init__(self) -> None:
        self.forms = CountTable()
        self.suffixes: dict[tuple[int, str], TableEntry] = {}
        self.fallback = "NOUN"

    def fit(self, records: Sequence[Mapping[str, Any]]) -> None:
        examples: list[tuple[str, str]] = []
        suffix_counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
        labels: list[str] = []
        for obj in records:
            tokens = obj["tokens"]
            upos = obj["targets"]["upos"]
            for token, label in zip(tokens, upos, strict=True):
                lowered = str(token).lower()
                label = str(label)
                examples.append((lowered, label))
                labels.append(label)
                for length in range(1, min(5, len(lowered)) + 1):
                    suffix_counts[(length, lowered[-length:])][label] += 1
        self.forms = fit_count_table(examples)
        self.fallback = majority_label(labels)
        for key, counts in suffix_counts.items():
            total = sum(counts.values())
            label, count = max(counts.items(), key=lambda item: (item[1], item[0]))
            if total >= 10:
                self.suffixes[key] = TableEntry(label, count / (total + ALPHA), count, total)

    def predict(self, token: str) -> str:
        entry = self.forms.lookup(token.lower())
        if entry is not None:
            return entry.label
        lowered = token.lower()
        for length in range(min(5, len(lowered)), 0, -1):
            suffix = self.suffixes.get((length, lowered[-length:]))
            if suffix is not None:
                return suffix.label
        if not any(char.isalnum() for char in token):
            return "PUNCT"
        return self.fallback


def load_pos_train(path: Path) -> list[Mapping[str, Any]]:
    """Load only GSD POS train for the explicitly allowed predicted-POS pass."""
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def is_punctuation(token: str) -> bool:
    return bool(token) and not any(char.isalnum() for char in token)


def suffix_shape(token: str) -> str:
    lowered = token.lower()
    if lowered.startswith("ge") and lowered.endswith(("t", "en")) and len(lowered) > 4:
        return "ge-participle"
    if lowered.endswith(("iert", "isiert")):
        return "iert-participle"
    if lowered.endswith(("en", "ern", "eln")) and len(lowered) > 4:
        return "bare-en"
    if lowered.endswith(("t", "te", "ten")) and len(lowered) > 3:
        return "bare-t"
    return "other"


def _span(tokens: Sequence[str], index: int, boundaries: frozenset[str]) -> tuple[int, int]:
    left = index
    while left > 0 and tokens[left - 1].lower() not in boundaries:
        left -= 1
    right = index + 1
    while right < len(tokens) and tokens[right].lower() not in boundaries:
        right += 1
    return left, right


def is_participle_like(token: str, pos: SurfacePosPredictor) -> bool:
    shape = suffix_shape(token)
    if shape in {"ge-participle", "iert-participle"}:
        return True
    return shape in {"bare-en", "bare-t"} and pos.predict(token) in {"VERB", "ADJ", "AUX"}


def is_infinitive_like(token: str, pos: SurfacePosPredictor) -> bool:
    return suffix_shape(token) == "bare-en" and pos.predict(token) in {"VERB", "AUX"}


def followed_by_noun_rule(tokens: Sequence[str], index: int) -> Candidate | None:
    """The app catalog's serve-honest noun proxy, isolated for testing."""
    if index + 1 < len(tokens) and tokens[index + 1][:1].isupper():
        return Candidate("DET", 1.0, "rule")
    return None


def paired_with_participle(
    tokens: Sequence[str], index: int, pos: SurfacePosPredictor
) -> bool:
    """Surface/POS-predicted participle pairing inside the hard-punctuation span."""
    left, right = _span(tokens, index, HARD_BOUNDARIES)
    return any(
        is_participle_like(tokens[position], pos)
        for position in range(left, right)
        if position != index
    )


def _form_family(form: str) -> str:
    if form in HABEN_FORMS:
        return "haben"
    if form in SEIN_FORMS:
        return "sein"
    if form in WERDEN_FORMS:
        return "werden"
    return "modal-or-other"


def context_features(
    task: str, tokens: Sequence[str], index: int, pos: SurfacePosPredictor
) -> dict[str, str]:
    previous = tokens[index - 1] if index else "<BOS>"
    following = tokens[index + 1] if index + 1 < len(tokens) else "<EOS>"
    previous2 = tokens[index - 2] if index > 1 else "<BOS2>"
    following2 = tokens[index + 2] if index + 2 < len(tokens) else "<EOS2>"
    form = tokens[index].lower()
    features = {
        "form": form,
        "previous_form": previous.lower(),
        "next_form": following.lower(),
        "previous_pos_pred": pos.predict(previous) if index else "BOS",
        "next_pos_pred": pos.predict(following) if index + 1 < len(tokens) else "EOS",
        "previous2_pos_pred": pos.predict(previous2) if index > 1 else "BOS",
        "next2_pos_pred": pos.predict(following2) if index + 2 < len(tokens) else "EOS",
        "previous_capitalized": str(previous[:1].isupper()),
        "next_capitalized": str(following[:1].isupper()),
        "previous_punctuation": str(is_punctuation(previous)),
        "next_punctuation": str(is_punctuation(following)),
        "bos": str(index == 0),
        "eos": str(index == len(tokens) - 1),
        "previous_suffix": suffix_shape(previous),
        "next_suffix": suffix_shape(following),
    }
    if task == "aux_verb":
        hard_left, hard_right = _span(tokens, index, HARD_BOUNDARIES)
        clause_left, clause_right = _span(tokens, index, CLAUSE_BOUNDARIES)

        def has(predicate: Callable[[str], bool], start: int, stop: int) -> str:
            return str(
                any(predicate(tokens[position]) for position in range(start, stop) if position != index)
            )

        features.update(
            {
                "form_family": _form_family(form),
                "paired_participle": str(paired_with_participle(tokens, index, pos)),
                "hard_left_participle": has(lambda token: is_participle_like(token, pos), hard_left, index),
                "hard_right_participle": has(
                    lambda token: is_participle_like(token, pos), index + 1, hard_right
                ),
                "clause_left_participle": has(
                    lambda token: is_participle_like(token, pos), clause_left, index
                ),
                "clause_right_participle": has(
                    lambda token: is_participle_like(token, pos), index + 1, clause_right
                ),
                "hard_left_infinitive": has(lambda token: is_infinitive_like(token, pos), hard_left, index),
                "hard_right_infinitive": has(
                    lambda token: is_infinitive_like(token, pos), index + 1, hard_right
                ),
                "clause_left_infinitive": has(
                    lambda token: is_infinitive_like(token, pos), clause_left, index
                ),
                "clause_right_infinitive": has(
                    lambda token: is_infinitive_like(token, pos), index + 1, clause_right
                ),
                "next_locative": str(following.lower() in LOCATIVE_FORMS),
                "previous_locative": str(previous.lower() in LOCATIVE_FORMS),
                "clause_has_locative": has(
                    lambda token: token.lower() in LOCATIVE_FORMS, clause_left, clause_right
                ),
            }
        )
    return features


DET_FEATURES = (
    "form",
    "previous_form",
    "next_form",
    "previous_pos_pred",
    "next_pos_pred",
    "previous2_pos_pred",
    "next2_pos_pred",
    "previous_capitalized",
    "next_capitalized",
    "previous_punctuation",
    "next_punctuation",
    "bos",
    "eos",
    "previous_suffix",
    "next_suffix",
)
AUX_FEATURES = DET_FEATURES + (
    "form_family",
    "paired_participle",
    "hard_left_participle",
    "hard_right_participle",
    "clause_left_participle",
    "clause_right_participle",
    "hard_left_infinitive",
    "hard_right_infinitive",
    "clause_left_infinitive",
    "clause_right_infinitive",
    "next_locative",
    "previous_locative",
    "clause_has_locative",
)


@dataclass
class TreeNode:
    entry: TableEntry
    feature: str | None = None
    common_values: frozenset[str] = frozenset()
    children: dict[str, TreeNode] | None = None

    @property
    def leaf_count(self) -> int:
        if not self.children:
            return 1
        return sum(child.leaf_count for child in self.children.values())


def _entry(labels: Sequence[str]) -> TableEntry:
    counts = Counter(labels)
    label, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return TableEntry(label, count / (len(labels) + ALPHA), count, len(labels))


def _entropy(labels: Sequence[str]) -> float:
    counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


class CountDecisionTree:
    """A deterministic multiway count tree; leaves are inspectable majority rules."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(feature_names)
        self.root: TreeNode | None = None

    def fit(self, rows: Sequence[tuple[Mapping[str, str], str]]) -> None:
        if not rows:
            raise ValueError("cannot fit context tree without rows")
        self.root = self._grow(rows, self.feature_names, 0)

    def _partition(
        self, rows: Sequence[tuple[Mapping[str, str], str]], feature: str
    ) -> tuple[dict[str, list[tuple[Mapping[str, str], str]]], frozenset[str]]:
        value_counts = Counter(features[feature] for features, _ in rows)
        common = frozenset(value for value, count in value_counts.items() if count >= TREE_MIN_LEAF)
        groups: dict[str, list[tuple[Mapping[str, str], str]]] = defaultdict(list)
        for features, label in rows:
            value = features[feature]
            groups[value if value in common else "<RARE>"].append((features, label))
        return dict(groups), common

    def _grow(
        self,
        rows: Sequence[tuple[Mapping[str, str], str]],
        features: Sequence[str],
        depth: int,
    ) -> TreeNode:
        labels = [label for _, label in rows]
        node = TreeNode(_entry(labels))
        if depth >= TREE_MAX_DEPTH or len(set(labels)) == 1 or not features:
            return node
        parent_entropy = _entropy(labels)
        choices: list[
            tuple[float, str, dict[str, list[tuple[Mapping[str, str], str]]], frozenset[str]]
        ] = []
        for feature in features:
            groups, common = self._partition(rows, feature)
            if len(groups) < 2:
                continue
            remainder = sum(
                len(group) / len(rows) * _entropy([label for _, label in group])
                for group in groups.values()
            )
            choices.append((parent_entropy - remainder, feature, groups, common))
        if not choices:
            return node
        gain, feature, groups, common = max(choices, key=lambda item: (item[0], item[1]))
        if gain < TREE_MIN_GAIN:
            return node
        remaining = tuple(name for name in features if name != feature)
        node.feature = feature
        node.common_values = common
        node.children = {
            value: self._grow(group, remaining, depth + 1)
            for value, group in sorted(groups.items())
        }
        return node

    def predict(self, features: Mapping[str, str]) -> TableEntry:
        if self.root is None:
            raise RuntimeError("context tree has not been fit")
        node = self.root
        while node.feature is not None and node.children:
            raw = features[node.feature]
            value = raw if raw in node.common_values else "<RARE>"
            child = node.children.get(value)
            if child is None:
                break
            node = child
        return node.entry

    @property
    def used_features(self) -> tuple[str, ...]:
        found: set[str] = set()

        def walk(node: TreeNode | None) -> None:
            if node is None or node.feature is None:
                return
            found.add(node.feature)
            for child in (node.children or {}).values():
                walk(child)

        walk(self.root)
        return tuple(sorted(found))

    @property
    def leaf_count(self) -> int:
        return self.root.leaf_count if self.root is not None else 0


class ConditionalCatalogRule:
    """Catalog first branch plus a train-mined complementary count tree."""

    def __init__(self, task: str, pos: SurfacePosPredictor) -> None:
        self.task = task
        self.pos = pos
        self.tree = CountDecisionTree(DET_FEATURES if task == "det_pron" else AUX_FEATURES)
        self.catalog_entry = TableEntry(LABELS[task][0], 0.0, 0, 0)

    def _catalog_fires(self, tokens: Sequence[str], index: int) -> bool:
        if self.task == "det_pron":
            return followed_by_noun_rule(tokens, index) is not None
        return paired_with_participle(tokens, index, self.pos)

    def fit(self, records: Sequence[TargetRecord]) -> None:
        rows: list[tuple[Mapping[str, str], str]] = []
        branch_labels: list[str] = []
        for record in records:
            for index, _, label in record.targets():
                rows.append((context_features(self.task, record.tokens, index, self.pos), label))
                if self._catalog_fires(record.tokens, index):
                    branch_labels.append(label)
        self.tree.fit(rows)
        if branch_labels:
            counts = Counter(branch_labels)
            expected = LABELS[self.task][0]
            count = counts[expected]
            self.catalog_entry = TableEntry(
                expected, count / (len(branch_labels) + ALPHA), count, len(branch_labels)
            )

    def predict(self, tokens: Sequence[str], index: int) -> TableEntry:
        if self._catalog_fires(tokens, index):
            return self.catalog_entry
        return self.tree.predict(context_features(self.task, tokens, index, self.pos))

    @property
    def description(self) -> str:
        used = ", ".join(self.tree.used_features)
        if self.task == "det_pron":
            catalog = "next token capitalized => DET"
        else:
            catalog = "surface/POS-predicted clause participle paired => AUX"
        return (
            f"{catalog}; otherwise train-mined count tree "
            f"({self.tree.leaf_count} leaves; features: {used})"
        )


class GermanR2TaskStudent:
    def __init__(self, task: str, pos: SurfacePosPredictor) -> None:
        self.task = task
        self.pos = pos
        self.form_table = CountTable()
        self.fallback = LABELS[task][0]
        self.rule = ConditionalCatalogRule(task, pos)
        self.rule_evidence: RuleEvidence | None = None
        self.catalog_evidence: CatalogEvidence | None = None

    def fit(self, train: Sequence[TargetRecord], dev: Sequence[TargetRecord]) -> None:
        examples = [
            (token.lower(), label)
            for record in train
            for _, token, label in record.targets()
        ]
        self.form_table = fit_count_table(examples)
        self.fallback = majority_label(label for _, label in examples)
        self.rule.fit(train)
        self._judge(dev)

    def memorizer_entry(self, token: str) -> TableEntry:
        entry = self.form_table.lookup(token.lower())
        return entry or TableEntry(self.fallback, 0.0, 0, 0)

    def _judge(self, dev: Sequence[TargetRecord]) -> None:
        support = correct = memory_correct = naive_correct = 0
        branch_support = branch_correct = 0
        for record in dev:
            for index, token, label in record.targets():
                support += 1
                memory = self.memorizer_entry(token).label
                prediction = self.rule.predict(record.tokens, index).label
                fires = self.rule._catalog_fires(record.tokens, index)
                correct += prediction == label
                memory_correct += memory == label
                naive_correct += (LABELS[self.task][0] if fires else memory) == label
                if fires:
                    branch_support += 1
                    branch_correct += LABELS[self.task][0] == label
        dev_accuracy = correct / support
        memory_accuracy = memory_correct / support
        margin = dev_accuracy - memory_accuracy
        admitted = (
            support >= DEV_MIN_SUPPORT
            and dev_accuracy >= DEV_MIN_ACCURACY
            and margin >= DEV_MIN_MARGIN
        )
        name = "noun-follower conditional" if self.task == "det_pron" else "participle-pair conditional"
        self.rule_evidence = RuleEvidence(
            name=name,
            description=self.rule.description,
            predicted_label=f"conditional {LABELS[self.task][0]}/{LABELS[self.task][1]}",
            dev_support=support,
            dev_accuracy=dev_accuracy,
            dev_memorizer_accuracy=memory_accuracy,
            margin=margin,
            admitted=admitted,
        )
        naive_accuracy = naive_correct / support
        branch_accuracy = branch_correct / branch_support if branch_support else 0.0
        if not admitted or not branch_support:
            status = "MISSED"
        elif dev_accuracy > naive_accuracy:
            status = "IMPROVED"
        else:
            status = "RECOVERED"
        self.catalog_evidence = CatalogEvidence(
            status, branch_support, branch_accuracy, naive_accuracy, dev_accuracy
        )

    def predict(self, tokens: Sequence[str], index: int, rules_on: bool = True) -> str:
        form = self.memorizer_entry(tokens[index])
        candidates = [Candidate(form.label, form.confidence, "form")]
        if rules_on and self.rule_evidence is not None and self.rule_evidence.admitted:
            entry = self.rule.predict(tokens, index)
            candidates.append(Candidate(entry.label, entry.confidence, "rule"))
        return choose_candidate(candidates, self.fallback)


def score_indexed(
    records: Sequence[TargetRecord],
    predictor: Callable[[Sequence[str], int], str],
) -> float:
    correct = total = 0
    for record in records:
        for index, _, label in record.targets():
            correct += predictor(record.tokens, index) == label
            total += 1
    if not total:
        raise ValueError("cannot score zero indexed targets")
    return correct / total


def evaluate_task(student: GermanR2TaskStudent, records: Sequence[TargetRecord]) -> TaskScores:
    total = sum(len(record.indices) for record in records)
    full = score_indexed(records, lambda tokens, index: student.predict(tokens, index, True))
    memorizer = score_indexed(records, lambda tokens, index: student.predict(tokens, index, False))
    return TaskScores(full, memorizer, total)


def verdict(det_pron_acc: float, aux_verb_acc: float, catalog_recovered: bool) -> str:
    return (
        "FIRES"
        if det_pron_acc >= BAR and aux_verb_acc >= BAR and catalog_recovered
        else "IN-BETWEEN/MISS"
    )


def print_scoreboard(
    students: Mapping[str, GermanR2TaskStudent],
    scores: Mapping[str, TaskScores],
    data_hash: str,
    guard: TestReadGuard,
) -> None:
    print("\nGERMAN R2 CODEX FINAL SCOREBOARD")
    print("run_tag: empirical")
    print(
        "supervision: GSD gold only; task train fits priors/rules; dev only admits rules; "
        "test only final evaluation; POS train fits a serve-time predicted-POS pass; no teacher data touched"
    )
    print("register_prior: not used; per-task form tables are mined from task train only")
    print("SOFT=0: count fitting and one train/dev/test pass; no gradient descent or wake/sleep")
    print(
        "judge_gate: dev_support>=5 AND dev_accuracy>=0.70 AND "
        "dev_marginal_over_same-subset_memorizer>=+0.01"
    )
    print(
        "tree_config: max_depth=6 min_leaf=8 min_information_gain=0.001; "
        "Laplace alpha=2.0"
    )
    print("serve-honest features used:")
    print("  - target lower-cased surface form and token length/affix shape")
    print("  - original-case capitalization and punctuation of adjacent tokens")
    print("  - literal previous/next surface forms")
    print("  - ge-...-t/en, -iert, and bare -t/-en neighbour morphology")
    print("  - clause-local participle/infinitive and locative literal signals")
    print("  - train-only surface-form/suffix predicted POS for neighbours (never neighbour gold POS)")
    print(
        "hash_order: det_pron train, aux_verb train, det_pron dev, aux_verb dev, "
        "det_pron test, aux_verb test"
    )
    print(f"data_hash_sha256: {data_hash}")
    if not guard.confirmed:
        raise RuntimeError(f"expected one test read per task, claimed={sorted(guard.claimed)}")
    print("TEST READ ONCE: CONFIRMED")
    print("\ntask       full(rules-on)  memorizer(rules-off)  rule_marginal  indexed_n")
    for task in TASKS:
        score = scores[task]
        print(
            f"{task:<10} {score.full:.6f}        {score.memorizer:.6f}               "
            f"{score.full - score.memorizer:+.6f}      {score.total}"
        )
    print("\nadmitted rules (dev judged):")
    for task in TASKS:
        evidence = students[task].rule_evidence
        if evidence is None or not evidence.admitted:
            if evidence is None:
                print(f"  {task}: NONE")
            else:
                print(
                    f"  {task}: NONE (candidate={evidence.name}; dev_support={evidence.dev_support}; "
                    f"dev_fired_accuracy={evidence.dev_accuracy:.6f}; "
                    f"memorizer_same_subset={evidence.dev_memorizer_accuracy:.6f}; "
                    f"margin={evidence.margin:+.6f})"
                )
            continue
        print(f"  {task}: {evidence.name}")
        print(f"    context: {evidence.description}")
        print(f"    predicted_label: {evidence.predicted_label}")
        print(
            f"    dev_support={evidence.dev_support} "
            f"dev_fired_accuracy={evidence.dev_accuracy:.6f} "
            f"memorizer_same_subset={evidence.dev_memorizer_accuracy:.6f} "
            f"margin={evidence.margin:+.6f}"
        )
    print("\ncatalog-recovery check:")
    catalog_names = {
        "det_pron": '"followed-by-NOUN => DET" (next-capitalized proxy)',
        "aux_verb": '"paired-with-participle => AUX"',
    }
    recovered = True
    for task in TASKS:
        evidence = students[task].catalog_evidence
        assert evidence is not None
        recovered &= evidence.status in {"RECOVERED", "IMPROVED"}
        print(
            f"  {catalog_names[task]}: {evidence.status} "
            f"(branch_dev_support={evidence.support}, branch_dev_accuracy={evidence.branch_accuracy:.6f}, "
            f"naive_catalog_rule_dev_accuracy={evidence.naive_accuracy:.6f}, "
            f"admitted_rule_dev_accuracy={evidence.full_rule_accuracy:.6f})"
        )
    result = verdict(scores["det_pron"].full, scores["aux_verb"].full, recovered)
    print(
        f"VERDICT: {result} (empirical; det_pron={scores['det_pron'].full:.6f}, "
        f"aux_verb={scores['aux_verb'].full:.6f}, both_catalog_rules={recovered})"
    )


def main() -> None:
    hasher = hashlib.sha256()
    print("Loading GSD POS train for the train-only serve-time POS predictor.", flush=True)
    pos = SurfacePosPredictor()
    pos.fit(load_pos_train(DATA_ROOT / "tasks" / "pos" / "train.jsonl"))

    train: dict[str, list[TargetRecord]] = {}
    dev: dict[str, list[TargetRecord]] = {}
    for task in TASKS:
        print(f"Loading {task} train gold.", flush=True)
        train[task] = load_task_split(DATA_ROOT, task, "train", hasher)
    for task in TASKS:
        print(f"Loading {task} dev for fixed-gate rule admission.", flush=True)
        dev[task] = load_task_split(DATA_ROOT, task, "dev", hasher)

    students: dict[str, GermanR2TaskStudent] = {}
    for task in TASKS:
        student = GermanR2TaskStudent(task, pos)
        student.fit(train[task], dev[task])
        students[task] = student
    del train, dev

    guard = TestReadGuard()
    test: dict[str, list[TargetRecord]] = {}
    for task in TASKS:
        print(f"FINAL EVAL: reading {task} test now (first and only read).", flush=True)
        test[task] = load_task_split(DATA_ROOT, task, "test", hasher, guard)
    scores = {task: evaluate_task(students[task], test[task]) for task in TASKS}
    print_scoreboard(students, scores, hasher.hexdigest(), guard)


if __name__ == "__main__":
    main()
