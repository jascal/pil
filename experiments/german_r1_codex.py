"""German R1: deterministic per-token count student with hard registers.

This is a pure-Python reimplementation of ``wyly_lm_v5.best_per_key`` and
``core_cover_sw``.  Dicts are clearer than padded GPU tensors for variable-length
sentences, while retaining the same gated majority, Laplace confidence, and
highest-confidence-wins semantics.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALPHA = 2.0
FORM_MINSUPP = 1
FORM_MINDET = 0.0
CONTEXT_MINSUPP = 2
CONTEXT_MINDET = 0.5
HEAD_TARGETS = {"pos": "upos", "morph_case": "case", "morph_gnn": "gnn"}
CASE_LABELS = ("Nom", "Acc", "Dat", "Gen", "-")
NP_POS = frozenset({"DET", "PRON", "ADJ", "NOUN", "PROPN", "NUM"})
DATA_ROOT = Path("/home/allans/code/germandata")

# R1's government coverage is PARTIAL: a parse-free forward NP-span scan has no
# attachment analysis, and lightweight verb stem/prefix matching misses pre-verbal
# objects and irregular forms. The reported register marginal is therefore a LOWER
# BOUND on what a full parse-based register layer (R3, out of scope) would deliver.
GOVERNMENT_CAVEAT = (
    "R1's government coverage is PARTIAL: the parse-free forward NP-span scan has no "
    "attachment analysis, and lightweight verb stem/prefix matching misses pre-verbal "
    "objects and irregular forms. The register marginal reported here is a LOWER BOUND "
    "on what a full parse-based register layer (R3, out of scope) would deliver."
)


@dataclass(frozen=True)
class TaskRecord:
    sent_id: str
    text: str
    tokens: tuple[str, ...]
    targets: tuple[str, ...]


@dataclass(frozen=True)
class Sentence:
    sent_id: str
    text: str
    tokens: tuple[str, ...]
    targets: Mapping[str, tuple[str, ...]]


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
class RegisterProposal:
    """A hard register constraint, singleton or ambiguous prune set."""

    labels: frozenset[str]


@dataclass(frozen=True)
class CaseScores:
    null_floor: float
    per_class_accuracy: Mapping[str, float]
    per_class_total: Mapping[str, int]


class CountTable:
    """A gated majority-per-key table carrying Laplace-shrunk confidence."""

    def __init__(self, entries: Mapping[Any, TableEntry] | None = None) -> None:
        self.entries = dict(entries or {})

    def lookup(self, key: Any) -> TableEntry | None:
        return self.entries.get(key)


class TestReadGuard:
    """Makes a second attempt to load the registered test split a hard error."""

    def __init__(self) -> None:
        self.read_count = 0

    def claim(self) -> None:
        if self.read_count:
            raise RuntimeError("GSD test split may be read only once")
        self.read_count = 1


def parse_task_record(raw: str | bytes | Mapping[str, Any], target_name: str) -> TaskRecord:
    """Parse and validate one task record; pre-expanded tokens pass through unchanged."""
    obj = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    tokens = tuple(obj["tokens"])
    targets = tuple(obj["targets"][target_name])
    if len(tokens) != len(targets):
        raise ValueError(
            f"{obj.get('sent_id', '<unknown>')}: {len(tokens)} tokens but "
            f"{len(targets)} {target_name} targets"
        )
    return TaskRecord(str(obj["sent_id"]), str(obj["text"]), tokens, targets)


def load_task_file(path: Path, target_name: str, hasher: Any | None = None) -> list[TaskRecord]:
    """Read one JSONL file exactly once, optionally adding its raw bytes to a hash."""
    records: list[TaskRecord] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            if hasher is not None:
                hasher.update(raw_line)
            if raw_line.strip():
                records.append(parse_task_record(raw_line, target_name))
    return records


def align_task_records(
    pos_records: Sequence[TaskRecord],
    case_records: Sequence[TaskRecord],
    gnn_records: Sequence[TaskRecord],
) -> list[Sentence]:
    """Validate 1:1 sentence/token alignment and combine the three gold heads."""
    if not (len(pos_records) == len(case_records) == len(gnn_records)):
        raise ValueError("task files have different sentence counts")
    combined: list[Sentence] = []
    for pos, case, gnn in zip(pos_records, case_records, gnn_records, strict=True):
        if pos.sent_id != case.sent_id or pos.sent_id != gnn.sent_id:
            raise ValueError(f"misaligned sent_id near {pos.sent_id}")
        if pos.tokens != case.tokens or pos.tokens != gnn.tokens:
            raise ValueError(f"misaligned tokens in {pos.sent_id}")
        combined.append(
            Sentence(
                sent_id=pos.sent_id,
                text=pos.text,
                tokens=pos.tokens,
                targets={"pos": pos.targets, "morph_case": case.targets, "morph_gnn": gnn.targets},
            )
        )
    return combined


def load_split(data_root: Path, split: str, hasher: Any | None = None) -> list[Sentence]:
    """Load a split in the documented hash order: POS, case, then GNN."""
    loaded: dict[str, list[TaskRecord]] = {}
    for head, target in HEAD_TARGETS.items():
        path = data_root / "tasks" / head / f"{split}.jsonl"
        loaded[head] = load_task_file(path, target, hasher)
    return align_task_records(loaded["pos"], loaded["morph_case"], loaded["morph_gnn"])


def load_test_once(
    data_root: Path, hasher: Any, guard: TestReadGuard
) -> list[Sentence]:
    guard.claim()
    return load_split(data_root, "test", hasher)


def fit_count_table(
    examples: Iterable[tuple[Any, str]],
    minsupp: int,
    mindet: float,
    alpha: float = ALPHA,
) -> CountTable:
    """Fit v5-style majority per key with support and raw-determinism gates."""
    counts: dict[Any, Counter[str]] = defaultdict(Counter)
    for key, label in examples:
        counts[key][label] += 1
    entries: dict[Any, TableEntry] = {}
    for key, label_counts in counts.items():
        total = sum(label_counts.values())
        # Lexicographic label order makes exact count ties deterministic.
        label, count = max(label_counts.items(), key=lambda item: (item[1], item[0]))
        if count >= minsupp and count / total >= mindet:
            entries[key] = TableEntry(label, count / (total + alpha), count, total)
    return CountTable(entries)


def predict_table_or_fallback(table: CountTable, key: Any, fallback: str) -> str:
    entry = table.lookup(key)
    return entry.label if entry is not None else fallback


def global_mode(labels: Iterable[str]) -> str:
    counts = Counter(labels)
    if not counts:
        raise ValueError("cannot find mode of empty labels")
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def choose_candidate(candidates: Iterable[Candidate], fallback: str) -> str:
    """Highest confidence wins; register > context > form on exact ties."""
    tier_rank = {"form": 0, "context": 1, "register": 2}
    available = list(candidates)
    if not available:
        return fallback
    return max(available, key=lambda c: (c.confidence, tier_rank[c.tier], c.label)).label


def narrow_and_arbitrate(
    count_candidates: Iterable[Candidate],
    register_proposals: Iterable[RegisterProposal],
    label_frequencies: Mapping[str, int],
    fallback: str,
) -> str:
    """Intersect hard constraints, then arbitrate only among allowed labels."""
    counts = list(count_candidates)
    proposals = list(register_proposals)
    if not proposals:
        return choose_candidate(counts, fallback)

    allowed = set(proposals[0].labels)
    for proposal in proposals[1:]:
        allowed.intersection_update(proposal.labels)
    if not allowed:
        return choose_candidate(counts, fallback)

    survivors = [candidate for candidate in counts if candidate.label in allowed]
    if survivors:
        return choose_candidate(survivors, fallback)
    return max(allowed, key=lambda label: (label_frequencies.get(label, 0), label))


def invert_article_table(table: Mapping[str, Any]) -> dict[str, set[tuple[str, str, str]]]:
    """Map lowercased article form to all (paradigm, case, gender-key) realizations."""
    inverse: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for paradigm, by_case in table.items():
        for case, by_gender in by_case.items():
            for gender_key, form in by_gender.items():
                inverse[str(form).lower()].add((str(paradigm), str(case), str(gender_key)))
    return dict(inverse)


def article_candidate_sets(
    inverse: Mapping[str, set[tuple[str, str, str]]], form: str
) -> tuple[set[str], set[str]]:
    triples = inverse.get(form.lower(), set())
    cases = {case for _, case, _ in triples}
    gnn = {_article_gnn(gender_key) for _, _, gender_key in triples}
    return cases, gnn


def _article_gnn(gender_key: str) -> str:
    return "-|Plur" if gender_key == "Plur" else f"{gender_key}|Sing"


def _pronoun_gnn(identity: str) -> str:
    parts = identity.split("_")
    number = "Plur" if "Plur" in parts or "Formal" in parts else "Sing"
    gender = next((part for part in parts if part in {"Masc", "Fem", "Neut"}), "-")
    return f"{gender}|{number}"


def invert_pronoun_table(table: Mapping[str, Any]) -> dict[str, set[tuple[str, str]]]:
    inverse: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for identity, by_case in table.items():
        for case, form in by_case.items():
            inverse[str(form).lower()].add((str(case), _pronoun_gnn(str(identity))))
    return dict(inverse)


def invert_adjective_endings(
    table: Mapping[str, Any], alpha: float = ALPHA
) -> dict[str, tuple[set[str], float]]:
    """Collect suffix case sets and their Laplace-shrunk majority confidence."""
    case_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for by_case in table.values():
        for case, by_gender in by_case.items():
            for suffix in by_gender.values():
                case_counts[str(suffix)][str(case)] += 1
    result: dict[str, tuple[set[str], float]] = {}
    for suffix, counts in case_counts.items():
        total = sum(counts.values())
        best = max(counts.values())
        result[suffix] = (set(counts), best / (total + alpha))
    return result


def invert_adjective_gnn(
    table: Mapping[str, Any], alpha: float = ALPHA
) -> dict[str, tuple[set[str], float]]:
    """Collect suffix gender-number sets analogously to adjective case sets."""
    gnn_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for by_case in table.values():
        for by_gender in by_case.values():
            for gender_key, suffix in by_gender.items():
                gnn_counts[str(suffix)][_article_gnn(str(gender_key))] += 1
    result: dict[str, tuple[set[str], float]] = {}
    for suffix, counts in gnn_counts.items():
        total = sum(counts.values())
        best = max(counts.values())
        result[suffix] = (set(counts), best / (total + alpha))
    return result


def _majority_entry(record: Mapping[str, Any]) -> TableEntry | None:
    """Apply the pilot's existing majority/support/determinism conventions."""
    counts = record["counts"]
    total = int(record["tot"])
    label, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    if count < FORM_MINSUPP or count / total < FORM_MINDET:
        return None
    return TableEntry(str(label), count / (total + ALPHA), count, total)


def _verb_stem(form: str) -> str:
    """Strip one common German infinitival or finite ending, without lemmatizing."""
    lowered = form.lower()
    for ending in ("test", "tet", "ten", "est", "st", "et", "en", "te", "t", "e", "n"):
        if lowered.endswith(ending) and len(lowered) > len(ending):
            return lowered[: -len(ending)]
    return lowered


def _load_register(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        envelope = json.load(handle)
    if not {"register", "origin", "language", "table"} <= envelope.keys():
        raise ValueError(f"bad register envelope: {path}")
    return envelope


class RegisterLayer:
    """Confidence-carrying authored and mined R1 register terms."""

    def __init__(
        self,
        article_inverse: Mapping[str, set[tuple[str, str, str]]],
        adjective_suffixes: Mapping[str, tuple[set[str], float]],
        adjective_gnn_suffixes: Mapping[str, tuple[set[str], float]],
        pronoun_inverse: Mapping[str, set[tuple[str, str]]],
        det_pron: Mapping[str, TableEntry],
        preposition_cases: Mapping[str, frozenset[str]],
        verb_government: Mapping[str, str],
    ) -> None:
        self.article_inverse = article_inverse
        self.adjective_suffixes = adjective_suffixes
        self.adjective_gnn_suffixes = adjective_gnn_suffixes
        self.pronoun_inverse = pronoun_inverse
        self.det_pron = det_pron
        self.preposition_cases = preposition_cases
        self.verb_government = verb_government

    @classmethod
    def from_directory(cls, register_dir: Path) -> RegisterLayer:
        article = _load_register(register_dir / "article_paradigm.json")
        adjective = _load_register(register_dir / "adjective_endings.json")
        pronoun = _load_register(register_dir / "pronoun_paradigm.json")
        det_pron_envelope = _load_register(register_dir / "det_pron_ambiguous.json")
        prep = _load_register(register_dir / "preposition_case.json")
        verb = _load_register(register_dir / "verb_government.json")

        det_pron: dict[str, TableEntry] = {}
        for form, record in det_pron_envelope["table"].items():
            entry = _majority_entry(record)
            if entry is not None:
                det_pron[form.lower()] = entry

        preposition_cases = {
            form.lower(): frozenset(str(case) for case in cases)
            for form, cases in prep["table"]["core"].items()
        }
        verb_government: dict[str, str] = {}
        for lemma, record in verb["table"].items():
            entry = _majority_entry(record)
            if entry is None or entry.label not in {"obj:Acc", "obj:Dat", "obj:Gen"}:
                continue
            verb_government[lemma.lower()] = entry.label.removeprefix("obj:")

        return cls(
            invert_article_table(article["table"]),
            invert_adjective_endings(adjective["table"]),
            invert_adjective_gnn(adjective["table"]),
            invert_pronoun_table(pronoun["table"]),
            det_pron,
            preposition_cases,
            verb_government,
        )

    def pos_candidate(self, form: str) -> Candidate | None:
        entry = self.det_pron.get(form.lower())
        return Candidate(entry.label, entry.confidence, "register") if entry else None

    def direct_case_candidates(self, form: str, predicted_pos: str) -> list[RegisterProposal]:
        candidates: list[RegisterProposal] = []
        article_cases, _ = article_candidate_sets(self.article_inverse, form)
        if article_cases:
            candidates.append(RegisterProposal(frozenset(article_cases)))

        pronoun = self.pronoun_inverse.get(form.lower(), set())
        pronoun_cases = {case for case, _ in pronoun}
        if pronoun_cases:
            candidates.append(RegisterProposal(frozenset(pronoun_cases)))

        if predicted_pos == "ADJ":
            matches = [
                (suffix, rule)
                for suffix, rule in self.adjective_suffixes.items()
                if suffix and form.lower().endswith(suffix)
            ]
            if matches:
                _, (cases, _) = max(matches, key=lambda item: len(item[0]))
                candidates.append(RegisterProposal(frozenset(cases)))
        return candidates

    def direct_gnn_candidates(
        self, form: str, predicted_pos: str | None = None
    ) -> list[RegisterProposal]:
        candidates: list[RegisterProposal] = []
        _, article_gnn = article_candidate_sets(self.article_inverse, form)
        if article_gnn:
            candidates.append(RegisterProposal(frozenset(article_gnn)))
        pronoun = self.pronoun_inverse.get(form.lower(), set())
        pronoun_gnn = {gnn for _, gnn in pronoun}
        if pronoun_gnn:
            candidates.append(RegisterProposal(frozenset(pronoun_gnn)))
        if predicted_pos in {None, "ADJ"}:
            matches = [
                (suffix, rule)
                for suffix, rule in self.adjective_gnn_suffixes.items()
                if suffix and form.lower().endswith(suffix)
            ]
            if matches:
                _, (gnn, _) = max(matches, key=lambda item: len(item[0]))
                candidates.append(RegisterProposal(frozenset(gnn)))
        return candidates

    def _verb_government_case(self, form: str) -> str | None:
        form_stem = _verb_stem(form)
        matches = [
            (len(_verb_stem(lemma)), lemma, case)
            for lemma, case in self.verb_government.items()
            if _verb_stem(lemma) == form_stem
        ]
        return max(matches)[2] if matches else None

    @staticmethod
    def _forward_np_span(anchor: int, predicted_pos: Sequence[str]) -> Iterable[int]:
        for index in range(anchor + 1, len(predicted_pos)):
            pos = predicted_pos[index]
            if pos == "PUNCT":
                continue
            if pos not in NP_POS:
                break
            yield index

    def government_candidates(
        self, tokens: Sequence[str], predicted_pos: Sequence[str]
    ) -> dict[int, list[RegisterProposal]]:
        """Project prep/verb government across a forward NP span to its boundary."""
        proposals: dict[int, list[RegisterProposal]] = defaultdict(list)
        for anchor, token in enumerate(tokens):
            pos = predicted_pos[anchor]
            allowed: frozenset[str] | None = None
            if pos == "ADP":
                allowed = self.preposition_cases.get(token.lower())
            elif pos in {"VERB", "AUX"}:
                verb_case = self._verb_government_case(token)
                if verb_case is not None:
                    allowed = frozenset({verb_case})
            if allowed is None:
                continue
            proposal = RegisterProposal(allowed)
            for index in self._forward_np_span(anchor, predicted_pos):
                proposals[index].append(proposal)
        return proposals


class GermanR1Student:
    """Three independent heads sharing the same form/context count machinery."""

    def __init__(self, registers: RegisterLayer) -> None:
        self.registers = registers
        self.form_tables: dict[str, CountTable] = {}
        self.context_tables: dict[str, CountTable] = {}
        self.fallbacks: dict[str, str] = {}
        self.label_frequencies: dict[str, Counter[str]] = {}

    def fit(self, train: Sequence[Sentence]) -> None:
        for head in HEAD_TARGETS:
            form_examples: list[tuple[str, str]] = []
            context_examples: list[tuple[tuple[str, str], str]] = []
            labels: list[str] = []
            for sentence in train:
                previous = "<BOS>"
                for token, label in zip(sentence.tokens, sentence.targets[head], strict=True):
                    form = token.lower()
                    form_examples.append((form, label))
                    context_examples.append(((form, previous), label))
                    labels.append(label)
                    previous = form
            self.form_tables[head] = fit_count_table(
                form_examples, FORM_MINSUPP, FORM_MINDET
            )
            self.context_tables[head] = fit_count_table(
                context_examples, CONTEXT_MINSUPP, CONTEXT_MINDET
            )
            self.label_frequencies[head] = Counter(labels)
            self.fallbacks[head] = global_mode(labels)

    def _count_candidates(self, head: str, form: str, previous: str) -> list[Candidate]:
        candidates: list[Candidate] = []
        form_entry = self.form_tables[head].lookup(form)
        if form_entry:
            candidates.append(Candidate(form_entry.label, form_entry.confidence, "form"))
        context_entry = self.context_tables[head].lookup((form, previous))
        if context_entry:
            candidates.append(Candidate(context_entry.label, context_entry.confidence, "context"))
        return candidates

    def _predict_pos(self, tokens: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
        full: list[str] = []
        off: list[str] = []
        memorizer: list[str] = []
        previous = "<BOS>"
        for token in tokens:
            form = token.lower()
            count_candidates = self._count_candidates("pos", form, previous)
            register = self.registers.pos_candidate(form)
            full_candidates = count_candidates + ([register] if register else [])
            fallback = self.fallbacks["pos"]
            full.append(choose_candidate(full_candidates, fallback))
            off.append(choose_candidate(count_candidates, fallback))
            memorizer.append(
                predict_table_or_fallback(self.form_tables["pos"], form, fallback)
            )
            previous = form
        return full, off, memorizer

    def _predict_morph_head(
        self,
        head: str,
        tokens: Sequence[str],
        full_pos: Sequence[str],
        government: Mapping[int, list[RegisterProposal]],
    ) -> tuple[list[str], list[str], list[str]]:
        full: list[str] = []
        off: list[str] = []
        memorizer: list[str] = []
        previous = "<BOS>"
        fallback = self.fallbacks[head]
        for index, token in enumerate(tokens):
            form = token.lower()
            count_candidates = self._count_candidates(head, form, previous)
            if head == "morph_case":
                register_candidates = self.registers.direct_case_candidates(
                    form, full_pos[index]
                ) + government.get(index, [])
            else:
                register_candidates = self.registers.direct_gnn_candidates(
                    form, full_pos[index]
                )
            full.append(
                narrow_and_arbitrate(
                    count_candidates,
                    register_candidates,
                    self.label_frequencies[head],
                    fallback,
                )
            )
            off.append(choose_candidate(count_candidates, fallback))
            memorizer.append(predict_table_or_fallback(self.form_tables[head], form, fallback))
            previous = form
        return full, off, memorizer

    def predict_sentence(self, sentence: Sentence) -> dict[str, dict[str, list[str]]]:
        result = {
            "full": {head: [] for head in HEAD_TARGETS},
            "registers_off": {head: [] for head in HEAD_TARGETS},
            "memorizer": {head: [] for head in HEAD_TARGETS},
        }
        pos_full, pos_off, pos_mem = self._predict_pos(sentence.tokens)
        result["full"]["pos"] = pos_full
        result["registers_off"]["pos"] = pos_off
        result["memorizer"]["pos"] = pos_mem

        government = self.registers.government_candidates(sentence.tokens, pos_full)
        for head in ("morph_case", "morph_gnn"):
            full, off, mem = self._predict_morph_head(
                head, sentence.tokens, pos_full, government if head == "morph_case" else {}
            )
            result["full"][head] = full
            result["registers_off"][head] = off
            result["memorizer"][head] = mem
        return result


def accuracy(predictions: Sequence[str], gold: Sequence[str]) -> float:
    if len(predictions) != len(gold):
        raise ValueError("prediction/gold length mismatch")
    if not gold:
        raise ValueError("cannot score an empty sequence")
    return sum(pred == target for pred, target in zip(predictions, gold, strict=True)) / len(gold)


def score_case(predictions: Sequence[str], gold: Sequence[str]) -> CaseScores:
    if len(predictions) != len(gold):
        raise ValueError("prediction/gold length mismatch")
    if not gold:
        raise ValueError("cannot score an empty sequence")
    per_class_accuracy: dict[str, float] = {}
    per_class_total: dict[str, int] = {}
    for label in CASE_LABELS:
        positions = [index for index, target in enumerate(gold) if target == label]
        per_class_total[label] = len(positions)
        per_class_accuracy[label] = (
            sum(predictions[index] == label for index in positions) / len(positions)
            if positions
            else 0.0
        )
    return CaseScores(
        null_floor=sum(target == "-" for target in gold) / len(gold),
        per_class_accuracy=per_class_accuracy,
        per_class_total=per_class_total,
    )


def verdict(pos_full: float, case_full: float) -> str:
    if pos_full >= 0.95 and case_full >= 0.90:
        return "FIRES"
    if pos_full < 0.85:
        return "HALTED"
    return "IN-BETWEEN"


def evaluate(
    student: GermanR1Student, test: Sequence[Sentence]
) -> tuple[dict[str, dict[str, float]], CaseScores]:
    predictions = {
        arm: {head: [] for head in HEAD_TARGETS}
        for arm in ("full", "registers_off", "memorizer")
    }
    gold = {head: [] for head in HEAD_TARGETS}
    for sentence in test:
        sentence_predictions = student.predict_sentence(sentence)
        for head in HEAD_TARGETS:
            gold[head].extend(sentence.targets[head])
            for arm in predictions:
                predictions[arm][head].extend(sentence_predictions[arm][head])
    scores = {
        arm: {head: accuracy(by_head[head], gold[head]) for head in HEAD_TARGETS}
        for arm, by_head in predictions.items()
    }
    return scores, score_case(predictions["full"]["morph_case"], gold["morph_case"])


def print_scoreboard(
    scores: Mapping[str, Mapping[str, float]],
    case_scores: CaseScores,
    data_hash: str,
    test_guard: TestReadGuard,
) -> None:
    print("\nGERMAN R1 FINAL SCOREBOARD")
    print("run_tag: empirical")
    print("supervision: GSD train gold ONLY; no teacher data read; dev used only for alignment validation")
    print(
        "config: alpha=2.0 form(minsupp=1,mindet=0.0) "
        "context=form+previous-form(minsupp=2,mindet=0.5) government=np-span-to-boundary"
    )
    print(
        "ambiguous_register_policy: intersect hard-prune sets; arbitrate within the set by "
        "count confidence/tier, then restricted training-label frequency"
    )
    print("verb_government: lightweight stem/prefix match; forward NP-span scan only")
    print("hash_order: pos/case/gnn for train, then dev, then test")
    print(f"data_hash_sha256: {data_hash}")
    if test_guard.read_count != 1:
        raise RuntimeError(f"expected one test read, saw {test_guard.read_count}")
    print("TEST READ ONCE: CONFIRMED (single aligned final evaluation; all three test task files read once)")
    print("\nhead          full       registers_off  memorizer   register_marginal")
    for head in HEAD_TARGETS:
        full = scores["full"][head]
        off = scores["registers_off"][head]
        mem = scores["memorizer"][head]
        print(f"{head:<13} {full:.6f}   {off:.6f}       {mem:.6f}   {full - off:+.6f}")
    print(f"\nmorph_case majority-null floor (-): {case_scores.null_floor:.6f}")
    print("morph_case per-class full-system accuracy:")
    for label in CASE_LABELS:
        print(
            f"  {label:<3} {case_scores.per_class_accuracy[label]:.6f} "
            f"(n={case_scores.per_class_total[label]})"
        )
    print(
        "government_heuristic: from predicted ADP/VERB/AUX anchors, scan forward across all "
        "DET/PRON/ADJ/NOUN/PROPN/NUM tokens to the first non-NP/non-PUNCT boundary, skipping "
        "punctuation. Strict prepositions and stem/prefix-matched verbs hard-select a case; "
        "two-way prepositions narrow to {Acc,Dat} and arbitrate."
    )
    print(GOVERNMENT_CAVEAT)
    pos_full = scores["full"]["pos"]
    case_full = scores["full"]["morph_case"]
    result = verdict(pos_full, case_full)
    print(f"VERDICT: {result} (pos={pos_full:.6f}, morph_case={case_full:.6f})")


def main() -> None:
    hasher = hashlib.sha256()
    print("Loading GSD train gold (POS, case, GNN); teacher paths are never accessed.", flush=True)
    train = load_split(DATA_ROOT, "train", hasher)
    print("Loading GSD dev for alignment validation only; fixed defaults are unchanged.", flush=True)
    dev = load_split(DATA_ROOT, "dev", hasher)
    print(f"Aligned sentences: train={len(train)} dev={len(dev)}", flush=True)
    del dev

    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    student = GermanR1Student(registers)
    student.fit(train)
    del train

    guard = TestReadGuard()
    print("FINAL EVAL: reading GSD test now (first and only test read).", flush=True)
    test = load_test_once(DATA_ROOT, hasher, guard)
    scores, case_scores = evaluate(student, test)
    print_scoreboard(scores, case_scores, hasher.hexdigest(), guard)


if __name__ == "__main__":
    main()
