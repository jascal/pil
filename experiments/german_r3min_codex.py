"""German R3-min: gold-attachment upper bounds for case and AUX/VERB.

This campaign deliberately supplies gold ``head_offset`` edges.  It measures an
oracle ceiling, not a serve-honest system: no attachment predictor is trained.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.german_r1_codex import (  # noqa: E402
    CASE_LABELS,
    DATA_ROOT,
    CaseScores,
    GermanR1Student,
    RegisterLayer,
    RegisterProposal,
    Sentence,
    TaskRecord,
    accuracy,
    load_split,
    narrow_and_arbitrate,
    score_case,
)
from experiments.german_r2_codex import (  # noqa: E402
    CLAUSE_BOUNDARIES,
    GermanR2TaskStudent,
    SurfacePosPredictor,
    TargetRecord,
    _span,
    is_infinitive_like,
    is_participle_like,
    load_pos_train,
    load_task_split,
    score_indexed,
    suffix_shape,
)

CASE_THRESHOLD = 0.88
CASE_BAR = 0.90
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "german_r3min_codex.json"
TEST_TASKS = frozenset({"pos", "morph_case", "head_deprel", "aux_verb"})
PARSE_FREE_NP_POS = frozenset({"DET", "PRON", "ADJ", "NOUN", "PROPN", "NUM"})
HASH_ORDER = (
    "pos/train",
    "morph_case/train",
    "morph_gnn/train",
    "pos/train (R2 SurfacePosPredictor read)",
    "aux_verb/train",
    "aux_verb/dev",
    "pos/test",
    "morph_case/test",
    "morph_gnn/test",
    "head_deprel/test",
    "aux_verb/test",
)


class TestReadGuard:
    """Claim each in-scope GSD test task once across both oracle arms."""

    __test__ = False

    def __init__(self) -> None:
        self.claimed: set[str] = set()

    def claim(self, task: str) -> None:
        if task not in TEST_TASKS:
            raise ValueError(f"unknown guarded test task: {task}")
        if task in self.claimed:
            raise RuntimeError(f"GSD {task} test split may be read only once")
        self.claimed.add(task)

    @property
    def confirmed(self) -> bool:
        return self.claimed == TEST_TASKS

    def assert_complete(self) -> None:
        if not self.confirmed:
            raise RuntimeError(
                "expected exactly one test read for pos, morph_case, head_deprel, "
                f"and aux_verb; claimed={sorted(self.claimed)}"
            )


@dataclass(frozen=True)
class _HashingReadPath:
    """Path-shaped adapter that hashes R2's otherwise uninstrumented POS read."""

    path: Path
    hasher: Any

    def read_bytes(self) -> bytes:
        raw = self.path.read_bytes()
        self.hasher.update(raw)
        return raw


@dataclass(frozen=True)
class CaseOracleDecision:
    label: str
    source: str


@dataclass(frozen=True)
class HeadDeprelRecord:
    sent_id: str
    text: str
    tokens: tuple[str, ...]
    head_offset: tuple[int, ...]
    deprel: tuple[str, ...]

    @property
    def targets(self) -> tuple[int, ...]:
        """Compatibility alias for the pre-existing attachment-only arms."""
        return self.head_offset


@dataclass(frozen=True)
class FullCasePass1Decision:
    label: str | None
    rule: int | None
    rule5_attempted: bool = False
    rule5_hit: bool = False


@dataclass(frozen=True)
class FullCaseSentenceResult:
    predictions: tuple[str, ...]
    fired_rules: tuple[int | None, ...]
    rule5_attempts: int
    rule5_hits: int


FULL_CASE_RULE_LABELS = {
    1: "subject -> Nom",
    2: "reverse-child copula -> Nom",
    3: "reverse-child case marker -> preposition register",
    4: "iobj -> Dat",
    5: "obj/obl verb-government register",
    6: "obj -> Acc default",
    7: "case-child-free nmod -> Gen",
    8: "agreement dependent inherits head pass-1 case",
    9: "final R1 baseline fallback",
}


def head_deprel_by_sent(
    records: Sequence[TaskRecord | HeadDeprelRecord],
) -> dict[str, TaskRecord | HeadDeprelRecord]:
    """Index head records by sentence ID, rejecting ambiguous duplicate IDs."""
    indexed: dict[str, TaskRecord | HeadDeprelRecord] = {}
    for record in records:
        if record.sent_id in indexed:
            raise ValueError(f"duplicate head_deprel sent_id: {record.sent_id}")
        indexed[record.sent_id] = record
    return indexed


def aligned_head_record(
    sentence: Sentence | TargetRecord,
    indexed: Mapping[str, TaskRecord | HeadDeprelRecord],
) -> TaskRecord | HeadDeprelRecord:
    """Join by sent_id and validate tokens before exposing attachment offsets."""
    record = indexed.get(sentence.sent_id)
    if record is None:
        raise ValueError(f"{sentence.sent_id}: missing head_deprel record")
    if record.tokens != sentence.tokens:
        raise ValueError(f"{sentence.sent_id}: head_deprel token mismatch")
    return record


def governor_index(index: int, head_offset: Sequence[int]) -> int | None:
    """Resolve a relative head, representing roots and invalid heads as None."""
    if index < 0 or index >= len(head_offset):
        raise IndexError(f"token index outside head_offset: {index}")
    governor = index + int(head_offset[index])
    if governor == index or governor < 0 or governor >= len(head_offset):
        return None
    return governor


def children(index: int, head_offset: Sequence[int]) -> list[int]:
    """Return gold children of an index, in surface order."""
    return [
        child
        for child in range(len(head_offset))
        if child != index and governor_index(child, head_offset) == index
    ]


def neighbors(index: int, head_offset: Sequence[int]) -> list[int]:
    """Return gold children followed by the gold parent, when one exists."""
    linked = children(index, head_offset)
    parent = governor_index(index, head_offset)
    if parent is not None:
        linked.append(parent)
    return linked


def oracle_case_decision(
    student: GermanR1Student,
    tokens: Sequence[str],
    pos_full: Sequence[str],
    head_offset: Sequence[int],
    index: int,
) -> CaseOracleDecision | None:
    """Derive one case from attachment/register evidence, without accepting gold case."""
    if not (len(tokens) == len(pos_full) == len(head_offset)):
        raise ValueError("tokens, predicted POS, and head_offset must align")
    governor = governor_index(index, head_offset)
    if governor is None:
        return None

    return _case_decision_from_anchor(student, tokens, pos_full, index, governor)


def _case_decision_from_anchor(
    student: GermanR1Student,
    tokens: Sequence[str],
    pos_full: Sequence[str],
    index: int,
    anchor: int,
) -> CaseOracleDecision | None:
    """Apply the shared case-register decision once an anchor is supplied."""
    if len(tokens) != len(pos_full):
        raise ValueError("tokens and predicted POS must align")

    governor_pos = pos_full[anchor]
    if governor_pos == "ADP":
        prep_cases = student.registers.preposition_cases.get(tokens[anchor].lower())
        if prep_cases is None:
            return None
        if len(prep_cases) == 1:
            return CaseOracleDecision(next(iter(prep_cases)), "preposition_one_way")
        label = _arbitrate_case_constraint(
            student, tokens, pos_full, index, prep_cases
        )
        return CaseOracleDecision(label, "preposition_two_way")

    if governor_pos in {"VERB", "AUX"}:
        verb_case = student.registers._verb_government_case(tokens[anchor])
        if verb_case is not None:
            return CaseOracleDecision(verb_case, "verb_government")
    return None


def _arbitrate_case_constraint(
    student: GermanR1Student,
    tokens: Sequence[str],
    pos_full: Sequence[str],
    index: int,
    allowed: frozenset[str],
) -> str:
    """Apply R1 count/declension arbitration inside one case constraint."""
    form = tokens[index].lower()
    previous_form = tokens[index - 1].lower() if index else "<BOS>"
    count_candidates = student._count_candidates("morph_case", form, previous_form)
    declension_candidates = student.registers.direct_case_candidates(
        form, pos_full[index]
    )
    return narrow_and_arbitrate(
        count_candidates,
        declension_candidates + [RegisterProposal(frozenset(allowed))],
        student.label_frequencies["morph_case"],
        student.fallbacks["morph_case"],
    )


def verb_aware_forward_np_span(
    anchor: int,
    pos_full: Sequence[str],
) -> list[int]:
    """Scan a forward NP span, explicitly hard-stopping at VERB/AUX."""
    reached: list[int] = []
    for index in range(anchor + 1, len(pos_full)):
        predicted_pos = pos_full[index]
        if predicted_pos in {"VERB", "AUX"}:
            break
        if predicted_pos == "PUNCT":
            continue
        if predicted_pos not in PARSE_FREE_NP_POS:
            break
        reached.append(index)
    return reached


def parse_free_case_decisions(
    student: GermanR1Student,
    tokens: Sequence[str],
    pos_full: Sequence[str],
) -> dict[int, CaseOracleDecision]:
    """Project register case through local verb-aware spans, with no attachment."""
    if len(tokens) != len(pos_full):
        raise ValueError("tokens and predicted POS must align")
    decisions: dict[int, CaseOracleDecision] = {}
    for anchor, predicted_pos in enumerate(pos_full):
        if predicted_pos == "ADP":
            if student.registers.preposition_cases.get(tokens[anchor].lower()) is None:
                continue
        elif predicted_pos in {"VERB", "AUX"}:
            if student.registers._verb_government_case(tokens[anchor]) is None:
                continue
        else:
            continue
        for index in verb_aware_forward_np_span(anchor, pos_full):
            decision = _case_decision_from_anchor(
                student, tokens, pos_full, index, anchor
            )
            if decision is not None:
                decisions[index] = decision
    return decisions


def full_case_pass1_decision(
    student: GermanR1Student,
    tokens: Sequence[str],
    pos_full: Sequence[str],
    gold_pos: Sequence[str],
    head_offset: Sequence[int],
    deprel: Sequence[str],
    index: int,
) -> FullCasePass1Decision:
    """Apply fixed full-oracle rules 1-7 to one token, never rule 8."""
    if not (
        len(tokens)
        == len(pos_full)
        == len(gold_pos)
        == len(head_offset)
        == len(deprel)
    ):
        raise ValueError("full case oracle inputs must align")
    relation = deprel[index]
    # csubj:pass is the clausal-subject passive analogue of nsubj:pass.
    if relation in {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}:
        return FullCasePass1Decision("Nom", 1)

    reverse_children = children(index, head_offset)
    if any(deprel[child] == "cop" for child in reverse_children):
        return FullCasePass1Decision("Nom", 2)

    case_children = [
        child for child in reverse_children if deprel[child] == "case"
    ]
    if case_children:
        marker = case_children[0]
        prep_cases = student.registers.preposition_cases.get(tokens[marker].lower())
        if prep_cases is None:
            return FullCasePass1Decision(None, None)
        if len(prep_cases) == 1:
            return FullCasePass1Decision(next(iter(prep_cases)), 3)
        return FullCasePass1Decision(
            _arbitrate_case_constraint(
                student, tokens, pos_full, index, prep_cases
            ),
            3,
        )

    if relation == "iobj":
        return FullCasePass1Decision("Dat", 4)

    obl_family = {"obl", "obl:arg", "obl:agent", "obl:tmod"}
    if relation == "obj" or relation in obl_family:
        governor = governor_index(index, head_offset)
        if governor is not None and gold_pos[governor] in {"VERB", "AUX"}:
            verb_case = student.registers._verb_government_case(tokens[governor])
            if verb_case is not None:
                return FullCasePass1Decision(
                    verb_case, 5, rule5_attempted=True, rule5_hit=True
                )
            if relation == "obj":
                return FullCasePass1Decision(
                    "Acc", 6, rule5_attempted=True, rule5_hit=False
                )
            return FullCasePass1Decision(
                None, None, rule5_attempted=True, rule5_hit=False
            )
        if relation == "obj":
            return FullCasePass1Decision("Acc", 6)
        return FullCasePass1Decision(None, None)

    if relation == "nmod":
        return FullCasePass1Decision("Gen", 7)
    return FullCasePass1Decision(None, None)


def full_oracle_case_sentence(
    student: GermanR1Student,
    tokens: Sequence[str],
    baseline: Sequence[str],
    pos_full: Sequence[str],
    gold_pos: Sequence[str],
    head_offset: Sequence[int],
    deprel: Sequence[str],
    eligible_indices: Sequence[int],
) -> FullCaseSentenceResult:
    """Run rules 1-7, then deterministic rule-8 inheritance, over a baseline copy."""
    if len(tokens) != len(baseline):
        raise ValueError("tokens and baseline must align")
    eligible = set(eligible_indices)
    pass1 = list(baseline)
    fired_rules: list[int | None] = [None] * len(tokens)
    rule5_attempts = 0
    rule5_hits = 0

    for index in sorted(eligible):
        decision = full_case_pass1_decision(
            student,
            tokens,
            pos_full,
            gold_pos,
            head_offset,
            deprel,
            index,
        )
        rule5_attempts += decision.rule5_attempted
        rule5_hits += decision.rule5_hit
        if decision.label is not None:
            pass1[index] = decision.label
            fired_rules[index] = decision.rule

    predictions = list(pass1)
    agreement_relations = {"det", "det:poss", "amod", "nummod"}
    for index in sorted(eligible):
        if deprel[index] in agreement_relations:
            governor = governor_index(index, head_offset)
            if governor is not None:
                predictions[index] = pass1[governor]
                fired_rules[index] = 8
            else:
                fired_rules[index] = 9
        elif fired_rules[index] is None:
            fired_rules[index] = 9

    return FullCaseSentenceResult(
        tuple(predictions),
        tuple(fired_rules),
        rule5_attempts,
        rule5_hits,
    )


def has_zu_infinitive_marker(
    tokens: Sequence[str],
    infinitive: int,
    head_offset: Sequence[int],
) -> bool:
    """Recognize surface ``zu``/``zur`` immediately before or attached to an infinitive."""
    return any(
        marker != infinitive
        and tokens[marker].lower() in {"zu", "zur"}
        and (
            marker == infinitive - 1
            or governor_index(marker, head_offset) == infinitive
        )
        for marker in range(len(tokens))
    )


def surface_participle_or_infinitive_hit(
    tokens: Sequence[str],
    index: int,
    head_offset: Sequence[int],
    pos: SurfacePosPredictor,
) -> bool:
    """Classify one linked token using only its surface form and train-only POS."""
    if is_participle_like(tokens[index], pos):
        return True
    return is_infinitive_like(tokens[index], pos) and has_zu_infinitive_marker(
        tokens, index, head_offset
    )


def aux_oracle_fires(
    tokens: Sequence[str],
    index: int,
    head_offset: Sequence[int],
    pos: SurfacePosPredictor,
) -> bool:
    """Whether a gold-linked parent/child supplies positive surface AUX evidence."""
    if len(tokens) != len(head_offset):
        raise ValueError("tokens and head_offset must align")
    return any(
        surface_participle_or_infinitive_hit(tokens, linked, head_offset, pos)
        for linked in neighbors(index, head_offset)
    )


def oracle_aux_prediction(
    tokens: Sequence[str],
    index: int,
    head_offset: Sequence[int],
    pos: SurfacePosPredictor,
    baseline_predictor: Callable[[Sequence[str], int], str],
) -> tuple[str, bool]:
    """Override toward AUX only on positive oracle-linked surface evidence."""
    fired = aux_oracle_fires(tokens, index, head_offset, pos)
    if fired:
        return "AUX", True
    return baseline_predictor(tokens, index), False


def parse_free_clause_aux_fires(
    tokens: Sequence[str],
    index: int,
    pos: SurfacePosPredictor,
) -> bool:
    """Search both directions inside one surface-delimited clause."""
    left, right = _span(tokens, index, CLAUSE_BOUNDARIES)
    for position in range(left, right):
        if position == index:
            continue
        token = tokens[position]
        shape = suffix_shape(token)
        if shape != "other" and is_participle_like(token, pos):
            return True
        if (
            shape != "other"
            and is_infinitive_like(token, pos)
            and position > left
            and tokens[position - 1].lower() in {"zu", "zur"}
        ):
            return True
    return False


def parse_free_clause_aux_prediction(
    tokens: Sequence[str],
    index: int,
    pos: SurfacePosPredictor,
    baseline_predictor: Callable[[Sequence[str], int], str],
) -> tuple[str, bool]:
    """Override toward AUX on clause-local surface evidence, else use baseline."""
    fired = parse_free_clause_aux_fires(tokens, index, pos)
    if fired:
        return "AUX", True
    return baseline_predictor(tokens, index), False


def case_verdict(oracle_accuracy: float) -> str:
    return "CONFIRMED" if oracle_accuracy >= CASE_THRESHOLD else "INCOMPLETE"


def _case_scores_json(scores: CaseScores) -> dict[str, dict[str, float | int]]:
    return {
        label: {
            "accuracy": scores.per_class_accuracy[label],
            "n": scores.per_class_total[label],
        }
        for label in CASE_LABELS
    }


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _load_case_test_once(
    hasher: Any,
    guard: TestReadGuard,
) -> list[Sentence]:
    guard.claim("pos")
    guard.claim("morph_case")
    return load_split(DATA_ROOT, "test", hasher)


def parse_head_deprel_record(
    raw: str | bytes | Mapping[str, Any],
) -> HeadDeprelRecord:
    """Parse both attachment targets from one already-read JSON record."""
    obj = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    tokens = tuple(str(token) for token in obj["tokens"])
    head_offset = tuple(int(offset) for offset in obj["targets"]["head_offset"])
    deprel = tuple(str(label) for label in obj["targets"]["deprel"])
    if not (len(tokens) == len(head_offset) == len(deprel)):
        raise ValueError(f"{obj.get('sent_id', '<unknown>')}: head targets misalign")
    return HeadDeprelRecord(
        str(obj["sent_id"]),
        str(obj["text"]),
        tokens,
        head_offset,
        deprel,
    )


def _load_head_test_once(
    hasher: Any,
    guard: TestReadGuard,
) -> list[HeadDeprelRecord]:
    guard.claim("head_deprel")
    raw = (DATA_ROOT / "tasks" / "head_deprel" / "test.jsonl").read_bytes()
    hasher.update(raw)
    return [parse_head_deprel_record(line) for line in raw.splitlines() if line.strip()]


def evaluate_case_arm(
    student: GermanR1Student,
    test: Sequence[Sentence],
    head_index: Mapping[str, TaskRecord | HeadDeprelRecord],
) -> dict[str, Any]:
    baseline_all: list[str] = []
    heuristic_all: list[str] = []
    oracle_all: list[str] = []
    full_oracle_all: list[str] = []
    gold_all: list[str] = []
    case_bearing = 0
    oracle_fired = 0
    heuristic_fired = 0
    oracle_source_counts = {
        "preposition_one_way": 0,
        "preposition_two_way": 0,
        "verb_government": 0,
    }
    heuristic_source_counts = {
        "preposition_one_way": 0,
        "preposition_two_way": 0,
        "verb_government": 0,
    }
    full_rule_counts = {rule: 0 for rule in FULL_CASE_RULE_LABELS}
    full_rule5_attempts = 0
    full_rule5_hits = 0
    full_rule5_correct = 0

    for sentence in test:
        head = aligned_head_record(sentence, head_index)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError(f"{sentence.sent_id}: deprel targets were not loaded")
        result = student.predict_sentence(sentence)
        baseline = list(result["full"]["morph_case"])
        pos_full = result["full"]["pos"]
        oracle = list(baseline)
        heuristic = list(baseline)
        gold = sentence.targets["morph_case"]
        eligible_indices = [
            index for index, gold_label in enumerate(gold) if gold_label != "-"
        ]

        full_result = full_oracle_case_sentence(
            student,
            sentence.tokens,
            baseline,
            pos_full,
            sentence.targets["pos"],
            head.head_offset,
            head.deprel,
            eligible_indices,
        )
        full_oracle = list(full_result.predictions)
        full_rule5_attempts += full_result.rule5_attempts
        full_rule5_hits += full_result.rule5_hits
        for index in eligible_indices:
            rule = full_result.fired_rules[index]
            if rule is None:
                raise RuntimeError(f"{sentence.sent_id}:{index}: full rule unaccounted")
            full_rule_counts[rule] += 1
            if rule == 5:
                full_rule5_correct += full_oracle[index] == gold[index]

        heuristic_decisions = parse_free_case_decisions(
            student, sentence.tokens, pos_full
        )
        for index, decision in heuristic_decisions.items():
            heuristic[index] = decision.label

        for index, gold_label in enumerate(gold):
            if gold_label == "-":
                continue
            case_bearing += 1
            heuristic_decision = heuristic_decisions.get(index)
            if heuristic_decision is not None:
                heuristic_fired += 1
                heuristic_source_counts[heuristic_decision.source] += 1
            decision = oracle_case_decision(
                student, sentence.tokens, pos_full, head.targets, index
            )
            if decision is not None:
                oracle[index] = decision.label
                oracle_fired += 1
                oracle_source_counts[decision.source] += 1

        baseline_all.extend(baseline)
        heuristic_all.extend(heuristic)
        oracle_all.extend(oracle)
        full_oracle_all.extend(full_oracle)
        gold_all.extend(gold)

    baseline_accuracy = accuracy(baseline_all, gold_all)
    heuristic_accuracy = accuracy(heuristic_all, gold_all)
    oracle_accuracy = accuracy(oracle_all, gold_all)
    full_oracle_accuracy = accuracy(full_oracle_all, gold_all)
    baseline_classes = score_case(baseline_all, gold_all)
    heuristic_classes = score_case(heuristic_all, gold_all)
    oracle_classes = score_case(oracle_all, gold_all)
    full_oracle_classes = score_case(full_oracle_all, gold_all)
    partial_verdict = case_verdict(oracle_accuracy)
    verdict = case_verdict(full_oracle_accuracy)
    return {
        "baseline_accuracy": baseline_accuracy,
        "heuristic_accuracy": heuristic_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "full_oracle_accuracy": full_oracle_accuracy,
        "delta": oracle_accuracy - baseline_accuracy,
        "heuristic_delta_vs_baseline": heuristic_accuracy - baseline_accuracy,
        "oracle_delta_vs_heuristic": oracle_accuracy - heuristic_accuracy,
        "oracle_total_gap_vs_baseline": oracle_accuracy - baseline_accuracy,
        "full_oracle_delta_vs_baseline": full_oracle_accuracy - baseline_accuracy,
        "full_oracle_delta_vs_partial_oracle": (
            full_oracle_accuracy - oracle_accuracy
        ),
        "per_class": {
            "baseline": _case_scores_json(baseline_classes),
            "heuristic": _case_scores_json(heuristic_classes),
            "oracle": _case_scores_json(oracle_classes),
            "full_oracle": _case_scores_json(full_oracle_classes),
        },
        "case_bearing_n": case_bearing,
        "oracle_branch_fired_n": oracle_fired,
        "oracle_branch_fired_fraction": _fraction(oracle_fired, case_bearing),
        "baseline_fallback_n": case_bearing - oracle_fired,
        "baseline_fallback_fraction": _fraction(
            case_bearing - oracle_fired, case_bearing
        ),
        "oracle_source_counts": oracle_source_counts,
        "heuristic_branch_fired_n": heuristic_fired,
        "heuristic_branch_fired_fraction": _fraction(
            heuristic_fired, case_bearing
        ),
        "heuristic_baseline_fallback_n": case_bearing - heuristic_fired,
        "heuristic_baseline_fallback_fraction": _fraction(
            case_bearing - heuristic_fired, case_bearing
        ),
        "heuristic_source_counts": heuristic_source_counts,
        "ladder": {
            "rung1_parse_free_baseline": baseline_accuracy,
            "rung2_parse_free_verb_aware_span": heuristic_accuracy,
            "rung3_gold_governor_oracle": oracle_accuracy,
            "rung4_full_deprel_case_oracle": full_oracle_accuracy,
            "rung2_minus_rung1": heuristic_accuracy - baseline_accuracy,
            "rung3_minus_rung2": oracle_accuracy - heuristic_accuracy,
            "rung3_minus_rung1_total_gap": oracle_accuracy - baseline_accuracy,
            "rung4_minus_rung3": full_oracle_accuracy - oracle_accuracy,
            "rung4_minus_rung1_total_gap": (
                full_oracle_accuracy - baseline_accuracy
            ),
        },
        "full_oracle_rule_coverage": {
            str(rule): {
                "label": label,
                "n": full_rule_counts[rule],
                "fraction": _fraction(full_rule_counts[rule], case_bearing),
            }
            for rule, label in FULL_CASE_RULE_LABELS.items()
        },
        "full_oracle_rule5_register": {
            "attempts": full_rule5_attempts,
            "hits": full_rule5_hits,
            "empty": full_rule5_attempts - full_rule5_hits,
            "hit_rate": _fraction(full_rule5_hits, full_rule5_attempts),
            "correct_hits": full_rule5_correct,
            "precision": _fraction(full_rule5_correct, full_rule5_hits),
            "note": (
                "Verb-government register precision is a known second limiter; "
                "the deterministic rule is applied as registered and scored here."
            ),
        },
        "threshold": CASE_THRESHOLD,
        "target_bar": CASE_BAR,
        "verdict": verdict,
        "partial_oracle_verdict": partial_verdict,
        "shortfall_from_0.90": CASE_BAR - full_oracle_accuracy,
        "partial_oracle_shortfall_from_0.90": CASE_BAR - oracle_accuracy,
        "nom_note": (
            "Nom is often not government-assigned (subject case is structurally "
            "different from object/preposition government), so it is expected to move little."
        ),
    }


def evaluate_aux_arm(
    student: GermanR2TaskStudent,
    pos: SurfacePosPredictor,
    test: Sequence[TargetRecord],
    head_index: Mapping[str, TaskRecord | HeadDeprelRecord],
) -> dict[str, Any]:
    def baseline_predictor(tokens: Sequence[str], index: int) -> str:
        return student.predict(tokens, index, True)

    baseline_accuracy = score_indexed(test, baseline_predictor)

    oracle_predictions: dict[tuple[int, int], tuple[str, bool]] = {}
    heuristic_predictions: dict[tuple[int, int], tuple[str, bool]] = {}
    total = oracle_fired = heuristic_fired = gold_aux = 0
    oracle_fired_gold_aux = heuristic_fired_gold_aux = 0
    for record in test:
        head = aligned_head_record(record, head_index)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError(f"{record.sent_id}: deprel targets were not loaded")
        for index, _, label in record.targets():
            oracle_prediction = oracle_aux_prediction(
                record.tokens,
                index,
                head.targets,
                pos,
                baseline_predictor,
            )
            heuristic_prediction = parse_free_clause_aux_prediction(
                record.tokens,
                index,
                pos,
                baseline_predictor,
            )
            key = (id(record.tokens), index)
            oracle_predictions[key] = oracle_prediction
            heuristic_predictions[key] = heuristic_prediction
            total += 1
            oracle_fired += oracle_prediction[1]
            heuristic_fired += heuristic_prediction[1]
            if label == "AUX":
                gold_aux += 1
                oracle_fired_gold_aux += oracle_prediction[1]
                heuristic_fired_gold_aux += heuristic_prediction[1]

    oracle_accuracy = score_indexed(
        test,
        lambda tokens, index: oracle_predictions[(id(tokens), index)][0],
    )
    heuristic_accuracy = score_indexed(
        test,
        lambda tokens, index: heuristic_predictions[(id(tokens), index)][0],
    )
    oracle_fired_fraction = _fraction(oracle_fired, total)
    oracle_aux_recall = _fraction(oracle_fired_gold_aux, gold_aux)
    oracle_fired_precision = _fraction(oracle_fired_gold_aux, oracle_fired)
    heuristic_fired_fraction = _fraction(heuristic_fired, total)
    heuristic_aux_recall = _fraction(heuristic_fired_gold_aux, gold_aux)
    heuristic_fired_precision = _fraction(
        heuristic_fired_gold_aux, heuristic_fired
    )
    close_to_sufficient = oracle_aux_recall > 0.90
    interpretation = (
        "Gold attachment structure alone is close to sufficient to recover AUX; "
        "this is a weaker result than a genuine morphology-driven rescore."
        if close_to_sufficient
        else "The gold-link surface signal is not close to sufficient by itself to recover AUX."
    )
    return {
        "baseline_accuracy": baseline_accuracy,
        "heuristic_accuracy": heuristic_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "delta": oracle_accuracy - baseline_accuracy,
        "heuristic_delta_vs_baseline": heuristic_accuracy - baseline_accuracy,
        "oracle_delta_vs_heuristic": oracle_accuracy - heuristic_accuracy,
        "oracle_total_gap_vs_baseline": oracle_accuracy - baseline_accuracy,
        "indexed_n": total,
        "ladder": {
            "rung1_parse_free_baseline": baseline_accuracy,
            "rung2_parse_free_clause_search": heuristic_accuracy,
            "rung3_gold_attachment_oracle": oracle_accuracy,
            "rung2_minus_rung1": heuristic_accuracy - baseline_accuracy,
            "rung3_minus_rung2": oracle_accuracy - heuristic_accuracy,
            "rung3_minus_rung1_total_gap": oracle_accuracy - baseline_accuracy,
        },
        "circularity_caveat": {
            "all_ambiguous_oracle_branch_fired_n": oracle_fired,
            "all_ambiguous_oracle_branch_fired_fraction": oracle_fired_fraction,
            "all_ambiguous_baseline_fallback_n": total - oracle_fired,
            "all_ambiguous_baseline_fallback_fraction": _fraction(
                total - oracle_fired, total
            ),
            "gold_aux_n": gold_aux,
            "gold_aux_oracle_branch_fired_n": oracle_fired_gold_aux,
            "gold_aux_oracle_branch_fired_fraction_recall": oracle_aux_recall,
            "oracle_branch_fired_precision_aux": oracle_fired_precision,
            "gold_attachment_close_to_sufficient": close_to_sufficient,
            "interpretation": interpretation,
        },
        "heuristic_circularity_caveat": {
            "all_ambiguous_heuristic_branch_fired_n": heuristic_fired,
            "all_ambiguous_heuristic_branch_fired_fraction": heuristic_fired_fraction,
            "all_ambiguous_baseline_fallback_n": total - heuristic_fired,
            "all_ambiguous_baseline_fallback_fraction": _fraction(
                total - heuristic_fired, total
            ),
            "gold_aux_n": gold_aux,
            "gold_aux_heuristic_branch_fired_n": heuristic_fired_gold_aux,
            "gold_aux_heuristic_branch_fired_fraction_recall": heuristic_aux_recall,
            "heuristic_branch_fired_precision_aux": heuristic_fired_precision,
        },
    }


def print_scoreboard(scoreboard: Mapping[str, Any], guard: TestReadGuard) -> None:
    case = scoreboard["morph_case"]
    aux = scoreboard["aux_verb"]
    caveat = aux["circularity_caveat"]
    heuristic_caveat = aux["heuristic_circularity_caveat"]
    full_rule5 = case["full_oracle_rule5_register"]

    print("\nGERMAN R3-MIN CODEX FINAL SCOREBOARD")
    print(
        "ORACLE/UPPER-BOUND measurement: GOLD head_offset attachment is supplied; "
        "the full case arm additionally uses GOLD deprel and governor POS. These are "
        "NOT serve-honest numbers and no attachment model is built."
    )
    print("hash_order: " + ", ".join(HASH_ORDER))
    print(f"data_hash_sha256: {scoreboard['data_hash_sha256']}")
    guard.assert_complete()
    print("TEST READ ONCE: CONFIRMED")
    print(
        "parse-free heuristic arms reuse the same in-memory test records; "
        "no additional test files are read."
    )

    print(
        "\nCASE LADDER — baseline -> arm (c) heuristic -> arm (a) partial oracle "
        "-> full deprel oracle"
    )
    print(f"rung1 parse-free baseline:           {case['baseline_accuracy']:.6f}")
    print(
        f"rung2 verb-aware span heuristic:     {case['heuristic_accuracy']:.6f} "
        f"(rung2-rung1={case['heuristic_delta_vs_baseline']:+.6f})"
    )
    print(
        f"rung3 gold-governor partial oracle:  {case['oracle_accuracy']:.6f} "
        f"(rung3-rung2={case['oracle_delta_vs_heuristic']:+.6f})"
    )
    print(f"total gap rung3-rung1:               {case['delta']:+.6f}")
    print(
        f"rung4 full deprel->case oracle:      {case['full_oracle_accuracy']:.6f} "
        f"(rung4-rung3={case['full_oracle_delta_vs_partial_oracle']:+.6f})"
    )
    print(
        "full-oracle delta vs baseline:       "
        f"{case['full_oracle_delta_vs_baseline']:+.6f}"
    )
    print("per-class accuracy (same gold n for all rungs):")
    print("  label  baseline  heuristic  partial   full      n")
    for label in CASE_LABELS:
        baseline = case["per_class"]["baseline"][label]
        heuristic = case["per_class"]["heuristic"][label]
        oracle = case["per_class"]["oracle"][label]
        full_oracle = case["per_class"]["full_oracle"][label]
        print(
            f"  {label:<3}    {baseline['accuracy']:.6f}  "
            f"{heuristic['accuracy']:.6f}   {oracle['accuracy']:.6f}  "
            f"{full_oracle['accuracy']:.6f}  {oracle['n']}"
        )
    print("Arm (c) parse-free verb-aware span coverage:")
    print(
        "  heuristic_branch_fired among case-bearing: "
        f"{case['heuristic_branch_fired_fraction']:.6f} "
        f"({case['heuristic_branch_fired_n']}/{case['case_bearing_n']})"
    )
    print(
        "  baseline_fallback among case-bearing: "
        f"{case['heuristic_baseline_fallback_fraction']:.6f} "
        f"({case['heuristic_baseline_fallback_n']}/{case['case_bearing_n']})"
    )
    print("Arm (a) gold-governor oracle coverage:")
    print(
        "  oracle_branch_fired among case-bearing: "
        f"{case['oracle_branch_fired_fraction']:.6f} "
        f"({case['oracle_branch_fired_n']}/{case['case_bearing_n']})"
    )
    print(
        "  baseline_fallback among case-bearing: "
        f"{case['baseline_fallback_fraction']:.6f} "
        f"({case['baseline_fallback_n']}/{case['case_bearing_n']})"
    )
    print("Full deprel->case oracle coverage (fraction of case-bearing tokens):")
    for rule in range(1, 10):
        coverage = case["full_oracle_rule_coverage"][str(rule)]
        print(
            f"  rule {rule}: {coverage['label']}: "
            f"{coverage['fraction']:.6f} "
            f"({coverage['n']}/{case['case_bearing_n']})"
        )
    print(
        "  rule-5 verb_government attempts: "
        f"hits={full_rule5['hits']} empty={full_rule5['empty']} "
        f"attempts={full_rule5['attempts']} hit_rate={full_rule5['hit_rate']:.6f}"
    )
    print(
        "  rule-5 hit precision: "
        f"{full_rule5['precision']:.6f} "
        f"({full_rule5['correct_hits']}/{full_rule5['hits']})"
    )
    print(f"  {full_rule5['note']}")
    print(case["nom_note"])
    print(
        f"CASE R3-GATING (partial oracle, retained): {case['partial_oracle_verdict']} "
        f"(oracle={case['oracle_accuracy']:.6f}, threshold>=0.880000, "
        f"shortfall_from_0.90={case['partial_oracle_shortfall_from_0.90']:+.6f})"
    )
    print(
        f"CASE R3-GATING: {case['verdict']} "
        f"(full_oracle={case['full_oracle_accuracy']:.6f}, threshold>=0.880000, "
        f"shortfall_from_0.90={case['shortfall_from_0.90']:+.6f})"
    )

    print("\nAUX_VERB LADDER — baseline -> arm (d) heuristic -> arm (b) oracle")
    print(f"rung1 parse-free baseline:           {aux['baseline_accuracy']:.6f}")
    print(
        f"rung2 clause-bounded heuristic:      {aux['heuristic_accuracy']:.6f} "
        f"(rung2-rung1={aux['heuristic_delta_vs_baseline']:+.6f})"
    )
    print(
        f"rung3 gold-attachment oracle:        {aux['oracle_accuracy']:.6f} "
        f"(rung3-rung2={aux['oracle_delta_vs_heuristic']:+.6f})"
    )
    print(f"total gap rung3-rung1:               {aux['delta']:+.6f}")
    print("Arm (d) parse-free clause-search circularity caveat:")
    print(
        "  all ambiguous — heuristic branch fired: "
        f"{heuristic_caveat['all_ambiguous_heuristic_branch_fired_fraction']:.6f} "
        f"({heuristic_caveat['all_ambiguous_heuristic_branch_fired_n']}/"
        f"{aux['indexed_n']}); baseline fallback: "
        f"{heuristic_caveat['all_ambiguous_baseline_fallback_fraction']:.6f} "
        f"({heuristic_caveat['all_ambiguous_baseline_fallback_n']}/"
        f"{aux['indexed_n']})"
    )
    print(
        "  gold AUX — heuristic branch fired recall: "
        f"{heuristic_caveat['gold_aux_heuristic_branch_fired_fraction_recall']:.6f} "
        f"({heuristic_caveat['gold_aux_heuristic_branch_fired_n']}/"
        f"{heuristic_caveat['gold_aux_n']})"
    )
    print(
        "  heuristic branch fired precision for AUX: "
        f"{heuristic_caveat['heuristic_branch_fired_precision_aux']:.6f} "
        f"({heuristic_caveat['gold_aux_heuristic_branch_fired_n']}/"
        f"{heuristic_caveat['all_ambiguous_heuristic_branch_fired_n']})"
    )
    print("Arm (b) gold-attachment oracle circularity caveat:")
    print(
        "  all ambiguous — oracle branch fired: "
        f"{caveat['all_ambiguous_oracle_branch_fired_fraction']:.6f} "
        f"({caveat['all_ambiguous_oracle_branch_fired_n']}/{aux['indexed_n']}); "
        "baseline fallback: "
        f"{caveat['all_ambiguous_baseline_fallback_fraction']:.6f} "
        f"({caveat['all_ambiguous_baseline_fallback_n']}/{aux['indexed_n']})"
    )
    print(
        "  gold AUX — oracle branch fired recall: "
        f"{caveat['gold_aux_oracle_branch_fired_fraction_recall']:.6f} "
        f"({caveat['gold_aux_oracle_branch_fired_n']}/{caveat['gold_aux_n']})"
    )
    print(
        "  oracle branch fired precision for AUX: "
        f"{caveat['oracle_branch_fired_precision_aux']:.6f} "
        f"({caveat['gold_aux_oracle_branch_fired_n']}/"
        f"{caveat['all_ambiguous_oracle_branch_fired_n']})"
    )
    print(f"  {caveat['interpretation']}")


def main() -> None:
    hasher = hashlib.sha256()

    print("FIT: loading R1 POS/case/GNN train in fixed hash order.", flush=True)
    train = load_split(DATA_ROOT, "train", hasher)
    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    case_student = GermanR1Student(registers)
    case_student.fit(train)
    del train

    print("FIT: loading POS train for R2's train-only surface predictor.", flush=True)
    pos = SurfacePosPredictor()
    pos.fit(
        load_pos_train(
            _HashingReadPath(DATA_ROOT / "tasks" / "pos" / "train.jsonl", hasher)
        )
    )
    print("FIT: loading aux_verb train and dev for R2's pre-existing rule gate.", flush=True)
    aux_train = load_task_split(DATA_ROOT, "aux_verb", "train", hasher)
    aux_dev = load_task_split(DATA_ROOT, "aux_verb", "dev", hasher)
    aux_student = GermanR2TaskStudent("aux_verb", pos)
    aux_student.fit(aux_train, aux_dev)
    del aux_train, aux_dev

    guard = TestReadGuard()
    print("FINAL EVAL: reading aligned POS/case/GNN test once.", flush=True)
    case_test = _load_case_test_once(hasher, guard)
    print("FINAL EVAL: reading shared head_deprel test once.", flush=True)
    head_test = _load_head_test_once(hasher, guard)
    head_index = head_deprel_by_sent(head_test)
    print("FINAL EVAL: reading aux_verb test once.", flush=True)
    aux_test = load_task_split(DATA_ROOT, "aux_verb", "test", hasher, guard)

    case_scores = evaluate_case_arm(case_student, case_test, head_index)
    aux_scores = evaluate_aux_arm(aux_student, pos, aux_test, head_index)
    guard.assert_complete()
    scoreboard = {
        "measurement": {
            "oracle_upper_bound": True,
            "gold_attachment": "head_offset",
            "full_case_oracle_gold_fields": [
                "head_offset",
                "deprel",
                "governor_upos",
            ],
            "serve_honest": False,
            "attachment_model_built": False,
            "statement": (
                "ORACLE/UPPER-BOUND measurement with gold attachment; the full case "
                "arm also uses gold deprel and governor POS; not serve-honest."
            ),
        },
        "data_hash_sha256": hasher.hexdigest(),
        "hash_order": list(HASH_ORDER),
        "test_read_once": True,
        "test_tasks_claimed": sorted(guard.claimed),
        "morph_case": case_scores,
        "aux_verb": aux_scores,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_scoreboard(scoreboard, guard)
    print(f"structured_scoreboard: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
