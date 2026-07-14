import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.german_r1_codex import (  # noqa: E402
    DATA_ROOT,
    GermanR1Student,
    RegisterLayer,
    Sentence,
    TaskRecord,
)
from experiments.german_r2_codex import SurfacePosPredictor  # noqa: E402
from experiments.german_r3min_codex import (  # noqa: E402
    aligned_head_record,
    case_verdict,
    full_case_pass1_decision,
    full_oracle_case_sentence,
    governor_index,
    head_deprel_by_sent,
    oracle_aux_prediction,
    oracle_case_decision,
    parse_free_clause_aux_prediction,
    parse_head_deprel_record,
    verb_aware_forward_np_span,
)


def _case_student() -> GermanR1Student:
    student = GermanR1Student(RegisterLayer.from_directory(DATA_ROOT / "registers"))
    student.fit(
        [
            Sentence(
                sent_id="train-case",
                text="für Haus in das",
                tokens=("für", "Haus", "in", "das"),
                targets={
                    "pos": ("ADP", "NOUN", "ADP", "DET"),
                    "morph_case": ("-", "Nom", "-", "Nom"),
                    "morph_gnn": ("-|-", "Neut|Sing", "-|-", "Neut|Sing"),
                },
            )
        ]
    )
    return student


def _surface_pos() -> SurfacePosPredictor:
    pos = SurfacePosPredictor()
    pos.fit(
        [
            {
                "tokens": ["arbeiten", "heute", "hat"],
                "targets": {"upos": ["VERB", "ADV", "AUX"]},
            }
        ]
    )
    return pos


def test_head_records_join_by_sent_id_and_reject_token_mismatch() -> None:
    sentence = Sentence(
        sent_id="fixture-align",
        text="für das Haus",
        tokens=("für", "das", "Haus"),
        targets={
            "pos": ("ADP", "DET", "NOUN"),
            "morph_case": ("-", "Acc", "Acc"),
            "morph_gnn": ("-|-", "Neut|Sing", "Neut|Sing"),
        },
    )
    head = TaskRecord(
        "fixture-align",
        "für das Haus",
        ("für", "das", "Haus"),
        (0, -1, -2),
    )
    indexed = head_deprel_by_sent([head])

    assert aligned_head_record(sentence, indexed) is head

    mismatch = TaskRecord(
        "fixture-align",
        "für ein Haus",
        ("für", "ein", "Haus"),
        (0, -1, -2),
    )
    with pytest.raises(ValueError, match="fixture-align"):
        aligned_head_record(sentence, head_deprel_by_sent([mismatch]))


def test_head_deprel_parser_keeps_offsets_and_relations_from_one_record() -> None:
    record = parse_head_deprel_record(
        {
            "sent_id": "head-both",
            "text": "Das Haus steht",
            "tokens": ["Das", "Haus", "steht"],
            "targets": {
                "head_offset": [1, 1, 0],
                "deprel": ["det", "nsubj", "root"],
            },
        }
    )

    assert record.head_offset == (1, 1, 0)
    assert record.deprel == ("det", "nsubj", "root")


def test_governor_index_resolves_offset_and_treats_root_as_none() -> None:
    head_offset = (2, 1, 0, -1)

    assert governor_index(0, head_offset) == 2
    assert governor_index(3, head_offset) == 2
    assert governor_index(2, head_offset) is None


def test_oracle_case_uses_one_way_preposition_without_gold_case_input() -> None:
    student = _case_student()

    decision = oracle_case_decision(
        student,
        ("für", "Haus"),
        ("ADP", "NOUN"),
        (0, -1),
        1,
    )

    assert "gold" not in inspect.signature(oracle_case_decision).parameters
    assert decision is not None
    assert decision.label == "Acc"
    assert decision.source == "preposition_one_way"


def test_oracle_case_two_way_preposition_uses_arbitration_branch() -> None:
    student = _case_student()

    decision = oracle_case_decision(
        student,
        ("in", "das"),
        ("ADP", "DET"),
        (0, -1),
        1,
    )

    assert decision is not None
    assert decision.label == "Acc"
    assert decision.source == "preposition_two_way"


def test_full_case_subject_resolves_to_nom_before_register_rules() -> None:
    decision = full_case_pass1_decision(
        _case_student(),
        ("Haus",),
        ("NOUN",),
        ("NOUN",),
        (0,),
        ("nsubj",),
        0,
    )

    assert (decision.label, decision.rule) == ("Nom", 1)


def test_full_case_reverse_child_copula_resolves_to_nom() -> None:
    decision = full_case_pass1_decision(
        _case_student(),
        ("Arzt", "ist"),
        ("NOUN", "AUX"),
        ("NOUN", "AUX"),
        (0, -1),
        ("root", "cop"),
        0,
    )

    assert (decision.label, decision.rule) == ("Nom", 2)


def test_full_case_child_preposition_beats_nmod_genitive_rule() -> None:
    decision = full_case_pass1_decision(
        _case_student(),
        ("Haus", "für"),
        ("NOUN", "ADP"),
        ("NOUN", "ADP"),
        (0, -1),
        ("nmod", "case"),
        0,
    )

    assert (decision.label, decision.rule) == ("Acc", 3)


def test_full_case_iobj_resolves_to_dat() -> None:
    decision = full_case_pass1_decision(
        _case_student(),
        ("hilft", "Mann"),
        ("VERB", "NOUN"),
        ("VERB", "NOUN"),
        (0, -1),
        ("root", "iobj"),
        1,
    )

    assert (decision.label, decision.rule) == ("Dat", 4)


def test_full_case_obj_uses_non_acc_verb_government_hit() -> None:
    student = _case_student()
    student.registers.verb_government["helfen"] = "Dat"

    decision = full_case_pass1_decision(
        student,
        ("helfen", "Objekt"),
        ("VERB", "NOUN"),
        ("VERB", "NOUN"),
        (0, -1),
        ("root", "obj"),
        1,
    )

    assert decision.label == "Dat"
    assert decision.rule == 5
    assert decision.rule5_attempted and decision.rule5_hit


def test_full_case_obj_defaults_to_acc_when_verb_register_is_empty() -> None:
    decision = full_case_pass1_decision(
        _case_student(),
        ("unregistriertesverb", "Objekt"),
        ("VERB", "NOUN"),
        ("VERB", "NOUN"),
        (0, -1),
        ("root", "obj"),
        1,
    )

    assert (decision.label, decision.rule) == ("Acc", 6)
    assert decision.rule5_attempted and not decision.rule5_hit


def test_full_case_nmod_without_case_child_resolves_to_gen() -> None:
    decision = full_case_pass1_decision(
        _case_student(),
        ("Dach", "Haus"),
        ("NOUN", "NOUN"),
        ("NOUN", "NOUN"),
        (0, -1),
        ("root", "nmod"),
        1,
    )

    assert (decision.label, decision.rule) == ("Gen", 7)


def test_full_case_dependent_inherits_definite_head_pass1_case() -> None:
    result = full_oracle_case_sentence(
        _case_student(),
        ("Haus", "das"),
        ("Dat", "Acc"),
        ("NOUN", "DET"),
        ("NOUN", "DET"),
        (0, -1),
        ("nsubj", "det"),
        (0, 1),
    )

    assert result.predictions == ("Nom", "Nom")
    assert result.fired_rules == (1, 8)


def test_full_case_dependent_inherits_head_pass1_baseline_fallback() -> None:
    result = full_oracle_case_sentence(
        _case_student(),
        ("Haus", "das"),
        ("Dat", "Acc"),
        ("NOUN", "DET"),
        ("NOUN", "DET"),
        (0, -1),
        ("root", "det"),
        (0, 1),
    )

    assert result.predictions == ("Dat", "Dat")
    assert result.fired_rules == (9, 8)


def test_verb_aware_span_stops_at_verb_in_fronted_v2_pp() -> None:
    tokens = ("In", "den", "Garten", "geht", "er", ".")
    pos_full = ("ADP", "DET", "NOUN", "VERB", "PRON", "PUNCT")

    reached = verb_aware_forward_np_span(0, pos_full)

    assert tokens[1:3] == ("den", "Garten")
    assert reached == [1, 2]
    assert 3 not in reached
    assert 4 not in reached


def test_aux_oracle_fires_for_surface_participle_via_gold_child() -> None:
    prediction = oracle_aux_prediction(
        ("hat", "gearbeitet"),
        0,
        (0, -1),
        _surface_pos(),
        lambda _tokens, _index: "VERB",
    )

    assert prediction == ("AUX", True)


def test_aux_oracle_falls_through_without_surface_hit() -> None:
    prediction = oracle_aux_prediction(
        ("hat", "heute"),
        0,
        (0, -1),
        _surface_pos(),
        lambda _tokens, _index: "VERB",
    )

    assert prediction == ("VERB", False)


def test_aux_oracle_recognizes_adjacent_zu_infinitive() -> None:
    prediction = oracle_aux_prediction(
        ("hat", "zu", "arbeiten"),
        0,
        (0, -1, -2),
        _surface_pos(),
        lambda _tokens, _index: "VERB",
    )

    assert prediction == ("AUX", True)


def test_aux_oracle_recognizes_gold_attached_nonadjacent_zu_infinitive() -> None:
    prediction = oracle_aux_prediction(
        ("hat", "zu", "heute", "arbeiten"),
        0,
        (0, 2, -2, -3),
        _surface_pos(),
        lambda _tokens, _index: "VERB",
    )

    assert prediction == ("AUX", True)


def test_parse_free_clause_search_finds_nonadjacent_participle() -> None:
    prediction = parse_free_clause_aux_prediction(
        ("hat", "das", "Buch", "gearbeitet", "."),
        0,
        _surface_pos(),
        lambda _tokens, _index: "VERB",
    )

    assert prediction == ("AUX", True)


def test_parse_free_clause_search_rejects_participle_across_boundary() -> None:
    prediction = parse_free_clause_aux_prediction(
        ("hat", ",", "er", "hat", "gearbeitet", "."),
        0,
        _surface_pos(),
        lambda _tokens, _index: "VERB",
    )

    assert prediction == ("VERB", False)


@pytest.mark.parametrize(
    ("oracle_accuracy", "expected"),
    [(0.879, "INCOMPLETE"), (0.88, "CONFIRMED"), (0.95, "CONFIRMED")],
)
def test_case_verdict_fixed_threshold(
    oracle_accuracy: float,
    expected: str,
) -> None:
    assert case_verdict(oracle_accuracy) == expected
