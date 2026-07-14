import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.german_r1_codex import (  # noqa: E402
    CASE_LABELS,
    DATA_ROOT,
    Candidate,
    GermanR1Student,
    RegisterLayer,
    RegisterProposal,
    Sentence,
    accuracy,
    align_task_records,
    article_candidate_sets,
    fit_count_table,
    invert_article_table,
    narrow_and_arbitrate,
    parse_task_record,
    predict_table_or_fallback,
    score_case,
    verdict,
)


def test_task_loader_keeps_preexpanded_tokens_aligned() -> None:
    tokens = ["Wir", "sind", "in", "dem", "Haus", "."]

    def record(target: str, labels: list[str]) -> str:
        return json.dumps(
            {
                "sent_id": "fixture-s1",
                "text": "Wir sind im Haus.",
                "tokens": tokens,
                "targets": {target: labels},
            }
        )

    pos = parse_task_record(record("upos", ["PRON", "AUX", "ADP", "DET", "NOUN", "PUNCT"]), "upos")
    case = parse_task_record(record("case", ["Nom", "-", "-", "Dat", "Dat", "-"]), "case")
    gnn = parse_task_record(
        record("gnn", ["-|Plur", "-|-", "-|-", "Neut|Sing", "Neut|Sing", "-|-"]),
        "gnn",
    )
    combined = align_task_records([pos], [case], [gnn])

    assert combined[0].tokens == tuple(tokens)
    assert combined[0].tokens[2:4] == ("in", "dem")
    assert all(len(values) == len(tokens) for values in combined[0].targets.values())


def test_article_inverse_preserves_unique_and_ambiguous_case_sets() -> None:
    fixture = {
        "definite": {
            "Nom": {"Masc": "der", "Fem": "die", "Neut": "das", "Plur": "die"},
            "Acc": {"Masc": "den", "Fem": "die", "Neut": "das", "Plur": "die"},
            "Dat": {"Masc": "dem", "Fem": "der", "Neut": "dem", "Plur": "den"},
            "Gen": {"Masc": "des", "Fem": "der", "Neut": "des", "Plur": "der"},
        }
    }
    inverse = invert_article_table(fixture)
    dem_cases, _ = article_candidate_sets(inverse, "dem")
    die_cases, _ = article_candidate_sets(inverse, "die")

    assert dem_cases == {"Dat"}
    assert {"Nom", "Acc"} <= die_cases
    assert len(die_cases) > 1


def test_majority_table_support_determinism_and_fallback() -> None:
    table = fit_count_table(
        [("haus", "NOUN"), ("haus", "NOUN"), ("haus", "VERB"), ("selten", "ADJ")],
        minsupp=2,
        mindet=0.6,
    )

    haus = table.lookup("haus")
    assert haus is not None
    assert haus.label == "NOUN"
    assert haus.count == 2 and haus.total == 3
    assert haus.confidence == pytest.approx(2 / 5)
    assert table.lookup("selten") is None
    assert predict_table_or_fallback(table, "selten", "NOUN") == "NOUN"


def test_narrow_and_arbitrate_uses_count_survivor_inside_syncretic_set() -> None:
    candidates = [
        Candidate("Acc", 0.4, "form"),
        Candidate("Dat", 0.9, "context"),
    ]

    prediction = narrow_and_arbitrate(
        candidates,
        [RegisterProposal(frozenset({"Nom", "Acc"}))],
        {"Nom": 20, "Acc": 10, "Dat": 30},
        "-",
    )

    assert prediction == "Acc"


def test_narrow_and_arbitrate_uses_restricted_frequency_fallback() -> None:
    prediction = narrow_and_arbitrate(
        [Candidate("Nom", 0.9, "context")],
        [RegisterProposal(frozenset({"Acc", "Dat"}))],
        {"Nom": 100, "Acc": 4, "Dat": 7},
        "-",
    )

    assert prediction == "Dat"


def test_narrow_and_arbitrate_ignores_empty_constraint_intersection() -> None:
    prediction = narrow_and_arbitrate(
        [Candidate("Nom", 0.8, "form"), Candidate("Dat", 0.9, "context")],
        [
            RegisterProposal(frozenset({"Nom", "Acc"})),
            RegisterProposal(frozenset({"Dat"})),
        ],
        {"Nom": 10, "Acc": 5, "Dat": 8},
        "-",
    )

    assert prediction == "Dat"


def test_morph_gnn_ambiguous_register_changes_only_full_arm() -> None:
    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    student = GermanR1Student(registers)
    train = [
        Sentence(
            sent_id="train-1",
            text="die",
            tokens=("die",),
            targets={"pos": ("DET",), "morph_case": ("Nom",), "morph_gnn": ("-|Sing",)},
        )
    ]
    student.fit(train)

    predictions = student.predict_sentence(train[0])

    assert any(
        len(proposal.labels) > 1
        for proposal in registers.direct_gnn_candidates("die", "DET")
    )
    assert predictions["registers_off"]["morph_gnn"] == ["-|Sing"]
    assert predictions["memorizer"]["morph_gnn"] == ["-|Sing"]
    assert predictions["full"]["morph_gnn"] == ["Fem|Sing"]


def test_government_scans_full_np_span_and_stops_at_boundary() -> None:
    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    tokens = ("mit", "dem", "sehr", "alten", "großen", "Haus", "steht", "Garten")
    predicted_pos = ("ADP", "DET", "PUNCT", "ADJ", "ADJ", "NOUN", "VERB", "NOUN")

    proposals = registers.government_candidates(tokens, predicted_pos)

    assert set(proposals) == {1, 3, 4, 5}
    assert proposals[5] == [RegisterProposal(frozenset({"Dat"}))]
    assert 7 not in proposals


def test_two_way_preposition_narrows_and_arbitrates() -> None:
    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    proposals = registers.government_candidates(
        ("in", "die", "Stadt", "fahren"),
        ("ADP", "DET", "NOUN", "VERB"),
    )

    assert proposals[1] == [RegisterProposal(frozenset({"Acc", "Dat"}))]
    assert (
        narrow_and_arbitrate(
            [Candidate("Nom", 0.9, "context"), Candidate("Acc", 0.4, "form")],
            proposals[1],
            {"Acc": 2, "Dat": 8, "Nom": 20},
            "-",
        )
        == "Acc"
    )


def test_verb_government_matches_inflected_form_and_scans_forward() -> None:
    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    assert "absolvieren" in registers.verb_government

    proposals = registers.government_candidates(
        ("absolviert", "die", "schwere", "Prüfung", "heute"),
        ("VERB", "DET", "ADJ", "NOUN", "ADV"),
    )

    expected = [RegisterProposal(frozenset({"Acc"}))]
    assert proposals == {1: expected, 2: expected, 3: expected}


def test_scorer_accuracy_null_floor_and_case_breakdown() -> None:
    gold = ["Nom", "Acc", "Dat", "Gen", "-", "-", "Nom"]
    predictions = ["Nom", "Nom", "Dat", "Acc", "-", "Acc", "Acc"]

    assert accuracy(predictions, gold) == pytest.approx(3 / 7)
    scores = score_case(predictions, gold)
    assert scores.null_floor == pytest.approx(2 / 7)
    assert scores.per_class_accuracy == {
        "Nom": pytest.approx(1 / 2),
        "Acc": pytest.approx(0.0),
        "Dat": pytest.approx(1.0),
        "Gen": pytest.approx(0.0),
        "-": pytest.approx(1 / 2),
    }
    assert sum(scores.per_class_total.values()) == len(gold)
    assert tuple(scores.per_class_total) == CASE_LABELS


@pytest.mark.parametrize(
    ("pos", "case", "expected"),
    [(0.96, 0.92, "FIRES"), (0.80, 0.99, "HALTED"), (0.90, 0.85, "IN-BETWEEN")],
)
def test_verdict_logic(pos: float, case: float, expected: str) -> None:
    assert verdict(pos, case) == expected
