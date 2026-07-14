from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.windowed_labeler_codex as module  # noqa: E402
from experiments.german_r1_codex import Sentence, fit_count_table  # noqa: E402
from experiments.german_r3_codex import (  # noqa: E402
    BackoffTables,
    CampaignTestReadGuard,
    _relation_key_levels,
)
from experiments.german_r3min_codex import HeadDeprelRecord  # noqa: E402
from experiments.windowed_labeler_codex import (  # noqa: E402
    WindowedDeprelLabeler,
    apply_preregistered_read,
    load_guarded_test_after_selection,
    score_relation_partition,
    sweep_dev_configurations,
    windowed_relation_key_levels,
)


def _sentence(sent_id: str, tokens: tuple[str, ...]) -> Sentence:
    return Sentence(
        sent_id=sent_id,
        text=" ".join(tokens),
        tokens=tokens,
        targets={
            "pos": ("NOUN",) * len(tokens),
            "morph_case": ("-",) * len(tokens),
            "morph_gnn": ("-|-",) * len(tokens),
        },
    )


def _unary_student(
    tokens: tuple[str, ...],
    pos: tuple[str, ...],
    offsets: tuple[int, ...],
    labels: tuple[str, ...],
    repetitions: int = 5,
) -> SimpleNamespace:
    key_rows = [
        _relation_key_levels(tokens, pos, index, offset)
        for index, offset in enumerate(offsets)
    ]
    tables = []
    for level in range(len(key_rows[0])):
        tables.append(
            fit_count_table(
                (
                    (keys[level], label)
                    for keys, label in zip(key_rows, labels, strict=True)
                    for _ in range(repetitions)
                ),
                minsupp=1,
                mindet=0.0,
            )
        )
    return SimpleNamespace(deprel_table=BackoffTables(tables, "dep"))


def test_windowed_key_builder_constructs_hand_worked_hierarchy() -> None:
    tokens = ("Der", "Hund", "bellt", "!")
    pos = ("DET", "NOUN", "VERB", "PUNCT")
    offsets = (1, 1, 0, -1)
    shape = (True, False, False)
    structure = ("RIGHT", 1, "VERB")
    unary = (
        (("hund", "NOUN", 1, "der", "DET", "bellt", "VERB", shape), structure),
        (("hund", "NOUN", 1, "DET", "VERB", shape), structure),
        (("NOUN", 1, "DET", "VERB", shape), structure),
        ("NOUN", ("NOUN", "DET", "VERB"), structure),
        ("NOUN", structure),
        ("NOUN", "RIGHT"),
    )
    governor_context = ((-1, "NOUN", "TITLE"), (1, "PUNCT", "PUNCT"))
    expected = (
        (
            "WINDOW",
            1,
            unary[2],
            ((-1, "DET", "TITLE"), (1, "VERB", "LOWER")),
            governor_context,
        ),
        ("WINDOW", 0, unary[2], (), governor_context),
        *unary,
    )

    assert windowed_relation_key_levels(tokens, pos, 1, offsets, 1) == expected


def test_backoff_by_t_selects_first_supported_window_level() -> None:
    tokens = ("Der", "Hund", "bellt", "!")
    pos = ("DET", "NOUN", "VERB", "PUNCT")
    offsets = (1, 1, 0, -1)
    student = _unary_student(tokens, pos, offsets, ("det", "nsubj", "root", "punct"))
    labeler = WindowedDeprelLabeler(student, max_radius=1, radius=1, min_support=5)
    levels = windowed_relation_key_levels(tokens, pos, 1, offsets, 1)
    labeler.window_tables = {
        1: fit_count_table(((levels[0], "rare") for _ in range(4)), 1, 0.0),
        0: fit_count_table(((levels[1], "supported") for _ in range(6)), 1, 0.0),
    }

    result = labeler.predict_with_diagnostics(tokens, pos, offsets)

    assert result.labels[1] == "supported"
    assert result.windowed_fired[1]
    assert result.fired_radius[1] == 0


def test_populated_tables_use_nonzero_tokens_own_position_key() -> None:
    tokens = ("Alpha", "links", "Beta")
    pos = ("DET", "VERB", "NOUN")
    offsets = (1, 0, -1)
    student = _unary_student(tokens, pos, offsets, ("det", "root", "obj"))
    labeler = WindowedDeprelLabeler(student, max_radius=0, radius=0, min_support=6)
    examples = [
        (tokens, pos, offsets, ("case", "root", "nmod"))
        for _ in range(5)
    ]
    labeler.fit_examples(examples)

    unary = student.deprel_table.predict(
        _relation_key_levels(tokens, pos, 2, offsets[2])
    )
    fallen_through = labeler.predict(tokens, pos, offsets)
    windowed = labeler.predict(tokens, pos, offsets, min_support=5)

    assert all(table.entries for table in student.deprel_table.tables)
    assert labeler.window_tables[0].entries
    assert unary == "obj"
    assert fallen_through[0] == "det"
    assert fallen_through[2] == "obj"
    assert fallen_through[2] != fallen_through[0]
    assert windowed[0] == "case"
    assert windowed[2] == "nmod"
    assert windowed[2] != unary


def test_dev_sweep_does_not_read_test_and_test_is_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = ("Alpha", "links", "Beta")
    pos = ("DET", "VERB", "NOUN")
    offsets = (1, 0, -1)
    sentence = _sentence("dev", tokens)
    record = HeadDeprelRecord("dev", sentence.text, tokens, offsets, ("det", "root", "obj"))
    student = _unary_student(tokens, pos, offsets, record.deprel)
    labeler = WindowedDeprelLabeler(student, max_radius=0)
    labeler.fit_examples([(tokens, pos, offsets, record.deprel)] * 5)
    guard = CampaignTestReadGuard()

    selection = sweep_dev_configurations(
        labeler,
        [sentence],
        {"dev": record},
        {"dev": pos},
        {"dev": offsets},
        radii=(0,),
        thresholds=(5,),
    )["selected"]

    assert guard.counts == {"shared": 0, "head_deprel": 0}

    def fake_shared(_root, _hasher, supplied_guard):
        supplied_guard.claim("shared")
        return [sentence]

    def fake_heads(_root, _hasher, supplied_guard):
        supplied_guard.claim("head_deprel")
        return [record]

    monkeypatch.setattr(module, "load_shared_test_once", fake_shared)
    monkeypatch.setattr(module, "load_head_test_once", fake_heads)
    loaded = load_guarded_test_after_selection(Path("unused"), hashlib.sha256(), guard, selection)

    assert loaded == ([sentence], [record])
    assert guard.counts == {"shared": 1, "head_deprel": 1}
    with pytest.raises(RuntimeError, match="may be read only once"):
        load_guarded_test_after_selection(Path("unused"), hashlib.sha256(), guard, selection)


def test_partition_metric_computes_coarse_matches_on_known_split() -> None:
    predicted = ("nsubj", "obj", "det", "amod", "case", "root")
    gold = ("nsubj:pass", "obl:arg", "det", "amod", "case:foo", "root")

    result = score_relation_partition(predicted, gold)

    assert result["pairwise_sensitive"] == {"accuracy": 0.5, "n": 2}
    assert result["local"] == {"accuracy": 1.0, "n": 3}
    assert result["excluded_n"] == 1


def _metrics(
    las: float,
    *,
    case: float,
    pairwise: float,
    local: float,
) -> dict[str, object]:
    return {
        "las_strict": las,
        "serve_honest_case_accuracy": case,
        "partition": {
            "pairwise_sensitive": {"accuracy": pairwise, "n": 10},
            "local": {"accuracy": local, "n": 10},
        },
    }


@pytest.mark.parametrize(
    ("windowed", "verdict"),
    [
        (_metrics(0.54, case=0.78, pairwise=0.78, local=0.71), "FIRES"),
        (_metrics(0.52, case=0.78, pairwise=0.74, local=0.71), "IN-BETWEEN"),
        (_metrics(0.50, case=0.76, pairwise=0.74, local=0.71), "HALTED"),
    ],
)
def test_section5_read_verdicts(windowed: dict[str, object], verdict: str) -> None:
    unary = _metrics(0.50, case=0.77, pairwise=0.70, local=0.70)

    read = apply_preregistered_read(unary, windowed, 1, 10)

    assert read["verdict"] == verdict
    assert read["dev_M"] == 1
    assert read["dev_T"] == 10
