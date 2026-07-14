import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.german_r1_codex import (  # noqa: E402
    CASE_LABELS,
    DATA_ROOT,
    GermanR1Student,
    RegisterLayer,
    Sentence,
)
from experiments.german_r3_codex import (  # noqa: E402
    MajorityOffsetBaseline,
    flatten_haiku_tree,
    load_head_deprel_file,
    resolve_governor_index,
    run_predicted_case_sentence,
    score_dependencies,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    full_oracle_case_sentence,
    head_deprel_by_sent,
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


def test_head_loader_and_relative_governor_arithmetic(tmp_path: Path) -> None:
    fixture = {
        "sent_id": "fixture-head",
        "text": "Das Haus steht",
        "tokens": ["Das", "Haus", "steht"],
        "targets": {
            "head_offset": [1, 1, 0],
            "deprel": ["det", "nsubj", "root"],
        },
    }
    path = tmp_path / "head.jsonl"
    path.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
    hasher = hashlib.sha256()

    records = load_head_deprel_file(path, hasher)

    assert len(records) == 1
    assert records[0].head_offset == (1, 1, 0)
    assert resolve_governor_index(0, records[0].head_offset[0], 3) == 1
    assert resolve_governor_index(2, records[0].head_offset[2], 3) == 2
    assert hasher.hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest()


def test_flatten_haiku_tree_preserves_parent_heads_and_root_self_loop() -> None:
    tree = {
        "word": "steht",
        "position": 2,
        "dep": "root",
        "children": [
            {"word": "Haus", "position": 1, "dep": "nsubj", "children": []},
            {
                "word": "neben",
                "position": 3,
                "dep": "obl",
                "children": [
                    {"word": "an", "position": 4, "dep": "case", "children": []}
                ],
            },
        ],
    }

    flattened = flatten_haiku_tree(tree)

    assert [(token.position, token.dep, token.head_position) for token in flattened] == [
        (1, "nsubj", 2),
        (2, "root", 2),
        (3, "obl", 2),
        (4, "case", 3),
    ]


def test_majority_offset_baseline_uses_predicted_pos_and_global_deprel_mode() -> None:
    train = [
        _sentence("a", ("A",)),
        _sentence("b", ("B",)),
        _sentence("c", ("C",)),
    ]
    heads = head_deprel_by_sent(
        [
            HeadDeprelRecord("a", "A", ("A",), (1,), ("det",)),
            HeadDeprelRecord("b", "B", ("B",), (1,), ("det",)),
            HeadDeprelRecord("c", "C", ("C",), (-1,), ("obj",)),
        ]
    )
    predicted_pos = {sentence.sent_id: ("NOUN",) for sentence in train}
    baseline = MajorityOffsetBaseline()
    baseline.fit(train, heads, predicted_pos)

    prediction = baseline.predict(("NOUN", "UNSEEN"))

    assert prediction.head_offset == (1, 1)
    assert prediction.deprel == ("det", "det")


def test_uas_las_scorer_has_exact_known_values() -> None:
    scores = score_dependencies(
        predicted_head_offset=(1, 0, 1, -2),
        predicted_deprel=("det", "nsubj", "obj", "punct"),
        gold_head_offset=(1, 0, -1, -2),
        gold_deprel=("det", "root", "obj", "punct"),
    )

    assert scores.uas == pytest.approx(0.75)
    assert scores.las == pytest.approx(0.50)
    assert scores.tokens == 4


def test_case_cascade_uses_only_predicted_dependency_and_pos_arrays() -> None:
    student = GermanR1Student(RegisterLayer.from_directory(DATA_ROOT / "registers"))
    student.registers.verb_government["helfen"] = "Dat"
    tokens = ("helfen", "Objekt")
    baseline = ("-", "Nom")

    predicted_result = run_predicted_case_sentence(
        student,
        tokens,
        baseline,
        predicted_pos=("NOUN", "NOUN"),
        predicted_head_offset=(0, 0),
        predicted_deprel=("root", "root"),
        eligible_indices=(1,),
    )
    gold_oracle_result = full_oracle_case_sentence(
        student,
        tokens,
        baseline,
        ("VERB", "NOUN"),
        ("VERB", "NOUN"),
        (0, -1),
        ("root", "obj"),
        (1,),
    )

    assert "gold" not in inspect.signature(run_predicted_case_sentence).parameters
    assert predicted_result.predictions == baseline
    assert gold_oracle_result.predictions == ("-", "Dat")
    assert predicted_result.predictions != gold_oracle_result.predictions
    assert len(predicted_result.predictions) == len(tokens)
    assert set(predicted_result.predictions) <= set(CASE_LABELS)
