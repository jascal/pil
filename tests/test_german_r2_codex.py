import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.german_r2_codex import (  # noqa: E402
    Candidate,
    SurfacePosPredictor,
    TargetRecord,
    fit_count_table,
    followed_by_noun_rule,
    parse_record,
    score_indexed,
    verdict,
)
from experiments.german_r2_codex import (
    TestReadGuard as ReadGuard,
)


def test_loader_extracts_only_aligned_sparse_targets() -> None:
    raw = json.dumps(
        {
            "sent_id": "fixture-1",
            "text": "Die kennen das Haus.",
            "tokens": ["Die", "kennen", "das", "Haus", "."],
            "targets": {"indices": [0, 2], "label": ["PRON", "DET"]},
        }
    )

    record = parse_record(raw)

    assert list(record.targets()) == [(0, "Die", "PRON"), (2, "das", "DET")]
    assert [token for _, token, _ in record.targets()] == ["Die", "das"]


def test_followed_by_noun_catalog_rule_fires_on_surface_capitalization() -> None:
    proposal = followed_by_noun_rule(("Ich", "sehe", "die", "Katze", "."), 2)

    assert proposal == Candidate("DET", 1.0, "rule")
    assert followed_by_noun_rule(("Die", "kennen", "wir", "."), 0) is None


def test_majority_per_form_and_unseen_fallback() -> None:
    table = fit_count_table(
        [("die", "DET"), ("die", "DET"), ("die", "PRON"), ("das", "PRON")]
    )

    die = table.lookup("die")
    assert die is not None
    assert die.label == "DET"
    assert (die.count, die.total) == (2, 3)
    assert die.confidence == pytest.approx(2 / 5)
    assert table.lookup("unseen") is None
    assert (table.lookup("unseen") or die).label == "DET"


def test_scorer_scores_indexed_tokens_only() -> None:
    record = TargetRecord(
        "fixture-score",
        "Die sehen das Haus.",
        ("Die", "sehen", "das", "Haus", "."),
        (0, 2),
        ("PRON", "DET"),
    )
    predictions = {0: "PRON", 1: "WRONG", 2: "DET", 3: "WRONG", 4: "WRONG"}

    score = score_indexed([record], lambda _tokens, index: predictions[index])

    assert score == 1.0
    assert score != 2 / 5


@pytest.mark.parametrize(
    ("det_acc", "aux_acc", "catalog", "expected"),
    [
        (0.97, 0.97, True, "FIRES"),
        (0.9699, 0.99, True, "IN-BETWEEN/MISS"),
        (0.99, 0.9699, True, "IN-BETWEEN/MISS"),
        (0.99, 0.99, False, "IN-BETWEEN/MISS"),
    ],
)
def test_verdict_requires_both_bars_and_catalog(
    det_acc: float, aux_acc: float, catalog: bool, expected: str
) -> None:
    assert verdict(det_acc, aux_acc, catalog) == expected


def test_test_read_guard_rejects_second_claim_per_task() -> None:
    guard = ReadGuard()
    guard.claim("det_pron")

    with pytest.raises(RuntimeError, match="may be read only once"):
        guard.claim("det_pron")


def test_surface_pos_predictor_uses_only_supplied_training_records() -> None:
    predictor = SurfacePosPredictor()
    predictor.fit(
        [
            {
                "tokens": ["Das", "Haus", "steht"],
                "targets": {"upos": ["DET", "NOUN", "VERB"]},
            }
        ]
    )

    assert predictor.predict("Haus") == "NOUN"
    assert predictor.predict("steht") == "VERB"
