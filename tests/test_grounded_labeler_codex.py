from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.german_r3_codex as german_r3  # noqa: E402
from experiments.attach_levers_codex import (  # noqa: E402
    predict_deprels,
    score_prediction_sequences,
)
from experiments.biaffine_labeler_codex import RelationLabelerConfig  # noqa: E402
from experiments.german_r1_codex import Sentence  # noqa: E402
from experiments.german_r3_codex import (  # noqa: E402
    CampaignTestReadGuard,
    GermanR3DependencyStudent,
    load_head_test_once,
    load_shared_test_once,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    head_deprel_by_sent,
)
from experiments.grounded_labeler_codex import (  # noqa: E402
    FEATURE_DIM,
    GroundedRelationLabeler,
    GroundedResidualProvider,
    assert_matched_uas,
    doubly_covered_indices,
    preregistered_read,
    select_dev_layer,
)


def _write_provider(
    path: Path,
    keys: list[tuple[str, int]],
    *,
    duplicate: bool = False,
) -> GroundedResidualProvider:
    if duplicate:
        keys = [*keys, keys[0]]
    residuals = np.zeros((len(keys), 4, FEATURE_DIM), dtype=np.float16)
    for row in range(len(keys)):
        for layer in range(4):
            residuals[row, layer, :] = row * 10 + layer
    np.savez(
        path,
        sent_ids=np.asarray([key[0] for key in keys]),
        token_index=np.asarray([key[1] for key in keys], dtype=np.int64),
        word_text=np.asarray(["x"] * len(keys)),
        checkpoint_layers=np.asarray([8, 17, 26, 35], dtype=np.int64),
        last_residual=residuals,
        mean_residual=residuals,
    )
    return GroundedResidualProvider(path)


def _sentence(sent_id: str = "s") -> Sentence:
    tokens = ("Kind", "sieht", "Hund")
    return Sentence(
        sent_id,
        " ".join(tokens),
        tokens,
        {
            "pos": ("NOUN", "VERB", "NOUN"),
            "morph_case": ("Nom", "-", "Acc"),
            "morph_gnn": ("Neut|Sing", "-|-", "Masc|Sing"),
        },
    )


class _FixedRelationTable:
    def __init__(self, label: str) -> None:
        self.label = label

    def predict(self, _keys: Any) -> str:
        return self.label


def _unary_student(label: str = "unary-fallback") -> GermanR3DependencyStudent:
    student = GermanR3DependencyStudent(1)
    student.deprel_table = _FixedRelationTable(label)  # type: ignore[assignment]
    return student


def _initialized_labeler(relation: str = "nsubj") -> GroundedRelationLabeler:
    labeler = GroundedRelationLabeler(
        RelationLabelerConfig(rank=2, epochs=1, learning_rate=0.01, seed=0)
    )
    labeler._initialize_parameters([relation])
    return labeler


def test_provider_lookup_layer_slice_and_duplicate_rejection(tmp_path: Path) -> None:
    provider = _write_provider(tmp_path / "ok.npz", [("s", 0), ("s", 2)])

    assert provider.covered("s", 0)
    assert not provider.covered("s", 1)
    assert provider.residual("s", 2, 3).dtype == np.float64
    assert np.all(provider.residual("s", 2, 3) == 13.0)
    with pytest.raises(KeyError):
        provider.residual("s", 1, 0)
    with pytest.raises(ValueError, match="duplicate grounded key"):
        _write_provider(tmp_path / "duplicate.npz", [("s", 0)], duplicate=True)


def test_doubly_covered_filter_root_and_unary_fallback(tmp_path: Path) -> None:
    provider = _write_provider(tmp_path / "coverage.npz", [("s", 0), ("s", 1)])
    sentence = _sentence()
    pos = sentence.targets["pos"]
    heads = (1, 0, -1)
    l0 = _unary_student()
    labeler = _initialized_labeler()

    selected = doubly_covered_indices("s", 3, heads, provider, 0)
    predicted = labeler.predict("s", sentence.tokens, pos, heads, provider, 0, l0)
    unary = predict_deprels(l0, sentence.tokens, pos, heads)

    assert selected == (0,)
    assert 1 not in selected  # ROOT, despite its covered dependent row.
    assert predicted[1] == "root"
    assert predicted[2] == unary[2] == "unary-fallback"


def test_fit_excludes_root_and_uncovered_arc_relations(tmp_path: Path) -> None:
    provider = _write_provider(tmp_path / "train.npz", [("s", 0), ("s", 1)])
    sentence = _sentence()
    heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "s",
                sentence.text,
                sentence.tokens,
                (1, 0, -1),
                ("nsubj", "root", "uncovered-only"),
            )
        ]
    )
    labeler = GroundedRelationLabeler(
        RelationLabelerConfig(rank=2, epochs=1, learning_rate=0.01, seed=0)
    )

    labeler.fit([sentence], provider, heads, 0)

    assert labeler.relations == ("nsubj",)
    assert "root" not in labeler.relations
    assert "uncovered-only" not in labeler.relations
    assert labeler.training_arc_count == 1
    assert labeler.training_sentence_count == 1


def test_uas_identity_on_synthetic_doubly_covered_subset() -> None:
    gold_heads = (1, -1)
    gold_relations = ("nsubj", "obj")
    common_heads = (1, 1)
    labels = {
        "unary": ("dep", "obj"),
        "surface119": ("nsubj", "dep"),
        "grounded": ("nsubj", "obj"),
    }
    metrics = {
        arm: score_prediction_sequences(
            common_heads, relations, gold_heads, gold_relations
        )
        for arm, relations in labels.items()
    }

    assert_matched_uas(metrics)
    assert {metric["uas"] for metric in metrics.values()} == {0.5}


def test_dev_layer_selection_argmax_and_smallest_tie_break() -> None:
    assert select_dev_layer({0: 0.4, 1: 0.7, 2: 0.6, 3: 0.5}) == 1
    assert select_dev_layer({0: 0.7, 1: 0.7, 2: 0.6, 3: 0.7}) == 0


def _metric(
    las: float,
    *,
    case: float = 0.80,
    pairwise: float = 0.75,
    local: float = 0.72,
) -> dict[str, object]:
    return {
        "uas": 0.70,
        "las_strict": las,
        "las_coarse": las + 0.01,
        "deprel_only_accuracy": las + 0.10,
        "case_bearing_deprel_accuracy": las,
        "case_bearing_deprel_n": 10,
        "serve_honest_morph_case": case,
        "serve_honest_morph_case_n": 20,
        "tokens": 100,
        "partition": {
            "pairwise_sensitive": {"accuracy": pairwise, "n": 30},
            "local": {"accuracy": local, "n": 40},
            "excluded_n": 30,
        },
    }


def test_preregistered_read_fires() -> None:
    read = preregistered_read(
        _metric(0.50, case=0.80, pairwise=0.70, local=0.70),
        _metric(0.52, case=0.80),
        _metric(0.54, case=0.81, pairwise=0.80, local=0.72),
        2,
        {"dev": 0.5, "test": 0.51},
    )

    assert read["verdict"] == "FIRES"
    assert read["deprel_las_gain_vs_unary"] == pytest.approx(0.04)
    assert read["gain_vs_surface119"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("grounded", "surface"),
    [
        (_metric(0.52, case=0.81, pairwise=0.80, local=0.72), _metric(0.51)),
        (_metric(0.54, case=0.81, pairwise=0.72, local=0.80), _metric(0.51)),
        (_metric(0.54, case=0.80, pairwise=0.80, local=0.72), _metric(0.51)),
        (_metric(0.54, case=0.81, pairwise=0.80, local=0.72), _metric(0.55)),
    ],
    ids=("small-gain", "diffuse", "no-case-improvement", "loses-surface119"),
)
def test_preregistered_read_in_between_for_each_failing_condition(
    grounded: dict[str, object], surface: dict[str, object]
) -> None:
    read = preregistered_read(
        _metric(0.50, case=0.80, pairwise=0.70, local=0.70),
        surface,
        grounded,
        0,
        {"dev": 0.5, "test": 0.5},
    )

    assert read["verdict"] == "IN-BETWEEN"


@pytest.mark.parametrize("grounded_las", [0.50, 0.49])
def test_preregistered_read_halted(grounded_las: float) -> None:
    read = preregistered_read(
        _metric(0.50, case=0.80, pairwise=0.70, local=0.70),
        _metric(0.48),
        _metric(grounded_las, case=0.81, pairwise=0.80, local=0.71),
        0,
        {"dev": 0.5, "test": 0.5},
    )

    assert read["verdict"] == "HALTED → rung 3 (R4 hybrid)"


def test_campaign_test_guard_rejects_second_loader_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(german_r3, "load_split", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        german_r3, "load_head_deprel_file", lambda *_args, **_kwargs: []
    )
    guard = CampaignTestReadGuard()
    hasher = hashlib.sha256()

    assert load_shared_test_once(tmp_path, hasher, guard) == []
    assert load_head_test_once(tmp_path, hasher, guard) == []
    guard.assert_complete()
    with pytest.raises(RuntimeError, match="may be read only once"):
        load_shared_test_once(tmp_path, hasher, guard)
    with pytest.raises(RuntimeError, match="may be read only once"):
        load_head_test_once(tmp_path, hasher, guard)
