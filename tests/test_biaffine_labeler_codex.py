from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.biaffine_labeler_codex import (  # noqa: E402
    BilinearRelationLabeler,
    RelationLabelerConfig,
    preregistered_read,
    score_relation_partition,
)
from experiments.german_r1_codex import Sentence  # noqa: E402
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
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


def _tiny_training_data() -> tuple[
    list[Sentence], dict[str, HeadDeprelRecord], dict[str, tuple[str, ...]]
]:
    sentences = [
        _sentence("a", ("Kind", "sieht", "Hund")),
        _sentence("b", ("Mann", "mag", "Frau")),
    ]
    heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "a",
                sentences[0].text,
                sentences[0].tokens,
                (1, 0, -1),
                ("nsubj", "root", "obj"),
            ),
            HeadDeprelRecord(
                "b",
                sentences[1].text,
                sentences[1].tokens,
                (1, 0, -1),
                ("nsubj", "root", "obj"),
            ),
        ]
    )
    predicted_pos = {
        "a": ("NOUN", "VERB", "NOUN"),
        "b": ("NOUN", "VERB", "NOUN"),
    }
    return sentences, heads, predicted_pos


def test_relation_labeler_shapes_and_finite_epoch_losses() -> None:
    sentences, heads, predicted_pos = _tiny_training_data()
    labeler = BilinearRelationLabeler(
        RelationLabelerConfig(rank=4, epochs=3, learning_rate=0.1, seed=0)
    )

    labeler.fit(sentences, heads, predicted_pos)

    phi = labeler.phi("Kind", "NOUN", 0, 3)
    assert phi.shape == (labeler.feature_dim,)
    assert labeler.U is not None and labeler.U.shape == (labeler.feature_dim, 4)
    assert labeler.V is not None and labeler.V.shape == (labeler.feature_dim, 4)
    assert labeler.C is not None and labeler.C.shape == (3, 4)
    assert labeler.P is not None and labeler.P.shape == (3, labeler.feature_dim)
    assert labeler.Q is not None and labeler.Q.shape == (3, labeler.feature_dim)
    assert labeler.b is not None and labeler.b.shape == (3,)
    assert labeler.relations == ("nsubj", "obj", "root")
    assert len(labeler.epoch_losses) == 3
    assert np.all(np.isfinite(labeler.epoch_losses))


def test_predict_is_independent_of_unreachable_gold_dependency_objects() -> None:
    sentences, heads, predicted_pos = _tiny_training_data()
    labeler = BilinearRelationLabeler(RelationLabelerConfig(rank=4, epochs=2, seed=0))
    labeler.fit(sentences, heads, predicted_pos)
    tokens = sentences[0].tokens
    pos = predicted_pos["a"]
    supplied_predicted_heads = (0, -1, -2)
    gold_a = HeadDeprelRecord(
        "gold-a", "", tokens, (1, 0, -1), ("nsubj", "root", "obj")
    )

    before = labeler.predict(tokens, pos, supplied_predicted_heads)
    gold_b = HeadDeprelRecord(
        "gold-b", "", tokens, (-2, 1, 0), ("punct", "det", "root")
    )
    after = labeler.predict(tokens, pos, supplied_predicted_heads)

    assert gold_a.head_offset != supplied_predicted_heads
    assert gold_b.head_offset != gold_a.head_offset
    assert gold_b.deprel != gold_a.deprel
    assert before == after


def test_invalid_supplied_head_falls_back_to_dep_without_scoring() -> None:
    sentences, heads, predicted_pos = _tiny_training_data()
    labeler = BilinearRelationLabeler(RelationLabelerConfig(rank=4, epochs=1, seed=0))
    labeler.fit(sentences, heads, predicted_pos)

    predicted = labeler.predict(
        sentences[0].tokens,
        predicted_pos["a"],
        (99, 0, -1),
    )

    assert predicted[0] == "dep"
    assert predicted[1] in labeler.relations
    assert predicted[2] in labeler.relations


def test_partition_metric_scores_buckets_and_excludes_other_relations() -> None:
    predicted = ("nsubj:pass", "obl", "det", "dep", "case", "punct")
    gold = ("nsubj:pass", "obj", "det", "amod", "case", "root")

    metric = score_relation_partition(predicted, gold)

    assert metric["pairwise_sensitive"] == {"accuracy": 0.5, "n": 2}
    assert metric["local"]["accuracy"] == pytest.approx(2 / 3)
    assert metric["local"]["n"] == 3
    assert metric["excluded_n"] == 1


def _metric(
    las: float,
    *,
    case: float = 0.61,
    pairwise: float = 0.78,
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


def test_preregistered_read_fires_when_every_condition_holds() -> None:
    unary = _metric(0.50, case=0.60, pairwise=0.70, local=0.70)
    seeds = {
        0: _metric(0.54, pairwise=0.79, local=0.72),
        1: _metric(0.53, pairwise=0.78, local=0.72),
        2: _metric(0.55, pairwise=0.80, local=0.72),
    }

    read = preregistered_read(unary, seeds)

    assert read["verdict"] == "FIRES"
    assert read["las_gain_mean"] == pytest.approx(0.04)
    assert read["partition_gains"]["pairwise_sensitive"] > read["partition_gains"]["local"]


@pytest.mark.parametrize(
    "seeds",
    [
        {
            0: _metric(0.52, pairwise=0.78, local=0.71),
            1: _metric(0.52, pairwise=0.78, local=0.71),
            2: _metric(0.52, pairwise=0.78, local=0.71),
        },
        {
            0: _metric(0.56, pairwise=0.78, local=0.71),
            1: _metric(0.55, pairwise=0.78, local=0.71),
            2: _metric(0.49, pairwise=0.78, local=0.71),
        },
        {
            0: _metric(0.54, case=0.60, pairwise=0.78, local=0.71),
            1: _metric(0.54, case=0.60, pairwise=0.78, local=0.71),
            2: _metric(0.54, case=0.60, pairwise=0.78, local=0.71),
        },
        {
            0: _metric(0.54, pairwise=0.72, local=0.78),
            1: _metric(0.54, pairwise=0.72, local=0.78),
            2: _metric(0.54, pairwise=0.72, local=0.78),
        },
    ],
    ids=("small-gain", "one-negative-seed", "case-not-improved", "diffuse"),
)
def test_preregistered_read_in_between_for_each_individual_condition(
    seeds: dict[int, dict[str, object]],
) -> None:
    unary = _metric(0.50, case=0.60, pairwise=0.70, local=0.70)

    read = preregistered_read(unary, seeds)

    assert read["verdict"] == "IN-BETWEEN"


def test_preregistered_read_halts_for_nonpositive_mean_las_gain() -> None:
    unary = _metric(0.50, case=0.60, pairwise=0.70, local=0.70)
    seeds = {
        0: _metric(0.51, pairwise=0.80, local=0.70),
        1: _metric(0.50, pairwise=0.80, local=0.70),
        2: _metric(0.49, pairwise=0.80, local=0.70),
    }

    read = preregistered_read(unary, seeds)

    assert read["verdict"] == "HALTED"
    assert read["las_gain_mean"] == pytest.approx(0.0)
