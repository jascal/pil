from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.attach_levers_codex import (  # noqa: E402
    BilinearAttachmentScorer,
    BilinearConfig,
    chu_liu_edmonds,
    evaluate_ladder,
    half_oracle_dependency_pairs,
    score_case_bearing_deprels,
    score_prediction_sequences,
)
from experiments.german_r1_codex import Sentence  # noqa: E402
from experiments.german_r3_codex import DependencyPrediction  # noqa: E402
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    head_deprel_by_sent,
)


def _sentence(
    sent_id: str,
    tokens: tuple[str, ...],
    *,
    case: tuple[str, ...] | None = None,
) -> Sentence:
    return Sentence(
        sent_id=sent_id,
        text=" ".join(tokens),
        tokens=tokens,
        targets={
            "pos": ("NOUN",) * len(tokens),
            "morph_case": case or ("-",) * len(tokens),
            "morph_gnn": ("-|-",) * len(tokens),
        },
    )


class _RelationTable:
    def predict(self, _keys: Any) -> str:
        return "dep"


class _FixtureL0:
    def __init__(self, offsets: tuple[int, ...]) -> None:
        self.offsets = offsets
        self.deprel_table = _RelationTable()

    def predict_heads(
        self,
        _tokens: tuple[str, ...],
        _predicted_pos: tuple[str, ...],
    ) -> tuple[int, ...]:
        return self.offsets

    def predict_sentence(
        self,
        _tokens: tuple[str, ...],
        _predicted_pos: tuple[str, ...],
    ) -> DependencyPrediction:
        return DependencyPrediction(self.offsets, ("dep",) * len(self.offsets))


class _FixtureBilinear:
    def __init__(self, scores: np.ndarray) -> None:
        self.scores = scores

    def score_matrix(
        self,
        _tokens: tuple[str, ...],
        _predicted_pos: tuple[str, ...],
    ) -> np.ndarray:
        return self.scores.copy()


class _FixtureCaseStudent:
    def predict_sentence(self, sentence: Sentence) -> dict[str, dict[str, list[str]]]:
        return {
            "full": {
                "morph_case": list(sentence.targets["morph_case"]),
                "pos": list(sentence.targets["pos"]),
            }
        }


def _passthrough_case(
    _student: Any,
    _tokens: tuple[str, ...],
    baseline: list[str],
    _predicted_pos: tuple[str, ...],
    _predicted_head_offset: tuple[int, ...],
    _predicted_deprel: tuple[str, ...],
    _eligible: list[int],
) -> SimpleNamespace:
    return SimpleNamespace(predictions=tuple(baseline))


def test_chu_liu_edmonds_breaks_cycle_for_known_optimum() -> None:
    # Greedy selects A->B and B->A (score 10 each), which is a cycle.  The
    # unique best single-root tree is C->ROOT, A->C, B->A, total 8+3+10=21.
    scores = np.asarray(
        [
            [1.0, -np.inf, 10.0, 3.0],
            [1.0, 10.0, -np.inf, 2.0],
            [8.0, 7.0, 6.0, -np.inf],
        ]
    )

    greedy = tuple(int(value) for value in np.argmax(scores, axis=1))
    decoded = chu_liu_edmonds(scores)

    assert greedy == (2, 1, 0)
    assert decoded == (3, 1, 0)
    assert sum(scores[index, governor] for index, governor in enumerate(decoded)) == 21.0
    assert decoded.count(0) == 1


def test_bilinear_feature_factor_and_score_shapes() -> None:
    scorer = BilinearAttachmentScorer(BilinearConfig(rank=4, epochs=1, seed=0))
    tokens = ("Das", "HAUS", ".")
    pos = ("DET", "NOUN", "PUNCT")

    phi = scorer.phi(tokens[0], pos[0], 0, len(tokens))
    scores = scorer.score_matrix(tokens, pos)

    assert phi.shape == (scorer.feature_dim,)
    assert scorer.U.shape == (scorer.feature_dim, 4)
    assert scorer.V.shape == (scorer.feature_dim, 4)
    assert scorer.W.shape == (scorer.feature_dim, scorer.feature_dim)
    assert np.linalg.matrix_rank(scorer.W) <= 4
    assert scores.shape == (3, 4)
    assert np.all(np.isneginf(scores[np.arange(3), np.arange(1, 4)]))


def test_bilinear_loss_decreases_over_several_sgd_epochs() -> None:
    # The assertion is initial-vs-final rather than stepwise: factorized SGD is
    # non-convex, so individual shuffled sentence updates need not be monotone.
    sentences = [
        _sentence("a", ("Kind", "sieht", "Hund")),
        _sentence("b", ("Mann", "mag", "Frau")),
    ]
    heads = head_deprel_by_sent(
        [
            HeadDeprelRecord("a", sentences[0].text, sentences[0].tokens, (1, 0, -1), ("dep",) * 3),
            HeadDeprelRecord("b", sentences[1].text, sentences[1].tokens, (1, 0, -1), ("dep",) * 3),
        ]
    )
    predicted_pos = {"a": ("NOUN", "VERB", "NOUN"), "b": ("NOUN", "VERB", "NOUN")}
    scorer = BilinearAttachmentScorer(
        BilinearConfig(rank=4, epochs=20, learning_rate=0.15, l2_penalty=0.0, seed=0)
    )
    initial = scorer.mean_loss(sentences, heads, predicted_pos)

    scorer.fit(sentences, heads, predicted_pos)
    final = scorer.mean_loss(sentences, heads, predicted_pos)

    assert final < initial


def test_shared_uas_las_and_deprel_only_scorer_has_known_values() -> None:
    metrics = score_prediction_sequences(
        predicted_head_offset=(1, 0, 1, -2),
        predicted_deprel=("det", "nsubj", "obj", "punct"),
        gold_head_offset=(1, 0, -1, -2),
        gold_deprel=("det", "root", "obj", "punct"),
    )

    assert metrics["uas"] == pytest.approx(0.75)
    assert metrics["las"] == pytest.approx(0.50)
    assert metrics["deprel_only_accuracy"] == pytest.approx(0.75)
    assert metrics["tokens"] == 4


def test_ladder_wiring_runs_all_four_rungs_end_to_end() -> None:
    sentence = _sentence("ladder", ("A", "B", "C"))
    heads = head_deprel_by_sent(
        [HeadDeprelRecord("ladder", sentence.text, sentence.tokens, (1, 1, 0), ("dep",) * 3)]
    )
    predicted_pos = {"ladder": ("NOUN", "NOUN", "VERB")}
    scores = np.asarray(
        [
            [0.0, -np.inf, 3.0, 1.0],
            [0.0, 1.0, -np.inf, 3.0],
            [3.0, 1.0, 1.0, -np.inf],
        ]
    )

    metrics, predictions = evaluate_ladder(
        [sentence],
        heads,
        predicted_pos,
        _FixtureL0((1, 1, 0)),
        _FixtureBilinear(scores),
        _FixtureCaseStudent(),
        _passthrough_case,
    )

    assert tuple(metrics) == ("L0", "L1", "L2", "L3")
    assert tuple(predictions["ladder"]) == ("L0", "L1", "L2", "L3")
    for rung in metrics.values():
        assert set(rung) == {
            "uas",
            "las",
            "deprel_only_accuracy",
            "tokens",
            "serve_honest_morph_case",
        }
        assert all(isinstance(value, int | float) for value in rung.values())


def test_case_plumbing_passes_predictions_not_gold_dependencies() -> None:
    sentence = _sentence("plumbing", ("A", "B", "C"), case=("Nom", "-", "-"))
    gold_offsets = (-2, -1, 0)
    gold_deprels = ("gold-a", "gold-b", "root")
    heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "plumbing",
                sentence.text,
                sentence.tokens,
                gold_offsets,
                gold_deprels,
            )
        ]
    )
    predicted_pos = {"plumbing": ("DET", "NOUN", "VERB")}
    scores = np.asarray(
        [
            [0.0, -np.inf, 3.0, 1.0],
            [0.0, 1.0, -np.inf, 3.0],
            [3.0, 1.0, 1.0, -np.inf],
        ]
    )
    calls: list[tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]] = []

    def spy_case(
        _student: Any,
        _tokens: tuple[str, ...],
        baseline: list[str],
        pos: tuple[str, ...],
        head_offset: tuple[int, ...],
        deprel: tuple[str, ...],
        _eligible: list[int],
    ) -> SimpleNamespace:
        calls.append((pos, head_offset, deprel))
        return SimpleNamespace(predictions=tuple(baseline))

    evaluate_ladder(
        [sentence],
        heads,
        predicted_pos,
        _FixtureL0((1, 1, 0)),
        _FixtureBilinear(scores),
        _FixtureCaseStudent(),
        spy_case,
    )

    assert len(calls) == 4
    assert all(call[0] == predicted_pos["plumbing"] for call in calls)
    assert all(call[1] != gold_offsets for call in calls)
    assert all(call[2] != gold_deprels for call in calls)


def test_case_bearing_deprel_accuracy_uses_only_exact_routing_relations() -> None:
    predicted = ("nsubj", "obj", "punct", "obl", "cop", "dep")
    gold = ("nsubj", "iobj", "punct", "obl:arg", "cop", "root")

    relation_accuracy, relation_n = score_case_bearing_deprels(predicted, gold)

    assert relation_n == 4
    assert relation_accuracy == pytest.approx(0.5)


def test_half_oracle_dependency_pairs_mix_only_the_requested_fields() -> None:
    predicted = DependencyPrediction(
        head_offset=(1, 0, -1),
        deprel=("pred-a", "pred-root", "pred-b"),
    )
    oracle = HeadDeprelRecord(
        sent_id="mix",
        text="A B C",
        tokens=("A", "B", "C"),
        head_offset=(2, 1, 0),
        deprel=("gold-a", "gold-b", "gold-root"),
    )

    pairs = half_oracle_dependency_pairs(predicted, oracle)

    assert pairs["predicted_heads_predicted_deprels"] == predicted
    assert pairs["oracle_heads_oracle_deprels"].head_offset == oracle.head_offset
    assert pairs["oracle_heads_oracle_deprels"].deprel == oracle.deprel
    assert pairs["oracle_heads_predicted_deprels"].head_offset == oracle.head_offset
    assert pairs["oracle_heads_predicted_deprels"].deprel == predicted.deprel
    assert pairs["predicted_heads_oracle_deprels"].head_offset == predicted.head_offset
    assert pairs["predicted_heads_oracle_deprels"].deprel == oracle.deprel
