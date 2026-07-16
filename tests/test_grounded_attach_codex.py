from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.german_r3_codex as german_r3  # noqa: E402
import experiments.grounded_attach_codex as grounded_attach  # noqa: E402
from experiments.attach_levers_codex import (  # noqa: E402
    BilinearConfig,
    chu_liu_edmonds,
    l0_edge_scores,
    score_prediction_sequences,
)
from experiments.biaffine_labeler_codex import RelationLabelerConfig  # noqa: E402
from experiments.german_r1_codex import Sentence  # noqa: E402
from experiments.german_r3_codex import (  # noqa: E402
    CampaignTestReadGuard,
    DependencyPrediction,
    load_head_test_once,
    load_shared_test_once,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    head_deprel_by_sent,
)
from experiments.grounded_attach_codex import (  # noqa: E402
    GroundedAttachmentScorer,
    covered_dependent_fraction,
    covered_dependent_indices,
    doubly_covered_fraction,
    evaluate_test_arms,
    grounded_geometry,
    mixed_edge_scores,
    preregistered_read,
)
from experiments.grounded_labeler_codex import (  # noqa: E402
    FEATURE_DIM,
    GroundedRelationLabeler,
    GroundedResidualProvider,
)


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


def _write_provider(
    path: Path,
    keys: list[tuple[str, int]],
    vectors: np.ndarray | None = None,
) -> GroundedResidualProvider:
    if vectors is None:
        vectors = np.zeros((len(keys), FEATURE_DIM), dtype=np.float64)
        for row in range(len(keys)):
            vectors[row, row : row + 4] = 1.0 + row
    residuals = np.repeat(vectors[:, None, :], 4, axis=1).astype(np.float16)
    np.savez(
        path,
        sent_ids=np.asarray([sent_id for sent_id, _index in keys]),
        token_index=np.asarray([index for _sent_id, index in keys], dtype=np.int64),
        word_text=np.asarray(["x"] * len(keys)),
        checkpoint_layers=np.asarray([8, 17, 26, 35], dtype=np.int64),
        last_residual=residuals,
    )
    return GroundedResidualProvider(path)


class _RelationTable:
    def __init__(self, label: str = "dep") -> None:
        self.label = label

    def predict(self, _keys: Any) -> str:
        return self.label


class _FixedL0:
    def __init__(self, offsets: tuple[int, ...], label: str = "dep") -> None:
        self.offsets = offsets
        self.deprel_table = _RelationTable(label)

    def predict_heads(
        self, _tokens: tuple[str, ...], _pos: tuple[str, ...]
    ) -> tuple[int, ...]:
        return self.offsets

    def predict_sentence(
        self, _tokens: tuple[str, ...], _pos: tuple[str, ...]
    ) -> DependencyPrediction:
        return DependencyPrediction(self.offsets, ("dep",) * len(self.offsets))


class _FixedSurface:
    def __init__(self, scores: np.ndarray) -> None:
        self.scores = scores

    def score_matrix(
        self, _tokens: tuple[str, ...], _pos: tuple[str, ...]
    ) -> np.ndarray:
        return self.scores.copy()


class _CaseStudent:
    def predict_sentence(self, sentence: Sentence) -> dict[str, dict[str, list[str]]]:
        return {"full": {"morph_case": ["Nom", "-", "Acc"]}}


def _heads(sentence: Sentence) -> dict[str, HeadDeprelRecord]:
    return head_deprel_by_sent(
        [
            HeadDeprelRecord(
                sentence.sent_id,
                sentence.text,
                sentence.tokens,
                (1, 0, -1),
                ("dep", "root", "dep"),
            )
        ]
    )  # type: ignore[return-value]


def test_grounded_attachment_fit_tracks_finite_decreasing_loss(tmp_path: Path) -> None:
    sentence = _sentence()
    provider = _write_provider(
        tmp_path / "fit.npz", [("s", 0), ("s", 1), ("s", 2)]
    )
    scorer = GroundedAttachmentScorer(
        BilinearConfig(
            rank=4,
            epochs=20,
            learning_rate=0.15,
            l2_penalty=0.0,
            seed=0,
        )
    )

    scorer.fit([sentence], provider, _heads(sentence), 0)

    assert scorer.training_sentence_count == 1
    assert scorer.training_arc_count == 2
    assert len(scorer.epoch_losses) == 20
    assert np.all(np.isfinite(scorer.epoch_losses))
    assert scorer.epoch_losses[-1] < scorer.epoch_losses[0]
    assert np.isfinite(scorer.score_edge(provider, "s", 0, 1, 0))
    assert scorer.W.shape == (FEATURE_DIM, FEATURE_DIM)


def test_covered_dependent_filter_and_both_coverage_fractions(tmp_path: Path) -> None:
    sentence = _sentence()
    provider = _write_provider(tmp_path / "coverage.npz", [("s", 0), ("s", 1)])

    selected = covered_dependent_indices("s", 3, provider, 0)

    assert selected == (0, 1)
    assert covered_dependent_fraction([sentence], provider, 0) == pytest.approx(2 / 3)
    assert doubly_covered_fraction(
        [sentence], _heads(sentence), provider, 0
    ) == pytest.approx(1 / 2)


def test_mixed_scores_preserve_uncovered_l0_fallback_and_decode(tmp_path: Path) -> None:
    sentence = _sentence()
    provider = _write_provider(tmp_path / "mixed.npz", [("s", 0), ("s", 1)])
    l0 = _FixedL0((1, 0, -1))
    scorer = GroundedAttachmentScorer(BilinearConfig(rank=2, epochs=1, seed=0))
    scorer.U.fill(0.0)
    scorer.V.fill(0.0)
    scorer.b = 0.75
    expected_l0 = l0_edge_scores(l0, sentence.tokens, sentence.targets["pos"])

    mixed = mixed_edge_scores(
        scorer, l0, provider, "s", sentence.tokens, sentence.targets["pos"], 0
    )

    assert mixed[0, 3] == expected_l0[0, 3]
    assert mixed[2, 1] == expected_l0[2, 1]
    assert mixed[0, 2] == pytest.approx(0.75)
    decoded = chu_liu_edmonds(mixed)
    assert len(decoded) == 3
    assert decoded.count(0) == 1
    assert all(0 <= governor <= 3 for governor in decoded)
    assert all(governor != index + 1 for index, governor in enumerate(decoded))


def test_three_head_arms_score_the_same_covered_indices_with_different_uas() -> None:
    selected = (0, 2)
    gold = (1, 0, -1)
    predictions = {
        "l0": (1, 0, 1),
        "surface119": (-2, 0, -1),
        "grounded": (1, 0, -1),
    }
    metrics = {}
    selected_by_arm = {arm: selected for arm in predictions}
    assert len(set(selected_by_arm.values())) == 1
    for arm, heads in predictions.items():
        picked = tuple(heads[index] for index in selected_by_arm[arm])
        picked_gold = tuple(gold[index] for index in selected)
        metrics[arm] = score_prediction_sequences(
            picked, ("dep", "dep"), picked_gold, ("dep", "dep")
        )

    assert metrics["l0"]["uas"] == 0.5
    assert metrics["surface119"]["uas"] == 0.5
    assert metrics["grounded"]["uas"] == 1.0
    assert {metric["tokens"] for metric in metrics.values()} == {2}


def test_conversion_arms_run_end_to_end_on_tiny_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentence = _sentence()
    provider = _write_provider(
        tmp_path / "conversion.npz", [("s", 0), ("s", 1), ("s", 2)]
    )
    l0 = _FixedL0((1, 0, 1), label="unary")
    surface_scores = np.asarray(
        [
            [0.0, -np.inf, 3.0, 1.0],
            [3.0, 1.0, -np.inf, 1.0],
            [0.0, 1.0, 3.0, -np.inf],
        ]
    )
    grounded_scorer = GroundedAttachmentScorer(
        BilinearConfig(rank=2, epochs=1, seed=0)
    )
    grounded_scorer.U.fill(0.0)
    grounded_scorer.V.fill(0.0)
    grounded_scorer.b = 0.5
    labeler = GroundedRelationLabeler(
        RelationLabelerConfig(rank=2, epochs=1, learning_rate=0.01, seed=0)
    )
    labeler._initialize_parameters(["dep"])

    def fake_case(
        _student: Any,
        _tokens: tuple[str, ...],
        baseline: list[str],
        _pos: tuple[str, ...],
        _heads: tuple[int, ...],
        _labels: tuple[str, ...],
        _eligible: list[int],
    ) -> SimpleNamespace:
        return SimpleNamespace(predictions=tuple(baseline))

    monkeypatch.setattr(grounded_attach, "run_predicted_case_sentence", fake_case)
    attach, conversion = evaluate_test_arms(
        [sentence],
        _heads(sentence),
        {"s": sentence.targets["pos"]},
        l0,
        _FixedSurface(surface_scores),
        grounded_scorer,
        provider,
        0,
        labeler,
        0,
        _CaseStudent(),
    )

    assert tuple(attach) == ("l0", "surface119", "grounded")
    assert tuple(conversion) == ("baseline", "#121", "full_grounded")
    assert {metric["covered"]["tokens"] for metric in attach.values()} == {3}
    assert all(metric["tokens"] == 3 for metric in conversion.values())
    assert all(metric["serve_honest_morph_case_n"] == 2 for metric in conversion.values())


def test_grounded_geometry_identical_and_random_vectors(tmp_path: Path) -> None:
    identical = np.ones((5, FEATURE_DIM), dtype=np.float64)
    identical_provider = _write_provider(
        tmp_path / "identical.npz", [("same", index) for index in range(5)], identical
    )
    rng = np.random.default_rng(4)
    random = rng.normal(size=(12, FEATURE_DIM))
    random_provider = _write_provider(
        tmp_path / "random.npz", [("random", index) for index in range(12)], random
    )

    same = grounded_geometry(identical_provider, 0, sample_size=5)
    varied = grounded_geometry(random_provider, 0, sample_size=12)

    assert same["cosine"] == pytest.approx(1.0)
    assert same["effrank"] == pytest.approx(1.0)
    assert np.isfinite(varied["cosine"])
    assert np.isfinite(varied["effrank"])
    assert varied["effrank"] > 2.0


def _read(**overrides: float) -> dict[str, Any]:
    values = {
        "uas_l0": 0.50,
        "uas_surface119": 0.52,
        "uas_grounded": 0.54,
        "las_base": 0.40,
        "las_121": 0.42,
        "las_full": 0.44,
        "case_base": 0.70,
        "case_121": 0.72,
        "case_full": 0.74,
    }
    values.update(overrides)
    return preregistered_read(
        **values,
        dev_layer=2,
        covered_dependent_fraction=0.60,
        doubly_covered_fraction=0.55,
        c7_cosine=0.01,
        c7_effrank=25.0,
    )


def test_preregistered_read_fires() -> None:
    read = _read()
    assert read["verdict"] == "FIRES"
    assert read["uas_gain_vs_l0"] == pytest.approx(0.04)
    assert read["las_full_gain_vs_base"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    "overrides",
    [
        {"uas_grounded": 0.52, "uas_surface119": 0.51},
        {"uas_grounded": 0.53, "uas_surface119": 0.54},
        {"las_full": 0.42},
        {"case_full": 0.70},
    ],
    ids=("small-uas-lift", "loses-surface", "small-las-conversion", "no-case-lift"),
)
def test_preregistered_read_in_between_for_each_failure(
    overrides: dict[str, float],
) -> None:
    assert _read(**overrides)["verdict"] == "IN-BETWEEN"


@pytest.mark.parametrize("grounded", [0.50, 0.49])
def test_preregistered_read_halted(grounded: float) -> None:
    assert _read(uas_grounded=grounded)["verdict"] == "HALTED"


def test_campaign_test_guard_rejects_second_claim(
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


def test_both_dev_sweeps_structurally_precede_all_test_access() -> None:
    source = inspect.getsource(grounded_attach.build_scoreboard)
    attach_sweep = source.index("_sweep_attach_layers(")
    label_sweep = source.index("_sweep_label_layers(")
    test_provider = source.index('GroundedResidualProvider(GROUND_PATHS["test"])')
    shared_test = source.index("load_shared_test_once(")
    head_test = source.index("load_head_test_once(")

    assert attach_sweep < test_provider < shared_test
    assert label_sweep < test_provider < head_test
    assert source.index("grounded_scorer.fit(") < test_provider
    assert source.index("grounded_labeler.fit(") < test_provider
