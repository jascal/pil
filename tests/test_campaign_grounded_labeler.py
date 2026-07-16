"""Unit tests for the grounded-φ relation-labeler gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.biaffine_labeler_codex import RelationLabelerConfig  # noqa: E402
from experiments.campaign_grounded_labeler import (  # noqa: E402
    CHECKPOINT_LAYERS,
    FEATURE_DIM,
    GroundedFeatureProvider,
    GroundedRelationLabeler,
    assert_uas_identity,
    is_doubly_covered,
    preregistered_read,
    select_dev_layer,
)
from experiments.german_r1_codex import Sentence  # noqa: E402
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    head_deprel_by_sent,
)


def _write_tiny_npz(path: Path) -> None:
    # Two sentences: s1 covers tokens 0,1,2; s2 covers only token 0.
    # layer slices differ so layer_index selection is observable.
    last_residual = np.zeros((4, 4, FEATURE_DIM), dtype=np.float16)
    # row 0: s1,0
    last_residual[0, 0, 0] = 1.0
    last_residual[0, 1, 0] = 10.0
    last_residual[0, 2, 0] = 100.0
    last_residual[0, 3, 0] = 1000.0
    # row 1: s1,1
    last_residual[1, 0, 1] = 2.0
    last_residual[1, 1, 1] = 20.0
    # row 2: s1,2
    last_residual[2, 0, 2] = 3.0
    last_residual[2, 1, 2] = 30.0
    # row 3: s2,0
    last_residual[3, 0, 0] = 4.0
    np.savez(
        path,
        sent_ids=np.array(["s1", "s1", "s1", "s2"], dtype="U8"),
        token_index=np.asarray([0, 1, 2, 0], dtype=np.int64),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=last_residual,
    )


def test_grounded_feature_provider_lookup_and_layer_slice(tmp_path: Path) -> None:
    npz_path = tmp_path / "tiny.npz"
    _write_tiny_npz(npz_path)

    provider0 = GroundedFeatureProvider(npz_path, layer_index=0)
    provider1 = GroundedFeatureProvider(npz_path, layer_index=1)

    assert provider0.feature_dim == FEATURE_DIM
    assert provider0.layer_index == 0
    assert provider0.layer == 8
    assert provider0.n_rows == 4
    assert provider0.n_sentences == 2
    assert provider0.covered("s1", 0) is True
    assert provider0.covered("s1", 3) is False
    assert provider0.get("s1", 3) is None
    assert provider0.get("missing", 0) is None

    vec0 = provider0.get("s1", 0)
    vec1 = provider1.get("s1", 0)
    assert vec0 is not None and vec1 is not None
    assert vec0.dtype == np.float64
    assert vec0[0] == pytest.approx(1.0)
    assert vec1[0] == pytest.approx(10.0)
    assert provider0.get("s1", 1) is not None
    assert provider0.get("s1", 1)[1] == pytest.approx(2.0)


def test_grounded_feature_provider_rejects_bad_checkpoint_order(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        sent_ids=np.array(["s1"], dtype="U8"),
        token_index=np.asarray([0], dtype=np.int64),
        checkpoint_layers=np.asarray([1, 2, 3, 4], dtype=np.int64),
        last_residual=np.zeros((1, 4, FEATURE_DIM), dtype=np.float16),
    )
    with pytest.raises(ValueError, match="checkpoint_layers"):
        GroundedFeatureProvider(path, layer_index=0)


def test_is_doubly_covered_hand_examples(tmp_path: Path) -> None:
    npz_path = tmp_path / "tiny.npz"
    _write_tiny_npz(npz_path)
    provider = GroundedFeatureProvider(npz_path, layer_index=0)
    # s1 tokens 0,1,2 covered; size=4 so token 3 is uncovered.
    size = 4
    # both endpoints covered: dep=0 → gov=1 (offset +1)
    assert is_doubly_covered("s1", 0, 1, size, provider) is True
    # only dependent covered: dep=0 → gov=3 (offset +3), governor uncovered
    assert is_doubly_covered("s1", 0, 3, size, provider) is False
    # ROOT always covered regardless of dependent coverage
    assert is_doubly_covered("s1", 3, 0, size, provider) is True
    # out-of-sentence predicted offset is not doubly-covered
    assert is_doubly_covered("s1", 0, 99, size, provider) is False


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


def test_grounded_predict_falls_back_to_unary_when_governor_uncovered(
    tmp_path: Path,
) -> None:
    # Cover only dependent at index 0 and leave governor (index 1) uncovered.
    path = tmp_path / "partial.npz"
    last_residual = np.zeros((1, 4, FEATURE_DIM), dtype=np.float16)
    last_residual[0, 0, :] = 0.1
    np.savez(
        path,
        sent_ids=np.array(["a"], dtype="U8"),
        token_index=np.asarray([0], dtype=np.int64),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=last_residual,
    )
    # Fit needs at least one kept arc: use a second sentence fully covered.
    full_path = tmp_path / "full.npz"
    last_full = np.zeros((3, 4, FEATURE_DIM), dtype=np.float16)
    last_full[:, 0, :] = np.linspace(0.0, 1.0, FEATURE_DIM, dtype=np.float16)
    last_full[1, 0, 0] = 1.0
    last_full[2, 0, 0] = -1.0
    np.savez(
        full_path,
        sent_ids=np.array(["b", "b", "b"], dtype="U8"),
        token_index=np.asarray([0, 1, 2], dtype=np.int64),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=last_full,
    )
    fit_provider = GroundedFeatureProvider(full_path, layer_index=0)
    sentences = [
        _sentence("b", ("Kind", "sieht", "Hund")),
    ]
    heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "b",
                sentences[0].text,
                sentences[0].tokens,
                (1, 0, -1),
                ("nsubj", "root", "obj"),
            ),
        ]
    )
    predicted_pos = {"b": ("NOUN", "VERB", "NOUN")}
    labeler = GroundedRelationLabeler(
        RelationLabelerConfig(rank=4, epochs=2, learning_rate=0.1, seed=0)
    )
    labeler.fit(sentences, heads, predicted_pos, fit_provider)

    predict_provider = GroundedFeatureProvider(path, layer_index=0)
    unary = ("FALLBACK_NSUBJ", "root", "FALLBACK_OBJ")
    # head 0→1: dependent covered, governor not → unary fallback
    predicted = labeler.predict(
        ("Kind", "sieht", "Hund"),
        ("NOUN", "VERB", "NOUN"),
        (1, 0, -1),
        "a",
        predict_provider,
        unary,
    )
    assert predicted[0] == "FALLBACK_NSUBJ"
    assert predicted[1] == "root"
    # head 2→1 also uncovered endpoints → unary
    assert predicted[2] == "FALLBACK_OBJ"


def test_assert_uas_identity_passes_and_fails() -> None:
    assert_uas_identity(
        {
            "unary": (1, 0, -1),
            "surface119": (1, 0, -1),
            "grounded": (1, 0, -1),
        }
    )
    with pytest.raises(AssertionError, match="UAS invariant"):
        assert_uas_identity(
            {
                "unary": (1, 0, -1),
                "surface119": (1, 0, -1),
                "grounded": (0, 0, -1),
            }
        )


def test_select_dev_layer_max_accuracy_smallest_index_tiebreak() -> None:
    rows = [
        {
            "layer_index": 0,
            "layer": 8,
            "dev_deprel_only_accuracy": 0.50,
            "dev_doubly_covered_fraction": 0.4,
        },
        {
            "layer_index": 1,
            "layer": 17,
            "dev_deprel_only_accuracy": 0.55,
            "dev_doubly_covered_fraction": 0.4,
        },
        {
            "layer_index": 2,
            "layer": 26,
            "dev_deprel_only_accuracy": 0.55,
            "dev_doubly_covered_fraction": 0.4,
        },
        {
            "layer_index": 3,
            "layer": 35,
            "dev_deprel_only_accuracy": 0.40,
            "dev_doubly_covered_fraction": 0.4,
        },
    ]
    # max accuracy 0.55 shared by indices 1 and 2 → pick smallest index 1
    assert select_dev_layer(rows) == 1
    # pure max
    rows[2]["dev_deprel_only_accuracy"] = 0.60
    assert select_dev_layer(rows) == 2


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


def _read(
    unary: dict[str, object],
    surface: dict[str, object],
    grounded: dict[str, object],
) -> dict[str, object]:
    return preregistered_read(
        unary,
        surface,
        grounded,
        dev_layer=17,
        dev_layer_index=1,
        doubly_covered_fraction={"dev": 0.5, "test": 0.48},
    )


def test_preregistered_read_fires_when_every_condition_holds() -> None:
    unary = _metric(0.50, case=0.60, pairwise=0.70, local=0.70)
    surface = _metric(0.48, case=0.60, pairwise=0.68, local=0.70)
    grounded = _metric(0.54, case=0.62, pairwise=0.79, local=0.72)

    read = _read(unary, surface, grounded)

    assert read["verdict"] == "FIRES"
    assert read["deprel_las_gain_vs_unary"] == pytest.approx(0.04)
    assert read["gain_vs_surface119"] == pytest.approx(0.06)
    assert read["partition_gains"]["pairwise_sensitive"] > read["partition_gains"]["local"]
    assert read["dev_layer"] == 17
    assert read["dev_layer_index"] == 1


def test_preregistered_read_halts_for_nonpositive_gain() -> None:
    unary = _metric(0.50, case=0.60, pairwise=0.70, local=0.70)
    surface = _metric(0.48, case=0.60, pairwise=0.68, local=0.70)
    grounded = _metric(0.49, case=0.62, pairwise=0.80, local=0.71)

    read = _read(unary, surface, grounded)

    assert read["verdict"] == "HALTED"
    assert read["deprel_las_gain_vs_unary"] == pytest.approx(-0.01)


@pytest.mark.parametrize(
    ("grounded", "expected_fragment"),
    [
        (
            _metric(0.52, case=0.62, pairwise=0.79, local=0.71),
            "strict LAS gain is below +0.03",
        ),
        (
            _metric(0.54, case=0.62, pairwise=0.71, local=0.79),
            "pairwise-sensitive gain is not greater than local gain",
        ),
        (
            _metric(0.54, case=0.60, pairwise=0.79, local=0.71),
            "serve-honest case does not improve vs unary",
        ),
        (
            _metric(0.54, case=0.62, pairwise=0.79, local=0.71),
            "grounded does not beat surface-#119 on strict LAS",
        ),
    ],
    ids=("small-gain", "diffuse", "case-not-improved", "fails-surface119"),
)
def test_preregistered_read_in_between_for_each_individual_condition(
    grounded: dict[str, object],
    expected_fragment: str,
) -> None:
    unary = _metric(0.50, case=0.60, pairwise=0.70, local=0.70)
    # For the surface-fail case, surface must match or beat grounded LAS.
    if "surface-#119" in expected_fragment:
        surface = _metric(0.55, case=0.60, pairwise=0.68, local=0.70)
    else:
        surface = _metric(0.48, case=0.60, pairwise=0.68, local=0.70)

    read = _read(unary, surface, grounded)

    assert read["verdict"] == "IN-BETWEEN"
    assert expected_fragment in read["text"]
