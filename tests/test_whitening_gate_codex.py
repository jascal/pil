from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.attach_levers_codex import (  # noqa: E402
    chu_liu_edmonds,
    governor_columns_to_offsets,
)
from experiments.german_r3_codex import CampaignTestReadGuard  # noqa: E402
from experiments.grounded_labeler_codex import (  # noqa: E402
    FEATURE_DIM,
    GroundedResidualProvider,
)
from experiments.whitening_gate_codex import (  # noqa: E402
    ATTACH_ARMS,
    LAYER_INDICES,
    VARIANTS,
    WhitenedResidualProvider,
    assert_identical_arc_sets,
    assert_matched_labeler_uas,
    centered_participation_ratio,
    fit_layer_whitening_transforms,
    pin_a_effrank,
    pin_a_row_keys,
    pin_b_edge_log_probs,
    section5_read,
    select_variant_layer,
)


def _write_provider(path: Path, matrix: np.ndarray) -> GroundedResidualProvider:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
        raise ValueError("fixture matrix must have FEATURE_DIM columns")
    residuals = np.repeat(values[:, None, :], 4, axis=1).astype(np.float16)
    np.savez(
        path,
        sent_ids=np.asarray(["fixture"] * len(values)),
        token_index=np.arange(len(values), dtype=np.int64),
        word_text=np.asarray(["x"] * len(values)),
        checkpoint_layers=np.asarray([8, 17, 26, 35], dtype=np.int64),
        last_residual=residuals,
        mean_residual=residuals,
    )
    return GroundedResidualProvider(path)


def test_v1_uses_train_only_and_matches_numpy_zscore(tmp_path: Path) -> None:
    train = np.zeros((4, FEATURE_DIM), dtype=np.float64)
    train[:, 0] = [1.0, 3.0, 5.0, 7.0]
    train[:, 1] = [4.0, 4.0, 4.0, 4.0]
    train[:, 2] = [-2.0, 0.0, 2.0, 4.0]
    held_out = np.zeros((2, FEATURE_DIM), dtype=np.float64)
    held_out[:, 0] = [1000.0, 2000.0]
    held_out[:, 1] = [-900.0, 900.0]
    train_provider = _write_provider(tmp_path / "train.npz", train)
    dev_provider = _write_provider(tmp_path / "dev.npz", held_out)

    transform = fit_layer_whitening_transforms(train_provider, 0)["v1"]
    adapter = WhitenedResidualProvider(dev_provider, transform)

    assert transform.fit_split == "train"
    assert transform.input_mean == pytest.approx(train.mean(axis=0))
    assert transform.sigma == pytest.approx(train.std(axis=0))
    assert transform.input_mean[0] != pytest.approx(held_out[:, 0].mean())
    expected = np.divide(
        held_out[0] - train.mean(axis=0),
        train.std(axis=0),
        out=np.zeros(FEATURE_DIM),
        where=train.std(axis=0) > 0.0,
    )
    actual = adapter.residual("fixture", 0, 0)
    assert actual == pytest.approx(expected)
    assert actual[1] == 0.0
    assert np.all(np.isfinite(actual))


def test_v2_removes_dominant_pc_and_raises_effective_rank(tmp_path: Path) -> None:
    rows = 40
    t = np.linspace(-2.0, 2.0, rows)
    matrix = np.zeros((rows, FEATURE_DIM), dtype=np.float64)
    matrix[:, 0] = 120.0 * t
    matrix[:, 1] = np.sin(np.arange(rows) * 0.7)
    matrix[:, 2] = np.cos(np.arange(rows) * 1.1)
    matrix[:, 3] = np.sin(np.arange(rows) * 1.7)
    provider = _write_provider(tmp_path / "dominant.npz", matrix)
    transform = fit_layer_whitening_transforms(provider, 0)["v2_k1"]
    whitened = WhitenedResidualProvider(provider, transform)
    keys = tuple(("fixture", index) for index in range(rows))

    raw_rank = pin_a_effrank(provider, 0, keys)["effrank"]
    whitened_rank = pin_a_effrank(whitened, 0, keys)["effrank"]

    assert abs(transform.components[0, 0]) > 0.999
    assert whitened_rank > raw_rank
    assert whitened_rank > 2.0


class _TinyProvider:
    def __init__(self, covered: set[int], vectors: np.ndarray | None = None) -> None:
        self.covered_indices = covered
        self.vectors = vectors
        count = len(vectors) if vectors is not None else max(covered, default=-1) + 1
        self._row_by_key = {("s", index): index for index in range(count)}

    @staticmethod
    def _validate_layer(layer: int) -> None:
        if layer != 0:
            raise ValueError("tiny provider only supports layer zero")

    def covered(self, sent_id: str, token_index: int) -> bool:
        return sent_id == "s" and token_index in self.covered_indices

    def residual(self, sent_id: str, token_index: int, layer: int) -> np.ndarray:
        self._validate_layer(layer)
        if sent_id != "s" or self.vectors is None:
            raise KeyError((sent_id, token_index))
        return np.asarray(self.vectors[token_index], dtype=np.float64)


class _FixedL0:
    def __init__(self, offsets: tuple[int, ...]) -> None:
        self.offsets = offsets

    def predict_heads(
        self, _tokens: tuple[str, ...], _predicted_pos: tuple[str, ...]
    ) -> tuple[int, ...]:
        return self.offsets


class _MappedScorer:
    def __init__(self, scores: dict[tuple[int, int], float]) -> None:
        self.scores = scores

    def score_edge(
        self,
        _provider: _TinyProvider,
        _sent_id: str,
        dependent: int,
        governor: int,
        _layer: int,
    ) -> float:
        return self.scores[(dependent, governor)]


def _manual_log_softmax(row: np.ndarray) -> np.ndarray:
    finite = np.isfinite(row)
    maximum = row[finite].max()
    result = np.full_like(row, -np.inf, dtype=np.float64)
    result[finite] = row[finite] - maximum - np.log(np.exp(row[finite] - maximum).sum())
    return result


def test_pin_b_log_probability_matrix_matches_hand_example_and_decodes() -> None:
    tokens = ("a", "b", "c", "d")
    pos = ("NOUN",) * 4
    provider = _TinyProvider({0, 1, 2})
    l0 = _FixedL0((1, 0, -1, -1))
    scorer = _MappedScorer(
        {
            (0, 1): 2.0,
            (0, 2): 0.0,
            (1, 0): 1.0,
            (1, 2): 3.0,
            (2, 0): -1.0,
            (2, 1): 0.0,
        }
    )
    # These are the L0 heuristic magnitudes calculated from the fixed offsets
    # above.  Self cells are invalid before either softmax.
    l0_raw = np.asarray(
        [
            [0.06, -np.inf, 1.0, 0.05333333333333334, 0.035833333333333335],
            [1.0, 0.055, -np.inf, 0.055, 0.03666666666666667],
            [0.06, 0.05333333333333334, 1.0, -np.inf, 0.03833333333333333],
            [0.06, 0.035833333333333335, 0.05333333333333334, 1.0, -np.inf],
        ]
    )
    expected = np.stack([_manual_log_softmax(row) for row in l0_raw])
    bilinear = {
        0: ((2, 3), np.asarray([2.0, 0.0])),
        1: ((1, 3), np.asarray([1.0, 3.0])),
        2: ((1, 2), np.asarray([-1.0, 0.0])),
    }
    for dependent, (columns, raw) in bilinear.items():
        expected[dependent, list(columns)] = _manual_log_softmax(raw)
        expected[dependent] = _manual_log_softmax(expected[dependent])

    actual = pin_b_edge_log_probs(scorer, l0, provider, "s", tokens, pos, 0)

    assert actual == pytest.approx(expected, abs=1e-12)
    assert np.exp(np.where(np.isfinite(actual), actual, -np.inf)).sum(axis=1) == pytest.approx(
        np.ones(4), abs=1e-12
    )
    decoded = chu_liu_edmonds(actual)
    assert decoded == (2, 0, 2, 3)


def test_pin_b_log_probs_drive_cle_cycle_breaking_to_single_root_tree() -> None:
    tokens = ("a", "b", "c")
    pos = ("NOUN",) * 3
    provider = _TinyProvider({0, 1, 2})
    l0 = _FixedL0((1, -1, -1))
    scorer = _MappedScorer(
        {
            (0, 1): 8.0,
            (0, 2): 0.0,
            (1, 0): 8.0,
            (1, 2): 0.0,
            (2, 0): 8.0,
            (2, 1): 0.0,
        }
    )

    log_probs = pin_b_edge_log_probs(scorer, l0, provider, "s", tokens, pos, 0)
    naive = tuple(int(column) for column in np.argmax(log_probs, axis=1))
    decoded = chu_liu_edmonds(log_probs)
    offsets = governor_columns_to_offsets(decoded)

    assert naive[:2] == (2, 1)  # token 0 <-> token 1 is the naive cycle.
    assert decoded.count(0) == 1
    assert all(governor != index + 1 for index, governor in enumerate(decoded))
    assert len(offsets) == 3


def test_pin_a_formula_matches_independent_singular_value_computation() -> None:
    matrix = np.asarray(
        [[1.0, 0.0, 2.0], [0.0, 2.0, 1.0], [3.0, 1.0, 0.0], [2.0, 4.0, 3.0]]
    )
    singular = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
    expected = np.square(np.square(singular).sum()) / np.power(singular, 4).sum()

    assert centered_participation_ratio(matrix) == pytest.approx(expected)


def test_pin_a_uses_exactly_800_rows_with_seed_zero() -> None:
    vectors = np.column_stack(
        (np.arange(805, dtype=np.float64), np.arange(805, dtype=np.float64) ** 2)
    )
    provider = _TinyProvider(set(range(805)), vectors)

    keys = pin_a_row_keys(provider, sample_size=800, seed=0)
    repeated = pin_a_row_keys(provider, sample_size=800, seed=0)
    metric = pin_a_effrank(provider, 0, keys)

    assert keys == repeated
    assert len(keys) == len(set(keys)) == 800
    assert metric["sample_size"] == 800
    assert metric["seed"] == 0


def test_matched_uas_and_arc_set_assertions_raise_on_mismatch() -> None:
    metrics = {
        arm: {"uas": 0.5}
        for arm in ("unary", "surface119", "raw121", "whitened")
    }
    metrics["whitened"] = {"uas": 0.6}
    with pytest.raises(AssertionError, match="UAS invariant"):
        assert_matched_labeler_uas(metrics)

    common = (("s", 0), ("s", 1))
    arc_sets = {arm: common for arm in ATTACH_ARMS}
    arc_sets["raw122"] = (("s", 0),)
    with pytest.raises(AssertionError, match="arc set differs"):
        assert_identical_arc_sets(arc_sets)


def _label_metrics(
    *, unary: float = 0.50, raw: float = 0.51, white: float = 0.54
) -> dict[str, dict[str, float]]:
    return {
        "unary": {"las_strict": unary},
        "surface119": {"las_strict": 0.52},
        "raw121": {"las_strict": raw},
        "whitened": {"las_strict": white},
    }


def _attach_metrics(
    *, l0: float = 0.50, raw: float = 0.51, white: float = 0.54
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "l0": {"covered": {"uas": l0}},
        "surface119": {"covered": {"uas": 0.52}},
        "raw122": {"covered": {"uas": raw}},
        "whitened": {"covered": {"uas": white}},
    }


def _conversion_metrics(
    *, baseline: float = 0.40, raw: float = 0.41, white: float = 0.44, case: float = 0.91
) -> dict[str, dict[str, float]]:
    return {
        "baseline": {"las_strict": baseline, "serve_honest_morph_case": 0.75},
        "#121_labels": {"las_strict": 0.42, "serve_honest_morph_case": 0.80},
        "raw122_full": {"las_strict": raw, "serve_honest_morph_case": 0.81},
        "full_grounded_labels": {
            "las_strict": white,
            "serve_honest_morph_case": case,
        },
    }


def _section_read(
    labeler: dict[str, Any] | None = None,
    attachment: dict[str, Any] | None = None,
    conversion: dict[str, Any] | None = None,
    *,
    robust: bool = True,
) -> dict[str, Any]:
    selection = {
        "labeler": {"variant": "v2_k1", "layer": 1},
        "attach": {"variant": "v1", "layer": 2},
    }
    geometry = {
        str(layer): {
            "raw": {"effrank": 1.2},
            "v1": {"effrank": 20.0},
            "v2_k1": {"effrank": 30.0},
            "v2_k2": {"effrank": 40.0},
        }
        for layer in LAYER_INDICES
    }
    return section5_read(
        labeler or _label_metrics(),
        attachment or _attach_metrics(),
        conversion or _conversion_metrics(),
        selection,
        geometry,
        {"labeler": "v2 wins", "attach": "v1 wins"},
        cross_vendor_robust=robust,
    )


def test_section5_read_fires_and_records_mechanism_fields() -> None:
    read = _section_read()

    assert read["verdict"] == "FIRES"
    assert read["fired_primitive"] == "labeler+attach"
    assert read["labeler_delta"] == pytest.approx(0.03)
    assert read["attach_uas_gain"] == pytest.approx(0.04)
    assert read["attach_conv_las_gain"] == pytest.approx(0.04)
    assert read["effrank_raw"] == {"labeler": 1.2, "attach": 1.2}
    assert read["effrank_whitened"] == {"labeler": 30.0, "attach": 20.0}


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        (
            {
                "labeler": _label_metrics(white=0.52),
                "attachment": _attach_metrics(white=0.52),
                "conversion": _conversion_metrics(white=0.42),
            },
            "gain-below-bars",
        ),
        ({"robust": False}, "cross-vendor-straddle"),
    ],
    ids=("improves-without-bar", "bar-without-robustness"),
)
def test_section5_read_in_between(kwargs: dict[str, Any], reason: str) -> None:
    del reason
    assert _section_read(**kwargs)["verdict"] == "IN-BETWEEN"


def test_section5_read_halted_when_whitening_is_no_better_on_both_primitives() -> None:
    read = _section_read(
        _label_metrics(raw=0.52, white=0.52),
        _attach_metrics(raw=0.53, white=0.52),
        _conversion_metrics(raw=0.43, white=0.42),
    )

    assert read["verdict"] == "HALTED"
    assert "R4 LLM-hybrid" in read["text"]


@pytest.mark.parametrize(
    ("case", "baseline", "expected"),
    [
        (0.91, 0.75, "met"),
        (0.80, 0.75, "plateau language"),
        (0.75, 0.75, "diagnose (at or below baseline)"),
        (0.74, 0.70, "diagnose"),
    ],
)
def test_section5_through_line_clauses(
    case: float, baseline: float, expected: str
) -> None:
    conversion = _conversion_metrics(case=case)
    conversion["baseline"]["serve_honest_morph_case"] = baseline

    assert _section_read(conversion=conversion)["through_line"] == expected


def test_campaign_guard_rejects_second_claim() -> None:
    guard = CampaignTestReadGuard()
    guard.claim("shared")
    guard.claim("head_deprel")
    guard.assert_complete()

    with pytest.raises(RuntimeError, match="may be read only once"):
        guard.claim("shared")


class _DevCurve(dict[tuple[str, int], float]):
    def __init__(self, values: dict[tuple[str, int], float]) -> None:
        super().__init__(values)
        self.keys_read: list[tuple[str, int]] = []

    def __getitem__(self, key: tuple[str, int]) -> float:
        self.keys_read.append(key)
        return super().__getitem__(key)


def test_dev_only_variant_layer_selection_consumes_only_dev_curve() -> None:
    curve = _DevCurve(
        {
            (variant, layer): (0.8 if (variant, layer) == ("v2_k1", 2) else 0.5)
            for variant in VARIANTS
            for layer in LAYER_INDICES
        }
    )
    held_out = SimpleNamespace(
        accuracy=property(lambda _self: pytest.fail("TEST accuracy was touched"))
    )

    selected = select_variant_layer(curve)

    assert selected == ("v2_k1", 2)
    assert set(curve.keys_read) == set(curve)
    assert held_out is not None  # The held-out counterpart is never an input.


def test_dev_selection_ties_prefer_lower_complexity_then_lower_layer() -> None:
    tied = {
        (variant, layer): 0.7 for variant in VARIANTS for layer in LAYER_INDICES
    }
    assert select_variant_layer(tied) == ("v1", 0)
