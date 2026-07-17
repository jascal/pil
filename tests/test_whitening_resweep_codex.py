"""Tests for the DEV-only whitening hyperparameter re-sweep."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.whitening_gate_codex import (  # noqa: E402
    CHECKPOINT_LAYERS,
    FEATURE_DIM,
    GermanR3DependencyStudent,
    GroundedResidualProvider,
    HeadDeprelRecord,
    Sentence,
    fit_layer_whitening_transforms,
    head_deprel_by_sent,
)
from experiments.whitening_resweep_codex import (  # noqa: E402
    FIXED_RECIPE,
    SEED,
    compute_fixed_anchor,
    hyperparameter_grid,
    setup_train_dev,
    summarize_grid,
    sweep_attach_grid,
    sweep_labeler_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "experiments" / "whitening_resweep_codex.py"

# Independently cross-checked against data/whitening_gate_codex.json and a live
# deterministic gate run during specification preparation.
REFERENCE_LABEL_RAW = 0.7605647148000758
REFERENCE_LABEL_WHITE = 0.727307182111048
REFERENCE_ATTACH_RAW = 0.5581156316916488
REFERENCE_ATTACH_WHITE = 0.4871948608137045


def _sentence(sent_id: str, tokens: tuple[str, ...]) -> Sentence:
    size = len(tokens)
    return Sentence(
        sent_id=sent_id,
        text=" ".join(tokens),
        tokens=tokens,
        targets={
            "pos": ("NOUN",) * size,
            "morph_case": ("-",) * size,
            "morph_gnn": ("-|-",) * size,
        },
    )


def _write_provider(
    path: Path,
    sent_ids: list[str],
    token_indices: list[int],
    rows: np.ndarray,
) -> GroundedResidualProvider:
    residuals = np.repeat(np.asarray(rows)[:, None, :], 4, axis=1).astype(np.float16)
    np.savez(
        path,
        sent_ids=np.asarray(sent_ids),
        token_index=np.asarray(token_indices, dtype=np.int64),
        word_text=np.asarray(["x"] * len(sent_ids)),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=residuals,
        mean_residual=residuals,
    )
    return GroundedResidualProvider(path)


def _synthetic_setup(tmp_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    train = [
        _sentence("tr0", ("A", "B", "C", "D", "E")),
        _sentence("tr1", ("F", "G", "H", "I", "J")),
    ]
    dev = [_sentence("dv0", ("K", "L", "M", "N", "O"))]
    offsets = (1, 0, -1, -2, -3)
    relations = ("nsubj", "root", "obj", "nmod", "amod")
    train_heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                sentence.sent_id,
                sentence.text,
                sentence.tokens,
                offsets,
                relations,
            )
            for sentence in train
        ]
    )
    dev_heads = head_deprel_by_sent(
        [HeadDeprelRecord("dv0", dev[0].text, dev[0].tokens, offsets, relations)]
    )
    predicted_pos = ("NOUN", "VERB", "NOUN", "NOUN", "ADJ")
    train_pos = {sentence.sent_id: predicted_pos for sentence in train}
    dev_pos = {"dv0": predicted_pos}

    train_rows = rng.normal(scale=0.5, size=(10, FEATURE_DIM))
    dev_rows = rng.normal(scale=0.5, size=(5, FEATURE_DIM))
    for index in range(10):
        train_rows[index, index % 5] = float(index % 5 + 1)
    for index in range(5):
        dev_rows[index, index] = float(index + 1)
    train_provider = _write_provider(
        tmp_path / "train.npz",
        ["tr0"] * 5 + ["tr1"] * 5,
        list(range(5)) * 2,
        train_rows,
    )
    dev_provider = _write_provider(
        tmp_path / "dev.npz", ["dv0"] * 5, list(range(5)), dev_rows
    )
    transform = fit_layer_whitening_transforms(train_provider, 0)["v1"]
    l0_student = GermanR3DependencyStudent(window=1)
    l0_student.fit(train, train_heads, train_pos)
    return {
        "train_provider": train_provider,
        "dev_provider": dev_provider,
        "transforms": {("v1", 0): transform},
        "grounded_train": train,
        "train_heads": train_heads,
        "dev": dev,
        "dev_heads": dev_heads,
        "dev_pos": dev_pos,
        "l0_student": l0_student,
    }


def _tiny_anchor() -> dict[str, Any]:
    return {
        "raw_layer": 0,
        "whitened_variant": "v1",
        "whitened_layer": 0,
        "fixed_raw": 0.25,
        "fixed_whitened": 0.5,
    }


def _assert_summary_consistent(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    for feature in ("raw", "whitened"):
        metric = f"{feature}_dev"
        best = max(float(row[metric]) for row in rows)
        assert summary[f"{feature}_best"] == best
        argmax = summary[f"{feature}_argmax"]
        matching = [
            row
            for row in rows
            if row["lr"] == argmax["lr"]
            and row["epochs"] == argmax["epochs"]
            and row["rank"] == argmax["rank"]
        ]
        assert len(matching) == 1
        assert matching[0][metric] == best


def test_grid_runs_and_summary_argmaxes_are_consistent(tmp_path: Path) -> None:
    setup = _synthetic_setup(tmp_path)
    grid = [(0.01, 5, 16), FIXED_RECIPE, (0.1, 5, 32)]
    anchor = _tiny_anchor()

    label_rows = sweep_labeler_grid(setup, anchor, grid)
    attach_rows = sweep_attach_grid(setup, anchor, grid)
    assert len(label_rows) == len(attach_rows) == len(grid)
    assert all(set(row) == {"lr", "epochs", "rank", "raw_dev", "whitened_dev"} for row in label_rows)
    assert all(set(row) == {"lr", "epochs", "rank", "raw_dev", "whitened_dev"} for row in attach_rows)
    _assert_summary_consistent(summarize_grid(label_rows, anchor), label_rows)
    _assert_summary_consistent(summarize_grid(attach_rows, anchor), attach_rows)


def test_fixed_recipe_anchor_reproduces_real_gate_dev_numbers() -> None:
    setup = setup_train_dev()
    anchor = compute_fixed_anchor(setup)

    assert anchor["labeler"]["raw_layer"] == 1
    assert anchor["labeler"]["whitened_variant"] == "v1"
    assert anchor["labeler"]["whitened_layer"] == 2
    assert anchor["attach"]["raw_layer"] == 1
    assert anchor["attach"]["whitened_variant"] == "v2_k2"
    assert anchor["attach"]["whitened_layer"] == 2
    assert anchor["labeler"]["fixed_raw"] == pytest.approx(REFERENCE_LABEL_RAW, rel=1e-9)
    assert anchor["labeler"]["fixed_whitened"] == pytest.approx(
        REFERENCE_LABEL_WHITE, rel=1e-9
    )
    assert anchor["attach"]["fixed_raw"] == pytest.approx(REFERENCE_ATTACH_RAW, rel=1e-9)
    assert anchor["attach"]["fixed_whitened"] == pytest.approx(
        REFERENCE_ATTACH_WHITE, rel=1e-9
    )


def test_module_source_has_no_held_out_read_hooks() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    forbidden = (
        'GROUND_PATHS["test"]',
        "GROUND_PATHS['test']",
        "CampaignTestReadGuard",
        "load_shared_test_once",
        "load_head_test_once",
        "qwen3b_mwt_gsd_test_n977",
        "section5_read",
    )
    for text in forbidden:
        assert text not in source


def test_small_grid_is_deterministic(tmp_path: Path) -> None:
    setup = _synthetic_setup(tmp_path)
    grid = [FIXED_RECIPE, (0.1, 5, 16)]
    anchor = _tiny_anchor()

    first_label = sweep_labeler_grid(setup, anchor, grid)
    second_label = sweep_labeler_grid(setup, anchor, grid)
    first_attach = sweep_attach_grid(setup, anchor, grid)
    second_attach = sweep_attach_grid(setup, anchor, grid)
    assert first_label == second_label
    assert first_attach == second_attach
    assert SEED == 0


def test_grid_shape_and_fixed_recipe() -> None:
    grid = hyperparameter_grid()
    assert len(grid) == 24
    assert len(set(grid)) == 24
    assert FIXED_RECIPE in grid
