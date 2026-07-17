"""Unit tests for the DEV-only whitening hyperparameter re-sweep diagnostic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.campaign_grounded_labeler import (  # noqa: E402
    CHECKPOINT_LAYERS,
    FEATURE_DIM,
)
from experiments.campaign_whitening_gate import (  # noqa: E402
    DEV_NPZ,
    TRAIN_NPZ,
)
from experiments.german_r1_codex import Sentence  # noqa: E402
from experiments.german_r3_codex import GermanR3DependencyStudent  # noqa: E402
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    head_deprel_by_sent,
)
from experiments.whitening_resweep_grok import (  # noqa: E402
    FIXED_EPOCHS,
    FIXED_LR,
    FIXED_RANK,
    GATE_JSON,
    SEED,
    build_providers,
    find_fixed_row,
    fit_eval_attach_config,
    fit_eval_labeler_config,
    iter_grid,
    load_gate_selection,
    pick_best_row,
    sweep_attach_grid,
    sweep_labeler_grid,
)

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "whitening_resweep_grok.py"

# Forbidden substrings proving the diagnostic never opens the TEST split.
_FORBIDDEN_SOURCE_SUBSTRINGS = (
    "TEST_NPZ",
    "CampaignTestReadGuard",
    "load_shared_test_once",
    "load_head_test_once",
    "qwen3b_mwt_gsd_test",
)


def _write_npz(
    path: Path,
    *,
    sent_ids: list[str],
    token_indices: list[int],
    residuals: np.ndarray,
) -> None:
    n = len(sent_ids)
    assert residuals.shape == (n, FEATURE_DIM)
    last = np.zeros((n, 4, FEATURE_DIM), dtype=np.float64)
    for layer in range(4):
        last[:, layer, :] = residuals
    np.savez(
        path,
        sent_ids=np.array(sent_ids, dtype="U16"),
        token_index=np.asarray(token_indices, dtype=np.int64),
        checkpoint_layers=np.asarray(CHECKPOINT_LAYERS, dtype=np.int64),
        last_residual=last,
    )


def _sentence(sent_id: str, tokens: tuple[str, ...]) -> Sentence:
    n = len(tokens)
    return Sentence(
        sent_id=sent_id,
        text=" ".join(tokens),
        tokens=tokens,
        targets={
            "pos": ("NOUN",) * n,
            "morph_case": ("-",) * n,
            "morph_gnn": ("-|-",) * n,
        },
    )


def _synthetic_bundle(tmp_path: Path) -> dict:
    """Tiny TRAIN/DEV fixtures with FEATURE_DIM residual columns + fitted L0.

    Sentences are length-5 so that with window=1 both LOCAL and LONG arcs exist
    (L0 fit requires non-empty LONG backoff tables).
    """
    rng = np.random.default_rng(0)

    # 5 tokens: offsets mix ROOT, LOCAL (|off|<=1), and LONG (|off|>1).
    # tree: 0→1 (+1 local), 1→ROOT, 2→1 (-1 local), 3→1 (-2 long), 4→1 (-3 long)
    head_offsets = (1, 0, -1, -2, -3)
    deprels = ("nsubj", "root", "obj", "nmod", "amod")
    train_sents = [
        _sentence("tr0", ("A", "B", "C", "D", "E")),
        _sentence("tr1", ("F", "G", "H", "I", "J")),
    ]
    dev_sents = [
        _sentence("dv0", ("K", "L", "M", "N", "O")),
    ]
    train_heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "tr0",
                train_sents[0].text,
                train_sents[0].tokens,
                head_offsets,
                deprels,
            ),
            HeadDeprelRecord(
                "tr1",
                train_sents[1].text,
                train_sents[1].tokens,
                head_offsets,
                deprels,
            ),
        ]
    )
    dev_heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "dv0",
                dev_sents[0].text,
                dev_sents[0].tokens,
                head_offsets,
                deprels,
            ),
        ]
    )
    pos5 = ("NOUN", "VERB", "NOUN", "NOUN", "ADJ")
    train_pos = {"tr0": pos5, "tr1": pos5}
    dev_pos = {"dv0": pos5}

    # Distinct residual rows per token (FEATURE_DIM).
    train_rows = []
    train_sent_ids: list[str] = []
    train_tok_idx: list[int] = []
    for sid in ("tr0", "tr1"):
        for ti in range(5):
            train_sent_ids.append(sid)
            train_tok_idx.append(ti)
            row = rng.normal(scale=0.5, size=FEATURE_DIM)
            row[ti] = float(ti + 1)
            train_rows.append(row)
    train_residuals = np.stack(train_rows)

    dev_rows = []
    for ti in range(5):
        row = rng.normal(scale=0.5, size=FEATURE_DIM)
        row[ti] = float(ti + 1)
        dev_rows.append(row)
    dev_residuals = np.stack(dev_rows)

    train_npz = tmp_path / "train.npz"
    dev_npz = tmp_path / "dev.npz"
    _write_npz(
        train_npz,
        sent_ids=train_sent_ids,
        token_indices=train_tok_idx,
        residuals=train_residuals,
    )
    _write_npz(
        dev_npz,
        sent_ids=["dv0"] * 5,
        token_indices=[0, 1, 2, 3, 4],
        residuals=dev_residuals,
    )

    # window=1 → offsets ±2,±3 are LONG so long_offset_table is non-empty.
    l0 = GermanR3DependencyStudent(window=1)
    l0.fit(train_sents, train_heads, train_pos)

    return {
        "train": train_sents,
        "train_heads": train_heads,
        "train_pos": train_pos,
        "dev": dev_sents,
        "dev_heads": dev_heads,
        "dev_pos": dev_pos,
        "l0_student": l0,
        "train_npz": train_npz,
        "dev_npz": dev_npz,
        "layer_index": 0,
        "variant": "v1",
    }


def test_iter_grid_has_24_cells_including_fixed() -> None:
    cells = iter_grid()
    assert len(cells) == 24
    assert (FIXED_LR, FIXED_EPOCHS, FIXED_RANK) in cells


def test_pick_best_row_tie_break() -> None:
    rows = [
        {"lr": 0.1, "epochs": 15, "rank": 32, "score": 0.5},
        {"lr": 0.01, "epochs": 5, "rank": 16, "score": 0.5},  # wins: smaller epochs/rank/lr
        {"lr": 0.04, "epochs": 5, "rank": 32, "score": 0.5},
        {"lr": 0.3, "epochs": 40, "rank": 16, "score": 0.49},
    ]
    best = pick_best_row(rows)
    assert best["lr"] == 0.01
    assert best["epochs"] == 5
    assert best["rank"] == 16


def test_grid_runs_labeler_on_synthetic(tmp_path: Path) -> None:
    """Full (or reduced) grid completes with well-formed finite-score rows."""
    fx = _synthetic_bundle(tmp_path)
    train_provider, dev_provider = build_providers(
        train_npz=fx["train_npz"],
        dev_npz=fx["dev_npz"],
        layer_index=fx["layer_index"],
        feature_set="raw",
    )
    # Representative subset of the real grid constants (cheap on tiny data).
    subset = [(0.04, 5, 16), (0.1, 5, 16), (0.04, 5, 32)]
    rows = sweep_labeler_grid(
        train=fx["train"],
        train_heads=fx["train_heads"],
        train_pos=fx["train_pos"],
        train_provider=train_provider,
        dev=fx["dev"],
        dev_heads=fx["dev_heads"],
        dev_pos=fx["dev_pos"],
        l0_student=fx["l0_student"],
        dev_provider=dev_provider,
        feature_set="raw",
        grid=subset,
    )
    assert len(rows) == len(subset)
    for row, (lr, epochs, rank) in zip(rows, subset, strict=True):
        assert set(row.keys()) == {"lr", "epochs", "rank", "score"}
        assert row["lr"] == lr
        assert row["epochs"] == epochs
        assert row["rank"] == rank
        assert np.isfinite(row["score"])
        assert 0.0 <= row["score"] <= 1.0


def test_grid_runs_attach_whitened_on_synthetic(tmp_path: Path) -> None:
    fx = _synthetic_bundle(tmp_path)
    train_provider, dev_provider = build_providers(
        train_npz=fx["train_npz"],
        dev_npz=fx["dev_npz"],
        layer_index=fx["layer_index"],
        feature_set="whitened",
        variant=fx["variant"],
    )
    subset = [(0.04, 5, 16), (0.01, 5, 16)]
    rows = sweep_attach_grid(
        train=fx["train"],
        train_heads=fx["train_heads"],
        train_provider=train_provider,
        dev=fx["dev"],
        dev_heads=fx["dev_heads"],
        dev_pos=fx["dev_pos"],
        l0_student=fx["l0_student"],
        dev_provider=dev_provider,
        feature_set="whitened",
        grid=subset,
    )
    assert len(rows) == 2
    for row in rows:
        assert set(row.keys()) == {"lr", "epochs", "rank", "score"}
        assert np.isfinite(row["score"])
    # fixed cell look-up works when fixed is in the grid
    full_like = [
        {"lr": FIXED_LR, "epochs": FIXED_EPOCHS, "rank": FIXED_RANK, "score": 0.4},
        {"lr": 0.1, "epochs": 5, "rank": 16, "score": 0.3},
    ]
    fixed = find_fixed_row(full_like)
    assert fixed["score"] == 0.4


def test_fixed_recipe_anchor_reproduces_gate_dev(tmp_path: Path) -> None:
    """Real-data fixed-recipe whitened cell matches campaign_whitening_gate.json DEV."""
    del tmp_path  # unused; real paths required
    if not TRAIN_NPZ.is_file() or not DEV_NPZ.is_file() or not GATE_JSON.is_file():
        pytest.skip("real grounded npz / gate JSON not present")

    gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
    selection = load_gate_selection(GATE_JSON)

    # Expected DEV scores from the gate's persisted sweep rows (not hardcoded).
    lab_sel = selection["labeler"]
    att_sel = selection["attach"]
    expected_lab = None
    for row in gate["dev_labeler_sweep"]:
        if (
            row["variant"] == lab_sel["variant"]
            and int(row["layer_index"]) == lab_sel["layer_index"]
        ):
            expected_lab = float(row["dev_deprel_only_accuracy"])
            break
    expected_att = None
    for row in gate["dev_attach_sweep"]:
        if (
            row["variant"] == att_sel["variant"]
            and int(row["layer_index"]) == att_sel["layer_index"]
        ):
            expected_att = float(row["dev_uas"])
            break
    assert expected_lab is not None
    assert expected_att is not None

    # Load TRAIN+DEV + L0 the same way the re-sweep does (DEV only; no TEST).
    from experiments.whitening_resweep_grok import load_train_dev_bundle

    bundle = load_train_dev_bundle()

    # Whitened labeler at fixed recipe.
    train_lab, dev_lab = build_providers(
        train_npz=TRAIN_NPZ,
        dev_npz=DEV_NPZ,
        layer_index=lab_sel["layer_index"],
        feature_set="whitened",
        variant=lab_sel["variant"],
    )
    lab_score = fit_eval_labeler_config(
        lr=FIXED_LR,
        epochs=FIXED_EPOCHS,
        rank=FIXED_RANK,
        train=bundle["train"],
        train_heads=bundle["train_heads"],
        train_pos=bundle["train_pos"],
        train_provider=train_lab,
        dev=bundle["dev"],
        dev_heads=bundle["dev_heads"],
        dev_pos=bundle["dev_pos"],
        l0_student=bundle["l0_student"],
        dev_provider=dev_lab,
    )
    assert lab_score == pytest.approx(expected_lab, abs=1e-6)

    # Whitened attach at fixed recipe.
    train_att, dev_att = build_providers(
        train_npz=TRAIN_NPZ,
        dev_npz=DEV_NPZ,
        layer_index=att_sel["layer_index"],
        feature_set="whitened",
        variant=att_sel["variant"],
    )
    att_score = fit_eval_attach_config(
        lr=FIXED_LR,
        epochs=FIXED_EPOCHS,
        rank=FIXED_RANK,
        train=bundle["train"],
        train_heads=bundle["train_heads"],
        train_provider=train_att,
        dev=bundle["dev"],
        dev_heads=bundle["dev_heads"],
        dev_pos=bundle["dev_pos"],
        l0_student=bundle["l0_student"],
        dev_provider=dev_att,
    )
    assert att_score == pytest.approx(expected_att, abs=1e-6)


def test_no_test_read_in_module_source() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in _FORBIDDEN_SOURCE_SUBSTRINGS:
        assert forbidden not in source, f"module must not contain {forbidden!r}"


def test_determinism_labeler_and_attach(tmp_path: Path) -> None:
    """Same seed/config/fixture → bit-identical DEV scores on two independent runs."""
    fx = _synthetic_bundle(tmp_path)
    train_provider, dev_provider = build_providers(
        train_npz=fx["train_npz"],
        dev_npz=fx["dev_npz"],
        layer_index=fx["layer_index"],
        feature_set="raw",
    )
    kwargs_lab = dict(
        lr=0.04,
        epochs=5,
        rank=16,
        train=fx["train"],
        train_heads=fx["train_heads"],
        train_pos=fx["train_pos"],
        train_provider=train_provider,
        dev=fx["dev"],
        dev_heads=fx["dev_heads"],
        dev_pos=fx["dev_pos"],
        l0_student=fx["l0_student"],
        dev_provider=dev_provider,
    )
    s1 = fit_eval_labeler_config(**kwargs_lab)
    s2 = fit_eval_labeler_config(**kwargs_lab)
    assert s1 == s2
    assert SEED == 0

    kwargs_att = dict(
        lr=0.04,
        epochs=5,
        rank=16,
        train=fx["train"],
        train_heads=fx["train_heads"],
        train_provider=train_provider,
        dev=fx["dev"],
        dev_heads=fx["dev_heads"],
        dev_pos=fx["dev_pos"],
        l0_student=fx["l0_student"],
        dev_provider=dev_provider,
    )
    a1 = fit_eval_attach_config(**kwargs_att)
    a2 = fit_eval_attach_config(**kwargs_att)
    assert a1 == a2


def test_missing_gate_json_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="whitening-gate campaign"):
        load_gate_selection(missing)
