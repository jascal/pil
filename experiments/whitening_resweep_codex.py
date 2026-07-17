"""DEV-only whitening hyperparameter re-sweep for grounded dependency models.

Every transform and model is fit on TRAIN. Geometry selection and scoring use
DEV. The diagnostic compares identical grids for raw and whitened residuals.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.whitening_gate_codex import (  # noqa: E402
    DATA_ROOT,
    GROUND_PATHS,
    SEED,
    BilinearConfig,
    GermanR1Student,
    GermanR3DependencyStudent,
    GroundedAttachmentScorer,
    GroundedRelationLabeler,
    GroundedResidualProvider,
    RegisterLayer,
    RelationLabelerConfig,
    WhitenedResidualProvider,
    _attachment_sweeps,
    _fit_transforms_and_geometry,
    _labeler_sweeps,
    _predicted_pos_by_sentence,
    choose_window,
    evaluate_attach_dev_candidate,
    evaluate_dev_layer,
    head_deprel_by_sent,
    load_head_deprel_file,
    load_split,
)

LR_GRID = (0.01, 0.04, 0.1, 0.3)
EPOCHS_GRID = (5, 15, 40)
RANK_GRID = (16, 32)
FIXED_RECIPE = (0.04, 5, 16)

# One absolute accuracy/UAS point is large enough to distinguish a practical
# whitening gain from small DEV fluctuations while leaving the raw margin visible.
CLEAR_MARGIN_THRESHOLD = 0.01

GridPoint = tuple[float, int, int]


def hyperparameter_grid() -> list[GridPoint]:
    """Return the mandated 24 cells in deterministic order."""
    return [
        (lr, epochs, rank)
        for lr in LR_GRID
        for epochs in EPOCHS_GRID
        for rank in RANK_GRID
    ]


def setup_train_dev() -> dict[str, Any]:
    """Replicate the gate's TRAIN+DEV setup and stop at the fitted L0 model."""
    hasher = hashlib.sha256()
    train_provider = GroundedResidualProvider(GROUND_PATHS["train"])
    dev_provider = GroundedResidualProvider(GROUND_PATHS["dev"])
    transforms, geometry = _fit_transforms_and_geometry(train_provider)

    train = load_split(DATA_ROOT, "train", hasher)
    train_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "train.jsonl", hasher
    )
    train_heads = head_deprel_by_sent(train_head_records)
    grounded_train = [
        sentence for sentence in train if sentence.sent_id in train_provider.sentence_ids
    ]
    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    case_student = GermanR1Student(registers)
    case_student.fit(train)
    train_pos = _predicted_pos_by_sentence(case_student, train)

    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    dev_heads = head_deprel_by_sent(dev_head_records)
    dev_pos = _predicted_pos_by_sentence(case_student, dev)
    window, dev_coverage, coverage_curve = choose_window(dev_head_records)
    l0_student = GermanR3DependencyStudent(window)
    l0_student.fit(train, train_heads, train_pos)

    return {
        "train_provider": train_provider,
        "dev_provider": dev_provider,
        "transforms": transforms,
        "geometry": geometry,
        "train": train,
        "train_heads": train_heads,
        "grounded_train": grounded_train,
        "case_student": case_student,
        "train_pos": train_pos,
        "dev": dev,
        "dev_heads": dev_heads,
        "dev_pos": dev_pos,
        "window": window,
        "dev_coverage": dev_coverage,
        "coverage_curve": coverage_curve,
        "l0_student": l0_student,
    }


def _selected_curve_score(
    curve: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    layer: int,
    variant: str | None = None,
) -> float:
    for row in curve:
        if int(row["layer"]) != layer:
            continue
        if variant is not None and row.get("variant") != variant:
            continue
        return float(row[metric])
    raise KeyError(f"missing selected curve row: metric={metric}, variant={variant}, layer={layer}")


def compute_fixed_anchor(setup: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Run the fixed gate sweeps and return its DEV-selected anchor points."""
    label_raw_curve, label_raw_layer, label_white_curve, label_white_selection = (
        _labeler_sweeps(
            setup["grounded_train"],
            setup["train_heads"],
            setup["train_provider"],
            setup["transforms"],
            setup["dev"],
            setup["dev_heads"],
            setup["dev_pos"],
            setup["dev_provider"],
            setup["l0_student"],
            RelationLabelerConfig(seed=SEED),
        )
    )
    label_variant, label_white_layer = label_white_selection

    attach_raw_curve, attach_raw_layer, attach_white_curve, attach_white_selection = (
        _attachment_sweeps(
            setup["grounded_train"],
            setup["train_heads"],
            setup["train_provider"],
            setup["transforms"],
            setup["dev"],
            setup["dev_heads"],
            setup["dev_pos"],
            setup["dev_provider"],
            setup["l0_student"],
            BilinearConfig(seed=SEED),
        )
    )
    attach_variant, attach_white_layer = attach_white_selection

    return {
        "labeler": {
            "raw_layer": label_raw_layer,
            "whitened_variant": label_variant,
            "whitened_layer": label_white_layer,
            "fixed_raw": _selected_curve_score(
                label_raw_curve,
                "deprel_only_accuracy",
                layer=label_raw_layer,
            ),
            "fixed_whitened": _selected_curve_score(
                label_white_curve,
                "deprel_only_accuracy",
                variant=label_variant,
                layer=label_white_layer,
            ),
        },
        "attach": {
            "raw_layer": attach_raw_layer,
            "whitened_variant": attach_variant,
            "whitened_layer": attach_white_layer,
            "fixed_raw": _selected_curve_score(
                attach_raw_curve,
                "covered_dependent_uas",
                layer=attach_raw_layer,
            ),
            "fixed_whitened": _selected_curve_score(
                attach_white_curve,
                "covered_dependent_uas",
                variant=attach_variant,
                layer=attach_white_layer,
            ),
        },
    }


def _fit_eval_labeler(
    setup: Mapping[str, Any],
    provider: Any,
    dev_provider: Any,
    layer: int,
    point: GridPoint,
) -> float:
    lr, epochs, rank = point
    labeler = GroundedRelationLabeler(
        RelationLabelerConfig(
            rank=rank,
            epochs=epochs,
            learning_rate=lr,
            seed=SEED,
        )
    )
    labeler.fit(setup["grounded_train"], provider, setup["train_heads"], layer)
    accuracy, _fraction, _count = evaluate_dev_layer(
        setup["dev"],
        setup["dev_heads"],
        setup["dev_pos"],
        setup["l0_student"],
        labeler,
        dev_provider,
        layer,
    )
    return float(accuracy)


def _fit_eval_attach(
    setup: Mapping[str, Any],
    provider: Any,
    dev_provider: Any,
    layer: int,
    point: GridPoint,
) -> float:
    lr, epochs, rank = point
    scorer = GroundedAttachmentScorer(
        BilinearConfig(
            rank=rank,
            epochs=epochs,
            learning_rate=lr,
            seed=SEED,
        )
    )
    scorer.fit(setup["grounded_train"], provider, setup["train_heads"], layer)
    uas, _count = evaluate_attach_dev_candidate(
        setup["dev"],
        setup["dev_heads"],
        setup["dev_pos"],
        setup["l0_student"],
        scorer,
        dev_provider,
        layer,
    )
    return float(uas)


def sweep_labeler_grid(
    setup: Mapping[str, Any],
    anchor: Mapping[str, Any],
    grid: Sequence[GridPoint] | None = None,
) -> list[dict[str, Any]]:
    """Fit and score paired raw/whitened labelers over one identical grid."""
    points = list(grid) if grid is not None else hyperparameter_grid()
    raw_layer = int(anchor["raw_layer"])
    variant = str(anchor["whitened_variant"])
    white_layer = int(anchor["whitened_layer"])
    train_white = WhitenedResidualProvider(
        setup["train_provider"], setup["transforms"][(variant, white_layer)]
    )
    dev_white = WhitenedResidualProvider(
        setup["dev_provider"], setup["transforms"][(variant, white_layer)]
    )
    rows: list[dict[str, Any]] = []
    for point in points:
        lr, epochs, rank = point
        raw_dev = _fit_eval_labeler(
            setup, setup["train_provider"], setup["dev_provider"], raw_layer, point
        )
        whitened_dev = _fit_eval_labeler(setup, train_white, dev_white, white_layer, point)
        rows.append(
            {
                "lr": lr,
                "epochs": epochs,
                "rank": rank,
                "raw_dev": raw_dev,
                "whitened_dev": whitened_dev,
            }
        )
        print(
            f"GRID LABELER: lr={lr:g} epochs={epochs} rank={rank} "
            f"raw={raw_dev:.12f} whitened={whitened_dev:.12f}",
            flush=True,
        )
    return rows


def sweep_attach_grid(
    setup: Mapping[str, Any],
    anchor: Mapping[str, Any],
    grid: Sequence[GridPoint] | None = None,
) -> list[dict[str, Any]]:
    """Fit and score paired raw/whitened attachment models over one grid."""
    points = list(grid) if grid is not None else hyperparameter_grid()
    raw_layer = int(anchor["raw_layer"])
    variant = str(anchor["whitened_variant"])
    white_layer = int(anchor["whitened_layer"])
    train_white = WhitenedResidualProvider(
        setup["train_provider"], setup["transforms"][(variant, white_layer)]
    )
    dev_white = WhitenedResidualProvider(
        setup["dev_provider"], setup["transforms"][(variant, white_layer)]
    )
    rows: list[dict[str, Any]] = []
    for point in points:
        lr, epochs, rank = point
        raw_dev = _fit_eval_attach(
            setup, setup["train_provider"], setup["dev_provider"], raw_layer, point
        )
        whitened_dev = _fit_eval_attach(setup, train_white, dev_white, white_layer, point)
        rows.append(
            {
                "lr": lr,
                "epochs": epochs,
                "rank": rank,
                "raw_dev": raw_dev,
                "whitened_dev": whitened_dev,
            }
        )
        print(
            f"GRID ATTACH: lr={lr:g} epochs={epochs} rank={rank} "
            f"raw={raw_dev:.12f} whitened={whitened_dev:.12f}",
            flush=True,
        )
    return rows


def _best_grid_row(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("grid must contain at least one point")
    return min(
        rows,
        key=lambda row: (
            -float(row[metric]),
            int(row["epochs"]),
            int(row["rank"]),
            float(row["lr"]),
        ),
    )


def summarize_grid(
    rows: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any]
) -> dict[str, Any]:
    """Combine full rows, deterministic argmaxes, selections, and anchor values."""
    raw = _best_grid_row(rows, "raw_dev")
    white = _best_grid_row(rows, "whitened_dev")
    raw_best = float(raw["raw_dev"])
    white_best = float(white["whitened_dev"])

    def argmax(row: Mapping[str, Any]) -> dict[str, float | int]:
        return {
            "lr": float(row["lr"]),
            "epochs": int(row["epochs"]),
            "rank": int(row["rank"]),
        }

    return {
        "raw_layer": int(anchor["raw_layer"]),
        "raw_best": raw_best,
        "raw_argmax": argmax(raw),
        "whitened_variant": str(anchor["whitened_variant"]),
        "whitened_layer": int(anchor["whitened_layer"]),
        "whitened_best": white_best,
        "whitened_argmax": argmax(white),
        "fixed_raw": float(anchor["fixed_raw"]),
        "fixed_whitened": float(anchor["fixed_whitened"]),
        "margin": white_best - raw_best,
        "full_grid": [dict(row) for row in rows],
    }


def _format_argmax(argmax: Mapping[str, Any]) -> str:
    return (
        f"lr={float(argmax['lr']):g},epochs={int(argmax['epochs'])},"
        f"rank={int(argmax['rank'])}"
    )


def _summary_text(labeler: Mapping[str, Any], attach: Mapping[str, Any], signal: str) -> str:
    return (
        f"On DEV with seed {SEED}, labeler best raw={labeler['raw_best']:.12f} and "
        f"whitened={labeler['whitened_best']:.12f} (margin={labeler['margin']:+.12f}; "
        f"fixed anchor raw={labeler['fixed_raw']:.12f}, "
        f"whitened={labeler['fixed_whitened']:.12f}); attachment best "
        f"raw={attach['raw_best']:.12f} and whitened={attach['whitened_best']:.12f} "
        f"(margin={attach['margin']:+.12f}; fixed anchor raw={attach['fixed_raw']:.12f}, "
        f"whitened={attach['fixed_whitened']:.12f}); signal={signal}."
    )


def build_resweep() -> dict[str, Any]:
    """Run the fixed selection plus the complete paired diagnostic grid."""
    print(f"SEED: {SEED} (single deterministic seed).", flush=True)
    setup = setup_train_dev()
    anchor = compute_fixed_anchor(setup)
    label_rows = sweep_labeler_grid(setup, anchor["labeler"])
    attach_rows = sweep_attach_grid(setup, anchor["attach"])
    labeler = summarize_grid(label_rows, anchor["labeler"])
    attach = summarize_grid(attach_rows, anchor["attach"])
    signal = (
        "REOPEN"
        if max(labeler["margin"], attach["margin"]) > CLEAR_MARGIN_THRESHOLD
        else "R4_CLEAN"
    )
    return {
        "seed": SEED,
        "grid": {
            "lr": list(LR_GRID),
            "epochs": list(EPOCHS_GRID),
            "rank": list(RANK_GRID),
        },
        "labeler": labeler,
        "attach": attach,
        "signal": signal,
        "clear_margin_threshold": CLEAR_MARGIN_THRESHOLD,
        "text": _summary_text(labeler, attach, signal),
    }


def print_report(resweep: Mapping[str, Any]) -> None:
    """Print the compact report consumed by manual verification."""
    print("\nDEV-ONLY HYPERPARAMETER RE-SWEEP", flush=True)
    print(f"seed={resweep['seed']}", flush=True)
    print(
        "primitive | raw best DEV (argmax) | whitened best DEV (argmax) | margin",
        flush=True,
    )
    for primitive in ("labeler", "attach"):
        row = resweep[primitive]
        print(
            f"{primitive} | {row['raw_best']:.12f} ({_format_argmax(row['raw_argmax'])}) "
            f"| {row['whitened_best']:.12f} ({_format_argmax(row['whitened_argmax'])}) "
            f"| {row['margin']:+.12f}",
            flush=True,
        )
    print("FIXED-RECIPE ANCHOR: lr=0.04,epochs=5,rank=16", flush=True)
    for primitive in ("labeler", "attach"):
        row = resweep[primitive]
        print(
            f"{primitive} | raw={row['fixed_raw']:.12f} "
            f"| whitened={row['fixed_whitened']:.12f}",
            flush=True,
        )
    print(
        f"SIGNAL: {resweep['signal']} | threshold={resweep['clear_margin_threshold']:.6f} "
        f"| labeler_margin={resweep['labeler']['margin']:+.12f} "
        f"| attach_margin={resweep['attach']['margin']:+.12f}",
        flush=True,
    )


def main() -> None:
    resweep = build_resweep()
    print_report(resweep)


if __name__ == "__main__":
    main()
