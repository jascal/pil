"""DEV-only hyperparameter re-sweep: raw φ vs whitened φ (mechanism diagnostic).

Fits on TRAIN, evaluates on DEV only. Never reads the TEST split. Emits
REOPEN / R4_CLEAN based on whether best-tuned whitened DEV beats best-tuned
raw DEV for either primitive (labeler, attach). Single seed=0 throughout.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.attach_levers_codex import BilinearConfig  # noqa: E402
from experiments.biaffine_labeler_codex import RelationLabelerConfig  # noqa: E402
from experiments.campaign_grounded_attach import GroundedAttachmentScorer  # noqa: E402
from experiments.campaign_grounded_labeler import GroundedRelationLabeler  # noqa: E402
from experiments.campaign_whitening_gate import (  # noqa: E402
    CHECKPOINT_LAYERS,
    DEV_NPZ,
    TRAIN_NPZ,
    GroundedFeatureProvider,
    WhitenedFeatureProvider,
    WhiteningTransform,
    evaluate_dev_attach_layer_pin_b,
    evaluate_dev_layer,
    load_layer_matrix,
)
from experiments.german_r1_codex import (  # noqa: E402
    DATA_ROOT,
    GermanR1Student,
    RegisterLayer,
    Sentence,
    load_split,
)
from experiments.german_r3_codex import (  # noqa: E402
    GermanR3DependencyStudent,
    _predicted_pos_by_sentence,
    choose_window,
    load_head_deprel_file,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    head_deprel_by_sent,
)

# Hyperparameter grid (identical for raw and whitened, both primitives).
LR_VALUES: tuple[float, ...] = (0.01, 0.04, 0.1, 0.3)
EPOCHS_VALUES: tuple[int, ...] = (5, 15, 40)
RANK_VALUES: tuple[int, ...] = (16, 32)
SEED = 0

# Fixed-recipe anchor cell (RelationLabelerConfig / BilinearConfig defaults).
FIXED_LR = 0.04
FIXED_EPOCHS = 5
FIXED_RANK = 16

GATE_JSON = Path(__file__).resolve().parents[1] / "data" / "campaign_whitening_gate.json"

FeatureSet = Literal["raw", "whitened"]
Primitive = Literal["labeler", "attach"]


def iter_grid() -> list[tuple[float, int, int]]:
    """Return the 24 (lr, epochs, rank) cells in deterministic order."""
    return [
        (lr, epochs, rank)
        for lr in LR_VALUES
        for epochs in EPOCHS_VALUES
        for rank in RANK_VALUES
    ]


def load_gate_selection(gate_path: Path = GATE_JSON) -> dict[str, dict[str, Any]]:
    """Load DEV-selected whitening (variant, layer_index) per primitive from gate JSON.

    Raises FileNotFoundError with a clear message if the gate artifact is missing.
    """
    if not gate_path.is_file():
        raise FileNotFoundError(
            f"whitening-gate campaign artifact missing at {gate_path}; "
            "run experiments/campaign_whitening_gate.py first"
        )
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    selected_labeler = payload["selected_labeler"]
    selected_attach = payload["selected_attach"]
    return {
        "labeler": {
            "variant": str(selected_labeler["variant"]),
            "layer_index": int(selected_labeler["layer_index"]),
            "layer": int(selected_labeler["layer"]),
        },
        "attach": {
            "variant": str(selected_attach["variant"]),
            "layer_index": int(selected_attach["layer_index"]),
            "layer": int(selected_attach["layer"]),
        },
    }


def load_train_dev_bundle() -> dict[str, Any]:
    """Load TRAIN+DEV splits, head files, POS, and fit the L0 student (no TEST)."""
    hasher = hashlib.sha256()
    print("LOAD: GSD train shared tasks and head_deprel.", flush=True)
    train = load_split(DATA_ROOT, "train", hasher)
    train_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "train.jsonl", hasher
    )
    train_heads = head_deprel_by_sent(train_head_records)

    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    r1_student = GermanR1Student(registers)
    r1_student.fit(train)
    train_pos = _predicted_pos_by_sentence(r1_student, train)

    print("LOAD: GSD dev once for L0 window + DEV eval.", flush=True)
    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_heads_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    window, _dev_coverage, _coverage_curve = choose_window(dev_heads_records)
    dev_heads = head_deprel_by_sent(dev_heads_records)
    dev_pos = _predicted_pos_by_sentence(r1_student, dev)

    l0_student = GermanR3DependencyStudent(window)
    l0_student.fit(train, train_heads, train_pos)
    print(f"LOAD: L0 window={window}; train={len(train)} dev={len(dev)}.", flush=True)

    return {
        "train": train,
        "train_heads": train_heads,
        "train_pos": train_pos,
        "dev": dev,
        "dev_heads": dev_heads,
        "dev_pos": dev_pos,
        "l0_student": l0_student,
        "window": window,
    }


def build_providers(
    *,
    train_npz: Path,
    dev_npz: Path,
    layer_index: int,
    feature_set: FeatureSet,
    variant: str | None = None,
) -> tuple[
    GroundedFeatureProvider | WhitenedFeatureProvider,
    GroundedFeatureProvider | WhitenedFeatureProvider,
]:
    """Build TRAIN and DEV providers for raw or whitened φ.

    Whitening is fit on TRAIN only (load_layer_matrix + WhiteningTransform.fit).
    """
    train_raw = GroundedFeatureProvider(train_npz, layer_index)
    dev_raw = GroundedFeatureProvider(dev_npz, layer_index)
    if feature_set == "raw":
        return train_raw, dev_raw
    if variant is None:
        raise ValueError("variant is required for whitened feature_set")
    transform = WhiteningTransform.fit(
        load_layer_matrix(train_npz, layer_index),
        variant=variant,
        layer_index=layer_index,
    )
    return (
        WhitenedFeatureProvider(train_raw, transform),
        WhitenedFeatureProvider(dev_raw, transform),
    )


def fit_eval_labeler_config(
    *,
    lr: float,
    epochs: int,
    rank: int,
    train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
    train_pos: Mapping[str, Sequence[str]],
    train_provider: GroundedFeatureProvider | WhitenedFeatureProvider,
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    dev_provider: GroundedFeatureProvider | WhitenedFeatureProvider,
) -> float:
    """Fit one RelationLabelerConfig on TRAIN; return DEV deprel-only accuracy."""
    config = RelationLabelerConfig(
        rank=rank, epochs=epochs, learning_rate=lr, seed=SEED
    )
    labeler = GroundedRelationLabeler(config)
    labeler.fit(train, train_heads, train_pos, train_provider)  # type: ignore[arg-type]
    metrics = evaluate_dev_layer(
        dev=dev,
        dev_heads=dev_heads,
        dev_pos=dev_pos,
        l0_student=l0_student,
        grounded_labeler=labeler,
        provider=dev_provider,  # type: ignore[arg-type]
    )
    return float(metrics["dev_deprel_only_accuracy"])


def fit_eval_attach_config(
    *,
    lr: float,
    epochs: int,
    rank: int,
    train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
    train_provider: GroundedFeatureProvider | WhitenedFeatureProvider,
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    dev_provider: GroundedFeatureProvider | WhitenedFeatureProvider,
) -> float:
    """Fit one BilinearConfig on TRAIN; return DEV UAS under PIN-B decode."""
    config = BilinearConfig(rank=rank, epochs=epochs, learning_rate=lr, seed=SEED)
    scorer = GroundedAttachmentScorer(config)
    scorer.fit(train, train_heads, train_provider)  # type: ignore[arg-type]
    metrics = evaluate_dev_attach_layer_pin_b(
        dev=dev,
        dev_heads=dev_heads,
        dev_pos=dev_pos,
        l0_student=l0_student,
        scorer=scorer,
        provider=dev_provider,
    )
    return float(metrics["dev_uas"])


def pick_best_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select max-score row.

    Tie-break on exact score ties: prefer smaller epochs, then smaller rank,
    then smaller lr.
    """
    if not rows:
        raise ValueError("cannot pick best from empty grid")
    best = min(
        rows,
        key=lambda row: (
            -float(row["score"]),
            int(row["epochs"]),
            int(row["rank"]),
            float(row["lr"]),
        ),
    )
    return dict(best)


def find_fixed_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Look up the fixed-recipe grid cell (lr=0.04, epochs=5, rank=16)."""
    for row in rows:
        if (
            float(row["lr"]) == FIXED_LR
            and int(row["epochs"]) == FIXED_EPOCHS
            and int(row["rank"]) == FIXED_RANK
        ):
            return dict(row)
    raise KeyError(
        f"fixed recipe (lr={FIXED_LR}, epochs={FIXED_EPOCHS}, rank={FIXED_RANK}) "
        "not found in grid rows"
    )


def sweep_labeler_grid(
    *,
    train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
    train_pos: Mapping[str, Sequence[str]],
    train_provider: GroundedFeatureProvider | WhitenedFeatureProvider,
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    dev_provider: GroundedFeatureProvider | WhitenedFeatureProvider,
    feature_set: FeatureSet,
    grid: Sequence[tuple[float, int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Sweep labeler configs; print each row as it completes."""
    cells = list(grid) if grid is not None else iter_grid()
    rows: list[dict[str, Any]] = []
    for lr, epochs, rank in cells:
        score = fit_eval_labeler_config(
            lr=lr,
            epochs=epochs,
            rank=rank,
            train=train,
            train_heads=train_heads,
            train_pos=train_pos,
            train_provider=train_provider,
            dev=dev,
            dev_heads=dev_heads,
            dev_pos=dev_pos,
            l0_student=l0_student,
            dev_provider=dev_provider,
        )
        row = {"lr": lr, "epochs": epochs, "rank": rank, "score": score}
        print(
            f"labeler/{feature_set}: lr={lr} epochs={epochs} rank={rank} "
            f"dev_deprel_only_accuracy={score:.6f}",
            flush=True,
        )
        rows.append(row)
    return rows


def sweep_attach_grid(
    *,
    train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
    train_provider: GroundedFeatureProvider | WhitenedFeatureProvider,
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    dev_provider: GroundedFeatureProvider | WhitenedFeatureProvider,
    feature_set: FeatureSet,
    grid: Sequence[tuple[float, int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Sweep attach configs; print each row as it completes."""
    cells = list(grid) if grid is not None else iter_grid()
    rows: list[dict[str, Any]] = []
    for lr, epochs, rank in cells:
        score = fit_eval_attach_config(
            lr=lr,
            epochs=epochs,
            rank=rank,
            train=train,
            train_heads=train_heads,
            train_provider=train_provider,
            dev=dev,
            dev_heads=dev_heads,
            dev_pos=dev_pos,
            l0_student=l0_student,
            dev_provider=dev_provider,
        )
        row = {"lr": lr, "epochs": epochs, "rank": rank, "score": score}
        print(
            f"attach/{feature_set}: lr={lr} epochs={epochs} rank={rank} "
            f"dev_uas={score:.6f}",
            flush=True,
        )
        rows.append(row)
    return rows


def _primitive_block(
    *,
    primitive: Primitive,
    variant: str,
    layer_index: int,
    layer: int,
    grid_raw: list[dict[str, Any]],
    grid_whitened: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_best_row = pick_best_row(grid_raw)
    white_best_row = pick_best_row(grid_whitened)
    fixed_raw = find_fixed_row(grid_raw)
    fixed_white = find_fixed_row(grid_whitened)
    return {
        "raw_best": float(raw_best_row["score"]),
        "raw_argmax": {
            "lr": float(raw_best_row["lr"]),
            "epochs": int(raw_best_row["epochs"]),
            "rank": int(raw_best_row["rank"]),
        },
        "whitened_best": float(white_best_row["score"]),
        "whitened_argmax": {
            "lr": float(white_best_row["lr"]),
            "epochs": int(white_best_row["epochs"]),
            "rank": int(white_best_row["rank"]),
        },
        "fixed_raw": float(fixed_raw["score"]),
        "fixed_whitened": float(fixed_white["score"]),
        "grid_raw": grid_raw,
        "grid_whitened": grid_whitened,
        "variant": variant,
        "layer_index": layer_index,
        "layer": layer,
    }


def run_resweep(
    *,
    bundle: dict[str, Any] | None = None,
    selection: dict[str, dict[str, Any]] | None = None,
    train_npz: Path = TRAIN_NPZ,
    dev_npz: Path = DEV_NPZ,
    grid: Sequence[tuple[float, int, int]] | None = None,
) -> dict[str, Any]:
    """Run the full DEV-only re-sweep for both primitives and both feature sets."""
    if bundle is None:
        bundle = load_train_dev_bundle()
    if selection is None:
        selection = load_gate_selection()

    train = bundle["train"]
    train_heads = bundle["train_heads"]
    train_pos = bundle["train_pos"]
    dev = bundle["dev"]
    dev_heads = bundle["dev_heads"]
    dev_pos = bundle["dev_pos"]
    l0_student = bundle["l0_student"]

    print(
        f"RE-SWEEP: seed={SEED}; grid lr={LR_VALUES} epochs={EPOCHS_VALUES} "
        f"rank={RANK_VALUES} ({len(iter_grid())} cells × 2 feature sets × 2 primitives).",
        flush=True,
    )

    # --- LABELER ---
    lab = selection["labeler"]
    lab_variant = str(lab["variant"])
    lab_layer_index = int(lab["layer_index"])
    lab_layer = int(lab.get("layer", CHECKPOINT_LAYERS[lab_layer_index]))
    print(
        f"LABELER selected whitening: variant={lab_variant} "
        f"layer_index={lab_layer_index} layer={lab_layer}",
        flush=True,
    )

    train_lab_raw, dev_lab_raw = build_providers(
        train_npz=train_npz,
        dev_npz=dev_npz,
        layer_index=lab_layer_index,
        feature_set="raw",
    )
    train_lab_white, dev_lab_white = build_providers(
        train_npz=train_npz,
        dev_npz=dev_npz,
        layer_index=lab_layer_index,
        feature_set="whitened",
        variant=lab_variant,
    )

    print("SWEEP: labeler raw φ", flush=True)
    grid_lab_raw = sweep_labeler_grid(
        train=train,
        train_heads=train_heads,
        train_pos=train_pos,
        train_provider=train_lab_raw,
        dev=dev,
        dev_heads=dev_heads,
        dev_pos=dev_pos,
        l0_student=l0_student,
        dev_provider=dev_lab_raw,
        feature_set="raw",
        grid=grid,
    )
    print("SWEEP: labeler whitened φ", flush=True)
    grid_lab_white = sweep_labeler_grid(
        train=train,
        train_heads=train_heads,
        train_pos=train_pos,
        train_provider=train_lab_white,
        dev=dev,
        dev_heads=dev_heads,
        dev_pos=dev_pos,
        l0_student=l0_student,
        dev_provider=dev_lab_white,
        feature_set="whitened",
        grid=grid,
    )
    labeler_block = _primitive_block(
        primitive="labeler",
        variant=lab_variant,
        layer_index=lab_layer_index,
        layer=lab_layer,
        grid_raw=grid_lab_raw,
        grid_whitened=grid_lab_white,
    )

    # --- ATTACH ---
    att = selection["attach"]
    att_variant = str(att["variant"])
    att_layer_index = int(att["layer_index"])
    att_layer = int(att.get("layer", CHECKPOINT_LAYERS[att_layer_index]))
    print(
        f"ATTACH selected whitening: variant={att_variant} "
        f"layer_index={att_layer_index} layer={att_layer}",
        flush=True,
    )

    train_att_raw, dev_att_raw = build_providers(
        train_npz=train_npz,
        dev_npz=dev_npz,
        layer_index=att_layer_index,
        feature_set="raw",
    )
    train_att_white, dev_att_white = build_providers(
        train_npz=train_npz,
        dev_npz=dev_npz,
        layer_index=att_layer_index,
        feature_set="whitened",
        variant=att_variant,
    )

    print("SWEEP: attach raw φ", flush=True)
    grid_att_raw = sweep_attach_grid(
        train=train,
        train_heads=train_heads,
        train_provider=train_att_raw,
        dev=dev,
        dev_heads=dev_heads,
        dev_pos=dev_pos,
        l0_student=l0_student,
        dev_provider=dev_att_raw,
        feature_set="raw",
        grid=grid,
    )
    print("SWEEP: attach whitened φ", flush=True)
    grid_att_white = sweep_attach_grid(
        train=train,
        train_heads=train_heads,
        train_provider=train_att_white,
        dev=dev,
        dev_heads=dev_heads,
        dev_pos=dev_pos,
        l0_student=l0_student,
        dev_provider=dev_att_white,
        feature_set="whitened",
        grid=grid,
    )
    attach_block = _primitive_block(
        primitive="attach",
        variant=att_variant,
        layer_index=att_layer_index,
        layer=att_layer,
        grid_raw=grid_att_raw,
        grid_whitened=grid_att_white,
    )

    margin_labeler = float(labeler_block["whitened_best"]) - float(labeler_block["raw_best"])
    margin_attach = float(attach_block["whitened_best"]) - float(attach_block["raw_best"])
    # Strict: REOPEN if either whitened_best > raw_best.
    signal = "REOPEN" if (margin_labeler > 0.0 or margin_attach > 0.0) else "R4_CLEAN"

    text = (
        f"DEV-only hyperparameter re-sweep (seed={SEED}). "
        f"Labeler (variant={lab_variant}, layer={lab_layer}): "
        f"raw_best={labeler_block['raw_best']:.6f} at {labeler_block['raw_argmax']}, "
        f"whitened_best={labeler_block['whitened_best']:.6f} at "
        f"{labeler_block['whitened_argmax']}; "
        f"fixed-recipe (lr={FIXED_LR}/epochs={FIXED_EPOCHS}/rank={FIXED_RANK}) "
        f"raw={labeler_block['fixed_raw']:.6f} whitened={labeler_block['fixed_whitened']:.6f}; "
        f"margin_labeler={margin_labeler:+.6f}. "
        f"Attach (variant={att_variant}, layer={att_layer}): "
        f"raw_best={attach_block['raw_best']:.6f} at {attach_block['raw_argmax']}, "
        f"whitened_best={attach_block['whitened_best']:.6f} at "
        f"{attach_block['whitened_argmax']}; "
        f"fixed-recipe raw={attach_block['fixed_raw']:.6f} "
        f"whitened={attach_block['fixed_whitened']:.6f}; "
        f"margin_attach={margin_attach:+.6f}. "
        f"Signal={signal} (REOPEN if either margin > 0, else R4_CLEAN). "
        f"Architect should weigh a clear margin vs a wash when interpreting close calls."
    )

    return {
        "labeler": labeler_block,
        "attach": attach_block,
        "signal": signal,
        "margin_labeler": margin_labeler,
        "margin_attach": margin_attach,
        "text": text,
        "seed": SEED,
    }


def _fmt_argmax(argmax: Mapping[str, Any]) -> str:
    return (
        f"lr={argmax['lr']} epochs={argmax['epochs']} rank={argmax['rank']}"
    )


def print_summary(resweep: Mapping[str, Any]) -> None:
    """Print the per-primitive table, fixed-recipe anchors, signal, and margins."""
    print("", flush=True)
    print("=" * 72, flush=True)
    print("WHITENING HYPERPARAMETER RE-SWEEP (DEV only, seed=0)", flush=True)
    print("=" * 72, flush=True)
    print(
        f"{'primitive':<10} {'feature':<10} {'best_DEV':>10}  winning hyperparameters",
        flush=True,
    )
    print("-" * 72, flush=True)
    for primitive in ("labeler", "attach"):
        block = resweep[primitive]
        print(
            f"{primitive:<10} {'raw':<10} {block['raw_best']:>10.6f}  "
            f"{_fmt_argmax(block['raw_argmax'])}",
            flush=True,
        )
        print(
            f"{primitive:<10} {'whitened':<10} {block['whitened_best']:>10.6f}  "
            f"{_fmt_argmax(block['whitened_argmax'])}",
            flush=True,
        )
    print("-" * 72, flush=True)
    print(
        f"Fixed-recipe anchor (lr={FIXED_LR}, epochs={FIXED_EPOCHS}, rank={FIXED_RANK}):",
        flush=True,
    )
    for primitive in ("labeler", "attach"):
        block = resweep[primitive]
        print(
            f"  {primitive}: fixed_raw={block['fixed_raw']:.6f}  "
            f"fixed_whitened={block['fixed_whitened']:.6f}",
            flush=True,
        )
    print("-" * 72, flush=True)
    print(f"signal:          {resweep['signal']}", flush=True)
    print(f"margin_labeler:  {resweep['margin_labeler']:+.6f}", flush=True)
    print(f"margin_attach:   {resweep['margin_attach']:+.6f}", flush=True)
    print("-" * 72, flush=True)
    print(resweep["text"], flush=True)
    print("=" * 72, flush=True)


def main() -> None:
    if not math.isfinite(SEED):  # pragma: no cover — seed is constant 0
        raise RuntimeError("SEED must be finite")
    resweep = run_resweep()
    print_summary(resweep)
    # Machine-readable dump to stdout only (no file write).
    print(json.dumps({k: v for k, v in resweep.items() if k != "text"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
