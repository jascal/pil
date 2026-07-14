"""Oracle-head and noise-matched diagnostics for the biaffine relation labeler."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.attach_levers_codex import (  # noqa: E402
    predict_deprels,
    score_prediction_sequences,
)
from experiments.biaffine_labeler_codex import (  # noqa: E402
    LOCAL_DEPRELS,
    PAIRWISE_SENSITIVE_DEPRELS,
    BilinearRelationLabeler,
    RelationLabelerConfig,
    score_relation_partition,
)
from experiments.german_r1_codex import (  # noqa: E402
    DATA_ROOT,
    GermanR1Student,
    RegisterLayer,
    Sentence,
    accuracy,
    load_split,
)
from experiments.german_r3_codex import (  # noqa: E402
    CampaignTestReadGuard,
    GermanR3DependencyStudent,
    _predicted_pos_by_sentence,
    choose_window,
    load_head_deprel_file,
    load_head_test_once,
    load_shared_test_once,
    run_predicted_case_sentence,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    aligned_head_record,
    head_deprel_by_sent,
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "biaffine_labeler_diagnostic_codex.json"
)
PRIMARY_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "biaffine_labeler_codex.json"
)
ROW_KEYS = (
    ("unary", "gold_heads"),
    ("unary", "predicted_heads"),
    ("biaffine_gold_trained", "gold_heads"),
    ("biaffine_gold_trained", "predicted_heads"),
    ("biaffine_noise_matched", "predicted_heads"),
)
GAIN_KEYS = (
    "deprel_only",
    "las_strict",
    "serve_honest_case",
    "pairwise",
    "local",
)


def _gold_train_targets(
    train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
) -> dict[str, tuple[str, ...]]:
    targets: dict[str, tuple[str, ...]] = {}
    for sentence in train:
        head = aligned_head_record(sentence, train_heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        targets[sentence.sent_id] = head.deprel
    return targets


def _train_predicted_governors(
    train: Sequence[Sentence],
    train_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
) -> dict[str, tuple[int, ...]]:
    return {
        sentence.sent_id: l0_student.predict_heads(
            sentence.tokens, train_pos[sentence.sent_id]
        )
        for sentence in train
    }


def _empty_accumulator() -> dict[str, list[Any]]:
    return {"heads": [], "deprels": [], "case": []}


def _evaluate_cells(
    test: Sequence[Sentence],
    test_heads: Mapping[str, HeadDeprelRecord],
    test_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    gold_trained: BilinearRelationLabeler,
    noise_matched: BilinearRelationLabeler,
    case_student: GermanR1Student,
) -> list[dict[str, Any]]:
    accumulated = {key: _empty_accumulator() for key in ROW_KEYS}
    gold_heads_all: list[int] = []
    gold_deprels_all: list[str] = []
    gold_case_all: list[str] = []

    for sentence in test:
        head = aligned_head_record(sentence, test_heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = test_pos[sentence.sent_id]
        gold_heads = head.head_offset
        predicted_heads = l0_student.predict_sentence(
            sentence.tokens, pos
        ).head_offset
        predictions = {
            ("unary", "gold_heads"): predict_deprels(
                l0_student, sentence.tokens, pos, gold_heads
            ),
            ("unary", "predicted_heads"): predict_deprels(
                l0_student, sentence.tokens, pos, predicted_heads
            ),
            ("biaffine_gold_trained", "gold_heads"): gold_trained.predict(
                sentence.tokens, pos, gold_heads
            ),
            (
                "biaffine_gold_trained",
                "predicted_heads",
            ): gold_trained.predict(sentence.tokens, pos, predicted_heads),
            (
                "biaffine_noise_matched",
                "predicted_heads",
            ): noise_matched.predict(sentence.tokens, pos, predicted_heads),
        }
        head_sources = {
            "gold_heads": gold_heads,
            "predicted_heads": predicted_heads,
        }
        baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        eligible = [
            index
            for index, label in enumerate(sentence.targets["morph_case"])
            if label != "-"
        ]
        for key, predicted_deprels in predictions.items():
            arm, head_source = key
            supplied_heads = head_sources[head_source]
            case_result = run_predicted_case_sentence(
                case_student,
                sentence.tokens,
                baseline,
                pos,
                supplied_heads,
                predicted_deprels,
                eligible,
            )
            accumulated[(arm, head_source)]["heads"].extend(supplied_heads)
            accumulated[(arm, head_source)]["deprels"].extend(predicted_deprels)
            accumulated[(arm, head_source)]["case"].extend(
                case_result.predictions[index] for index in eligible
            )
        gold_heads_all.extend(gold_heads)
        gold_deprels_all.extend(head.deprel)
        gold_case_all.extend(
            sentence.targets["morph_case"][index] for index in eligible
        )

    table: list[dict[str, Any]] = []
    for arm, head_source in ROW_KEYS:
        cell = accumulated[(arm, head_source)]
        strict = score_prediction_sequences(
            cell["heads"],
            cell["deprels"],
            gold_heads_all,
            gold_deprels_all,
        )
        partition = score_relation_partition(cell["deprels"], gold_deprels_all)
        table.append(
            {
                "arm": arm,
                "head_source": head_source,
                "deprel_only": strict["deprel_only_accuracy"],
                "las_strict": strict["las"],
                "serve_honest_case": accuracy(cell["case"], gold_case_all),
                "pairwise": partition["pairwise_sensitive"]["accuracy"],
                "local": partition["local"]["accuracy"],
            }
        )
    return table


def _memorization_scores(
    train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
    train_pos: Mapping[str, Sequence[str]],
    labeler: BilinearRelationLabeler,
    test_gold_accuracy: float,
) -> dict[str, float]:
    predicted: list[str] = []
    gold: list[str] = []
    for sentence in train:
        head = aligned_head_record(sentence, train_heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        predicted.extend(
            labeler.predict(
                sentence.tokens,
                train_pos[sentence.sent_id],
                head.head_offset,
            )
        )
        gold.extend(head.deprel)
    train_accuracy = accuracy(predicted, gold)
    return {
        "train_deprel_only": train_accuracy,
        "test_gold_heads_deprel_only": test_gold_accuracy,
        "train_minus_test": train_accuracy - test_gold_accuracy,
    }


def _indexed_table(
    table: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(row["arm"], row["head_source"]): row for row in table}


def _gains(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    return {key: float(candidate[key]) - float(baseline[key]) for key in GAIN_KEYS}


def _format_gains(gains: Mapping[str, float]) -> str:
    return ", ".join(f"{key}={gains[key]:+.6f}" for key in GAIN_KEYS)


def _primary_crosscheck(table: Sequence[Mapping[str, Any]]) -> str:
    if not PRIMARY_OUTPUT_PATH.is_file():
        raise FileNotFoundError(
            f"primary campaign artifact required for cross-check: {PRIMARY_OUTPUT_PATH}"
        )
    primary = json.loads(PRIMARY_OUTPUT_PATH.read_text(encoding="utf-8"))
    indexed = _indexed_table(table)
    comparisons = {
        ("unary", "predicted_heads"): primary["arms"]["unary"],
        ("biaffine_gold_trained", "predicted_heads"): primary["arms"][
            "biaffine_by_seed"
        ]["0"]["metrics"],
    }
    paths = {
        "deprel_only": ("deprel_only_accuracy",),
        "las_strict": ("las_strict",),
        "serve_honest_case": ("serve_honest_morph_case",),
        "pairwise": ("partition", "pairwise_sensitive", "accuracy"),
        "local": ("partition", "local", "accuracy"),
    }
    for key, expected in comparisons.items():
        for metric, path in paths.items():
            value: Any = expected
            for part in path:
                value = value[part]
            actual = indexed[key][metric]
            if not math.isclose(float(actual), float(value), abs_tol=1e-12):
                raise AssertionError(
                    f"primary cross-check mismatch for {key} {metric}: "
                    f"diagnostic={actual}, primary={value}"
                )
    return "predicted-head unary and biaffine s0 cells match the primary artifact"


def _make_reads(
    table: Sequence[Mapping[str, Any]],
    memorization: Mapping[str, float],
    crosscheck: str,
) -> dict[str, str]:
    indexed = _indexed_table(table)
    unary_gold = indexed[("unary", "gold_heads")]
    biaffine_gold = indexed[("biaffine_gold_trained", "gold_heads")]
    gold_gains = _gains(biaffine_gold, unary_gold)
    deprel_wins = gold_gains["deprel_only"] > 0.0
    case_wins = gold_gains["serve_honest_case"] > 0.0
    if deprel_wins and case_wins:
        q1_conclusion = "weak heads are implicated"
    elif not deprel_wins and not case_wins:
        q1_conclusion = "labeler form remains limiting even with correct heads"
    else:
        q1_conclusion = "the deprel/case evidence is mixed"
    q1 = (
        "Q1 FORM vs HEADS: on gold heads biaffine_gold_trained "
        f"{'beats' if deprel_wins else 'does not beat'} unary on deprel-only "
        f"({gold_gains['deprel_only']:+.6f}) and "
        f"{'beats' if case_wins else 'does not beat'} it on case "
        f"({gold_gains['serve_honest_case']:+.6f}); {q1_conclusion}; {crosscheck}."
    )

    unary_predicted = indexed[("unary", "predicted_heads")]
    gold_predicted = indexed[("biaffine_gold_trained", "predicted_heads")]
    noise_predicted = indexed[("biaffine_noise_matched", "predicted_heads")]
    gold_predicted_gains = _gains(gold_predicted, unary_predicted)
    noise_gains = _gains(noise_predicted, unary_predicted)
    narrowed = sum(
        noise_gains[key] > gold_predicted_gains[key] for key in GAIN_KEYS
    )
    reached = sum(noise_gains[key] >= 0.0 for key in GAIN_KEYS)
    if narrowed == len(GAIN_KEYS):
        closure = "all"
    elif narrowed:
        closure = "some"
    else:
        closure = "none"
    q2 = (
        f"Q2 NOISE MATCH: closes {closure} of the gold-trained gaps by narrowing "
        f"{narrowed}/{len(GAIN_KEYS)} metrics and reaches/exceeds unary on "
        f"{reached}/{len(GAIN_KEYS)}; noise-matched gains [{_format_gains(noise_gains)}] "
        f"versus gold-trained gains [{_format_gains(gold_predicted_gains)}]."
    )

    gap = float(memorization["train_minus_test"])
    if gap >= 0.10:
        gap_read = "large"
        implication = "consistent with memorization"
    elif gap <= 0.05:
        gap_read = "small"
        implication = "generalization is roughly intact"
    else:
        gap_read = "moderate"
        implication = "suggestive but not decisive for memorization"
    q3 = (
        f"Q3 MEMORIZATION: the train-test gap is {gap_read} ({gap:+.6f}), "
        f"{implication}; caveat: this conflates memorization with ordinary "
        "train/test distribution shift, so it does not establish causation."
    )
    return {"q1": q1, "q2": q2, "q3": q3}


def build_diagnostic() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit both diagnostic labelers and score five cells from one test read."""
    hasher = hashlib.sha256()
    print("FIT: loading GSD train shared tasks and head_deprel once.", flush=True)
    train = load_split(DATA_ROOT, "train", hasher)
    train_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "train.jsonl", hasher
    )
    train_heads_raw = head_deprel_by_sent(train_head_records)
    train_heads = {
        sent_id: record
        for sent_id, record in train_heads_raw.items()
        if isinstance(record, HeadDeprelRecord)
    }
    if len(train_heads) != len(train_heads_raw):
        raise TypeError("labeled dependency records required")

    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    r1_student = GermanR1Student(registers)
    r1_student.fit(train)
    train_pos = _predicted_pos_by_sentence(r1_student, train)

    print("DEV: loading GSD dev once for the shipped L0 window choice only.", flush=True)
    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_heads = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    window, dev_coverage, coverage_curve = choose_window(dev_heads)
    del dev

    l0_student = GermanR3DependencyStudent(window)
    l0_student.fit(train, train_heads, train_pos)
    config = RelationLabelerConfig()
    gold_trained = BilinearRelationLabeler(config)
    print(
        f"FIT: biaffine_gold_trained seed={config.seed} rank={config.rank} "
        f"epochs={config.epochs} lr={config.learning_rate}.",
        flush=True,
    )
    gold_trained.fit(train, train_heads, train_pos)
    print(
        "FIT: biaffine_gold_trained epoch losses="
        + ", ".join(f"{loss:.6f}" for loss in gold_trained.epoch_losses),
        flush=True,
    )

    predicted_train_heads = _train_predicted_governors(
        train, train_pos, l0_student
    )
    train_deprels = _gold_train_targets(train, train_heads)
    noise_matched = BilinearRelationLabeler(config)
    print(
        f"FIT: biaffine_noise_matched seed={config.seed} rank={config.rank} "
        f"epochs={config.epochs} lr={config.learning_rate}.",
        flush=True,
    )
    noise_matched.fit_on_supplied_governors(
        train,
        predicted_train_heads,
        train_deprels,
        train_pos,
    )
    print(
        "FIT: biaffine_noise_matched epoch losses="
        + ", ".join(f"{loss:.6f}" for loss in noise_matched.epoch_losses),
        flush=True,
    )

    guard = CampaignTestReadGuard()
    print(
        "FINAL DIAGNOSTIC: reading shared test tasks once and head_deprel test once.",
        flush=True,
    )
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_head_records = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads_raw = head_deprel_by_sent(test_head_records)
    test_heads = {
        sent_id: record
        for sent_id, record in test_heads_raw.items()
        if isinstance(record, HeadDeprelRecord)
    }
    if len(test_heads) != len(test_heads_raw):
        raise TypeError("labeled dependency records required")
    test_pos = _predicted_pos_by_sentence(r1_student, test)
    table = _evaluate_cells(
        test,
        test_heads,
        test_pos,
        l0_student,
        gold_trained,
        noise_matched,
        r1_student,
    )
    guard.assert_complete()

    indexed = _indexed_table(table)
    test_gold_accuracy = float(
        indexed[("biaffine_gold_trained", "gold_heads")]["deprel_only"]
    )
    memorization = _memorization_scores(
        train,
        train_heads,
        train_pos,
        gold_trained,
        test_gold_accuracy,
    )
    crosscheck = _primary_crosscheck(table)
    reads = _make_reads(table, memorization, crosscheck)
    artifact: dict[str, Any] = {
        "table": table,
        "reads": reads,
        "memorization": memorization,
        "noise_matched_gains_vs_unary_predicted_heads": _gains(
            indexed[("biaffine_noise_matched", "predicted_heads")],
            indexed[("unary", "predicted_heads")],
        ),
        "data_hash_sha256": hasher.hexdigest(),
        "test_read_counts": dict(guard.counts),
        "l0_window": {
            "k": window,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "fixed_hyperparameters": {
            "rank": config.rank,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "l2_penalty": config.l2_penalty,
            "seed": config.seed,
        },
        "partition_definitions": {
            "pairwise": sorted(PAIRWISE_SENSITIVE_DEPRELS),
            "local": sorted(LOCAL_DEPRELS),
        },
        "primary_crosscheck": crosscheck,
    }
    return artifact, guard


def print_diagnostic(
    artifact: Mapping[str, Any], guard: CampaignTestReadGuard
) -> None:
    """Print the five diagnostic cells and all computed reads."""
    print("\nBIAFFINE RELATION-LABELER DIAGNOSTIC — GSD TEST")
    print(
        "arm                       head_source      deprel_only las_strict "
        "serve_honest_case pairwise  local"
    )
    for row in artifact["table"]:
        print(
            f"{row['arm']:<26} {row['head_source']:<16} "
            f"{row['deprel_only']:.6f}    {row['las_strict']:.6f}   "
            f"{row['serve_honest_case']:.6f}          "
            f"{row['pairwise']:.6f}  {row['local']:.6f}"
        )
    print()
    for key in ("q1", "q2", "q3"):
        print(artifact["reads"][key])
    memorization = artifact["memorization"]
    print(
        "MEMORIZATION GAP: "
        f"train={memorization['train_deprel_only']:.6f} "
        f"test_gold_heads={memorization['test_gold_heads_deprel_only']:.6f} "
        f"train-test={memorization['train_minus_test']:+.6f}"
    )
    guard.assert_complete()
    print("DIAGNOSTIC TEST READ ONCE: CONFIRMED")


def main() -> None:
    artifact, guard = build_diagnostic()
    OUTPUT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"structured_output: {OUTPUT_PATH}")
    print_diagnostic(artifact, guard)


if __name__ == "__main__":
    main()
