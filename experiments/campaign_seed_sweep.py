"""Registered phased B0 seed sweep for admission-composition stability (slice #84).

The default mode resumes phases A, B1, B2, and B3 in order. ``--worker`` performs exactly one
B0 regeneration. Importing this module is safe for unit tests: no sweep runs at import time.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# A worker must establish import-time recipe constants and caller-side seeds before importing the
# #81 helper, because that helper imports wyly_lm_v5.  PYTHONHASHSEED is intentionally absent here:
# only the orchestrator's child-process environment can establish it before interpreter startup.
_BOOTSTRAP_RECIPE = {
    "WYLY_TAG": "pythia70m",
    "WYLY_DS": "wikitext",
    "WYLY_LIB": "mined",
    "WYLY_JUDGE": "cover",
    "WYLY_ONLINE": "1",
    "WYLY_COVER": "sw",
    "WYLY_CONCEPTS": "1",
    "WYLY_POINTER": "1",
    "WYLY_TPOINTER": "1",
    "WYLY_DX": "1",
    "WYLY_CX": "1",
    "WYLY_FOLDS": "3",
}
for _bootstrap_key, _bootstrap_value in _BOOTSTRAP_RECIPE.items():
    os.environ[_bootstrap_key] = _bootstrap_value
if "--worker" in sys.argv:
    try:
        _bootstrap_seed = int(sys.argv[sys.argv.index("--seed") + 1])
    except (ValueError, IndexError):
        _bootstrap_seed = 0  # argparse emits the useful error later for malformed worker argv.
    random.seed(_bootstrap_seed)
    np.random.seed(_bootstrap_seed)
    torch.manual_seed(_bootstrap_seed)
    if os.environ.get("SEED_SWEEP_DETERMINISTIC") == "1":
        torch.use_deterministic_algorithms(True)

# Reuse #81's canonical recipe and name normalization instead of reproducing either concept.
import wyly_lm_v5 as v5  # noqa: E402

from experiments.campaign_wikitext_gated_rescue import (  # noqa: E402
    _RECIPE,
    _base_rule_name,
    freeze_b0,
    load_frozen_b0,
)

assert _RECIPE == _BOOTSTRAP_RECIPE

SWEEP_DIR = REPO / "data" / "seed_sweep"
SUMMARY_PATH = REPO / "data" / "seed_sweep_summary.json"
FORBIDDEN_OUTPUT = (REPO / "data" / "wikitext_gated_rescue_b0.pt").resolve()
ARCHIVE_PATH = REPO / "data" / "wyly_v5_mined_pythia70m_wikitext_cov_ol_sw_cn_pt_tp_dx_cx_f3.pt"
SEED0_CAPTURE = SWEEP_DIR / "seed0_rep0_fitted.pt"
B1_SEED0_CAPTURE = SWEEP_DIR / "b1_seed0_rep0_fitted.pt"
CORE_FIELDS = ("agree", "cover", "agree_fired")
TOLERANCE = 1e-6
CONSENSUS_THRESHOLD = 7
NOISE_THRESHOLD = 5
SLOW_RUN_SECONDS = 600.0

PART_A_BINARY_RULE = (
    "REGISTERED BINARY: deterministic iff all three have identical admitted compositions "
    "(as sets of names) AND core_sw (all three fields: agree/cover/agree_fired) within 1e-6 "
    "pairwise (all 3 choose 2 pairs)."
)
PRE_NAMED_READINGS = (
    "PRE-NAMED READINGS: STABLE-CORE iff a nonempty set of (base-name) rules appears in >= "
    "7/8 B3 internal-seed runs; SEED-NOISE iff most admitted (base-name) rules appear in < "
    "5/8 B3 internal-seed runs."
)
CONSENSUS_CAVEAT = (
    "Pre-named caveat: if the re-score dips vs. the seed-0 run's own FULL (all admitted rules, "
    "not just consensus) core_sw.agree, check whether the specific consensus-member rule(s) in "
    "THIS internal-seed-0 run have an unusually WEAK fit relative to how the SAME base-name rule "
    "fits in the 8 B3 internal-seed runs (e.g., compare per-seed marginal/support numbers for that rule "
    "name if you have them handy from the per-run JSONs) BEFORE concluding \"fringe rules carry "
    "the score\"."
)
HONESTY_NOTE = (
    "The native WYLY_SEED knob measures internal-seed diversity while the unchanged caller-side "
    "bootstrap seeds measure external-seed invariance."
)

def output_path(seed: int, rep: int, condition: str = "default") -> Path:
    """Return an isolated per-run JSON path, guarded against the #81 frozen artifact."""
    safe_condition = re.sub(r"[^a-z0-9_-]+", "-", condition.lower()).strip("-")
    prefix = "run" if safe_condition == "default" else f"run_{safe_condition}"
    path = (SWEEP_DIR / f"{prefix}_seed{int(seed)}_rep{int(rep)}.json").resolve()
    if path == FORBIDDEN_OUTPUT:
        raise AssertionError("seed-sweep output must never overwrite the #81 frozen B0 artifact")
    if SWEEP_DIR.resolve() not in path.parents:
        raise AssertionError("seed-sweep output escaped data/seed_sweep")
    return path


def build_subprocess_env(
    seed: int,
    condition: str = "default",
    *,
    internal_seed: int = 0,
    force_cpu: bool = False,
) -> dict[str, str]:
    """Build the interpreter-start environment; PYTHONHASHSEED is deliberately always fixed."""
    env = {**os.environ, **_RECIPE, "PYTHONHASHSEED": "0", "SEED_SWEEP_SEED": str(int(seed))}
    env.pop("SEED_SWEEP_DETERMINISTIC", None)
    env["WYLY_SEED"] = str(int(internal_seed))
    if condition == "cpu" or force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif condition == "deterministic":
        env["SEED_SWEEP_DETERMINISTIC"] = "1"
    elif condition not in {"default", "b1", "b2", "b3"}:
        raise ValueError(f"unknown run condition: {condition}")
    return env


def base_name_set(names: list[str] | set[str] | tuple[str, ...]) -> set[str]:
    return {_base_rule_name(name) for name in names}


def identical_compositions(
    records: list[dict[str, Any]], *, normalize: bool = True
) -> bool:
    sets = [
        base_name_set(record["admitted_names"])
        if normalize
        else set(record["admitted_names"])
        for record in records
    ]
    return bool(sets) and all(names == sets[0] for names in sets[1:])


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def jaccard_matrix(name_sets: list[set[str]]) -> list[list[float]]:
    return [[jaccard(left, right) for right in name_sets] for left in name_sets]


def frequency_table(
    seed_to_names: dict[int, set[str]],
) -> list[dict[str, Any]]:
    seeds_by_name: dict[str, list[int]] = defaultdict(list)
    for seed, names in sorted(seed_to_names.items()):
        for name in sorted(names):
            seeds_by_name[name].append(seed)
    return [
        {"name": name, "count": len(seeds), "seeds": seeds}
        for name, seeds in sorted(
            seeds_by_name.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]


def consensus_membership(
    seed_to_names: dict[int, set[str]], threshold: int = CONSENSUS_THRESHOLD
) -> set[str]:
    return {
        row["name"] for row in frequency_table(seed_to_names) if row["count"] >= threshold
    }


def evaluate_part_a(records: list[dict[str, Any]], *, normalize: bool = True) -> bool:
    if len(records) != 3 or not identical_compositions(records, normalize=normalize):
        return False
    for left_index in range(3):
        for right_index in range(left_index + 1, 3):
            left, right = records[left_index], records[right_index]
            if any(
                abs(float(left["core_sw"][field]) - float(right["core_sw"][field]))
                > TOLERANCE
                for field in CORE_FIELDS
            ):
                return False
    return True


def conditional_conditions(part_a_passed: bool) -> tuple[str, ...]:
    return () if part_a_passed else ("cpu", "deterministic")


def evaluate_pair(records: list[dict[str, Any]], *, normalize: bool = True) -> bool | None:
    if len(records) != 2 or any(record.get("status") != "ok" for record in records):
        return None
    if not identical_compositions(records, normalize=normalize):
        return False
    return all(
        abs(float(records[0]["core_sw"][field]) - float(records[1]["core_sw"][field]))
        <= TOLERANCE
        for field in CORE_FIELDS
    )


def first_divergence(left_path: Path, right_path: Path) -> dict[str, Any] | None:
    left = left_path.read_text(errors="replace").splitlines()
    right = right_path.read_text(errors="replace").splitlines()
    for line_number, pair in enumerate(zip(left, right, strict=False), start=1):
        left_line, right_line = pair
        if left_line != right_line:
            prior = left[: line_number - 1]
            episode = next(
                (line.strip() for line in reversed(prior) if "sleep " in line), None
            )
            return {
                "line_number": line_number,
                "last_sleep_context": episode,
                "left": left_line,
                "right": right_line,
            }
    return None


def _guard_worker_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == FORBIDDEN_OUTPUT or SWEEP_DIR.resolve() not in resolved.parents:
        raise ValueError("--out must be under data/seed_sweep and cannot be the #81 artifact")
    return resolved


def _worker(seed: int, rep: int, out: Path) -> int:
    out = _guard_worker_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for key, value in _RECIPE.items():
        os.environ[key] = value
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    deterministic = os.environ.get("SEED_SWEEP_DETERMINISTIC") == "1"
    internal_seed = int(os.environ.get("WYLY_SEED", "0"))
    if deterministic:
        torch.use_deterministic_algorithms(True)

    original = v5.core_cover_sw
    captured: dict[str, Any] = {}

    def capture(model, rules, ids, yv, cls, idxs, **kwargs):
        captured["model"] = model
        captured["rules"] = list(rules)
        return original(model, rules, ids, yv, cls, idxs, **kwargs)

    # Never let v5.main overwrite the archived reference.  Its small ordinary state file is
    # temporary; the explicitly requested JSON/log outputs are the durable per-run records.
    temporary_state = out.with_suffix(".state.pt")
    v5.STATE = temporary_state
    start = time.perf_counter()
    v5.core_cover_sw = capture
    try:
        try:
            v5.main()
        except RuntimeError as error:
            if not deterministic:
                raise
            result = {
                "status": "not_feasible",
                "seed": seed,
                "external_seed": seed,
                "internal_seed": internal_seed,
                "rep": rep,
                "error": str(error),
                "wall_seconds": time.perf_counter() - start,
                "device": str(v5.DEV),
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                "torch_deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
            }
            out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return 0
    finally:
        v5.core_cover_sw = original
        temporary_state.unlink(missing_ok=True)
    wall = time.perf_counter() - start
    if not captured:
        raise RuntimeError("core_cover_sw capture wrapper was never called")

    model = captured["model"]
    rules = list(model.rules)
    ids, y, cls, _uv, _tr, te = v5.load_ds()
    yv = cls[y]
    core_sw, _pred = original(model, rules, ids, yv, cls, te, return_pred=True)
    leave_one_out = {}
    for name, _fn in rules:
        reduced = [rule for rule in rules if rule[0] != name]
        score = original(model, reduced, ids, yv, cls, te)
        leave_one_out[name] = {
            **score,
            "agree_delta_full_minus_without": core_sw["agree"] - score["agree"],
        }

    if os.environ.get("SEED_SWEEP_FREEZE"):
        freeze_path = _guard_worker_path(Path(os.environ["SEED_SWEEP_FREEZE"]))
        freeze_b0(freeze_path, model, rules, core_sw, wall)

    result = {
        "status": "ok",
        "seed": seed,
        "external_seed": seed,
        "internal_seed": internal_seed,
        "rep": rep,
        "admitted_names": [name for name, _fn in rules],
        "admitted_order_kind": "final model.rules order (after admission/elimination/swap)",
        "core_sw": core_sw,
        "rule_leave_one_out": leave_one_out,
        "wall_seconds": wall,
        "device": str(v5.DEV),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def _run_one(
    seed: int,
    rep: int,
    condition: str = "default",
    *,
    freeze: bool = False,
    force: bool = False,
    internal_seed: int = 0,
    artifact_seed: int | None = None,
    force_cpu: bool = False,
):
    """Run or resume one subprocess-backed regeneration.

    ``seed`` is always the external seed. ``artifact_seed`` only controls the human-readable
    filename (B3 uses its internal seed there).
    """
    out = output_path(seed if artifact_seed is None else artifact_seed, rep, condition)
    log_path = out.with_suffix(".log")
    if out.exists() and not force:
        record = json.loads(out.read_text())
        record.setdefault("external_seed", seed)
        record.setdefault("internal_seed", internal_seed)
        record["condition"] = condition
        record["log_path"] = str(log_path.relative_to(REPO))
        record["resumed"] = True
        print(
            f"resumed {condition} external_seed={seed} internal_seed={internal_seed} "
            f"rep={rep}: {out.relative_to(REPO)}",
            flush=True,
        )
        return record

    env = build_subprocess_env(
        seed, condition, internal_seed=internal_seed, force_cpu=force_cpu
    )
    if freeze:
        capture_path = B1_SEED0_CAPTURE if condition == "b1" else SEED0_CAPTURE
        env["SEED_SWEEP_FREEZE"] = str(capture_path)
    command = [
        sys.executable,
        __file__,
        "--worker",
        "--seed",
        str(seed),
        "--rep",
        str(rep),
        "--out",
        str(out),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, env=env, check=False, capture_output=True, text=True
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(
        completed.stdout
        + ("\n=== STDERR ===\n" + completed.stderr if completed.stderr else "")
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode, command, completed.stdout, completed.stderr
        )
    record = json.loads(out.read_text())
    record["condition"] = condition
    record["log_path"] = str(log_path.relative_to(REPO))
    record["exceeded_10_minutes"] = elapsed > SLOW_RUN_SECONDS
    record["resumed"] = False
    print(
        f"completed {condition} external_seed={seed} internal_seed={internal_seed} "
        f"rep={rep}: {elapsed:.1f}s"
        + (" (EXCEEDED 10 MINUTES; allowed to finish)" if elapsed > SLOW_RUN_SECONDS else ""),
        flush=True,
    )
    return record


def _spread(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(record["core_sw"][field]) for record in records]
    return {"min": min(values), "median": median(values), "max": max(values)}


def _archived_base_names() -> tuple[set[str], list[str]]:
    payload = torch.load(ARCHIVE_PATH, map_location="cpu", weights_only=False)
    names = list(payload["rules"])
    return base_name_set(names), names


def _consensus_rescore(
    consensus: set[str], capture_path: Path = B1_SEED0_CAPTURE
) -> tuple[dict[str, float], list[str]]:
    model, rules, _core, _wall = load_frozen_b0(capture_path)
    selected = [rule for rule in rules if _base_rule_name(rule[0]) in consensus]
    ids, y, cls, _uv, _tr, te = v5.load_ds()
    yv = cls[y]
    return v5.core_cover_sw(model, selected, ids, yv, cls, te), [name for name, _ in selected]


def _reading_flags(frequencies: list[dict[str, Any]]) -> dict[str, bool]:
    stable = any(row["count"] >= CONSENSUS_THRESHOLD for row in frequencies)
    # "Most admitted rules" is interpreted over distinct admitted base names.
    noisy = bool(frequencies) and sum(
        row["count"] < NOISE_THRESHOLD for row in frequencies
    ) > len(frequencies) / 2
    return {"STABLE-CORE": stable, "SEED-NOISE": noisy}


def _marginal_caveat_check(
    seed0: dict[str, Any], part_b: list[dict[str, Any]], consensus: set[str]
) -> list[dict[str, Any]]:
    rows = []
    for base in sorted(consensus):
        observations = []
        for record in [seed0, *part_b]:
            for actual, metrics in record.get("rule_leave_one_out", {}).items():
                if _base_rule_name(actual) == base:
                    support_match = re.search(r"\[(\d+)\]$", actual)
                    observations.append(
                        {
                            "seed": record.get("internal_seed", record["seed"]),
                            "rep": record["rep"],
                            "actual_name": actual,
                            "support_suffix": (
                                int(support_match.group(1)) if support_match else None
                            ),
                            "agree_delta_full_minus_without": metrics[
                                "agree_delta_full_minus_without"
                            ],
                        }
                    )
        seed0_observation = next(
            (row for row in observations if row["seed"] == 0 and row["rep"] == 0), None
        )
        rows.append(
            {"base_name": base, "seed0": seed0_observation, "all_observations": observations}
        )
    return rows


def _run_phase(phase: str, *, force: bool) -> list[dict[str, Any]]:
    if phase == "a":
        return [
            _run_one(0, rep, freeze=(rep == 0), force=force, force_cpu=True)
            for rep in range(3)
        ]
    if phase == "b1":
        return [
            _run_one(0, rep, "b1", freeze=(rep == 0), force=force)
            for rep in range(2)
        ]
    if phase == "b2":
        return [_run_one(seed, 0, "b2", force=force) for seed in range(1, 5)]
    if phase == "b3":
        return [
            _run_one(
                0,
                0,
                "b3",
                force=force,
                internal_seed=internal_seed,
                artifact_seed=internal_seed,
            )
            for internal_seed in range(1, 9)
        ]
    raise ValueError(f"unknown phase: {phase}")


def _off_diagonal(matrix: list[list[float]]) -> list[float]:
    return [matrix[i][j] for i in range(len(matrix)) for j in range(i + 1, len(matrix))]


def _print_phase_scoreboard(phase: str, records: list[dict[str, Any]]) -> None:
    if phase == "a":
        print(PART_A_BINARY_RULE)
        print(
            "PART A REGISTERED BINARY (base-name sets): "
            + ("PASS" if evaluate_part_a(records) else "FAIL")
        )
    elif phase == "b1":
        result = (
            evaluate_pair(records)
            if all(str(record.get("device", "")).startswith("cuda") for record in records)
            else None
        )
        print(
            "B1 GPU same-seed determinism binary: "
            + ("NOT FEASIBLE" if result is None else "PASS" if result else "FAIL")
        )
    else:
        print(f"Phase {phase.upper()} complete: {len(records)} run records available.")


def _analyze(
    part_a: list[dict[str, Any]],
    b1: list[dict[str, Any]],
    b2: list[dict[str, Any]],
    b3: list[dict[str, Any]],
    *,
    total_wall: float,
) -> None:
    archived_base, archived_raw = _archived_base_names()
    part_a_base_pass = evaluate_part_a(part_a, normalize=True)
    part_a_raw_pass = evaluate_part_a(part_a, normalize=False)
    b1_gpu_device = all(
        str(record.get("device", "")).startswith("cuda") for record in b1
    )
    b1_base_pass = evaluate_pair(b1, normalize=True) if b1_gpu_device else None
    b1_raw_pass = evaluate_pair(b1, normalize=False) if b1_gpu_device else None

    print("=== PART A: CPU SAME-SEED DETERMINISM ===")
    print(PART_A_BINARY_RULE)
    print(f"PART A REGISTERED BINARY (base-name sets): {'PASS' if part_a_base_pass else 'FAIL'}")
    print(f"Supplementary raw-name binary: {'PASS' if part_a_raw_pass else 'FAIL'}")
    print("=== B1: GPU SAME-SEED DETERMINISM ===")
    print(f"B1 devices: {[record.get('device') for record in b1]}")
    print(
        "B1 GPU same-seed determinism binary: "
        + ("NOT FEASIBLE" if b1_base_pass is None else "PASS" if b1_base_pass else "FAIL")
    )
    print(
        "B1 supplementary raw-name binary: "
        + ("NOT FEASIBLE" if b1_raw_pass is None else "PASS" if b1_raw_pass else "FAIL")
    )

    b1_reference = b1[0]
    reference_names = base_name_set(b1_reference["admitted_names"])
    b2_composition_mismatches = [
        record["external_seed"]
        for record in b2
        if base_name_set(record["admitted_names"]) != reference_names
    ]
    b2_core_drift = [
        {
            "external_seed": record["external_seed"],
            **{
                field: float(record["core_sw"][field])
                - float(b1_reference["core_sw"][field])
                for field in CORE_FIELDS
            },
        }
        for record in b2
    ]
    b2_core_matches = all(
        abs(row[field]) <= TOLERANCE for row in b2_core_drift for field in CORE_FIELDS
    )
    print("=== B2: EXTERNAL-SEED INVARIANCE (INTERNAL SEED 0) ===")
    print(f"B2 devices: {[record.get('device') for record in b2]}")
    print(
        "B2 compositions vs B1 rep 0: "
        + ("all 4 match" if not b2_composition_mismatches else f"mismatches={b2_composition_mismatches}")
    )
    print(f"B2 core_sw drift vs B1 rep 0: {json.dumps(b2_core_drift, sort_keys=True)}")
    print(f"B2 core_sw invariant within 1e-6: {b2_core_matches}")

    seed_to_base = {
        record["internal_seed"]: base_name_set(record["admitted_names"]) for record in b3
    }
    seed_to_raw = {
        record["internal_seed"]: set(record["admitted_names"]) for record in b3
    }
    base_matrix = jaccard_matrix(list(seed_to_base.values()))
    raw_matrix = jaccard_matrix(list(seed_to_raw.values()))
    base_off_diagonal = _off_diagonal(base_matrix)
    raw_off_diagonal = _off_diagonal(raw_matrix)
    frequencies = frequency_table(seed_to_base)
    consensus = consensus_membership(seed_to_base)
    flags = _reading_flags(frequencies)
    spread = {field: _spread(b3, field) for field in CORE_FIELDS}
    anomalies = [
        {"internal_seed": record["internal_seed"], "agree": record["core_sw"]["agree"]}
        for record in b3
        if not 0.341 <= record["core_sw"]["agree"] <= 0.351
    ]

    print("=== B3: 8 INTERNAL-SEED COMPOSITION SWEEP ===")
    print(f"B3 devices: {[record.get('device') for record in b3]}")
    print("Base-name pairwise Jaccard matrix (rows/columns internal seeds 1..8):")
    for seed, row in zip(seed_to_base, base_matrix, strict=True):
        print(f"  {seed:2d}: " + " ".join(f"{value:.3f}" for value in row))
    print(
        "Jaccard summary (base): "
        f"min={min(base_off_diagonal):.3f} median={median(base_off_diagonal):.3f} "
        f"max={max(base_off_diagonal):.3f}"
    )
    print(
        "Jaccard summary (raw supplementary): "
        f"min={min(raw_off_diagonal):.3f} median={median(raw_off_diagonal):.3f} "
        f"max={max(raw_off_diagonal):.3f}"
    )
    print("Per-rule admission frequency (base name):")
    for row in frequencies:
        print(f"  {row['name']}: {row['count']}/8 internal seeds={row['seeds']}")
    print(f"core_sw spread: {json.dumps(spread, sort_keys=True)}")
    print(f"Registered [0.341, 0.351] per-seed anomalies: {anomalies or 'none'}")
    print(PRE_NAMED_READINGS)
    applies = [name for name, applies in flags.items() if applies]
    print(f"PRE-NAMED READINGS applying: {', '.join(applies) if applies else 'neither'}")

    print("=== CONSENSUS-CORE RE-SCORE ===")
    consensus_core, consensus_actual_names = _consensus_rescore(consensus)
    print(f"Membership (>=7/8 B3 internal seeds): {sorted(consensus)}")
    print(f"B1 internal-seed-0 rep-0 actual variants used: {consensus_actual_names}")
    print(f"Consensus core_sw: {json.dumps(consensus_core, sort_keys=True)}")
    print("This tests STABLE MEMBERSHIP with B1 rep 0 parameters, not an ensemble.")
    print(CONSENSUS_CAVEAT)
    caveat_check = _marginal_caveat_check(b1_reference, b3, consensus)
    dip = consensus_core["agree"] < b1_reference["core_sw"]["agree"]

    cpu_names = base_name_set(part_a[0]["admitted_names"])
    cross_device = {
        "cpu_only": sorted(cpu_names - reference_names),
        "gpu_only": sorted(reference_names - cpu_names),
        "symmetric_difference": sorted(cpu_names ^ reference_names),
        "core_sw_agree_delta_gpu_minus_cpu": (
            float(b1_reference["core_sw"]["agree"])
            - float(part_a[0]["core_sw"]["agree"])
        ),
    }
    b2_invariant = not b2_composition_mismatches and b2_core_matches
    device_difference_observed = bool(cross_device["symmetric_difference"]) or abs(
        cross_device["core_sw_agree_delta_gpu_minus_cpu"]
    ) > TOLERANCE
    hypothesis_consistent = bool(
        part_a_base_pass and b1_base_pass and b2_invariant and device_difference_observed
    )
    hypothesis_note = (
        "CONSISTENT with the amended hypothesis: composition is deterministic per (code version, "
        "device), external seed is invariant, and a CPU/GPU difference is observed."
        if hypothesis_consistent
        else "NOT FULLY CONSISTENT with the amended hypothesis in this run: at least one of "
        "per-device determinism, external-seed invariance, or an observable CPU/GPU difference "
        "was absent."
    )
    print("=== CROSS-DEVICE VERDICT: PART A CPU VS B1 GPU REP 0 ===")
    print(json.dumps(cross_device, sort_keys=True))
    print(hypothesis_note)

    print("=== DRIFT VS. ARCHIVED ARTIFACT (B3) ===")
    per_seed_drift = {
        seed: sorted(names ^ archived_base) for seed, names in seed_to_base.items()
    }
    drift_sets = [set(names) for names in per_seed_drift.values()]
    systematic = sorted(set.intersection(*drift_sets))
    seed_varying = sorted(set.union(*drift_sets) - set(systematic))
    print(f"Archived raw names: {archived_raw}")
    for seed, differences in per_seed_drift.items():
        print(f"  internal seed {seed}: symmetric difference={differences}")
    print(f"SYSTEMATIC: {systematic}")
    print(f"SEED-VARYING: {seed_varying}")

    recommendation = (
        "RECOMMENDATION: if a package needs a fixed, auditable rule set, freeze the >=7/8 B3 "
        "consensus core rather than any single regeneration's full output."
    )
    print("=== HONESTY / POLICY ===")
    print(HONESTY_NOTE)
    print("Everything beyond the registered binaries is DESCRIPTIVE; it is not a pass/fail gate.")
    print(recommendation)
    print(f"Total orchestrator wall time: {total_wall:.1f}s ({total_wall / 60:.1f} min)")

    summary = {
        "honesty_note": HONESTY_NOTE,
        "comparison_choices": {
            "part_a_and_b1": "base-name sets; raw-name results also reported",
            "b2_and_b3": "base-name sets; raw-name B3 Jaccard also reported",
            "order": "final model.rules order, not reconstructed greedy-only order",
        },
        "part_a": {
            "runs": part_a,
            "registered_base_name_binary": part_a_base_pass,
            "supplementary_raw_name_binary": part_a_raw_pass,
        },
        "b1": {
            "runs": b1,
            "all_runs_gpu": b1_gpu_device,
            "gpu_same_seed_determinism_binary": b1_base_pass,
            "supplementary_raw_name_binary": b1_raw_pass,
            "consensus_fit_reference_rep": 0,
        },
        "b2": {
            "runs": b2,
            "reference": "B1 rep 0 (external=0, internal=0)",
            "composition_mismatch_external_seeds": b2_composition_mismatches,
            "all_compositions_match": not b2_composition_mismatches,
            "core_sw_drift": b2_core_drift,
            "core_sw_invariant_within_1e-6": b2_core_matches,
        },
        "b3": {
            "runs": b3,
            "base_name_jaccard_matrix": base_matrix,
            "raw_name_jaccard_matrix": raw_matrix,
            "frequency_table": frequencies,
            "core_sw_spread": spread,
            "registered_band_anomalies": anomalies,
            "readings": flags,
        },
        "consensus": {
            "membership": sorted(consensus),
            "threshold": ">=7/8 B3 internal-seed runs",
            "fit_reference": "B1 internal-seed-0 rep 0",
            "actual_names": consensus_actual_names,
            "core_sw": consensus_core,
            "full_reference_core_sw": b1_reference["core_sw"],
            "dip": dip,
            "marginal_support_check": caveat_check,
            "interpretation": "stable membership with B1 rep 0 parameters, not an ensemble",
        },
        "cross_device": {
            **cross_device,
            "device_difference_observed": device_difference_observed,
            "amended_hypothesis_consistent": hypothesis_consistent,
            "verdict": hypothesis_note,
        },
        "drift": {
            "archived_raw_names": archived_raw,
            "per_internal_seed_symmetric_difference": per_seed_drift,
            "SYSTEMATIC": systematic,
            "SEED-VARYING": seed_varying,
        },
        "recommendation": recommendation,
        "total_wall_seconds": total_wall,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def orchestrate(phase: str = "all", *, force: bool = False) -> int:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    phases = ("a", "b1", "b2", "b3") if phase == "all" else (phase,)
    records_by_phase = {}
    for selected in phases:
        print(f"=== RUN PHASE {selected.upper()} ===", flush=True)
        records_by_phase[selected] = _run_phase(selected, force=force)
        _print_phase_scoreboard(selected, records_by_phase[selected])
    if phase == "all":
        _analyze(
            records_by_phase["a"],
            records_by_phase["b1"],
            records_by_phase["b2"],
            records_by_phase["b3"],
            total_wall=time.perf_counter() - total_start,
        )
    else:
        print("Phase-only invocation complete; full analysis/summary is written by --phase all.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--phase", choices=("a", "b1", "b2", "b3", "all"), default="all"
    )
    parser.add_argument(
        "--force", action="store_true", help="rerun and overwrite selected run artifacts"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rep", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.worker and any(value is None for value in (args.seed, args.rep, args.out)):
        parser.error("--worker requires --seed, --rep, and --out")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return _worker(args.seed, args.rep, args.out)
    return orchestrate(args.phase, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
