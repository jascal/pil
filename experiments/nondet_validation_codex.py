"""Slice 5 non-deterministic beam validation pilot (independent Codex lane).

The registered thresholds, M sweep, strata, and prune-bite definition are fixed by
PIL_NONDET_VALIDATION_PREREG.md. Generated puzzles remain in memory; the only written
artifact is the aggregate/audit report at ``data/nondet_validation_codex.json``.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))
sys.path.insert(0, str(REPO / "experiments"))

import campaign_gate_b_pilot as gbp
import campaign_sudoku_forced_move as osc
from beam_engine import beam_decode
from sudoku_oracle import SudokuOracle

OUTPUT = REPO / "data" / "nondet_validation_codex.json"

# Registered constants -- do not tune.
BEAM_WIDTH = 8
WYLY_SEED = 0
RULE_ID = "decide_energy_v1"
M_SWEEP = list(range(1, 9))
SEPARATION_THRESHOLD = 0.15
BEAM_ACC_FLOOR = 0.50

# Reproducibility/runtime constants. These are population-generation controls, not gates.
GENERATOR_SEED = 20260713
HOLES = 45
BRANCH_CELL_SOFT_TARGET = 1_000
GENERATION_ATTEMPT_CAP = 30_000

Board = list[list[int]]
CountTable = list[list[list[int]]]


def log(message: str = "") -> None:
    print(message, flush=True)


def fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def copied(board: Board) -> Board:
    return [row[:] for row in board]


def generate_full_grid(rng: random.Random) -> Board:
    """Generate one valid completed grid by randomized row-major backtracking."""
    board = [[0] * 9 for _ in range(9)]

    def rec(cell: int) -> bool:
        if cell == 81:
            return True
        r, c = divmod(cell, 9)
        values = list(range(1, 10))
        rng.shuffle(values)
        legal = osc.cell_candidates(board, r, c)
        for value in values:
            if value not in legal:
                continue
            board[r][c] = value
            if rec(cell + 1):
                return True
        board[r][c] = 0
        return False

    if not rec(0):
        raise RuntimeError("randomized full-grid construction unexpectedly failed")
    return board


def remove_cells(solution: Board, rng: random.Random, n_holes: int = HOLES) -> Board:
    board = copied(solution)
    for cell in rng.sample(range(81), n_holes):
        r, c = divmod(cell, 9)
        board[r][c] = 0
    return board


def propagation_fixpoint(board: Board) -> tuple[Board, bool]:
    """Apply the shared synchronous propagation pass until fixpoint or contradiction."""
    grid = copied(board)
    while True:
        forced, contradiction = gbp.propagate_one_pass(grid)
        if contradiction:
            return grid, True
        if not forced:
            return grid, False
        for r, c, value in forced:
            grid[r][c] = value


def count_solutions(board: Board, stop_after: int = 2) -> int:
    """Count solutions with an MRV pivot, stopping once ``stop_after`` are found."""
    found = 0

    def rec(grid: Board) -> None:
        nonlocal found
        if found >= stop_after:
            return
        empties = [(r, c) for r in range(9) for c in range(9) if grid[r][c] == 0]
        if not empties:
            found += 1
            return
        pivot = min(empties, key=lambda rc: (len(osc.cell_candidates(grid, *rc)), rc))
        pr, pc = pivot
        candidates = sorted(osc.cell_candidates(grid, pr, pc))
        if not candidates:
            return
        for value in candidates:
            grid[pr][pc] = value
            rec(grid)
            grid[pr][pc] = 0
            if found >= stop_after:
                return

    rec(copied(board))
    return found


def branch_cells(board: Board) -> list[tuple[int, int]]:
    """Return empty cells on which shared single-cell propagation reaches a stall."""
    return [
        (r, c)
        for r in range(9)
        for c in range(9)
        if board[r][c] == 0 and gbp.propagation_depth(board, r, c) is None
    ]


def board_is_consistent(board: Board) -> bool:
    """Check filled assignments using the shared candidate implementation."""
    return all(
        board[r][c] == 0 or board[r][c] in osc.cell_candidates(board, r, c)
        for r in range(9)
        for c in range(9)
    )


def canonical_solve_with_r(board: Board) -> tuple[Board, list[list[int | None]]] | None:
    """Solve by the registered canonical DFS and mark accepted-path nested-guess depths."""

    def rec(
        grid: Board, rgrid: list[list[int | None]], depth: int
    ) -> tuple[Board, list[list[int | None]]] | None:
        while True:
            forced, contradiction = gbp.propagate_one_pass(grid)
            if contradiction:
                return None
            if not forced:
                break
            for r, c, value in forced:
                grid[r][c] = value
                rgrid[r][c] = depth
            # Synchronous singles can expose a failed guess by assigning duplicate
            # peer values in the same pass. Treat that as branch failure even when
            # no empty zero-candidate cell remains for the next pass to flag.
            if not board_is_consistent(grid):
                return None

        empties = [(r, c) for r in range(9) for c in range(9) if grid[r][c] == 0]
        if not empties:
            return grid, rgrid
        pr, pc = min(empties, key=lambda rc: (len(osc.cell_candidates(grid, *rc)), rc))
        for value in sorted(osc.cell_candidates(grid, pr, pc)):
            child = copied(grid)
            child[pr][pc] = value
            child_rgrid = [row[:] for row in rgrid]
            child_rgrid[pr][pc] = depth + 1
            result = rec(child, child_rgrid, depth + 1)
            if result is not None:
                return result
        return None

    rgrid0: list[list[int | None]] = [
        [0 if board[r][c] != 0 else None for c in range(9)] for r in range(9)
    ]
    return rec(copied(board), rgrid0, 0)


def generate_population() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(GENERATOR_SEED)
    puzzles: list[dict[str, Any]] = []
    n_attempted = 0
    n_stall = 0
    n_branch = 0
    n_contradictions = 0
    started = time.monotonic()

    while n_attempted < GENERATION_ATTEMPT_CAP and n_branch < BRANCH_CELL_SOFT_TARGET:
        n_attempted += 1
        solution = generate_full_grid(rng)
        clues = remove_cells(solution, rng)
        fixpoint, contradiction = propagation_fixpoint(clues)
        if contradiction:
            n_contradictions += 1
            log(f"WARNING: generated valid partial grid contradicted at attempt {n_attempted}; discarded")
            continue
        if not any(0 in row for row in fixpoint):
            continue
        n_stall += 1
        if count_solutions(clues) != 1:
            continue
        cells = branch_cells(clues)
        if not cells:
            raise RuntimeError("stall+unique puzzle unexpectedly has no branch cells")
        puzzles.append({"clues": clues, "solution": solution, "branch_cells": cells})
        n_branch += len(cells)
        if n_attempted % 100 == 0 or n_branch >= BRANCH_CELL_SOFT_TARGET:
            log(
                f"generation progress: attempts={n_attempted} stalls={n_stall} "
                f"kept={len(puzzles)} branch_cells={n_branch}"
            )

    cap_hit = n_attempted >= GENERATION_ATTEMPT_CAP and n_branch < BRANCH_CELL_SOFT_TARGET
    provenance = {
        "seed": GENERATOR_SEED,
        "n_attempted": n_attempted,
        "n_propagation_stall": n_stall,
        "n_unique_and_stall": len(puzzles),
        "n_kept": len(puzzles),
        "stall_rate_of_attempted": fraction(n_stall, n_attempted),
        "stall_and_unique_rate_of_attempted": fraction(len(puzzles), n_attempted),
        "n_puzzles_kept": len(puzzles),
        "n_branch_cells_total": n_branch,
        "attempt_cap": GENERATION_ATTEMPT_CAP,
        "attempt_cap_hit": cap_hit,
        "soft_branch_cell_target": BRANCH_CELL_SOFT_TARGET,
        "unexpected_generation_contradictions": n_contradictions,
        "generation_seconds": time.monotonic() - started,
    }
    return puzzles, provenance


def actual_attempted_steps(result: dict[str, Any]) -> int:
    """Actual outer steps attempted, including an early total-collapse step."""
    return len(result["surviving_counts"]) + int(result["anomaly_total_contradiction"])


def prune_bite(results: list[dict[str, Any]]) -> float:
    return fraction(
        sum(int(result["prune_events"]) for result in results),
        sum(actual_attempted_steps(result) for result in results),
    )


def accuracy(cells: list[dict[str, Any]], m: int) -> float | None:
    if not cells:
        return None
    correct = sum(cell["runs"][m]["committed_value"] == cell["gold"] for cell in cells)
    return fraction(correct, len(cells))


def compute_metrics(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_m: dict[str, dict[str, Any]] = {}
    n_cells = len(cells)
    for m in M_SWEEP:
        stratum = [cell for cell in cells if 1 <= cell["refutation_depth"] <= m]
        diagonal = [cell for cell in cells if cell["refutation_depth"] == m]
        horizon = [cell for cell in cells if cell["refutation_depth"] > m]
        beam_correct = sum(cell["runs"][m]["committed_value"] == cell["gold"] for cell in stratum)
        conf_correct = sum(cell["runs"][1]["committed_value"] == cell["gold"] for cell in stratum)
        beam_acc = fraction(beam_correct, len(stratum))
        conf_acc = fraction(conf_correct, len(stratum))
        m_results = [cell["runs"][m] for cell in stratum]
        prune_events = sum(int(result["prune_events"]) for result in m_results)
        attempted_steps = sum(actual_attempted_steps(result) for result in m_results)
        # Full fixed branch-cell population (no r filter): fair across M.
        pop_correct = sum(
            cell["runs"][m]["committed_value"] == cell["gold"] for cell in cells
        )
        per_m[str(m)] = {
            "n_stratum": len(stratum),
            "beam_acc": beam_acc,
            "conf_acc": conf_acc,
            "separation": beam_acc - conf_acc,
            "prune_bite": fraction(prune_events, attempted_steps),
            "prune_events": prune_events,
            "actual_attempted_steps": attempted_steps,
            "diagonal": {
                "n": len(diagonal),
                "beam_acc": accuracy(diagonal, m),
            },
            "horizon_beyond_M": {
                "n": len(horizon),
                "beam_acc": accuracy(horizon, m),
            },
            "population": {
                "n": n_cells,
                "beam_acc": fraction(pop_correct, n_cells),
            },
        }
    return per_m


def compute_verdict(per_m: dict[str, dict[str, Any]]) -> dict[str, Any]:
    positive_prune_ms = [m for m in M_SWEEP if per_m[str(m)]["prune_bite"] > 0]
    sep_met_ms = [m for m in M_SWEEP if per_m[str(m)]["separation"] >= SEPARATION_THRESHOLD]
    floor_met_ms = [m for m in M_SWEEP if per_m[str(m)]["beam_acc"] >= BEAM_ACC_FLOOR]
    sep_and_floor_ms = [m for m in M_SWEEP if m in sep_met_ms and m in floor_met_ms]
    fire_eligible_ms = [m for m in sep_and_floor_ms if m in positive_prune_ms]
    deciding_m = sep_and_floor_ms[0] if sep_and_floor_ms else None

    # Rising signature: population beam_acc(M) over the full fixed branch-cell set.
    population_points = [
        (m, per_m[str(m)]["population"]["beam_acc"]) for m in M_SWEEP
    ]
    population_holds = all(
        left[1] <= right[1] + 1e-15
        for left, right in zip(population_points, population_points[1:], strict=False)
    )

    if not positive_prune_ms:
        verdict = "INVALID"
        explanation = "prune_bite was zero at every M; no registered verdict is permitted"
    else:
        separation_fails = all(
            per_m[str(m)]["separation"] < SEPARATION_THRESHOLD for m in positive_prune_ms
        )
        floor_fails = all(per_m[str(m)]["beam_acc"] < BEAM_ACC_FLOOR for m in M_SWEEP)
        if fire_eligible_ms and population_holds:
            verdict = "FIRES"
            explanation = (
                "at least one M met separation, accuracy floor, and positive pruning, and "
                "population beam_acc(M) is monotonically non-decreasing across M=1..8"
            )
        elif separation_fails or floor_fails:
            verdict = "FAILS"
            reasons = []
            if separation_fails:
                reasons.append("separation stayed below 0.15 at every positive-prune M")
            if floor_fails:
                reasons.append("beam accuracy never reached 0.50")
            explanation = "; ".join(reasons)
        else:
            verdict = "BOUNDARY"
            explanation = (
                "registered negative criteria did not hold, but the complete FIRES conditions "
                "were not all satisfied (population beam_acc(M) not monotonically "
                "non-decreasing across M=1..8, or other conjuncts missing)"
            )

    return {
        "verdict": verdict,
        "deciding_M": deciding_m,
        "sep_met_ms": sep_met_ms,
        "floor_met_ms": floor_met_ms,
        "separation_and_floor_met_ms": sep_and_floor_ms,
        "positive_prune_ms": positive_prune_ms,
        "fire_eligible_ms": fire_eligible_ms,
        "population_signature_holds": population_holds,
        "population_points": [{"M": m, "beam_acc": acc} for m, acc in population_points],
        "separation_and_floor_condition_holds": bool(fire_eligible_ms),
        "explanation": explanation,
    }


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def print_scoreboard(
    provenance: dict[str, Any], r_histogram: dict[str, int], per_m: dict[str, dict[str, Any]],
    verdict: dict[str, Any]
) -> None:
    log("\nPROVENANCE")
    for key, value in provenance.items():
        log(f"  {key}: {value}")
    log("\nR HISTOGRAM")
    for depth, count in r_histogram.items():
        log(f"  r={depth}: {count}")
    log("\nPER-M SCOREBOARD (registered stratum: 1 <= r <= M)")
    log(
        "  M  n     beam_acc  conf_acc  separation  prune_bite  "
        "diag_n diag_acc  horizon_n horizon_acc"
    )
    for m in M_SWEEP:
        row = per_m[str(m)]
        diag = row["diagonal"]
        horizon = row["horizon_beyond_M"]
        diag_acc = "NA" if diag["beam_acc"] is None else f"{diag['beam_acc']:.6f}"
        horizon_acc = "NA" if horizon["beam_acc"] is None else f"{horizon['beam_acc']:.6f}"
        log(
            f"  {m:>1}  {row['n_stratum']:>4}  {row['beam_acc']:.6f}  {row['conf_acc']:.6f}  "
            f"{row['separation']:+.6f}   {row['prune_bite']:.6f}  "
            f"{diag['n']:>6} {diag_acc:>8}  {horizon['n']:>9} {horizon_acc:>11}"
        )
    log("\nREGISTERED CONDITIONS")
    log(
        "  separation+floor+prune condition: "
        f"{verdict['separation_and_floor_condition_holds']} "
        f"(eligible Ms={verdict['fire_eligible_ms']})"
    )
    log(
        f"  rising population signature: {verdict['population_signature_holds']} "
        f"(points={verdict['population_points']})"
    )
    deciding = verdict["deciding_M"]
    if deciding is None:
        deciding_text = "none"
    else:
        row = per_m[str(deciding)]
        deciding_text = (
            f"M={deciding}, separation={row['separation']:.6f}, "
            f"beam_acc={row['beam_acc']:.6f}, prune_bite={row['prune_bite']:.6f}"
        )
    log(
        f"\nVERDICT: {verdict['verdict']} | deciding {deciding_text} | "
        f"{verdict['explanation']}"
    )


def main() -> int:
    run_started = time.monotonic()
    log("Slice 5 non-deterministic beam validation -- CODEX LANE")
    log("Building fixed corpus count table...")
    count_table, n_grids = gbp.build_sudoku_count_table()
    log(f"Count table ready: {n_grids} grids from {gbp.CORPUS}")
    log("Generating deterministic in-memory stall+unique puzzle population...")
    puzzles, generation = generate_population()
    if not puzzles:
        raise RuntimeError("generation produced no unique propagation-stalling puzzles")

    cells: list[dict[str, Any]] = []
    log("Computing canonical refutation depths and running M=1..8 beam sweep...")
    for puzzle_index, puzzle in enumerate(puzzles):
        canonical = canonical_solve_with_r(puzzle["clues"])
        if canonical is None:
            raise RuntimeError(f"canonical solver failed on unique puzzle {puzzle_index}")
        solved, rgrid = canonical
        if solved != puzzle["solution"]:
            raise RuntimeError(f"canonical solution disagrees with generator gold at puzzle {puzzle_index}")
        for r, c in puzzle["branch_cells"]:
            depth = rgrid[r][c]
            if depth is None or depth < 1:
                raise RuntimeError(
                    f"branch cell ({puzzle_index}, {r}, {c}) has invalid refutation depth {depth}"
                )
            runs: dict[int, dict[str, Any]] = {}
            for m in M_SWEEP:
                oracle = SudokuOracle(puzzle["clues"], r, c, count_table, WYLY_SEED)
                runs[m] = beam_decode(oracle, m, BEAM_WIDTH, RULE_ID, WYLY_SEED)
            cells.append(
                {
                    "puzzle_index": puzzle_index,
                    "row": r,
                    "col": c,
                    "gold": puzzle["solution"][r][c],
                    "refutation_depth": depth,
                    "runs": runs,
                }
            )
        if (puzzle_index + 1) % 10 == 0 or puzzle_index + 1 == len(puzzles):
            log(
                f"beam progress: puzzles={puzzle_index + 1}/{len(puzzles)} "
                f"cells={len(cells)}/{generation['n_branch_cells_total']}"
            )

    r_counts = Counter(int(cell["refutation_depth"]) for cell in cells)
    r_histogram = {str(depth): r_counts[depth] for depth in sorted(r_counts)}
    per_m = compute_metrics(cells)
    verdict = compute_verdict(per_m)
    provenance = {
        **generation,
        "beam_width": BEAM_WIDTH,
        "wyly_seed": WYLY_SEED,
        "rule_id": RULE_ID,
        "m_sweep": M_SWEEP,
        "separation_threshold": SEPARATION_THRESHOLD,
        "beam_acc_floor": BEAM_ACC_FLOOR,
        "count_table_corpus_path": str(gbp.CORPUS),
        "count_table_n_grids": n_grids,
        "git_commit_hash": git_commit_hash(),
        "total_runtime_seconds": time.monotonic() - run_started,
    }
    report = {
        "provenance": provenance,
        "r_histogram": r_histogram,
        "per_m": per_m,
        "verdict": verdict,
        "cell_runs": cells,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print_scoreboard(provenance, r_histogram, per_m, verdict)
    log(f"\nReport written to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
