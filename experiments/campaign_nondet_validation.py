"""Non-deterministic beam validation pilot (Slice 5).

Pre-registration: /home/allans/code/PIL_NONDET_VALIDATION_PREREG.md (SIGNED 2026-07-13).
All thresholds / M-sweep / strata / prune_bite requirement are FIXED by that prereg.

Question: on GUESS-REQUIRING sudoku, does the M-step beam commit gold at BRANCH cells
better than the M=1 det_rank baseline, and does that improvement track refutation-depth r?
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent  # -> pil/
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

import campaign_gate_b_pilot as gbp  # noqa: E402
import campaign_sudoku_forced_move as osc  # noqa: E402
from beam_engine import beam_decode  # noqa: E402
from sudoku_oracle import SudokuOracle  # noqa: E402

OUTPUT = REPO / "data" / "nondet_validation_pilot.json"

# Registered constants -- do not tune (THIS prereg; not gbp's older M_SWEEP).
BEAM_WIDTH = 8
RULE_ID = "decide_energy_v1"
WYLY_SEED = 0
M_SWEEP = [1, 2, 3, 4, 5, 6, 7, 8]
SEPARATION_THRESHOLD = 0.15
BEAM_ACC_FLOOR = 0.50

GEN_SEED = 0
N_HOLES = 45  # 36 clues remain -- SIGNED
BRANCH_CELL_TARGET = 1500
ATTEMPT_CAP = 20_000


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Generation (adapt gen_sudoku: full-solve + 45-hole, keep stall + unique)
# ---------------------------------------------------------------------------


def _full_solve(rng: random.Random) -> list[list[int]]:
    """Backtracking full-solve with rng.shuffle per cell (row-major), mirroring gen_sudoku."""

    grid = [[0] * 9 for _ in range(9)]
    cells = [(r, c) for r in range(9) for c in range(9)]

    def solve(cells_left: list[tuple[int, int]]) -> bool:
        if not cells_left:
            return True
        r, c = cells_left[0]
        vals = list(range(1, 10))
        rng.shuffle(vals)
        for v in vals:
            if (
                all(grid[r][j] != v for j in range(9))
                and all(grid[i][c] != v for i in range(9))
                and all(
                    grid[3 * (r // 3) + i][3 * (c // 3) + j] != v
                    for i in range(3)
                    for j in range(3)
                )
            ):
                grid[r][c] = v
                if solve(cells_left[1:]):
                    return True
                grid[r][c] = 0
        return False

    assert solve(cells)
    return grid


def prop_to_fixpoint(board: list[list[int]]) -> list[list[int]] | None:
    """Apply synchronous naked+hidden-single passes to fixpoint.

    Returns the fixpoint board, or None on contradiction.
    """
    g = [row[:] for row in board]
    while True:
        forced, contradiction = gbp.propagate_one_pass(g)
        if contradiction:
            return None
        if not forced:
            return g
        for r, c, v in forced:
            g[r][c] = v


def count_solutions(board: list[list[int]], limit: int = 2) -> int:
    """Backtracking solution counter; stops once `limit` solutions are found."""
    g = [row[:] for row in board]

    def bt() -> int:
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    n = 0
                    for v in sorted(osc.cell_candidates(g, r, c)):
                        g[r][c] = v
                        n += bt()
                        g[r][c] = 0
                        if n >= limit:
                            return n
                    return n
        return 1

    return bt()


def is_fully_filled(board: list[list[int]]) -> bool:
    return all(board[r][c] != 0 for r in range(9) for c in range(9))


def branch_cells(clues: list[list[int]]) -> list[tuple[int, int]]:
    """Empty cells with propagation_depth is None (SIGNED branch-cell definition)."""
    out: list[tuple[int, int]] = []
    for r in range(9):
        for c in range(9):
            if clues[r][c] == 0 and gbp.propagation_depth(clues, r, c) is None:
                out.append((r, c))
    return out


def generate_guess_requiring(
    seed: int = GEN_SEED,
    branch_target: int = BRANCH_CELL_TARGET,
    attempt_cap: int = ATTEMPT_CAP,
) -> dict[str, Any]:
    """Generate unique-solution, propagation-stalling puzzles until ~branch_target cells."""
    rng = random.Random(seed)
    n_attempts = 0
    n_stall = 0
    kept: list[dict[str, Any]] = []
    n_branch_cells = 0

    while n_branch_cells < branch_target and n_attempts < attempt_cap:
        n_attempts += 1
        full = _full_solve(rng)
        clues = [row[:] for row in full]
        holes = rng.sample(range(81), N_HOLES)
        for h in holes:
            clues[h // 9][h % 9] = 0

        # STALL CHECK: prop to fixpoint; discard if fully filled (propagation-solvable)
        # or contradiction (defensive).
        fix = prop_to_fixpoint(clues)
        if fix is None:
            continue
        if is_fully_filled(fix):
            continue
        n_stall += 1

        # UNIQUENESS: exactly one solution
        if count_solutions(clues, limit=2) != 1:
            continue

        bcells = branch_cells(clues)
        if not bcells:
            # Stall with empties but every empty is somehow forced? Should not happen.
            continue

        kept.append(
            {
                "clues": clues,
                "solution": full,
                "branch_cells": bcells,
            }
        )
        n_branch_cells += len(bcells)

    stall_rate = (n_stall / n_attempts) if n_attempts else 0.0
    return {
        "seed": seed,
        "n_attempts": n_attempts,
        "n_stall": n_stall,
        "stall_rate": stall_rate,
        "n_kept": len(kept),
        "n_branch_cells": n_branch_cells,
        "puzzles": kept,
    }


# ---------------------------------------------------------------------------
# Ground truth — refutation-depth r (SIGNED algorithm, exact)
# ---------------------------------------------------------------------------


def solve_with_depths(
    board: list[list[int]], depth: int = 0
) -> tuple[list[list[int]], dict[tuple[int, int], int]] | None:
    """Canonical min-candidate branch-and-prune solve with per-cell depths.

    Returns (solved_board, local_depths) on success, else None.
    local_depths: {(r,c): depth} for every cell filled by THIS call's own
    propagation-to-fixpoint (all at `depth`) plus everything filled deeper by the
    recursive call that succeeded.
    """
    board = [row[:] for row in board]
    local: dict[tuple[int, int], int] = {}
    while True:
        forced, contradiction = gbp.propagate_one_pass(board)
        if contradiction:
            return None
        if not forced:
            break
        for r, c, v in forced:
            board[r][c] = v
            local[(r, c)] = depth
    empties = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
    if not empties:
        return board, local
    pivot = min(empties, key=lambda rc: (len(osc.cell_candidates(board, *rc)), rc))
    pr, pc = pivot
    for v in sorted(osc.cell_candidates(board, pr, pc)):
        trial = [row[:] for row in board]
        trial[pr][pc] = v
        result = solve_with_depths(trial, depth + 1)
        if result is not None:
            solved_board, deeper = result
            merged = dict(local)
            merged[(pr, pc)] = depth + 1
            merged.update(deeper)
            return solved_board, merged
    return None  # every candidate at this pivot failed -> caller backtracks


def compute_r_for_puzzle(
    clues: list[list[int]], bcells: list[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    """r(cell) for every branch cell via one solve_with_depths call."""
    result = solve_with_depths(clues, 0)
    assert result is not None, "unique stall puzzle must be solvable"
    _solved, depths = result
    return {rc: depths[rc] for rc in bcells}


# ---------------------------------------------------------------------------
# Metric tables + verdict
# ---------------------------------------------------------------------------


def run_decodes(
    puzzles: list[dict[str, Any]],
    count_table: list[list[list[int]]],
) -> list[dict[str, Any]]:
    """For every (puzzle, branch_cell, M): one fresh beam_decode. Returns flat records."""
    records: list[dict[str, Any]] = []
    n_total = sum(len(p["branch_cells"]) for p in puzzles) * len(M_SWEEP)
    done = 0
    t0 = time.time()
    for pi, puz in enumerate(puzzles):
        clues = puz["clues"]
        sol = puz["solution"]
        r_map = puz["r_map"]
        for r, c in puz["branch_cells"]:
            gold = int(sol[r][c])
            cell_r = int(r_map[(r, c)])
            for M in M_SWEEP:
                res = beam_decode(
                    SudokuOracle(clues, r, c, count_table, WYLY_SEED),
                    M,
                    BEAM_WIDTH,
                    RULE_ID,
                    WYLY_SEED,
                )
                commit = res["committed_value"]
                records.append(
                    {
                        "puzzle_idx": pi,
                        "row": r,
                        "col": c,
                        "r": cell_r,
                        "M": M,
                        "gold": gold,
                        "commit": commit,
                        "correct": commit == gold,
                        "prune_events": int(res["prune_events"]),
                        "pruned_this_decision": int(res["prune_events"]) > 0,
                    }
                )
                done += 1
                if done % 500 == 0 or done == n_total:
                    elapsed = time.time() - t0
                    log(
                        f"  decode progress {done}/{n_total} "
                        f"({100.0 * done / n_total:.1f}%) elapsed={elapsed:.1f}s"
                    )
    return records


def build_tables(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Core per-(r,M) table + per-M stratum / diagonal / horizon aggregates."""
    # Index: (cell_key) -> {M: record} for conf_acc (needs M=1 column on same cells)
    # cell_key = (puzzle_idx, row, col)
    by_cell: dict[tuple[int, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        key = (rec["puzzle_idx"], rec["row"], rec["col"])
        by_cell[key][rec["M"]] = rec

    # Unique cells with their r
    cells: list[tuple[tuple[int, int, int], int]] = []
    for key, m_map in by_cell.items():
        r_val = m_map[1]["r"]  # r is independent of M
        cells.append((key, r_val))

    r_values = sorted({r for _, r in cells})
    r_histogram = Counter(r for _, r in cells)

    # Core: for each (r_val, M)
    core: dict[str, dict[str, Any]] = {}
    for r_val in r_values:
        subset_keys = [k for k, rv in cells if rv == r_val]
        for M in M_SWEEP:
            n = len(subset_keys)
            if n == 0:
                acc = None
                prune_bite = None
            else:
                correct = sum(1 for k in subset_keys if by_cell[k][M]["correct"])
                pruned = sum(
                    1 for k in subset_keys if by_cell[k][M]["pruned_this_decision"]
                )
                acc = correct / n
                prune_bite = pruned / n
            core[f"r={r_val},M={M}"] = {
                "r": r_val,
                "M": M,
                "n": n,
                "acc": acc,
                "prune_bite": prune_bite,
            }

    def stratum_keys(M: int, mode: str) -> list[tuple[int, int, int]]:
        if mode == "signal":
            return [k for k, rv in cells if 1 <= rv <= M]
        if mode == "diagonal":
            return [k for k, rv in cells if rv == M]
        if mode == "horizon":
            return [k for k, rv in cells if rv > M]
        raise ValueError(mode)

    per_m: dict[str, Any] = {}
    diagonal: dict[str, Any] = {}
    horizon: dict[str, Any] = {}
    population: dict[str, Any] = {}
    n_cells = len(cells)
    all_keys = [k for k, _ in cells]

    for M in M_SWEEP:
        sig = stratum_keys(M, "signal")
        n_sig = len(sig)
        if n_sig == 0:
            beam_acc = None
            conf_acc = None
            separation = None
            prune_bite = None
        else:
            beam_correct = sum(1 for k in sig if by_cell[k][M]["correct"])
            conf_correct = sum(1 for k in sig if by_cell[k][1]["correct"])
            pruned = sum(1 for k in sig if by_cell[k][M]["pruned_this_decision"])
            beam_acc = beam_correct / n_sig
            conf_acc = conf_correct / n_sig
            separation = beam_acc - conf_acc
            prune_bite = pruned / n_sig
        per_m[str(M)] = {
            "M": M,
            "n": n_sig,
            "beam_acc": beam_acc,
            "conf_acc": conf_acc,
            "SEPARATION": separation,
            "prune_bite": prune_bite,
        }

        diag_keys = stratum_keys(M, "diagonal")
        n_diag = len(diag_keys)
        if n_diag == 0:
            diag_acc = None
        else:
            diag_acc = sum(1 for k in diag_keys if by_cell[k][M]["correct"]) / n_diag
        diagonal[str(M)] = {"M": M, "r": M, "n": n_diag, "acc": diag_acc}

        hor_keys = stratum_keys(M, "horizon")
        n_hor = len(hor_keys)
        if n_hor == 0:
            hor_acc = None
        else:
            hor_acc = sum(1 for k in hor_keys if by_cell[k][M]["correct"]) / n_hor
        horizon[str(M)] = {"M": M, "n": n_hor, "beam_acc": hor_acc}

        # Full fixed branch-cell population (no r filter): fair across M.
        pop_correct = sum(1 for k in all_keys if by_cell[k][M]["correct"])
        population[str(M)] = {
            "M": M,
            "n": n_cells,
            "beam_acc": pop_correct / n_cells if n_cells > 0 else None,
        }

    return {
        "r_histogram": {str(k): int(v) for k, v in sorted(r_histogram.items())},
        "r_values": r_values,
        "core": core,
        "per_m": per_m,
        "diagonal": diagonal,
        "horizon": horizon,
        "population": population,
        "n_cells": n_cells,
    }


def population_nondecreasing_up_to(population: dict[str, Any], up_to_M: int) -> bool:
    """population beam_acc(M) non-decreasing for M=1..up_to_M (ties allowed)."""
    seq: list[float] = []
    for M in M_SWEEP:
        if M > up_to_M:
            break
        seq.append(float(population[str(M)]["beam_acc"]))
    for i in range(1, len(seq)):
        if seq[i] + 1e-15 < seq[i - 1]:
            return False
    return True


def decide_verdict(tables: dict[str, Any]) -> dict[str, Any]:
    per_m = tables["per_m"]
    population = tables["population"]

    any_prune = any(
        (per_m[str(M)]["prune_bite"] or 0) > 0 for M in M_SWEEP
    )
    if not any_prune:
        return {
            "verdict": "invalid",
            "deciding_M": None,
            "deciding": None,
            "explanation": (
                "INVALID: prune_bite(M)==0 for every M in M_SWEEP — the generated "
                "puzzles never forced the beam to branch-and-prune. Re-generate harder; "
                "not a ranker failure."
            ),
        }

    # FIRES candidates
    firing_Ms: list[int] = []
    for M in M_SWEEP:
        row = per_m[str(M)]
        sep = row["SEPARATION"]
        bacc = row["beam_acc"]
        pb = row["prune_bite"]
        if sep is None or bacc is None or pb is None:
            continue
        if (
            sep >= SEPARATION_THRESHOLD
            and bacc >= BEAM_ACC_FLOOR
            and pb > 0
            and population_nondecreasing_up_to(population, M)
        ):
            firing_Ms.append(M)

    if firing_Ms:
        M_star = firing_Ms[0]
        row = per_m[str(M_star)]
        return {
            "verdict": "fires",
            "deciding_M": M_star,
            "deciding": {
                "M": M_star,
                "SEPARATION": row["SEPARATION"],
                "beam_acc": row["beam_acc"],
                "conf_acc": row["conf_acc"],
                "prune_bite": row["prune_bite"],
                "n": row["n"],
            },
            "firing_Ms": firing_Ms,
            "explanation": (
                f"FIRES: at M={M_star} on r in [1,{M_star}], "
                f"SEPARATION={row['SEPARATION']:.6f} >= {SEPARATION_THRESHOLD}, "
                f"beam_acc={row['beam_acc']:.6f} >= {BEAM_ACC_FLOOR}, "
                f"prune_bite={row['prune_bite']:.6f} > 0, and population beam_acc(M) "
                f"is monotonically non-decreasing across M=1..{M_star}."
            ),
        }

    # FAILS: every M with prune_bite>0 has SEPARATION < 0.15, OR beam_acc never reaches 0.50
    prune_Ms = [
        M
        for M in M_SWEEP
        if (per_m[str(M)]["prune_bite"] or 0) > 0
    ]
    all_sep_low = all(
        (per_m[str(M)]["SEPARATION"] is not None)
        and (per_m[str(M)]["SEPARATION"] < SEPARATION_THRESHOLD)
        for M in prune_Ms
    )
    never_floor = all(
        (per_m[str(M)]["beam_acc"] is None)
        or (per_m[str(M)]["beam_acc"] < BEAM_ACC_FLOOR)
        for M in M_SWEEP
    )
    if all_sep_low or never_floor:
        return {
            "verdict": "fails",
            "deciding_M": None,
            "deciding": {
                "prune_Ms": prune_Ms,
                "all_sep_low": all_sep_low,
                "never_floor": never_floor,
                "per_m_summary": {
                    str(M): {
                        "SEPARATION": per_m[str(M)]["SEPARATION"],
                        "beam_acc": per_m[str(M)]["beam_acc"],
                        "prune_bite": per_m[str(M)]["prune_bite"],
                    }
                    for M in M_SWEEP
                },
            },
            "explanation": (
                f"FAILS: registered negative. all_sep_low_on_prune_Ms={all_sep_low}, "
                f"beam_acc_never_reaches_{BEAM_ACC_FLOOR}={never_floor}. "
                f"prune_Ms={prune_Ms}."
            ),
        }

    # Neither strict FAILS (sep/floor) nor FIRES (population signature blocked, etc.).
    # Report fails (only fires/fails/invalid are registered outcomes) with the
    # best near-miss M that cleared sep+floor+prune_bite if any.
    near: list[int] = []
    for M in M_SWEEP:
        row = per_m[str(M)]
        sep, bacc, pb = row["SEPARATION"], row["beam_acc"], row["prune_bite"]
        if (
            sep is not None
            and bacc is not None
            and pb is not None
            and sep >= SEPARATION_THRESHOLD
            and bacc >= BEAM_ACC_FLOOR
            and pb > 0
        ):
            near.append(M)
    if near:
        M_star = near[0]
        row = per_m[str(M_star)]
        return {
            "verdict": "fails",
            "deciding_M": M_star,
            "deciding": {
                "M": M_star,
                "SEPARATION": row["SEPARATION"],
                "beam_acc": row["beam_acc"],
                "conf_acc": row["conf_acc"],
                "prune_bite": row["prune_bite"],
                "n": row["n"],
                "blocked_by": "population_not_nondecreasing",
                "near_miss_Ms": near,
            },
            "explanation": (
                f"FAILS: at M={M_star} SEPARATION={row['SEPARATION']:.6f} >= "
                f"{SEPARATION_THRESHOLD}, beam_acc={row['beam_acc']:.6f} >= "
                f"{BEAM_ACC_FLOOR}, prune_bite={row['prune_bite']:.6f} > 0, but "
                f"population beam_acc(M) is not monotonically non-decreasing "
                f"across M=1..{M_star} (rising signature fails). near_miss_Ms={near}."
            ),
        }
    return {
        "verdict": "fails",
        "deciding_M": None,
        "deciding": {
            "note": "partial conditions only; FIRES criteria not jointly met",
            "per_m_summary": {
                str(M): {
                    "SEPARATION": per_m[str(M)]["SEPARATION"],
                    "beam_acc": per_m[str(M)]["beam_acc"],
                    "prune_bite": per_m[str(M)]["prune_bite"],
                }
                for M in M_SWEEP
            },
        },
        "explanation": (
            "FAILS (no M jointly meets SEPARATION, beam_acc floor, prune_bite>0, "
            "and population beam_acc(M) monotonically non-decreasing)."
        ),
    }


def print_scoreboard(
    gen: dict[str, Any],
    tables: dict[str, Any],
    verdict: dict[str, Any],
) -> None:
    log("=" * 72)
    log("NONDET VALIDATION PILOT -- beam-energy vs M=1 at BRANCH cells")
    log("=" * 72)
    log(
        f"Registered: BEAM_WIDTH={BEAM_WIDTH} M_SWEEP={M_SWEEP} "
        f"DELTA={SEPARATION_THRESHOLD} FLOOR={BEAM_ACC_FLOOR} "
        f"RULE_ID={RULE_ID} WYLY_SEED={WYLY_SEED}"
    )
    log()
    log("--- generation ---")
    log(f"  seed={gen['seed']} n_attempts={gen['n_attempts']} n_stall={gen['n_stall']}")
    log(f"  stall_rate={gen['stall_rate']:.6f}")
    log(f"  n_kept={gen['n_kept']} n_branch_cells={gen['n_branch_cells']}")
    log()
    log("--- r histogram (branch cells) ---")
    for k, v in tables["r_histogram"].items():
        log(f"  r={k}: {v}")
    log()
    log("--- core table acc(r, M) ---")
    r_values = tables["r_values"]
    header = "r\\M".ljust(6) + "".join(f"{M:>10}" for M in M_SWEEP)
    log(header)
    for r_val in r_values:
        row = f"{r_val:<6}"
        for M in M_SWEEP:
            cell = tables["core"][f"r={r_val},M={M}"]
            if cell["acc"] is None:
                row += f"{'—':>10}"
            else:
                row += f"{cell['acc']:>10.4f}"
        log(row)
    log()
    log("--- core prune_bite(r, M) ---")
    log(header)
    for r_val in r_values:
        row = f"{r_val:<6}"
        for M in M_SWEEP:
            cell = tables["core"][f"r={r_val},M={M}"]
            if cell["prune_bite"] is None:
                row += f"{'—':>10}"
            else:
                row += f"{cell['prune_bite']:>10.4f}"
        log(row)
    log()
    log("--- per-M stratum r in [1, M] ---")
    log(
        f"{'M':>3} {'n':>6} {'beam_acc':>10} {'conf_acc':>10} "
        f"{'SEPARATION':>12} {'prune_bite':>12}"
    )
    for M in M_SWEEP:
        row = tables["per_m"][str(M)]
        def fmt(x: Any) -> str:
            return "—" if x is None else f"{x:.6f}"

        log(
            f"{M:>3} {row['n']:>6} {fmt(row['beam_acc']):>10} "
            f"{fmt(row['conf_acc']):>10} {fmt(row['SEPARATION']):>12} "
            f"{fmt(row['prune_bite']):>12}"
        )
    log()
    log("--- diagonal acc(r=M, M) (informational; not used in verdict) ---")
    for M in M_SWEEP:
        d = tables["diagonal"][str(M)]
        acc_s = "—" if d["acc"] is None else f"{d['acc']:.6f}"
        log(f"  M={M}: n={d['n']} acc={acc_s}")
    log()
    log("--- population beam_acc(M) (full fixed branch-cell set; rising signature) ---")
    for M in M_SWEEP:
        p = tables["population"][str(M)]
        acc_s = "—" if p["beam_acc"] is None else f"{p['beam_acc']:.6f}"
        log(f"  M={M}: n={p['n']} beam_acc={acc_s}")
    log()
    log("--- horizon r > M beam_acc ---")
    for M in M_SWEEP:
        h = tables["horizon"][str(M)]
        acc_s = "—" if h["beam_acc"] is None else f"{h['beam_acc']:.6f}"
        log(f"  M={M}: n={h['n']} beam_acc={acc_s}")
    log()
    log("--- verdict ---")
    log(f"  {verdict['verdict'].upper()}")
    log(f"  {verdict['explanation']}")
    if verdict.get("deciding") and verdict["verdict"] == "fires":
        d = verdict["deciding"]
        log(
            f"  deciding M={d['M']}: SEPARATION={d['SEPARATION']:.6f} "
            f"beam_acc={d['beam_acc']:.6f} conf_acc={d['conf_acc']:.6f} "
            f"prune_bite={d['prune_bite']:.6f} n={d['n']}"
        )


def main() -> int:
    t0 = time.time()
    log("=" * 72)
    log("NONDET VALIDATION PILOT")
    log("=" * 72)
    log(
        f"Registered: BEAM_WIDTH={BEAM_WIDTH} M_SWEEP={M_SWEEP} "
        f"DELTA={SEPARATION_THRESHOLD} FLOOR={BEAM_ACC_FLOOR}"
    )
    log()

    log("--- building count table from corpus ---")
    count_table, n_grids = gbp.build_sudoku_count_table()
    log(f"CountTable: n_grids={n_grids}")
    log()

    log(
        f"--- generating guess-requiring puzzles "
        f"(seed={GEN_SEED}, holes={N_HOLES}, target_branch_cells~{BRANCH_CELL_TARGET}) ---"
    )
    gen = generate_guess_requiring()
    log(
        f"  n_attempts={gen['n_attempts']} n_stall={gen['n_stall']} "
        f"stall_rate={gen['stall_rate']:.6f}"
    )
    log(f"  n_kept={gen['n_kept']} n_branch_cells={gen['n_branch_cells']}")
    log()

    log("--- computing refutation-depth r per branch cell ---")
    for puz in gen["puzzles"]:
        puz["r_map"] = compute_r_for_puzzle(puz["clues"], puz["branch_cells"])
    r_hist: Counter[int] = Counter()
    for puz in gen["puzzles"]:
        for rc in puz["branch_cells"]:
            r_hist[puz["r_map"][rc]] += 1
    log(f"  r_histogram={dict(sorted(r_hist.items()))}")
    log()

    log(f"--- running beam_decode for {gen['n_branch_cells']} cells x {len(M_SWEEP)} M ---")
    records = run_decodes(gen["puzzles"], count_table)
    log(f"  done: {len(records)} decodes")
    log()

    tables = build_tables(records)
    verdict = decide_verdict(tables)

    print_scoreboard(gen, tables, verdict)

    payload = {
        "generation": {
            "seed": gen["seed"],
            "n_attempts": gen["n_attempts"],
            "n_stall": gen["n_stall"],
            "stall_rate": gen["stall_rate"],
            "n_kept": gen["n_kept"],
            "n_branch_cells": gen["n_branch_cells"],
            "n_holes": N_HOLES,
            "branch_cell_target": BRANCH_CELL_TARGET,
            "attempt_cap": ATTEMPT_CAP,
        },
        "r_histogram": tables["r_histogram"],
        "core": tables["core"],
        "per_m": tables["per_m"],
        "diagonal": tables["diagonal"],
        "horizon": tables["horizon"],
        "population": tables["population"],
        "beam_width": BEAM_WIDTH,
        "rule_id": RULE_ID,
        "wyly_seed": WYLY_SEED,
        "M_SWEEP": M_SWEEP,
        "SEPARATION_THRESHOLD": SEPARATION_THRESHOLD,
        "BEAM_ACC_FLOOR": BEAM_ACC_FLOOR,
        "git_commit_hash": gbp.git_commit_hash(),
        "count_table_n_grids": n_grids,
        "verdict": verdict["verdict"],
        "deciding_M": verdict.get("deciding_M"),
        "deciding": verdict.get("deciding"),
        "explanation": verdict.get("explanation"),
        "elapsed_sec": time.time() - t0,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    log()
    log(f"wrote {OUTPUT}")
    log(f"elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
