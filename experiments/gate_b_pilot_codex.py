"""Independent Gate (b) pilot: propagation-aligned beam with pinned PIC margins.

The registered length-robust ordering is important.  For two synthetic paths,
``margins=(1, 1)`` has zero guesses, while ``margins=(1, 0, 1, 1, 1)`` has one.
The first path wins because ``-n_guesses`` is compared before the raw sorted margin
tuple; an all-forced path therefore beats a guessed path regardless of path length.

The constrained arm uses one synchronous whole-board naked/hidden-single primitive
for both certified propagation depth and beam progress.  When propagation stalls,
the pivot is selected deterministically by minimum candidate count and then row-major
position.  That pivot rule is a selection rule, not part of the registered energy.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import cmp_to_key, total_ordering
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"
OUTPUT = REPO / "data" / "gate_b_pilot_codex.json"
SUDOKU_DATA = REPO / "data" / "wyly_nexttoken_sudoku_L256.pt"
WIKITEXT_DATA = REPO / "data" / "wyly_nexttoken_wikitext_L256.pt"

M_SWEEP = (1, 2, 3, 4, 5)
BEAM_WIDTH = 8
SEPARATION_BAR = 0.15
BEAM_ACCURACY_FLOOR = 0.50
N_SCAN_MAX = 5000
UNCONSTRAINED_LIMIT = 256
CONFIDENCE_INTEGER_SCALE = 1_000_000

WYLY_ENV: dict[str, str] = {
    "WYLY_TAG": "pythia70m",
    "WYLY_DS": "sudoku",
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
    "WYLY_LABELS": "corpus",
}

_RUNTIME: tuple[Any, Any, Any] | None = None


def _sudoku_runtime() -> tuple[Any, Any, Any]:
    """Lazily import dataset-aware modules so pure unit tests do not load datasets."""
    global _RUNTIME
    if _RUNTIME is None:
        for key, value in WYLY_ENV.items():
            os.environ.setdefault(key, value)
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(EXPERIMENTS))
        osc = importlib.import_module("campaign_sudoku_forced_move")
        uniform = importlib.import_module("campaign_uniform_reveal")
        v5 = importlib.import_module("wyly_lm_v5")
        _RUNTIME = osc, uniform, v5
    return _RUNTIME


def propagate_pass(
    board: list[list[int]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], frozenset[int]], bool]:
    """Run one synchronous whole-board naked+hidden-single propagation pass.

    The input board is never mutated.  Candidate and forced-cell predicates are the
    reusable sudoku primitives from ``campaign_sudoku_forced_move``.
    """
    osc, _, _ = _sudoku_runtime()
    candidates: dict[tuple[int, int], frozenset[int]] = {}
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                candidates[(r, c)] = frozenset(osc.cell_candidates(board, r, c))
    contradiction = any(not values for values in candidates.values())
    if contradiction:
        return {}, candidates, True

    forced: dict[tuple[int, int], int] = {}
    for (r, c), values in candidates.items():
        if len(values) == 1 and osc.is_naked_single(board, r, c):
            forced[(r, c)] = next(iter(values))
            continue
        for value in sorted(values):
            if osc.is_hidden_single(board, r, c, value):
                forced[(r, c)] = value
                break
    return forced, candidates, False


def propagation_depth(board: list[list[int]], cell0: tuple[int, int]) -> int | None:
    """Return the 1-based pass where ``cell0`` is first forced, or ``None`` at fixpoint."""
    work = [row[:] for row in board]
    work[cell0[0]][cell0[1]] = 0
    for pass_index in range(1, 82):
        forced, _, contradiction = propagate_pass(work)
        if contradiction:
            return None
        if cell0 in forced:
            return pass_index
        if not forced:
            return None
        for (r, c), value in forced.items():
            if (r, c) != cell0:
                work[r][c] = value
    return None


def compare_det_rank(count_a: int, denom_a: int, count_b: int, denom_b: int) -> int:
    """Compare two rational determinism ranks by integer cross multiplication only."""
    left = int(count_a) * int(denom_b)
    right = int(count_b) * int(denom_a)
    return (left > right) - (left < right)


@total_ordering
@dataclass(frozen=True)
class DetRank:
    """A ratio whose ordering never performs floating-point division."""

    count: int
    total: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetRank):
            return NotImplemented
        return compare_det_rank(self.count, self.total, other.count, other.total) == 0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, DetRank):
            return NotImplemented
        return compare_det_rank(self.count, self.total, other.count, other.total) < 0


Assignment = tuple[int, int, int, str]


@dataclass
class PathState:
    board: list[list[int]]
    margins: tuple[int, ...] = ()
    assignments: tuple[Assignment, ...] = ()
    branch_ranks: tuple[DetRank, ...] = ()
    seed: int = 0


def det_rank_tuple(path: PathState) -> tuple[DetRank, ...]:
    return path.branch_ranks


def path_quality_key(path: PathState) -> tuple[Any, ...]:
    """Registered best-first key: guess count, margins, det-rank, deterministic tie."""
    n_guesses = path.margins.count(0)
    rule_ids = tuple(assignment[3] for assignment in path.assignments)
    assignment_tuple = tuple(path.assignments)
    return (
        -n_guesses,
        tuple(sorted(path.margins)),
        det_rank_tuple(path),
        (rule_ids, path.seed, assignment_tuple),
    )


def _rank_for(table: dict[str, Any], column: int, value: int) -> DetRank:
    counts = table["counts"]
    count = int(counts[column][value])
    denominators = table.get("denominators")
    total = int(denominators[column][value] if denominators is not None else table["totals"][column])
    return DetRank(count, total)


def _candidate_cmp(
    value_a: int,
    value_b: int,
    *,
    column: int,
    table: dict[str, Any],
    rule_id: str,
    seed: int,
) -> int:
    rank_a = _rank_for(table, column, value_a)
    rank_b = _rank_for(table, column, value_b)
    ranked = compare_det_rank(rank_a.count, rank_a.total, rank_b.count, rank_b.total)
    if ranked:
        return ranked
    tie_a = (rule_id, seed, value_a)
    tie_b = (rule_id, seed, value_b)
    return (tie_a > tie_b) - (tie_a < tie_b)


def rank_candidates(
    values: set[int] | frozenset[int],
    column: int,
    table: dict[str, Any],
    rule_id: str,
    seed: int,
) -> list[int]:
    """Best-first candidate order with rational comparison and registered final tie."""
    cmp = lambda a, b: _candidate_cmp(  # noqa: E731
        a, b, column=column, table=table, rule_id=rule_id, seed=seed
    )
    return sorted(values, key=cmp_to_key(cmp), reverse=True)


def _apply_forced(board: list[list[int]], forced: dict[tuple[int, int], int]) -> list[list[int]]:
    new_board = [row[:] for row in board]
    for (r, c), value in sorted(forced.items()):
        new_board[r][c] = value
    return new_board


def _assignment_source(path: PathState, cell0: tuple[int, int]) -> str | None:
    for r, c, _, rule_id in path.assignments:
        if (r, c) == cell0:
            return "forced" if rule_id == "single" else "branch"
    return None


def decide_energy(
    board: list[list[int]],
    cell0: tuple[int, int],
    M: int,
    table: dict[str, Any],
    beam_width: int = BEAM_WIDTH,
) -> dict[str, Any]:
    """Commit the target value with the registered propagation-aligned beam."""
    if M < 1:
        raise ValueError("M must be at least 1")
    _, _, v5 = _sudoku_runtime()
    initial = [row[:] for row in board]
    initial[cell0[0]][cell0[1]] = 0
    paths = [PathState(initial, seed=v5.WYLY_SEED)]
    surviving_counts: list[int] = []
    prune_fired: list[bool] = []
    pruned_extensions = 0
    attempted_extensions = 0

    for _step in range(1, M + 1):
        next_paths: list[PathState] = []
        step_pruned = False
        for path in paths:
            if path.board[cell0[0]][cell0[1]] != 0:
                next_paths.append(path)
                continue
            forced, candidates, contradiction = propagate_pass(path.board)
            if contradiction:
                attempted_extensions += 1
                pruned_extensions += 1
                step_pruned = True
                continue
            if forced:
                attempted_extensions += 1
                new_board = _apply_forced(path.board, forced)
                _, _, post_contradiction = propagate_pass(new_board)
                if post_contradiction:
                    pruned_extensions += 1
                    step_pruned = True
                    continue
                assignments = tuple((r, c, value, "single") for (r, c), value in sorted(forced.items()))
                next_paths.append(
                    PathState(
                        new_board,
                        path.margins + (1,) * len(forced),
                        path.assignments + assignments,
                        path.branch_ranks,
                        path.seed,
                    )
                )
                continue

            pivot, pivot_values = min(candidates.items(), key=lambda item: (len(item[1]), item[0]))
            if not pivot_values:
                attempted_extensions += 1
                pruned_extensions += 1
                step_pruned = True
                continue
            ranked_values = rank_candidates(
                pivot_values, pivot[1], table, "branch", v5.WYLY_SEED
            )[:beam_width]
            for value in ranked_values:
                attempted_extensions += 1
                new_board = [row[:] for row in path.board]
                new_board[pivot[0]][pivot[1]] = value
                _, _, post_contradiction = propagate_pass(new_board)
                if post_contradiction:
                    pruned_extensions += 1
                    step_pruned = True
                    continue
                next_paths.append(
                    PathState(
                        new_board,
                        path.margins + (0,),
                        path.assignments + ((pivot[0], pivot[1], value, "branch"),),
                        path.branch_ranks + (_rank_for(table, pivot[1], value),),
                        path.seed,
                    )
                )
        if not next_paths:
            paths = []
            surviving_counts.append(0)
            prune_fired.append(step_pruned)
            break
        next_paths.sort(key=path_quality_key, reverse=True)
        paths = next_paths[:beam_width]
        surviving_counts.append(len(paths))
        prune_fired.append(step_pruned)

    result: dict[str, Any] = {
        "commit": None,
        "source": "no_surviving_path",
        "stats": {
            "surviving_path_counts": surviving_counts,
            "prune_fired": prune_fired,
            "pruned_extensions": pruned_extensions,
            "attempted_extensions": attempted_extensions,
        },
    }
    if not paths:
        return result
    best = paths[0]
    if best.board[cell0[0]][cell0[1]] != 0:
        result["commit"] = best.board[cell0[0]][cell0[1]]
        result["source"] = _assignment_source(best, cell0)
        return result
    legal = frozenset(_sudoku_runtime()[0].cell_candidates(best.board, *cell0))
    if legal:
        result["commit"] = rank_candidates(
            legal, cell0[1], table, "det_rank_fallback", v5.WYLY_SEED
        )[0]
        result["source"] = "det_rank_fallback"
    else:
        result["source"] = "no_legal_candidate"
    return result


def _load_sudoku() -> tuple[Any, Any, Any, Any, Any, Any]:
    _, _, v5 = _sudoku_runtime()
    loaded = v5.load_ds()
    return tuple(item.detach().cpu() for item in loaded)  # type: ignore[return-value]


def _raw_gold_digit(
    row_index: int, y: Any, cls: Any, uv: Any, value_map: dict[int, int | None]
) -> int | None:
    raw_token = int(uv[int(cls[int(y[row_index])])])
    value = value_map.get(raw_token)
    return int(value) if value is not None else None


def _reconstruct_row(
    row_index: int,
    ids: Any,
    uv: Any,
    sol_pat: list[int],
    puz_pat: list[int],
    value_map: dict[int, int | None],
) -> dict[str, Any] | None:
    osc, uniform, _ = _sudoku_runtime()
    raw = osc.raw_window_from_compact(ids[row_index], uv)
    return uniform.reconstruct_block_for_target(raw, sol_pat, puz_pat, value_map, 256)


def _build_count_table(
    ids: Any, y: Any, cls: Any, uv: Any, tr: Any, sol_pat: list[int], puz_pat: list[int],
    value_map: dict[int, int | None],
) -> tuple[dict[str, Any], dict[str, int]]:
    counts = [[0] * 10 for _ in range(9)]
    stats = Counter(scanned=0, target_rows=0, board_valid=0, gold_resolved=0)
    for row_index in tr.tolist():
        stats["scanned"] += 1
        rec = _reconstruct_row(row_index, ids, uv, sol_pat, puz_pat, value_map)
        if rec is None:
            continue
        stats["target_rows"] += 1
        if not rec["board_valid"]:
            continue
        stats["board_valid"] += 1
        gold = _raw_gold_digit(row_index, y, cls, uv, value_map)
        if gold is None:
            continue
        stats["gold_resolved"] += 1
        counts[int(rec["c"])][gold] += 1
    totals = [sum(row[1:]) for row in counts]
    return {"counts": counts, "totals": totals}, dict(stats)


def _scan_test_population(
    ids: Any, y: Any, cls: Any, uv: Any, te: Any, sol_pat: list[int], puz_pat: list[int],
    value_map: dict[int, int | None],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = Counter(scanned=0, target_rows=0, board_valid=0, gold_resolved=0, forced_step1=0)
    for row_index in te.tolist():
        stats["scanned"] += 1
        rec = _reconstruct_row(row_index, ids, uv, sol_pat, puz_pat, value_map)
        if rec is None:
            continue
        stats["target_rows"] += 1
        if not rec["board_valid"]:
            continue
        stats["board_valid"] += 1
        gold = _raw_gold_digit(row_index, y, cls, uv, value_map)
        if gold is None:
            continue
        stats["gold_resolved"] += 1
        board = [row[:] for row in rec["board"]]
        cell0 = (int(rec["r"]), int(rec["c"]))
        board[cell0[0]][cell0[1]] = 0
        forced, _, contradiction = propagate_pass(board)
        if contradiction:
            stats["depth_contradiction"] += 1
            continue
        if cell0 in forced:
            stats["forced_step1"] += 1
            continue
        depth = propagation_depth(board, cell0)
        rows.append({"row_index": row_index, "board": board, "cell0": cell0, "gold": gold, "d": depth})
    stats["population"] = len(rows)
    return rows, dict(stats)


def _accuracy(correct: int, count: int) -> float | None:
    return correct / count if count else None


def _aggregate_beam_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [n for result in results for n in result["stats"]["surviving_path_counts"]]
    flags = [flag for result in results for flag in result["stats"]["prune_fired"]]
    pruned = sum(result["stats"]["pruned_extensions"] for result in results)
    attempted = sum(result["stats"]["attempted_extensions"] for result in results)
    return {
        "average_surviving_path_count_per_step": sum(counts) / len(counts) if counts else None,
        "legality_prune_fired_step_fraction": sum(flags) / len(flags) if flags else None,
        "steps": len(flags),
        "pruned_extensions": pruned,
        "attempted_extensions": attempted,
    }


def _depth_key(depth: int | None) -> str:
    return "never" if depth is None else str(depth)


def _run_constrained(table: dict[str, Any], population: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: dict[int, list[dict[str, Any]]] = {}
    for M in M_SWEEP:
        decisions[M] = [decide_energy(row["board"], row["cell0"], M, table) for row in population]

    hist = Counter(_depth_key(row["d"]) for row in population)
    histogram = {
        key: {"count": count, "fraction": count / len(population) if population else 0.0}
        for key, count in sorted(
            hist.items(),
            key=lambda item: (item[0] == "never", int(item[0]) if item[0].isdigit() else 99),
        )
    }
    per_d: dict[str, Any] = {}
    for depth_key, hist_row in histogram.items():
        indices = [i for i, row in enumerate(population) if _depth_key(row["d"]) == depth_key]
        per_d[depth_key] = {"count": hist_row["count"], "fraction": hist_row["fraction"], "by_M": {}}
        for M in M_SWEEP:
            correct = sum(decisions[M][i]["commit"] == population[i]["gold"] for i in indices)
            baseline = sum(decisions[1][i]["commit"] == population[i]["gold"] for i in indices)
            per_d[depth_key]["by_M"][str(M)] = {
                "beam_acc": _accuracy(correct, len(indices)),
                "conf_acc": _accuracy(baseline, len(indices)),
                "SEPARATION": _accuracy(correct - baseline, len(indices)),
            }

    per_m: dict[str, Any] = {}
    qualifying: list[dict[str, Any]] = []
    for M in M_SWEEP:
        signal = [i for i, row in enumerate(population) if row["d"] is not None and 2 <= row["d"] <= M]
        signal_beam = sum(decisions[M][i]["commit"] == population[i]["gold"] for i in signal)
        signal_conf = sum(decisions[1][i]["commit"] == population[i]["gold"] for i in signal)
        beam_acc = _accuracy(signal_beam, len(signal))
        conf_acc = _accuracy(signal_conf, len(signal))
        separation = None if beam_acc is None or conf_acc is None else beam_acc - conf_acc

        diagonal = [i for i, row in enumerate(population) if row["d"] == M]
        diagonal_now = sum(decisions[M][i]["commit"] == population[i]["gold"] for i in diagonal)
        diagonal_prev = sum(decisions[max(1, M - 1)][i]["commit"] == population[i]["gold"] for i in diagonal)
        diagonal_acc = _accuracy(diagonal_now, len(diagonal))
        previous_acc = _accuracy(diagonal_prev, len(diagonal))
        rises = bool(
            M >= 2 and diagonal_acc is not None and previous_acc is not None and diagonal_acc > previous_acc
        )
        conditions = {
            "separation_at_least_0.15": separation is not None and separation >= SEPARATION_BAR,
            "beam_acc_at_least_0.50": beam_acc is not None and beam_acc >= BEAM_ACCURACY_FLOOR,
            "d_equals_M_slice_rises_when_crossed": rises,
        }
        row = {
            "signal_stratum": f"d in [2,{M}]",
            "count": len(signal),
            "beam_acc": beam_acc,
            "conf_acc": conf_acc,
            "SEPARATION": separation,
            "d_equals_M": {
                "d": M,
                "count": len(diagonal),
                "beam_acc_at_M_minus_1": previous_acc if M >= 2 else None,
                "beam_acc_at_M": diagonal_acc,
                "rise": rises,
            },
            "conditions": conditions,
            "beam_statistics": _aggregate_beam_stats(decisions[M]),
        }
        per_m[str(M)] = row
        if all(conditions.values()):
            qualifying.append({"M": M, **row})

    numeric_separation_exists = any(
        row["SEPARATION"] is not None and row["SEPARATION"] >= SEPARATION_BAR
        for row in per_m.values()
    )
    majority_exists = any(
        row["beam_acc"] is not None and row["beam_acc"] >= BEAM_ACCURACY_FLOOR
        for row in per_m.values()
    )
    if qualifying:
        deciding = qualifying[0]
        verdict = "FIRES"
    else:
        candidates = [row for row in ({"M": int(M), **values} for M, values in per_m.items()) if row["count"]]
        deciding = max(
            candidates,
            key=lambda row: (
                row["SEPARATION"] if row["SEPARATION"] is not None else -2.0,
                row["beam_acc"] if row["beam_acc"] is not None else -1.0,
                -row["M"],
            ),
            default=None,
        )
        verdict = "FAILS"

    honest_limit: dict[str, Any] = {}
    for M in M_SWEEP:
        indices = [i for i, row in enumerate(population) if row["d"] is None or row["d"] > M]
        beam_correct = sum(decisions[M][i]["commit"] == population[i]["gold"] for i in indices)
        conf_correct = sum(decisions[1][i]["commit"] == population[i]["gold"] for i in indices)
        honest_limit[str(M)] = {
            "stratum": f"d>{M} or never",
            "count": len(indices),
            "beam_acc": _accuracy(beam_correct, len(indices)),
            "conf_acc": _accuracy(conf_correct, len(indices)),
            "SEPARATION": _accuracy(beam_correct - conf_correct, len(indices)),
        }

    return {
        "status": "COMPLETE",
        "population_count": len(population),
        "d_histogram": histogram,
        "per_d": per_d,
        "per_M": per_m,
        "honest_limit": honest_limit,
        "beam_statistics_overall": _aggregate_beam_stats(
            [result for M in M_SWEEP for result in decisions[M]]
        ),
        "verdict": {
            "value": verdict,
            "separation_condition_exists": numeric_separation_exists,
            "majority_condition_exists": majority_exists,
            "rising_signature_required": True,
            "deciding_pair": deciding,
        },
    }


@dataclass
class UnconstrainedPath:
    context: Any
    margins: tuple[int, ...] = ()
    tokens: tuple[int, ...] = ()
    rule_ids: tuple[str, ...] = ()


def _unconstrained_key(path: UnconstrainedPath, seed: int) -> tuple[Any, ...]:
    return tuple(sorted(path.margins)), path.rule_ids, seed, path.tokens


def _load_frozen_rules(payload: dict[str, Any], rescue: Any, v5: Any) -> list[tuple[str, Any]]:
    rules: list[tuple[str, Any]] = []
    for frozen in payload["rules"]:
        name = frozen["name"]
        if frozen["kind"] == "induction":
            fn = v5.mir_induction_L(frozen["n"])
            confidence = float(frozen["conf"])

            def conf_fn(w: Any, fn: Any = fn, confidence: float = confidence) -> tuple[Any, Any]:
                import torch

                pred = fn(w)
                conf = torch.full((len(w),), confidence, dtype=torch.float64, device=w.device)
                return pred, conf

            rules.append((name, conf_fn))
        else:
            _, conf_fn = rescue._reconstruct_rule(name, frozen["spec"])
            rules.append((name, conf_fn))
    if [name for name, _ in rules] != payload["rule_names"]:
        raise AssertionError("frozen rule order changed during reconstruction")
    return rules


def _fired_candidates(contexts: Any, rules: list[tuple[str, Any]]) -> list[dict[int, tuple[int, str]]]:
    import torch

    candidates: list[dict[int, tuple[int, str]]] = [dict() for _ in range(len(contexts))]
    with torch.no_grad():
        for name, conf_fn in rules:
            pred, confidence = conf_fn(contexts)
            for row in (pred >= 0).nonzero(as_tuple=False).flatten().tolist():
                value = int(pred[row])
                margin = int(round(float(confidence[row]) * CONFIDENCE_INTEGER_SCALE))
                prior = candidates[row].get(value)
                if prior is None or (margin, name) > prior:
                    candidates[row][value] = (margin, name)
    return candidates


def _unconstrained_trajectory(
    context: Any, rules: list[tuple[str, Any]], seed: int
) -> tuple[dict[int, int | None], list[int]]:
    import torch

    paths = [UnconstrainedPath(context.clone())]
    commits: dict[int, int | None] = {}
    surviving: list[int] = []
    for M in M_SWEEP:
        contexts = torch.stack([path.context for path in paths])
        fired = _fired_candidates(contexts, rules)
        next_paths: list[UnconstrainedPath] = []
        for path, row_candidates in zip(paths, fired, strict=True):
            for value, (margin, rule_id) in row_candidates.items():
                new_context = torch.cat(
                    [path.context[1:], torch.tensor([value], dtype=path.context.dtype)]
                )
                next_paths.append(
                    UnconstrainedPath(
                        new_context,
                        path.margins + (margin,),
                        path.tokens + (value,),
                        path.rule_ids + (rule_id,),
                    )
                )
        next_paths.sort(key=lambda path: _unconstrained_key(path, seed), reverse=True)
        paths = next_paths[:BEAM_WIDTH]
        surviving.append(len(paths))
        commits[M] = paths[0].tokens[0] if paths else None
        if not paths:
            for remaining in M_SWEEP[M:]:
                commits[remaining] = None
                surviving.append(0)
            break
    return commits, surviving


def _run_unconstrained() -> dict[str, Any]:
    _, _, v5 = _sudoku_runtime()
    sys.path.insert(0, str(EXPERIMENTS))
    rescue = importlib.import_module("campaign_wikitext_gated_rescue")
    freeze_path = Path(rescue.B0_FREEZE)
    missing = [str(path) for path in (freeze_path, WIKITEXT_DATA) if not path.exists()]
    if missing:
        return {
            "status": "SKIPPED",
            "reason": f"missing required file(s): {', '.join(missing)}",
            "legality_prune": "NONE",
            "rows_found": 0,
        }

    import torch

    payload = torch.load(freeze_path, map_location="cpu", weights_only=False)
    rules = _load_frozen_rules(payload, rescue, v5)
    dataset = torch.load(WIKITEXT_DATA, map_location="cpu", weights_only=False)
    raw_ids = dataset["kept_ids"].detach().cpu()
    labels = dataset["target"].detach().cpu()
    top = torch.bincount(labels).argsort(descending=True)[: v5.V]
    keep = torch.isin(labels, top)
    raw_ids = raw_ids[keep]
    labels = labels[keep]
    n, width = raw_ids.shape
    flat = torch.cat([raw_ids.reshape(-1), labels])
    _, inverse = flat.unique(return_inverse=True)
    ids = inverse[: n * width].reshape(n, width).detach().cpu()
    gold = inverse[n * width :].detach().cpu()

    holdout_start = int(0.85 * n)
    scan_end = min(n, holdout_start + N_SCAN_MAX)
    ambiguous_indices: list[int] = []
    batch_size = 256
    scanned = 0
    for start in range(holdout_start, scan_end, batch_size):
        stop = min(start + batch_size, scan_end)
        batch_candidates = _fired_candidates(ids[start:stop], rules)
        for offset, candidates in enumerate(batch_candidates):
            scanned += 1
            if len(candidates) >= 2:
                ambiguous_indices.append(start + offset)
                if len(ambiguous_indices) >= UNCONSTRAINED_LIMIT:
                    break
        if len(ambiguous_indices) >= UNCONSTRAINED_LIMIT:
            break

    correct = Counter()
    surviving_by_step: list[int] = []
    for row_index in ambiguous_indices:
        commits, surviving = _unconstrained_trajectory(ids[row_index], rules, v5.WYLY_SEED)
        surviving_by_step.extend(surviving)
        target = int(gold[row_index])
        for M in M_SWEEP:
            correct[M] += commits[M] == target
    count = len(ambiguous_indices)
    conf_acc = _accuracy(correct[1], count)
    per_m = {}
    for M in M_SWEEP:
        beam_acc = _accuracy(correct[M], count)
        per_m[str(M)] = {
            "count": count,
            "beam_acc": beam_acc,
            "conf_acc": conf_acc,
            "SEPARATION": None if beam_acc is None or conf_acc is None else beam_acc - conf_acc,
        }
    return {
        "status": "COMPLETE",
        "legality_prune": "NONE",
        "energy": f"fired-rule confidence margins scaled to integers by {CONFIDENCE_INTEGER_SCALE}",
        "candidate_definition": "distinct predictions from fired frozen-cover rules",
        "holdout": "last 15% of filtered corpus, deterministic chronological order",
        "rows_scanned": scanned,
        "rows_found": count,
        "requested_limit": UNCONSTRAINED_LIMIT,
        "scan_limit": N_SCAN_MAX,
        "per_M": per_m,
        "beam_statistics": {
            "average_surviving_path_count_per_step": (
                sum(surviving_by_step) / len(surviving_by_step) if surviving_by_step else None
            ),
            "legality_prune_fired_step_fraction": 0.0,
            "steps": len(surviving_by_step),
        },
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def run_pilot() -> dict[str, Any]:
    """Run both registered arms and return the complete serializable report."""
    started = time.perf_counter()
    if not SUDOKU_DATA.exists():
        raise FileNotFoundError(f"required sudoku dataset is missing: {SUDOKU_DATA}")
    osc, _, v5 = _sudoku_runtime()
    ids, y, cls, uv, tr, te = _load_sudoku()
    codec = v5.load_codec()
    puz_pat, sol_pat = osc.header_patterns(codec)
    value_map = osc.build_value_map(codec)

    table, train_stats = _build_count_table(
        ids, y, cls, uv, tr, sol_pat, puz_pat, value_map
    )
    population, test_stats = _scan_test_population(
        ids, y, cls, uv, te, sol_pat, puz_pat, value_map
    )
    constrained = _run_constrained(table, population)
    unconstrained = _run_unconstrained()
    elapsed = time.perf_counter() - started
    report = {
        "pilot": "Gate (b) independent CODEX lane, correction v2/v3",
        "constrained": constrained,
        "unconstrained": unconstrained,
        "population_scan": {"train": train_stats, "test": test_stats},
        "provenance": {
            "git_commit": _git_commit(),
            "dataset_paths": {"sudoku": str(SUDOKU_DATA), "wikitext": str(WIKITEXT_DATA)},
            "uniform_reveal_parse_function": "campaign_uniform_reveal.reconstruct_block_for_target",
            "det_rank": {
                "key": "P(digit | target column index c)",
                "comparison": "integer cross-multiplication",
                "train_row_count": train_stats["gold_resolved"],
                "raw_count_table": table["counts"],
                "column_totals": table["totals"],
            },
            "beam_width": BEAM_WIDTH,
            "M_sweep": list(M_SWEEP),
            "separation_bar": SEPARATION_BAR,
            "beam_accuracy_floor": BEAM_ACCURACY_FLOOR,
            "WYLY_SEED": v5.WYLY_SEED,
            "no_cover_regeneration": True,
            "unconstrained_bounds": {
                "N_SCAN_MAX": N_SCAN_MAX,
                "UNCONSTRAINED_LIMIT": UNCONSTRAINED_LIMIT,
                "rows_actually_found": unconstrained.get("rows_found", 0),
            },
            "wall_time_seconds": elapsed,
        },
        "verdict": constrained["verdict"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def print_scoreboard(report: dict[str, Any]) -> None:
    constrained = report["constrained"]
    print("GATE (b) PILOT — CODEX INDEPENDENT LANE")
    print("CONSTRAINED ARM (gated)")
    print(f"population={constrained['population_count']}")
    print("M  signal        n     beam     conf      separation  d=M: prev -> now  rises")
    for M, row in constrained["per_M"].items():
        diagonal = row["d_equals_M"]
        print(
            f"{M:>1}  {row['signal_stratum']:<12} {row['count']:>5}  "
            f"{_fmt(row['beam_acc']):>7}  {_fmt(row['conf_acc']):>7}  {_fmt(row['SEPARATION']):>10}  "
            f"n={diagonal['count']:<4} {_fmt(diagonal['beam_acc_at_M_minus_1'])} -> "
            f"{_fmt(diagonal['beam_acc_at_M'])}  {diagonal['rise']}"
        )
    print("d histogram and per-d beam accuracy by M")
    for depth, row in constrained["per_d"].items():
        accuracies = " ".join(f"M{M}={_fmt(values['beam_acc'])}" for M, values in row["by_M"].items())
        print(f"d={depth:<5} n={row['count']:<5} frac={row['fraction']:.4f}  {accuracies}")
    print("honest-limit strata")
    for M, row in constrained["honest_limit"].items():
        print(
            f"M={M} {row['stratum']:<14} n={row['count']:<5} beam={_fmt(row['beam_acc'])} "
            f"conf={_fmt(row['conf_acc'])} sep={_fmt(row['SEPARATION'])}"
        )
    for M, row in constrained["per_M"].items():
        stats = row["beam_statistics"]
        print(
            f"beam-stats M={M}: avg_survivors/step="
            f"{_fmt(stats['average_surviving_path_count_per_step'])} "
            f"prune-fired-step-frac={_fmt(stats['legality_prune_fired_step_fraction'])} "
            f"pruned={stats['pruned_extensions']}/{stats['attempted_extensions']}"
        )

    unconstrained = report["unconstrained"]
    print("UNCONSTRAINED ARM (contrast; not gated)")
    print(
        f"status={unconstrained['status']} legality_prune={unconstrained['legality_prune']} "
        f"rows_scanned={unconstrained.get('rows_scanned', 0)} rows_found={unconstrained.get('rows_found', 0)}"
    )
    if unconstrained["status"] == "COMPLETE":
        print("M  n     beam     conf      separation")
        for M, row in unconstrained["per_M"].items():
            print(
                f"{M:>1}  {row['count']:>4}  {_fmt(row['beam_acc']):>7}  "
                f"{_fmt(row['conf_acc']):>7}  {_fmt(row['SEPARATION']):>10}"
            )
        stats = unconstrained["beam_statistics"]
        print(
            f"beam-stats: avg_survivors/step={_fmt(stats['average_surviving_path_count_per_step'])} "
            f"prune-fired-step-frac={_fmt(stats['legality_prune_fired_step_fraction'])}"
        )
    else:
        print(f"reason={unconstrained['reason']}")

    provenance = report["provenance"]
    print("PROVENANCE")
    print(
        f"git={provenance['git_commit']} beam_width={provenance['beam_width']} "
        f"WYLY_SEED={provenance['WYLY_SEED']} no_cover_regeneration="
        f"{provenance['no_cover_regeneration']} wall={provenance['wall_time_seconds']:.3f}s"
    )
    print(
        f"parse={provenance['uniform_reveal_parse_function']} det_rank_key="
        f"{provenance['det_rank']['key']} train_rows={provenance['det_rank']['train_row_count']}"
    )
    print(f"raw_count_table={provenance['det_rank']['raw_count_table']}")
    print(f"datasets={provenance['dataset_paths']}")
    print(f"unconstrained_bounds={provenance['unconstrained_bounds']}")

    verdict = report["verdict"]
    deciding = verdict["deciding_pair"]
    print("REGISTERED CONDITIONS")
    print(
        f"exists separation>=0.15: {verdict['separation_condition_exists']}; "
        f"exists beam_acc>=0.50: {verdict['majority_condition_exists']}; "
        "rising d==M crossing signature required: True"
    )
    if deciding is not None:
        print(
            f"deciding pair: M={deciding['M']} {deciding['signal_stratum']} n={deciding['count']} "
            f"beam={_fmt(deciding['beam_acc'])} conf={_fmt(deciding['conf_acc'])} "
            f"separation={_fmt(deciding['SEPARATION'])} "
            f"rise={deciding['d_equals_M']['rise']}"
        )
    print(f"VERDICT: {verdict['value']}")


def main() -> None:
    report = run_pilot()
    print_scoreboard(report)


if __name__ == "__main__":
    main()
