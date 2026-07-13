"""Independently verify the last-vs-first SOLUTION-header anchor correction.

This read-mostly cross-check does not read the uniform-reveal implementation or tests and does
not call the existing solution-header location/reconstruction helpers whose behavior it checks.
It implements rightmost-header search, target location, grid extraction, and board validation
from scratch. It deliberately reuses only anchor-agnostic helpers for token metadata, overlaying
the revealed solution prefix, batched legality features, and reveal-third aggregation.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

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
WYLY_ENV_WAS_SET = {key: key in os.environ for key in WYLY_ENV}
for _key, _value in WYLY_ENV.items():
    os.environ.setdefault(_key, _value)

import campaign_legality_pinning as clp  # noqa: E402
import campaign_sudoku_forced_move as osc  # noqa: E402
import wyly_lm_v5 as v5  # noqa: E402

GROK_N = 34_614
GROK_NAKED = (0.551, 0.694, 0.857)
GROK_UNION = (0.736, 0.869, 0.950)


def find_last_subsequence(seq: list[int], pattern: list[int]) -> int:
    """Return the rightmost exact occurrence of pattern in seq, or -1."""
    pattern_len = len(pattern)
    if pattern_len == 0 or pattern_len > len(seq):
        return -1
    for start in range(len(seq) - pattern_len, -1, -1):
        matches = True
        for offset in range(pattern_len):
            if seq[start + offset] != pattern[offset]:
                matches = False
                break
        if matches:
            return start
    return -1


def corrected_target_location(
    token_ids: list[int],
    length: int,
    solution_pattern: list[int],
) -> tuple[int, int, int] | None:
    """Locate the target cell using the rightmost SOLUTION header."""
    sol_at = find_last_subsequence(token_ids, solution_pattern)
    if sol_at < 0:
        return None
    offset = length - (sol_at + len(solution_pattern))
    if not 0 <= offset < 90 or offset % 10 == 9:
        return None
    row, col = divmod(offset, 10)
    return sol_at, row, col


def extract_grid(
    token_ids: list[int],
    body_start: int,
    n_offsets: int,
    value_map: dict[int, int | None],
) -> list[list[int]] | None:
    """Extract cell tokens from a body prefix into a zero-filled 9x9 grid."""
    if body_start < 0 or n_offsets < 0 or body_start + n_offsets > len(token_ids):
        return None
    grid = [[0 for _ in range(9)] for _ in range(9)]
    for offset in range(n_offsets):
        row, col = divmod(offset, 10)
        if col == 9:
            continue
        token_id = token_ids[body_start + offset]
        if token_id not in value_map:
            return None
        value = value_map[token_id]
        grid[row][col] = 0 if value is None else int(value)
    return grid


def board_is_valid(board: list[list[int]]) -> bool:
    """Return whether rows, columns, and boxes contain no duplicate nonzero digit."""
    if len(board) != 9 or any(len(row) != 9 for row in board):
        return False

    def unit_is_valid(values: list[int]) -> bool:
        nonzero = [value for value in values if value != 0]
        return all(1 <= value <= 9 for value in nonzero) and len(nonzero) == len(set(nonzero))

    for row in board:
        if not unit_is_valid(row):
            return False
    for col in range(9):
        if not unit_is_valid([board[row][col] for row in range(9)]):
            return False
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = [
                board[row][col]
                for row in range(box_row, box_row + 3)
                for col in range(box_col, box_col + 3)
            ]
            if not unit_is_valid(box):
                return False
    return True


def main() -> int:
    print("Independent uniform-reveal diagnostic (rightmost SOLUTION header)")
    print("WYLY environment:")
    for key, default in WYLY_ENV.items():
        origin = "pre-existing" if WYLY_ENV_WAS_SET[key] else "setdefault"
        print(f"  {key}={os.environ[key]} ({origin}; requested default={default})")
    print(f"v5.DATA={v5.DATA}")

    codec = v5.load_codec()
    puzzle_pattern, solution_pattern = osc.header_patterns(codec)
    value_map = osc.build_value_map(codec)
    print(f"PUZZLE header pattern={puzzle_pattern}")
    print(f"SOLUTION header pattern={solution_pattern}")

    dataset = torch.load(v5.DATA, map_location="cpu")
    kept_ids = dataset["kept_ids"]
    targets = dataset["target"]
    n_total = int(kept_ids.shape[0])

    counts = Counter()
    clue_counts: Counter[int] = Counter()
    boards: list[list[list[int]]] = []
    rows: list[int] = []
    cols: list[int] = []
    true_values: list[int] = []
    cell_indices: list[int] = []
    not_in_map = object()

    for index in range(n_total):
        token_ids = [int(token_id) for token_id in kept_ids[index].tolist()]
        length = len(token_ids)
        sol_at = find_last_subsequence(token_ids, solution_pattern)
        if sol_at < 0:
            counts["no_anchor"] += 1
            continue

        location = corrected_target_location(token_ids, length, solution_pattern)
        if location is None:
            counts["target_out_of_range"] += 1
            continue
        located_sol_at, row, col = location
        if located_sol_at != sol_at:
            raise RuntimeError("independent anchor searches unexpectedly disagreed")
        offset = row * 10 + col

        gold_token_id = int(targets[index])
        gold_digit = value_map.get(gold_token_id, not_in_map)
        if gold_digit is not_in_map or gold_digit is None:
            counts["gold_not_digit"] += 1
            continue
        true_value = int(gold_digit)

        puzzle_body_start = sol_at - osc.PUZ_BODY
        if puzzle_body_start < 0:
            counts["puzzle_not_visible"] += 1
            continue
        puzzle = extract_grid(token_ids, puzzle_body_start, osc.PUZ_BODY, value_map)
        if puzzle is None:
            counts["puzzle_extraction_failed"] += 1
            continue

        solution_body_start = sol_at + len(solution_pattern)
        solution = extract_grid(token_ids, solution_body_start, offset, value_map)
        if solution is None:
            counts["solution_extraction_failed"] += 1
            continue

        board = osc.board_state_before_cell(puzzle, solution, row, col)
        counts["candidates"] += 1
        clue_counts[sum(value != 0 for puzzle_row in puzzle for value in puzzle_row)] += 1
        if not board_is_valid(board):
            counts["invalid"] += 1
            continue

        boards.append(board)
        rows.append(row)
        cols.append(col)
        true_values.append(true_value)
        cell_indices.append(row * 9 + col)

    n_valid = len(boards)
    print("\nFunnel counts:")
    print(f"  n_total_windows={n_total}")
    print(f"  n_no_anchor={counts['no_anchor']}")
    print(f"  n_target_out_of_range={counts['target_out_of_range']}")
    print(f"  n_gold_not_digit={counts['gold_not_digit']}")
    puzzle_failures = counts["puzzle_not_visible"] + counts["puzzle_extraction_failed"]
    print(f"  n_puzzle_not_visible={counts['puzzle_not_visible']}")
    print(f"  n_puzzle_extraction_failed={counts['puzzle_extraction_failed']}")
    print(f"  n_puzzle_not_visible/extraction_failed={puzzle_failures}")
    print(f"  n_solution_extraction_failed={counts['solution_extraction_failed']}")
    print(f"  n_candidates={counts['candidates']}")
    print(f"  n_invalid={counts['invalid']}")
    print(f"  n_board_valid_classifiable={n_valid}")

    clue_distribution = dict(sorted(clue_counts.items()))
    collapses_to_36 = clue_distribution == {36: counts["candidates"]}
    print("\nCandidate clue-count distribution:")
    print(f"  {clue_distribution}")
    print(f"  collapses_to_{{36: n_candidates}}={collapses_to_36}")

    grids_t = torch.tensor(boards, dtype=torch.long)
    rows_t = torch.tensor(rows, dtype=torch.long)
    cols_t = torch.tensor(cols, dtype=torch.long)
    true_values_t = torch.tensor(true_values, dtype=torch.long)
    features = clp.legality_feature_tensor(grids_t, rows_t, cols_t, true_values_t)
    forced_eq_gold = features["forced_value"] == true_values_t
    strat = clp.stratify_by_reveal_third(
        cell_indices,
        features["is_naked"],
        features["is_hidden"],
        forced_eq_gold,
        error_mask=None,
    )

    print("\nReveal-third distribution (board-valid classifiable rows):")
    for third in (0, 1, 2):
        third_n = int(strat[third]["n_error"])
        percentage = 100.0 * third_n / n_valid if n_valid else 0.0
        print(f"  third {third}: n={third_n}, pct={percentage:.3f}%")
    print("  grok reference: approximately 33% / 33% / 34%")

    print("\nRecovery by reveal third:")
    print("third  my_naked  grok_naked  delta     my_union  grok_union  delta")
    deltas: list[tuple[str, int, float]] = []
    for third in (0, 1, 2):
        my_naked = float(strat[third]["recovered_naked"])
        my_union = float(strat[third]["recovered_union"])
        naked_delta = my_naked - GROK_NAKED[third]
        union_delta = my_union - GROK_UNION[third]
        deltas.extend((("naked", third, naked_delta), ("union", third, union_delta)))
        print(
            f"{third:>5}  {my_naked:>9.3f}  {GROK_NAKED[third]:>10.3f}  {naked_delta:>+7.3f}"
            f"    {my_union:>8.3f}  {GROK_UNION[third]:>10.3f}  {union_delta:>+7.3f}"
        )
    print(f"grok n reference={GROK_N}; my n={n_valid}; n delta={n_valid - GROK_N:+d}")

    offending = [(kind, third, delta) for kind, third, delta in deltas if abs(delta) > 0.01]
    if offending:
        details = ", ".join(f"{kind}[{third}]={delta:+.3f}" for kind, third, delta in offending)
        verdict = "DIFFER"
        print(f"VERDICT: DIFFER (|delta| > 0.01: {details})")
    else:
        verdict = "AGREE"
        print("VERDICT: AGREE (all recovery-rate |delta| <= 0.01)")

    print(f"SUMMARY: final n={n_valid}; verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
