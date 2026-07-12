"""Slice #91: certify and admit the frozen NAKED-ONLY sudoku legality feature.

The certificate proves only #90's naked-single feature for the concrete hand-authored
row∪col∪box scope on the scored windows. It does not certify hidden singles, sudoku
legality universally, or the abstract pluggable scope hook.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# Copied verbatim from #90. v5 binds these at import time.
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
for _k, _v in WYLY_ENV.items():
    os.environ.setdefault(_k, _v)

import campaign_legality_pinning as pin  # noqa: E402
import campaign_sudoku_forced_move as osc  # noqa: E402
import wyly_lm_v5 as v5  # noqa: E402

OUTPUT = REPO / "data" / "legality_certificate.json"
CORPUS = REPO / "data" / "corpus_sudoku.txt"

REVEAL_INDEX_NOTE = (
    "the sudoku dataset presents ONLY late-reveal solution cells (index ~59-80); early/mid cells "
    "never appear as prediction targets, so the register is measured only on the near-solved "
    "sub-task and its capability on early/mid cells is UNTESTED by this data."
)

FRAMING_NOTE = (
    "sudoku-only, LABELS=corpus, no natural-text blending; scope hand-authored; cert covers "
    "the concrete row∪col∪box instance, not the abstract pluggable hook."
)

DESIGN_REGISTERED = """\
## Pre-registered -- constraint-register build 2: certificate + admission (post-#90; consolidation gate \
satisfied)

**Recorded 2026-07-12 before any code/numbers.** Advisor-vetted. Builds on #90's NAKED-ONLY
freeze. Sudoku only, LABELS=corpus; late-reveal-only caveat propagates verbatim; scope
hand-authored (framing); no natural-text blending.

- **PLUGGABLE SCOPE (refactor in now):** the legality feature takes the peer-set/scope
  (row∪col∪box) as a pluggable input. **The certificate covers ONLY the concrete
  row∪col∪box instantiation** -- a future mined scope reuses the form + cert harness but
  MUST earn its own cert run (stated in the doc; pluggability is never "certified for any
  scope").
- **SOUFFLÉ CERTIFICATE (extend wyly_derived_certify.py -- NOT pil/certify.py):** prove the
  tensor legality feature ≡ its Datalog program window-by-window. Compared value = the FULL
  per-row output tuple {n_candidates, is_naked, forced_value} INCLUDING the -1/abstain rows
  (abstain disagreements hide parse bugs). Three-way agreement: **numpy oracle ≡ tensor
  feature ≡ Soufflé**. BAR: 100% on the scored windows = `proved` OVER THAT STATED DOMAIN
  (not "sudoku" universally). Any mismatch -> exact window/row reported, promotion BLOCKED.
- **CORPUS VALIDITY SCAN (registered precondition):** one-time scan that every corpus grid
  is complete and all-different-consistent (valid, uniquely-solvable sudoku). The cert
  proves feature ≡ Datalog; that forced_value = the corpus label additionally requires this
  premise (then naked-single value = solution value by all-different math). conf=1.0 is
  `proved` CONDITIONAL on the scan, which discharges it. Invalid grids -> report, premise fails.
- **CONF PROMOTION (strict, tag discipline):** measured-1.0 -> certified-constant-1.0 ONLY
  IF cert = 100% AND validity scan clean AND proposer regressions = 0. Else conf stays at
  measured fired-accuracy (register ships honest-uncertified).
- **ADMISSION + REGRESSION CHECK:** admit the naked-only forced-move proposer through the
  existing judge on the sudoku cover; report held-out marginal AND regressions (cover-correct
  rows the proposer flips). Pre-named: regressions = 0 expected; ANY nonzero is a
  premise/pipeline flag to RESOLVE (not explain away) before the conf promotion ships.
"""


# Python supplies only linear cell positions and values. All unit recovery is intensional.
SOUFFLE_PROGRAM = r"""
.decl tok(w:number, p:number, t:number)
.input tok
.decl target(w:number, p:number)
.input target

.decl digit(d:number)
digit(1). digit(2). digit(3). digit(4). digit(5).
digit(6). digit(7). digit(8). digit(9).

// Normalize serialized raw grid values: 1..9 are digits; 0 is empty/no val row.
.decl val(w:number, p:number, d:number)
val(W, P, D) :- tok(W, P, T), digit(T), D = T.
.decl empty(w:number, p:number)
empty(W, P) :- tok(W, P, 0).

// Recover coordinates solely from row-major p in 0..80.
.decl cellrc(w:number, p:number, r:number, c:number, b:number)
cellrc(W, P, R, C, B) :- tok(W, P, _), R = P / 9, C = P % 9,
                          BR = R / 3, BC = C / 3, B = BR * 3 + BC.
.decl targetrc(w:number, r:number, c:number, b:number)
targetrc(W, R, C, B) :- target(W, P), cellrc(W, P, R, C, B).

// Treat the target itself as empty, even if its serialized board cell is filled.
.decl filled(w:number, p:number, d:number)
filled(W, P, D) :- val(W, P, D), !target(W, P).
.decl rowused(w:number, r:number, d:number)
rowused(W, R, D) :- filled(W, P, D), cellrc(W, P, R, _, _).
.decl colused(w:number, c:number, d:number)
colused(W, C, D) :- filled(W, P, D), cellrc(W, P, _, C, _).
.decl boxused(w:number, b:number, d:number)
boxused(W, B, D) :- filled(W, P, D), cellrc(W, P, _, _, B).
.decl used(w:number, d:number)
used(W, D) :- targetrc(W, R, _, _), rowused(W, R, D).
used(W, D) :- targetrc(W, _, C, _), colused(W, C, D).
used(W, D) :- targetrc(W, _, _, B), boxused(W, B, D).
.decl candidate(w:number, d:number)
candidate(W, D) :- target(W, _), digit(D), !used(W, D).
.decl candcount(w:number, n:number)
candcount(W, N) :- target(W, _), N = count : { candidate(W, _) }.

.decl feat(w:number, n:number, is_naked:number, forced_value:number)
feat(W, N, 1, D) :- candcount(W, N), N = 1, candidate(W, D).
feat(W, N, 0, -1) :- candcount(W, N), N != 1.
.output feat
"""

FeatureScope = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    dict[str, torch.Tensor],
]


def row_col_box_scope(
    grids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    true_v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The concrete #90 row∪col∪box implementation, without alteration."""
    return pin.legality_feature_tensor(grids, rows, cols, true_v)


def legality_feature_with_scope(
    grids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    true_v: torch.Tensor,
    scope: FeatureScope = row_col_box_scope,
) -> dict[str, torch.Tensor]:
    """Run a legality feature through an explicit scope implementation.

    A future mined scope can reuse this form and harness, but must earn its own certificate;
    the pluggable hook is never itself certified. The default delegates directly to #90.
    """
    return scope(grids, rows, cols, true_v)


def normalize_grid_value(value: int) -> int | None:
    """Mirror Soufflé val(): 1..9 map to themselves and 0 maps to empty."""
    value = int(value)
    if value == 0:
        return None
    if 1 <= value <= 9:
        return value
    raise ValueError(f"grid value must be in 0..9, got {value}")


def recover_cell_index(position: int) -> tuple[int, int, int]:
    """Recover (row, column, box) from a row-major position using integer division."""
    position = int(position)
    if not 0 <= position < 81:
        raise ValueError(f"cell position must be in 0..80, got {position}")
    row, col = divmod(position, 9)
    return row, col, (row // 3) * 3 + col // 3


def souffle_legality(
    grids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> dict[int, tuple[int, bool, int]]:
    """Run the independent Soufflé mirror over a batch of sudoku boards."""
    if grids.ndim != 3 or tuple(grids.shape[1:]) != (9, 9):
        raise ValueError(f"grids must be [N,9,9], got {tuple(grids.shape)}")
    n = int(grids.shape[0])
    rows_l = [int(x) for x in rows.detach().cpu().tolist()]
    cols_l = [int(x) for x in cols.detach().cpu().tolist()]
    if len(rows_l) != n or len(cols_l) != n:
        raise ValueError("rows and cols must have one entry per grid")
    values = grids.detach().cpu().to(dtype=torch.long)
    if values.numel() and (int(values.min()) < 0 or int(values.max()) > 9):
        raise ValueError("grid values must be in 0..9")

    with tempfile.TemporaryDirectory() as td_raw:
        td = Path(td_raw)
        (td / "prog.dl").write_text(SOUFFLE_PROGRAM)
        with (td / "tok.facts").open("w") as facts:
            for w in range(n):
                facts.writelines(
                    f"{w}\t{p}\t{int(value)}\n"
                    for p, value in enumerate(values[w].reshape(-1).tolist())
                )
        (td / "target.facts").write_text(
            "".join(
                f"{w}\t{rows_l[w] * 9 + cols_l[w]}\n"
                for w in range(n)
            )
        )
        subprocess.run(
            ["souffle", "-F", str(td), "-D", str(td), str(td / "prog.dl")],
            check=True,
            capture_output=True,
            text=True,
        )
        out: dict[int, tuple[int, bool, int]] = {}
        for line in (td / "feat.csv").read_text().splitlines():
            w, n_candidates, is_naked, forced_value = line.split("\t")
            out[int(w)] = (int(n_candidates), bool(int(is_naked)), int(forced_value))
    return out


def oracle_naked_tuple(
    grid: list[list[int]], row: int, col: int, true_v: int,
) -> tuple[int, bool, int]:
    """Read #89's oracle as the frozen naked-only output tuple."""
    candidates = osc.cell_candidates(grid, row, col)
    flags = osc.classify_forced(grid, row, col, true_v)
    is_naked = bool(flags["naked"])
    forced = int(next(iter(candidates))) if is_naked else -1
    return len(candidates), is_naked, forced


def three_way_agreement(
    grids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    true_v: torch.Tensor,
    *,
    dataset_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compare full oracle/tensor/Soufflé tuples, including all abstain rows."""
    feat = legality_feature_with_scope(grids, rows, cols, true_v, row_col_box_scope)
    datalog = souffle_legality(grids, rows, cols)
    n = int(grids.shape[0])
    source_rows = list(range(n)) if dataset_rows is None else [int(x) for x in dataset_rows]
    if len(source_rows) != n:
        raise ValueError("dataset_rows must align with grids")
    grid_l = grids.detach().cpu().tolist()
    row_l = [int(x) for x in rows.detach().cpu().tolist()]
    col_l = [int(x) for x in cols.detach().cpu().tolist()]
    true_l = [int(x) for x in true_v.detach().cpu().tolist()]
    tensor_n = feat["n_candidates"].detach().cpu().tolist()
    tensor_naked = feat["is_naked"].detach().cpu().tolist()
    # #90's shared tensor also computes the frozen-out hidden-single path. The certified
    # NAKED-ONLY tuple projects that shared result to -1 whenever is_naked is false.
    tensor_forced = torch.where(
        feat["is_naked"], feat["forced_value"],
        torch.full_like(feat["forced_value"], -1),
    ).detach().cpu().tolist()
    mismatches: list[dict[str, Any]] = []
    for w in range(n):
        oracle = oracle_naked_tuple(grid_l[w], row_l[w], col_l[w], true_l[w])
        tensor = (int(tensor_n[w]), bool(tensor_naked[w]), int(tensor_forced[w]))
        souffle = datalog.get(w)
        if not (oracle == tensor == souffle):
            mismatches.append({
                "window": w,
                "row": source_rows[w],
                "target": [row_l[w], col_l[w]],
                "oracle": list(oracle),
                "tensor": list(tensor),
                "souffle": None if souffle is None else list(souffle),
            })
    checked = n
    count = checked - len(mismatches)
    return {
        "checked": checked,
        "agreement_count": count,
        "mismatches": len(mismatches),
        "percent": 100.0 * count / checked if checked else 0.0,
        "mismatch_rows": mismatches,
        "proved_over_stated_domain": checked > 0 and not mismatches,
    }


def validate_solution_grid(grid: Sequence[Sequence[int]]) -> list[str]:
    """Return completeness/all-different failures for one proposed solution grid."""
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        return ["shape: expected 9x9"]
    reasons: list[str] = []
    wanted = set(range(1, 10))
    flat = [int(value) for row in grid for value in row]
    if any(value not in wanted for value in flat):
        reasons.append("complete: cells must all be digits 1..9")
    for row in range(9):
        if set(int(x) for x in grid[row]) != wanted:
            reasons.append(f"row {row}: not a permutation of 1..9")
    for col in range(9):
        if {int(grid[row][col]) for row in range(9)} != wanted:
            reasons.append(f"column {col}: not a permutation of 1..9")
    for box in range(9):
        br, bc = (box // 3) * 3, (box % 3) * 3
        values = {
            int(grid[row][col])
            for row in range(br, br + 3)
            for col in range(bc, bc + 3)
        }
        if values != wanted:
            reasons.append(f"box {box}: not a permutation of 1..9")
    return reasons


def scan_solution_grids(grids: Iterable[Sequence[Sequence[int]]]) -> dict[str, Any]:
    """Validate an iterable of solution grids and identify every invalid grid."""
    invalid: list[dict[str, Any]] = []
    scanned = 0
    for index, grid in enumerate(grids):
        scanned += 1
        reasons = validate_solution_grid(grid)
        if reasons:
            invalid.append({"grid": index, "reasons": reasons})
    return {
        "scanned": scanned,
        "invalid": len(invalid),
        "invalid_grids": invalid,
        "clean": scanned > 0 and not invalid,
    }


def parse_corpus_solutions(path: Path = CORPUS) -> list[list[list[int]]]:
    """Parse every SOLUTION grid from the corpus's registered block format."""
    lines = path.read_text().splitlines()
    solutions: list[list[list[int]]] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if lines[index] != "PUZZLE":
            raise ValueError(f"line {index + 1}: expected PUZZLE")
        index += 10
        if index >= len(lines) or lines[index] != "SOLUTION":
            raise ValueError(f"line {index + 1}: expected SOLUTION")
        index += 1
        grid: list[list[int]] = []
        for _ in range(9):
            if index >= len(lines):
                raise ValueError("truncated SOLUTION grid")
            fields = lines[index].split()
            if len(fields) != 9:
                raise ValueError(f"line {index + 1}: expected 9 solution cells")
            try:
                grid.append([int(field) for field in fields])
            except ValueError as exc:
                raise ValueError(f"line {index + 1}: non-integer solution cell") from exc
            index += 1
        solutions.append(grid)
    return solutions


def scan_corpus_validity(path: Path = CORPUS) -> dict[str, Any]:
    return scan_solution_grids(parse_corpus_solutions(path))


def confidence_promotion(
    cert_agreement_fraction: float,
    validity_scan_clean: bool,
    proposer_regressions: int,
    measured_conf: float,
) -> tuple[bool, float, str]:
    """Promote iff all three registered conditions pass exactly."""
    failures: list[str] = []
    if cert_agreement_fraction != 1.0:
        failures.append("certificate agreement is not 100%")
    if not validity_scan_clean:
        failures.append("corpus validity scan is not clean")
    if proposer_regressions != 0:
        failures.append(f"proposer regressions are not 0 ({proposer_regressions})")
    if failures:
        return False, float(measured_conf), "promotion withheld: " + "; ".join(failures)
    return (
        True,
        1.0,
        "promoted: certificate agreement is 100%, corpus validity scan is clean, and "
        "proposer regressions are exactly 0; confidence is certified-constant-1.0",
    )


def compute_regressions(
    cover_pred_without: Sequence[int] | torch.Tensor,
    cover_pred_with: Sequence[int] | torch.Tensor,
    gold: Sequence[int] | torch.Tensor,
    *,
    row_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Count rows changed from correct without proposer to wrong/abstain with it."""
    def values(x: Sequence[int] | torch.Tensor) -> list[int]:
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().tolist()]
        return [int(v) for v in x]

    without, with_, expected = values(cover_pred_without), values(cover_pred_with), values(gold)
    if not (len(without) == len(with_) == len(expected)):
        raise ValueError("prediction and gold arrays must have equal length")
    rows = list(range(len(expected))) if row_indices is None else [int(x) for x in row_indices]
    if len(rows) != len(expected):
        raise ValueError("row_indices must align with predictions")
    regressed = [
        rows[i]
        for i, (before, after, target) in enumerate(zip(without, with_, expected, strict=True))
        if before == target and after != target
    ]
    return {"count": len(regressed), "row_indices": regressed}


def log(message: str = "") -> None:
    print(message, flush=True)


def main() -> int:
    started = time.time()
    log("=" * 72)
    log("SLICE #91 -- legality certificate + admission")
    log("=" * 72)
    log(DESIGN_REGISTERED)
    log(f"FRAMING: {FRAMING_NOTE}")
    log(f"CAVEAT: {REVEAL_INDEX_NOTE}")
    env_used = {key: os.environ.get(key, "") for key in WYLY_ENV}
    log("\nWYLY_* env used for this run:")
    for key, value in env_used.items():
        log(f"  {key}={value}")

    log("\n--- CORPUS VALIDITY SCAN ---")
    validity = scan_corpus_validity()
    log(f"solution grids scanned={validity['scanned']} invalid={validity['invalid']}")
    for invalid in validity["invalid_grids"][:20]:
        log(f"  invalid grid {invalid['grid']}: {invalid['reasons']}")

    codec = v5.load_codec()
    puz_pat, sol_pat = osc.header_patterns(codec)
    value_map = osc.build_value_map(codec)
    bare = {digit: int(codec.encode(str(digit))[0]) for digit in range(1, 10)}
    space = {digit: int(codec.encode(f" {digit}")[0]) for digit in range(1, 10)}

    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    ids_cpu = ids.detach().cpu()
    y_cpu = y.detach().cpu()
    cls_cpu = cls.detach().cpu()
    uv_cpu = uv.detach().cpu() if torch.is_tensor(uv) else uv
    stack = pin.build_grid_stack_for_rows(
        list(range(len(ids))), ids_cpu, y_cpu, cls_cpu, uv_cpu,
        sol_pat, puz_pat, value_map,
    )
    dataset_rows = [int(meta["row"]) for meta in stack["meta"]]
    if not dataset_rows:
        raise RuntimeError("no classifiable solution-cell rows for certificate")

    log("\n--- THREE-WAY CERTIFICATE (full tuple, including -1 rows) ---")
    cert = three_way_agreement(
        stack["grids"], stack["rows"], stack["cols"], stack["true_v"],
        dataset_rows=dataset_rows,
    )
    log(
        f"oracle == tensor == Soufflé: {cert['agreement_count']}/{cert['checked']} "
        f"({cert['percent']:.6f}%) mismatches={cert['mismatches']}"
    )
    for mismatch in cert["mismatch_rows"]:
        log(f"  MISMATCH {mismatch}")
    if cert["proved_over_stated_domain"]:
        log("proved over the stated scored-window domain for concrete row∪col∪box scope")
    else:
        log("NOT proved; promotion blocked")

    log("\n--- COVER REGENERATION + NAKED-ONLY ADMISSION ---")
    model, rules, original_cover = osc.run_cover_regeneration()
    bare_c, space_c = pin.digit_to_compact_maps(uv_cpu, bare, space)
    tensor = legality_feature_with_scope(
        stack["grids"], stack["rows"], stack["cols"], stack["true_v"], row_col_box_scope,
    )
    naked_digits = torch.where(
        tensor["is_naked"], tensor["forced_value"],
        torch.full_like(tensor["forced_value"], -1),
    )
    compact = pin.forced_digits_to_compact(naked_digits, stack["cols"], bare_c, space_c)
    forced_class = torch.full((len(ids),), -1, dtype=torch.long)
    for index, meta in enumerate(stack["meta"]):
        forced_class[int(meta["row"])] = compact[index]
    proposer = pin.make_lookup_proposer(ids_cpu, forced_class)

    import wyly_lm_v2 as v2  # noqa: E402
    nval = min(int(v2.NVAL), max(1, len(tr) // 10))
    val = tr[-nval:]
    measured_conf = float(v5.val_fired_acc(proposer, ids, yv, val))
    val_pred = proposer(ids[val])
    n_val_fired = int((val_pred >= 0).sum().item())
    log(
        f"ValFiredAccuracy measured={measured_conf:.6f} "
        f"(fired {n_val_fired}/{len(val)} validation rows)"
    )

    out_without, pred_without = original_cover(
        model, rules, ids, yv, cls, te, return_pred=True,
    )
    rules_with = list(rules) + [("legality_forced_naked", proposer)]
    out_with, pred_with = original_cover(
        model, rules_with, ids, yv, cls, te,
        extra_conf={"legality_forced_naked": measured_conf}, return_pred=True,
    )
    marginal = float(out_with["agree"]) - float(out_without["agree"])
    te_rows = [int(row) for row in te.detach().cpu().tolist()]
    te_position = {row: position for position, row in enumerate(te_rows)}
    te_set = set(te_rows)
    te_solution_rows = [row for row in dataset_rows if row in te_set]
    positions = [te_position[row] for row in te_solution_rows]
    regression = compute_regressions(
        pred_without.detach().cpu()[positions],
        pred_with.detach().cpu()[positions],
        yv.detach().cpu()[te_solution_rows],
        row_indices=te_solution_rows,
    )
    regression_sample = regression["row_indices"][:50]
    log(
        f"te agree without={float(out_without['agree']):.6f} "
        f"with={float(out_with['agree']):.6f} held-out marginal={marginal:+.6f}"
    )
    log(
        f"solution-cell regressions={regression['count']} "
        f"sample_rows={regression_sample}"
    )

    cert_fraction = cert["agreement_count"] / cert["checked"] if cert["checked"] else 0.0
    promoted, final_conf, reason = confidence_promotion(
        cert_fraction, validity["clean"], regression["count"], measured_conf,
    )
    log(f"\nCONF PROMOTION: promoted={promoted} final_conf={final_conf:.6f}")
    log(f"reason: {reason}")

    wall = time.time() - started
    report = {
        "slice": 91,
        "framing_note": FRAMING_NOTE,
        "late_reveal_caveat": REVEAL_INDEX_NOTE,
        "wyly_env": env_used,
        "validity_scan": validity,
        "certificate": cert,
        "admission": {
            "agree_without": float(out_without["agree"]),
            "agree_with": float(out_with["agree"]),
            "held_out_marginal": marginal,
            "regressions": regression["count"],
            "regression_sample_rows": regression_sample,
            "te_solution_rows": len(te_solution_rows),
        },
        "confidence_promotion": {
            "measured_conf": measured_conf,
            "promoted": promoted,
            "final_conf": final_conf,
            "reason": reason,
            "n_val": len(val),
            "n_val_fired": n_val_fired,
        },
        "wall_time_s": wall,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))

    log("\n" + "=" * 72)
    log("SCOREBOARD")
    log("=" * 72)
    log(f"FRAMING: {FRAMING_NOTE}")
    log(DESIGN_REGISTERED)
    log(f"Validity scan: scanned={validity['scanned']} invalid={validity['invalid']}")
    log(
        f"Three-way cert: {cert['agreement_count']}/{cert['checked']} "
        f"({cert['percent']:.6f}%) mismatches={cert['mismatches']}"
    )
    log(f"Admission held-out marginal: {marginal:+.6f}")
    log(f"Admission regressions: {regression['count']} sample_rows={regression_sample}")
    log(f"Confidence: promoted={promoted} final_conf={final_conf:.6f}; {reason}")
    log(f"CAVEAT: {REVEAL_INDEX_NOTE}")
    log(f"FRAMING: {FRAMING_NOTE}")
    log(f"wall time total: {wall:.1f}s")
    log(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
