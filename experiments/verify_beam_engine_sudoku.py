"""Gate: generic beam_decode + SudokuOracle bit-identical to pilot decide_energy.

Rebuilds the #101 population (same reconstruction loop as campaign_gate_b_pilot.main),
compares engine vs pilot on every record x M in 1..5 (four fields), recomputes the
FIRES scoreboard from engine commits, and asserts exact match against the frozen
#101 fixture (tests/fixtures/gate_b_pilot_reference.json).

Standalone module + validation experiment only — does not touch serve dispatch.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

import campaign_gate_b_pilot as gbp  # noqa: E402
import campaign_sudoku_forced_move as osc  # noqa: E402
import campaign_uniform_reveal as unif  # noqa: E402
import wyly_lm_v5 as v5  # noqa: E402
from beam_engine import beam_decode  # noqa: E402
from sudoku_oracle import SudokuOracle  # noqa: E402

# Frozen #101 constants — do not read os.environ for the seed.
SEED = 0
COMPARE_FIELDS = (
    "committed_value",
    "anomaly_total_contradiction",
    "prune_events",
    "surviving_counts",
)
FIXTURE_PATH = REPO / "tests" / "fixtures" / "gate_b_pilot_reference.json"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _slice_compare(d: dict[str, Any]) -> dict[str, Any]:
    return {k: d[k] for k in COMPARE_FIELDS}


def build_population() -> list[dict[str, Any]]:
    """Duplicate the pilot main() population reconstruction loop (no cover-state cache)."""
    codec = v5.load_codec()
    puz_pat, sol_pat = osc.header_patterns(codec)
    value_map = osc.build_value_map(codec)
    ids, y, cls, uv, tr, te = v5.load_ds()
    n_kept = int(ids.shape[0])
    log(f"load_ds: N={n_kept} tr={len(tr)} te={len(te)}")

    ids_cpu = ids.detach().cpu()
    y_cpu = y.detach().cpu()
    cls_cpu = cls.detach().cpu()
    uv_cpu = uv.detach().cpu() if hasattr(uv, "detach") else uv
    L = int(ids_cpu.shape[1])

    records: list[dict[str, Any]] = []
    n_none = 0
    n_invalid = 0
    n_nongold = 0
    for i in range(n_kept):
        seq = osc.raw_window_from_compact(ids_cpu[i], uv_cpu)
        info = unif.reconstruct_block_for_target(seq, sol_pat, puz_pat, value_map, L=L)
        if info is None:
            n_none += 1
            continue
        if not info["board_valid"]:
            n_invalid += 1
            continue
        gold_tid = int(uv_cpu[int(cls_cpu[int(y_cpu[i])].item())].item())
        tv = value_map.get(gold_tid, ...)
        if tv is ... or tv is None:
            n_nongold += 1
            continue
        tv = int(tv)
        r, c = int(info["r"]), int(info["c"])
        board = info["board"]
        d = gbp.propagation_depth(board, r, c)
        records.append(
            {
                "row": int(i),
                "r": r,
                "c": c,
                "gold": tv,
                "board": board,
                "d": d,
            }
        )
    log(
        f"population_all_valid={len(records)}  "
        f"(skipped none={n_none} invalid={n_invalid} nongold={n_nongold})"
    )
    return records


def recompute_scoreboard(
    records: list[dict[str, Any]],
    commits_engine: dict[int, list[int | None]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    golds = [int(rec["gold"]) for rec in records]
    ds = [rec["d"] for rec in records]
    per_m: dict[int, dict[str, Any]] = {}
    diagonal_acc: dict[int, float] = {}
    diagonal_n: dict[int, int] = {}

    for m in gbp.M_SWEEP:
        mask_stratum = [(d is not None and 2 <= int(d) <= m) for d in ds]
        beam_acc, n_stratum = gbp._acc_on(mask_stratum, commits_engine[m], golds)
        conf_acc, _ = gbp._acc_on(mask_stratum, commits_engine[1], golds)
        separation = beam_acc - conf_acc

        mask_diag = [(d is not None and int(d) == m) for d in ds]
        b_diag, n_diag = gbp._acc_on(mask_diag, commits_engine[m], golds)
        diagonal_acc[m] = b_diag
        diagonal_n[m] = n_diag

        per_m[m] = {
            "beam_acc": beam_acc,
            "conf_acc": conf_acc,
            "separation": separation,
            "n_stratum": n_stratum,
            "beam_acc_on_d_eq_M": b_diag,
            "n_d_eq_M": n_diag,
        }

    verdict_block = gbp._verdict_from(per_m, diagonal_acc, diagonal_n)
    return per_m, verdict_block


def main() -> int:
    t0 = time.time()
    log("=" * 72)
    log("VERIFY beam_decode + SudokuOracle vs pilot decide_energy (#101 gate)")
    log("=" * 72)
    log(
        f"BEAM_WIDTH={gbp.BEAM_WIDTH} M_SWEEP={gbp.M_SWEEP} "
        f"RULE_ID={gbp.RULE_ID} SEED={SEED}"
    )

    count_table, n_grids = gbp.build_sudoku_count_table(gbp.CORPUS)
    log(f"CountTable: n_grids={n_grids} from {gbp.CORPUS}")

    records = build_population()
    n_records = len(records)
    if n_records == 0:
        log("ERROR: empty population")
        return 1

    commits_engine: dict[int, list[int | None]] = {m: [] for m in gbp.M_SWEEP}
    n_compared = 0

    log("--- bit-identity sweep (engine vs pilot) ---")
    for idx, rec in enumerate(records):
        if idx > 0 and idx % 5000 == 0:
            log(f"  ... progress {idx}/{n_records}")
        for m in gbp.M_SWEEP:
            pilot_res = gbp.decide_energy(
                rec["board"],
                rec["r"],
                rec["c"],
                m,
                gbp.BEAM_WIDTH,
                count_table,
                SEED,
                gbp.RULE_ID,
            )
            oracle = SudokuOracle(
                rec["board"], rec["r"], rec["c"], count_table, SEED
            )
            engine_res = beam_decode(
                oracle, m, gbp.BEAM_WIDTH, gbp.RULE_ID, SEED
            )
            n_compared += 1
            pilot_s = _slice_compare(pilot_res)
            engine_s = _slice_compare(engine_res)
            if pilot_s != engine_s:
                log("MISMATCH — first differing (record, M):")
                log(
                    f"  record: row={rec['row']} r={rec['r']} c={rec['c']} "
                    f"gold={rec['gold']} d={rec['d']}"
                )
                log(f"  m={m}")
                log(f"  pilot_res  = {pilot_res!r}")
                log(f"  engine_res = {engine_res!r}")
                return 1
            commits_engine[m].append(engine_res["committed_value"])

    log(
        f"BIT-IDENTICAL: engine == pilot decide_energy on N={n_records} records x "
        f"M={gbp.M_SWEEP} (all 4 compared fields), 0 mismatches."
    )
    log(f"  (total comparisons = {n_compared})")

    per_m, verdict_block = recompute_scoreboard(records, commits_engine)

    log()
    log("=" * 72)
    log("SCOREBOARD -- constrained (gated), recomputed from engine commits")
    log("=" * 72)
    log(
        f"{'M':>3}  {'n':>7}  {'beam_acc':>10}  {'conf_acc':>10}  {'SEPARATION':>12}  "
        f"{'d==M n':>7}  {'beam_d==M':>10}"
    )
    for m in gbp.M_SWEEP:
        row = per_m[m]
        log(
            f"{m:>3}  {row['n_stratum']:>7}  {row['beam_acc']:>10.6f}  "
            f"{row['conf_acc']:>10.6f}  {row['separation']:>+12.6f}  "
            f"{row['n_d_eq_M']:>7}  {row['beam_acc_on_d_eq_M']:>10.6f}"
        )
    log()
    v = verdict_block["verdict"]
    deciding = verdict_block["deciding"]
    if deciding:
        decide_str = (
            f"M={deciding['M']} stratum={deciding['stratum']} "
            f"sep={deciding['separation']:.6f} beam_acc={deciding['beam_acc']:.6f} "
            f"n={deciding['n_stratum']}"
        )
    else:
        decide_str = "no M met both SEPARATION and floor"
    log(f"VERDICT: {v} -- {decide_str}")
    log(verdict_block["explanation"])

    if not FIXTURE_PATH.exists():
        log(f"ERROR: frozen fixture missing: {FIXTURE_PATH}")
        return 1
    fixture = json.loads(FIXTURE_PATH.read_text())

    # Compare scoreboard subset against frozen #101 reference.
    score_ok = True
    for m in gbp.M_SWEEP:
        mkey = str(m)
        ref = fixture["per_m"][mkey]
        got = {
            k: per_m[m][k]
            for k in (
                "beam_acc",
                "conf_acc",
                "separation",
                "n_stratum",
                "beam_acc_on_d_eq_M",
                "n_d_eq_M",
            )
        }
        if got != ref:
            score_ok = False
            log(f"SCOREBOARD MISMATCH vs frozen #101 reference at M={m}")
            log(f"  fixture = {ref!r}")
            log(f"  got     = {got!r}")
    if verdict_block != fixture["verdict"]:
        score_ok = False
        log("SCOREBOARD MISMATCH vs frozen #101 reference (verdict block)")
        log(f"  fixture = {fixture['verdict']!r}")
        log(f"  got     = {verdict_block!r}")

    if not score_ok:
        return 1

    wall = time.time() - t0
    log()
    log(
        f"GATE PASSED: bit-identical reproduction + scoreboard matches frozen #101 "
        f"reference (verdict={v})."
    )
    log(f"wall_time_s={wall:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
