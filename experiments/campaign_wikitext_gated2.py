"""Regression-aware two-sided gating follow-up to the Wikitext rescue campaign.

GATED1 is reproduced descriptively with the slice-81 orchestration.  GATED2 admits
the same B1 candidates but claims only when B0 is weak and B1 is strong.
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from experiments import campaign_wikitext_gated_rescue as campaign  # noqa: E402

OUTPUT = REPO / "data" / "wikitext_gated2.json"
EXPECTED_TAU_LOW_GRID = [
    0.08503663539886475,
    0.15015951693058016,
    0.2123691976070404,
    0.30124656558036805,
    0.44849786162376404,
    0.5126050710678101,
    0.5625,
    0.7023643851280212,
    0.8238235354423527,
    0.9989785552024841,
]
EXPECTED_FLAT_TEST_AGREE = 0.3496747314929962
EXPECTED_GATED1_TEST_AGREE = 0.3585037291049957
EXPECTED_GATED1_REGRESSIONS = 62
EXPECTED_GATED1_TAU = 0.30124656558036805
REPRO_TOLERANCE = 1e-7
TAU_TOLERANCE = 1e-7
HONESTY_NOTE = (
    "candidate families are template_fixed (not learned end-to-end); frac_induced is "
    "unaffected; the frozen B0′ artifact was reused and never regenerated this run; "
    "GATED2 receives a single test_te evaluation with no repeated test peeking."
)


def log(message: str = "") -> None:
    print(message, flush=True)


def two_sided_merge(
    b0_pred: torch.Tensor,
    b0_conf: torch.Tensor,
    b1_pred: torch.Tensor,
    b1_conf: torch.Tensor,
    tau_low: float,
    tau_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Let B1 claim only on B0-low/B1-high rows where B1 fires."""
    claim = (b1_pred >= 0) & (b0_conf <= tau_low) & (b1_conf >= tau_high)
    pred = torch.where(claim, b1_pred, b0_pred)
    conf = torch.where(claim, b1_conf, b0_conf)
    return pred, conf, claim


def flat_gate_passes(agree: float) -> bool:
    """Registered Rule 1, inclusive at both tolerance boundaries."""
    return 0.3497 - 0.005 <= agree <= 0.3497 + 0.005


def select_operating_point(
    frontier: list[dict[str, Any]],
    flat_val_agree: float,
    flat_val_regressions: int,
) -> dict[str, Any] | None:
    """Apply the registered validation filter and deterministic tie-breaks."""
    admissible = [
        row
        for row in frontier
        if row["val_agree"] >= flat_val_agree + 0.005
        and row["val_regressions"] <= flat_val_regressions
    ]
    if not admissible:
        return None
    return max(
        admissible,
        key=lambda row: (
            row["val_agree"],
            -row["val_regressions"],
            row["gated_coverage"],
        ),
    )


def resolve_operating_point(
    selected: dict[str, Any] | None,
    evaluate_test: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Keep the no-admissible path structurally incapable of touching GATED2 test data."""
    if selected is None:
        log("no admissible operating point")
        return {"outcome": "no admissible operating point", "gated2_test": None}
    return {"outcome": "selected", "gated2_test": evaluate_test(selected)}


def _write(report: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    log(f"JSON: {OUTPUT}")


def _print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    log(title)
    for row in rows:
        log("  " + json.dumps(row, sort_keys=True))


def _b0_conf_fns(rules) -> dict[str, Any]:
    conf_fns = {
        name: campaign.v5.CONF_FNS[name]
        for name, _ in rules
        if name in campaign.v5.CONF_FNS
    }
    for name, fn in rules:
        if name not in conf_fns:
            scalar = float(campaign.v5.RULE_CONF.get(name, 0.0))
            conf_fns[name] = lambda w, fn=fn, scalar=scalar: (
                fn(w),
                torch.full((len(w),), scalar, device=w.device),
            )
    return conf_fns


def _predict_gated1(stack, rows: torch.Tensor, counts_fn):
    stack.forward(rows, counts_row_fn=counts_fn)
    state = stack.last_carried[-1]
    return state.pred, state.conf


def _metrics(pred, target, base_pred) -> dict[str, Any]:
    correct = pred == target
    base_correct = base_pred == target
    return {
        "agree": float(correct.float().mean()),
        "regressions": int((base_correct & ~correct).sum()),
        "correct": correct,
    }


def _family(name: str) -> str | None:
    for family in campaign.CANDIDATE_FAMILIES:
        if name == family or name.startswith(family + " ") or name.startswith(family + "["):
            return family
    return None


def _family_set(admitted: list[str]) -> set[str]:
    return {family for name in admitted if (family := _family(name)) is not None}


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "correct"}


def main() -> int:
    started = time.perf_counter()
    report: dict[str, Any] = {
        "honesty_note": HONESTY_NOTE,
        "architecture": (
            "GATED1 uses BlockStack(carry='gated'); GATED2 bypasses BlockStack merging and "
            "uses direct per-block SW-cover predictions with local two_sided_merge."
        ),
        "constants": {
            "candidate_families": list(campaign.CANDIDATE_FAMILIES),
            "admit_thresh": campaign.ADMIT_THRESH,
            "max_rules": campaign.MAX_RULES,
        },
        "gaps_or_ambiguities": [],
    }

    log("=== B0′ LOAD + DEFENSIVE BAND CHECK ===")
    if not campaign.B0_FREEZE.exists():
        report["stopped"] = "frozen B0′ artifact missing"
        log(f"STOP: frozen B0′ artifact missing: {campaign.B0_FREEZE}")
        _write(report)
        return 2
    try:
        model, rules, frozen_core, frozen_wall = campaign.load_frozen_b0(campaign.B0_FREEZE)
    except Exception as exc:  # report faithfully; regeneration is forbidden
        report["stopped"] = f"frozen B0′ load failed: {type(exc).__name__}: {exc}"
        log(f"STOP: {report['stopped']}")
        _write(report)
        return 2
    b0_band = campaign.b0_within_registered_band(frozen_core["agree"])
    report["b0"] = {
        "artifact": str(campaign.B0_FREEZE),
        "source": "loaded frozen B0′; never regenerated",
        "original_fit_wall_seconds": frozen_wall,
        "core_sw": frozen_core,
        "rules": [name for name, _ in rules],
        "defensive_registered_band_pass": b0_band,
    }
    log(f"artifact: {campaign.B0_FREEZE}")
    log("source: loaded frozen B0′ (regeneration forbidden and not attempted)")
    log(f"frozen core_sw: {json.dumps(frozen_core, sort_keys=True)}")
    log(f"B0′ registered band [{campaign.B0_SCORE_MIN:.3f}, {campaign.B0_SCORE_MAX:.3f}]: {b0_band}")
    if not b0_band:
        report["stopped"] = "defensive B0′ registered-band check failed"
        log("STOP: defensive B0′ registered-band check failed")
        _write(report)
        return 2

    ids, y, cls, uv, tr, te = campaign.v5.load_ds()
    yv = cls[y]
    val_te, test_te = campaign.split_te(te)
    active_families, exclusions = campaign.active_candidate_families(
        [name for name, _ in rules]
    )
    counts_fn = campaign._counts_row_fn(model, cls)
    b0_conf_fns = _b0_conf_fns(rules)
    b0 = campaign.WylyBlock(0, "B0")
    b0.rules = list(rules)
    b0.conf_fns = b0_conf_fns
    b0_pred_val, b0_conf_val = b0.predict_cover(ids[val_te], counts_row_fn=counts_fn)
    b0_pred_test, _ = b0.predict_cover(ids[test_te], counts_row_fn=counts_fn)
    b0_pred_te, _ = b0.predict_cover(ids[te], counts_row_fn=counts_fn)
    traced, traced_pred, trace = campaign.core_cover_sw_traced(
        model, rules, ids, yv, cls, te
    )
    if not torch.equal(b0_pred_te, traced_pred):
        raise AssertionError("direct B0 WylyBlock prediction differs from traced core cover")
    report["b0"]["traced_core_sw"] = traced
    report["split"] = {"seed": 0, "val_te": len(val_te), "test_te": len(test_te)}
    report["pool"] = {
        "active_families": list(active_families),
        "exclusions": exclusions,
    }
    log(f"B0′ rules: {[name for name, _ in rules]}")
    log(f"traced/direct prediction parity on te: {torch.equal(b0_pred_te, traced_pred)}")
    log(f"split: val_te={len(val_te)} test_te={len(test_te)} seed=0")
    log(f"active candidate families: {list(active_families)}")
    log(f"pool exclusions: {exclusions}")

    _, _, tr_trace = campaign.core_cover_sw_traced(model, rules, ids, yv, cls, tr)
    val_pos = torch.searchsorted(te, val_te)
    val_conf_from_trace = trace["conf"][val_pos]
    if not torch.equal(val_conf_from_trace, b0_conf_val):
        raise AssertionError("B0 val confidence extraction differs from direct block cover")
    taus = campaign.tau_deciles(val_conf_from_trace)
    grid_matches = all(
        abs(actual - expected) <= TAU_TOLERANCE
        for actual, expected in zip(taus, EXPECTED_TAU_LOW_GRID, strict=True)
    )
    report["tau_low_grid"] = taus
    report["tau_low_grid_matches_registered_literals"] = grid_matches

    log("=== STEP 0: #81 FLAT/GATED1 REPRODUCTION ===")
    log(f"tau grid: {taus}")
    log(f"tau grid matches registered literal values: {grid_matches}")
    if not grid_matches:
        report["stopped"] = "B0 val-confidence tau grid drifted from registered values"
        log("STOP: B0 val-confidence tau grid drifted from registered values")
        _write(report)
        return 2

    step0_runs: list[dict[str, Any]] = []
    fitted_step0: list[tuple[list[Any], Any]] = []
    for tau in taus:
        candidates, low_train, mining_log = campaign.fit_candidate_pool(
            model,
            ids,
            yv,
            cls,
            uv,
            tr,
            tr_trace["conf"],
            tau,
            val_te,
            test_te,
            active_families,
        )
        stack, admitted, marginals, val_agree = campaign._run_gated_tau(
            rules, b0_conf_fns, candidates, counts_fn, ids, yv, val_te, tau
        )
        step0_runs.append(
            {
                "tau": tau,
                "val_agree": val_agree,
                "admitted": admitted,
                "marginals": marginals,
                "mined_train_rows": len(low_train),
                "mining_log_rows": len(mining_log),
            }
        )
        fitted_step0.append((candidates, stack))
        log(
            f"GATED1 tau={tau:.9f} val_agree={val_agree:.9f} "
            f"admitted={admitted} mined_train={len(low_train)}"
        )
    best_index = min(
        range(10), key=lambda i: (-step0_runs[i]["val_agree"], step0_runs[i]["tau"])
    )
    chosen_gated1 = step0_runs[best_index]
    chosen_candidates, gated1_stack = fitted_step0[best_index]
    flat, flat_admitted, flat_marginals = campaign._run_flat(
        model, rules, b0_conf_fns, chosen_candidates, counts_fn, ids, yv, val_te
    )

    flat_pred_val, _ = flat.predict_cover(ids[val_te], counts_row_fn=counts_fn)
    flat_pred_test, _ = flat.predict_cover(ids[test_te], counts_row_fn=counts_fn)
    gated1_pred_test, _ = _predict_gated1(gated1_stack, ids[test_te], counts_fn)
    flat_val = _metrics(flat_pred_val, yv[val_te], b0_pred_val)
    flat_test = _metrics(flat_pred_test, yv[test_te], b0_pred_test)
    gated1_test = _metrics(gated1_pred_test, yv[test_te], b0_pred_test)
    gated1_delta = gated1_test["agree"] - flat_test["agree"]
    reproduced = (
        abs(flat_test["agree"] - EXPECTED_FLAT_TEST_AGREE) <= REPRO_TOLERANCE
        and abs(gated1_test["agree"] - EXPECTED_GATED1_TEST_AGREE) <= REPRO_TOLERANCE
        and gated1_test["regressions"] == EXPECTED_GATED1_REGRESSIONS
        and abs(chosen_gated1["tau"] - EXPECTED_GATED1_TAU) <= TAU_TOLERANCE
    )
    rule1 = flat_gate_passes(flat_test["agree"])
    report["step0"] = {
        "tau_runs": step0_runs,
        "chosen_tau": chosen_gated1["tau"],
        "flat": {
            "admitted": flat_admitted,
            "marginals": flat_marginals,
            "val": _public_metrics(flat_val),
            "test": _public_metrics(flat_test),
        },
        "gated1": {
            "admitted": chosen_gated1["admitted"],
            "marginals": chosen_gated1["marginals"],
            "test": _public_metrics(gated1_test),
            "test_delta_over_flat": gated1_delta,
            "plus_0.0088_and_62_reproduced": reproduced,
        },
        "rule1_flat_gate_pass": rule1,
    }
    log(f"chosen GATED1 tau: {chosen_gated1['tau']:.9f}")
    log(f"FLAT admitted: {flat_admitted}")
    _print_rows("FLAT admission marginals:", flat_marginals)
    log(f"GATED1 admitted: {chosen_gated1['admitted']}")
    _print_rows("GATED1 admission marginals:", chosen_gated1["marginals"])
    log(
        "GATED1 reference: "
        f"flat_test_agree={flat_test['agree']:.9f} "
        f"gated1_test_agree={gated1_test['agree']:.9f} "
        f"delta={gated1_delta:+.9f} regressions={gated1_test['regressions']}"
    )
    log(f"+0.0088/62 reproduced: {'yes' if reproduced else 'no'}")
    log(
        f"REGISTERED RULE 1: abs({flat_test['agree']:.9f} - 0.3497) <= 0.005: {rule1}"
    )
    if not reproduced:
        report["stopped"] = "GATED1 reference reproduction red flag"
        log("STOP: GATED1 reference reproduction red flag")
        _write(report)
        return 2
    if not rule1:
        report["stopped"] = "Registered Rule 1 FLAT agreement gate failed"
        log("STOP: Registered Rule 1 FLAT agreement gate failed")
        _write(report)
        return 2

    log("=== STEP 1: GATED2 VALIDATION FRONTIER ===")
    frontier: list[dict[str, Any]] = []
    gated2_by_tau: dict[int, dict[str, Any]] = {}
    per_tau_admission: list[dict[str, Any]] = []
    for tau_index, tau_low in enumerate(taus):
        candidates, low_train, mining_log = campaign.fit_candidate_pool(
            model,
            ids,
            yv,
            cls,
            uv,
            tr,
            tr_trace["conf"],
            tau_low,
            val_te,
            test_te,
            active_families,
        )
        stack, admitted, marginals, gated1_val_agree = campaign._run_gated_tau(
            rules, b0_conf_fns, candidates, counts_fn, ids, yv, val_te, tau_low
        )
        b1 = stack.blocks[1]
        b1_pred_val, b1_conf_val = b1.predict_cover(ids[val_te], counts_row_fn=None)
        tau_high_grid = campaign.tau_deciles(b1_conf_val)
        gated2_by_tau[tau_index] = {
            "b1": b1,
            "admitted": admitted,
            "marginals": marginals,
            "tau_high_grid": tau_high_grid,
        }
        per_tau_admission.append(
            {
                "tau_index": tau_index,
                "tau_low": tau_low,
                "admitted": admitted,
                "admitted_families": sorted(_family_set(admitted)),
                "marginals": marginals,
                "pointer_marginals": [
                    row for row in marginals if _family(row["name"]) == "pointer"
                ],
                "gated1_val_agree": gated1_val_agree,
                "mined_train_rows": len(low_train),
                "mining_log_rows": len(mining_log),
                "tau_high_grid": tau_high_grid,
            }
        )
        for tau_high in tau_high_grid:
            merged, _, claim = two_sided_merge(
                b0_pred_val,
                b0_conf_val,
                b1_pred_val,
                b1_conf_val,
                tau_low,
                tau_high,
            )
            merged_metrics = _metrics(merged, yv[val_te], b0_pred_val)
            frontier.append(
                {
                    "tau_index": tau_index,
                    "tau_low": tau_low,
                    "tau_high": tau_high,
                    "val_agree": merged_metrics["agree"],
                    "val_regressions": merged_metrics["regressions"],
                    "gated_coverage": float(claim.float().mean()),
                    "gated_coverage_count": int(claim.sum()),
                    "admitted": admitted,
                }
            )
    report["gated2"] = {
        "per_tau_low_admission": per_tau_admission,
        "frontier": frontier,
    }
    _print_rows("FULL GATED2 VALIDATION FRONTIER:", frontier)

    selected = select_operating_point(
        frontier, flat_val["agree"], flat_val["regressions"]
    )
    report["gated2"]["rule2_baseline"] = {
        "flat_val_agree": flat_val["agree"],
        "flat_val_regressions": flat_val["regressions"],
        "required_agree": flat_val["agree"] + 0.005,
        "maximum_regressions": flat_val["regressions"],
    }
    report["gated2"]["selected"] = selected
    log("=== REGISTERED RULE 2: VALIDATION SELECTION ===")
    log(
        f"FLAT val agree={flat_val['agree']:.9f} regressions={flat_val['regressions']}; "
        f"required GATED2 agree>={flat_val['agree'] + 0.005:.9f}, "
        f"regressions<={flat_val['regressions']}"
    )

    def evaluate_test(point: dict[str, Any]) -> dict[str, Any]:
        tau_state = gated2_by_tau[point["tau_index"]]
        b1_pred_test, b1_conf_test = tau_state["b1"].predict_cover(
            ids[test_te], counts_row_fn=None
        )
        merged_test, _, claim_test = two_sided_merge(
            b0_pred_test,
            b0.predict_cover(ids[test_te], counts_row_fn=counts_fn)[1],
            b1_pred_test,
            b1_conf_test,
            point["tau_low"],
            point["tau_high"],
        )
        gated2_metrics = _metrics(merged_test, yv[test_te], b0_pred_test)
        flat_correct = flat_test["correct"]
        gated2_correct = gated2_metrics["correct"]
        b = int((flat_correct & ~gated2_correct).sum())
        c = int((gated2_correct & ~flat_correct).sum())
        pvalue = campaign.exact_discordant_p(b, c)
        chosen_families = _family_set(tau_state["admitted"])
        flat_families = _family_set(flat_admitted)
        clauses = {
            "i_gated2_admits_family_flat_declined": bool(chosen_families - flat_families),
            "ii_agree_plus_0.005": gated2_metrics["agree"] >= flat_test["agree"] + 0.005,
            "iii_two_sided_exact_p_lt_0.05": pvalue < 0.05,
            "iv_regressions_no_more_than_flat": (
                gated2_metrics["regressions"] <= flat_test["regressions"]
            ),
        }
        first_three = all(list(clauses.values())[:3])
        if all(clauses.values()):
            verdict = "PASS"
        elif (
            first_three
            and not clauses["iv_regressions_no_more_than_flat"]
            and gated2_metrics["regressions"] <= 12
        ):
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"
        result = {
            "agree": gated2_metrics["agree"],
            "regressions": gated2_metrics["regressions"],
            "gated_coverage": float(claim_test.float().mean()),
            "gated_coverage_count": int(claim_test.sum()),
            "admitted": tau_state["admitted"],
            "admitted_families": sorted(chosen_families),
            "flat_admitted_families": sorted(flat_families),
            "new_families_vs_flat": sorted(chosen_families - flat_families),
            "discordant_convention": "b=flat-correct/GATED2-wrong; c=GATED2-correct/flat-wrong",
            "b": b,
            "c": c,
            "two_sided_exact_binomial_p": pvalue,
            "clauses": clauses,
            "verdict": verdict,
        }
        return result

    resolved = resolve_operating_point(selected, evaluate_test)
    report["gated2"].update(resolved)
    if selected is None:
        log("GATED2 test_te: not touched (no admissible validation point)")
        _print_rows("PER-tau_low ADMISSION MARGINALS (including pointer):", per_tau_admission)
    else:
        test_result = resolved["gated2_test"]
        log("selected operating point: " + json.dumps(selected, sort_keys=True))
        selected_admission = per_tau_admission[selected["tau_index"]]
        _print_rows("CHOSEN tau_low ADMISSION MARGINALS:", selected_admission["marginals"])
        log(
            "pointer behind the two-sided gate (all greedy-round marginals): "
            + json.dumps(selected_admission["pointer_marginals"], sort_keys=True)
        )
        log("=== REGISTERED RULE 3: ONE-SHOT test_te VERDICT ===")
        log("GATED2 test metrics: " + json.dumps(test_result, sort_keys=True))
        for clause, value in test_result["clauses"].items():
            log(f"  {clause}: {value}")
        log(f"FINAL VERDICT: {test_result['verdict']}")
        shrinkage = {
            "validation": {
                "gated2_agree": selected["val_agree"],
                "gated2_regressions": selected["val_regressions"],
                "delta_agree_over_flat": selected["val_agree"] - flat_val["agree"],
                "delta_regressions_vs_flat": (
                    selected["val_regressions"] - flat_val["regressions"]
                ),
            },
            "test": {
                "gated2_agree": test_result["agree"],
                "gated2_regressions": test_result["regressions"],
                "delta_agree_over_flat": test_result["agree"] - flat_test["agree"],
                "delta_regressions_vs_flat": (
                    test_result["regressions"] - flat_test["regressions"]
                ),
            },
            "val_to_test": {
                "gated2_agree_change": test_result["agree"] - selected["val_agree"],
                "gated2_regressions_change": (
                    test_result["regressions"] - selected["val_regressions"]
                ),
                "delta_over_flat_shrinkage": (
                    (test_result["agree"] - flat_test["agree"])
                    - (selected["val_agree"] - flat_val["agree"])
                ),
            },
        }
        report["gated2"]["val_to_test_shrinkage"] = shrinkage
        log("val -> test shrinkage: " + json.dumps(shrinkage, sort_keys=True))

    total_wall = time.perf_counter() - started
    report["total_wall_seconds"] = total_wall
    log("=== DESCRIPTIVE EXTRAS ===")
    log(
        "GATED1 +0.0088/62 reproduced: "
        f"{'yes' if reproduced else 'no'} (actual delta={gated1_delta:+.9f}, "
        f"regressions={gated1_test['regressions']})"
    )
    log("honesty note: " + HONESTY_NOTE)
    log(f"total wall time: {total_wall:.3f}s")
    _write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
