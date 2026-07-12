"""Frontier-row measurement across the registered #81, #83, and #86 points (slice #87).

This is a measurement campaign: it reproduces historical operating points and adds row-level
tracing and statistics.  It introduces no fitting, mining, or admission design.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

OUTPUT = REPO / "data" / "frontier_rows.json"
FLOOR = 20
PERMUTATIONS = 10_000
BOOTSTRAPS = 5_000
NO_RUNNER_CONF = -1e9
AXES = ("confidence", "gap", "teacher_consensus", "decile_index")
HONESTY_NOTE = (
    "Stage 2 is descriptive by registered exclusion: it is the same GATED1 mechanism as Stage 1 "
    "at finer key granularity, not independent evidence. Stage 3 votes only if both deduplicated "
    "classes clear the universal floor. All measurements are single historical operating points "
    "(GATED1's chosen tau and #86's already-registered grid); no new sweep produced them."
)
REFUTED_AMBIGUITY = (
    "The REFUTED rule is not explicitly power-gated. The registered literal (without a power "
    "gate) determines the verdict; the power-gated result is also reported for transparency."
)
VERDICT_OVERLAP_RESOLUTION = (
    "If CONFIRMED and literal REFUTED are simultaneously true because a significant refuting "
    "axis was excluded as underpowered from CONFIRMED, the stage is MIXED; neither mutually "
    "exclusive registered label holds cleanly."
)


def log(message: str = "") -> None:
    print(message, flush=True)


def voting_eligible(gains: int, regressions: int, floor: int = FLOOR) -> bool:
    """Apply the universal inclusive per-class floor."""
    return gains >= floor and regressions >= floor


def extract_row_sets(
    base_correct: torch.Tensor, candidate_correct: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return positional gain, regression, and untouched row indices."""
    if base_correct.shape != candidate_correct.shape:
        raise ValueError("correctness tensors must have identical shapes")
    gains = ~base_correct & candidate_correct
    regressions = base_correct & ~candidate_correct
    untouched = ~(gains | regressions)
    return {
        "gains": gains.nonzero(as_tuple=False).flatten().cpu(),
        "regressions": regressions.nonzero(as_tuple=False).flatten().cpu(),
        "untouched": untouched.nonzero(as_tuple=False).flatten().cpu(),
    }


def discordance_crosscheck(
    base_correct: torch.Tensor,
    flat_correct: torch.Tensor,
    gated_correct: torch.Tensor,
) -> dict[str, int]:
    """Explain B0-vs-GATED counts in terms of the distinct FLAT-vs-GATED contrast."""
    if not (base_correct.shape == flat_correct.shape == gated_correct.shape):
        raise ValueError("correctness tensors must have identical shapes")
    base_gain = ~base_correct & gated_correct
    base_reg = base_correct & ~gated_correct
    b = flat_correct & ~gated_correct
    c = ~flat_correct & gated_correct
    return {
        "b0_to_gated_gains": int(base_gain.sum()),
        "b0_to_gated_regressions": int(base_reg.sum()),
        "flat_to_gated_b": int(b.sum()),
        "flat_to_gated_c": int(c.sum()),
        "gains_also_c": int((base_gain & c).sum()),
        "gains_not_c": int((base_gain & ~c).sum()),
        "c_not_b0_gain": int((c & ~base_gain).sum()),
        "regressions_also_b": int((base_reg & b).sum()),
        "regressions_not_b": int((base_reg & ~b).sum()),
        "b_not_b0_regression": int((b & ~base_reg).sum()),
    }


def pool_point_rows(points: Iterable[dict[str, Iterable[int]]]) -> dict[str, Any]:
    """Pool per-point gain/regression identities, retaining pre-dedup totals."""
    rows = list(points)
    gain_sets = [set(map(int, row["gains"])) for row in rows]
    regression_sets = [set(map(int, row["regressions"])) for row in rows]
    gains = set().union(*gain_sets) if gain_sets else set()
    regressions = set().union(*regression_sets) if regression_sets else set()
    return {
        "pre_dedup_gains": sum(len(row) for row in gain_sets),
        "post_dedup_gains": len(gains),
        "pre_dedup_regressions": sum(len(row) for row in regression_sets),
        "post_dedup_regressions": len(regressions),
        "gains": sorted(gains),
        "regressions": sorted(regressions),
    }


def auc_mann_whitney(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Probability that a gain has a larger value than a regression, ties worth one half."""
    x = torch.as_tensor(positive, dtype=torch.float64)
    y = torch.as_tensor(negative, dtype=torch.float64)
    if not len(x) or not len(y):
        raise ValueError("AUC requires two non-empty classes")
    ordered_y = torch.sort(y).values
    below = torch.searchsorted(ordered_y, x, right=False)
    at_or_below = torch.searchsorted(ordered_y, x, right=True)
    u = below.double().sum() + 0.5 * (at_or_below - below).double().sum()
    return float(u / (len(x) * len(y)))


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """One-based average ranks with exact tie handling."""
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty(len(values), dtype=torch.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return ranks


def auc_statistics(
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    permutations: int = PERMUTATIONS,
    bootstraps: int = BOOTSTRAPS,
    seed: int = 0,
) -> dict[str, Any]:
    """AUC, seeded two-sided permutation p-value, and percentile bootstrap interval."""
    x = torch.as_tensor(positive, dtype=torch.float64).cpu()
    y = torch.as_tensor(negative, dtype=torch.float64).cpu()
    if not len(x) or not len(y):
        raise ValueError("statistics require two non-empty classes")
    observed = auc_mann_whitney(x, y)
    pooled = torch.cat((x, y))
    ranks = _average_ranks(pooled)
    nx, ny = len(x), len(y)
    generator = torch.Generator().manual_seed(seed)
    extreme = 0
    for _ in range(permutations):
        selected = torch.randperm(len(pooled), generator=generator)[:nx]
        u = float(ranks[selected].sum()) - nx * (nx + 1) / 2
        perm_auc = u / (nx * ny)
        extreme += abs(perm_auc - 0.5) >= abs(observed - 0.5) - 1e-15
    pvalue = (extreme + 1) / (permutations + 1)

    samples = torch.empty(bootstraps, dtype=torch.float64)
    for index in range(bootstraps):
        xb = x[torch.randint(nx, (nx,), generator=generator)]
        yb = y[torch.randint(ny, (ny,), generator=generator)]
        samples[index] = auc_mann_whitney(xb, yb)
    low, high = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.float64))
    width = float(high - low)
    powered = 0.65 - width / 2 > 0.5
    return {
        "auc": observed,
        "permutation_p_two_sided": pvalue,
        "bootstrap_95_ci": [float(low), float(high)],
        "bootstrap_method": "percentile",
        "ci_width": width,
        "powered_for_auc_0.65": powered,
        "power_heuristic": (
            "CI width recentered at 0.65 excludes 0.5" if powered else
            "CI width recentered at 0.65 does not exclude 0.5"
        ),
        "permutations": permutations,
        "bootstraps": bootstraps,
        "seed": seed,
        "n_positive": nx,
        "n_negative": ny,
    }


def calibration_deciles(confidence: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Map confidence to 1..10 using the reference's fixed 10%..90% quantiles."""
    if not len(reference):
        raise ValueError("decile reference must not be empty")
    quantiles = torch.arange(1, 10, dtype=torch.float64, device=reference.device) / 10
    boundaries = torch.quantile(reference.double(), quantiles)
    return torch.bucketize(confidence.double(), boundaries, right=True) + 1


def core_cover_sw_runner_traced(model, rules, ids, yv, cls, idxs):
    """Read-only mirror of core_cover_sw that additionally tracks the runner-up confidence.

    If no second eligible candidate fires, ``runner_conf`` remains -1e9 and the resulting large
    gap denotes "no real competitor" rather than a calibrated numeric margin.
    """
    from experiments import campaign_wikitext_gated_rescue as gated_rescue

    v5 = gated_rescue.v5
    w = ids[idxs]
    pred = torch.full_like(w[:, -1], -1)
    conf = torch.full((len(w),), NO_RUNNER_CONF, device=w.device)
    runner_conf = torch.full_like(conf, NO_RUNNER_CONF)
    winner = ["abstain"] * len(w)

    def consider(a: torch.Tensor, c: torch.Tensor, name: str, eligible=None) -> None:
        nonlocal pred, conf, runner_conf, winner
        valid = a >= 0 if eligible is None else eligible & (a >= 0)
        beats_winner = valid & (c > conf)
        beats_runner_only = valid & (c > runner_conf) & ~beats_winner
        runner_conf = torch.where(
            beats_winner, conf, torch.where(beats_runner_only, c, runner_conf)
        )
        pred = torch.where(beats_winner, a, pred)
        conf = torch.where(beats_winner, c, conf)
        for index in beats_winner.nonzero(as_tuple=False).flatten().cpu().tolist():
            winner[index] = name

    for name, fn in rules:
        if name in v5.CONF_FNS:
            value, confidence = v5.CONF_FNS[name](w)
        else:
            value = fn(w)
            confidence = torch.full(
                (len(w),), float(v5.RULE_CONF.get(name, 0.0)), device=w.device
            )
        consider(value, confidence, name)

    token = w[:, -1]
    row = model.counts[token]
    maximum, argmax = row.max(1)
    total = row.sum(1)
    value = torch.where(maximum >= 1, cls[argmax], torch.full_like(token, -1))
    confidence = maximum.float() / (total + v5.ALPHA)
    calibrated = getattr(model, "counts_calib", None)
    if calibrated is not None:
        calibrated_conf = calibrated[token]
        confidence = torch.where(calibrated_conf >= 0, calibrated_conf, confidence)
    consider(value, confidence, "counts")

    if v5.STRATA and getattr(model, "rules2", None):
        low = conf < v5.STRATA_TAU
        for name, fn in model.rules2:
            if name in v5.CONF_FNS:
                value, confidence = v5.CONF_FNS[name](w)
            else:
                value = fn(w)
                confidence = torch.full(
                    (len(w),), float(v5.RULE_CONF.get(name, 0.0)), device=w.device
                )
            consider(value, confidence, f"stratum-2/{name}", eligible=low)

    correct = pred == yv[idxs]
    fired = pred >= 0
    out = {
        "agree": float(correct.float().mean()),
        "cover": float(fired.float().mean()),
        "agree_fired": float(correct[fired].float().mean()) if bool(fired.any()) else 0.0,
    }
    trace = {
        "name": winner,
        "conf": conf,
        "runner_conf": runner_conf,
        "gap": conf - runner_conf,
        "correct": correct,
        "pred": pred,
    }
    return out, pred, trace


def assert_runner_trace_parity(model, rules, ids, yv, cls, idxs) -> dict[str, Any]:
    """Prove that runner-up bookkeeping does not perturb original prediction/confidence."""
    from experiments import campaign_wikitext_gated_rescue as gated_rescue

    _, original_pred = gated_rescue.v5.core_cover_sw(
        model, rules, ids, yv, cls, idxs, return_pred=True
    )
    _, traced_pred, trace = core_cover_sw_runner_traced(model, rules, ids, yv, cls, idxs)
    _, prior_pred, prior_trace = gated_rescue.core_cover_sw_traced(
        model, rules, ids, yv, cls, idxs
    )
    if not torch.equal(original_pred, traced_pred) or not torch.equal(prior_pred, traced_pred):
        raise AssertionError("runner-up tracer prediction parity failed")
    if not torch.equal(prior_trace["conf"], trace["conf"]):
        raise AssertionError("runner-up tracer confidence parity failed")
    return trace


def _axis_values(trace: dict[str, Any], consensus: torch.Tensor) -> dict[str, torch.Tensor]:
    conf = trace["conf"].detach().cpu()
    return {
        "confidence": conf,
        "gap": trace["gap"].detach().cpu(),
        "teacher_consensus": consensus.detach().cpu().double(),
        "decile_index": calibration_deciles(conf, conf).cpu().double(),
    }


def analyze_row_sets(
    axes: dict[str, torch.Tensor],
    gains: Sequence[int],
    regressions: Sequence[int],
    untouched: Sequence[int],
    *,
    seed: int,
    permutations: int = PERMUTATIONS,
    bootstraps: int = BOOTSTRAPS,
) -> dict[str, Any]:
    """Compute the four guarded-axis contrasts and the gap boundary clause."""
    gain_idx = torch.as_tensor(gains, dtype=torch.long)
    regression_idx = torch.as_tensor(regressions, dtype=torch.long)
    untouched_idx = torch.as_tensor(untouched, dtype=torch.long)
    statistics = {}
    for offset, name in enumerate(AXES):
        statistics[name] = auc_statistics(
            axes[name][gain_idx], axes[name][regression_idx],
            permutations=permutations, bootstraps=bootstraps, seed=seed + offset,
        )
    changed = torch.unique(torch.cat((gain_idx, regression_idx)))
    # The hypothesis predicts *smaller* gaps for changed rows. Orient this boundary AUC as the
    # probability that an untouched row has a larger gap than a changed row, so >=.65 has the
    # registered frontier meaning rather than silently testing the opposite direction.
    statistics["boundary_gap_untouched_higher_than_changed"] = auc_statistics(
        axes["gap"][untouched_idx], axes["gap"][changed],
        permutations=permutations, bootstraps=bootstraps, seed=seed + len(AXES),
    )
    return statistics


def resolve_verdict(statistics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply registered CONFIRMED/REFUTED/MIXED rules and expose the power asymmetry."""
    guarded = {name: statistics[name] for name in AXES}
    powered = {name: row for name, row in guarded.items() if row["powered_for_auc_0.65"]}
    excluded = sorted(set(guarded) - set(powered))
    boundary = statistics["boundary_gap_untouched_higher_than_changed"]
    confirmed = (
        bool(powered)
        and all(row["auc"] <= 0.55 for row in powered.values())
        and boundary["auc"] >= 0.65
        and boundary["permutation_p_two_sided"] < 0.05
    )
    literal_refuting = sorted(
        name for name, row in guarded.items()
        if row["auc"] >= 0.65 and row["permutation_p_two_sided"] < 0.05
    )
    powered_refuting = sorted(name for name in literal_refuting if name in powered)
    if confirmed and literal_refuting:
        verdict = "MIXED"
    elif confirmed:
        verdict = "CONFIRMED"
    elif literal_refuting:
        verdict = "REFUTED"
    else:
        verdict = "MIXED"
    return {
        "verdict": verdict,
        "powered_axes": sorted(powered),
        "underpowered_axes_excluded_from_confirmed": excluded,
        "refuted_guarded_axes_literal": literal_refuting,
        "refuted_guarded_axes_with_power_gate": powered_refuting,
        "refuted_power_ambiguity_resolution": REFUTED_AMBIGUITY,
        "verdict_overlap_resolution": VERDICT_OVERLAP_RESOLUTION,
    }


def overall_verdict(stage_verdicts: Sequence[str]) -> str:
    if not stage_verdicts:
        return "MIXED"
    return stage_verdicts[0] if len(set(stage_verdicts)) == 1 else "MIXED"


def guarded_stage3_rows(
    score_point: Callable[[Any, torch.Tensor], dict[str, Any]],
    points: Sequence[Any],
    val_te: torch.Tensor,
    test_te: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The only Stage-3 scoring route; reject aliases and pass val_te to every point."""
    if val_te.data_ptr() == test_te.data_ptr() or torch.equal(val_te, test_te):
        raise AssertionError("Stage 3 val_te aliases #86 test_te")
    rows = []
    for point in points:
        scored = score_point(point, val_te)
        sets = extract_row_sets(scored["base_correct"], scored["candidate_correct"])
        rows.append({
            "point": point,
            "gains": sets["gains"].tolist(),
            "regressions": sets["regressions"].tolist(),
            "gain_count": len(sets["gains"]),
            "regression_count": len(sets["regressions"]),
        })
    return rows, pool_point_rows(rows)


def _teacher_consensus(gated_rescue, split: torch.Tensor) -> torch.Tensor:
    raw_positions = gated_rescue._raw_alignment()[split.cpu()]
    _agree70, largest_cluster = gated_rescue._load_teacher_overlay(raw_positions)
    return largest_cluster


def _b0_conf_fns(gated_rescue, rules) -> dict[str, Any]:
    out = {name: gated_rescue.v5.CONF_FNS[name] for name, _ in rules
           if name in gated_rescue.v5.CONF_FNS}
    for name, fn in rules:
        if name not in out:
            scalar = float(gated_rescue.v5.RULE_CONF.get(name, 0.0))
            out[name] = lambda w, fn=fn, scalar=scalar: (
                fn(w), torch.full((len(w),), scalar, device=w.device)
            )
    return out


def _stage1_and_2() -> tuple[dict[str, Any], dict[str, Any]]:
    from experiments import campaign_wikitext_gated_ra as gated_ra
    from experiments import campaign_wikitext_gated_rescue as gated_rescue

    model, rules, frozen_core, _ = gated_rescue.load_frozen_b0(gated_rescue.B0_FREEZE)
    if not gated_rescue.b0_within_registered_band(frozen_core["agree"]):
        raise RuntimeError("STOP: frozen B0 fails registered band")
    ids, y, cls, uv, tr, te = gated_rescue.v5.load_ds()
    yv = cls[y]
    val_te, test_te = gated_rescue.split_te(te)
    active, exclusions = gated_rescue.active_candidate_families([name for name, _ in rules])
    counts_fn = gated_rescue._counts_row_fn(model, cls)
    conf_fns = _b0_conf_fns(gated_rescue, rules)
    b0 = gated_rescue.WylyBlock(0, "B0")
    b0.rules, b0.conf_fns = list(rules), conf_fns
    _, _, tr_trace = core_cover_sw_runner_traced(model, rules, ids, yv, cls, tr)
    test_trace = assert_runner_trace_parity(model, rules, ids, yv, cls, test_te)
    val_trace = assert_runner_trace_parity(model, rules, ids, yv, cls, val_te)
    val_conf = val_trace["conf"]
    taus = gated_rescue.tau_deciles(val_conf)

    tau_runs, fitted = [], []
    for tau in taus:
        candidates, low_train, mining_log = gated_rescue.fit_candidate_pool(
            model, ids, yv, cls, uv, tr, tr_trace["conf"], tau, val_te, test_te, active
        )
        stack, admitted, marginals, agree = gated_rescue._run_gated_tau(
            rules, conf_fns, candidates, counts_fn, ids, yv, val_te, tau
        )
        tau_runs.append({
            "tau": tau, "val_agree": agree, "admitted": admitted,
            "marginals": marginals, "mined_train_rows": len(low_train),
            "mining_log": mining_log,
        })
        fitted.append((candidates, stack))
    best = min(range(len(taus)), key=lambda i: (-tau_runs[i]["val_agree"], tau_runs[i]["tau"]))
    chosen = tau_runs[best]
    chosen_candidates, stack = fitted[best]
    flat, _, _ = gated_rescue._run_flat(
        model, rules, conf_fns, chosen_candidates, counts_fn, ids, yv, val_te
    )
    base_pred, _ = b0.predict_cover(ids[test_te], counts_row_fn=counts_fn)
    flat_pred, _ = flat.predict_cover(ids[test_te], counts_row_fn=counts_fn)
    stack.forward(ids[test_te], counts_row_fn=counts_fn)
    gated_pred = stack.last_carried[-1].pred
    target = yv[test_te]
    base_correct, flat_correct, gated_correct = (
        base_pred == target, flat_pred == target, gated_pred == target
    )
    row_sets = extract_row_sets(base_correct, gated_correct)
    if not voting_eligible(len(row_sets["gains"]), len(row_sets["regressions"])):
        raise RuntimeError("STOP: Stage 1 failed the registered >=20 per-class floor")
    axes = _axis_values(test_trace, _teacher_consensus(gated_rescue, test_te))
    stats = analyze_row_sets(axes, **row_sets, seed=100)
    stage1_verdict = resolve_verdict(stats)
    crosscheck = discordance_crosscheck(base_correct, flat_correct, gated_correct)
    stage1 = {
        "name": "Stage 1 / #81 GATED1 on test_te",
        "votes": True,
        "counts": {name: len(rows) for name, rows in row_sets.items()},
        "chosen_tau": chosen["tau"],
        "admitted": chosen["admitted"],
        "tau_runs": tau_runs,
        "active_candidate_families": list(active),
        "pool_exclusions": exclusions,
        "crosscheck_with_81_flat_vs_gated": crosscheck,
        "registered_81_flat_vs_gated_reference": {"b": 59, "c": 135},
        "crosscheck_note": (
            "#81 b/c compare FLAT with GATED1, while these gain/regression classes compare B0 "
            "with GATED1; the overlap decomposition explains why equality is not asserted."
        ),
        "decile_reference": "B0 winner confidence on Stage 1 test_te",
        "teacher_consensus_definition": "largest agreeing cluster among the eight teachers (0..8)",
        "statistics": stats,
        **stage1_verdict,
    }

    keyed_candidates, _, _ = gated_ra.fit_candidate_pool_with_keys(
        model, ids, yv, cls, uv, tr, tr_trace["conf"], chosen["tau"],
        val_te, test_te, active,
    )
    b1_pred, b1_conf, winner, key = gated_ra.trace_b1_cover(
        keyed_candidates, chosen["admitted"], ids[val_te]
    )
    b0_pred_val, _ = b0.predict_cover(ids[val_te], counts_row_fn=counts_fn)
    attribution = gated_ra.attribution_counts(
        b0_pred_val, yv[val_te], b1_pred, b1_conf, winner, key, chosen["tau"]
    )
    cell = torch.tensor(
        [name == "pointer" for name in winner], device=key.device
    ) & (key == 18)
    claimed_cell = cell & attribution["claim"]
    gains = (claimed_cell & attribution["gains"]).nonzero(as_tuple=False).flatten().cpu()
    regressions = (
        (claimed_cell & attribution["regressions"]).nonzero(as_tuple=False).flatten().cpu()
    )
    untouched = (
        claimed_cell & ~(attribution["gains"] | attribution["regressions"])
    ).nonzero(
        as_tuple=False
    ).flatten().cpu()
    cell_counts = attribution["counts"].get(
        ("pointer", 18), {"claimed": 0, "val_gains": 0, "val_regressions": 0}
    )
    stage2_stats = None
    if len(gains) and len(regressions) and len(untouched):
        axes2 = _axis_values(val_trace, _teacher_consensus(gated_rescue, val_te))
        stage2_stats = analyze_row_sets(
            axes2, gains, regressions, untouched, seed=200
        )
    stage2 = {
        "name": "Stage 2 / #83 pointer key=18 on val_te",
        "votes": False,
        "exclusion_reason": (
            "REGISTERED EXCLUSION, not a floor failure: descriptive only because this duplicates "
            "Stage 1's mechanism at finer key granularity."
        ),
        "counts": {
            **cell_counts,
            "cell_untouched": len(untouched),
            "global_gain_rows": val_te[gains].cpu().tolist(),
            "global_regression_rows": val_te[regressions].cpu().tolist(),
        },
        "decile_reference": "B0 winner confidence on Stage 2 val_te",
        "teacher_consensus_definition": "largest agreeing cluster among the eight teachers (0..8)",
        "statistics": stage2_stats,
        "verdict": "DESCRIPTIVE ONLY (REGISTERED EXCLUSION)",
    }
    return stage1, stage2


def _artifact_points(frames_at_scale) -> list[tuple[int, float]]:
    artifact = json.loads(frames_at_scale.OUTPUT.read_text())
    return [
        (int(row["support_gate"]), float(row["interaction_gate"]))
        for row in artifact["scales"]["2.8b"]["grid"]
        if row["marginal"] >= frames_at_scale.ADMIT_THRESH
    ]


def _stage3_child(destination: Path) -> int:
    # Import order matters: frames binds the 2.8b v5 configuration before importing 70m helpers.
    from experiments import campaign_frames_at_scale as frames_at_scale
    from experiments import campaign_wikitext_gated_rescue as gated_rescue

    if os.environ.get("WYLY_SEED") != "0":
        raise AssertionError("Stage 3 requires WYLY_SEED=0")
    historical_points = _artifact_points(frames_at_scale)
    model, captured_rules, _ = frames_at_scale._regenerate()
    ids, y, cls, uv, tr, te = frames_at_scale.v5.load_ds()
    yv = cls[y]
    val_te, test_te = frames_at_scale.split_te(te)
    fit = frames_at_scale._fit_region(tr)
    baseline_rules = [rule for rule in captured_rules if not rule[0].startswith("mined frames")]
    mined_by_point, score_by_point, fresh_points = {}, {}, []
    for support, interaction in frames_at_scale.registered_grid():
        mined, _ = frames_at_scale._mine_once(
            model, baseline_rules, ids, yv, cls, uv, fit, support, interaction
        )
        scored = frames_at_scale._score_candidate(
            model, baseline_rules, mined, ids, yv, cls, val_te
        )
        point = (support, interaction)
        mined_by_point[point] = mined
        score_by_point[point] = scored
        if scored["marginal"] >= frames_at_scale.ADMIT_THRESH:
            fresh_points.append(point)

    def score_point(point, idxs):
        if idxs.data_ptr() != val_te.data_ptr():
            raise AssertionError("Stage 3 attempted to score anything other than val_te")
        return score_by_point[point]

    point_rows, pooled = guarded_stage3_rows(
        score_point, fresh_points, val_te, test_te
    )
    base_trace = assert_runner_trace_parity(
        model, baseline_rules, ids, yv, cls, val_te
    )
    axes = _axis_values(base_trace, _teacher_consensus(gated_rescue, val_te))
    changed = set(pooled["gains"]) | set(pooled["regressions"])
    untouched = sorted(set(range(len(val_te))) - changed)
    votes = voting_eligible(pooled["post_dedup_gains"], pooled["post_dedup_regressions"])
    statistics = None
    verdict: dict[str, Any] = {"verdict": "DESCRIPTIVE ONLY (FLOOR FAILURE)"}
    if pooled["post_dedup_gains"] and pooled["post_dedup_regressions"] and untouched:
        statistics = analyze_row_sets(
            axes, pooled["gains"], pooled["regressions"], untouched, seed=300
        )
        if votes:
            verdict = resolve_verdict(statistics)
    stage3 = {
        "name": "Stage 3 / #86 judge-clearing 2.8b points on val_te",
        "votes": votes,
        "floor_outcome": (
            "votes: both post-dedup classes >=20" if votes else
            "descriptive only: at least one post-dedup class <20"
        ),
        "historical_artifact_points": historical_points,
        "fresh_regeneration_points": fresh_points,
        "point_set_drift": historical_points != fresh_points,
        "point_rows": point_rows,
        "counts": {**{key: value for key, value in pooled.items() if not isinstance(value, list)},
                   "untouched": len(untouched)},
        "decile_reference": "B0 winner confidence on Stage 3 2.8b val_te",
        "teacher_consensus_definition": "largest agreeing cluster among the eight teachers (0..8)",
        "structural_guard": "Only val_te is accepted by score_point; #86 test_te is never scored",
        "statistics": statistics,
        **verdict,
    }
    destination.write_text(json.dumps(stage3, indent=2, sort_keys=True))
    return 0


def _run_stage3_subprocess() -> dict[str, Any]:
    env = dict(os.environ)
    env.update({
        "WYLY_TAG": "pythia2.8b", "WYLY_DS": "wikitext", "WYLY_LIB": "mined",
        "WYLY_JUDGE": "cover", "WYLY_ONLINE": "1", "WYLY_COVER": "sw", "WYLY_SEED": "0",
    })
    for key in (
        "WYLY_CONCEPTS", "WYLY_POINTER", "WYLY_TPOINTER", "WYLY_DX", "WYLY_CX", "WYLY_FOLDS"
    ):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="frontier-rows-") as temporary:
        destination = Path(temporary) / "stage3.json"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--stage3-child", str(destination)],
            env=env, check=False,
        )
        if completed.returncode != 0 or not destination.exists():
            raise RuntimeError(f"Stage 3 subprocess failed with exit {completed.returncode}")
        return json.loads(destination.read_text())


def _print_stage(stage: dict[str, Any]) -> None:
    log(f"=== {stage['name']} ===")
    log("counts: " + json.dumps(stage["counts"], sort_keys=True))
    if stage.get("statistics"):
        log("axis\tAUC\tp(two-sided perm)\tbootstrap 95% CI\tpowered@0.65")
        for name, row in stage["statistics"].items():
            log(
                f"{name}\t{row['auc']:.6f}\t{row['permutation_p_two_sided']:.6g}\t"
                f"[{row['bootstrap_95_ci'][0]:.6f}, {row['bootstrap_95_ci'][1]:.6f}]\t"
                f"{row['powered_for_auc_0.65']}"
            )
            log(f"  power note: {row['power_heuristic']} (CI width={row['ci_width']:.6f})")
    log("VERDICT: " + stage["verdict"])
    if not stage["votes"]:
        log(stage.get("exclusion_reason", stage.get("floor_outcome", "descriptive only")))


def _contract_corollary(overall: str, voting: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if overall == "CONFIRMED":
        regressions = sum(stage["counts"].get(
            "regressions", stage["counts"].get("post_dedup_regressions", 0)
        ) for stage in voting)
        gains = sum(stage["counts"].get(
            "gains", stage["counts"].get("post_dedup_gains", 0)
        ) for stage in voting)
        domains = ["pythia70m/wikitext" if "Stage 1" in row["name"] else
                   "pythia2.8b/wikitext" for row in voting]
        return {"regression_budget": {
            "rate": regressions / (regressions + gains),
            "rate_definition": "deduplicated regressions / (deduplicated gains + regressions)",
            "eval_unit": "held-out split row",
            "measured_on": [row["name"] for row in voting],
            "sweep@sha": "#81@3c5a96e + #86@9681595 (data/frames_at_scale.json)",
            "domain": domains[0] if len(set(domains)) == 1 else domains,
        }}
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-child", type=Path)
    args = parser.parse_args(argv)
    if args.stage3_child:
        return _stage3_child(args.stage3_child)

    stage1, stage2 = _stage1_and_2()
    stage3 = _run_stage3_subprocess()
    stages = [stage1, stage2, stage3]
    voting = [stage for stage in stages if stage["votes"]]
    overall = overall_verdict([stage["verdict"] for stage in voting])
    corollary = _contract_corollary(overall, voting)
    report = {
        "honesty_note": HONESTY_NOTE,
        "runner_gap_sentinel": NO_RUNNER_CONF,
        "runner_gap_sentinel_note": (
            "runner_conf=-1e9 when no real competitor fired; the large gap is categorical "
            "maximal confidence/no competitor"
        ),
        "statistics": {
            "auc_orientation": (
                "guarded axes: positive=GAIN, negative=REGRESSION; boundary gap: "
                "positive=UNTOUCHED, negative=GAIN-union-REGRESSION because frontier rows are "
                "registered as the lower-gap class"
            ),
            "permutations": PERMUTATIONS, "permutation_p": "two-sided around AUC=0.5",
            "bootstraps": BOOTSTRAPS, "bootstrap_ci": "percentile 95%",
            "power_heuristic": "recenter observed CI width at 0.65 and ask whether it excludes 0.5",
        },
        "refuted_power_ambiguity": REFUTED_AMBIGUITY,
        "verdict_overlap_resolution": VERDICT_OVERLAP_RESOLUTION,
        "boundary_auc_direction_resolution": (
            "The clause says changed-vs-untouched AUC but the hypothesis predicts lower changed "
            "gaps. It is oriented as P(gap_untouched > gap_changed), so >=0.65 supports the "
            "stated decision-boundary hypothesis."
        ),
        "stages": stages,
        "overall_verdict": overall,
        "contract_corollary": corollary,
    }
    for stage in stages:
        _print_stage(stage)
    log("=== OVERALL ===")
    log("OVERALL VERDICT: " + overall)
    if overall == "CONFIRMED":
        log("RECOMMENDATION ONLY (not built): " + json.dumps(corollary, sort_keys=True))
    elif overall == "REFUTED":
        axes = sorted({axis for stage in voting
                       for axis in stage.get("refuted_guarded_axes_literal", [])})
        log("discriminating axis/axes: " + ", ".join(axes))
        log("Live follow-ups: multi-episode mining sweeps and the existing WYLY_STRATA gate.")
    else:
        log("No recommendation is made.")
    log("honesty note: " + HONESTY_NOTE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    log(f"JSON: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
