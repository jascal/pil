"""Key-retreat admission follow-up to the one-sided Wikitext rescue campaign (#83).

The fitted candidate objects are unchanged.  This module adds tracing orchestration so validation
regressions can be attributed to literal candidate-table keys, then masks those keys and repeats
the registered one-sided admission pass.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from experiments import campaign_wikitext_gated_rescue as campaign  # noqa: E402
from experiments.campaign_wikitext_gated2 import (  # noqa: E402
    EXPECTED_FLAT_TEST_AGREE,
    EXPECTED_GATED1_REGRESSIONS,
    EXPECTED_GATED1_TAU,
    EXPECTED_GATED1_TEST_AGREE,
    EXPECTED_TAU_LOW_GRID,
    REPRO_TOLERANCE,
    TAU_TOLERANCE,
    flat_gate_passes,
    resolve_operating_point,
    select_operating_point,
)

B0_FREEZE = campaign.B0_FREEZE
load_frozen_b0 = campaign.load_frozen_b0
split_te = campaign.split_te
tau_deciles = campaign.tau_deciles
core_cover_sw_traced = campaign.core_cover_sw_traced
active_candidate_families = campaign.active_candidate_families
_run_flat = campaign._run_flat
_run_gated_tau = campaign._run_gated_tau
_counts_row_fn = campaign._counts_row_fn
b0_within_registered_band = campaign.b0_within_registered_band
CANDIDATE_FAMILIES = campaign.CANDIDATE_FAMILIES
ADMIT_THRESH = campaign.ADMIT_THRESH
MAX_RULES = campaign.MAX_RULES
exact_discordant_p = campaign.exact_discordant_p
WylyBlock = campaign.WylyBlock
BlockStack = campaign.BlockStack
v5 = campaign.v5

OUTPUT = REPO / "data" / "wikitext_gated_ra.json"
GATED2_OUTPUT = REPO / "data" / "wikitext_gated2.json"
ABSTAIN_CONF = -1e9
HONESTY_NOTE = (
    "val_te used three ways in this run — key bans, admission marginals, and operating-point "
    "selection — over a single split; a single test_te evaluation; candidate families remain "
    "template_fixed (not learned end-to-end); frac_induced is unaffected by this experiment."
)
PARTIAL_PREDICTION = (
    "50-80% coverage predicts PARTIAL-fail at the reference point (~12-31 residual "
    "regressions); clause-(iv) pass needs ~95%; the coverage number is itself the finding."
)
NEGATIVE_FINDING = (
    "regressions are row-idiosyncratic at key granularity; structural retreat cannot fix them"
)
VERDICT_SCOPE = "families with meaningful key granularity (pointer cells, kgram suffixes, frame anchors)"
DSTATE_SCOPE = (
    "dstate's ~10-value keyspace degenerates retreat to family on/off — its rows reported but "
    "excluded from the verdict reading"
)

Candidate = tuple[str, Callable[[torch.Tensor], torch.Tensor], Callable[..., Any], Callable[..., Any]]


def log(message: str = "") -> None:
    print(message, flush=True)


def _write(report: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    log(f"JSON: {OUTPUT}")


def _public(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not isinstance(value, torch.Tensor)}


def _metrics(pred: torch.Tensor, target: torch.Tensor, base: torch.Tensor) -> dict[str, Any]:
    correct = pred == target
    base_correct = base == target
    return {
        "agree": float(correct.float().mean()),
        "regressions": int((base_correct & ~correct).sum()),
        "gains": int((~base_correct & correct).sum()),
        "correct": correct,
    }


def _b0_conf_fns(rules) -> dict[str, Any]:
    out = {name: v5.CONF_FNS[name] for name, _ in rules if name in v5.CONF_FNS}
    for name, fn in rules:
        if name not in out:
            scalar = float(v5.RULE_CONF.get(name, 0.0))
            out[name] = lambda w, fn=fn, scalar=scalar: (
                fn(w), torch.full((len(w),), scalar, device=w.device)
            )
    return out


def _fit_table_candidate_with_key(name, ids, yv, tr, feature, vocab) -> Candidate:
    table, _ = v5.fit_dgate_feature(ids, yv, tr, feature, vocab)
    base = vocab + 2

    def evaluate(w):
        feat = feature(w)
        key = (feat + 1) * base + w[:, -1]
        value, confidence = table.lookup_conf(key)
        miss = feat < 0
        value = torch.where(miss, torch.full_like(value, -1), value)
        confidence = torch.where(miss, torch.full_like(confidence, ABSTAIN_CONF), confidence)
        return value, confidence, torch.where(value >= 0, key, torch.full_like(key, -1))

    return name, lambda w: evaluate(w)[0], lambda w: evaluate(w)[:2], lambda w: evaluate(w)[2]


def _mined_evaluate(mined, w):
    """Exact lookup_conf_all arbitration with its winning composite key retained."""
    best_v = torch.full((len(w),), -1, dtype=torch.long, device=w.device)
    best_c = torch.full((len(w),), ABSTAIN_CONF, device=w.device)
    best_k = torch.full((len(w),), -1, dtype=torch.long, device=w.device)
    for o in mined.offs:
        key = (o * mined.B + mined.A(w[:, -o])) * mined.B + w[:, -1]
        value, confidence = mined.t1.lookup_conf(key)
        take = (value >= 0) & (confidence > best_c)
        best_v = torch.where(take, value, best_v)
        best_c = torch.where(take, confidence, best_c)
        best_k = torch.where(take, key, best_k)
    for o1, o2 in mined.pair_offs:
        key = (
            (o1 * 16 + o2) * mined.B**2
            + mined.A(w[:, -o1]) * mined.B
            + mined.A(w[:, -o2])
        ) * mined.B + w[:, -1]
        value, confidence = mined.t2.lookup_conf(key)
        take = (value >= 0) & (confidence > best_c)
        best_v = torch.where(take, value, best_v)
        best_c = torch.where(take, confidence, best_c)
        best_k = torch.where(take, key, best_k)
    return best_v, best_c, best_k


def fit_candidate_pool_with_keys(
    model, ids, yv, cls, uv, tr, tr_conf, tau, val_te, test_te, active_families
):
    """Use #81's fitting primitives and return each candidate's literal firing key."""
    vocab = len(uv)
    candidates: list[Candidate] = []
    if "pointer" in active_families:
        pointer = v5.PointerRule(vocab, v5.DEV)
        pointer.refresh(ids, yv, tr)

        def pointer_evaluate(w):
            pred, confidence = pointer.lookup_conf(w)
            _, length, concept_length, has = pointer._feats(w)
            key = length * (pointer.lmax + 2) + concept_length
            key = torch.where(has & (pred >= 0), key, torch.full_like(key, -1))
            return pred, confidence, key

        candidates.append(
            (
                "pointer",
                lambda w: pointer_evaluate(w)[0],
                lambda w: pointer_evaluate(w)[:2],
                lambda w: pointer_evaluate(w)[2],
            )
        )

    sent, quote, attr, _ = campaign._member_sets(uv, vocab)
    if "dstate gate" in active_families:
        def dstate(w):
            return v5.dstate_feature(w, sent, quote)

        candidates.append(
            _fit_table_candidate_with_key("dstate gate", ids, yv, tr, dstate, vocab)
        )

    if "sentpair/dgate2" in active_families:
        def senthead_pos(w):
            return v5.recent_member_pos(w, sent)

        def previous_head(w):
            return v5.at_pos(w, v5.prev_occ_pos(w, senthead_pos(w)), succ=1)

        def current_head(w):
            return v5.at_pos(w, senthead_pos(w), succ=1)

        table, _ = v5.fit_dgate2(ids, yv, tr, previous_head, current_head, vocab)
        base = vocab + 2

        def pair_evaluate(w):
            fa, fb = previous_head(w), current_head(w)
            valid = (fa >= 0) & (fb >= 0)
            key = (fa + 1) * base + fb
            value, confidence = table.lookup_conf(key)
            value = torch.where(valid, value, torch.full_like(value, -1))
            confidence = torch.where(
                valid, confidence, torch.full_like(confidence, ABSTAIN_CONF)
            )
            key = torch.where(value >= 0, key, torch.full_like(key, -1))
            return value, confidence, key

        candidates.append(
            (
                "sentpair/dgate2",
                lambda w: pair_evaluate(w)[0],
                lambda w: pair_evaluate(w)[:2],
                lambda w: pair_evaluate(w)[2],
            )
        )

    if "attrib-subj gate" in active_families:
        def attrib(w):
            return v5.at_pos(w, v5.recent_member_pos(w, attr), succ=-1)

        candidates.append(
            _fit_table_candidate_with_key("attrib-subj gate", ids, yv, tr, attrib, vocab)
        )

    low_train = campaign.mining_fit_slice(tr, tr_conf, tau, val_te, test_te)
    mining_log: list[str] = []
    if "mined frames" in active_families:
        mined = v5.MinedGates(vocab, v5.DEV)
        if len(low_train):
            mined.mine(
                model, ids, yv, cls, low_train, torch.Generator().manual_seed(0),
                mining_log, 0, sample_n=min(6000, len(low_train)),
            )
        candidates.append(
            (
                "mined frames",
                lambda w: _mined_evaluate(mined, w)[0],
                lambda w: _mined_evaluate(mined, w)[:2],
                lambda w: _mined_evaluate(mined, w)[2],
            )
        )
    assert tuple(row[0] for row in candidates) == tuple(active_families)
    return candidates, low_train, mining_log


def strip_keys(candidates: Iterable[Candidate]):
    return [(name, fn, conf_fn) for name, fn, conf_fn, _ in candidates]


def trace_b1_cover(candidates: Iterable[Candidate], admitted: Iterable[str], w):
    """Mirror SW consider() for admitted B1 rules only; no counts tier."""
    by_name = {candidate[0]: candidate for candidate in candidates}
    pred = torch.full_like(w[:, -1], -1)
    conf = torch.full((len(w),), ABSTAIN_CONF, device=w.device)
    key = torch.full_like(w[:, -1], -1)
    winner = ["abstain"] * len(w)
    for name in admitted:
        _, _, conf_fn, key_fn = by_name[name]
        value, confidence = conf_fn(w)
        candidate_key = key_fn(w)
        take = (value >= 0) & (confidence > conf)
        pred = torch.where(take, value, pred)
        conf = torch.where(take, confidence, conf)
        key = torch.where(take, candidate_key, key)
        for index in take.nonzero(as_tuple=False).flatten().cpu().tolist():
            winner[index] = name
    return pred, conf, winner, key


def attribution_counts(
    b0_pred, target, b1_pred, b1_conf, winner: list[str], key, tau: float
) -> dict[str, Any]:
    """Attribute claimed one-sided gains/regressions to (winning rule, literal key)."""
    claim = (b1_pred >= 0) & (b1_conf > tau)
    base_correct = b0_pred == target
    b1_correct = b1_pred == target
    gains = claim & ~base_correct & b1_correct
    regressions = claim & base_correct & ~b1_correct
    counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"claimed": 0, "val_gains": 0, "val_regressions": 0}
    )
    for index in claim.nonzero(as_tuple=False).flatten().cpu().tolist():
        item = (winner[index], int(key[index]))
        if item[0] == "abstain" or item[1] < 0:
            raise AssertionError("claimed B1 row lacks a winning rule/key")
        counts[item]["claimed"] += 1
        counts[item]["val_gains"] += int(gains[index])
        counts[item]["val_regressions"] += int(regressions[index])
    return {
        "claim": claim,
        "gains": gains,
        "regressions": regressions,
        "counts": dict(counts),
        "regression_attributions": [
            (winner[index], int(key[index]))
            for index in regressions.nonzero(as_tuple=False).flatten().cpu().tolist()
        ],
    }


def ban_criterion(val_gains: int, val_regressions: int) -> bool:
    return val_regressions >= val_gains and val_regressions >= 1


def compute_ban_set(counts: dict[tuple[str, int], dict[str, int]]) -> dict[str, set[int]]:
    bans: dict[str, set[int]] = defaultdict(set)
    for (rule, key), row in sorted(counts.items()):
        if ban_criterion(row.get("val_gains", 0), row.get("val_regressions", 0)):
            bans[rule].add(key)
    return dict(bans)


def val_to_test_coverage(
    bans: dict[str, set[int]], test_regression_attributions: Iterable[tuple[str, int]]
) -> float:
    rows = list(test_regression_attributions)
    if not rows:
        return 0.0
    covered = sum(key in bans.get(rule, set()) for rule, key in rows)
    return covered / len(rows)


def retreat_candidate(candidate: Candidate, banned_keys: set[int]) -> Candidate:
    name, fn, conf_fn, key_fn = candidate

    def mask(w):
        key = key_fn(w)
        if not banned_keys:
            return torch.zeros_like(key, dtype=torch.bool)
        banned = torch.tensor(sorted(banned_keys), device=key.device, dtype=key.dtype)
        return torch.isin(key, banned)

    def wrapped_fn(w):
        value = fn(w)
        return torch.where(mask(w), torch.full_like(value, -1), value)

    def wrapped_conf(w):
        value, confidence = conf_fn(w)
        banned = mask(w)
        return (
            torch.where(banned, torch.full_like(value, -1), value),
            torch.where(banned, torch.full_like(confidence, ABSTAIN_CONF), confidence),
        )

    def wrapped_key(w):
        key = key_fn(w)
        return torch.where(mask(w), torch.full_like(key, -1), key)

    return name, wrapped_fn, wrapped_conf, wrapped_key


def gate_part_b(coverage: float, run_part_b: Callable[[], Any]) -> Any:
    """Structural Rule-1 guard: the callback may be the only route into Part B/test state."""
    log(PARTIAL_PREDICTION)
    if coverage < 0.50:
        log(NEGATIVE_FINDING)
        return None
    return run_part_b()


def _serial_counts(counts):
    return [
        {"rule": rule, "key": key, **row}
        for (rule, key), row in sorted(counts.items())
    ]


def _serial_bans(bans):
    return {name: sorted(keys) for name, keys in sorted(bans.items())}


def _family_set(names):
    return set(names) & set(CANDIDATE_FAMILIES)


def _cardinality(attribution, names=CANDIDATE_FAMILIES):
    out = {}
    for name in names:
        seen = {key for (rule, key) in attribution["counts"] if rule == name}
        regression = {
            key for rule, key in attribution["regression_attributions"] if rule == name
        }
        out[name] = {"claimed_keys": len(seen), "regression_keys": len(regression)}
    return out


def _static_zero_regression_reference():
    if not GATED2_OUTPUT.exists():
        return None
    payload = json.loads(GATED2_OUTPUT.read_text())
    rows = [row for row in payload.get("gated2", {}).get("frontier", []) if row["val_regressions"] == 0]
    return max(rows, key=lambda row: row["val_agree"]) if rows else None


def main() -> int:
    started = time.perf_counter()
    report: dict[str, Any] = {
        "honesty_note": HONESTY_NOTE,
        "key_design": {
            "pointer": "l * (pointer.lmax + 2) + lc",
            "dstate gate": "(feature + 1) * (vocab + 2) + last token (full table key)",
            "sentpair/dgate2": "(previous_head + 1) * (vocab + 2) + current_head",
            "attrib-subj gate": "(feature + 1) * (vocab + 2) + last token (full table key)",
            "mined frames": "winning singleton/pair MinedGates composite table key",
            "kgram": "inapplicable: frozen B0 family, not a B1 candidate",
        },
        "gaps_or_ambiguities": [
            "The registered scope phrase mentions kgram suffixes, but kgram is B0-only here.",
            "Dstate uses the requested full composite table-entry key; feature-only "
            "cardinality is reported separately.",
        ],
    }
    log("=== B0′ LOAD + REGISTERED DEFENSIVE CHECK ===")
    if not B0_FREEZE.exists():
        report["stopped"] = f"frozen B0′ artifact missing: {B0_FREEZE}"
        log("STOP: " + report["stopped"])
        _write(report)
        return 2
    try:
        model, rules, frozen_core, frozen_wall = load_frozen_b0(B0_FREEZE)
    except Exception as exc:
        report["stopped"] = f"frozen B0′ load failed: {type(exc).__name__}: {exc}"
        log("STOP: " + report["stopped"])
        _write(report)
        return 2
    b0_band = b0_within_registered_band(frozen_core["agree"])
    report["b0"] = {
        "artifact": str(B0_FREEZE), "source": "loaded; never regenerated",
        "original_fit_wall_seconds": frozen_wall, "core_sw": frozen_core,
        "rules": [name for name, _ in rules], "registered_band_pass": b0_band,
    }
    log(f"artifact: {B0_FREEZE}")
    log("source: LOADED frozen B0′; regeneration forbidden and not attempted")
    log(f"frozen core_sw: {json.dumps(frozen_core, sort_keys=True)}")
    log(f"B0′ registered band: {b0_band}")
    if not b0_band:
        report["stopped"] = "defensive B0′ registered-band check failed"
        log("STOP: " + report["stopped"])
        _write(report)
        return 2

    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    val_te, test_te = split_te(te)
    active, exclusions = active_candidate_families([name for name, _ in rules])
    counts_fn = _counts_row_fn(model, cls)
    b0_conf_fns = _b0_conf_fns(rules)
    b0 = WylyBlock(0, "B0")
    b0.rules, b0.conf_fns = list(rules), b0_conf_fns
    b0_pred_val, b0_conf_val = b0.predict_cover(ids[val_te], counts_row_fn=counts_fn)
    b0_pred_test, _ = b0.predict_cover(ids[test_te], counts_row_fn=counts_fn)
    b0_pred_te, _ = b0.predict_cover(ids[te], counts_row_fn=counts_fn)
    traced, traced_pred, trace = core_cover_sw_traced(model, rules, ids, yv, cls, te)
    if not torch.equal(b0_pred_te, traced_pred):
        raise AssertionError("direct B0 prediction differs from traced core cover")
    _, _, tr_trace = core_cover_sw_traced(model, rules, ids, yv, cls, tr)
    val_pos = torch.searchsorted(te, val_te)
    if not torch.equal(trace["conf"][val_pos], b0_conf_val):
        raise AssertionError("B0 val confidence differs from traced extraction")
    taus = tau_deciles(b0_conf_val)
    grid_matches = len(taus) == len(EXPECTED_TAU_LOW_GRID) and all(
        abs(a - b) <= TAU_TOLERANCE for a, b in zip(taus, EXPECTED_TAU_LOW_GRID, strict=True)
    )
    report["split"] = {"seed": 0, "val_te": len(val_te), "test_te": len(test_te)}
    report["pool"] = {"active_families": list(active), "exclusions": exclusions}
    report["tau_low_grid"] = taus
    report["tau_low_grid_matches_registered_literals"] = grid_matches
    log(f"B0′ rules: {[name for name, _ in rules]}")
    log(f"traced/direct prediction parity on te: {torch.equal(b0_pred_te, traced_pred)}")
    log(f"split: val_te={len(val_te)} test_te={len(test_te)} seed=0")
    log(f"active B1 families: {list(active)}")
    log(f"tau grid: {taus}")
    log(f"tau grid matches registered literal values: {grid_matches}")
    if not grid_matches:
        report["stopped"] = "B0 val-confidence tau grid drifted from registered values"
        log("STOP: " + report["stopped"])
        _write(report)
        return 2

    log("=== PART A / STEP 0: FLAT + GATED1 REPRODUCTION ===")
    step0, fitted = [], []
    for tau_index, tau in enumerate(taus):
        candidates, low_train, mining_log = fit_candidate_pool_with_keys(
            model, ids, yv, cls, uv, tr, tr_trace["conf"], tau, val_te, test_te, active
        )
        stack, admitted, marginals, val_agree = _run_gated_tau(
            rules, b0_conf_fns, strip_keys(candidates), counts_fn, ids, yv, val_te, tau
        )
        row = {
            "tau_index": tau_index, "tau": tau, "val_agree": val_agree,
            "admitted": admitted, "marginals": marginals,
            "mined_train_rows": len(low_train), "mining_log_rows": len(mining_log),
        }
        step0.append(row)
        fitted.append((candidates, stack))
        log(
            f"GATED1 tau={tau:.9f} val_agree={val_agree:.9f} admitted={admitted} "
            f"mined_train={len(low_train)}"
        )
    best_index = min(range(10), key=lambda i: (-step0[i]["val_agree"], step0[i]["tau"]))
    chosen = step0[best_index]
    chosen_candidates, chosen_stack = fitted[best_index]
    flat, flat_admitted, flat_marginals = _run_flat(
        model, rules, b0_conf_fns, strip_keys(chosen_candidates), counts_fn, ids, yv, val_te
    )
    flat_pred_val, _ = flat.predict_cover(ids[val_te], counts_row_fn=counts_fn)
    flat_pred_test, _ = flat.predict_cover(ids[test_te], counts_row_fn=counts_fn)
    chosen_stack.forward(ids[test_te], counts_row_fn=counts_fn)
    gated1_pred_test = chosen_stack.last_carried[-1].pred
    flat_val = _metrics(flat_pred_val, yv[val_te], b0_pred_val)
    flat_test = _metrics(flat_pred_test, yv[test_te], b0_pred_test)
    gated1_test = _metrics(gated1_pred_test, yv[test_te], b0_pred_test)
    reproduced = (
        abs(flat_test["agree"] - EXPECTED_FLAT_TEST_AGREE) <= REPRO_TOLERANCE
        and abs(gated1_test["agree"] - EXPECTED_GATED1_TEST_AGREE) <= REPRO_TOLERANCE
        and gated1_test["regressions"] == EXPECTED_GATED1_REGRESSIONS
        and abs(chosen["tau"] - EXPECTED_GATED1_TAU) <= TAU_TOLERANCE
    )
    rule0 = flat_gate_passes(flat_test["agree"])
    log(f"chosen GATED1 tau: {chosen['tau']:.9f}")
    log(f"FLAT admitted: {flat_admitted}")
    log(f"FLAT admission marginals: {json.dumps(flat_marginals, sort_keys=True)}")
    log(f"GATED1 admitted: {chosen['admitted']}")
    log(f"GATED1 admission marginals: {json.dumps(chosen['marginals'], sort_keys=True)}")
    log(
        f"GATED1 reference: flat_test_agree={flat_test['agree']:.9f} "
        f"gated1_test_agree={gated1_test['agree']:.9f} "
        f"delta={gated1_test['agree'] - flat_test['agree']:+.9f} "
        f"regressions={gated1_test['regressions']}"
    )
    log(f"reproduction check: {reproduced}")
    log(f"Registered Rule 0 FLAT gate: {rule0}")
    report["step0"] = {
        "tau_runs": step0, "chosen_tau": chosen["tau"],
        "flat": {"admitted": flat_admitted, "marginals": flat_marginals,
                 "val": _public(flat_val), "test": _public(flat_test)},
        "gated1": {"admitted": chosen["admitted"], "marginals": chosen["marginals"],
                   "test": _public(gated1_test)},
        "reproduction_pass": reproduced, "rule0_flat_gate_pass": rule0,
    }
    if not (reproduced and rule0):
        report["stopped"] = "Rule 0 reproduction/FLAT gate failed"
        log("STOP: " + report["stopped"])
        _write(report)
        return 2

    def trace_at(candidates, admitted, idxs, tau):
        pred, conf, winner, key = trace_b1_cover(candidates, admitted, ids[idxs])
        return attribution_counts(b0.predict_cover(ids[idxs], counts_row_fn=counts_fn)[0],
                                  yv[idxs], pred, conf, winner, key, tau)

    val_attr = trace_at(chosen_candidates, chosen["admitted"], val_te, chosen["tau"])
    test_attr = trace_at(chosen_candidates, chosen["admitted"], test_te, chosen["tau"])
    bans = compute_ban_set(val_attr["counts"])
    coverage = val_to_test_coverage(bans, test_attr["regression_attributions"])
    cardinality = _cardinality(val_attr)
    vocab_base = len(uv) + 2
    dstate_composite = {
        key for rule, key in val_attr["counts"] if rule == "dstate gate"
    }
    dstate_features = {(key // vocab_base) - 1 for key in dstate_composite}
    cardinality["dstate gate"]["feature_only_claimed_values"] = len(dstate_features)
    ordered = sorted(
        val_attr["counts"],
        key=lambda item: (-val_attr["counts"][item]["val_regressions"], item[0], item[1]),
    )
    concentration = {}
    for k in (5, 10):
        top = ordered[:k]
        hit = sum(item in set(top) for item in test_attr["regression_attributions"])
        total = len(test_attr["regression_attributions"])
        concentration[f"top_{k}"] = {
            "keys": [{"rule": rule, "key": key,
                      "val_regressions": val_attr["counts"][(rule, key)]["val_regressions"]}
                     for rule, key in top],
            "test_regression_fraction": hit / total if total else 0.0,
            "covered": hit, "total": total,
        }
    report["part_a"] = {
        "val_counts": _serial_counts(val_attr["counts"]), "ban_set": _serial_bans(bans),
        "key_cardinality": cardinality, "concentration": concentration,
        "val_to_test_coverage": coverage,
        "test_regressions": len(test_attr["regression_attributions"]),
        "rule1_pass": coverage >= 0.50,
    }
    log("=== PART A: REGRESSION ANATOMY ===")
    log("per-family key cardinality: " + json.dumps(cardinality, sort_keys=True))
    log("val ban set: " + json.dumps(_serial_bans(bans), sort_keys=True))
    log("top-5 concentration: " + json.dumps(concentration["top_5"], sort_keys=True))
    log("top-10 concentration: " + json.dumps(concentration["top_10"], sort_keys=True))
    log(
        f"REGISTERED VAL->TEST BAN-SET COVERAGE: {coverage:.9f} "
        f"({sum(key in bans.get(rule, set()) for rule, key in test_attr['regression_attributions'])}/"
        f"{len(test_attr['regression_attributions'])})"
    )

    def run_part_b():
        log("=== PART B: GATED-RA RETREAT-INSIDE-ADMISSION ===")
        frontier, states = [], {}
        for tau_index, tau in enumerate(taus):
            raw_candidates, _ = fitted[tau_index]
            _, prelim_admitted, prelim_marginals, prelim_agree = _run_gated_tau(
                rules, b0_conf_fns, strip_keys(raw_candidates), counts_fn, ids, yv, val_te, tau
            )
            prelim_attr = trace_at(raw_candidates, prelim_admitted, val_te, tau)
            tau_bans = compute_ban_set(prelim_attr["counts"])
            retreated = [retreat_candidate(row, tau_bans.get(row[0], set())) for row in raw_candidates]
            stack, admitted, marginals, val_agree = _run_gated_tau(
                rules, b0_conf_fns, strip_keys(retreated), counts_fn, ids, yv, val_te, tau
            )
            post_attr = trace_at(retreated, admitted, val_te, tau)
            row = {
                "tau_index": tau_index, "tau_low": tau, "val_agree": val_agree,
                "val_regressions": int(post_attr["regressions"].sum()),
                "gated_coverage": float(post_attr["claim"].float().mean()),
                "gated_coverage_count": int(post_attr["claim"].sum()), "admitted": admitted,
                "bans": _serial_bans(tau_bans),
                "ban_counts": {name: len(tau_bans.get(name, set())) for name in active},
                "key_cardinality": _cardinality(prelim_attr),
                "preliminary": {"admitted": prelim_admitted, "marginals": prelim_marginals,
                                "val_agree": prelim_agree},
                "marginals": marginals,
                "retained_vs_banned_gain": {
                    "original_val_gains": int(prelim_attr["gains"].sum()),
                    "retained_original_val_gains": sum(
                        values["val_gains"] for item, values in prelim_attr["counts"].items()
                        if item[1] not in tau_bans.get(item[0], set())
                    ),
                    "banned_original_val_gains": sum(
                        values["val_gains"] for item, values in prelim_attr["counts"].items()
                        if item[1] in tau_bans.get(item[0], set())
                    ),
                    "post_retreat_val_gains": int(post_attr["gains"].sum()),
                },
            }
            frontier.append(row)
            states[tau_index] = {"candidates": retreated, "stack": stack, "admitted": admitted}
            log("  " + json.dumps(row, sort_keys=True))
        selected = select_operating_point(frontier, flat_val["agree"], flat_val["regressions"])
        part_b = {
            "frontier": frontier, "selected": selected,
            "selection_baseline": {"flat_val_agree": flat_val["agree"],
                                   "flat_val_regressions": flat_val["regressions"]},
        }

        def evaluate_test(point):
            state = states[point["tau_index"]]
            attr = trace_at(state["candidates"], state["admitted"], test_te, point["tau_low"])
            b1_pred, b1_conf, _, _ = trace_b1_cover(
                state["candidates"], state["admitted"], ids[test_te]
            )
            claim = (b1_pred >= 0) & (b1_conf > point["tau_low"])
            pred = torch.where(claim, b1_pred, b0_pred_test)
            metrics = _metrics(pred, yv[test_te], b0_pred_test)
            b = int((flat_test["correct"] & ~metrics["correct"]).sum())
            c = int((metrics["correct"] & ~flat_test["correct"]).sum())
            pvalue = exact_discordant_p(b, c)
            new_families = _family_set(state["admitted"]) - _family_set(flat_admitted)
            clauses = {
                "i_gated_ra_admits_family_flat_declined": bool(new_families),
                "ii_agree_plus_0.005": metrics["agree"] >= flat_test["agree"] + 0.005,
                "iii_two_sided_exact_p_lt_0.05": pvalue < 0.05,
                "iv_regressions_no_more_than_flat": metrics["regressions"] <= flat_test["regressions"],
            }
            first_three = all(list(clauses.values())[:3])
            verdict = (
                "PASS" if all(clauses.values()) else
                "PARTIAL" if first_three and not clauses["iv_regressions_no_more_than_flat"]
                and metrics["regressions"] <= 12 else "FAIL"
            )
            return {
                **_public(metrics), "gated_coverage": float(claim.float().mean()),
                "gated_coverage_count": int(claim.sum()), "admitted": state["admitted"],
                "new_families_vs_flat": sorted(new_families), "b": b, "c": c,
                "two_sided_exact_binomial_p": pvalue, "clauses": clauses, "verdict": verdict,
                "test_key_regressions": attr["regression_attributions"],
            }

        # Reuse #82's structural test guard; its None branch cannot invoke evaluate_test.
        resolved = resolve_operating_point(selected, evaluate_test)
        if selected is None:
            log("no admissible operating point (post-retreat)")
            log("GATED-RA test_te: not touched after validation selection")
            part_b.update({"outcome": "no admissible operating point (post-retreat)", "test": None})
        else:
            test_result = resolved["gated2_test"]
            part_b.update({"outcome": "selected", "test": test_result})
            shrinkage = {
                "validation": {"agree": selected["val_agree"],
                               "regressions": selected["val_regressions"],
                               "delta_agree_over_flat": selected["val_agree"] - flat_val["agree"]},
                "test": {"agree": test_result["agree"],
                         "regressions": test_result["regressions"],
                         "delta_agree_over_flat": test_result["agree"] - flat_test["agree"]},
                "val_to_test": {
                    "agree_change": test_result["agree"] - selected["val_agree"],
                    "regressions_change": (
                        test_result["regressions"] - selected["val_regressions"]
                    ),
                },
            }
            part_b["val_to_test_shrinkage"] = shrinkage
            log("selected operating point: " + json.dumps(selected, sort_keys=True))
            log("GATED-RA test metrics: " + json.dumps(test_result, sort_keys=True))
            log("val -> test shrinkage: " + json.dumps(shrinkage, sort_keys=True))
            for clause, value in test_result["clauses"].items():
                log(f"  {clause}: {value}")
            log(f"FINAL VERDICT: {test_result['verdict']}")
        return part_b

    part_b = gate_part_b(coverage, run_part_b)
    if part_b is None:
        report["stopped"] = "Registered Rule 1 coverage below 0.50; Part B not entered"
        report["part_b"] = None
        report["static_references"] = {"gated2_zero_regression": _static_zero_regression_reference()}
        report["total_wall_seconds"] = time.perf_counter() - started
        log("=== DESCRIPTIVE EXTRAS ===")
        log(
            "#82 zero-regression reference: "
            + json.dumps(
                report["static_references"]["gated2_zero_regression"], sort_keys=True
            )
        )
        log("honesty note: " + HONESTY_NOTE)
        log(f"total wall time: {report['total_wall_seconds']:.3f}s")
        _write(report)
        return 0

    report["part_b"] = part_b
    report["static_references"] = {"gated2_zero_regression": _static_zero_regression_reference()}
    log("=== VERDICT SCOPE + DESCRIPTIVE EXTRAS ===")
    log(VERDICT_SCOPE)
    log("commentary: kgram suffixes are inapplicable because kgram is not a B1 pool member")
    log(DSTATE_SCOPE)
    log(
        f"dstate actual cardinality: composite={len(dstate_composite)}, "
        f"feature_only={len(dstate_features)}"
    )
    small_others = {
        name: row["claimed_keys"] for name, row in cardinality.items()
        if name != "dstate gate" and 0 < row["claimed_keys"] <= 10
    }
    log("other <=10-key families at GATED1 point: " + json.dumps(small_others, sort_keys=True))
    log(
        "#82 zero-regression reference: "
        + json.dumps(
            report["static_references"]["gated2_zero_regression"], sort_keys=True
        )
    )
    log("honesty note: " + HONESTY_NOTE)
    report["total_wall_seconds"] = time.perf_counter() - started
    log(f"total wall time: {report['total_wall_seconds']:.3f}s")
    _write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
