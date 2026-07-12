"""Slice #89-shaped code-legality residual probe (measurement only, no register).

MEASURE-FIRST: residual errors of the corpus code cover vs mate_feature recoverability,
per PIL_CODE_REGISTER_PREREG.md. Thresholds FIXED there — never tuned to hit a verdict.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# Code recipe (NOT babi-flavored #89 defaults). Plain assignment so setdefaults in the
# sudoku harness no-op when that module is imported next. Explicitly pin the babi-only
# flags off (empty/"0") BEFORE importing campaign_sudoku_forced_move, whose setdefault
# would otherwise inject CONCEPTS/POINTER/… and pollute v5's import-time STATE path.
WYLY_ENV: dict[str, str] = {
    "WYLY_TAG": "pythia70m",
    "WYLY_DS": "code",
    "WYLY_LIB": "mined",
    "WYLY_JUDGE": "cover",
    "WYLY_ONLINE": "1",
    "WYLY_COVER": "sw",
    "WYLY_LABELS": "corpus",
}
for _k, _v in WYLY_ENV.items():
    os.environ[_k] = _v
for _k in ("WYLY_CONCEPTS", "WYLY_POINTER", "WYLY_TPOINTER", "WYLY_DX", "WYLY_CX"):
    os.environ[_k] = ""
os.environ["WYLY_FOLDS"] = "0"

# v5 MUST bind before sudoku's setdefaults can affect import-time constants.
import wyly_lm_v5 as v5  # noqa: E402  # isort: skip
import campaign_sudoku_forced_move as sudoku  # noqa: E402  # isort: skip
import campaign_composite_discriminator as cd  # noqa: E402  # isort: skip

OUTPUT = REPO / "data" / "code_legality_probe.json"
NO_RUNNER_CONF = -1e9
PAIR = {"(": ")", "[": "]", "{": "}"}
AXES_USED = ("confidence", "gap", "decile_index")
# Descriptive admissible-point cutoffs for THIS probe (not #88 CLAUSE-2 rescue replay).
ADMISSIBLE_PRECISION = 0.8
ADMISSIBLE_RECALL = 0.1

HONESTY_NOTE = (
    "measure-first probe only (no register build); mate/depth features are already Datalog-"
    "certified; GATE = mate-recoverable residual / all residual; thresholds fixed in "
    "PIL_CODE_REGISTER_PREREG.md before numbers; LABELS=corpus; control omits teacher_consensus "
    "(no 8-teacher code ensemble); depth_flag uses clamp-at-0 proxy for 'would go negative'."
)

BRACKET_CHAR_RESOLUTION_NOTE = (
    "bracket_sets returns flat compact ids for openers/closers; on this code tokenizer there "
    "are typically two surface variants per bracket character (bare and space-prefixed). "
    "Mate-forced closer == gold is checked at CHARACTER level after .strip() (same convention "
    "as bracket_sets itself), not literal compact-id equality — whitespace variant is a "
    "formatting choice unrelated to bracket legality."
)

DEPTH_FLAG_NOTE = (
    "depth_feature clamps depth to [0, cap]; depth==0 with a closer prediction is the only "
    "observable proxy for 'would drive depth negative' under that clamp."
)


# ---------------------------------------------------------------------------
# Bracket char maps + mate-recoverable classifier
# ---------------------------------------------------------------------------

def bracket_char_maps(
    op: list[int], cl: list[int], uv: torch.Tensor, ts: Any
) -> tuple[dict[int, str], dict[int, str]]:
    """Map compact opener/closer ids -> stripped surface character.

    Collapses bare and space-prefixed variants (e.g. ')' and ' )') into the same char so
    mate-forced matching is legality-level, not formatting-level. See
    BRACKET_CHAR_RESOLUTION_NOTE.
    """

    def char_of(i: int) -> str:
        return ts.token_str(int(uv[i].item())).strip()

    return {i: char_of(i) for i in op}, {i: char_of(i) for i in cl}


def mate_recoverable_mask(
    mate_opener: torch.Tensor,
    gold: torch.Tensor,
    pred: torch.Tensor,
    opener_char: dict[int, str],
    closer_char: dict[int, str],
    pair: dict[str, str] | None = None,
) -> torch.Tensor:
    """Per-row mate-recoverable flags (prereg definition + explicit pred!=gold).

    A residual row is mate-recoverable iff ALL of:
      (a) mate_opener >= 0 and its char is a known opener
      (b) gold is a known closer (any surface variant)
      (c) closer_char[gold] == PAIR[opener_char[mate_opener]]
      (d) pred != gold  (load-bearing; residual rows already satisfy this by construction)
    """
    pairs = pair if pair is not None else PAIR
    n = int(mate_opener.shape[0])
    out = torch.zeros(n, dtype=torch.bool, device=mate_opener.device)
    mo = mate_opener.detach().cpu()
    g = gold.detach().cpu()
    p = pred.detach().cpu()
    for i in range(n):
        m_i = int(mo[i].item())
        g_i = int(g[i].item())
        p_i = int(p[i].item())
        if m_i < 0 or m_i not in opener_char:
            continue
        if g_i not in closer_char:
            continue
        forced = pairs.get(opener_char[m_i])
        if forced is None or closer_char[g_i] != forced:
            continue
        if p_i == g_i:
            continue
        out[i] = True
    return out


def depth_flag_row(pred: int, is_close: torch.Tensor, depth: int) -> bool:
    """True if wrong pred is a closer at depth==0 (clamp proxy for illegal closer)."""
    if pred < 0:
        return False
    if not bool(is_close[pred].item() if torch.is_tensor(is_close[pred]) else is_close[pred]):
        return False
    return int(depth) == 0


def depth_flag_fraction(
    pred: torch.Tensor,
    depths: torch.Tensor,
    is_close: torch.Tensor,
) -> float:
    """Fraction of residual rows where closer-pred @ depth==0."""
    n = int(pred.shape[0])
    if n == 0:
        return 0.0
    flags = [
        depth_flag_row(int(pred[i].item()), is_close, int(depths[i].item()))
        for i in range(n)
    ]
    return float(sum(flags) / n)


def calibration_deciles(confidence: torch.Tensor) -> torch.Tensor:
    """Self-referential decile buckets from the same array's 10..90th percentiles.

    Deviation from #87's two-arg form (external reference distribution): no registered
    code-specific external reference exists for this probe.
    """
    q = torch.arange(1, 10, dtype=torch.float64) / 10
    conf = confidence.detach().double().cpu()
    boundaries = torch.quantile(conf, q)
    return torch.bucketize(conf, boundaries, right=True) + 1


# ---------------------------------------------------------------------------
# Local read-only core_cover_sw with runner-up confidence
# ---------------------------------------------------------------------------

def core_cover_sw_runner_traced(
    model: Any,
    rules: list,
    ids: torch.Tensor,
    yv: torch.Tensor,
    cls: torch.Tensor,
    idxs: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, dict[str, Any]]:
    """Read-only mirror of v5.core_cover_sw that also tracks runner-up confidence.

    Same consider() winner-selection semantics (highest conf wins); runner_conf is the
    second-highest eligible confidence (NO_RUNNER_CONF if none). Does NOT import
    campaign_wikitext_gated_rescue (avoids that module's WYLY_DS reassignment).
    """
    w = ids[idxs]
    pred = torch.full_like(w[:, -1], -1)
    conf = torch.full((len(w),), NO_RUNNER_CONF, device=w.device)
    runner_conf = torch.full_like(conf, NO_RUNNER_CONF)

    def consider(a: torch.Tensor, c: torch.Tensor, eligible: torch.Tensor | None = None) -> None:
        nonlocal pred, conf, runner_conf
        valid = a >= 0 if eligible is None else eligible & (a >= 0)
        beats_winner = valid & (c > conf)
        beats_runner_only = valid & (c > runner_conf) & ~beats_winner
        runner_conf = torch.where(
            beats_winner, conf, torch.where(beats_runner_only, c, runner_conf)
        )
        pred = torch.where(beats_winner, a, pred)
        conf = torch.where(beats_winner, c, conf)

    for name, fn in rules:
        if name in v5.CONF_FNS:
            value, confidence = v5.CONF_FNS[name](w)
        else:
            value = fn(w)
            confidence = torch.full(
                (len(w),), float(v5.RULE_CONF.get(name, 0.0)), device=w.device
            )
        consider(value, confidence)

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
    consider(value, confidence)

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
            consider(value, confidence, eligible=low)

    correct = pred == yv[idxs]
    fired = pred >= 0
    out = {
        "agree": float(correct.float().mean()),
        "cover": float(fired.float().mean()),
        "agree_fired": float(correct[fired].float().mean()) if bool(fired.any()) else 0.0,
    }
    trace = {
        "conf": conf,
        "runner_conf": runner_conf,
        "gap": conf - runner_conf,
        "pred": pred,
    }
    return out, pred, trace


def assert_trace_parity(
    pred_traced: torch.Tensor, pred_ref: torch.Tensor
) -> bool:
    """Raise if traced preds differ from the real core_cover_sw preds."""
    if not torch.equal(pred_traced, pred_ref):
        n_diff = int((pred_traced != pred_ref).sum().item())
        raise AssertionError(
            f"runner-up tracer prediction parity failed: {n_diff} rows differ"
        )
    return True


# ---------------------------------------------------------------------------
# Control (statistical separability of mate-recoverable residual rows)
# ---------------------------------------------------------------------------

def _precision_recall(
    scores: torch.Tensor, labels: torch.Tensor, threshold: float
) -> tuple[float, float]:
    y = labels.detach().long().cpu().view(-1)
    s = scores.detach().double().cpu().view(-1)
    pred_pos = s >= threshold
    tp = int((pred_pos & (y == 1)).sum().item())
    fp = int((pred_pos & (y == 0)).sum().item())
    fn = int((~pred_pos & (y == 1)).sum().item())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return precision, recall


def run_control(
    confidence: torch.Tensor,
    gap: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_folds: int = 5,
) -> dict[str, Any]:
    """#88-adapted control on residual rows; skip if either class has < n_folds rows."""
    omitted = {
        "teacher_consensus": (
            "requires an 8-teacher ensemble from campaign_wikitext_gated_rescue; "
            "no such ensemble exists for the code dataset (LABELS=corpus; only "
            "pythia70m/pythia410m teacher artifacts). Not fabricated via 2-teacher proxy."
        )
    }
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    base: dict[str, Any] = {
        "axes_used": list(AXES_USED),
        "omitted_axes": omitted,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "admissible_precision_cutoff": ADMISSIBLE_PRECISION,
        "admissible_recall_cutoff": ADMISSIBLE_RECALL,
        "admissible_note": (
            "descriptive cutoffs for this probe only (precision>=0.8 AND recall>=0.1); "
            "NOT the same procedure as #88 CLAUSE-2 rescue-replay admissible points"
        ),
        "decile_note": (
            "self-referential calibration_deciles from residual-row confidences "
            "(no external code reference distribution)"
        ),
    }
    if n_pos < n_folds or n_neg < n_folds:
        return {
            **base,
            "skipped": True,
            "reason": (
                f"stratified {n_folds}-fold requires >= {n_folds} rows per class; "
                f"got n_pos={n_pos}, n_neg={n_neg}"
            ),
            "cv_auc": None,
            "perm_p": None,
            "passes": None,
            "admissible_points": [],
            "parity_check_passed": None,
        }

    decile = calibration_deciles(confidence).to(dtype=torch.float64)
    features = torch.stack(
        [
            confidence.detach().double().cpu(),
            gap.detach().double().cpu(),
            decile.cpu(),
        ],
        dim=1,
    )
    y = labels.detach().double().cpu().view(-1)

    cv = cd.cv_auc_distribution(features, y)
    cv_auc = float(cv["median"])
    null_aucs = cd.permutation_null_aucs(features, y)
    perm_p = float(cd.empirical_perm_p(cv_auc, null_aucs))
    oof = cd.pooled_oof_scores(features, y, fold_seed=0)
    thresholds = cd.score_deciles(oof)
    admissible: list[dict[str, float]] = []
    for thr in thresholds:
        prec, rec = _precision_recall(oof, y, thr)
        if prec >= ADMISSIBLE_PRECISION and rec >= ADMISSIBLE_RECALL:
            admissible.append(
                {"threshold": float(thr), "precision": prec, "recall": rec}
            )

    # PASSES iff rows are NOT statistically separable at the #88 CLAUSE-1 bar.
    passes = not (cv_auc >= 0.70 and perm_p < 0.05)
    return {
        **base,
        "skipped": False,
        "reason": None,
        "cv_auc": cv_auc,
        "cv_auc_min": float(cv["min"]),
        "cv_auc_max": float(cv["max"]),
        "perm_p": perm_p,
        "passes": bool(passes),
        "admissible_points": admissible,
        "parity_check_passed": None,  # filled by main
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def log(msg: str = "") -> None:
    print(msg, flush=True)


def _write_report(report: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    log(f"wrote {OUTPUT}")


def main() -> int:
    t0 = time.time()
    log("=" * 72)
    log("CODE LEGALITY PROBE -- measure-first residual recoverability (#89-shaped)")
    log("=" * 72)
    log(f"HONESTY: {HONESTY_NOTE}")
    log()
    log("WYLY_* env used for this run:")
    env_used = {k: os.environ.get(k, "") for k in WYLY_ENV}
    for k, v in env_used.items():
        log(f"  {k}={v}")
    log(
        f"  (module-bound) TAG={v5.TAG} DS={v5.DS} LIB={v5.LIB} JUDGE={v5.JUDGE} "
        f"LABELS={v5.LABELS} SWCOVER={v5.SWCOVER} STATE={v5.STATE}"
    )
    log(f"  STRATA={v5.STRATA} (expected False for this recipe)")
    log()

    # --- Cover regeneration ---
    log("--- run_cover_regeneration (v5.main) ---")
    t_reg = time.time()
    model, rules, original_cover = sudoku.run_cover_regeneration()
    log(f"regeneration wall time: {time.time() - t_reg:.1f}s")
    admitted = [n for n, _ in rules]
    log(f"admitted rules ({len(admitted)}): {admitted}")
    log(f"v5.STATE path: {v5.STATE}")

    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    log(
        f"load_ds: N={ids.shape[0]} shape={tuple(ids.shape)} "
        f"tr={len(tr)} te={len(te)} vocab_compact={len(uv)}"
    )

    te_idx = te
    out_te, pred_te = v5.core_cover_sw(
        model, rules, ids, yv, cls, te_idx, return_pred=True
    )
    log(
        f"regenerated core_sw (te): agree={out_te['agree']:.4f} "
        f"cover={out_te['cover']:.4f} agree_fired={out_te['agree_fired']:.4f}"
    )

    err_mask = pred_te != yv[te_idx]
    err_rows = te_idx[err_mask]
    n_te = int(len(te_idx))
    n_residual = int(len(err_rows))
    residual_rate = n_residual / n_te if n_te else 0.0
    log(f"residual_rate={residual_rate:.6f} n_residual={n_residual} / n_te={n_te}")

    module_bound = {
        "TAG": v5.TAG,
        "DS": v5.DS,
        "LIB": v5.LIB,
        "JUDGE": v5.JUDGE,
        "LABELS": v5.LABELS,
        "STATE": str(v5.STATE),
    }
    load_ds_info = {
        "n_total": int(ids.shape[0]),
        "n_tr": int(len(tr)),
        "n_te": int(len(te)),
        "vocab": int(len(uv)),
    }

    abort_msg = sudoku.abort_if_residual_too_small(residual_rate)
    if abort_msg is not None:
        log(abort_msg)
        report = {
            "aborted": True,
            "honesty_note": HONESTY_NOTE,
            "wyly_env": env_used,
            "module_bound": module_bound,
            "load_ds": load_ds_info,
            "regenerated_core_sw": out_te,
            "admitted_rules": admitted,
            "residual_rate": residual_rate,
            "n_residual": n_residual,
            "gate": None,
            "n_mate_recoverable": None,
            "verdict": "ABORT",
            "depth_flag_frac": None,
            "bracket_subset": None,
            "bracket_char_resolution_note": BRACKET_CHAR_RESOLUTION_NOTE,
            "control": None,
            "wall_time_s": time.time() - t0,
            "abort_message": abort_msg,
        }
        _write_report(report)
        return 2

    # --- Bracket sets + char maps ---
    ts = v5.load_codec()
    op, cl = v5.bracket_sets(uv, ts)
    opener_char, closer_char = bracket_char_maps(op, cl, uv, ts)
    log(
        f"bracket_sets: n_open={len(op)} n_close={len(cl)} "
        f"opener_chars={sorted(set(opener_char.values()))} "
        f"closer_chars={sorted(set(closer_char.values()))}"
    )
    log(f"bracket_char_resolution: {BRACKET_CHAR_RESOLUTION_NOTE}")

    vocab = len(uv)  # match wyly_mate_certify: compact-vocab-sized masks
    is_open = torch.zeros(vocab, dtype=torch.bool, device=ids.device)
    is_close = torch.zeros(vocab, dtype=torch.bool, device=ids.device)
    if op:
        is_open[torch.tensor(op, device=ids.device)] = True
    if cl:
        is_close[torch.tensor(cl, device=ids.device)] = True

    # --- Mate-recoverable residual ---
    w_err = ids[err_rows]
    mate_opener = v5.mate_feature(w_err, is_open, is_close)
    gold_err = yv[err_rows]
    pred_err = pred_te[err_mask]
    rec_mask = mate_recoverable_mask(
        mate_opener, gold_err, pred_err, opener_char, closer_char
    )
    n_mate_recoverable = int(rec_mask.sum().item())
    gate = n_mate_recoverable / n_residual if n_residual else 0.0
    verdict = sudoku.verdict_from_gate(gate)
    log(
        f"GATE={gate:.6f} n_mate_recoverable={n_mate_recoverable} VERDICT={verdict}"
    )

    # --- Secondary: depth-flag fraction ---
    depths = v5.depth_feature(w_err, is_open, is_close, cap=8)
    depth_flag_frac = depth_flag_fraction(pred_err, depths, is_close)
    log(f"depth_flag_frac={depth_flag_frac:.6f} ({DEPTH_FLAG_NOTE})")

    # --- Context: bracket-relevant residual subset ---
    bracket_token_set = set(op) | set(cl)
    gold_cpu = gold_err.detach().cpu()
    pred_cpu = pred_err.detach().cpu()
    subset_idx: list[int] = []
    for i in range(n_residual):
        g_i = int(gold_cpu[i].item())
        p_i = int(pred_cpu[i].item())
        if g_i in bracket_token_set or (p_i >= 0 and p_i in bracket_token_set):
            subset_idx.append(i)
    n_bracket = len(subset_idx)
    if n_bracket == 0:
        within_recovery: float | None = None
        bracket_subset_note = "empty bracket subset; within_recovery null (no divide-by-zero)"
    else:
        n_rec_in = int(sum(bool(rec_mask[i].item()) for i in subset_idx))
        within_recovery = n_rec_in / n_bracket
        bracket_subset_note = None
    bracket_subset = {
        "n": n_bracket,
        "within_recovery": within_recovery,
        "note": bracket_subset_note,
    }
    log(
        f"bracket_subset: n={n_bracket} within_recovery="
        f"{within_recovery if within_recovery is not None else 'null'}"
    )

    # --- CONTROL ---
    log("--- control (statistical separability of mate-recoverable residual) ---")
    _, pred_traced, trace = core_cover_sw_runner_traced(
        model, rules, ids, yv, cls, te_idx
    )
    try:
        parity_ok = assert_trace_parity(pred_traced, pred_te)
    except AssertionError as exc:
        log(f"PARITY FAIL: {exc}")
        raise
    log(f"parity_check_passed={parity_ok} (traced pred == core_cover_sw pred on te)")
    log(f"STRATA active in tracer branch: {bool(v5.STRATA)}")

    conf_err = trace["conf"][err_mask]
    gap_err = trace["gap"][err_mask]
    labels = rec_mask.to(dtype=torch.long)
    control = run_control(conf_err, gap_err, labels)
    control["parity_check_passed"] = bool(parity_ok)

    if control.get("skipped"):
        log(
            f"control SKIPPED: {control['reason']} "
            f"(n_pos={control['n_pos']} n_neg={control['n_neg']})"
        )
    else:
        log(
            f"control: axes={control['axes_used']} "
            f"omitted={list(control['omitted_axes'].keys())} "
            f"cv_auc={control['cv_auc']:.4f} perm_p={control['perm_p']:.4g} "
            f"passes={control['passes']} "
            f"n_admissible={len(control['admissible_points'])}"
        )

    wall = time.time() - t0

    # --- Scoreboard ---
    log()
    log("=" * 72)
    log("SCOREBOARD")
    log("=" * 72)
    log(f"env: {env_used}")
    log(f"STATE: {v5.STATE}")
    log(
        f"regenerated core_sw: agree={out_te['agree']:.4f} "
        f"cover={out_te['cover']:.4f} agree_fired={out_te['agree_fired']:.4f}"
    )
    log(f"residual_rate={residual_rate:.6f} n_residual={n_residual}")
    log(
        f"GATE={gate:.6f} n_mate_recoverable={n_mate_recoverable} VERDICT={verdict}"
    )
    log(f"depth_flag_frac={depth_flag_frac:.6f}")
    log(
        f"bracket_subset: n={n_bracket} within_recovery="
        f"{within_recovery if within_recovery is not None else 'null'}"
    )
    if control.get("skipped"):
        log(
            f"control: SKIPPED reason={control['reason']} "
            f"axes_used={control['axes_used']} "
            f"omitted={list(control['omitted_axes'].keys())} "
            f"n_pos={control['n_pos']} n_neg={control['n_neg']}"
        )
    else:
        log(
            f"control: axes_used={control['axes_used']} "
            f"omitted={list(control['omitted_axes'].keys())} "
            f"cv_auc={control['cv_auc']:.4f} perm_p={control['perm_p']:.4g} "
            f"passes={control['passes']}"
        )
    log(f"HONESTY: {HONESTY_NOTE}")
    log(f"wall time total: {wall:.1f}s")

    report = {
        "aborted": False,
        "honesty_note": HONESTY_NOTE,
        "wyly_env": env_used,
        "module_bound": module_bound,
        "load_ds": load_ds_info,
        "regenerated_core_sw": out_te,
        "admitted_rules": admitted,
        "residual_rate": residual_rate,
        "n_residual": n_residual,
        "gate": gate,
        "n_mate_recoverable": n_mate_recoverable,
        "verdict": verdict,
        "depth_flag_frac": depth_flag_frac,
        "depth_flag_note": DEPTH_FLAG_NOTE,
        "bracket_subset": bracket_subset,
        "bracket_char_resolution_note": BRACKET_CHAR_RESOLUTION_NOTE,
        "control": control,
        "wall_time_s": wall,
        "n_openers": len(op),
        "n_closers": len(cl),
        "strata": bool(v5.STRATA),
    }
    _write_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
