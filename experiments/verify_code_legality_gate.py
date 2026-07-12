"""Independent cross-check of the registered code-legality residual gate."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# wyly_lm_v5 binds its recipe constants at import time. Lock this probe's recipe first.
WYLY_ENV: dict[str, str] = {
    "WYLY_TAG": "pythia70m",
    "WYLY_DS": "code",
    "WYLY_LIB": "mined",
    "WYLY_JUDGE": "cover",
    "WYLY_ONLINE": "1",
    "WYLY_COVER": "sw",
    "WYLY_LABELS": "corpus",
}
for _key, _value in WYLY_ENV.items():
    os.environ.setdefault(_key, _value)

import wyly_lm_v5 as v5  # noqa: E402

# Import only after v5 has bound the code-probe recipe above.
# isort: split
import campaign_sudoku_forced_move as csfm  # noqa: E402

ABORT_RESIDUAL = 0.02
FIRES_THRESHOLD = 0.30
DEAD_THRESHOLD = 0.10
PREVIOUS_RECOVERABLE = 76
PREVIOUS_RESIDUAL = 5109
PREVIOUS_GATE = 0.0149
PREVIOUS_VERDICT = "DEAD"


def log(message: str = "") -> None:
    print(message, flush=True)


def verdict_for(gate: float) -> str:
    if gate >= FIRES_THRESHOLD:
        return "FIRES"
    if gate < DEAD_THRESHOLD:
        return "DEAD"
    return "MIDDLE"


def token_character_maps(
    uv: torch.Tensor,
    ts,
    open_indices: list[int],
    close_indices: list[int],
) -> tuple[dict[int, str], dict[str, set[int]]]:
    """Build stripped-character maps directly from the registered definition."""
    opener_char = {
        int(j): ts.token_str(int(uv[int(j)])).strip()
        for j in open_indices
    }
    closer_char = {
        int(j): ts.token_str(int(uv[int(j)])).strip()
        for j in close_indices
    }
    closer_by_char: dict[str, set[int]] = {}
    for uv_index, character in closer_char.items():
        closer_by_char.setdefault(character, set()).add(uv_index)
    return opener_char, closer_by_char


def classify_recoverable(
    mate: torch.Tensor,
    gold_err: torch.Tensor,
    opener_char: dict[int, str],
    closer_by_char: dict[str, set[int]],
) -> torch.Tensor:
    """Classify residual rows from the preregistered stripped-character rule."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    flags: list[bool] = []
    for mate_value, gold_value in zip(mate.detach().cpu().tolist(),
                                      gold_err.detach().cpu().tolist(), strict=True):
        mate_index = int(mate_value)
        gold_index = int(gold_value)
        if mate_index == -1:
            flags.append(False)
            continue
        opener = opener_char.get(mate_index)
        if opener not in pairs:
            flags.append(False)
            continue
        required_closer = pairs[opener]
        flags.append(gold_index in closer_by_char.get(required_closer, set()))
    return torch.tensor(flags, dtype=torch.bool, device=mate.device)


def bracket_relevant_rows(
    gold_err: torch.Tensor,
    pred_err: torch.Tensor,
    is_close: torch.Tensor,
) -> torch.Tensor:
    """Mark residuals whose gold or non-abstaining wrong prediction is a closer."""
    relevant = is_close[gold_err].clone()
    predicted_token = pred_err >= 0
    if bool(predicted_token.any()):
        relevant[predicted_token] |= is_close[pred_err[predicted_token]]
    return relevant


def comparison_line(
    n_recoverable: int,
    n_residual: int,
    gate: float,
    verdict: str,
) -> str:
    count_match = (
        abs(n_recoverable - PREVIOUS_RECOVERABLE) <= 5
        and abs(n_residual - PREVIOUS_RESIDUAL) <= 5
    )
    agrees = count_match and verdict == PREVIOUS_VERDICT
    status = "AGREE" if agrees else "DIFFER"
    comparison = (
        f"{status}: independent=(n_recoverable={n_recoverable}, n_residual={n_residual}, "
        f"gate={gate:.6f}, verdict={verdict}); "
        f"previous=(n_recoverable={PREVIOUS_RECOVERABLE}, n_residual={PREVIOUS_RESIDUAL}, "
        f"gate={PREVIOUS_GATE:.4f}, verdict={PREVIOUS_VERDICT})"
    )
    if agrees:
        return comparison
    if abs(n_residual - PREVIOUS_RESIDUAL) > 5:
        diagnosis = (
            "regenerated residual count changed materially, so the previous result likely "
            "reflects a different regeneration or runtime state"
        )
    elif abs(n_recoverable - PREVIOUS_RECOVERABLE) > 5:
        diagnosis = (
            "residual count is stable but recovery count is not, pointing to a classifier "
            "discrepancy in the previous result"
        )
    else:
        diagnosis = "near-matching counts imply the previous threshold verdict was applied incorrectly"
    return f"{comparison}; diagnosis: {diagnosis}"


def main() -> int:
    log("Regenerating cover from scratch (the only verification path)...")
    regeneration_start = time.monotonic()
    model, rules, _original = csfm.run_cover_regeneration()
    regeneration_seconds = time.monotonic() - regeneration_start

    ids, y, cls, uv, _tr, te = v5.load_ds()
    yv = cls[y]
    out, pred = v5.core_cover_sw(model, rules, ids, yv, cls, te, return_pred=True)

    gold_te = yv[te]
    mism = pred != gold_te
    if bool(((pred == -1) & ~mism).any()):
        raise RuntimeError("an abstaining prediction unexpectedly matched a gold uv-index")
    err = te[mism]
    pred_err = pred[mism]
    gold_err = yv[err]
    n_residual = int(mism.sum().item())
    residual_rate = n_residual / len(te)

    ts = v5.load_codec()
    open_indices, close_indices = v5.bracket_sets(uv, ts)
    is_open = torch.zeros(len(uv), dtype=torch.bool, device=ids.device)
    is_close = torch.zeros(len(uv), dtype=torch.bool, device=ids.device)
    is_open[open_indices] = True
    is_close[close_indices] = True

    mate = v5.mate_feature(ids[err], is_open, is_close)
    opener_char, closer_by_char = token_character_maps(
        uv, ts, open_indices, close_indices
    )
    recoverable = classify_recoverable(mate, gold_err, opener_char, closer_by_char)
    n_recoverable = int(recoverable.sum().item())
    gate = n_recoverable / n_residual if n_residual else float("nan")
    verdict = verdict_for(gate)
    abort_triggered = residual_rate < ABORT_RESIDUAL

    bracket_relevant = bracket_relevant_rows(gold_err, pred_err, is_close)
    n_bracket_relevant = int(bracket_relevant.sum().item())
    within_bracket_recovery = (
        n_recoverable / n_bracket_relevant if n_bracket_relevant else None
    )
    sanity_ok = n_recoverable <= n_bracket_relevant

    log()
    log("=" * 72)
    log("FINAL INDEPENDENT CODE-LEGALITY GATE SUMMARY")
    log("=" * 72)
    log(
        "step 2 path: regenerate (run_cover_regeneration; no cache-skip), "
        f"wall-clock={regeneration_seconds:.1f}s"
    )
    log(f"len(te): {len(te)}")
    log(f"n_residual: {n_residual}")
    log(f"residual_rate: {residual_rate:.6f}")
    log(
        "regenerated core_sw: "
        f"agree={out['agree']:.6f}, cover={out['cover']:.6f}, "
        f"agree_fired={out['agree_fired']:.6f}"
    )
    log(f"n_recoverable: {n_recoverable}")
    log(f"gate: {gate:.6f}")
    log(f"verdict: {verdict}")
    log(f"ABORT condition triggered: {abort_triggered}")
    if abort_triggered:
        log(f"ABORT note: residual_rate < {ABORT_RESIDUAL:.2f}; measurement still reported")
    log(f"n_bracket_relevant: {n_bracket_relevant}")
    if within_bracket_recovery is None:
        log("within_bracket_recovery: None (no bracket-relevant residual rows)")
    else:
        log(f"within_bracket_recovery: {within_bracket_recovery:.6f}")
    if sanity_ok:
        log("sanity check: PASS (every recoverable row is bracket-relevant)")
    else:
        log("SANITY CHECK VIOLATION: n_recoverable > n_bracket_relevant")
    log(comparison_line(n_recoverable, n_residual, gate, verdict))
    log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
