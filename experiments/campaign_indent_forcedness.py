"""Slice #89-shaped indent-forcedness probe (measurement only, no register).

MEASURE-FIRST: what fraction of the code cover's whitespace_indent residual is exactly
recoverable by a causal indent-continuation rule. Thresholds FIXED — never tuned.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# wyly_lm_v5 binds its recipe constants at import time. Lock the registered recipe first.
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
    os.environ.setdefault(_k, _v)

import wyly_lm_v5 as v5  # noqa: E402  # isort: skip
import campaign_sudoku_forced_move as csfm  # noqa: E402  # isort: skip
import campaign_code_residual_anatomy as cra  # noqa: E402  # isort: skip
import campaign_code_legality_probe as clp  # noqa: E402  # isort: skip

OUTPUT = REPO / "data" / "indent_forcedness.json"
FIRES_THRESHOLD = 0.5
DEAD_THRESHOLD = 0.2
REGISTERED_N_CLASS = 531
CEILING_NOTE = (
    "this class's mass is ~0.104 of residual per PIL_CODE_REGISTER_PREREG.md "
    "(informational only, not a gate input)"
)
FRAMING_NOTE = (
    "C/C++ indent is convention-consistency, NOT grammar legality -- a file-local "
    "STYLE deriver, not a legality register"
)
HONESTY_NOTE = (
    "Case B2 declines mid-line horizontal padding (predicted empty) rather than guessing; "
    "true 'dedent before an as-yet-ungenerated closer' is not causally available and is not "
    "implemented — only depth change already present on the completed line (Case C-structural) "
    "is used. Newline-placement Case A/B/C was chosen from observed corpus whitespace_indent "
    "examples (mid-run space tokens and pure blank-line tokens dominate), not to maximize the gate."
)


def log(message: str = "") -> None:
    print(message, flush=True)


def verdict_for(gate: float) -> str:
    """Local thresholds (0.5 / 0.2) — NOT #89's sudoku 0.30 / 0.10 pair."""
    if gate >= FIRES_THRESHOLD:
        return "FIRES"
    if gate < DEAD_THRESHOLD:
        return "DEAD"
    return "MIDDLE"


# ---------------------------------------------------------------------------
# Pure predictor helpers (testable without cover / codec)
# ---------------------------------------------------------------------------


def leading_ws(s: str) -> str:
    """Maximal prefix of s consisting only of space/tab."""
    i = 0
    while i < len(s) and s[i] in (" ", "\t"):
        i += 1
    return s[:i]


def infer_indent_unit(context_text: str) -> tuple[str, int, bool]:
    """Infer (unit_char, unit_size, used_default) from context_text's own lines."""
    lines = context_text.split("\n")
    non_blank_prefixes: list[str] = []
    for line in lines:
        prefix = leading_ws(line)
        if line[len(prefix) :]:  # non-blank: has a non-whitespace char after prefix
            non_blank_prefixes.append(prefix)

    positive_increments: list[int] = []
    for i in range(1, len(non_blank_prefixes)):
        inc = len(non_blank_prefixes[i]) - len(non_blank_prefixes[i - 1])
        if inc > 0:
            positive_increments.append(inc)

    if not positive_increments:
        return " ", 4, True

    non_empty = [p for p in non_blank_prefixes if p]
    if non_empty:
        tab_only = sum(1 for p in non_empty if all(c == "\t" for c in p))
        unit_char = "\t" if tab_only / len(non_empty) > 0.5 else " "
    else:
        unit_char = " "

    counts = Counter(positive_increments)
    max_count = max(counts.values())
    # mode; ties broken by smallest value
    unit_size = min(v for v, c in counts.items() if c == max_count)
    return unit_char, unit_size, False


def naive_indent_chars(context_text: str) -> str:
    """Leading-ws of the most recently completed line in context_text."""
    if "\n" not in context_text:
        return ""
    lines = context_text.split("\n")
    # lines[-1] is the (possibly empty / in-progress) last segment
    return leading_ws(lines[-2])


def structural_depths(
    ids_row: torch.Tensor,
    is_open: torch.Tensor,
    is_close: torch.Tensor,
    token_str_fn: Callable[[int], str],
) -> tuple[int, int]:
    """Return (d_end, d_line_start) matching depth_feature semantics (cap=8)."""
    sign = is_open[ids_row].long() - is_close[ids_row].long()
    cumdepth = sign.cumsum(0)
    d_end = int(cumdepth[-1].clamp(0, 8).item())
    j_last_nl: int | None = None
    length = int(ids_row.shape[0])
    for j in range(length - 1, -1, -1):
        compact = int(ids_row[j].item())
        if "\n" in token_str_fn(compact):
            j_last_nl = j
            break
    if j_last_nl is not None:
        d_line_start = int(cumdepth[j_last_nl].clamp(0, 8).item())
    else:
        d_line_start = d_end
    return d_end, d_line_start


def predict_structural_indent(
    context_text: str,
    ids_row: torch.Tensor,
    is_open: torch.Tensor,
    is_close: torch.Tensor,
    token_str_fn: Callable[[int], str],
) -> tuple[str, dict[str, Any]]:
    """Causal indent-continuation predictor (§5). Returns (predicted_surface, meta)."""
    unit_char, unit_size, used_default = infer_indent_unit(context_text)
    d_end, d_line_start = structural_depths(ids_row, is_open, is_close, token_str_fn)
    target_indent_chars = unit_char * (unit_size * d_end)

    if context_text == "" or context_text[-1] not in (" ", "\t", "\n"):
        # Case C: context ends right after a real (non-whitespace) token
        if d_end == d_line_start:
            current_line = (
                context_text.rsplit("\n", 1)[-1] if "\n" in context_text else context_text
            )
            predicted = "\n" + leading_ws(current_line)
            case = "C-same-depth"
        else:
            predicted = "\n" + target_indent_chars
            case = "C-structural"
    elif context_text[-1] == "\n":
        # Case A: already at a fresh, untyped line start
        predicted = target_indent_chars
        case = "A"
    else:
        # Case B: mid-whitespace-run continuation
        i = len(context_text)
        while i > 0 and context_text[i - 1] in (" ", "\t"):
            i -= 1
        already = context_text[i:]
        preceding = context_text[i - 1] if i > 0 else ""
        if preceding == "\n" or preceding == "":
            # B1: true indent-run continuation since a real newline (or window start)
            if target_indent_chars.startswith(already):
                predicted = target_indent_chars[len(already) :]
            else:
                predicted = ""
            case = "B1"
        else:
            # B2: mid-line horizontal padding — decline
            predicted = ""
            case = "B2"

    meta: dict[str, Any] = {
        "case": case,
        "d_end": d_end,
        "d_line_start": d_line_start,
        "unit_char": unit_char,
        "unit_size": unit_size,
        "used_default": used_default,
        "target_indent_chars": target_indent_chars,
    }
    return predicted, meta


def predict_naive_indent(context_text: str) -> str:
    """Naive baseline: same Case A/B/C skeleton, no structural awareness (§6)."""
    naive = naive_indent_chars(context_text)

    if context_text == "" or context_text[-1] not in (" ", "\t", "\n"):
        # Case C: unconditional copy of previous-completed-line indent (no depth check).
        # Uses naive_indent_chars (most recently completed line), not the just-ended
        # segment — so after a bare closer line the baseline still re-emits the prior
        # body indent (does not dedent). See test_naive_baseline_blindness_on_dedent.
        return "\n" + naive
    if context_text[-1] == "\n":
        # Case A
        return naive
    # Case B
    i = len(context_text)
    while i > 0 and context_text[i - 1] in (" ", "\t"):
        i -= 1
    already = context_text[i:]
    preceding = context_text[i - 1] if i > 0 else ""
    if preceding == "\n" or preceding == "":
        if naive.startswith(already):
            return naive[len(already) :]
        return ""
    return ""


def build_is_open_close(
    uv: torch.Tensor, codec: Any, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact-id-indexed opener/closer masks via certified bracket_sets."""
    op, cl = v5.bracket_sets(uv, codec)
    vocab = len(uv)
    is_open = torch.zeros(vocab, dtype=torch.bool, device=device)
    is_close = torch.zeros(vocab, dtype=torch.bool, device=device)
    if op:
        is_open[torch.tensor(op, device=device)] = True
    if cl:
        is_close[torch.tensor(cl, device=device)] = True
    return is_open, is_close


def comparison_line_n_class(n_class: int) -> str:
    """Informational drift note vs the #98 prereg record (~531); never aborts."""
    delta = n_class - REGISTERED_N_CLASS
    if abs(delta) <= 5:
        return (
            f"n_class={n_class} within ±5 of registered whitespace_indent count "
            f"{REGISTERED_N_CLASS} (delta={delta:+d})"
        )
    return (
        f"n_class={n_class} differs from registered whitespace_indent count "
        f"{REGISTERED_N_CLASS} by {delta:+d} (informational; no abort)"
    )


# ---------------------------------------------------------------------------
# Main measurement
# ---------------------------------------------------------------------------


def main() -> int:
    t0 = time.time()
    log("=" * 72)
    log("INDENT FORCEDNESS PROBE — measure-first whitespace_indent recoverability")
    log("=" * 72)
    log(f"FRAMING: {FRAMING_NOTE}")
    log(f"HONESTY: {HONESTY_NOTE}")
    log()

    state_path = Path(v5.STATE)
    log(f"v5.STATE path: {state_path}")
    if not state_path.exists():
        log(f"FATAL: state file missing/unreadable: {state_path}")
        log("STOP — do not fabricate numbers; do not fall back to a smaller recipe.")
        return 1

    log("--- run_cover_regeneration (v5.main) ---")
    t_reg = time.time()
    try:
        model, rules, _original = csfm.run_cover_regeneration()
    except Exception as exc:
        log(f"FATAL: run_cover_regeneration failed: {exc!r}")
        log("STOP — do not fabricate numbers; do not fall back to a smaller recipe.")
        return 1
    log(f"regeneration wall time: {time.time() - t_reg:.1f}s")

    ids, y, cls, uv, _tr, te = v5.load_ds()
    yv = cls[y]
    out_te, pred_te = v5.core_cover_sw(model, rules, ids, yv, cls, te, return_pred=True)
    log(
        f"core_sw (te): agree={out_te['agree']:.4f} cover={out_te['cover']:.4f} "
        f"agree_fired={out_te['agree_fired']:.4f}"
    )

    err_mask = pred_te != yv[te]
    err_rows = te[err_mask]
    n_residual = int(err_mask.sum().item())
    log(f"n_residual={n_residual}")

    codec = v5.load_codec()
    uv_cpu = uv.detach().cpu()
    ws_row_list: list[int] = []
    gold_surfaces: list[str] = []
    for r in err_rows.detach().cpu().tolist():
        r_i = int(r)
        gold_compact = int(yv[r_i].item())
        gold_raw = int(uv_cpu[gold_compact].item())
        gold_surface = codec.token_str(gold_raw)
        if cra.classify_surface(gold_surface) == "whitespace_indent":
            ws_row_list.append(r_i)
            gold_surfaces.append(gold_surface)

    n_class = len(ws_row_list)
    n_class_note = comparison_line_n_class(n_class)
    log(f"n_class={n_class} ({n_class_note})")

    ws_rows = torch.tensor(ws_row_list, dtype=torch.long, device=ids.device)
    is_open, is_close = build_is_open_close(uv, codec, device=ids.device)

    def token_str_fn(compact_id: int) -> str:
        return codec.token_str(int(uv_cpu[int(compact_id)].item()))

    n_forced = 0
    n_naive = 0
    recoverable_flags: list[bool] = []
    unit_char_space = 0
    unit_char_tab = 0
    used_default_n = 0
    unit_size_hist: Counter[int] = Counter()
    example_rec: list[dict[str, Any]] = []
    example_non: list[dict[str, Any]] = []

    for idx, r_i in enumerate(ws_row_list):
        raw_window_ids = csfm.raw_window_from_compact(ids[r_i], uv)
        context_text = codec.decode(raw_window_ids)
        ids_row = ids[r_i]
        predicted, meta = predict_structural_indent(
            context_text, ids_row, is_open, is_close, token_str_fn
        )
        naive_pred = predict_naive_indent(context_text)
        gold_surface = gold_surfaces[idx]
        recoverable = predicted == gold_surface
        recoverable_flags.append(recoverable)
        if recoverable:
            n_forced += 1
        if naive_pred == gold_surface:
            n_naive += 1

        if meta["unit_char"] == "\t":
            unit_char_tab += 1
        else:
            unit_char_space += 1
        if meta["used_default"]:
            used_default_n += 1
        unit_size_hist[int(meta["unit_size"])] += 1

        bucket = example_rec if recoverable else example_non
        if len(bucket) < 15:
            tail = context_text[-120:] if len(context_text) > 120 else context_text
            bucket.append(
                {
                    "context_tail": repr(tail),
                    "gold_surface": repr(gold_surface),
                    "predicted_surface": repr(predicted),
                    "naive_surface": repr(naive_pred),
                    "case": meta["case"],
                    "d_end": meta["d_end"],
                    "d_line_start": meta["d_line_start"],
                    "unit_char": meta["unit_char"],
                    "unit_size": meta["unit_size"],
                    "recoverable": recoverable,
                }
            )

    gate_class = n_forced / n_class if n_class else 0.0
    gate_abs = n_forced / n_residual if n_residual else 0.0
    naive_baseline_gate = n_naive / n_class if n_class else 0.0
    verdict = verdict_for(gate_class)

    log(
        f"n_forced_recoverable={n_forced} gate_class={gate_class:.6f} "
        f"gate_abs={gate_abs:.6f} naive_baseline_gate={naive_baseline_gate:.6f} "
        f"verdict={verdict}"
    )
    log(f"CEILING: {CEILING_NOTE}")

    # --- Control (reuse #88) ---
    log("--- control (statistical separability of forced-recoverable class rows) ---")
    recoverable_tensor = torch.tensor(recoverable_flags, dtype=torch.bool, device=ids.device)
    _, pred_traced, trace = clp.core_cover_sw_runner_traced(
        model, rules, ids, yv, cls, ws_rows
    )
    pred_ref = v5.core_cover_sw(model, rules, ids, yv, cls, ws_rows, return_pred=True)[1]
    parity_ok = clp.assert_trace_parity(pred_traced, pred_ref)
    labels = recoverable_tensor.to(dtype=torch.long)
    control = clp.run_control(trace["conf"], trace["gap"], labels)
    control["parity_check_passed"] = bool(parity_ok)

    if control.get("skipped"):
        log(
            f"control SKIPPED: {control['reason']} "
            f"(n_pos={control['n_pos']} n_neg={control['n_neg']})"
        )
        control_summary = f"SKIPPED ({control.get('reason', 'unknown')})"
    else:
        log(
            f"control: cv_auc={control['cv_auc']:.4f} perm_p={control['perm_p']:.4g} "
            f"passes={control['passes']}"
        )
        control_summary = (
            f"cv_auc={control['cv_auc']:.4f} perm_p={control['perm_p']:.4g} "
            f"passes={control['passes']}"
        )

    file_unit_stats = {
        "unit_char_space": unit_char_space,
        "unit_char_tab": unit_char_tab,
        "used_default": used_default_n,
        "unit_size_histogram": {str(k): int(v) for k, v in sorted(unit_size_hist.items())},
    }
    examples = example_rec + example_non

    report = {
        "n_residual": n_residual,
        "n_class": n_class,
        "n_class_vs_registered_531_note": n_class_note,
        "n_forced_recoverable": n_forced,
        "gate_class": gate_class,
        "gate_abs": gate_abs,
        "naive_baseline_gate": naive_baseline_gate,
        "verdict": verdict,
        "ceiling_note": CEILING_NOTE,
        "file_unit_stats": file_unit_stats,
        "control": control,
        "framing_note": FRAMING_NOTE,
        "honesty_note": HONESTY_NOTE,
        "examples": examples,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

    wall = time.time() - t0
    log()
    log("=" * 72)
    log("SCOREBOARD")
    log("=" * 72)
    log(f"n_residual={n_residual}")
    log(f"n_class={n_class}")
    log(f"n_forced_recoverable={n_forced}")
    log(f"gate_class={gate_class:.6f}")
    log(f"gate_abs={gate_abs:.6f}")
    log(f"naive_baseline_gate={naive_baseline_gate:.6f}")
    log(f"verdict={verdict}")
    log(f"control: {control_summary}")
    log(f"framing_note: {FRAMING_NOTE}")
    log(f"wall_time_s={wall:.1f}")
    log(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
