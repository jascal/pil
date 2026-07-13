"""Independent verification of slice #99's indent-forcedness verdict.

This script implements the registered predictor and naive baseline from the
preregistration text, without consulting the other #99 implementation or its output.

The operationalization is deliberately causal: it does not perform same-line
lookahead to deduct one unit for a closer that has not yet appeared (for example, a
``}`` in the token after the whitespace token being predicted). Such a deduction
would peek past the predicted position and break the left-context-only discipline of
the other compute-registers, including ``mate_feature`` and ``depth_feature``. This
is a documented, conservative reading of the registered rule, not an oversight.
Openers and closers already present anywhere in the left context are reflected in
the certified depth feature and therefore require no special case here.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "data" / "indent_forcedness_verify.json"

# wyly_lm_v5 binds these recipe constants at import time, so set the registered
# code-cover recipe before importing any experiment modules.
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

_LEADING_WHITESPACE = re.compile(r"^[ \t]*")


def _runtime_modules():
    """Import model-dependent modules only after fixing the registered environment."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "experiments"))
    import wyly_lm_v5 as v5

    # Import only after v5 has bound the code recipe above.
    # isort: split
    import campaign_sudoku_forced_move as csfm

    return v5, csfm


def infer_indent_unit(context_text: str) -> tuple[str, int, list[str]]:
    """Infer this window's file-local indentation unit from complete lines only.

    The first split entry may be a truncated line and the last is empty or an
    in-progress line, so neither is evidence. Blank complete lines are also ignored.
    Any tab evidence selects one tab per level. Otherwise the unit is the smallest
    mode of nonzero absolute changes between consecutive space-indent counts, with
    four spaces as the no-evidence default.
    """
    lines = context_text.split("\n")
    complete_lines = lines[1:-1]
    indents = [
        _LEADING_WHITESPACE.match(line).group()
        for line in complete_lines
        if line.strip()
    ]
    if any("\t" in indent for indent in indents):
        return "\t", 1, lines

    counts = [len(indent) for indent in indents]
    differences = [
        abs(current - previous)
        for previous, current in zip(counts, counts[1:], strict=False)
        if current != previous
    ]
    if not differences:
        return " ", 4, lines
    frequencies = Counter(differences)
    highest_frequency = max(frequencies.values())
    unit_size = min(size for size, frequency in frequencies.items() if frequency == highest_frequency)
    return " ", unit_size, lines


def remaining_indent(context_text: str, target: str) -> str:
    """Apply the registered causal newline/tail handling to a full target indent."""
    last_nl = context_text.rfind("\n")
    if last_nl == -1:
        return "\n" + target

    tail = context_text[last_nl + 1 :]
    if tail == "":
        return target
    if all(character in " \t" for character in tail):
        return target[len(tail) :] if len(target) > len(tail) else ""

    # Inline whitespace is intentionally not filtered out: the same uniform rule
    # applies, and will normally score these cases as misses.
    return target


def verdict(gate_class: float) -> str:
    """Apply the verbatim registered thresholds for slice #99."""
    if gate_class >= 0.5:
        return "FIRES"
    if gate_class < 0.2:
        return "DEAD"
    return "MIDDLE (escalate)"


def main() -> int:
    v5, csfm = _runtime_modules()
    model, rules, _original = csfm.run_cover_regeneration()

    ids, y, cls, uv, _tr, te = v5.load_ds()
    yv = cls[y]
    _, pred = v5.core_cover_sw(model, rules, ids, yv, cls, te, return_pred=True)
    mism = pred != yv[te]
    err = te[mism]
    n_residual = len(err)

    codec = v5.load_codec()
    uv_cpu = uv.detach().cpu()
    err_rows = err.detach().cpu().tolist()
    err_ws_rows = []
    for row in err_rows:
        class_id = int(yv[row].item())
        raw_surface = codec.token_str(int(uv_cpu[class_id].item()))
        if raw_surface and all(character in " \n\t" for character in raw_surface):
            err_ws_rows.append(row)

    err_ws = torch.tensor(err_ws_rows, dtype=torch.long, device=ids.device)
    n_class = len(err_ws_rows)

    # Every scored row must already be a cover error; this is the registered
    # "AND the cover predicted otherwise" clause, asserted rather than recomputed.
    assert bool(torch.all(pred[mism] != yv[err]).item())

    opl, cll = v5.bracket_sets(uv, codec)
    is_open = torch.zeros(len(uv), dtype=torch.bool, device=ids.device)
    is_close = torch.zeros(len(uv), dtype=torch.bool, device=ids.device)
    is_open[opl] = True
    is_close[cll] = True
    depths = v5.depth_feature(ids[err_ws], is_open, is_close)

    n_forced_recoverable = 0
    n_naive_recoverable = 0
    recoverable_examples: list[dict[str, object]] = []
    missed_examples: list[dict[str, object]] = []

    for position, row in enumerate(err_ws_rows):
        raw_ids = uv_cpu[ids[row].detach().cpu()].tolist()
        context_text = codec.decode(raw_ids)
        class_id = int(yv[row].item())
        gold = codec.token_str(int(uv_cpu[class_id].item()))
        depth = int(depths[position].item())

        unit_char, unit_size, lines = infer_indent_unit(context_text)
        # The certified current depth incorporates same-depth continuation and all
        # prior +1 opener / -1 closer effects in this row's left context.
        target = unit_char * (unit_size * depth)
        predicted = remaining_indent(context_text, target)
        forced_recoverable = predicted == gold
        n_forced_recoverable += int(forced_recoverable)

        previous_complete_line = lines[-2] if len(lines) >= 2 else ""
        prev_indent = _LEADING_WHITESPACE.match(previous_complete_line).group()
        naive_predicted = remaining_indent(context_text, prev_indent)
        n_naive_recoverable += int(naive_predicted == gold)

        example = {
            "row": row,
            "gold": gold,
            "predicted": predicted,
            "unit_char": unit_char,
            "unit_size": unit_size,
            "depth": depth,
            "forced_recoverable": forced_recoverable,
        }
        destination = recoverable_examples if forced_recoverable else missed_examples
        if len(destination) < 2:
            destination.append(example)

    gate_class = n_forced_recoverable / n_class if n_class else 0.0
    naive_gate = n_naive_recoverable / n_class if n_class else 0.0
    gate_absolute_residual_points = gate_class * n_class / n_residual if n_residual else 0.0
    result_verdict = verdict(gate_class)
    examples = recoverable_examples + missed_examples

    report = {
        "n_residual": n_residual,
        "n_class": n_class,
        "n_forced_recoverable": n_forced_recoverable,
        "gate_class": gate_class,
        "n_naive_recoverable": n_naive_recoverable,
        "naive_gate": naive_gate,
        "gate_absolute_residual_points": gate_absolute_residual_points,
        "verdict": result_verdict,
        "examples": examples,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

    print("\n" + "=" * 72)
    print("INDENT-FORCEDNESS INDEPENDENT VERIFICATION — SLICE #99")
    print("=" * 72)
    print(f"n_residual: {n_residual}")
    print(f"n_class: {n_class}")
    print(f"n_forced_recoverable: {n_forced_recoverable}")
    print(f"gate_class: {gate_class:.6f}")
    print(f"n_naive_recoverable: {n_naive_recoverable}")
    print(f"naive_gate: {naive_gate:.6f}")
    print(f"gate_absolute_residual_points: {gate_absolute_residual_points:.6f}")
    print(f"verdict: {result_verdict}")
    print("worked examples (repr-escaped):")
    if not examples:
        print("  none")
    for example in examples:
        print(
            f"  row={example['row']} recoverable={example['forced_recoverable']} "
            f"gold={example['gold']!r} predicted={example['predicted']!r} "
            f"unit=({example['unit_char']!r}, {example['unit_size']}) D={example['depth']}"
        )
    if gate_class < 0.2:
        agreement = "AGREE with the DEAD/plateau verdict"
    else:
        agreement = "the DEAD/MIDDLE boundary is operationalization-sensitive"
    print(f"Conclusion: gate_class={gate_class:.6f} is {'<' if gate_class < 0.2 else '>='} 0.2; {agreement}.")
    print(
        f"Direct comparison: this gate_class={gate_class:.6f}; grok reported 0.102; "
        f"difference={gate_class - 0.102:+.6f}."
    )
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
