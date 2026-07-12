"""Independent cross-check of slice #98 whitespace-only residual mass (code cover).

Re-implements only the whitespace classification from first principles. Does not
read or import campaign_code_residual_anatomy.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# Code recipe. Plain assignment so setdefaults in the sudoku harness no-op when
# that module is imported next. Explicitly pin babi-only flags off (empty/"0")
# BEFORE importing campaign_sudoku_forced_move, whose setdefault would otherwise
# inject CONCEPTS/POINTER/… and pollute v5's import-time STATE path.
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
import wyly_lm_v5 as v5  # isort: skip
import campaign_sudoku_forced_move as sudoku  # isort: skip

WHITESPACE_CHARS = frozenset({" ", "\n", "\t"})
PRIOR_N_RESIDUAL = 5109
PRIOR_WHITESPACE_MASS = 0.1039
MASS_BAR = 0.10


def is_pure_whitespace(surface: str) -> bool:
    """True iff surface is non-empty and consists only of space / newline / tab."""
    return len(surface) > 0 and all(c in WHITESPACE_CHARS for c in surface)


def main() -> None:
    print("=== verify_code_anatomy: independent whitespace residual cross-check ===", flush=True)
    print("step 1-2: run_cover_regeneration (v5.main + capture) ...", flush=True)
    model, _rules, _orig = sudoku.run_cover_regeneration()

    print("step 3: load_ds ...", flush=True)
    ids, y, cls, uv, _tr, te = v5.load_ds()
    yv = cls[y]

    print("step 4: core_cover_sw(..., return_pred=True) ...", flush=True)
    _, pred = v5.core_cover_sw(model, model.rules, ids, yv, cls, te, return_pred=True)
    err = te[pred != yv[te]]
    n_residual = int(len(err))
    print(f"n_residual = {n_residual}", flush=True)

    print("step 5-6: load_codec + classify pure-whitespace gold surfaces ...", flush=True)
    codec = v5.load_codec()
    n_whitespace = 0
    for i in err.tolist():
        gold_compact = int(yv[i].item())
        real_id = int(uv[gold_compact].item())
        surface = codec.token_str(real_id)
        if is_pure_whitespace(surface):
            n_whitespace += 1

    if n_residual == 0:
        print("ERROR: n_residual is 0; cannot compute whitespace_mass.", flush=True)
        raise SystemExit(1)

    whitespace_mass = n_whitespace / n_residual

    # --- final report ---
    print(flush=True)
    print("=" * 72, flush=True)
    print("FINAL REPORT: independent whitespace residual mass (slice #98 cross-check)", flush=True)
    print("=" * 72, flush=True)
    print(f"n_residual = {n_residual}", flush=True)
    print(f"n_whitespace = {n_whitespace}", flush=True)
    print(f"whitespace_mass = {whitespace_mass:.4f}", flush=True)
    print(flush=True)
    print(
        f"PRIOR SLICE #98 reported whitespace_indent mass = {PRIOR_WHITESPACE_MASS:.4f} "
        f"on n_residual ~{PRIOR_N_RESIDUAL}.",
        flush=True,
    )

    residual_delta = n_residual - PRIOR_N_RESIDUAL
    if residual_delta == 0:
        residual_note = (
            f"n_residual matches the codex-reported ~{PRIOR_N_RESIDUAL} exactly."
        )
    else:
        residual_note = (
            f"n_residual differs from codex-reported ~{PRIOR_N_RESIDUAL} by "
            f"{residual_delta:+d} (this run: {n_residual})."
        )
    print(residual_note, flush=True)

    mass_delta = whitespace_mass - PRIOR_WHITESPACE_MASS
    print(
        f"whitespace_mass vs codex 0.1039: this run {whitespace_mass:.4f} "
        f"(delta {mass_delta:+.4f}).",
        flush=True,
    )

    if whitespace_mass >= MASS_BAR:
        verdict = (
            "AGREE with codex: ESCALATE (whitespace_indent clears the 0.10 bar)"
        )
    else:
        verdict = (
            "DISAGREE / boundary is implementation-sensitive: PLATEAU_N1 by this "
            "independent measurement"
        )
    print(f"VERDICT: {verdict}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise
