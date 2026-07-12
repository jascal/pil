"""Slice #98: descriptive anatomy of the code cover's residual gold tokens."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "data" / "code_residual_anatomy.json"

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
for _key, _value in WYLY_ENV.items():
    os.environ.setdefault(_key, _value)

CATEGORIES = (
    "whitespace_indent",
    "bracket",
    "terminator_sep",
    "operator",
    "keyword",
    "identifier",
    "literal",
    "comment",
    "other",
)
CONSTRAINT_SHAPED = frozenset({"whitespace_indent", "bracket", "terminator_sep"})
LIVE_CANDIDATES = frozenset({"whitespace_indent", "terminator_sep"})
DECISION_THRESHOLD = 0.10

BRACKETS = frozenset("()[]{}")
TERMINATORS = frozenset({";", ","})
OPERATOR_CHARS = frozenset("+-*/=<>&|!%^~.:?")
C_KEYWORDS = frozenset(
    {
        "if",
        "for",
        "while",
        "do",
        "return",
        "int",
        "void",
        "char",
        "struct",
        "union",
        "enum",
        "typedef",
        "static",
        "const",
        "unsigned",
        "signed",
        "long",
        "short",
        "float",
        "double",
        "sizeof",
        "break",
        "continue",
        "switch",
        "case",
        "default",
        "goto",
        "extern",
    }
)
_FULL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAME_FRAGMENT = re.compile(r"^[A-Za-z0-9_]+$")
_LEADING_ALPHA = re.compile(r"^([A-Za-z]+)(?![A-Za-z])")
_ESCAPED_LITERAL = re.compile(r"^\\(?:.|[0-7]{1,3}|x[0-9A-Fa-f]+)$", re.DOTALL)


def classify_surface(raw_surface: str) -> str:
    """Classify one unstripped gold-token surface by the preregistered priority."""
    if raw_surface and all(character in " \n\t" for character in raw_surface):
        return "whitespace_indent"

    surface = raw_surface.strip()
    if surface in BRACKETS:
        return "bracket"
    if surface in TERMINATORS:
        return "terminator_sep"
    if surface and all(character in OPERATOR_CHARS for character in surface):
        return "operator"

    leading_alpha = _LEADING_ALPHA.match(surface)
    if surface in C_KEYWORDS or (
        leading_alpha is not None and leading_alpha.group(1) in C_KEYWORDS
    ):
        return "keyword"

    # A digit-leading fragment is kept for the later, explicit numeric-literal rule.
    if _FULL_IDENTIFIER.fullmatch(surface) or (
        surface
        and not surface[0].isdigit()
        and _NAME_FRAGMENT.fullmatch(surface) is not None
    ):
        return "identifier"

    if surface[:1].isdigit() or surface in {"'", '"'} or _ESCAPED_LITERAL.fullmatch(surface):
        return "literal"
    if any(marker in surface for marker in ("//", "/*", "*/")):
        return "comment"
    return "other"


def _runtime_modules():
    """Import the model-dependent modules only when the measurement is run."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "experiments"))
    import wyly_lm_v5 as v5

    # Import only after v5 has bound the code recipe above.
    # isort: split
    import campaign_sudoku_forced_move as csfm

    return v5, csfm


def _cache_status(state_path: Path, before_mtime_ns: int | None) -> str:
    """Distinguish an untouched existing state from a newly written regeneration."""
    if before_mtime_ns is None or not state_path.exists():
        return "regenerated"
    return "cache_hit" if state_path.stat().st_mtime_ns == before_mtime_ns else "regenerated"


def main() -> int:
    v5, csfm = _runtime_modules()
    state_path = Path(v5.STATE)
    before_mtime_ns = state_path.stat().st_mtime_ns if state_path.exists() else None

    print("--- run_cover_regeneration (v5.main) ---", flush=True)
    regeneration_start = time.monotonic()
    model, rules, _original = csfm.run_cover_regeneration()
    regeneration_seconds = time.monotonic() - regeneration_start
    cache_status = _cache_status(state_path, before_mtime_ns)

    ids, y, cls, uv, _tr, te = v5.load_ds()
    yv = cls[y]
    _, pred = v5.core_cover_sw(model, rules, ids, yv, cls, te, return_pred=True)
    mism = pred != yv[te]
    err = te[mism]
    n_residual = int(mism.sum().item())

    codec = v5.load_codec()
    gold_classes = yv[err].detach().cpu().tolist()
    uv_cpu = uv.detach().cpu()
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {category: [] for category in CATEGORIES}
    for class_index in gold_classes:
        raw_surface = codec.token_str(int(uv_cpu[int(class_index)].item()))
        category = classify_surface(raw_surface)
        counts[category] += 1
        rendered = repr(raw_surface)
        if len(examples[category]) < 5 and rendered not in examples[category]:
            examples[category].append(rendered)

    mass_unsorted = {
        category: counts[category] / n_residual if n_residual else 0.0
        for category in CATEGORIES
    }
    sorted_categories = sorted(CATEGORIES, key=lambda category: (-mass_unsorted[category], category))
    mass = {category: mass_unsorted[category] for category in sorted_categories}
    constraint_shaped_mass = sum(mass_unsorted[category] for category in CONSTRAINT_SHAPED)
    live_candidate_mass = {
        category: mass_unsorted[category] for category in sorted(LIVE_CANDIDATES)
    }
    winning_class = max(LIVE_CANDIDATES, key=lambda category: mass_unsorted[category])
    winning_mass = mass_unsorted[winning_class]
    verdict = "ESCALATE" if winning_mass >= DECISION_THRESHOLD else "PLATEAU_N1"

    report = {
        "n_residual": n_residual,
        "mass": mass,
        "constraint_shaped_mass": constraint_shaped_mass,
        "live_candidate_mass": live_candidate_mass,
        "max_live_candidate_mass": winning_mass,
        "winning_class": winning_class if verdict == "ESCALATE" else None,
        "verdict": verdict,
        "cache_status": cache_status,
        "regeneration_seconds": regeneration_seconds,
        "examples": examples,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

    print("\n" + "=" * 72)
    print("CODE RESIDUAL ANATOMY — SLICE #98")
    print("=" * 72)
    print(
        f"cover state: {cache_status}; run_cover_regeneration wall-clock: "
        f"{regeneration_seconds:.3f}s"
    )
    print(f"n_residual: {n_residual}")
    print("mass distribution:")
    for category, category_mass in mass.items():
        print(f"  {category}: {category_mass:.6f}")
    print(f"constraint_shaped_mass: {constraint_shaped_mass:.6f}")
    print("live_candidate_mass:")
    for category, category_mass in live_candidate_mass.items():
        print(f"  {category}: {category_mass:.6f}")
    print(f"max_live_candidate: {winning_class} ({winning_mass:.6f})")
    if verdict == "ESCALATE":
        print(f"VERDICT: ESCALATE — {winning_class} mass {winning_mass:.6f}")
    else:
        print(
            "VERDICT: PLATEAU_N1 — computed-route-on-code plateaued at register n=1; "
            "recipe plateau, achievability remains open"
        )
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
