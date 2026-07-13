"""Wikitext accuracy pilot: energy-beam DECIDE(M>1) vs M=1 corner on real text.

Pre-registration (SIGNED, fixed thresholds — do not tune):
  /home/allans/code/PIL_WIKITEXT_ACCURACY_PREREG.md

Fixed protocol:
  - Delta threshold = 0.01
  - M-sweep = {2,3,4,5} vs M=1 corner (bit-exact serve_sw reduction)
  - beam_width = 8 for M>1 arms
  - Reading rule:
      GAIN       iff exists M with Delta_agree(M) >= +0.01
      NULL       iff |Delta_agree(M)| < 0.01 for all M  (pre-registered EXPECTED result)
      REGRESSION iff any Delta_agree(M) <= -0.01

Route for prune_events: decide() / serve_energy() do not surface beam stats, so M>1
arms call beam_decode via TextOracle exactly as serve_energy does (same rule_id /
wyly_seed / commit construction), and keep prune_events / n_steps from the beam
result. The served answer is asserted to match decide() on a smoke sample of windows.

Usage: .venv/bin/python experiments/campaign_wikitext_accuracy.py
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

# Production wikitext recipe + energy emit; PKG_SUFFIX keeps this lane's package isolated.
os.environ.setdefault("WYLY_TAG", "pythia70m")
os.environ.setdefault("WYLY_LIB", "mined")
os.environ.setdefault("WYLY_JUDGE", "cover")
os.environ.setdefault("WYLY_ONLINE", "1")
os.environ.setdefault("WYLY_COVER", "sw")
os.environ.setdefault("WYLY_DS", "wikitext")
os.environ.setdefault("WYLY_EMIT", "1")
os.environ.setdefault("WYLY_ENERGY_MODE", "1")
os.environ.setdefault("WYLY_PKG_SUFFIX", "_energy_pilot_grok")

import text_oracle  # noqa: E402
import wyly_lm_v5 as v5  # noqa: E402
from beam_engine import beam_decode, det_rank_compare  # noqa: E402
from serve_package import decide, load_package, serve_energy  # noqa: E402
from text_oracle import TextOracle  # noqa: E402

# ---------------------------------------------------------------------------
# Registered constants (fixed by the prereg — do not tune)
# ---------------------------------------------------------------------------
DELTA = 0.01
M_SWEEP = (2, 3, 4, 5)
BEAM_WIDTH = 8
COUNT_KINDS = frozenset({"ngram", "gate", "dgate", "dgate2"})
# STOP thresholds for the degenerate-beam confound check.
MIN_FRAC_WITH_COUNTS = 0.05
MIN_DET_RANK_DIFF = 0.02
# Held-out measurement: full te (seed-free split from load_ds). Documented floor is 5000.
MIN_SAMPLE = 5000
OUTPUT = REPO / "data" / "wikitext_accuracy_pilot.json"
PKG_SUFFIX = os.environ.get("WYLY_PKG_SUFFIX", "_energy_pilot_grok")
PKG_DIR = REPO / "data" / f"wyly_expert_package_v5{PKG_SUFFIX}"
RECIPE_ENV = {
    "WYLY_TAG": os.environ.get("WYLY_TAG", "pythia70m"),
    "WYLY_LIB": os.environ.get("WYLY_LIB", "mined"),
    "WYLY_JUDGE": os.environ.get("WYLY_JUDGE", "cover"),
    "WYLY_ONLINE": os.environ.get("WYLY_ONLINE", "1"),
    "WYLY_COVER": os.environ.get("WYLY_COVER", "sw"),
    "WYLY_DS": os.environ.get("WYLY_DS", "wikitext"),
    "WYLY_EMIT": os.environ.get("WYLY_EMIT", "1"),
    "WYLY_ENERGY_MODE": os.environ.get("WYLY_ENERGY_MODE", "1"),
    "WYLY_PKG_SUFFIX": PKG_SUFFIX,
}

# Expansion-count monkeypatch (TextOracle is constructed fresh inside serve_energy / our
# beam path; patch the class method once at import time).
_orig_expand = text_oracle.TextOracle.expand
_expansion_count = [0]


def _counting_expand(self, ctx):
    children = _orig_expand(self, ctx)
    _expansion_count[0] += len(children)
    return children


text_oracle.TextOracle.expand = _counting_expand


def log(msg: str = "") -> None:
    print(msg, flush=True)


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def agreement_cover(pairs: list[tuple[Any | None, Any]]) -> tuple[float, float]:
    """Token-level served agreement and cover over (served_or_None, gold) pairs.

    agree = (# correct fires) / n   — abstains count as disagreement with gold
    cover = (# fires) / n
    """
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    n_agree = 0
    n_fired = 0
    for served, gold in pairs:
        if served is not None:
            n_fired += 1
            if served == gold:
                n_agree += 1
    return n_agree / n, n_fired / n


def collect_count_pairs(rules: list[dict]) -> list[tuple[int, int]]:
    """Distinct (cnt, tot) pairs from count-bearing rule kinds in a raw manifest."""
    seen: set[tuple[int, int]] = set()
    for r in rules:
        if r.get("kind") not in COUNT_KINDS:
            continue
        c = r.get("counts")
        if c is None:
            continue
        if isinstance(c, (list, tuple)) and len(c) == 2 and not isinstance(c[0], (list, tuple, dict)):
            seen.add((int(c[0]), int(c[1])))
        elif isinstance(c, dict):
            for v in c.values():
                if v is not None and len(v) == 2:
                    seen.add((int(v[0]), int(v[1])))
    return sorted(seen)


def frac_with_counts(rules: list[dict]) -> float:
    eligible = [r for r in rules if r.get("kind") in COUNT_KINDS]
    if not eligible:
        return 0.0
    n_ok = sum(1 for r in eligible if r.get("counts") is not None)
    return n_ok / len(eligible)


def det_rank_differentiates_frac(
    count_pairs: list[tuple[int, int]],
    *,
    n_samples: int = 2000,
    seed: int = 0,
) -> float:
    """Fraction of random distinct-tuple pairs where det_rank_compare is non-zero."""
    if len(count_pairs) < 2:
        return 0.0
    rng = random.Random(seed)
    # All ordered pairs of distinct indices if small; else sample with replacement of pairs.
    n_distinct = len(count_pairs)
    max_unordered = n_distinct * (n_distinct - 1) // 2
    if max_unordered <= n_samples:
        pairs = [
            (count_pairs[i], count_pairs[j])
            for i in range(n_distinct)
            for j in range(i + 1, n_distinct)
        ]
    else:
        pairs = []
        for _ in range(n_samples):
            i, j = rng.sample(range(n_distinct), 2)
            pairs.append((count_pairs[i], count_pairs[j]))
    n_diff = sum(1 for a, b in pairs if det_rank_compare(a, b) != 0)
    return n_diff / len(pairs) if pairs else 0.0


def _commit_from_beam(beam_result: dict, oracle: TextOracle) -> dict | None:
    """Replicate serve_energy's commit packaging (answer dict or None)."""
    commit = beam_result["committed_value"]
    if commit is None:
        fallback = oracle.fallback()
        if fallback is not None:
            fallback = dict(fallback)
            fallback["cert_kind"] = "per-token"
        return fallback
    if isinstance(commit, dict):
        out = dict(commit)
        out["cert_kind"] = "per-token"
        return out
    token, meta = commit
    result = {key: value for key, value in meta.items() if key != "conf"}
    result["answer"] = token
    result["confidence"] = round(meta["conf"], 4) if meta.get("conf") is not None else 0.0
    result["cert_kind"] = "M-step-lookahead"
    return result


def decide_energy_with_stats(
    ctx: list[int],
    idioms: list,
    ngrams: dict,
    m: dict,
) -> tuple[dict | None, dict[str, int | float]]:
    """Energy-beam decide that also returns beam stats.

    M=1, beam_width=1: delegates to serve_energy (serve_sw reduction); expansions N/A.
    M>1: TextOracle + beam_decode (same as serve_energy), returning prune_events / n_steps
    and the monkeypatched expansion count.
    """
    W = m.get("W", 1)
    M = int(m.get("M", 1))
    beam_width = int(m.get("beam_width", 1))
    m_derived = m.get("_derived")
    cmap = m.get("_cmap")
    m_tau = m.get("_tau")
    wyly_seed = int(m.get("wyly_seed", 0))

    if M == 1 and beam_width == 1:
        _expansion_count[0] = 0
        t0 = time.perf_counter()
        result = serve_energy(
            ctx, idioms, ngrams, W, M=1, beam_width=1,
            m_derived=m_derived, cmap=cmap, m_tau=m_tau, wyly_seed=wyly_seed,
        )
        elapsed = time.perf_counter() - t0
        return result, {
            "prune_events": 0,
            "n_steps": 0,
            "expansions": 0,
            "wall_s": elapsed,
        }

    _expansion_count[0] = 0
    t0 = time.perf_counter()
    oracle = TextOracle(ctx, idioms, ngrams, W, m_derived=m_derived, cmap=cmap, m_tau=m_tau)
    beam_result = beam_decode(oracle, M, beam_width, "decide_energy_v1", wyly_seed)
    result = _commit_from_beam(beam_result, oracle)
    elapsed = time.perf_counter() - t0
    return result, {
        "prune_events": int(beam_result["prune_events"]),
        "n_steps": int(beam_result["n_steps"]),
        "expansions": int(_expansion_count[0]),
        "wall_s": elapsed,
    }


def ensure_package() -> Path:
    """Load or emit the energy-mode package under PKG_DIR."""
    man_path = PKG_DIR / "manifest.json"
    rebuild = os.environ.get("WYLY_PILOT_REBUILD", "") == "1"
    if man_path.exists() and not rebuild:
        log(f"cache hit: {man_path}")
        return man_path
    if rebuild:
        log(f"WYLY_PILOT_REBUILD=1 — re-running v5.main() into {PKG_DIR}")
    else:
        log(f"package missing at {man_path} — running v5.main() to emit")
    v5.main()
    if not man_path.exists():
        log(f"STOP: v5.main() finished but {man_path} still missing")
        sys.exit(2)
    return man_path


def make_manifest_views(m_base: dict) -> tuple[dict, dict[int, dict]]:
    """One corner (M=1,bw=1) and one beam manifest per M in M_SWEEP."""
    m_corner = dict(m_base)
    m_corner["cover"] = "energy-beam"
    m_corner["M"] = 1
    m_corner["beam_width"] = 1
    m_beams: dict[int, dict] = {}
    for M in M_SWEEP:
        mb = dict(m_base)
        mb["cover"] = "energy-beam"
        mb["M"] = M
        mb["beam_width"] = BEAM_WIDTH
        m_beams[M] = mb
    return m_corner, m_beams


def _smoke_answer_parity(
    te_sample: list[int],
    ids,
    uvl: list[int],
    idioms,
    ngrams,
    m_beams: dict[int, dict],
    n_check: int = 8,
) -> None:
    """Assert our beam path's served answer matches decide() on a few windows."""
    for M, m_beam in m_beams.items():
        for wi in te_sample[:n_check]:
            ctx = [uvl[c] for c in ids[wi].tolist()]
            d_ref = decide(ctx, idioms, ngrams, m_beam)
            d_ours, _ = decide_energy_with_stats(ctx, idioms, ngrams, m_beam)
            a_ref = None if d_ref is None else d_ref.get("answer")
            a_ours = None if d_ours is None else d_ours.get("answer")
            if a_ref != a_ours:
                log(f"STOP: answer mismatch vs decide() at M={M} wi={wi}: "
                    f"ref={a_ref!r} ours={a_ours!r}")
                sys.exit(3)


def main() -> None:
    log("=" * 72)
    log("Wikitext accuracy pilot — energy-beam DECIDE(M>1) vs M=1 corner")
    log("prereg: /home/allans/code/PIL_WIKITEXT_ACCURACY_PREREG.md")
    log(f"DELTA={DELTA}  M_SWEEP={list(M_SWEEP)}  beam_width={BEAM_WIDTH}")
    log(f"package dir: {PKG_DIR}")
    log("=" * 72)

    man_path = ensure_package()
    raw = json.loads(man_path.read_text())
    rules = raw.get("rules", [])
    n_rules = int(raw.get("n_rules", len(rules)))
    cover = raw.get("cover")
    W = raw.get("W")

    # ---- Non-degeneracy (BEFORE any agreement measurement) ----
    fwc = frac_with_counts(rules)
    count_pairs = collect_count_pairs(rules)
    dr_frac = det_rank_differentiates_frac(count_pairs)
    log("")
    log("--- Non-degeneracy (load-bearing confound check) ---")
    log(f"frac_with_counts     = {fwc:.6f}  "
        f"(eligible kinds {sorted(COUNT_KINDS)}; n_rules={n_rules})")
    log(f"det_rank differentiates = {dr_frac:.6f}  "
        f"(distinct (cnt,tot) pairs={len(count_pairs)})")
    if fwc < MIN_FRAC_WITH_COUNTS or dr_frac < MIN_DET_RANK_DIFF:
        log("")
        log("STOP: degenerate beam substrate")
        log(f"  frac_with_counts={fwc:.6f} (need >= {MIN_FRAC_WITH_COUNTS})")
        log(f"  det_rank_differs={dr_frac:.6f} (need >= {MIN_DET_RANK_DIFF})")
        log("  No agreement/verdict computed.")
        sys.exit(1)
    log("non-degeneracy: PASS")

    idioms, ngrams, m_base = load_package(man_path)
    m_corner, m_beams = make_manifest_views(m_base)

    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    uvl = uv.tolist()
    te_l = te.tolist()
    # Full held-out split (seed-free, fixed by load_ds). n=11477 under the production recipe.
    te_sample = te_l
    n = len(te_sample)
    if n < MIN_SAMPLE:
        log(f"STOP: held-out sample n={n} < MIN_SAMPLE={MIN_SAMPLE}")
        sys.exit(1)
    log(f"held-out windows: n={n} (full te; seed-free split from load_ds)")

    _smoke_answer_parity(te_sample, ids, uvl, idioms, ngrams, m_beams)

    # ---- M=1 corner ----
    log("")
    log("--- Measurement ---")
    corner_pairs: list[tuple[Any | None, Any]] = []
    corner_walls: list[float] = []
    t_corner0 = time.perf_counter()
    for wi in te_sample:
        ctx = [uvl[c] for c in ids[wi].tolist()]
        gold = uvl[yv[wi]]
        d, st = decide_energy_with_stats(ctx, idioms, ngrams, m_corner)
        corner_walls.append(float(st["wall_s"]))
        ans = None if d is None else d.get("answer")
        corner_pairs.append((ans, gold))
    corner_total_s = time.perf_counter() - t_corner0
    agree_corner, cover_corner = agreement_cover(corner_pairs)
    mean_wall_corner = sum(corner_walls) / n if n else 0.0
    log(f"M=1 corner: agree={agree_corner:.6f}  cover={cover_corner:.6f}  "
        f"mean_wall_s={mean_wall_corner:.6e}  total_s={corner_total_s:.1f}")

    # ---- M>1 arms (same windows as corner) ----
    per_m: dict[str, dict[str, Any]] = {}
    for M in M_SWEEP:
        m_beam = m_beams[M]
        pairs: list[tuple[Any | None, Any]] = []
        walls: list[float] = []
        sum_prune = 0
        sum_steps = 0
        sum_exp = 0
        n_fired = 0
        t0 = time.perf_counter()
        for wi in te_sample:
            ctx = [uvl[c] for c in ids[wi].tolist()]
            gold = uvl[yv[wi]]
            d, st = decide_energy_with_stats(ctx, idioms, ngrams, m_beam)
            walls.append(float(st["wall_s"]))
            ans = None if d is None else d.get("answer")
            pairs.append((ans, gold))
            # prune_bite denominator: total beam decision steps (n_steps == M per call).
            # Pattern precedent: gate_b prune_bite_rate = total_prune_events / total_decision_steps.
            sum_prune += int(st["prune_events"])
            sum_steps += int(st["n_steps"])
            sum_exp += int(st["expansions"])
            if ans is not None:
                n_fired += 1
        total_s = time.perf_counter() - t0
        agree_beam, cover_beam = agreement_cover(pairs)
        # Apples-to-apples: same n windows as corner (no silent subsetting).
        assert len(pairs) == n == len(corner_pairs)
        d_agree = agree_beam - agree_corner
        d_cover = cover_beam - cover_corner
        mean_wall = sum(walls) / n if n else 0.0
        latency_ratio = (mean_wall / mean_wall_corner) if mean_wall_corner > 0 else float("inf")
        # prune_bite: sum(prune_events) / sum(n_steps) over all n decisions (each contributes M steps).
        prune_bite = (sum_prune / sum_steps) if sum_steps else 0.0
        mean_expansions = sum_exp / n if n else 0.0
        per_m[str(M)] = {
            "agree_beam": agree_beam,
            "delta_agree": d_agree,
            "cover_beam": cover_beam,
            "delta_cover": d_cover,
            "prune_bite": prune_bite,
            "latency_ratio": latency_ratio,
            "mean_expansions": mean_expansions,
            "mean_wall_s": mean_wall,
            "sum_prune_events": sum_prune,
            "sum_n_steps": sum_steps,
            "n_fired": n_fired,
            "total_s": total_s,
        }
        log(
            f"M={M}: agree={agree_beam:.6f}  d_agree={d_agree:+.6f}  "
            f"cover={cover_beam:.6f}  d_cover={d_cover:+.6f}  "
            f"prune_bite={prune_bite:.6f}  lat_ratio={latency_ratio:.3f}  "
            f"mean_exp={mean_expansions:.2f}  total_s={total_s:.1f}"
        )

    # ---- Verdict (fixed reading rule) ----
    deltas = {int(M): per_m[str(M)]["delta_agree"] for M in M_SWEEP}
    deciding_m: int | None = None
    if any(d <= -DELTA for d in deltas.values()):
        verdict = "REGRESSION"
        deciding_m = min(deltas, key=lambda m: deltas[m])
    elif any(d >= DELTA for d in deltas.values()):
        verdict = "GAIN"
        deciding_m = max(deltas, key=lambda m: deltas[m])
    else:
        verdict = "NULL"
        deciding_m = None

    log("")
    log("=" * 72)
    log("SCOREBOARD")
    log(f"{'M':>3}  {'agree_beam':>10}  {'d_agree':>10}  {'cover_beam':>10}  "
        f"{'d_cover':>10}  {'prune_bite':>10}  {'lat_ratio':>10}  {'mean_exp':>10}")
    log(f"{'1c':>3}  {agree_corner:10.6f}  {'—':>10}  {cover_corner:10.6f}  "
        f"{'—':>10}  {'n/a':>10}  {'1.000':>10}  {'0 (n/a)':>10}")
    for M in M_SWEEP:
        r = per_m[str(M)]
        log(
            f"{M:3d}  {r['agree_beam']:10.6f}  {r['delta_agree']:+10.6f}  "
            f"{r['cover_beam']:10.6f}  {r['delta_cover']:+10.6f}  "
            f"{r['prune_bite']:10.6f}  {r['latency_ratio']:10.3f}  "
            f"{r['mean_expansions']:10.2f}"
        )
    log("-" * 72)
    log(f"frac_with_counts          = {fwc:.6f}")
    log(f"det_rank differentiates   = {dr_frac:.6f}")
    log(f"n                         = {n}")
    log(f"agree_corner              = {agree_corner:.6f}")
    log(f"cover_corner              = {cover_corner:.6f}")
    if verdict == "REGRESSION":
        log(f"*** VERDICT: REGRESSION (deciding M={deciding_m}, "
            f"d_agree={deltas[deciding_m]:+.6f}) ***")
    else:
        log(f"VERDICT: {verdict}"
            + (f" (deciding M={deciding_m})" if deciding_m is not None else " (all |d_agree| < 0.01)"))
    log("=" * 72)

    out = {
        "provenance": {
            "n_rules": n_rules,
            "cover": cover,
            "W": W,
            "package_dir": str(PKG_DIR),
            "manifest_path": str(man_path),
            "pil_git": git_commit(REPO),
            "rosetta_git": git_commit(REPO.parent / "rosetta"),
            "recipe_env": RECIPE_ENV,
            "frac_with_counts": fwc,
            "det_rank_differentiates": dr_frac,
            "n_distinct_count_pairs": len(count_pairs),
            "energy_mode": raw.get("energy_mode"),
            "schema_version": raw.get("schema_version"),
        },
        "protocol": {
            "delta": DELTA,
            "M_sweep": list(M_SWEEP),
            "beam_width": BEAM_WIDTH,
            "sample": "full te from v5.load_ds() (seed-free 0.85 holdout)",
            "n": n,
            "prune_bite_definition": (
                "sum(prune_events) / sum(n_steps) over all n decisions; "
                "each M>1 call contributes n_steps=M from beam_decode"
            ),
            "latency_metric": "mean wall-time per token (perf_counter) beam/corner",
            "expansion_metric": "TextOracle.expand monkeypatch; M=1 corner is structurally 0/N-A",
            "m_gt1_route": (
                "TextOracle + beam_decode (mirror of serve_energy); smoke-checked vs decide()"
            ),
        },
        "agree_corner": agree_corner,
        "cover_corner": cover_corner,
        "mean_wall_corner_s": mean_wall_corner,
        "per_m": per_m,
        "verdict": verdict,
        "deciding_m": deciding_m,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    log(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
