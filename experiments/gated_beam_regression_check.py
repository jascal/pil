"""Prove the gated text beam is bit-exact to the M=1 corner on full held-out text.

Uses the existing production energy package without rebuilding it, and routes every
arm through serve_package.decide so the serve_energy hard-term gate is exercised.

Usage: .venv/bin/python experiments/gated_beam_regression_check.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

os.environ.setdefault("WYLY_TAG", "pythia70m")
os.environ.setdefault("WYLY_LIB", "mined")
os.environ.setdefault("WYLY_JUDGE", "cover")
os.environ.setdefault("WYLY_ONLINE", "1")
os.environ.setdefault("WYLY_COVER", "sw")
os.environ.setdefault("WYLY_DS", "wikitext")
os.environ.setdefault("WYLY_EMIT", "1")
os.environ.setdefault("WYLY_ENERGY_MODE", "1")
os.environ.setdefault("WYLY_PKG_SUFFIX", "_energy_pilot_grok")

import wyly_lm_v5 as v5  # noqa: E402
from serve_package import decide, load_package  # noqa: E402

M_SWEEP = (2, 3, 4, 5)
BEAM_WIDTH = 8
PKG_SUFFIX = os.environ.get("WYLY_PKG_SUFFIX", "_energy_pilot_grok")
PKG_DIR = REPO / "data" / f"wyly_expert_package_v5{PKG_SUFFIX}"
RECORDED_RESULT = REPO / "data" / "wikitext_accuracy_pilot.json"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def agreement_cover(pairs: list[tuple[Any | None, Any]]) -> tuple[float, float]:
    """Token-level served agreement and cover over (served_or_None, gold) pairs."""
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


def ensure_package() -> Path:
    """Return the fixed production package, failing rather than rebuilding it."""
    man_path = PKG_DIR / "manifest.json"
    if not man_path.is_file():
        log(f"STOP: required existing package is missing: {man_path}")
        log("This proof never rebuilds or substitutes a package.")
        raise FileNotFoundError(man_path)
    log(f"cache hit: {man_path}")
    return man_path


def make_manifest_views(m_base: dict) -> tuple[dict, dict[int, dict]]:
    """One corner (M=1,bw=1) and one beam manifest per M in M_SWEEP."""
    m_corner = dict(m_base)
    m_corner["cover"] = "energy-beam"
    m_corner["M"] = 1
    m_corner["beam_width"] = 1
    m_beams: dict[int, dict] = {}
    for M in M_SWEEP:
        m_beam = dict(m_base)
        m_beam["cover"] = "energy-beam"
        m_beam["M"] = M
        m_beam["beam_width"] = BEAM_WIDTH
        m_beams[M] = m_beam
    return m_corner, m_beams


def main() -> int:
    log("=" * 88)
    log("Gated beam regression check — actual decide() path on full held-out wikitext")
    log(f"M_SWEEP={list(M_SWEEP)}  beam_width={BEAM_WIDTH}")
    log(f"package dir: {PKG_DIR}")
    log("=" * 88)

    try:
        man_path = ensure_package()
    except FileNotFoundError:
        return 2

    if not RECORDED_RESULT.is_file():
        log(f"STOP: recorded ungated result is missing: {RECORDED_RESULT}")
        return 2

    raw = json.loads(man_path.read_text())
    recorded = json.loads(RECORDED_RESULT.read_text())
    idioms, ngrams, m_base = load_package(man_path)
    m_corner, m_beams = make_manifest_views(m_base)

    ids, y, cls, uv, _tr, te = v5.load_ds()
    yv = cls[y]
    uvl = uv.tolist()
    te_sample = te.tolist()
    n = len(te_sample)
    log(
        f"manifest: n_rules={raw.get('n_rules', len(raw.get('rules', [])))}  "
        f"cover={raw.get('cover')}  energy_mode={raw.get('energy_mode')}"
    )
    log(f"held-out windows: n={n} (full te; seed-free split from load_ds)")
    log(f"recorded ungated verdict: {recorded.get('verdict')}")
    log()
    log("--- Measurement and bit-exact gate proof ---")

    corner_pairs: list[tuple[Any | None, Any]] = []
    beam_pairs: dict[int, list[tuple[Any | None, Any]]] = {
        M: [] for M in M_SWEEP
    }
    for position, wi in enumerate(te_sample):
        ctx = [uvl[c] for c in ids[wi].tolist()]
        gold = uvl[yv[wi]]
        d_corner = decide(ctx, idioms, ngrams, m_corner)
        corner_answer = None if d_corner is None else d_corner.get("answer")
        corner_pairs.append((corner_answer, gold))

        for M in M_SWEEP:
            d_beam = decide(ctx, idioms, ngrams, m_beams[M])
            if d_beam != d_corner:
                log("STOP: hard-gate mismatch (first mismatch)")
                log(f"  position={position}  window_index={wi}  M={M}")
                log(f"  ctx={ctx!r}")
                log(f"  corner={d_corner!r}")
                log(f"  beam={d_beam!r}")
                return 1
            beam_answer = None if d_beam is None else d_beam.get("answer")
            beam_pairs[M].append((beam_answer, gold))

        if position > 0 and position % 2500 == 0:
            log(f"  ... progress {position}/{n}")

    agree_corner, cover_corner = agreement_cover(corner_pairs)
    per_m: dict[int, dict[str, float]] = {}
    passed = True
    for M in M_SWEEP:
        agree_beam, cover_beam = agreement_cover(beam_pairs[M])
        delta_agree = agree_beam - agree_corner
        delta_cover = cover_beam - cover_corner
        recorded_delta = float(recorded["per_m"][str(M)]["delta_agree"])
        per_m[M] = {
            "agree_beam": agree_beam,
            "cover_beam": cover_beam,
            "delta_agree": delta_agree,
            "delta_cover": delta_cover,
            "recorded_delta_agree": recorded_delta,
        }
        passed = passed and delta_agree == 0.0 and delta_cover == 0.0

    log()
    log("=" * 88)
    log("SCOREBOARD")
    log(
        f"{'M':>3}  {'agree_beam':>12}  {'gated d_agree':>14}  "
        f"{'ungated d_agree':>16}  {'cover_beam':>12}  {'gated d_cover':>14}"
    )
    log(
        f"{'1c':>3}  {agree_corner:12.6f}  {'—':>14}  {'—':>16}  "
        f"{cover_corner:12.6f}  {'—':>14}"
    )
    for M in M_SWEEP:
        row = per_m[M]
        log(
            f"{M:3d}  {row['agree_beam']:12.6f}  "
            f"{row['delta_agree']:+14.6f}  "
            f"{row['recorded_delta_agree']:+16.6f}  "
            f"{row['cover_beam']:12.6f}  {row['delta_cover']:+14.6f}"
        )
    log("-" * 88)
    log(f"bit-exact comparisons = {n * len(M_SWEEP)}; mismatches = 0")
    if passed:
        log("VERDICT: PASS — bit-exact to M=1 corner; text regression eliminated")
        return 0
    log("VERDICT: FAIL — at least one gated agreement/cover delta is non-zero")
    return 1


if __name__ == "__main__":
    sys.exit(main())
