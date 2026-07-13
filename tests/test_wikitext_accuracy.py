"""Fast unit tests for the wikitext accuracy pilot (hand-built toy fixtures only).

No full dataset/model load. Mirrors tests/test_beam_engine.py and
tests/test_gate_b_pilot.py style.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

from beam_engine import beam_decode, det_rank_compare  # noqa: E402
from serve_package import decide, serve_sw  # noqa: E402
from text_oracle import TextOracle  # noqa: E402

from experiments import campaign_wikitext_accuracy as pilot  # noqa: E402


def _ngram(out, conf, cite, counts):
    """ngrams[k][suffix] = (out, basis, cite, conf, stratum, origin, counts)."""
    return (out, "observational", cite, conf, 1, "teacher", counts)


def _strip_cert(d):
    if d is None:
        return None
    out = dict(d)
    out.pop("cert_kind", None)
    return out


# ---------------------------------------------------------------------------
# (a) Non-degenerate beam: det_rank discriminates; None/equal rates tie
# ---------------------------------------------------------------------------

def test_a_det_rank_discriminates_and_ties():
    # Higher rate ranks better (returns -1 so it sorts first under ascending cmp).
    assert det_rank_compare((9, 10), (1, 10)) == -1
    assert det_rank_compare((1, 10), (9, 10)) == 1
    # Negative controls: None or equal rates must tie.
    assert det_rank_compare(None, (9, 10)) == 0
    assert det_rank_compare((9, 10), None) == 0
    assert det_rank_compare(None, None) == 0
    assert det_rank_compare((5, 10), (5, 10)) == 0
    assert det_rank_compare((1, 2), (2, 4)) == 0  # equal rates via cross-multiply


def test_a_beam_commits_higher_rate_candidate():
    """Two same-conf answers; beam ranks by det_rank and picks the higher rate.

    Order puts the *lower*-rate rule first so serve_sw (conf-tie → first) would
    commit 20; M=1 with beam_width>1 takes the genuine beam path (not the serve_sw
    short-circuit) and must commit 10 via det_rank — proving non-hash discrimination.
    """
    idioms = [
        {
            "kind": "gate", "id": "lo", "frame": {}, "slot": 1,
            "table": {1: 20}, "confs": {1: 0.8}, "counts": {1: (1, 10)},
            "cite": "lo", "stratum": 1, "origin": "teacher",
        },
        {
            "kind": "gate", "id": "hi", "frame": {}, "slot": 1,
            "table": {1: 10}, "confs": {1: 0.8}, "counts": {1: (9, 10)},
            "cite": "hi", "stratum": 1, "origin": "teacher",
        },
    ]
    ngrams = defaultdict(dict)
    ctx = [1]
    assert det_rank_compare((9, 10), (1, 10)) == -1
    # Conf-tie first-in-order baseline would pick lo's answer.
    d_sw = serve_sw(ctx, idioms, ngrams, 1, m_tau=0.0)
    assert d_sw is not None and d_sw["answer"] == 20

    # M=1 + beam_width>1: beam_decode ranks first-step candidates by det_rank.
    m = {
        "cover": "energy-beam", "W": 1, "M": 1, "beam_width": 4,
        "_derived": [], "_cmap": {}, "_tau": 0.0, "wyly_seed": 0,
    }
    d, st = pilot.decide_energy_with_stats(ctx, idioms, ngrams, m)
    assert d is not None
    assert d["answer"] == 10, f"expected higher-rate answer 10, got {d!r}"
    assert st["n_steps"] == 1
    # Negative control: equal counts => det_rank ties.
    assert det_rank_compare((5, 10), (5, 10)) == 0


# ---------------------------------------------------------------------------
# (b) M=1 corner == serve_sw bit-exact (modulo cert_kind)
# ---------------------------------------------------------------------------

def test_b_m1_corner_matches_serve_sw_fire_and_abstain():
    # Firing case: single ngram on ctx.
    ngrams_fire = defaultdict(dict)
    ngrams_fire[1][(7,)] = _ngram(99, 0.9, "ng-fire", (9, 10))
    idioms: list = []
    ctx_fire = [7]
    W = 1

    d_sw = serve_sw(ctx_fire, idioms, ngrams_fire, W)
    m_e = {
        "cover": "energy-beam", "W": W, "M": 1, "beam_width": 1,
        "_derived": [], "_cmap": {}, "_tau": 0.0,
    }
    d_e = decide(ctx_fire, idioms, ngrams_fire, m_e)
    assert d_sw is not None and d_e is not None
    assert d_e.get("cert_kind") == "per-token"
    assert _strip_cert(d_e) == d_sw
    assert d_e["answer"] == 99

    # Abstaining case: empty package, no rules fire.
    ngrams_empty = defaultdict(dict)
    ctx_abs = [3, 4, 5]
    d_sw_a = serve_sw(ctx_abs, idioms, ngrams_empty, W)
    d_e_a = decide(ctx_abs, idioms, ngrams_empty, m_e)
    assert d_sw_a is None
    assert d_e_a is None


# ---------------------------------------------------------------------------
# (c) prune_events == 0 on text multi-step beam (no hard/legality term)
# ---------------------------------------------------------------------------

def test_c_prune_events_zero_on_text_multistep():
    """Chain of ngrams so the beam commits over M>=2 steps without empty expands."""
    ngrams = defaultdict(dict)
    # step0: ctx=(1,) -> 2; step1: ctx=(1,2) -> 3; step2: ctx=(1,2,3) -> 4
    ngrams[1][(1,)] = _ngram(2, 0.9, "s0", (9, 10))
    ngrams[1][(2,)] = _ngram(3, 0.8, "s1", (8, 10))
    ngrams[1][(3,)] = _ngram(4, 0.7, "s2", (7, 10))
    # Second candidate via a gate so expand returns 2 children at step 0.
    idioms = [
        {
            "kind": "gate", "id": "alt", "frame": {}, "slot": 1,
            "table": {1: 5}, "confs": {1: 0.5}, "counts": {1: (2, 10)},
            "cite": "alt", "stratum": 1, "origin": "teacher",
        },
    ]
    # Also wire continuations for the alt branch so expand never returns empty mid-beam.
    ngrams[1][(5,)] = _ngram(6, 0.4, "alt-s1", (1, 10))

    oracle = TextOracle((1,), idioms, ngrams, W=1)
    result = beam_decode(oracle, M=2, beam_width=4, rule_id="decide_energy_v1", wyly_seed=0)
    assert result["n_steps"] == 2
    assert result["prune_events"] == 0, f"expected no prune on text, got {result!r}"
    assert result["committed_value"] is not None

    # Via the pilot helper as well.
    m = {
        "cover": "energy-beam", "W": 1, "M": 2, "beam_width": 4,
        "_derived": [], "_cmap": {}, "_tau": 0.0, "wyly_seed": 0,
    }
    d, st = pilot.decide_energy_with_stats([1], idioms, ngrams, m)
    assert d is not None
    assert st["prune_events"] == 0
    assert st["n_steps"] == 2


# ---------------------------------------------------------------------------
# (d) agreement_cover pure helper
# ---------------------------------------------------------------------------

def test_d_agreement_cover_counter():
    # 5 windows: correct fire, incorrect fire, abstain, correct fire, abstain
    pairs = [
        (10, 10),   # correct fire
        (7, 10),    # incorrect fire
        (None, 3),  # abstain
        (5, 5),     # correct fire
        (None, 9),  # abstain
    ]
    # agree = 2/5 = 0.4; cover = 3/5 = 0.6
    agree, cover = pilot.agreement_cover(pairs)
    assert agree == 0.4
    assert cover == 0.6

    assert pilot.agreement_cover([]) == (0.0, 0.0)
    assert pilot.agreement_cover([(None, 1), (None, 2)]) == (0.0, 0.0)
    assert pilot.agreement_cover([(1, 1), (2, 2)]) == (1.0, 1.0)


def test_frac_with_counts_and_collect_pairs():
    rules = [
        {"kind": "ngram", "counts": [3, 5]},
        {"kind": "ngram"},  # missing counts
        {"kind": "gate", "counts": {"1": [9, 10], "2": [1, 10]}},
        {"kind": "relation"},  # not a count kind
        {"kind": "dgate2", "counts": None},
    ]
    assert abs(pilot.frac_with_counts(rules) - 2 / 4) < 1e-12  # ngram,ngram,gate,dgate2
    pairs = pilot.collect_count_pairs(rules)
    assert (3, 5) in pairs
    assert (9, 10) in pairs
    assert (1, 10) in pairs
    # det_rank differentiates on these pairs should be material
    frac = pilot.det_rank_differentiates_frac(pairs, n_samples=100, seed=0)
    assert frac > 0.5
