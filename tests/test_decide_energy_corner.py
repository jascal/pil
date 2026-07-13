"""DECIDE_ENERGY Slice 1: M=1/beam_width=1 corner is bit-exact to serve_sw.

Hand-built manifests (no training pipeline). Two covers share identical rules;
decide() must agree on answer/confidence/meta for every context, with the sole
additive difference that energy-beam stamps cert_kind="per-token" on commits.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

from serve_package import decide, load_package  # noqa: E402

# Shared rule inventory for the bit-exact gate (ngram + gate + dgate + relation).
RULES = [
    # High-confidence unigram: tau-gate fast path (context ending in 5).
    {
        "kind": "ngram", "id": "ng5", "ctx": [5], "out": 50,
        "confidence": 0.8, "basis": "observational", "citation": "ng5",
    },
    # Stratum-1 low-confidence unigram (context ending in 7).
    {
        "kind": "ngram", "id": "ng7s1", "ctx": [7], "out": 70,
        "confidence": 0.2, "basis": "observational", "citation": "ng7s1",
        "stratum": 1,
    },
    # Two equal-confidence gates on token 9 — first-in-manifest wins (strict >).
    {
        "kind": "gate", "id": "g9a", "frame": {}, "slot": 1,
        "table": {"9": 100}, "confs": {"9": 0.55},
        "citation": "g9a",
    },
    {
        "kind": "gate", "id": "g9b", "frame": {}, "slot": 1,
        "table": {"9": 200}, "confs": {"9": 0.55},
        "citation": "g9b",
    },
    # Stratum-2 gate on token 7: higher conf than ng7s1 → pool fallthrough winner.
    {
        "kind": "gate", "id": "g7s2", "frame": {}, "slot": 1,
        "table": {"7": 71}, "confs": {"7": 0.6},
        "citation": "g7s2", "stratum": 2,
    },
    # dgate over recent-member feature (members include 10; table key f:last).
    {
        "kind": "dgate", "id": "dg", "feature": "m0",
        "table": {"10:11": 42}, "confs": {"10:11": 0.7},
        "citation": "dg",
    },
    # relation: eq last two equal → copy last (offset 1).
    {
        "kind": "relation", "id": "rel", "eq": [[1, 2]], "copy": 1,
        "confidence": 0.9, "citation": "rel",
    },
]

DERIVED = [
    {"id": "m0", "kind": "recent-member", "members": [10], "succ": 0},
]

STRATA_TAU = 0.35
W = 2


def _write_pair(tmp_path):
    base = {
        "W": W,
        "strata_tau": STRATA_TAU,
        "rules": RULES,
        "derived": DERIVED,
        "origin": "teacher",
    }
    path_sw = tmp_path / "manifest_sw.json"
    path_e = tmp_path / "manifest_e.json"
    man_sw = {**base, "cover": "support-weighted"}
    man_e = {
        **base,
        "cover": "energy-beam",
        "M": 1,
        "beam_width": 1,
    }
    path_sw.write_text(json.dumps(man_sw))
    path_e.write_text(json.dumps(man_e))
    return path_sw, path_e


def _strip_cert(d):
    """Energy side stamps cert_kind; strip it so the rest can be compared bit-exact."""
    if d is None:
        return None
    out = dict(d)
    out.pop("cert_kind", None)
    return out


def _assert_parity(ctx, idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e):
    sw = decide(ctx, idioms_sw, ngrams_sw, m_sw)
    en = decide(ctx, idioms_e, ngrams_e, m_e)
    if sw is None:
        assert en is None, f"ctx={ctx}: sw abstained but energy returned {en!r}"
        return None
    assert en is not None, f"ctx={ctx}: sw fired {sw!r} but energy abstained"
    assert en.get("cert_kind") == "per-token", (
        f"ctx={ctx}: energy commit missing cert_kind=per-token: {en!r}")
    assert _strip_cert(en) == sw, (
        f"ctx={ctx}: bit-exact miss\n  sw={sw!r}\n  energy(sans cert)={_strip_cert(en)!r}")
    return en


@pytest.fixture
def packages(tmp_path):
    path_sw, path_e = _write_pair(tmp_path)
    idioms_sw, ngrams_sw, m_sw = load_package(path_sw)
    idioms_e, ngrams_e, m_e = load_package(path_e)
    assert m_sw["cover"] == "support-weighted"
    assert m_e["cover"] == "energy-beam"
    assert m_e.get("M", 1) == 1 and m_e.get("beam_width", 1) == 1
    return idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e


def test_corner_high_conf_tau_fast_path(packages):
    """(a) stratum-1 candidate with conf >= strata_tau → fast-return path."""
    idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e = packages
    en = _assert_parity([5], idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e)
    assert en["answer"] == 50
    assert en["confidence"] == 0.8
    assert en["stratum"] == 1


def test_corner_abstain(packages):
    """(b) no rule matches → both decide() return None (no cert_kind)."""
    idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e = packages
    sw = decide([99, 98], idioms_sw, ngrams_sw, m_sw)
    en = decide([99, 98], idioms_e, ngrams_e, m_e)
    assert sw is None and en is None


def test_corner_stratum2_pool_fallthrough(packages):
    """(c) stratum-1 conf < tau, higher stratum-2 conf wins the pool."""
    idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e = packages
    en = _assert_parity([7], idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e)
    # s1 ngram conf 0.2 < tau 0.35; s2 gate conf 0.6 wins pool
    assert en["answer"] == 71
    assert en["confidence"] == 0.6
    assert en["stratum"] == 2
    assert en["rule"] == "g7s2"


def test_corner_first_considered_tie(packages):
    """(d) equal confidence → first-in-manifest-order answer (strict >)."""
    idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e = packages
    en = _assert_parity([9], idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e)
    assert en["answer"] == 100  # g9a before g9b
    assert en["rule"] == "g9a"
    assert en["confidence"] == 0.55


def test_corner_dgate_and_relation(packages):
    """dgate (derived recent-member) and relation also agree bit-exact."""
    idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e = packages
    # dgate: member 10 appears, last token 11 → feature 10, key (10,11) → 42
    en_dg = _assert_parity(
        [10, 11], idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e)
    assert en_dg["answer"] == 42
    assert en_dg["circuit"] == "dgate"
    # relation eq(1,2) copy=1: last two equal → copy last token
    en_rel = _assert_parity(
        [3, 8, 8], idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e)
    assert en_rel["answer"] == 8
    assert en_rel["circuit"] == "relation"


def test_corner_small_token_contexts(packages):
    """Plain single-token contexts ([5], [7], [1]) for ngram/gate coverage."""
    idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e = packages
    for ctx in ([5], [7], [1], [9], [11]):
        _assert_parity(ctx, idioms_sw, ngrams_sw, m_sw, idioms_e, ngrams_e, m_e)


def test_m_gt1_raises():
    """M>1 / beam_width>1 is a future slice — hard NotImplementedError."""
    m = {
        "cover": "energy-beam", "W": 1, "M": 2, "beam_width": 1,
        "_tau": 0.35, "_derived": [], "_cmap": {},
    }
    with pytest.raises(NotImplementedError, match="Slice 2"):
        decide([1], [], {}, m)
    m2 = {
        "cover": "energy-beam", "W": 1, "M": 1, "beam_width": 2,
        "_tau": 0.35, "_derived": [], "_cmap": {},
    }
    with pytest.raises(NotImplementedError, match="Slice 2"):
        decide([1], [], {}, m2)
