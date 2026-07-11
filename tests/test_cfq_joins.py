"""Golden fixtures and multiset join_f1 tests for CFQ join representation."""
# ruff: noqa: E501
from __future__ import annotations

import math
import sys
from pathlib import Path

from pil.cfq_edges import join_f1, parse_sparql_joins

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # experiments/ has no __init__.py

from experiments.campaign_cfq_join_headroom import (  # noqa: E402
    fit_majority_join_prior,
    predict_majority_join,
)

# ---------------------------------------------------------------------------
# Synthetic SPARQL fixtures (star vs cycle, var identity, single-incidence)
# ---------------------------------------------------------------------------

STAR_3 = """SELECT count(*) WHERE {
?x0 ns:p1 ?y1 .
?x0 ns:p2 ?y2 .
?x0 ns:p3 ?y3 .
}"""

CYCLE_3 = """SELECT count(*) WHERE {
?x0 ns:p1 ?x1 .
?x1 ns:p2 ?x2 .
?x2 ns:p3 ?x0 .
}"""

SHARED_VAR = """SELECT count(*) WHERE {
?x0 ns:p1 M1 .
?x0 ns:p2 M2 .
}"""

SAME_TYPE_UNSHARED = """SELECT count(*) WHERE {
?x0 a ns:sometype .
?x1 a ns:sometype .
?x0 ns:p1 M1 .
?x1 ns:p2 M2 .
}"""

SINGLE_INCIDENCE = """SELECT count(*) WHERE {
?x0 ns:p1 M1 .
?x1 ns:p2 M2 .
}"""


# ---------------------------------------------------------------------------
# 1. star vs cycle — core correctness of the representation
# ---------------------------------------------------------------------------


def test_star_vs_cycle():
    star = parse_sparql_joins(STAR_3)
    cycle = parse_sparql_joins(CYCLE_3)

    assert len(star.signatures) == 1
    assert len(cycle.signatures) == 3
    assert star.signatures != cycle.signatures

    # star: one join var with 3 incidences
    star_sig = star.signatures[0]
    # signature = (var_type_or_VAR, tuple of (pred, slot) incidences)
    assert len(star_sig[1]) == 3

    # cycle: three join vars, each with 2 incidences
    for sig in cycle.signatures:
        assert len(sig[1]) == 2

    assert star.n_join_vars == 1
    assert cycle.n_join_vars == 3


# ---------------------------------------------------------------------------
# 2. var identity preserved (no type-based merge of distinct vars)
# ---------------------------------------------------------------------------


def test_var_identity_shared():
    pq = parse_sparql_joins(SHARED_VAR)
    assert pq.n_join_vars == 1
    assert len(pq.signatures) == 1
    assert len(pq.signatures[0][1]) == 2  # two incidences on ?x0


def test_var_identity_same_type_unshared():
    """Same-type distinct vars must NOT merge into a join just because of type."""
    pq = parse_sparql_joins(SAME_TYPE_UNSHARED)
    assert pq.n_join_vars == 0
    assert pq.signatures == []


# ---------------------------------------------------------------------------
# 3. join_f1 multiset
# ---------------------------------------------------------------------------


def test_join_f1_identical():
    s = ("VAR", (("ns:p1", "subj"), ("ns:p2", "subj")))
    assert join_f1([s], [s]) == 1.0
    assert join_f1([s, s], [s, s]) == 1.0


def test_join_f1_disjoint():
    a = ("VAR", (("ns:p1", "subj"), ("ns:p2", "subj")))
    b = ("VAR", (("ns:p3", "obj"), ("ns:p4", "obj")))
    assert join_f1([a], [b]) == 0.0


def test_join_f1_multiset_double_vs_single():
    """pred=[s,s], gold=[s] → inter=1 → P=1/2, R=1 → F1=2/3."""
    s = ("VAR", (("ns:p1", "subj"), ("ns:p2", "subj")))
    assert math.isclose(join_f1([s, s], [s]), 2.0 / 3.0, abs_tol=1e-6)


def test_join_f1_both_empty():
    assert join_f1([], []) == 1.0


def test_join_f1_one_empty():
    s = ("VAR", (("ns:p1", "subj"), ("ns:p2", "subj")))
    assert join_f1([], [s]) == 0.0
    assert join_f1([s], []) == 0.0


# ---------------------------------------------------------------------------
# 4. leak guard — unseen key → empty prediction (no gold fallback)
# ---------------------------------------------------------------------------


def test_majority_join_unseen_key_no_gold_fallback():
    """Unseen predicate-multiset key must predict empty, never gold signatures."""
    # Fit on a star with ns:p1/p2/p3
    fit_pairs = [
        ("q1", STAR_3),
        ("q2", STAR_3),
    ]
    maj_join, n_fail = fit_majority_join_prior(fit_pairs)
    assert n_fail == 0
    assert len(maj_join) >= 1

    # Test query with a completely different predicate multiset
    unseen_sparql = """SELECT count(*) WHERE {
?x0 ns:zz1 ?y1 .
?x0 ns:zz2 ?y2 .
}"""
    gold = parse_sparql_joins(unseen_sparql)
    assert gold.n_join_vars == 1  # has a join, so empty pred is a real miss
    key = tuple(sorted(gold.predicates))
    assert key not in maj_join

    pred = predict_majority_join(key, maj_join)
    assert pred == []
    # and join_f1 against non-empty gold is 0.0
    assert join_f1(pred, gold.signatures) == 0.0


def test_majority_join_seen_key_returns_majority():
    """Seen key returns the fit majority signature multiset."""
    fit_pairs = [
        ("q1", STAR_3),
        ("q2", STAR_3),
    ]
    maj_join, _ = fit_majority_join_prior(fit_pairs)
    gold = parse_sparql_joins(STAR_3)
    key = tuple(sorted(gold.predicates))
    pred = predict_majority_join(key, maj_join)
    assert pred == list(maj_join[key])
    assert join_f1(pred, gold.signatures) == 1.0


# ---------------------------------------------------------------------------
# 5. single-incidence vars are not joins
# ---------------------------------------------------------------------------


def test_single_incidence_not_join():
    pq = parse_sparql_joins(SINGLE_INCIDENCE)
    assert pq.signatures == []
    assert pq.n_join_vars == 0


# ---------------------------------------------------------------------------
# Extra: predicates multiset + n_vars_total sanity
# ---------------------------------------------------------------------------


def test_predicates_and_n_vars():
    star = parse_sparql_joins(STAR_3)
    assert star.predicates == ["ns:p1", "ns:p2", "ns:p3"]
    # ?x0, ?y1, ?y2, ?y3
    assert star.n_vars_total == 4

    typed = parse_sparql_joins(SAME_TYPE_UNSHARED)
    # ?x0, ?x1 from type triples + non-type
    assert typed.n_vars_total == 2
    assert typed.predicates == ["ns:p1", "ns:p2"]
