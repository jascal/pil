"""Golden fixtures and multiset edge_f1 tests for ``pil.cfq_edges``."""
# ruff: noqa: E501
from __future__ import annotations

import math

from pil.cfq_edges import edge_f1, is_concrete, parse_sparql_edges, role_typed

# ---------------------------------------------------------------------------
# Golden SPARQL fixtures (verbatim)
# ---------------------------------------------------------------------------

FIXTURE_0 = """SELECT count(*) WHERE {
?x0 a ns:people.person .
?x0 ns:influence.influence_node.influenced M1 .
?x0 ns:influence.influence_node.influenced M2 .
?x0 ns:people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses ?x1 .
?x1 a ns:film.cinematographer .
FILTER ( ?x0 != ?x1 )
}"""

FIXTURE_1 = """SELECT count(*) WHERE {
?x0 ns:influence.influence_node.influenced_by M0 .
?x0 ns:people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses ?x1 .
?x1 a ns:people.person .
FILTER ( ?x0 != ?x1 )
}"""

FIXTURE_2 = """SELECT count(*) WHERE {
?x0 ns:people.person.sibling_s/ns:people.sibling_relationship.sibling|ns:fictional_universe.fictional_character.siblings/ns:fictional_universe.sibling_relationship_of_fictional_characters.siblings ?x1 .
?x1 a ns:people.person .
?x1 ns:people.person.gender ns:m.05zppz .
FILTER ( ?x0 != ?x1 ) .
M2 ns:film.film.directed_by M3 .
M2 ns:film.film.directed_by M4 .
M2 ns:film.film.written_by ?x0
}"""


# ---------------------------------------------------------------------------
# parse_sparql_edges — golden fixtures
# ---------------------------------------------------------------------------


def test_fixture_0():
    pq = parse_sparql_edges(FIXTURE_0)
    assert len(pq.edges) == 3
    assert pq.filter_count == 1

    e_m1 = ("ns:people.person", "ns:influence.influence_node.influenced", "M1")
    e_m2 = ("ns:people.person", "ns:influence.influence_node.influenced", "M2")
    assert e_m1 in pq.edges
    assert e_m2 in pq.edges
    assert pq.anchored[pq.edges.index(e_m1)] is True
    assert pq.anchored[pq.edges.index(e_m2)] is True

    spouse_edges = [
        (e, a)
        for e, a in zip(pq.edges, pq.anchored, strict=True)
        if e[0] == "ns:people.person" and e[2] == "ns:film.cinematographer"
    ]
    assert len(spouse_edges) == 1
    (s_role, pred, o_role), anc = spouse_edges[0]
    assert s_role == "ns:people.person"
    assert o_role == "ns:film.cinematographer"
    assert "/" in pred and "|" in pred
    assert pred.startswith("ns:people.person.spouse_s")
    assert anc is False  # var-join: both endpoints were variables

    assert all(e[1] != "a" for e in pq.edges)


def test_fixture_1():
    pq = parse_sparql_edges(FIXTURE_1)
    assert len(pq.edges) == 2
    assert pq.filter_count == 1

    e_inf = ("VAR", "ns:influence.influence_node.influenced_by", "M0")
    assert e_inf in pq.edges
    assert pq.anchored[pq.edges.index(e_inf)] is True

    spouse = [
        (e, a)
        for e, a in zip(pq.edges, pq.anchored, strict=True)
        if e[0] == "VAR" and e[2] == "ns:people.person"
    ]
    assert len(spouse) == 1
    (s_role, pred, o_role), anc = spouse[0]
    assert s_role == "VAR"
    assert o_role == "ns:people.person"
    assert "/" in pred and "|" in pred
    assert pred.startswith("ns:people.person.spouse_s")
    assert anc is False


def test_fixture_2():
    pq = parse_sparql_edges(FIXTURE_2)
    assert len(pq.edges) == 5
    assert pq.filter_count == 1

    e_m3 = ("M2", "ns:film.film.directed_by", "M3")
    e_m4 = ("M2", "ns:film.film.directed_by", "M4")
    assert pq.edges.count(e_m3) == 1
    assert pq.edges.count(e_m4) == 1
    assert pq.anchored[pq.edges.index(e_m3)] is True
    assert pq.anchored[pq.edges.index(e_m4)] is True

    e_gender = ("ns:people.person", "ns:people.person.gender", "ns:m.05zppz")
    assert e_gender in pq.edges
    assert pq.anchored[pq.edges.index(e_gender)] is True

    e_written = ("M2", "ns:film.film.written_by", "VAR")
    assert e_written in pq.edges
    assert pq.anchored[pq.edges.index(e_written)] is True

    sibling = [
        (e, a)
        for e, a in zip(pq.edges, pq.anchored, strict=True)
        if e[0] == "VAR" and e[2] == "ns:people.person"
    ]
    assert len(sibling) == 1
    (s_role, pred, o_role), anc = sibling[0]
    assert s_role == "VAR"
    assert o_role == "ns:people.person"
    assert "/" in pred and "|" in pred
    assert pred.startswith("ns:people.person.sibling_s")
    assert anc is False

    # 4 entity-anchored, 1 var-join
    assert sum(pq.anchored) == 4
    assert len(pq.anchored) - sum(pq.anchored) == 1


# ---------------------------------------------------------------------------
# edge_f1
# ---------------------------------------------------------------------------


def test_edge_f1_identical():
    x = [("A", "p", "B"), ("C", "q", "D")]
    assert edge_f1(x, x) == 1.0


def test_edge_f1_disjoint():
    pred = [("A", "p", "B")]
    gold = [("C", "q", "D")]
    assert edge_f1(pred, gold) == 0.0


def test_edge_f1_both_empty():
    assert edge_f1([], []) == 1.0


def test_edge_f1_one_empty():
    assert edge_f1([], [("A", "p", "B")]) == 0.0
    assert edge_f1([("A", "p", "B")], []) == 0.0


def test_edge_f1_multiset():
    """Multiset intersection: pred=[e,e], gold=[e] → inter=1 → F1 = 2/3."""
    e = ("A", "p", "B")
    pred = [e, e]
    gold = [e]
    # inter=min(2,1)=1; P=1/2; R=1/1; F1=2*0.5*1.0/(0.5+1.0)=2/3
    assert math.isclose(edge_f1(pred, gold), 2.0 / 3.0, abs_tol=1e-6)


def test_fixture_2_anchored_counts():
    """Anchored vs var-join split on FIXTURE_2 (also covered in test_fixture_2)."""
    pq = parse_sparql_edges(FIXTURE_2)
    assert sum(pq.anchored) == 4
    assert len(pq.anchored) - sum(pq.anchored) == 1


# ---------------------------------------------------------------------------
# is_concrete / role_typed (Step-B foundation)
# ---------------------------------------------------------------------------


def test_is_concrete():
    assert is_concrete("M2") is True
    assert is_concrete("M17") is True
    assert is_concrete("ns:m.05zppz") is True
    assert is_concrete("ns:g.something") is True
    assert is_concrete("VAR") is False
    assert is_concrete("ns:people.person") is False
    assert is_concrete("ns:film.cinematographer") is False
    assert is_concrete("x0") is False  # bare variable-looking token, not concrete


def test_role_typed_fixture_2():
    """role_typed on real FIXTURE_2 edges: ENT abstraction, type-strings/VAR stay."""
    pq = parse_sparql_edges(FIXTURE_2)
    edges = pq.edges

    # directed_by: both endpoints concrete M-ids → ENT / ENT
    e_directed = ("M2", "ns:film.film.directed_by", "M3")
    assert e_directed in edges
    assert role_typed(e_directed) == (
        "ENT",
        "ns:film.film.directed_by",
        "ENT",
    )

    # gender: object is grounded MID → ENT; subject type-string stays
    e_gender = ("ns:people.person", "ns:people.person.gender", "ns:m.05zppz")
    assert e_gender in edges
    assert role_typed(e_gender) == (
        "ns:people.person",
        "ns:people.person.gender",
        "ENT",
    )

    # var-join sibling: VAR + type-string — neither concrete, unchanged
    sibling = [e for e in edges if e[0] == "VAR" and e[2] == "ns:people.person"]
    assert len(sibling) == 1
    assert role_typed(sibling[0]) == sibling[0]
