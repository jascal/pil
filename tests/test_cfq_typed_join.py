"""Unit tests for CFQ typed-schema join assembler probe."""
# ruff: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

from pil.cfq_edges import TypedIncidence, parse_sparql_typed_slots

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # experiments/ has no __init__.py

from experiments.campaign_cfq_typed_join import (  # noqa: E402
    SENTINEL_PREFIX,
    SlotTables,
    assemble_from_incidences,
    assemble_joins_by_type,
    merge_ambiguity,
    mine_slot_tables,
)

# ---------------------------------------------------------------------------
# (a) parse_sparql_typed_slots
# ---------------------------------------------------------------------------


MIXED_ENTITY_VAR = """SELECT count(*) WHERE {
?x0 ns:p1 M1 .
M2 ns:p2 ?x1 .
FILTER (?x0 != ?x1)
?x0 a ns:person .
}"""


def test_parse_typed_slots_mixed_entity_var_filter_and_late_a_triple():
    """Entity/var mix; a-triple AFTER edge still types; FILTER skipped; 2/triple."""
    incs = parse_sparql_typed_slots(MIXED_ENTITY_VAR)
    # two non-type triples → exactly 4 incidences (subj then obj each)
    assert len(incs) == 4

    # triple 1: ?x0 ns:p1 M1  — subj var (typed ns:person via late a), obj entity
    assert incs[0] == TypedIncidence(
        pred="ns:p1", slot="subj", is_var=True, var_type="ns:person",
    )
    assert incs[1] == TypedIncidence(
        pred="ns:p1", slot="obj", is_var=False, var_type=None,
    )
    # triple 2: M2 ns:p2 ?x1  — subj entity, obj var untyped
    assert incs[2] == TypedIncidence(
        pred="ns:p2", slot="subj", is_var=False, var_type=None,
    )
    assert incs[3] == TypedIncidence(
        pred="ns:p2", slot="obj", is_var=True, var_type=None,
    )


def test_parse_typed_slots_a_triple_before_edge():
    """a-triple before edge also resolves (full-scan path)."""
    sparql = """SELECT count(*) WHERE {
?x0 a ns:film .
?x0 ns:directed_by ?x1 .
}"""
    incs = parse_sparql_typed_slots(sparql)
    assert len(incs) == 2
    assert incs[0].var_type == "ns:film"
    assert incs[0].is_var is True
    assert incs[1].is_var is True
    assert incs[1].var_type is None


# ---------------------------------------------------------------------------
# (b) mine_slot_tables
# ---------------------------------------------------------------------------


def test_mine_slot_tables_var_rate_req_type_sentinel_label_maj():
    # Fit set:
    # Q1: ?x0 ns:p1 M1  with ?x0 a ns:person  → (p1,subj) var typed person; (p1,obj) not var
    # Q2: ?x0 ns:p1 M2  with ?x0 a ns:person  → same
    # Q3: M3 ns:p1 ?x0                       → (p1,subj) not var; (p1,obj) var untyped
    # Q4: ?x0 ns:p2 M1                       → (p2,subj) var untyped always; (p2,obj) not var
    #     (p2,subj never typed → sentinel)
    q1 = """SELECT count(*) WHERE {
?x0 a ns:person .
?x0 ns:p1 M1 .
}"""
    q2 = """SELECT count(*) WHERE {
?x0 a ns:person .
?x0 ns:p1 M2 .
}"""
    q3 = """SELECT count(*) WHERE {
M3 ns:p1 ?x0 .
}"""
    q4 = """SELECT count(*) WHERE {
?x0 ns:p2 M1 .
}"""
    # Extra typed majority for p1 subj: person twice, film once → person wins
    q5 = """SELECT count(*) WHERE {
?x0 a ns:film .
?x0 ns:p1 M4 .
}"""
    tables = mine_slot_tables([
        ("q1", q1), ("q2", q2), ("q3", q3), ("q4", q4), ("q5", q5),
    ])
    assert tables.n_parse_fail == 0

    # (p1, subj): appears in q1,q2,q3,q5 — 3 var (q1,q2,q5) + 1 non-var (q3) → 3/4
    assert abs(tables.var_rate[("ns:p1", "subj")] - 0.75) < 1e-9
    # (p1, obj): q1,q2,q5 non-var; q3 var → 1/4
    assert abs(tables.var_rate[("ns:p1", "obj")] - 0.25) < 1e-9
    # (p2, subj): always var in its only occurrence
    assert tables.var_rate[("ns:p2", "subj")] == 1.0

    # req_type majority over TYPED var occurrences of (p1,subj): person,person,film → person
    assert tables.req_type[("ns:p1", "subj")] == "ns:person"
    # (p2,subj) is a var slot but never typed → sentinel
    assert tables.req_type[("ns:p2", "subj")].startswith(SENTINEL_PREFIX)
    assert tables.req_type[("ns:p2", "subj")] == f"{SENTINEL_PREFIX}:ns:p2:subj"

    # label_maj for (p2,subj): only untyped var → "VAR"
    assert tables.label_maj[("ns:p2", "subj")] == "VAR"
    # label_maj for (p1,subj): person,person,film → person (majority of surface labels)
    assert tables.label_maj[("ns:p1", "subj")] == "ns:person"


# ---------------------------------------------------------------------------
# Helpers for synthetic SlotTables
# ---------------------------------------------------------------------------


def _tables(
    var_rate: dict[tuple[str, str], float],
    req_type: dict[tuple[str, str], str],
    label_maj: dict[tuple[str, str], str] | None = None,
) -> SlotTables:
    if label_maj is None:
        label_maj = {k: "VAR" for k in var_rate}
    return SlotTables(
        var_rate=var_rate,
        req_type=req_type,
        label_maj=label_maj,
        n_parse_fail=0,
    )


# ---------------------------------------------------------------------------
# (c) assembler core
# ---------------------------------------------------------------------------


def test_assemble_merge_same_req_type():
    """Two predicates sharing req_type → one 2-incidence join var."""
    tables = _tables(
        var_rate={
            ("ns:p1", "subj"): 1.0,
            ("ns:p1", "obj"): 0.0,
            ("ns:p2", "subj"): 1.0,
            ("ns:p2", "obj"): 0.0,
        },
        req_type={
            ("ns:p1", "subj"): "ns:person",
            ("ns:p1", "obj"): "ns:thing",
            ("ns:p2", "subj"): "ns:person",
            ("ns:p2", "obj"): "ns:thing",
        },
        label_maj={
            ("ns:p1", "subj"): "ns:person",
            ("ns:p2", "subj"): "VAR",
        },
    )
    pred = assemble_joins_by_type(["ns:p1", "ns:p2"], tables)
    # two person-typed subj slots → one join; labels person vs VAR (tie) →
    # lex-smallest = "VAR" (not ns:person)
    assert len(pred) == 1
    label, incs = pred[0]
    assert label == "VAR"
    assert incs == (("ns:p1", "subj"), ("ns:p2", "subj"))

    # clear majority: two person votes beat one VAR
    tables2 = _tables(
        var_rate={
            ("ns:p1", "subj"): 1.0, ("ns:p1", "obj"): 0.0,
            ("ns:p2", "subj"): 1.0, ("ns:p2", "obj"): 0.0,
            ("ns:p3", "subj"): 1.0, ("ns:p3", "obj"): 0.0,
        },
        req_type={
            ("ns:p1", "subj"): "ns:person",
            ("ns:p2", "subj"): "ns:person",
            ("ns:p3", "subj"): "ns:person",
            ("ns:p1", "obj"): "X",
            ("ns:p2", "obj"): "X",
            ("ns:p3", "obj"): "X",
        },
        label_maj={
            ("ns:p1", "subj"): "ns:person",
            ("ns:p2", "subj"): "ns:person",
            ("ns:p3", "subj"): "VAR",
        },
    )
    pred2 = assemble_joins_by_type(["ns:p1", "ns:p2", "ns:p3"], tables2)
    assert len(pred2) == 1
    assert pred2[0][0] == "ns:person"


def test_assemble_distinct_req_types_no_merge():
    """Distinct req_types → no multi-member group → empty output."""
    tables = _tables(
        var_rate={
            ("ns:p1", "subj"): 1.0,
            ("ns:p1", "obj"): 0.0,
            ("ns:p2", "subj"): 1.0,
            ("ns:p2", "obj"): 0.0,
        },
        req_type={
            ("ns:p1", "subj"): "ns:person",
            ("ns:p1", "obj"): "ns:thing",
            ("ns:p2", "subj"): "ns:film",
            ("ns:p2", "obj"): "ns:thing",
        },
    )
    pred = assemble_joins_by_type(["ns:p1", "ns:p2"], tables)
    assert pred == []


def test_assemble_var_rate_below_half_excluded():
    """Slot with var_rate < 0.5 is not a var slot in assemble_joins_by_type."""
    tables = _tables(
        var_rate={
            ("ns:p1", "subj"): 0.4,  # below threshold
            ("ns:p1", "obj"): 0.0,
            ("ns:p2", "subj"): 1.0,
            ("ns:p2", "obj"): 0.0,
        },
        req_type={
            ("ns:p1", "subj"): "ns:person",
            ("ns:p1", "obj"): "ns:thing",
            ("ns:p2", "subj"): "ns:person",
            ("ns:p2", "obj"): "ns:thing",
        },
    )
    # Only p2 subj is a var slot → singleton → no join var
    pred = assemble_joins_by_type(["ns:p1", "ns:p2"], tables)
    assert pred == []


def test_assemble_predicate_multiset_multiplicity():
    """Same predicate twice merges two (pred,slot) occurrences into one join var."""
    tables = _tables(
        var_rate={
            ("ns:p1", "subj"): 1.0,
            ("ns:p1", "obj"): 0.0,
        },
        req_type={
            ("ns:p1", "subj"): "ns:person",
            ("ns:p1", "obj"): "ns:thing",
        },
        label_maj={("ns:p1", "subj"): "VAR"},
    )
    pred = assemble_joins_by_type(["ns:p1", "ns:p1"], tables)
    assert len(pred) == 1
    label, incs = pred[0]
    assert label == "VAR"
    assert incs == (("ns:p1", "subj"), ("ns:p1", "subj"))


def test_assemble_sentinel_never_merges():
    """Sentinel-typed slots never merge — even two occurrences of the same (pred,slot)."""
    sent = f"{SENTINEL_PREFIX}:ns:p1:subj"
    tables = _tables(
        var_rate={
            ("ns:p1", "subj"): 1.0,
            ("ns:p1", "obj"): 0.0,
        },
        req_type={
            ("ns:p1", "subj"): sent,
            ("ns:p1", "obj"): f"{SENTINEL_PREFIX}:ns:p1:obj",
        },
    )
    # Two occurrences of the same sentinel string must NOT merge
    pred = assemble_joins_by_type(["ns:p1", "ns:p1"], tables)
    assert pred == []

    # Different (pred,slot) pairs have different sentinel strings and also no merge
    tables2 = _tables(
        var_rate={
            ("ns:p1", "subj"): 1.0,
            ("ns:p1", "obj"): 0.0,
            ("ns:p2", "subj"): 1.0,
            ("ns:p2", "obj"): 0.0,
        },
        req_type={
            ("ns:p1", "subj"): f"{SENTINEL_PREFIX}:ns:p1:subj",
            ("ns:p1", "obj"): f"{SENTINEL_PREFIX}:ns:p1:obj",
            ("ns:p2", "subj"): f"{SENTINEL_PREFIX}:ns:p2:subj",
            ("ns:p2", "obj"): f"{SENTINEL_PREFIX}:ns:p2:obj",
        },
    )
    assert assemble_joins_by_type(["ns:p1", "ns:p2"], tables2) == []


def test_assemble_deterministic_sorted():
    tables = _tables(
        var_rate={
            ("ns:b", "subj"): 1.0,
            ("ns:b", "obj"): 0.0,
            ("ns:a", "subj"): 1.0,
            ("ns:a", "obj"): 0.0,
            ("ns:c", "subj"): 1.0,
            ("ns:c", "obj"): 0.0,
            ("ns:d", "subj"): 1.0,
            ("ns:d", "obj"): 0.0,
        },
        req_type={
            ("ns:b", "subj"): "T1",
            ("ns:a", "subj"): "T1",
            ("ns:c", "subj"): "T2",
            ("ns:d", "subj"): "T2",
            ("ns:b", "obj"): "X",
            ("ns:a", "obj"): "X",
            ("ns:c", "obj"): "X",
            ("ns:d", "obj"): "X",
        },
    )
    preds = ["ns:b", "ns:a", "ns:c", "ns:d"]
    r1 = assemble_joins_by_type(preds, tables)
    r2 = assemble_joins_by_type(preds, tables)
    assert r1 == r2
    assert r1 == sorted(r1)
    # two groups: T1→(a,b), T2→(c,d); sorted by full signature tuple
    assert len(r1) == 2
    assert r1[0][1] == (("ns:a", "subj"), ("ns:b", "subj"))
    assert r1[1][1] == (("ns:c", "subj"), ("ns:d", "subj"))


# ---------------------------------------------------------------------------
# (d) star/cycle over-merge limitation (documented characterization, not a bug)
# ---------------------------------------------------------------------------


def test_overmerge_cycle_same_type_collapses_to_one():
    """KNOWN LIMITATION: 3 same-typed join vars collapse to one over-merged group.

    Gold cycle would have 3 join vars of 2 incidences each; type-merge fuses
    all person-typed incidences into a single star-like signature.
    """
    # Cycle-shaped multiset: knows x3 (each edge contributes person subj + person obj)
    tables = _tables(
        var_rate={
            ("ns:knows", "subj"): 1.0,
            ("ns:knows", "obj"): 1.0,
        },
        req_type={
            ("ns:knows", "subj"): "ns:person",
            ("ns:knows", "obj"): "ns:person",
        },
        label_maj={
            ("ns:knows", "subj"): "ns:person",
            ("ns:knows", "obj"): "ns:person",
        },
    )
    # 3 edges → 6 person-typed var-slot incidences → ONE over-merged join var
    pred = assemble_joins_by_type(
        ["ns:knows", "ns:knows", "ns:knows"], tables,
    )
    assert len(pred) == 1
    _label, incs = pred[0]
    assert len(incs) == 6  # all incidences fused, not 3 pairs of 2

    # Same via assemble_from_incidences with gold occupancy of a 3-cycle
    # (6 incidences, no var identity):
    gold_occ = [
        ("ns:knows", "subj"), ("ns:knows", "obj"),
        ("ns:knows", "subj"), ("ns:knows", "obj"),
        ("ns:knows", "subj"), ("ns:knows", "obj"),
    ]
    pred_inc = assemble_from_incidences(gold_occ, tables)
    assert len(pred_inc) == 1
    assert len(pred_inc[0][1]) == 6


# ---------------------------------------------------------------------------
# (e) assemble_from_incidences ignores var_rate
# ---------------------------------------------------------------------------


def test_assemble_from_incidences_ignores_var_rate():
    """Given occupancy is used even when var_rate would exclude the slot."""
    tables = _tables(
        var_rate={
            ("ns:p1", "subj"): 0.1,  # would exclude under assemble_joins_by_type
            ("ns:p1", "obj"): 0.0,
            ("ns:p2", "subj"): 0.1,
            ("ns:p2", "obj"): 0.0,
        },
        req_type={
            ("ns:p1", "subj"): "ns:person",
            ("ns:p1", "obj"): "ns:thing",
            ("ns:p2", "subj"): "ns:person",
            ("ns:p2", "obj"): "ns:thing",
        },
        label_maj={
            ("ns:p1", "subj"): "VAR",
            ("ns:p2", "subj"): "VAR",
        },
    )
    # assemble_joins_by_type excludes both → empty
    assert assemble_joins_by_type(["ns:p1", "ns:p2"], tables) == []
    # assemble_from_incidences still merges the given incidences
    pred = assemble_from_incidences(
        [("ns:p1", "subj"), ("ns:p2", "subj")], tables,
    )
    assert len(pred) == 1
    assert pred[0][1] == (("ns:p1", "subj"), ("ns:p2", "subj"))


# ---------------------------------------------------------------------------
# (f) merge_ambiguity
# ---------------------------------------------------------------------------


def test_merge_ambiguity_true_and_false_and_sentinel_exclusion():
    tables = _tables(
        var_rate={},
        req_type={
            ("ns:p1", "subj"): "ns:person",
            ("ns:p2", "subj"): "ns:person",
            ("ns:p3", "subj"): "ns:film",
            ("ns:p4", "subj"): "ns:film",
            ("ns:s1", "subj"): f"{SENTINEL_PREFIX}:ns:s1:subj",
            ("ns:s2", "subj"): f"{SENTINEL_PREFIX}:ns:s2:subj",
            # force same sentinel string only if same key; different keys differ
            ("ns:sa", "subj"): f"{SENTINEL_PREFIX}:shared",  # artificial same sentinel
            ("ns:sb", "subj"): f"{SENTINEL_PREFIX}:shared",
        },
    )
    # True: two gold join vars both touch ns:person
    gold_ambig = [
        ("VAR", (("ns:p1", "subj"), ("ns:p2", "subj"))),
        ("VAR", (("ns:p2", "subj"), ("ns:p3", "subj"))),
    ]
    assert merge_ambiguity(gold_ambig, tables) is True

    # False: disjoint non-sentinel types
    gold_ok = [
        ("VAR", (("ns:p1", "subj"), ("ns:p2", "subj"))),  # person
        ("VAR", (("ns:p3", "subj"), ("ns:p4", "subj"))),  # film
    ]
    assert merge_ambiguity(gold_ok, tables) is False

    # Sentinel-only "overlap" must NOT count as ambiguous
    gold_sent = [
        ("VAR", (("ns:sa", "subj"), ("ns:p1", "subj"))),  # sentinel + person
        ("VAR", (("ns:sb", "subj"), ("ns:p3", "subj"))),  # sentinel + film
    ]
    # sa/sb share sentinel string but sentinels are excluded; person ∩ film = empty
    assert merge_ambiguity(gold_sent, tables) is False


# ---------------------------------------------------------------------------
# (g) leak guard — type-merge answer, not gold
# ---------------------------------------------------------------------------


def test_assemble_no_gold_leak_disagrees_with_gold():
    """Rig tables so type-merge ≠ gold; assembler must return type-merge answer."""
    # Gold structure (not passed to assembler): two join vars —
    #   (VAR, ((p1,subj),(p2,subj))) and (VAR, ((p3,subj),(p4,subj)))
    # But tables force ALL four subj slots to the same req_type → one over-merge.
    tables = _tables(
        var_rate={
            ("ns:p1", "subj"): 1.0, ("ns:p1", "obj"): 0.0,
            ("ns:p2", "subj"): 1.0, ("ns:p2", "obj"): 0.0,
            ("ns:p3", "subj"): 1.0, ("ns:p3", "obj"): 0.0,
            ("ns:p4", "subj"): 1.0, ("ns:p4", "obj"): 0.0,
        },
        req_type={
            ("ns:p1", "subj"): "ns:person",
            ("ns:p2", "subj"): "ns:person",
            ("ns:p3", "subj"): "ns:person",
            ("ns:p4", "subj"): "ns:person",
            ("ns:p1", "obj"): "X",
            ("ns:p2", "obj"): "X",
            ("ns:p3", "obj"): "X",
            ("ns:p4", "obj"): "X",
        },
        label_maj={
            ("ns:p1", "subj"): "VAR",
            ("ns:p2", "subj"): "VAR",
            ("ns:p3", "subj"): "VAR",
            ("ns:p4", "subj"): "VAR",
        },
    )
    gold = [
        ("VAR", (("ns:p1", "subj"), ("ns:p2", "subj"))),
        ("VAR", (("ns:p3", "subj"), ("ns:p4", "subj"))),
    ]
    type_merge = assemble_joins_by_type(
        ["ns:p1", "ns:p2", "ns:p3", "ns:p4"], tables,
    )
    # Type-merge: ONE 4-incidence signature, not gold's two pairs
    assert len(type_merge) == 1
    assert len(type_merge[0][1]) == 4
    assert type_merge != gold
    # And it matches the pure type-merge prediction, not gold
    expected = [
        ("VAR", (
            ("ns:p1", "subj"),
            ("ns:p2", "subj"),
            ("ns:p3", "subj"),
            ("ns:p4", "subj"),
        )),
    ]
    assert type_merge == expected
