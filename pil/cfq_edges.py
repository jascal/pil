"""CFQ SPARQL structure extraction: triple-level edges + multiset edge-F1.

**Scope (Step A foundation, decode-agnostic).** Given a CFQ SPARQL string, this
module extracts the WHERE body, parses triple lines into a MULTISET of
normalized edges ``(s_role, pred, o_role)``, and classifies each edge as
entity-anchored vs var-join. It also provides multiset ``edge_f1`` over two
edge bags.

**What is implemented:**
  - Body extraction between the first ``{`` and last ``}``; line-oriented triple
    parse (whitespace split into exactly 3 tokens).
  - Compound property paths (``/`` and ``|`` chained, e.g.
    ``ns:people.person.spouse_s/ns:people.marriage.spouse|...``) are ONE
    predicate token — never decomposed on ``/`` or ``|``.
  - ``rdf:type`` triples (``pred == "a"``) are consumed to build a
    variable→type map and are never emitted as edges.
  - Variables normalize to their recorded type object if present, else the
    literal ``"VAR"``; concrete tokens (``M\\d+``, ``ns:m.*``, ``ns:g.*``) and
    all other non-variables stay verbatim; predicates are never rewritten.
  - Output is a MULTISET: source order, duplicates, and structurally identical
    normalized edges are all preserved (never collapsed to a set).
  - Multiset F1 via ``collections.Counter`` intersection (min multiplicity).

**Out of scope (do not overclaim):** no compositional generation, no
exact-SPARQL scoring, no model/decode logic. This is a structure-bag extractor
plus multiset F1 only — foundation infra for a later join-structure diagnostic
(Step B). Public ``is_concrete`` / ``role_typed`` below are additive Step-B
helpers for role-typed structure metrics (entity identity abstracted away).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

Edge = tuple[str, str, str]  # (s_role, pred, o_role)

_CONCRETE_M = re.compile(r"M\d+")


def _is_concrete(tok: str) -> bool:
    """True for M-mention tokens or grounded Freebase MIDs (ns:m.* / ns:g.*)."""
    return (
        _CONCRETE_M.fullmatch(tok) is not None
        or tok.startswith("ns:m.")
        or tok.startswith("ns:g.")
    )


# ---------------------------------------------------------------------------
# Step-B foundation: role-typed structure diagnostic helpers (additive)
# ---------------------------------------------------------------------------


def is_concrete(tok: str) -> bool:
    """Public alias for the M-mention / grounded-MID concreteness check.

    Same rule as ``_is_concrete``: True iff tok fullmatches ``M\\d+``, or
    ``tok.startswith("ns:m.")``, or ``tok.startswith("ns:g.")``.
    """
    return _is_concrete(tok)


def role_typed(edge: Edge) -> Edge:
    """Abstract concrete entity tokens in an edge's subject/object roles to ``ENT``.

    Leave type-strings (e.g. ``ns:people.person``), the literal ``VAR``, and the
    predicate untouched::

        role_typed((s, p, o)) == (
            'ENT' if is_concrete(s) else s,
            p,
            'ENT' if is_concrete(o) else o,
        )

    M-ids / grounded MIDs are query-local identifiers, not globally predictable
    across questions, so role-typing measures argument-slot KIND +
    directionality + multiplicity — not entity identity. This is a structure
    metric, not an entity-linking metric.
    """
    s, p, o = edge
    return (
        "ENT" if is_concrete(s) else s,
        p,
        "ENT" if is_concrete(o) else o,
    )


def _is_var(tok: str) -> bool:
    return tok.startswith("?")


def _normalize_role(tok: str, var_type: dict[str, str]) -> str:
    if _is_var(tok):
        return var_type.get(tok, "VAR")
    return tok


@dataclass(frozen=True)
class ParsedQuery:
    edges: list[Edge]  # multiset (order = source order)
    anchored: list[bool]  # parallel to edges; True = entity-anchored
    filter_count: int


def parse_sparql_edges(sparql: str) -> ParsedQuery:
    """Parse a CFQ SPARQL string into a multiset of normalized edges.

    See module docstring for normalization rules. Malformed triple lines (not
    exactly 3 whitespace-separated tokens after stripping) raise ``ValueError``.
    """
    first = sparql.find("{")
    last = sparql.rfind("}")
    if first < 0 or last < 0 or last <= first:
        raise ValueError(f"SPARQL missing WHERE braces: {sparql!r}")
    body = sparql[first + 1 : last]

    var_type: dict[str, str] = {}
    # (s_raw, pred, o_raw, anchored) collected top-to-bottom; normalize after
    # the full scan so type triples later in the query still apply.
    raw: list[tuple[str, str, str, bool]] = []
    filter_count = 0

    for line in body.split("\n"):
        line = line.strip()
        if line.endswith("."):
            line = line[:-1].strip()
        if not line:
            continue
        if line.startswith("FILTER"):
            filter_count += 1
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(
                f"expected exactly 3 tokens (subj pred obj), got {len(parts)}: {line!r}"
            )
        subj, pred, obj = parts
        if pred == "a":
            if _is_var(subj):
                var_type[subj] = obj
            # consumed — never an edge
            continue
        anchored = _is_concrete(subj) or _is_concrete(obj)
        raw.append((subj, pred, obj, anchored))

    edges: list[Edge] = []
    anchored_flags: list[bool] = []
    for subj, pred, obj, anc in raw:
        s_role = _normalize_role(subj, var_type)
        o_role = _normalize_role(obj, var_type)
        edges.append((s_role, pred, o_role))
        anchored_flags.append(anc)

    return ParsedQuery(edges=edges, anchored=anchored_flags, filter_count=filter_count)


def edge_f1(pred: list[Edge], gold: list[Edge]) -> float:
    """MULTISET F1 over two lists of Edge tuples.

    - Both empty -> 1.0.
    - Exactly one empty (other non-empty) -> 0.0.
    - Otherwise: build ``collections.Counter(pred)`` and ``Counter(gold)``,
      intersect via Counter ``&`` (multiset intersection: min count per key),
      let inter = sum of the intersection Counter's values. precision =
      inter / len(pred), recall = inter / len(gold). If inter == 0, return 0.0.
      Else return 2 * precision * recall / (precision + recall).
    """
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = sum((Counter(pred) & Counter(gold)).values())
    if inter == 0:
        return 0.0
    precision = inter / len(pred)
    recall = inter / len(gold)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Join-variable representation (label-free / isomorphism-invariant)
# ---------------------------------------------------------------------------
#
# Additive Step-B foundation for the join-structure headroom diagnostic.
#
# This is a label-free / isomorphism-invariant join-variable representation:
# each join var is described by (its type-or-VAR, sorted multiset of
# (predicate, slot) incidences) with no ``?x0``-style variable name in the
# signature. Scoring is multiset-level over those signatures (Counter F1),
# not occurrence-aligned / not full graph-isomorphism matching beyond the
# per-var-signature granularity. Oracle-predicate regime assumptions belong
# to the diagnostic script, not here.
#
# Design intent: a 3-STAR (one var with 3 incidences) and a 3-CYCLE (three
# vars each with 2 incidences) produce non-colliding signature multisets.
# Do NOT reduce this to pairwise (var,var) or (pred,pred) co-occurrence —
# those representations would collide star and cycle.


# Per-var join signature: (var_type_or_VAR, sorted multiset of (pred, slot))
JoinSignature = tuple[str, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class JoinQuery:
    """Label-free join structure of a CFQ SPARQL query.

    ``signatures`` is the multiset of per-join-var signatures (one entry per
    variable with >=2 non-type incidences), sorted for determinism.
    ``predicates`` is the multiset of all non-type edge predicates (source
    order, multiplicity preserved). ``n_join_vars`` / ``n_vars_total`` are
    derived counts (join vars meeting the threshold vs all distinct vars).
    """

    signatures: list[tuple]  # multiset of JoinSignature; sorted for determinism
    predicates: list[str]  # multiset of non-type edge preds; source order
    n_join_vars: int
    n_vars_total: int


def parse_sparql_joins(sparql: str) -> JoinQuery:
    """Parse a CFQ SPARQL string into a label-free join-variable representation.

    Body extraction and triple-line parsing match ``parse_sparql_edges``
    (first ``{`` / last ``}``, strip trailing ``.``, skip FILTER, exactly 3
    tokens per triple), but **variable identity is preserved**: raw ``?x0``
    tokens are never rewritten via ``_normalize_role``.

    For each non-type triple, every variable endpoint records an incidence
    ``(pred, "subj"|"obj")``. A variable is a join var iff it accumulates
    >=2 incidences (across triples, or twice within one triple). Each join
    var's signature is::

        (var_type.get(var, "VAR"), tuple(sorted(incidences)))

    which is label-free (no variable name) and isomorphism-invariant.
    """
    first = sparql.find("{")
    last = sparql.rfind("}")
    if first < 0 or last < 0 or last <= first:
        raise ValueError(f"SPARQL missing WHERE braces: {sparql!r}")
    body = sparql[first + 1 : last]

    var_type: dict[str, str] = {}
    # (subj, pred, obj) non-type triples, source order; normalize after full scan
    # so type triples later in the body still apply (same as parse_sparql_edges).
    raw_edges: list[tuple[str, str, str]] = []
    all_vars: set[str] = set()

    for line in body.split("\n"):
        line = line.strip()
        if line.endswith("."):
            line = line[:-1].strip()
        if not line:
            continue
        if line.startswith("FILTER"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(
                f"expected exactly 3 tokens (subj pred obj), got {len(parts)}: {line!r}"
            )
        subj, pred, obj = parts
        if _is_var(subj):
            all_vars.add(subj)
        if _is_var(obj):
            all_vars.add(obj)
        if pred == "a":
            if _is_var(subj):
                var_type[subj] = obj
            # type triple — not an edge incidence, but subject counts as a var
            continue
        raw_edges.append((subj, pred, obj))

    # Collect incidences and predicates (preserve raw var tokens)
    incidences: dict[str, list[tuple[str, str]]] = {}
    predicates: list[str] = []
    for subj, pred, obj in raw_edges:
        predicates.append(pred)
        if _is_var(subj):
            incidences.setdefault(subj, []).append((pred, "subj"))
        if _is_var(obj):
            incidences.setdefault(obj, []).append((pred, "obj"))

    # Join vars: >=2 incidences; signature is label-free
    signatures: list[tuple] = []
    for var, incs in incidences.items():
        if len(incs) < 2:
            continue
        sig: JoinSignature = (var_type.get(var, "VAR"), tuple(sorted(incs)))
        signatures.append(sig)
    signatures.sort()  # deterministic multiset order

    return JoinQuery(
        signatures=signatures,
        predicates=predicates,
        n_join_vars=len(signatures),
        n_vars_total=len(all_vars),
    )


def join_f1(pred_sigs: list, gold_sigs: list) -> float:
    """MULTISET F1 over two lists of join-signature tuples.

    Identical shape/logic to ``edge_f1``: both empty -> 1.0; exactly one empty
    -> 0.0; else Counter intersection, precision/recall, harmonic mean.
    Elements must be hashable (signature tuples of tuples, not lists).
    """
    if not pred_sigs and not gold_sigs:
        return 1.0
    if not pred_sigs or not gold_sigs:
        return 0.0
    inter = sum((Counter(pred_sigs) & Counter(gold_sigs)).values())
    if inter == 0:
        return 0.0
    precision = inter / len(pred_sigs)
    recall = inter / len(gold_sigs)
    return 2 * precision * recall / (precision + recall)
