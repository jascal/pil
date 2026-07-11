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
(Step B).
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
