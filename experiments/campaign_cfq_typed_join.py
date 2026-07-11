"""CFQ typed-schema join assembler headroom (measurement only).

Oracle-predicate regime: predictors are handed the gold predicate multiset
and must predict JOIN STRUCTURE via a deterministic TYPE-DRIVEN assembler
that mines ``(predicate, slot) -> required arg-type`` from CFQ train, then
merges variable slots sharing a required type into join variables.

This is the next candidate after #75 (campaign_cfq_join_headroom): that probe
showed CFQ join structure is neither majority-determined by the exact
predicate multiset (A=0.33 IID / 0.09-0.19 MCD) nor lexically predictable
(B~=0.02). This slice measures whether mined required types close the gap.

Pre-measurements (architect; do not re-derive): 64-68% of gold join vars have
surface label "VAR" (no ``a`` triple), so merging must use MINED required
types, not surface labels; surface-label within-query ambiguity is 0.002
(mcd1) / 0.166 (random).

Measurement only — no induction mechanism, no Souffle encoding. Ceiling is
1.0 by construction (access to gold.signatures trivially scores 1.0; not
implemented). A low score is a RECIPE PLATEAU for THIS predictor family;
achievability is OPEN — never claim "unbridgeable".

Known limitation (v1, advisor-vetted scope): maximal same-type merging
collapses a cycle of same-typed variables into a single over-merged star
(exact required-type equality grouping, no type-compatibility classes).

Tables are mined from CFQ train, NOT the real Freebase ontology;
schema-derived joins from this or any future such assembler must NOT count
toward frac_induced in the generality-metric accounting.

Run (from repo root, defaults to mcd1):
  .venv/bin/python experiments/campaign_cfq_typed_join.py
Env: CFQ_SPLITS=mcd1,mcd2,mcd3  CFQ_MAX_VAL=2000
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.campaign_cfq_join_headroom import (  # noqa: E402
    fit_majority_join_prior,
    predict_majority_join,
)
from experiments.campaign_cfq_residual import load_split  # noqa: E402
from pil.cfq_edges import (  # noqa: E402
    join_f1,
    parse_sparql_joins,
    parse_sparql_typed_slots,
)

SENTINEL_PREFIX = "__UNTYPED__"  # req_type for never-observed-typed slots


def log(m: str = "") -> None:
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Slot tables (fit-mined) + type-driven assembler
# ---------------------------------------------------------------------------


@dataclass
class SlotTables:
    """Fit-mined (pred, slot) -> var-rate / req-type / majority-label tables."""

    var_rate: dict[tuple[str, str], float]
    # (pred, slot) -> frac of occurrences (over ALL fit occurrences of that
    # (pred,slot), var or not) holding a variable
    req_type: dict[tuple[str, str], str]
    # majority explicit type over TYPED var occurrences of (pred,slot) only
    # (occurrences where is_var and var_type is not None); if a (pred,slot)
    # never has a typed-var occurrence at fit ->
    # f"{SENTINEL_PREFIX}:{pred}:{slot}"
    label_maj: dict[tuple[str, str], str]
    # majority gold surface label over VAR occurrences of (pred,slot)
    # (label = var_type if that occurrence is typed else "VAR"); ties broken
    # lex-smallest label
    n_parse_fail: int


def mine_slot_tables(fit_pairs: list[tuple[str, str]]) -> SlotTables:
    """Mine (pred, slot) -> var-rate / majority-required-type / majority-label.

    From fit (question, sparql) pairs via ``parse_sparql_typed_slots``. Skip
    queries that raise ValueError, counting them in n_parse_fail (mirror
    fit_majority_join_prior's try/except ValueError style in #75).
    """
    total: Counter[tuple[str, str]] = Counter()
    var_count: Counter[tuple[str, str]] = Counter()
    typed_type_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    label_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    n_fail = 0

    for _q, sparql in fit_pairs:
        try:
            incs = parse_sparql_typed_slots(sparql)
        except ValueError:
            n_fail += 1
            continue
        for ti in incs:
            key = (ti.pred, ti.slot)
            total[key] += 1
            if ti.is_var:
                var_count[key] += 1
                label = ti.var_type if ti.var_type is not None else "VAR"
                label_counts[key][label] += 1
                if ti.var_type is not None:
                    typed_type_counts[key][ti.var_type] += 1

    var_rate: dict[tuple[str, str], float] = {
        k: var_count[k] / total[k] for k in total
    }

    req_type: dict[tuple[str, str], str] = {}
    for k in total:
        ctr = typed_type_counts.get(k)
        if ctr:
            max_c = max(ctr.values())
            candidates = sorted(t for t, c in ctr.items() if c == max_c)
            req_type[k] = candidates[0]
        else:
            pred, slot = k
            req_type[k] = f"{SENTINEL_PREFIX}:{pred}:{slot}"

    label_maj: dict[tuple[str, str], str] = {}
    for k, ctr in label_counts.items():
        max_c = max(ctr.values())
        candidates = sorted(lab for lab, c in ctr.items() if c == max_c)
        label_maj[k] = candidates[0]

    return SlotTables(
        var_rate=var_rate,
        req_type=req_type,
        label_maj=label_maj,
        n_parse_fail=n_fail,
    )


def _req_type_of(
    pred: str, slot: str, tables: SlotTables,
) -> str:
    """Lookup required type; unseen (pred,slot) -> synthetic sentinel."""
    key = (pred, slot)
    if key in tables.req_type:
        return tables.req_type[key]
    return f"{SENTINEL_PREFIX}:{pred}:{slot}"


def _is_sentinel(req_type: str) -> bool:
    return req_type.startswith(SENTINEL_PREFIX)


def _occupancy_from_predicates(
    predicates: list[str],
    tables: SlotTables,
) -> tuple[list[tuple[str, str]], int]:
    """Step-1 occupancy from oracle predicate multiset + var_rate tables.

    Returns (var_slot_occurrences, n_unseen_pred_slot). Multiplicity preserved.
    """
    occurrences: list[tuple[str, str]] = []
    n_unseen = 0
    for p in predicates:
        for slot in ("subj", "obj"):
            key = (p, slot)
            if key not in tables.var_rate:
                n_unseen += 1
                continue
            if tables.var_rate[key] >= 0.5:
                occurrences.append(key)
    return occurrences, n_unseen


def _assemble_from_var_slots(
    occurrences: list[tuple[str, str]],
    tables: SlotTables,
) -> tuple[list[tuple], int]:
    """Shared merge+label core for both assemblers (steps 2-4).

    Grouping is by EXACT STRING EQUALITY of the req_type value only — no
    type-compatibility/subtyping classes in v1 (advisor-vetted, v1 scope).

    KNOWN LIMITATION: maximal same-type merging will collapse a cycle of
    same-typed variables (e.g. three distinct people-typed vars each
    appearing twice) into a single over-merged star-shaped join var — a
    known failure mode of exact required-type equality grouping.

    Returns (sorted signatures, n_sentinel_incidences).
    """
    # Group non-sentinel occurrences by exact req_type string equality.
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    n_sentinel = 0
    for pred, slot in occurrences:
        rt = _req_type_of(pred, slot, tables)
        if _is_sentinel(rt):
            # Sentinel occurrences never merge with anything and never form
            # multi-member groups — each is a discarded singleton.
            n_sentinel += 1
            continue
        groups[rt].append((pred, slot))

    signatures: list[tuple] = []
    for _rt, members in groups.items():
        if len(members) < 2:
            continue
        # predicted_label = majority vote of label_maj over member occurrences
        # (vote by occurrence / multiplicity counts); ties -> lex-smallest
        label_ctr: Counter[str] = Counter()
        for pred, slot in members:
            lab = tables.label_maj.get((pred, slot), "VAR")
            label_ctr[lab] += 1
        max_c = max(label_ctr.values())
        pred_label = sorted(lab for lab, c in label_ctr.items() if c == max_c)[0]
        sig = (pred_label, tuple(sorted(members)))
        signatures.append(sig)

    signatures.sort()
    return signatures, n_sentinel


def assemble_joins_by_type(
    predicates: list[str], tables: SlotTables,
) -> list[tuple]:
    """Deterministic type-driven join assembly from an oracle predicate multiset.

    Reads ONLY the predicate multiset + fit-mined tables. NEVER receives gold
    signatures (leak guard) — this must be structurally true, not just documented.

    Multiplicity of ``predicates`` is preserved (a multiset — do not dedupe).
    Grouping by exact required-type string equality only (v1, no subtyping).
    """
    occurrences, _n_unseen = _occupancy_from_predicates(predicates, tables)
    sigs, _n_sentinel = _assemble_from_var_slots(occurrences, tables)
    return sigs


def assemble_from_incidences(
    incidences: list[tuple[str, str]], tables: SlotTables,
) -> list[tuple]:
    """DIAGNOSTIC-ONLY family ceiling: same merge+label rule, gold occupancy.

    Slot occupancy comes from gold var-occupied (pred, slot) incidences
    (var identity dropped). Does NOT consult var_rate. Never receives gold
    signatures or variable names.
    """
    sigs, _n_sentinel = _assemble_from_var_slots(list(incidences), tables)
    return sigs


def merge_ambiguity(gold_signatures: list[tuple], tables: SlotTables) -> bool:
    """True iff >=2 gold join-var signatures have intersecting non-sentinel
    req_type sets — i.e. maximal type-merge would fuse them (upper-bounds a
    confound C, informal name only, not a return field).
    """
    type_sets: list[set[str]] = []
    for sig in gold_signatures:
        incs = sig[1] if len(sig) > 1 else ()
        types: set[str] = set()
        for pred, slot in incs:
            rt = _req_type_of(pred, slot, tables)
            if _is_sentinel(rt):
                continue
            types.add(rt)
        type_sets.append(types)

    for i in range(len(type_sets)):
        for j in range(i + 1, len(type_sets)):
            if type_sets[i] & type_sets[j]:
                return True
    return False


# ---------------------------------------------------------------------------
# Per-split run
# ---------------------------------------------------------------------------


def run_split(tag: str) -> dict:
    blob = load_split(tag)
    tr_q, tr_a = blob["train_q"], blob["train_sparql"]
    te_q, te_a = blob["test_q"], blob["test_sparql"]

    # fit / val from train — VERBATIM from campaign_cfq_join_headroom
    idx = list(range(len(tr_q)))
    random.Random(0).shuffle(idx)
    nval = min(int(os.environ.get("CFQ_MAX_VAL", "2000")), max(1, len(idx) // 10))
    val_i, fit_i = idx[:nval], idx[nval:]
    fit_q = [tr_q[i] for i in fit_i]
    fit_a = [tr_a[i] for i in fit_i]
    # val unused further (no admit/CELF); seed/split kept for fit/te parity
    _val_pairs = [(tr_q[i], tr_a[i]) for i in val_i]
    te_pairs = list(zip(te_q, te_a, strict=True))
    if len(te_pairs) > 3000:
        te_pairs = [
            te_pairs[i]
            for i in range(0, len(te_pairs), max(1, len(te_pairs) // 3000))
        ][:3000]

    fit_pairs = list(zip(fit_q, fit_a, strict=True))

    # --- fit predictors ---
    maj_join, n_parse_fail_fit_maj = fit_majority_join_prior(fit_pairs)
    tables = mine_slot_tables(fit_pairs)
    n_parse_fail_fit = max(n_parse_fail_fit_maj, tables.n_parse_fail)
    # Stable refs for leak-guard identity checks in the score loop (must stay
    # the same fit-time objects; a future edit that rebuilds them from gold
    # per query would rebind the locals and trip the asserts below).
    _fit_maj_join = maj_join
    _fit_tables = tables

    # --- score test (oracle-predicate regime) ---
    scores_a: list[float] = []
    scores_a_seen: list[float] = []
    scores_c: list[float] = []
    scores_c_seen: list[float] = []
    scores_c_unseen_joinful: list[float] = []
    scores_a_unseen_joinful: list[float] = []
    scores_c_inc: list[float] = []
    n_merge_ambiguous = 0
    n_unseen_key = 0
    n_with_join = 0
    n_parse_fail_test = 0
    n_scored = 0
    n_sentinel_incidences = 0
    n_var_slot_incidences = 0
    n_unseen_pred_slot = 0

    for _q, sparql in te_pairs:
        try:
            gold = parse_sparql_joins(sparql)
        except ValueError:
            n_parse_fail_test += 1
            continue

        n_scored += 1
        if gold.n_join_vars >= 1:
            n_with_join += 1

        # -----------------------------------------------------------------
        # LEAK GUARD INVARIANT (mirrors #75):
        # Predictors below read ONLY the predicate-multiset key (A) or the
        # predicate multiset + fit-mined SlotTables (C), never gold.signatures
        # as an assembler input. gold.signatures only ever flows into
        # join_f1(...) calls and merge_ambiguity(...) calls — never into
        # assemble_joins_by_type or assemble_from_incidences as an argument.
        # assemble_from_incidences reads gold SLOT OCCUPANCY (which (pred,slot)
        # pairs are var-occupied) but never gold VARIABLE IDENTITY (no var
        # names, no which-vars-are-shared info). Its output (pred_c_inc /
        # c_inc_f1) is quarantined from every verdict branch except the
        # "family fails" check (verdict step 3) — i.e. C_inc_mean must not
        # be compared in verdict branches 1 and 2.
        # -----------------------------------------------------------------

        key = tuple(sorted(gold.predicates))
        seen = key in maj_join
        if not seen:
            n_unseen_key += 1

        # (A) majority-join prior — imported from #75; empty on unseen key
        pred_a = predict_majority_join(key, maj_join)
        a_f1 = join_f1(pred_a, gold.signatures)
        scores_a.append(a_f1)
        if seen:
            scores_a_seen.append(a_f1)

        # (C) type-driven assembler — oracle predicates + fit tables only
        occs, n_uns = _occupancy_from_predicates(gold.predicates, tables)
        pred_c, n_sent = _assemble_from_var_slots(occs, tables)
        n_unseen_pred_slot += n_uns
        n_sentinel_incidences += n_sent
        n_var_slot_incidences += len(occs)
        c_f1 = join_f1(pred_c, gold.signatures)
        scores_c.append(c_f1)
        if seen:
            scores_c_seen.append(c_f1)

        # join_f1 scores both-empty as 1.0, so predictor A is NOT 0 on
        # unseen-key queries overall (many unseen-key queries have
        # gold.n_join_vars == 0, i.e. gold.signatures == [], and A also
        # predicts [] on unseen keys, scoring 1.0 there) — so the
        # unseen-key A-vs-C comparison is only meaningful restricted to the
        # JOINFUL subset (gold.n_join_vars >= 1), which is exactly why
        # C_unseen_joinful / A_unseen_joinful exist as separate tracked
        # quantities from the plain unseen-key mean.
        if (not seen) and gold.n_join_vars >= 1:
            scores_c_unseen_joinful.append(c_f1)
            scores_a_unseen_joinful.append(a_f1)

        # (C_inc) DIAGNOSTIC-ONLY family ceiling — gold slot occupancy,
        # var identity dropped. Quarantined from verdict branches 1-2.
        try:
            typed_slots = parse_sparql_typed_slots(sparql)
        except ValueError:
            # Should raise identically to parse_sparql_joins; count separately
            # but still record C_inc as missing this query (skip).
            n_parse_fail_test += 1
            typed_slots = None

        if typed_slots is not None:
            inc_list = [(ti.pred, ti.slot) for ti in typed_slots if ti.is_var]
            pred_c_inc = assemble_from_incidences(inc_list, tables)
            c_inc_f1 = join_f1(pred_c_inc, gold.signatures)
            scores_c_inc.append(c_inc_f1)

        if merge_ambiguity(gold.signatures, tables):
            n_merge_ambiguous += 1

        assert maj_join is _fit_maj_join
        assert tables is _fit_tables

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    a_mean = _mean(scores_a)
    a_seen_mean = _mean(scores_a_seen)
    c_mean = _mean(scores_c)
    c_seen_mean = _mean(scores_c_seen)
    c_unseen_joinful = _mean(scores_c_unseen_joinful)
    a_unseen_joinful = _mean(scores_a_unseen_joinful)
    c_inc_mean = _mean(scores_c_inc)
    frac_merge_ambiguous = n_merge_ambiguous / n_scored if n_scored else 0.0
    unseen_key_rate = n_unseen_key / n_scored if n_scored else 0.0
    frac_queries_with_join = n_with_join / n_scored if n_scored else 0.0

    # Verdict: exact thresholds, checked in this exact order, first match wins.
    if c_mean >= 0.9:
        verdict = "TYPE-DETERMINED: joins are schema-typed assembly"
    elif (
        c_mean > a_mean + 0.05
        and c_unseen_joinful >= 0.2
        and c_seen_mean >= a_seen_mean - 0.02
    ):
        verdict = "REAL TYPE SIGNAL: build-out warranted"
    elif c_inc_mean < 0.5:
        verdict = (
            "TYPE-MERGE FAMILY FAILS even with oracle slot occupancy "
            "(decisive negative for this family; achievability OPEN — "
            "NOT unbridgeable)"
        )
    else:
        verdict = (
            "RECIPE PLATEAU: mined-type assembly insufficient at this "
            "granularity; achievability OPEN"
        )

    log(f"\n=== CFQ/{tag} typed-join headroom (oracle-predicate regime) ===")
    log(
        f"  n_test={n_scored}  n_fit={len(fit_pairs)}  "
        f"n_parse_fail_fit={n_parse_fail_fit}  n_parse_fail_test={n_parse_fail_test}"
    )
    log(f"  A majority-join-prior mean join_f1: {a_mean:.4f}")
    log(f"  A_seen mean join_f1:                {a_seen_mean:.4f}")
    log(f"  C typed-assemble mean join_f1:      {c_mean:.4f}")
    log(f"  C_seen mean join_f1:                {c_seen_mean:.4f}")
    log(f"  C_unseen_joinful mean join_f1:      {c_unseen_joinful:.4f}")
    log(f"  A_unseen_joinful mean join_f1:      {a_unseen_joinful:.4f}")
    log(
        f"  [DIAGNOSTIC-ONLY / not part of verdict logic 1-2] "
        f"C_inc mean join_f1: {c_inc_mean:.4f}"
    )
    log(f"  frac_merge_ambiguous: {frac_merge_ambiguous:.4f}")
    log(f"  unseen_key_rate: {unseen_key_rate:.4f}")
    log(f"  frac_queries_with_join: {frac_queries_with_join:.4f}")
    log(
        f"  n_sentinel_incidences={n_sentinel_incidences}  "
        f"n_var_slot_incidences={n_var_slot_incidences}  "
        f"n_unseen_pred_slot={n_unseen_pred_slot}"
    )
    log(f"  VERDICT: {verdict}")
    log(f"  maj_join size={len(maj_join)}  req_type size={len(tables.req_type)}")

    return {
        "split": tag,
        "n_test": n_scored,
        "n_fit": len(fit_pairs),
        "n_parse_fail_fit": n_parse_fail_fit,
        "n_parse_fail_test": n_parse_fail_test,
        "A": a_mean,
        "A_seen": a_seen_mean,
        "C": c_mean,
        "C_seen": c_seen_mean,
        "C_unseen_joinful": c_unseen_joinful,
        "A_unseen_joinful": a_unseen_joinful,
        "C_inc_mean": c_inc_mean,
        "frac_merge_ambiguous": frac_merge_ambiguous,
        "unseen_key_rate": unseen_key_rate,
        "frac_queries_with_join": frac_queries_with_join,
        "n_sentinel_incidences": n_sentinel_incidences,
        "n_var_slot_incidences": n_var_slot_incidences,
        "n_unseen_pred_slot": n_unseen_pred_slot,
        "verdict": verdict,
        "maj_join_size": len(maj_join),
        "req_type_size": len(tables.req_type),
    }


def main() -> None:
    want = [
        s.strip()
        for s in os.environ.get("CFQ_SPLITS", "mcd1").split(",")
        if s.strip()
    ]
    results: list[dict] = []
    for tag in want:
        pt = REPO / "data" / f"cfq_{tag}.pt"
        if not pt.exists():
            log(f"skip {tag}: missing data/cfq_{tag}.pt")
            continue
        results.append(run_split(tag))

    log("\n" + "=" * 100)
    log("CFQ TYPED-JOIN HEADROOM SCOREBOARD (oracle-predicate; multiset join_f1)")
    log("=" * 100)
    log(
        f"{'split':8} {'n_te':>5} {'A':>7} {'A_seen':>7} {'C':>7} {'C_seen':>7} "
        f"{'C_unsJF':>7} {'A_unsJF':>7} {'ambig':>7} {'unseen':>7}"
    )
    for r in results:
        log(
            f"{r['split']:8} {r['n_test']:5d} "
            f"{r['A']:7.3f} "
            f"{r['A_seen']:7.3f} "
            f"{r['C']:7.3f} "
            f"{r['C_seen']:7.3f} "
            f"{r['C_unseen_joinful']:7.3f} "
            f"{r['A_unseen_joinful']:7.3f} "
            f"{r['frac_merge_ambiguous']:7.3f} "
            f"{r['unseen_key_rate']:7.3f}"
        )
        log(
            f"  [diagnostic] C_inc={r['C_inc_mean']:.3f}  "
            f"(DIAGNOSTIC-ONLY / not part of verdict logic 1-2)  "
            f"verdict={r['verdict']}"
        )

    out_dir = REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "tables are mined from CFQ train, NOT the real Freebase ontology; "
            "schema-derived joins from this or any future such assembler must "
            "NOT count toward frac_induced in the generality-metric accounting. "
            "Oracle-predicate regime; measurement only. Ceiling 1.0 by "
            "construction. Low score is a recipe plateau for this predictor "
            "family; achievability OPEN — not unbridgeable."
        ),
        "splits": results,
    }
    out = out_dir / "cfq_typed_join.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
