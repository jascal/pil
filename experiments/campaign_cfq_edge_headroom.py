"""CFQ edge-metric Step B: role-typed structure headroom diagnostic.

Measurement only — no lift/ceiling promised. Answers, over the predicate-bag
baseline: how much argument-role structure exists in CFQ SPARQL edges, how much
a per-predicate majority-role prior captures, and where residual headroom sits
(anchored-entity edges vs var-join edges). Does not build a decoder.

Run (from repo root, defaults to mcd1):
  .venv/bin/python experiments/campaign_cfq_edge_headroom.py
Env: CFQ_SPLITS=mcd1  CFQ_MAX_VAL=2000
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.campaign_cfq_residual import (  # noqa: E402
    content_words,
    load_split,
    mean_set_f1,
    mine_base_atoms,
    predict_paths,
)
from pil.cfq_edges import (  # noqa: E402
    Edge,
    edge_f1,
    parse_sparql_edges,
    role_typed,
)
from pil.residual_template import MapDict  # noqa: E402


def log(m: str = "") -> None:
    print(m, flush=True)


def is_anchored_rt(e: Edge) -> bool:
    """True iff role-typed edge has ENT in subject or object role."""
    return "ENT" in (e[0], e[2])


def mine_edge_pred_atoms(
    questions: list[str],
    sparqls: list[str],
    *,
    min_support: int = 2,
) -> tuple[MapDict, int]:
    """Word → full edge-predicate path atoms (all edges, not max_ns=1).

    Mirrors mine_base_atoms shape (MapDict of (word, pred) → [pred]) so
    predict_paths can be reused unmodified. Uses the FULL compound predicate
    string from parse_sparql_edges (not sparql_ns fragments). Dedupes
    predicates within a single query. Returns (maps, n_parse_fail).
    """
    votes: Counter[tuple[str, str]] = Counter()
    n_fail = 0
    for q, a in zip(questions, sparqls, strict=True):
        try:
            pq = parse_sparql_edges(a)
        except ValueError:
            n_fail += 1
            continue
        # set of unique full edge-predicates in this query (all edges, no max_ns)
        preds = {e[1] for e in pq.edges}
        if not preds:
            continue
        words = content_words(q)
        for w in words:
            for p in preds:
                votes[(w, p)] += 1
    maps: MapDict = {}
    for (w, p), c in votes.items():
        if c >= min_support:
            maps[(w, p)] = [p]
    return maps, n_fail


def majority_roles(
    fit_q: list[str],
    fit_a: list[str],
) -> tuple[dict[str, tuple[str, str]], int]:
    """Per full edge-predicate: majority (s_role, o_role) after role_typed.

    Tie-break: among max-count candidates, pick lexicographically smallest
    (s_role, o_role) tuple. Returns (maj_role, n_parse_fail_fit).
    """
    role_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    n_fail = 0
    for _q, a in zip(fit_q, fit_a, strict=True):
        try:
            pq = parse_sparql_edges(a)
        except ValueError:
            n_fail += 1
            continue
        for e in pq.edges:
            rt = role_typed(e)
            role_counts[rt[1]][(rt[0], rt[2])] += 1
    maj_role: dict[str, tuple[str, str]] = {}
    for p, ctr in role_counts.items():
        if not ctr:
            continue
        max_c = max(ctr.values())
        # deterministic tie-break: lex-smallest among tied max-count pairs
        candidates = sorted(pair for pair, c in ctr.items() if c == max_c)
        maj_role[p] = candidates[0]
    return maj_role, n_fail


def stratum_f1(pred: list[Edge], gold: list[Edge], *, anchored: bool) -> float:
    """edge_f1 after filtering both bags to anchored or var-join role-typed edges."""
    if anchored:
        p = [e for e in pred if is_anchored_rt(e)]
        g = [e for e in gold if is_anchored_rt(e)]
    else:
        p = [e for e in pred if not is_anchored_rt(e)]
        g = [e for e in gold if not is_anchored_rt(e)]
    return edge_f1(p, g)


def run_split(tag: str) -> dict:
    blob = load_split(tag)
    tr_q, tr_a = blob["train_q"], blob["train_sparql"]
    te_q, te_a = blob["test_q"], blob["test_sparql"]

    # fit / val from train — VERBATIM from campaign_cfq_residual.run_split
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

    # --- gold role-typed + parse test (consistent index set) ---
    gold_rt: list[list[Edge]] = []
    parsed_te: list[tuple[str, str]] = []  # successfully-parsed test pairs
    n_parse_fail_test = 0
    for q, a in te_pairs:
        try:
            pq = parse_sparql_edges(a)
        except ValueError:
            n_parse_fail_test += 1
            continue
        rt_edges = [role_typed(e) for e in pq.edges]
        # Equivalence: pq.anchored[j] == ('ENT' in (rt[0], rt[2])) for every edge
        for j, (rt, anc) in enumerate(zip(rt_edges, pq.anchored, strict=True)):
            ent_in = "ENT" in (rt[0], rt[2])
            if anc != ent_in:
                raise RuntimeError(
                    f"anchored-mask equivalence broke on split={tag} "
                    f"edge_j={j} raw_edge={pq.edges[j]!r} role_typed={rt!r} "
                    f"pq.anchored={anc} ENT_in_rt={ent_in}"
                )
        gold_rt.append(rt_edges)
        parsed_te.append((q, a))

    n_test = len(parsed_te)
    n_fit = len(fit_q)

    # --- bag set-F1 (REFERENCE ONLY — incommensurable with edge-F1) ---
    # empirical: predicate-bag baseline for context, not comparable to edge-F1
    base_maps = mine_base_atoms(fit_q, fit_a, max_ns=1, min_support=2)
    bag_setf1 = mean_set_f1(te_pairs, base_maps)  # empirical; full te_pairs as campaign

    # --- majority role-kind prior on fit (full edge-predicate space) ---
    maj_role, n_parse_fail_fit_maj = majority_roles(fit_q, fit_a)

    # --- word→edge-predicate maps (reuse predict_paths) ---
    edge_pred_maps, n_parse_fail_fit_mine = mine_edge_pred_atoms(
        fit_q, fit_a, min_support=2,
    )
    # fit parse fails counted once (same parser; take max of the two passes)
    n_parse_fail_fit = max(n_parse_fail_fit_maj, n_parse_fail_fit_mine)

    # --- role-prior decode + oracle-pred+majrole + stratification ---
    role_prior_scores: list[float] = []
    role_prior_anch: list[float] = []
    role_prior_vj: list[float] = []
    oracle_scores: list[float] = []
    oracle_anch: list[float] = []
    oracle_vj: list[float] = []

    for i, (q, _a) in enumerate(parsed_te):
        gold_i = gold_rt[i]

        # sanity: identity edge_f1 must be 1.0 for every test query
        assert edge_f1(gold_i, gold_i) == 1.0, (
            f"edge_f1(gold_rt, gold_rt) != 1.0 at test i={i}"
        )

        # 5. realistic: predicted predicates from word maps + maj role
        predicted_preds = predict_paths(q, edge_pred_maps)
        pred_rt: list[Edge] = []
        for ep in predicted_preds:
            if ep not in maj_role:
                continue  # never seen in fit — skip, do not guess
            s_role, o_role = maj_role[ep]
            pred_rt.append((s_role, ep, o_role))
        role_prior_scores.append(edge_f1(pred_rt, gold_i))
        role_prior_anch.append(stratum_f1(pred_rt, gold_i, anchored=True))
        role_prior_vj.append(stratum_f1(pred_rt, gold_i, anchored=False))

        # 6. oracle-pred + majority-role (perfect predicate multiset recall)
        # re-parse not needed: reconstruct gold edge list from gold_rt predicates
        # but we need gold role_typed edges for unseen-predicate fallback —
        # gold_rt[i] IS role_typed of each edge occurrence with multiplicity.
        pq_edges_rt = gold_i  # already role_typed multiset, source order
        # gold predicates WITH multiplicity (from role-typed edges' pred field)
        gold_preds_with_multiplicity = [e[1] for e in pq_edges_rt]
        oracle_rt: list[Edge] = []
        for j, ep in enumerate(gold_preds_with_multiplicity):
            if ep in maj_role:
                s_role, o_role = maj_role[ep]
                oracle_rt.append((s_role, ep, o_role))
            else:
                # unseen in fit: fall back to THIS query's own gold role-typed edge
                oracle_rt.append(pq_edges_rt[j])
        oracle_scores.append(edge_f1(oracle_rt, gold_i))
        oracle_anch.append(stratum_f1(oracle_rt, gold_i, anchored=True))
        oracle_vj.append(stratum_f1(oracle_rt, gold_i, anchored=False))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    # empirical averages over successfully-parsed test queries
    role_prior_overall = _mean(role_prior_scores)
    role_prior_anchored = _mean(role_prior_anch)
    role_prior_varjoin = _mean(role_prior_vj)
    oracle_overall = _mean(oracle_scores)
    oracle_anchored = _mean(oracle_anch)
    oracle_varjoin = _mean(oracle_vj)

    # gap = predicate-recall cost (oracle has perfect pred multiset; may be <0 — report honestly)
    gap = oracle_overall - role_prior_overall  # empirical

    # verdict branches on measured oracle_pred_majrole overall (X)
    x = oracle_overall
    if x >= 0.95:
        verdict = (
            f"ROLE ~DETERMINISTIC GIVEN PREDICATE: majority-role recovers {x:.3f} "
            f"of structure; residual loss is predicate-recall/multiplicity, little "
            f"per-question role headroom."
        )
    else:
        verdict = (
            f"PER-QUESTION ROLE HEADROOM: oracle-pred+majrole={x:.3f}<1 ⇒ up to "
            f"{1 - x:.3f} role structure varies per instance "
            f"(future question-conditioned decode)."
        )

    sanity_msg = (
        f"sanity: edge_f1(gold_rt,gold_rt)==1.0 for all {n_test} test queries: PASS"
    )

    log(f"\n=== CFQ/{tag} edge headroom (role-typed structure diagnostic) ===")
    log(f"  n_test={n_test}  n_fit={n_fit}  "
        f"n_parse_fail_fit={n_parse_fail_fit}  n_parse_fail_test={n_parse_fail_test}")
    log(
        f"  bag_setf1={bag_setf1:.4f}  "
        f"(REFERENCE ONLY — predicate-bag set-F1, INCOMMENSURABLE with edge-F1 "
        f"below, do not compare directly)"
    )
    log(
        f"  role_prior_edge_f1: overall={role_prior_overall:.4f}  "
        f"anchored={role_prior_anchored:.4f}  var-join={role_prior_varjoin:.4f}"
    )
    log(
        f"  oracle_pred_majrole_edge_f1: overall={oracle_overall:.4f}  "
        f"anchored={oracle_anchored:.4f}  var-join={oracle_varjoin:.4f}"
    )
    log(
        f"  predicate-recall cost (gap = oracle - role_prior): {gap:+.4f}  "
        f"[empirical; not asserted ≥0]"
    )
    log(f"  VERDICT: {verdict}")
    log(f"  {sanity_msg}")
    log(f"  maj_role size={len(maj_role)}  edge_pred_maps size={len(edge_pred_maps)}")

    # sample of maj_role for JSON (small, deterministic)
    sample_keys = sorted(maj_role.keys())[:8]
    maj_sample = {k: maj_role[k] for k in sample_keys}

    return {
        "split": tag,
        "n_test": n_test,
        "n_fit": n_fit,
        "n_parse_fail_fit": n_parse_fail_fit,
        "n_parse_fail_test": n_parse_fail_test,
        "bag_setf1": bag_setf1,  # empirical; REFERENCE ONLY / incommensurable
        "role_prior_edge_f1": {
            "overall": role_prior_overall,  # empirical
            "anchored": role_prior_anchored,  # empirical
            "var_join": role_prior_varjoin,  # empirical
        },
        "oracle_pred_majrole_edge_f1": {
            "overall": oracle_overall,  # empirical
            "anchored": oracle_anchored,  # empirical
            "var_join": oracle_varjoin,  # empirical
        },
        "gap_predicate_recall_cost": gap,  # empirical
        "verdict": verdict,
        "sanity_pass": True,
        "maj_role_size": len(maj_role),
        "maj_role_sample": maj_sample,
        "edge_pred_maps_size": len(edge_pred_maps),
        "note": (
            "bag_setf1 is REFERENCE ONLY (predicate-bag set-F1) and INCOMMENSURABLE "
            "with edge-F1 numbers. This is a measurement diagnostic with no promised "
            "lift; it measures role-typed structure headroom over a majority-role prior."
        ),
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

    log("\n" + "=" * 90)
    log("CFQ EDGE HEADROOM SCOREBOARD (role-typed edge-F1; bag_setf1 REFERENCE ONLY)")
    log("=" * 90)
    log(
        f"{'split':8} {'n_te':>5} {'bagF1':>7} "
        f"{'rpO':>7} {'rpA':>7} {'rpV':>7} "
        f"{'orO':>7} {'orA':>7} {'orV':>7}"
    )
    for r in results:
        rp = r["role_prior_edge_f1"]
        op = r["oracle_pred_majrole_edge_f1"]
        log(
            f"{r['split']:8} {r['n_test']:5d} {r['bag_setf1']:7.3f} "
            f"{rp['overall']:7.3f} {rp['anchored']:7.3f} {rp['var_join']:7.3f} "
            f"{op['overall']:7.3f} {op['anchored']:7.3f} {op['var_join']:7.3f}"
        )

    out_dir = REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "bag_setf1 is reference-only / incommensurable with edge-F1. "
            "Measurement diagnostic only — no promised lift."
        ),
        "splits": results,
    }
    out = out_dir / "cfq_edge_headroom.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
