"""CFQ join-structure headroom diagnostic (measurement only).

Oracle-predicate regime: predictors are handed the gold predicate multiset
and must predict JOIN STRUCTURE (which predicates share which variables).
This measures join-assembly given known predicates — a DIFFERENT question
from campaign_cfq_edge_headroom's role diagnostic (predicate recall / role
kinds). Scoring is multiset-level over label-free / isomorphism-invariant
per-var join signatures (Counter F1), NOT occurrence-aligned to any full
graph matching beyond per-signature overlap.

A low B (question-feature mean) is a LEXICAL-VOTE-FEATURE-RECIPE PLATEAU,
not a claim that join structure is "unbridgeable" or unlearnable in
principle. Measurement only — no mechanism/decoder is proposed here.
Ceiling is 1.0 by construction (access to gold.signatures trivially scores
1.0); we state that rather than implement a circular oracle path.

Run (from repo root, defaults to mcd1):
  .venv/bin/python experiments/campaign_cfq_join_headroom.py
Env: CFQ_SPLITS=mcd1,mcd2,mcd3  CFQ_MAX_VAL=2000
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
)
from pil.cfq_edges import join_f1, parse_sparql_joins  # noqa: E402


def log(m: str = "") -> None:
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Majority-join prior (predictor a) — factored for testability / leak guard
# ---------------------------------------------------------------------------


def fit_majority_join_prior(
    fit_pairs: list[tuple[str, str]],
) -> tuple[dict[tuple[str, ...], tuple], int]:
    """Fit majority join-signature multiset per oracle predicate multiset key.

    For each successfully-parsed fit query:
      key = tuple(sorted(pq.predicates))
      sig_multiset_key = tuple(sorted(pq.signatures))
    Count sig_multiset_key per key; collapse to majority (lex-smallest among
    max-count candidates on ties).

    Returns (maj_join, n_parse_fail). maj_join maps key -> winning
    signature-multiset tuple (list-ify at predict time).
    """
    counts: dict[tuple[str, ...], Counter[tuple]] = defaultdict(Counter)
    n_fail = 0
    for _q, sparql in fit_pairs:
        try:
            pq = parse_sparql_joins(sparql)
        except ValueError:
            n_fail += 1
            continue
        key = tuple(sorted(pq.predicates))
        sig_key = tuple(sorted(pq.signatures))
        counts[key][sig_key] += 1

    maj_join: dict[tuple[str, ...], tuple] = {}
    for key, ctr in counts.items():
        if not ctr:
            continue
        max_c = max(ctr.values())
        # deterministic tie-break: lex-smallest among max-count candidates
        candidates = sorted(sig for sig, c in ctr.items() if c == max_c)
        maj_join[key] = candidates[0]
    return maj_join, n_fail


def predict_majority_join(
    key: tuple[str, ...],
    maj_join: dict[tuple[str, ...], tuple],
) -> list:
    """Predict join signatures for an oracle predicate-multiset key.

    If key is unseen at fit time, return EMPTY list — NEVER fall back to gold.
    (Unlike campaign_cfq_edge_headroom's oracle_pred_majrole unseen-predicate
    fallback, which is a different, permitted choice there.)
    """
    if key not in maj_join:
        return []
    return list(maj_join[key])


# ---------------------------------------------------------------------------
# Question-feature -> join predictor (predictor b)
# ---------------------------------------------------------------------------


def mine_join_sig_atoms(
    fit_pairs: list[tuple[str, str]],
    *,
    min_support: int = 2,
) -> tuple[dict[str, set[tuple]], int]:
    """Word → set of join signatures (vote-then-threshold, mirror mine_edge_pred_atoms).

    For each fit query: content_words(q) x each signature in pq.signatures
    votes (word, signature). Keep pairs with count >= min_support.
    Returns (word -> set of signatures, n_parse_fail).
    """
    votes: Counter[tuple[str, tuple]] = Counter()
    n_fail = 0
    for q, sparql in fit_pairs:
        try:
            pq = parse_sparql_joins(sparql)
        except ValueError:
            n_fail += 1
            continue
        if not pq.signatures:
            continue
        words = content_words(q)
        for w in words:
            for sig in pq.signatures:
                votes[(w, sig)] += 1

    maps: dict[str, set[tuple]] = defaultdict(set)
    for (w, sig), c in votes.items():
        if c >= min_support:
            maps[w].add(sig)
    return dict(maps), n_fail


def predict_join_from_question(
    question: str,
    word_maps: dict[str, set[tuple]],
    *,
    oracle_predicates: list[str] | None = None,
) -> list:
    """Predicted signatures = mined sigs whose trigger word appears in question.

    Optionally restrict candidate signatures to those whose incidence predicates
    are a subset of the oracle predicate multiset (stronger, still no gold
    signatures). Default: no restriction (all mined sigs triggered by words).
    """
    words = content_words(question)
    pred_set: set[tuple] = set()
    for w in words:
        if w in word_maps:
            pred_set |= word_maps[w]

    if oracle_predicates is not None:
        # Optional consistency filter: every incidence pred in the signature
        # must appear in the oracle predicate multiset.
        allowed = set(oracle_predicates)
        filtered: set[tuple] = set()
        for sig in pred_set:
            # sig = (var_type, ((pred, slot), ...))
            incs = sig[1] if len(sig) > 1 else ()
            if all(pred in allowed for pred, _slot in incs):
                filtered.add(sig)
        pred_set = filtered

    return list(pred_set)


# ---------------------------------------------------------------------------
# Per-split run
# ---------------------------------------------------------------------------


def run_split(tag: str) -> dict:
    blob = load_split(tag)
    tr_q, tr_a = blob["train_q"], blob["train_sparql"]
    te_q, te_a = blob["test_q"], blob["test_sparql"]

    # fit / val from train — VERBATIM from campaign_cfq_edge_headroom / residual
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
    word_maps, n_parse_fail_fit_mine = mine_join_sig_atoms(fit_pairs, min_support=2)
    n_parse_fail_fit = max(n_parse_fail_fit_maj, n_parse_fail_fit_mine)

    # --- score test (oracle-predicate regime) ---
    scores_a: list[float] = []
    scores_b: list[float] = []
    n_join_vars_list: list[int] = []
    n_unseen_key = 0
    n_parse_fail_test = 0
    n_scored = 0

    for q, sparql in te_pairs:
        try:
            gold = parse_sparql_joins(sparql)
        except ValueError:
            n_parse_fail_test += 1
            continue

        n_scored += 1
        n_join_vars_list.append(gold.n_join_vars)

        # (a) majority-join prior — oracle predicates only; empty on unseen key
        key = tuple(sorted(gold.predicates))
        pred_a = predict_majority_join(key, maj_join)
        if key not in maj_join:
            n_unseen_key += 1
        scores_a.append(join_f1(pred_a, gold.signatures))

        # (b) question-feature predictor — question + optional oracle-pred filter
        # Restrict candidates to signatures whose incidence preds ⊆ oracle
        # predicate multiset (stronger; still never reads gold.signatures).
        pred_b = predict_join_from_question(
            q, word_maps, oracle_predicates=gold.predicates,
        )
        scores_b.append(join_f1(pred_b, gold.signatures))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    a_mean = _mean(scores_a)
    b_mean = _mean(scores_b)
    unseen_key_rate = n_unseen_key / n_scored if n_scored else 0.0
    frac_queries_with_join = (
        sum(1 for n in n_join_vars_list if n >= 1) / n_scored if n_scored else 0.0
    )
    mean_join_vars = _mean([float(n) for n in n_join_vars_list])

    # ceiling: 1.0 by construction (predictor with gold.signatures scores 1.0)
    ceiling_note = (
        "oracle/ceiling = 1.0 by construction "
        "(access to gold.signatures trivially scores 1.0; not computed)"
    )

    # verdict from measured numbers — exact thresholds 0.9 and +0.02
    if a_mean >= 0.9:
        verdict = (
            f"SCHEMA-DETERMINISTIC: join ≈ determined by predicate multiset "
            f"(A={a_mean:.3f}); CFQ join is assembly, not induction."
        )
    elif b_mean > a_mean + 0.02:
        verdict = (
            f"QUESTION-PREDICTABLE: question features add join signal "
            f"(B={b_mean:.3f} > A={a_mean:.3f})."
        )
    else:
        verdict = (
            f"NOT QUESTION-PREDICTABLE UNDER LEXICAL-VOTE FEATURES: "
            f"A={a_mean:.3f} B={b_mean:.3f} "
            f"(recipe plateau; achievability OPEN — NOT unbridgeable)."
        )

    log(f"\n=== CFQ/{tag} join headroom (oracle-predicate regime) ===")
    log(
        f"  n_test={n_scored}  n_fit={len(fit_pairs)}  "
        f"n_parse_fail_fit={n_parse_fail_fit}  n_parse_fail_test={n_parse_fail_test}"
    )
    log(f"  A majority-join-prior mean join_f1: {a_mean:.4f}")
    log(f"  B question-feature mean join_f1:    {b_mean:.4f}")
    log(f"  unseen_key_rate: {unseen_key_rate:.4f}")
    log(f"  frac_queries_with_join: {frac_queries_with_join:.4f}")
    log(f"  mean_join_vars: {mean_join_vars:.4f}")
    log(f"  {ceiling_note}")
    log(f"  VERDICT: {verdict}")
    log(f"  maj_join size={len(maj_join)}  word_maps size={len(word_maps)}")

    return {
        "split": tag,
        "n_test": n_scored,
        "n_fit": len(fit_pairs),
        "n_parse_fail_fit": n_parse_fail_fit,
        "n_parse_fail_test": n_parse_fail_test,
        "majority_join_prior_mean": a_mean,  # A — empirical
        "question_feature_mean": b_mean,  # B — empirical
        "unseen_key_rate": unseen_key_rate,  # empirical
        "frac_queries_with_join": frac_queries_with_join,  # empirical
        "mean_join_vars": mean_join_vars,  # empirical
        "ceiling_note": ceiling_note,
        "verdict": verdict,
        "maj_join_size": len(maj_join),
        "word_maps_size": len(word_maps),
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
    log("CFQ JOIN HEADROOM SCOREBOARD (oracle-predicate; multiset join_f1)")
    log("=" * 90)
    log(
        f"{'split':8} {'n_te':>5} {'A_maj':>7} {'B_q':>7} "
        f"{'unseen':>7} {'wJoin':>7} {'mnJV':>7}"
    )
    for r in results:
        log(
            f"{r['split']:8} {r['n_test']:5d} "
            f"{r['majority_join_prior_mean']:7.3f} "
            f"{r['question_feature_mean']:7.3f} "
            f"{r['unseen_key_rate']:7.3f} "
            f"{r['frac_queries_with_join']:7.3f} "
            f"{r['mean_join_vars']:7.3f}"
        )

    out_dir = REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "Oracle-predicate regime: predictors get gold predicate multiset, "
            "predict join structure. Multiset join_f1 over label-free per-var "
            "signatures. Low B is a lexical-vote-feature-recipe plateau, not "
            "unbridgeable. Ceiling 1.0 by construction. Measurement only."
        ),
        "splits": results,
    }
    out = out_dir / "cfq_join_headroom.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
