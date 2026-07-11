"""CFQ role-decode measurement: question-conditioned per-slot roles vs majority prior.

MEASUREMENT ONLY — no lift promised. Oracle-predicate regime: the test-time
predicate multiset is taken from gold SPARQL (NOT predicted). Roles are scored
at multiset edge-F1 level (not occurrence-aligned to a specific edge slot within
a query).

LEAK GUARD: no decoder or baseline may read gold ROLES from test SPARQL at
predict time. Only the gold PREDICATE multiset (with multiplicity) + the raw
question string are used for prediction. Unseen predicates fall back to a
fit-derived global_default (s_role, o_role) pair — NEVER to the current test
query's own gold role_typed edge (that was a leak in the older headroom
campaign oracle_pred path; this file deliberately does not reproduce it).

Scoring against gold role_typed edges after prediction is standard supervised
eval and is required; predicting from gold roles is the forbidden leak.

Run (from repo root):
  CFQ_SPLITS=mcd1,mcd2,mcd3 .venv/bin/python experiments/campaign_cfq_role_decode.py
Env: CFQ_SPLITS=mcd1  CFQ_MAX_VAL=2000
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.campaign_cfq_edge_headroom import (  # noqa: E402
    is_anchored_rt,
    majority_roles,
    mine_edge_pred_atoms,
    stratum_f1,
)

# Reuse-contract import (stratum helpers); keep name bound for API parity with headroom.
assert callable(is_anchored_rt)
from experiments.campaign_cfq_residual import content_words, load_split  # noqa: E402
from pil.cfq_edges import Edge, edge_f1, parse_sparql_edges, role_typed  # noqa: E402

# Mirrored from campaign_cfq_residual._STOP so local raw-token filtering aligns
# with content_words without modifying that module.
_STOP = frozenset({
    "a", "an", "the", "of", "to", "and", "or", "was", "is", "did", "do", "does",
    "were", "are", "by", "with", "from", "in", "on", "for", "s", "that", "which",
    "who", "what", "how", "many", "whose", "'s", ",", "?", ".",
})

_MENTION_RE = re.compile(r"M\d+")
_MENTION_WORD_RE = re.compile(r"\bM\d+\b")
_RAW_TOK_RE = re.compile(r"[A-Za-z0-9']+")


def log(m: str = "") -> None:
    print(m, flush=True)


def _majority_str(ctr: Counter[str], default: str = "VAR") -> str:
    """Most common key; lex-smallest among max-count ties; default if empty."""
    if not ctr:
        return default
    max_c = max(ctr.values())
    return sorted(k for k, c in ctr.items() if c == max_c)[0]


def _majority_pair(ctr: Counter[tuple[str, str]]) -> tuple[str, str]:
    """Most common (s,o) pair; lex-smallest among max-count ties."""
    max_c = max(ctr.values())
    return sorted(pair for pair, c in ctr.items() if c == max_c)[0]


def _content_words_with_pos(q: str) -> tuple[list[str], list[int], list[str]]:
    """Align content_words indices to raw token indices.

    raw_toks use ``[A-Za-z0-9']+`` (case preserved). Kept content words are
    filtered from those same raw tokens by lowercasing + stopword/len filter
    (matching content_words intent). Returns (kept_words, kept_raw_indices, raw_toks).

    Anchor positions come from kept_words (same filter as content_words for
    typical CFQ tokens). ENT/NONENT windowing always uses raw_toks slices around
    the mapped raw index — including stopwords like "a"/"an" that content_words
    drops.
    """
    raw_toks = _RAW_TOK_RE.findall(q)
    kept_words: list[str] = []
    kept_pos: list[int] = []
    for i, tok in enumerate(raw_toks):
        low = tok.lower()
        # content_words tokenizes on [A-Za-z0-9_]+ after lowercasing the whole
        # string; for alnum tokens this is equivalent to lowercasing the token.
        if low in _STOP or len(low) <= 1:
            continue
        kept_words.append(low)
        kept_pos.append(i)
    return kept_words, kept_pos, raw_toks


def _classify_window(raw_window: list[str]) -> str:
    """Classify a raw-token window as ENT / NONENT / AMBIG.

    Rule (same for subj and obj windows):
      - ENT if any window token fullmatches M\\d+
      - elif NONENT if any window token lowercased is in {a, an}
      - else AMBIG
    """
    if any(_MENTION_RE.fullmatch(t) for t in raw_window):
        return "ENT"
    if any(t.lower() in {"a", "an"} for t in raw_window):
        return "NONENT"
    return "AMBIG"


def fit_role_stats(fit_q: list[str], fit_a: list[str]) -> dict[str, Any]:
    """Fit-time role statistics for baselines + decoder (no test data)."""
    maj_role, n_parse_fail_maj = majority_roles(fit_q, fit_a)
    edge_pred_maps, n_parse_fail_mine = mine_edge_pred_atoms(fit_q, fit_a, min_support=2)
    n_parse_fail = max(n_parse_fail_maj, n_parse_fail_mine)

    subj_counts: dict[str, Counter[str]] = defaultdict(Counter)
    obj_counts: dict[str, Counter[str]] = defaultdict(Counter)
    nonent_subj: dict[str, Counter[str]] = defaultdict(Counter)
    nonent_obj: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_counts: dict[tuple[str, int], Counter[tuple[str, str]]] = defaultdict(Counter)
    global_ctr: Counter[tuple[str, str]] = Counter()
    preds_seen: set[str] = set()

    for q, a in zip(fit_q, fit_a, strict=True):
        try:
            pq = parse_sparql_edges(a)
        except ValueError:
            # already counted via majority_roles / mine passes; skip edges
            continue
        bucket = min(len(_MENTION_WORD_RE.findall(q)), 3)
        for e in pq.edges:
            rt = role_typed(e)
            s_role, p, o_role = rt
            preds_seen.add(p)
            subj_counts[p][s_role] += 1
            obj_counts[p][o_role] += 1
            if s_role != "ENT":
                nonent_subj[p][s_role] += 1
            if o_role != "ENT":
                nonent_obj[p][o_role] += 1
            bucket_counts[(p, bucket)][(s_role, o_role)] += 1
            global_ctr[(s_role, o_role)] += 1

    maj_subj: dict[str, str] = {}
    maj_obj: dict[str, str] = {}
    maj_nonent_subj: dict[str, str] = {}
    maj_nonent_obj: dict[str, str] = {}
    for p in preds_seen:
        maj_subj[p] = _majority_str(subj_counts[p], default="VAR")
        maj_obj[p] = _majority_str(obj_counts[p], default="VAR")
        maj_nonent_subj[p] = _majority_str(nonent_subj[p], default="VAR")
        maj_nonent_obj[p] = _majority_str(nonent_obj[p], default="VAR")

    maj_role_bucket: dict[tuple[str, int], tuple[str, str]] = {}
    for key, ctr in bucket_counts.items():
        if ctr:
            maj_role_bucket[key] = _majority_pair(ctr)

    if global_ctr:
        global_default = _majority_pair(global_ctr)
    else:
        global_default = ("VAR", "VAR")

    return {
        "maj_role": maj_role,
        "maj_nonent_subj": maj_nonent_subj,
        "maj_nonent_obj": maj_nonent_obj,
        "maj_subj": maj_subj,
        "maj_obj": maj_obj,
        "maj_role_bucket": maj_role_bucket,
        "edge_pred_maps": edge_pred_maps,
        "global_default": global_default,
        "n_parse_fail": n_parse_fail,
    }


def predict_majority_joint(preds: list[str], stats: dict[str, Any]) -> list[Edge]:
    """Per-predicate joint majority roles; unseen → fit global_default (no gold leak).

    Note: fair no-leak baseline; will differ slightly from the older headroom
    campaign's ~0.427 oracle_pred_majrole figure, which fell back to the test
    query's own gold role_typed edge on unseen predicates. That discrepancy is
    expected — do not "fix" this path to match the leaked number.
    """
    maj_role: dict[str, tuple[str, str]] = stats["maj_role"]
    gd0, gd1 = stats["global_default"]
    out: list[Edge] = []
    for p in preds:
        if p in maj_role:
            s, o = maj_role[p]
            out.append((s, p, o))
        else:
            out.append((gd0, p, gd1))
    return out


def predict_per_slot_marginal(preds: list[str], stats: dict[str, Any]) -> list[Edge]:
    """Independent per-predicate marginal subject/object majority roles."""
    maj_subj: dict[str, str] = stats["maj_subj"]
    maj_obj: dict[str, str] = stats["maj_obj"]
    gd0, gd1 = stats["global_default"]
    out: list[Edge] = []
    for p in preds:
        out.append((maj_subj.get(p, gd0), p, maj_obj.get(p, gd1)))
    return out


def predict_global_feature_prior(
    preds: list[str],
    q: str,
    stats: dict[str, Any],
) -> list[Edge]:
    """Joint majority conditioned on mention-count bucket of the raw question."""
    bucket = min(len(re.findall(r"M\d+", q)), 3)
    maj_role_bucket: dict[tuple[str, int], tuple[str, str]] = stats["maj_role_bucket"]
    maj_role: dict[str, tuple[str, str]] = stats["maj_role"]
    gd0, gd1 = stats["global_default"]
    out: list[Edge] = []
    for p in preds:
        if (p, bucket) in maj_role_bucket:
            s, o = maj_role_bucket[(p, bucket)]
            out.append((s, p, o))
        elif p in maj_role:
            s, o = maj_role[p]
            out.append((s, p, o))
        else:
            out.append((gd0, p, gd1))
    return out


def decode_roles(
    preds: list[str],
    q: str,
    stats: dict[str, Any],
) -> tuple[list[Edge], dict[str, int]]:
    """Per-slot local-surface role decoder (oracle predicates, no gold roles).

    For each predicate occurrence: locate a unique content-word anchor for that
    predicate via edge_pred_maps; classify subject/object windows on RAW tokens
    around the anchor (ENT via M\\d+, NONENT via a/an); map NONENT to the
    fit-time non-ENT majority role for that slot. Ambiguous anchors/windows fall
    back to maj_role[p]; unseen predicates fall back to global_default.
    """
    maj_role: dict[str, tuple[str, str]] = stats["maj_role"]
    maj_nonent_subj: dict[str, str] = stats["maj_nonent_subj"]
    maj_nonent_obj: dict[str, str] = stats["maj_nonent_obj"]
    edge_pred_maps = stats["edge_pred_maps"]
    gd0, gd1 = stats["global_default"]

    kept_words, kept_pos, raw_toks = _content_words_with_pos(q)
    # Prefer residual content_words for anchor membership (contract reuse);
    # position indices are taken from kept_words which mirrors that filter.
    cw = content_words(q)
    # If tokenizers diverge (rare apostrophe edge cases), fall back to kept_words
    # for both membership and indices so windows stay consistent.
    if cw != kept_words:
        cw = kept_words

    out: list[Edge] = []
    fallback = 0
    unseen = 0
    total = 0

    for p in preds:
        total += 1
        if p not in maj_role:
            out.append((gd0, p, gd1))
            unseen += 1
            continue

        anchor_words = [w for w in cw if (w, p) in edge_pred_maps]
        positions: set[int] = set()
        for w in anchor_words:
            for i, tok in enumerate(cw):
                if tok == w:
                    positions.add(i)
        if len(positions) != 1:
            s, o = maj_role[p]
            out.append((s, p, o))
            fallback += 1
            continue

        i = next(iter(positions))
        if i >= len(kept_pos):
            s, o = maj_role[p]
            out.append((s, p, o))
            fallback += 1
            continue
        ri = kept_pos[i]
        # Object: up to 3 raw tokens strictly AFTER anchor; subject: up to 3 BEFORE.
        obj_window = raw_toks[ri + 1 : ri + 4]
        subj_window = raw_toks[max(0, ri - 3) : ri]
        subj_cls = _classify_window(subj_window)
        obj_cls = _classify_window(obj_window)
        if subj_cls == "AMBIG" or obj_cls == "AMBIG":
            s, o = maj_role[p]
            out.append((s, p, o))
            fallback += 1
            continue

        if subj_cls == "ENT":
            s_role = "ENT"
        else:
            s_role = maj_nonent_subj.get(p, "VAR")
        if obj_cls == "ENT":
            o_role = "ENT"
        else:
            o_role = maj_nonent_obj.get(p, "VAR")
        out.append((s_role, p, o_role))

    return out, {"fallback": fallback, "unseen": unseen, "total": total}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_split(tag: str) -> dict[str, Any]:
    blob = load_split(tag)
    tr_q, tr_a = blob["train_q"], blob["train_sparql"]
    te_q, te_a = blob["test_q"], blob["test_sparql"]

    # fit / val from train — VERBATIM from campaign_cfq_edge_headroom.run_split
    idx = list(range(len(tr_q)))
    random.Random(0).shuffle(idx)
    nval = min(int(os.environ.get("CFQ_MAX_VAL", "2000")), max(1, len(idx) // 10))
    val_i, fit_i = idx[:nval], idx[nval:]
    fit_q = [tr_q[i] for i in fit_i]
    fit_a = [tr_a[i] for i in fit_i]
    _ = val_i  # unused here (no admit loop); kept for split parity
    te_pairs = list(zip(te_q, te_a, strict=True))
    if len(te_pairs) > 3000:
        te_pairs = [
            te_pairs[i]
            for i in range(0, len(te_pairs), max(1, len(te_pairs) // 3000))
        ][:3000]

    stats = fit_role_stats(fit_q, fit_a)
    n_parse_fail_fit = stats["n_parse_fail"]
    n_fit = len(fit_q)

    methods = ("majority_joint", "per_slot_marginal", "global_feature_prior", "decoder")
    scores: dict[str, list[float]] = {m: [] for m in methods}
    anch: dict[str, list[float]] = {m: [] for m in methods}
    vj: dict[str, list[float]] = {m: [] for m in methods}

    sum_fallback = 0
    sum_unseen = 0
    sum_total = 0
    n_parse_fail_test = 0

    for q, a in te_pairs:
        try:
            pq = parse_sparql_edges(a)
        except ValueError:
            n_parse_fail_test += 1
            continue
        gold_rt = [role_typed(e) for e in pq.edges]
        gold_preds = [e[1] for e in gold_rt]

        preds_by_method: dict[str, list[Edge]] = {
            "majority_joint": predict_majority_joint(gold_preds, stats),
            "per_slot_marginal": predict_per_slot_marginal(gold_preds, stats),
            "global_feature_prior": predict_global_feature_prior(gold_preds, q, stats),
        }
        dec_edges, counts = decode_roles(gold_preds, q, stats)
        preds_by_method["decoder"] = dec_edges
        sum_fallback += counts["fallback"]
        sum_unseen += counts["unseen"]
        sum_total += counts["total"]

        for m, pred in preds_by_method.items():
            scores[m].append(edge_f1(pred, gold_rt))
            anch[m].append(stratum_f1(pred, gold_rt, anchored=True))
            vj[m].append(stratum_f1(pred, gold_rt, anchored=False))

    n_test = len(scores["majority_joint"])
    overall = {m: _mean(scores[m]) for m in methods}
    anchored = {m: _mean(anch[m]) for m in methods}
    var_join = {m: _mean(vj[m]) for m in methods}

    fallback_rate = (sum_fallback / sum_total) if sum_total else 0.0
    unseen_rate = (sum_unseen / sum_total) if sum_total else 0.0

    decoder_overall = overall["decoder"]
    majority_joint_overall = overall["majority_joint"]
    decoder_wins_split = decoder_overall > majority_joint_overall
    anchored_gain = anchored["decoder"] - anchored["majority_joint"]
    overall_gain = decoder_overall - majority_joint_overall

    if decoder_wins_split:
        if anchored_gain > overall_gain:
            gain_note = "measured gain concentrated in anchored stratum"
        else:
            gain_note = "measured gain not concentrated in anchored stratum"
        verdict = (
            f"decoder beats majority_joint (no-leak) on {tag}: "
            f"overall {decoder_overall:.4f} > {majority_joint_overall:.4f} "
            f"(overall_gain={overall_gain:+.4f}, anchored_gain={anchored_gain:+.4f}; "
            f"{gain_note})"
        )
    else:
        verdict = (
            f"decoder does not beat majority_joint (no-leak) on {tag}: "
            f"overall {decoder_overall:.4f} <= {majority_joint_overall:.4f} "
            f"(overall_gain={overall_gain:+.4f}, anchored_gain={anchored_gain:+.4f}; "
            f"empirical loss or tie — no spin)"
        )

    log(f"\n=== CFQ/{tag} role-decode (oracle-pred, no-role-leak, multiset edge-F1) ===")
    log(
        f"  n_test={n_test}  n_fit={n_fit}  "
        f"n_parse_fail_fit={n_parse_fail_fit}  n_parse_fail_test={n_parse_fail_test}"
    )
    for m in methods:
        log(
            f"  {m:22s}  overall={overall[m]:.4f}  "
            f"anchored={anchored[m]:.4f}  var-join={var_join[m]:.4f}"
        )
    log(
        f"  decoder fallback_rate={fallback_rate:.4f}  unseen_rate={unseen_rate:.4f}  "
        f"(fallback={sum_fallback} unseen={sum_unseen} total={sum_total})"
    )
    log(f"  VERDICT: {verdict}")
    log(
        f"  maj_role size={len(stats['maj_role'])}  "
        f"edge_pred_maps size={len(stats['edge_pred_maps'])}  "
        f"global_default={stats['global_default']!r}"
    )

    return {
        "split": tag,
        "n_test": n_test,
        "n_fit": n_fit,
        "n_parse_fail_fit": n_parse_fail_fit,
        "n_parse_fail_test": n_parse_fail_test,
        "overall": overall,
        "anchored": anchored,
        "var_join": var_join,
        "decoder_fallback_rate": fallback_rate,
        "decoder_unseen_rate": unseen_rate,
        "decoder_fallback_n": sum_fallback,
        "decoder_unseen_n": sum_unseen,
        "decoder_total_n": sum_total,
        "decoder_wins_split": decoder_wins_split,
        "anchored_gain": anchored_gain,
        "overall_gain": overall_gain,
        "verdict": verdict,
        "maj_role_size": len(stats["maj_role"]),
        "edge_pred_maps_size": len(stats["edge_pred_maps"]),
        "global_default": stats["global_default"],
        "note": (
            "Oracle-predicate regime (gold predicate multiset, predicted roles). "
            "Multiset edge-F1 (not occurrence-aligned). No gold ROLE used at predict "
            "time; unseen predicates use fit-derived global_default only."
        ),
    }


def main() -> None:
    want = [
        s.strip()
        for s in os.environ.get("CFQ_SPLITS", "mcd1").split(",")
        if s.strip()
    ]
    results: list[dict[str, Any]] = []
    for tag in want:
        pt = REPO / "data" / f"cfq_{tag}.pt"
        if not pt.exists():
            log(f"skip {tag}: missing data/cfq_{tag}.pt")
            continue
        results.append(run_split(tag))

    log("\n" + "=" * 100)
    log(
        "CFQ ROLE-DECODE SCOREBOARD (oracle-pred, no-role-leak, multiset role-typed edge-F1)"
    )
    log("=" * 100)
    log(
        f"{'split':8} {'n_te':>5} "
        f"{'mjO':>7} {'margO':>7} {'gfpO':>7} {'decO':>7} "
        f"{'decA':>7} {'decV':>7} "
        f"{'fbR':>7} {'unR':>7} {'win':>5}"
    )
    for r in results:
        o = r["overall"]
        a = r["anchored"]
        v = r["var_join"]
        log(
            f"{r['split']:8} {r['n_test']:5d} "
            f"{o['majority_joint']:7.3f} {o['per_slot_marginal']:7.3f} "
            f"{o['global_feature_prior']:7.3f} {o['decoder']:7.3f} "
            f"{a['decoder']:7.3f} {v['decoder']:7.3f} "
            f"{r['decoder_fallback_rate']:7.3f} {r['decoder_unseen_rate']:7.3f} "
            f"{str(r['decoder_wins_split']):>5}"
        )

    wins = [bool(r["decoder_wins_split"]) for r in results]
    all_win = bool(wins) and all(wins)
    log(
        f"decoder beats majority_joint (no-leak) on ALL {len(results)} run splits: "
        f"{all_win}"
    )

    out_dir = REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "Oracle-predicate regime: test predicate multiset from gold, roles predicted "
            "from question + fit stats only. Multiset-level edge-F1 (not occurrence-aligned). "
            "LEAK GUARD: no gold ROLE at predict time; unseen predicates → fit global_default. "
            "Measurement only — no promised lift."
        ),
        "decoder_beats_majority_joint_on_all_splits": all_win,
        "n_splits": len(results),
        "splits": results,
    }
    out = out_dir / "cfq_role_decode.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
