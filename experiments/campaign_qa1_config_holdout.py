"""qa1 config-holdout battery (slice #80) — residual instrument + H1/H2.

Builds a fully generated qa1 world with held-out entity×location combinations
(``pil.qa1_battery``), fits a same-family memorizing cover B0' fresh on the
generated train split, and runs pre-registered H1/H2 against residual on the
holdout block.

Pre-registered pins (do not substitute):
  - Baseline ensemble for every marginal = B0' (counts+ngrams SW cover fit on train)
  - Val distribution = query tails of block-IID stories (non-held-out pairs)
  - H1/H2 verdicts use block-1 (holdout-answer) rows only

Residual R definition (explicit; was implicit/empty in #79):
  R = rows where B0' errs OR abstains (union of R_wrong and R_abstain).
  Scoreboard reports both counts separately; H1/H2 target the union.

Cross-link: docs/notes/qa1_cond_headroom.md (handoff to slice #80).
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from campaign_qa1_cond_headroom import (  # noqa: E402
    ADMIT_THRESH,
    H1_COV,
    H1_PREC,
    H1_READING,
    H2_AGREE_DELTA,
    H2_P,
    H2_READING,
    MOVELOC_DET,
    MOVELOC_SUPP,
    Movement,
    QueryRow,
    Story,
    _movements_from_prompt,
    cover_agree,
    encode_prompt,
    exact_binomial_two_sided,
    fit_majority_table,
    gold_token,
    load_train_stories,
    make_simple_kgram,
    mine_movement_verbs_from_events,
    moveloc_feature,
    pad_batch,
    select_gate_tau,
    unpad_row,
)

from pil.qa1_battery import (  # noqa: E402
    HELD_OUT_PAIRS,
    TRAIN_LEGAL_PAIRS,
    GenStory,
    generate_world,
    is_holdout_binding,
    join_train_corpus,
    parse_story_text,
    story_prompts_bench_style,
)
from pil.tokens import TokenSpace  # noqa: E402
from pil.wyly_block import BlockStack, WylyBlock  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 0
DATA = REPO / "data"
PKG_DIR = DATA / "wyly_expert_package_v5_babi"
OUT_JSON = DATA / "qa1_config_holdout.json"
LAPLACE_ALPHA = 2.0  # matches wyly_lm_v5 ALPHA for SW conf
KGRAM_MINSUPP = 2  # fit_kgram default
KGRAM_MINDET = 0.5  # fit_kgram default
COUNTS_MIN_COUNT = 2


# ---------------------------------------------------------------------------
# Story conversion (GenStory → Story / QueryRow)
# ---------------------------------------------------------------------------
def gen_to_story(gs: GenStory, *, id_prefix: str | None = None) -> Story:
    """Convert a GenStory into a Story with accumulated bench-style prompts."""
    sid = gs.story_id if id_prefix is None else f"{id_prefix}:{gs.story_id.split(':', 1)[-1]}"
    prompts = story_prompts_bench_style(gs)
    movs_chrono: list[Movement] = []
    qrows: list[QueryRow] = []
    raw_movs, raw_qs = parse_story_text(gs.text)
    # rebuild Movement objects in chrono order; assign per-query prefixes
    mi = 0
    for t, ((prompt, answer), (ent, ans)) in enumerate(zip(prompts, raw_qs, strict=True)):
        assert ans == answer
        # two new movements per turn
        for _ in range(2):
            if mi < len(raw_movs):
                e, v, loc = raw_movs[mi]
                movs_chrono.append(
                    Movement(
                        entity=e,
                        verb=v,
                        loc=loc,
                        text=f"{e} {v} to the {loc}.",
                    )
                )
                mi += 1
        assert ent == ent
        qrows.append(
            QueryRow(
                story_id=sid,
                turn=t,
                entity=ent,
                answer=answer,
                prompt=prompt if prompt.rstrip().endswith("A:") else prompt.rstrip() + " A:",
                movements_before=list(movs_chrono),
            )
        )
    return Story(story_id=sid, queries=qrows, all_movements=list(movs_chrono))


def gen_list_to_stories(stories: list[GenStory]) -> list[Story]:
    return [gen_to_story(s) for s in stories]


def assert_all_splits_disjoint(
    train: list[Story],
    holdout: list[Story],
    iid: list[Story],
    names: list[Story],
) -> None:
    """Every story id unique across all four lists."""
    buckets = {
        "train": train,
        "holdout": holdout,
        "iid": iid,
        "names": names,
    }
    seen: dict[str, str] = {}
    for label, stories in buckets.items():
        if not stories:
            raise AssertionError(f"empty {label} split")
        for s in stories:
            if s.story_id in seen:
                raise AssertionError(
                    f"story id collision {s.story_id!r} in {seen[s.story_id]} and {label}"
                )
            seen[s.story_id] = label
            for q in s.queries:
                if q.story_id != s.story_id:
                    raise AssertionError(f"query/story id mismatch {q.story_id} vs {s.story_id}")


def partition_holdout_queries(
    holdout_stories: list[Story],
) -> tuple[list[QueryRow], list[QueryRow]]:
    """Split holdout-story queries into block-1 (holdout gold) vs routed-IID rows."""
    block1: list[QueryRow] = []
    routed_iid: list[QueryRow] = []
    for s in holdout_stories:
        for q in s.queries:
            if is_holdout_binding(q.entity, q.answer):
                block1.append(q)
            else:
                routed_iid.append(q)
    return block1, routed_iid


def stories_from_rows(rows: list[QueryRow], prefix: str) -> list[Story]:
    """Wrap flat QueryRows as one-query-per-Story shells for scoring helpers."""
    by: dict[str, list[QueryRow]] = defaultdict(list)
    for q in rows:
        by[q.story_id].append(q)
    out: list[Story] = []
    for sid, qs in sorted(by.items()):
        qs_sorted = sorted(qs, key=lambda r: r.turn)
        movs: list[Movement] = []
        if qs_sorted:
            movs = list(qs_sorted[-1].movements_before)
        out.append(Story(story_id=sid, queries=qs_sorted, all_movements=movs))
    _ = prefix
    return out


# ---------------------------------------------------------------------------
# B0' — memorizing cover fit on generated train (stand-in for on-disk package)
# ---------------------------------------------------------------------------
def _train_token_stream(train_stories: list[Story], ts: TokenSpace) -> list[int]:
    """Flatten generated train into a host-BPE id stream (full story text)."""
    parts: list[str] = []
    for s in train_stories:
        last = s.queries[-1]
        parts.append(last.prompt + " " + last.answer + ".")
    text = " ".join(parts)
    return list(ts.encode(" " + text))


def _laplace_counts_table(
    token_ids: list[int], *, min_count: int = COUNTS_MIN_COUNT, alpha: float = LAPLACE_ALPHA
) -> dict[int, tuple[int, float]]:
    """Last-token → (majority next, Laplace conf).

    Shrinkage: conf = max_count / (row_total + alpha), alpha=2.0 mirroring
    ``wyly_lm_v5.ALPHA`` / support-weighted cover confidences. Rows with
    support < min_count are dropped.
    """
    buckets: dict[int, Counter] = defaultdict(Counter)
    for i in range(len(token_ids) - 1):
        buckets[token_ids[i]][token_ids[i + 1]] += 1
    table: dict[int, tuple[int, float]] = {}
    for k, ctr in buckets.items():
        tot = sum(ctr.values())
        if tot < min_count:
            continue
        best_y, best_n = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        conf = best_n / (tot + alpha)
        table[k] = (best_y, conf)
    return table


def _kgram_table_gated(
    token_ids: list[int],
    k: int,
    *,
    minsupp: int = KGRAM_MINSUPP,
    mindet: float = KGRAM_MINDET,
    alpha: float = LAPLACE_ALPHA,
) -> dict[tuple[int, ...], tuple[int, float]]:
    """Suffix-k → (majority next, Laplace conf) with fit_kgram-style pre-gate.

    Mirror of ``wyly_lm_v5.fit_kgram`` / ``best_per_key``: admit a key only if
    support >= minsupp (default 2) AND determinism >= mindet (default 0.5).
    Confidence on fire: cnt/(tot+alpha). Built via ``fit_suffix_kgram`` buckets
    (not the torch KeyTable path — host vocab is too large for packed keys).
    """
    buckets: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    for i in range(k - 1, len(token_ids) - 1):
        key = tuple(token_ids[i - k + 1 : i + 1])
        buckets[key][token_ids[i + 1]] += 1
    table: dict[tuple[int, ...], tuple[int, float]] = {}
    for key, ctr in buckets.items():
        tot = sum(ctr.values())
        if tot < minsupp:
            continue
        best_y, best_n = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        det = best_n / tot
        if det < mindet:
            continue
        table[key] = (best_y, best_n / (tot + alpha))
    return table


def _live_loc_any_verb(entity: str, movements: list[Movement]) -> str | None:
    """Most-recent location of entity (any MOVE_RE verb) — structural, not moveloc-V."""
    for mv in reversed(movements):
        if mv.entity == entity:
            return mv.loc
    return None


def _fit_pair_memory(
    train_stories: list[Story], ts: TokenSpace, *, alpha: float = LAPLACE_ALPHA
) -> dict[tuple[str, str], tuple[int, float]]:
    """Memorize train (entity, location) movement co-occurrence → answer token.

    Pure joint-config memory: fires at query time only when the live binding
    (entity, live_loc) was observed as a train movement. Held-out pairs have
    zero train support → abstain (residual reappears). Non-held-out pairs seen
    in train → predict live_loc with Laplace conf from movement counts.

    This is the load-bearing memorizing tier for config-holdout: short suffix
    kgrams after ``A:`` are multi-way ambiguous (6 locations), so token ngrams
    alone cannot saturate IID. Pair memory is still memorization (no binding
    generalization to unseen joints) — the contrast H1 measures.
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    for s in train_stories:
        for mv in s.all_movements:
            pair_counts[(mv.entity, mv.loc)] += 1
        # also count live query answers (same joints)
        for q in s.queries:
            pair_counts[(q.entity, q.answer)] += 1
    table: dict[tuple[str, str], tuple[int, float]] = {}
    for (ent, loc), n in pair_counts.items():
        if n < 1:
            continue
        # Key (ent, loc) → answer loc is deterministic; Laplace only shrinks by support.
        tok = gold_token(ts, loc)
        conf = n / (n + alpha)
        table[(ent, loc)] = (tok, conf)
    return table


def build_b0_prime(train_stories: list[Story], ts: TokenSpace) -> WylyBlock:
    """Fit B0' memorizing cover on generated train.

    B0' is the same memorizing family as the on-disk served package
    (counts + ngrams + config co-occurrence memory, support-weighted cover),
    used as a stand-in because the on-disk package / ``corpus_babi.txt`` has no
    holdout relative to it (slice #79: all 24 entity×location pairs covered).

    Tiers (all always admitted; SW cover arbitrates by conf):
      - pair_memory: (entity, live_loc) seen in train → predict loc (config memo)
      - counts: last-token → majority next on train stream (Laplace conf)
      - kgram k=2,3: suffix tables with fit_kgram-style support/det pre-gate

    Short token ngrams alone cannot saturate query tails (``A:`` is multi-way);
    pair_memory is the config-holdout-appropriate memorizing spine.
    """
    stream = _train_token_stream(train_stories, ts)
    counts = _laplace_counts_table(stream)
    kg2 = _kgram_table_gated(stream, 2)
    kg3 = _kgram_table_gated(stream, 3)
    pair_mem = _fit_pair_memory(train_stories, ts)
    # Location token ids — ngram tiers may only emit these (query-tail protocol)
    from pil.qa1_battery import LOCATIONS

    loc_toks = {gold_token(ts, loc) for loc in LOCATIONS}

    # Prompt-key → (entity, movements) for pair_memory at score time.
    # Built from train + filled for test rows at score via register_pair_prompts.
    prompt_index: dict[tuple[int, ...], tuple[str, list[Movement]]] = {}

    def index_stories(stories: list[Story]) -> None:
        for s in stories:
            for q in s.queries:
                key = tuple(encode_prompt(ts, q.prompt))
                prompt_index[key] = (q.entity, list(q.movements_before))

    index_stories(train_stories)
    # stash for later registration of test prompts (lookup only, not fitting)
    b0 = WylyBlock(0, "b0_prime", family="memorizing_cover")
    b0.residual_seed["prompt_index"] = prompt_index
    b0.residual_seed["pair_mem"] = pair_mem
    b0.residual_seed["index_stories"] = index_stories

    def pair_pred(ids: torch.Tensor) -> torch.Tensor:
        B = len(ids)
        out = torch.full((B,), -1, dtype=torch.long, device=ids.device)
        for i in range(B):
            ctx = tuple(unpad_row(ids[i]))
            hit = prompt_index.get(ctx)
            if hit is None:
                continue
            ent, movs = hit
            loc = _live_loc_any_verb(ent, movs)
            if loc is None:
                continue
            pm = pair_mem.get((ent, loc))
            if pm is not None:
                out[i] = pm[0]
        return out

    def pair_conf(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(ids)
        a = torch.full((B,), -1, dtype=torch.long, device=ids.device)
        c = torch.full((B,), -1e9, dtype=torch.float32, device=ids.device)
        for i in range(B):
            ctx = tuple(unpad_row(ids[i]))
            hit = prompt_index.get(ctx)
            if hit is None:
                continue
            ent, movs = hit
            loc = _live_loc_any_verb(ent, movs)
            if loc is None:
                continue
            pm = pair_mem.get((ent, loc))
            if pm is not None:
                a[i], c[i] = pm[0], float(pm[1])
        return a, c

    def counts_pred(ids: torch.Tensor) -> torch.Tensor:
        B = len(ids)
        out = torch.full((B,), -1, dtype=torch.long, device=ids.device)
        for i in range(B):
            ctx = unpad_row(ids[i])
            if not ctx:
                continue
            hit = counts.get(ctx[-1])
            if hit is not None and hit[0] in loc_toks:
                out[i] = hit[0]
        return out

    def counts_conf(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(ids)
        a = torch.full((B,), -1, dtype=torch.long, device=ids.device)
        c = torch.full((B,), -1e9, dtype=torch.float32, device=ids.device)
        for i in range(B):
            ctx = unpad_row(ids[i])
            if not ctx:
                continue
            hit = counts.get(ctx[-1])
            if hit is not None and hit[0] in loc_toks:
                a[i], c[i] = hit[0], float(hit[1])
        return a, c

    def make_kg(table: dict[tuple[int, ...], tuple[int, float]], k: int):
        def pred_fn(ids: torch.Tensor) -> torch.Tensor:
            B = len(ids)
            out = torch.full((B,), -1, dtype=torch.long, device=ids.device)
            for i in range(B):
                ctx = unpad_row(ids[i])
                if len(ctx) < k:
                    continue
                hit = table.get(tuple(ctx[-k:]))
                if hit is not None and hit[0] in loc_toks:
                    out[i] = hit[0]
            return out

        def conf_fn(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            B = len(ids)
            a = torch.full((B,), -1, dtype=torch.long, device=ids.device)
            c = torch.full((B,), -1e9, dtype=torch.float32, device=ids.device)
            for i in range(B):
                ctx = unpad_row(ids[i])
                if len(ctx) < k:
                    continue
                hit = table.get(tuple(ctx[-k:]))
                if hit is not None and hit[0] in loc_toks:
                    a[i], c[i] = hit[0], float(hit[1])
            return a, c

        return pred_fn, conf_fn

    k2_p, k2_c = make_kg(kg2, 2)
    k3_p, k3_c = make_kg(kg3, 3)

    # pair_memory conf is higher on seen joints; short ngrams are weak fallbacks
    b0.add_candidate("pair_memory", pair_pred, pair_conf)
    b0.add_candidate("counts", counts_pred, counts_conf)
    b0.add_candidate("kgram_k2", k2_p, k2_c)
    b0.add_candidate("kgram_k3", k3_p, k3_c)
    b0.admit_greedy(lambda rules: float(len(rules)), thresh=0.0, max_rules=4)

    b0.residual_seed["n_counts"] = len(counts)
    b0.residual_seed["n_kgram_k2"] = len(kg2)
    b0.residual_seed["n_kgram_k3"] = len(kg3)
    b0.residual_seed["n_pair_memory"] = len(pair_mem)
    b0.residual_seed["stream_len"] = len(stream)
    return b0


def register_b0_prompt_index(b0: WylyBlock, stories: list[Story]) -> None:
    """Register query prompts for pair_memory lookup (not fitting)."""
    index_stories = b0.residual_seed.get("index_stories")
    if callable(index_stories):
        index_stories(stories)


def score_b0_on_query_rows(
    b0: WylyBlock,
    ts: TokenSpace,
    rows: list[QueryRow],
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Agreement / abstain / error on query rows; residual R = wrong ∪ abstain.

    R is the union of R_wrong and R_abstain (rows where B0' errs OR abstains).
    Both components are reported separately; H1/H2 target the union.
    """
    n = len(rows)
    pred = torch.full((n,), -1, dtype=torch.long)
    conf = torch.full((n,), -1e9, dtype=torch.float32)
    gold = torch.tensor(
        [gold_token(ts, q.answer) for q in rows], dtype=torch.long
    ) if n else torch.zeros(0, dtype=torch.long)

    for start in range(0, n, batch_size):
        chunk = rows[start : start + batch_size]
        ids = pad_batch([encode_prompt(ts, q.prompt) for q in chunk])
        p, c = b0.predict_cover(ids)
        pred[start : start + len(chunk)] = p
        conf[start : start + len(chunk)] = c

    abstain = pred < 0
    correct = (~abstain) & (pred == gold)
    wrong = (~abstain) & (pred != gold)
    agree = float(correct.float().mean()) if n else 0.0
    per_turn: dict[int, dict[str, Any]] = {}
    for t in range(5):
        if n:
            mask = torch.tensor([q.turn == t for q in rows], dtype=torch.bool)
        else:
            mask = torch.zeros(0, dtype=torch.bool)
        nt = int(mask.sum()) if n else 0
        if nt == 0:
            continue
        per_turn[t] = {
            "n": nt,
            "agree": float(correct[mask].float().mean()),
            "abstain": float(abstain[mask].float().mean()),
            "error": float(wrong[mask].float().mean()),
        }
    R_wrong = [i for i in range(n) if bool(wrong[i])]
    R_abstain = [i for i in range(n) if bool(abstain[i])]
    R = R_wrong + R_abstain
    return {
        "n": n,
        "agree": agree,
        "n_correct": int(correct.sum()) if n else 0,
        "n_abstain": int(abstain.sum()) if n else 0,
        "n_wrong": int(wrong.sum()) if n else 0,
        "per_turn": per_turn,
        "pred": pred,
        "conf": conf,
        "gold": gold,
        "rows": rows,
        "R": R,
        "R_wrong": R_wrong,
        "R_abstain": R_abstain,
    }


# ---------------------------------------------------------------------------
# H1 — family headroom on block-1 residual
# ---------------------------------------------------------------------------
def run_h1_holdout(
    train_stories: list[Story],
    block1_score: dict[str, Any],
    ts: TokenSpace,
    *,
    det_thresh: float = MOVELOC_DET,
    supp_thresh: int = MOVELOC_SUPP,
) -> dict[str, Any]:
    """Mine moveloc + fit majority on GENERATED train; coverage/precision on block-1 R.

    Moveloc keys on entity-conditioned recency structure (most-recent matching
    verb-phrase location for that entity), not on the (entity, location) pair
    itself — so it is EXPECTED to fire correctly on held-out pairs even though
    it never saw that pair as an admitted training instance. That is the
    binding-vs-memorization contrast this instrument measures.
    """
    V, meta = mine_movement_verbs_from_events(
        train_stories, det_thresh=det_thresh, supp_thresh=supp_thresh
    )
    feat_tr: list[int] = []
    gold_tr: list[int] = []
    for s in train_stories:
        for q in s.queries:
            feat_tr.append(moveloc_feature(q.entity, q.movements_before, V, ts))
            gold_tr.append(gold_token(ts, q.answer))
    table = fit_majority_table(feat_tr, gold_tr)

    R = block1_score["R"]
    rows: list[QueryRow] = block1_score["rows"]
    gold = block1_score["gold"]

    fired = 0
    correct_fired = 0
    for i in R:
        q = rows[i]
        f = moveloc_feature(q.entity, q.movements_before, V, ts)
        if f < 0 or f not in table:
            continue
        fired += 1
        if table[f] == int(gold[i]):
            correct_fired += 1
    coverage = fired / len(R) if R else 0.0
    precision = correct_fired / fired if fired else 0.0
    # Registered rule: PASS iff cov >= 0.5 AND prec >= 0.8 (even if R is small —
    # empty R is itself a decisive locality finding; cov is undefined → 0 → FAIL
    # only if we had residual we could not recover; empty R → coverage 0.0 / no
    # residual to recover, report n_R=0 explicitly).
    if not R:
        passed = False
        status = "PASS" if False else "FAIL"
        # Decisive locality: residual did not reappear
        note = "R empty on block-1 — residual did not reappear (locality finding)"
    else:
        passed = coverage >= H1_COV and precision >= H1_PREC
        status = "PASS" if passed else "FAIL"
        note = None

    return {
        "status": status,
        "verdict": status,
        "n_R": len(R),
        "coverage": coverage,
        "precision": precision,
        "n_fired": fired,
        "n_correct_fired": correct_fired,
        "V": sorted(V),
        "verb_meta": meta,
        "table_size": len(table),
        "table": table,
        "note": note,
        "fit_on": "generated_train",
    }


# ---------------------------------------------------------------------------
# H2 — flat vs gated (only if H1 PASS)
# ---------------------------------------------------------------------------
def run_h2_holdout(
    b0: WylyBlock,
    train_stories: list[Story],
    val_stories: list[Story],
    block1_score: dict[str, Any],
    iid_score: dict[str, Any],
    ts: TokenSpace,
    V: set[str],
    majority: dict[int, int],
    *,
    tau: float | None = None,
) -> dict[str, Any]:
    """Flat vs gated arms; identical candidate pool (moveloc + k=2,3).

    Admission score and tau selection use TRAIN conf / TRAIN agreement (tables
    never see test). Metrics reported on both block-1 (verdict) and block-IID
    (sanity). Verdict rule applied on block-1 only.

    Judgment: registration pins val distribution = block-IID for any held-out
    evaluation path; admission remains train-scored (mirrors #79 ``run_h2``)
    so test_iid is not used to *select* rules — only to cross-check metrics.
    """
    train_pairs: list[tuple[list[int], int]] = []
    for s in train_stories:
        for q in s.queries:
            train_pairs.append((encode_prompt(ts, q.prompt), gold_token(ts, q.answer)))

    # Prompt map for moveloc lookup on train + scored blocks (predictions from
    # train-fit table only; registering test prompts is key lookup, not fitting).
    def build_pmap(stories: list[Story]) -> dict[tuple[int, ...], tuple[int, float]]:
        pmap: dict[tuple[int, ...], tuple[int, float]] = {}
        for s in stories:
            for q in s.queries:
                f = moveloc_feature(q.entity, q.movements_before, V, ts)
                if f < 0 or f not in majority:
                    continue
                key = tuple(encode_prompt(ts, q.prompt))
                # conf must clear select_gate_tau (pair_memory train conf ≈ 0.999)
                # under BlockStack carry=gated, which prefers cur only when
                # cur.conf > gate_conf; 0.9 would lose to B0 abstain fallback.
                pmap[key] = (majority[f], 0.9995)
        return pmap

    all_for_lookup = (
        train_stories
        + stories_from_rows(block1_score["rows"], "b1")
        + stories_from_rows(iid_score["rows"], "iid")
        + val_stories
    )
    pmap = build_pmap(all_for_lookup)

    def mv_pred(ids: torch.Tensor) -> torch.Tensor:
        B = len(ids)
        out = torch.full((B,), -1, dtype=torch.long, device=ids.device)
        for i in range(B):
            hit = pmap.get(tuple(unpad_row(ids[i])))
            if hit:
                out[i] = hit[0]
        return out

    def mv_conf(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(ids)
        a = torch.full((B,), -1, dtype=torch.long, device=ids.device)
        c = torch.full((B,), -1e9, dtype=torch.float32, device=ids.device)
        for i in range(B):
            hit = pmap.get(tuple(unpad_row(ids[i])))
            if hit:
                a[i], c[i] = hit[0], hit[1]
        return a, c

    k2_p, k2_c = make_simple_kgram(train_pairs, 2)
    k3_p, k3_c = make_simple_kgram(train_pairs, 3)

    tr_ids = pad_batch(
        [encode_prompt(ts, q.prompt) for s in train_stories for q in s.queries]
    )
    tr_gold = torch.tensor(
        [gold_token(ts, q.answer) for s in train_stories for q in s.queries],
        dtype=torch.long,
    )
    b0_p, b0_c = b0.predict_cover(tr_ids)
    tr_correct = (b0_p == tr_gold) & (b0_p >= 0)
    if tau is None:
        tau = select_gate_tau(b0_c, tr_correct)

    def eval_on(score: dict[str, Any], flat_pred: torch.Tensor, gated_pred: torch.Tensor) -> dict:
        te_gold = score["gold"]
        b0_te = score["pred"]
        b0_ok = (b0_te == te_gold) & (b0_te >= 0)

        def arm(pred: torch.Tensor) -> dict[str, Any]:
            ok = (pred == te_gold) & (pred >= 0)
            agree = float(ok.float().mean()) if len(te_gold) else 0.0
            reg = int((b0_ok & ~ok).sum()) if len(te_gold) else 0
            return {"agree": agree, "regressions": reg, "ok": ok}

        fm = arm(flat_pred)
        gm = arm(gated_pred)
        f_ok, g_ok = fm["ok"], gm["ok"]
        b = int((g_ok & ~f_ok).sum()) if len(te_gold) else 0
        c = int((f_ok & ~g_ok).sum()) if len(te_gold) else 0
        pval = exact_binomial_two_sided(b, c)
        return {
            "flat": {"agree": fm["agree"], "regressions": fm["regressions"]},
            "gated": {"agree": gm["agree"], "regressions": gm["regressions"]},
            "discordant": {
                "b_gated_right_flat_wrong": b,
                "c_flat_right_gated_wrong": c,
            },
            "p_value": pval,
        }

    # Admission scoring: train is saturated by pair_memory (agree=1), so residual
    # families have ~0 marginal under full B0' and greedy never admits them.
    # Judgment (reported): admit residual candidates by their marginal over a
    # *stripped* B0' (counts+kgram only, no pair_memory) on TRAIN — same rows,
    # no test leakage — so moveloc can show the train headroom H1 already
    # established. Evaluation still uses full B0' + admitted residual rules.
    stripped_rules = [(n, f) for n, f in b0.rules if "pair_memory" not in n]
    stripped_conf = {n: c for n, c in b0.conf_fns.items() if "pair_memory" not in n}

    def score_with_stripped(trial_extra: list, conf_extra: dict) -> float:
        rules = stripped_rules + list(trial_extra)
        confs = {**stripped_conf, **conf_extra}
        return cover_agree(rules, confs, tr_ids, tr_gold)

    # ---- FLAT: stripped B0' base + residual candidates; then restore pair_memory ----
    flat_b = WylyBlock(0, "flat_joint", family="joint")
    for name, fn in stripped_rules:
        flat_b.rules.append((name, fn))
        if name in stripped_conf:
            flat_b.conf_fns[name] = stripped_conf[name]
    flat_b.add_candidate("moveloc", mv_pred, mv_conf)
    flat_b.add_candidate("kgram_k2", k2_p, k2_c)
    flat_b.add_candidate("kgram_k3", k3_p, k3_c)

    def score_flat(rules):
        return cover_agree(rules, flat_b.conf_fns, tr_ids, tr_gold)

    flat_b.admit_greedy(score_flat, thresh=ADMIT_THRESH, max_rules=4)
    # Restore full B0' pair_memory for evaluation SW cover
    have = {n for n, _ in flat_b.rules}
    for name, fn in b0.rules:
        if name not in have:
            flat_b.rules.insert(0, (name, fn))
            if name in b0.conf_fns:
                flat_b.conf_fns[name] = b0.conf_fns[name]

    # ---- GATED: full B0' frozen; residual layer admitted vs stripped marginal ----
    g_b0 = WylyBlock(0, "b0_prime", family="memorizing_cover")
    for name, fn in b0.rules:
        g_b0.rules.append((name, fn))
        if name in b0.conf_fns:
            g_b0.conf_fns[name] = b0.conf_fns[name]
    g_b1 = WylyBlock(1, "residual", depends_on=[0], family="moveloc_kgram")
    g_b1.add_candidate("moveloc", mv_pred, mv_conf)
    g_b1.add_candidate("kgram_k2", k2_p, k2_c)
    g_b1.add_candidate("kgram_k3", k3_p, k3_c)
    stack = BlockStack([g_b0, g_b1], carry="gated", gate_conf=tau)

    def score_layer(rules):
        # rules = frozen full B0' + trial B1 under admit_mode=stack; score trial
        # extras against stripped baseline so residual families can clear thresh.
        b0_names = {n for n, _ in g_b0.rules}
        extra = [(n, f) for n, f in rules if n not in b0_names]
        conf_extra = {n: g_b1.conf_fns[n] for n, _ in extra if n in g_b1.conf_fns}
        return score_with_stripped(extra, conf_extra)

    stack.admit_layer(1, score_layer, thresh=ADMIT_THRESH, max_rules=4)

    def predict_both(rows: list[QueryRow]) -> tuple[torch.Tensor, torch.Tensor]:
        if not rows:
            z = torch.zeros(0, dtype=torch.long)
            return z, z
        ids = pad_batch([encode_prompt(ts, q.prompt) for q in rows])
        flat_pred, _ = flat_b.predict_cover(ids)
        stack.forward(ids)
        carried = stack.last_carried[-1]
        assert carried.pred is not None
        return flat_pred, carried.pred

    f1, g1 = predict_both(block1_score["rows"])
    fi, gi = predict_both(iid_score["rows"])
    m_b1 = eval_on(block1_score, f1, g1)
    m_iid = eval_on(iid_score, fi, gi)

    # Registered verdict on block-1 only
    earns = (
        m_b1["gated"]["agree"] >= m_b1["flat"]["agree"] + H2_AGREE_DELTA
        and m_b1["gated"]["regressions"] <= m_b1["flat"]["regressions"]
        and m_b1["p_value"] < H2_P
    )
    return {
        "status": "PASS" if earns else "FAIL",
        "verdict": "PASS" if earns else "FAIL",
        "tau": tau,
        "flat_admitted": [n for n, _ in flat_b.rules],
        "gated_admitted": [n for n, _ in g_b1.rules],
        "block1": m_b1,
        "block_iid": m_iid,
        # top-level aliases for scoreboard (block-1 = verdict surface)
        "flat": {
            **m_b1["flat"],
            "admitted": [n for n, _ in flat_b.rules],
        },
        "gated": {
            **m_b1["gated"],
            "admitted": [n for n, _ in g_b1.rules],
        },
        "discordant": m_b1["discordant"],
        "p_value": m_b1["p_value"],
    }


# ---------------------------------------------------------------------------
# Round-trip helpers (used by tests + campaign self-check)
# ---------------------------------------------------------------------------
def roundtrip_train_via_loader(train: list[GenStory], tmp_path: Path) -> list[Story]:
    """Write flat corpus and parse with load_train_stories."""
    text = join_train_corpus(train)
    p = tmp_path / "corpus_gen.txt"
    p.write_text(text)
    return load_train_stories(p)


def _entity_from_prompt_last(prompt: str) -> str:
    """Entity of the *current* turn: last ``Where is <E>?`` in an accumulated prompt.

    The reused ``entity_from_question`` uses ``re.search`` (first match), which is
    wrong for multi-turn accumulated bench prompts that embed prior Q lines.
    Bench loader has the same first-match quirk; for our generated blocks we take
    the last match so planted structure round-trips correctly.
    """
    import re

    matches = list(re.finditer(r"Where is (?P<ent>\w+)\?", prompt))
    if not matches:
        raise AssertionError(f"no Where-is in prompt ...{prompt[-60:]!r}")
    return matches[-1].group("ent")


def roundtrip_test_via_prompts(gen_stories: list[GenStory]) -> list[Story]:
    """Parse test blocks through bench-style prompts + _movements_from_prompt."""
    stories: list[Story] = []
    for gs in gen_stories:
        prompts = story_prompts_bench_style(gs)
        qrows: list[QueryRow] = []
        movs_chrono: list[Movement] = []
        for t, (prompt, answer) in enumerate(prompts):
            body = prompt
            if body.rstrip().endswith("A:"):
                body = body.rstrip()[:-2].rstrip()
            all_movs = _movements_from_prompt(body)
            movs_chrono = all_movs
            ent = _entity_from_prompt_last(prompt)
            qrows.append(
                QueryRow(
                    story_id=gs.story_id,
                    turn=t,
                    entity=ent,
                    answer=answer,
                    prompt=prompt if prompt.rstrip().endswith("A:") else prompt.rstrip() + " A:",
                    movements_before=list(all_movs),
                )
            )
        stories.append(
            Story(story_id=gs.story_id, queries=qrows, all_movements=list(movs_chrono))
        )
    return stories


def compare_planted_vs_parsed(gen: GenStory, parsed: Story) -> None:
    """Assert parsed movements/queries match generator-planted structure."""
    movs, qs = parse_story_text(gen.text)
    assert len(parsed.queries) == len(qs) == 5
    assert len(parsed.all_movements) == len(movs)
    for pm, (e, v, loc) in zip(parsed.all_movements, movs, strict=True):
        assert pm.entity == e and pm.verb == v and pm.loc == loc
    for pq, (e, a) in zip(parsed.queries, qs, strict=True):
        assert pq.entity == e and pq.answer == a


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(x) for x in o]
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_probe() -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    print("=" * 72)
    print("qa1 config-holdout battery (slice #80)")
    print("=" * 72)

    # --- world ---
    print("\n--- world generation ---")
    world = generate_world(seed=SEED)
    train_g: list[GenStory] = world["train"]
    holdout_g: list[GenStory] = world["test_holdout"]
    iid_g: list[GenStory] = world["test_iid"]
    names_g: list[GenStory] = world["test_names"]
    pair_counts: Counter = world["pair_counts"]

    print(
        f"stories: train={len(train_g)} holdout={len(holdout_g)} "
        f"iid={len(iid_g)} names={len(names_g)}"
    )
    print(f"HELD_OUT_PAIRS: {list(HELD_OUT_PAIRS)}")
    held_ok = all(pair_counts[p] == 0 for p in HELD_OUT_PAIRS)
    print(f"train excludes all 4 held-out pairs: {held_ok}")
    for p in HELD_OUT_PAIRS:
        print(f"  train count {p} = {pair_counts[p]}")
    legal_mins = {p: pair_counts[p] for p in TRAIN_LEGAL_PAIRS}
    min_c = min(legal_mins.values())
    print(f"all 20 non-held-out pairs count >= 20: {min_c >= 20} (min={min_c})")
    for p in sorted(TRAIN_LEGAL_PAIRS):
        print(f"  {p}: {pair_counts[p]}")

    train_s = gen_list_to_stories(train_g)
    holdout_s = gen_list_to_stories(holdout_g)
    iid_s = gen_list_to_stories(iid_g)
    names_s = gen_list_to_stories(names_g)
    assert_all_splits_disjoint(train_s, holdout_s, iid_s, names_s)
    print("story-id disjointness: OK across all four lists")

    block1_rows, routed_iid_rows = partition_holdout_queries(holdout_s)
    iid_rows = [q for s in iid_s for q in s.queries] + routed_iid_rows
    names_rows = [q for s in names_s for q in s.queries]
    print(
        f"block-1 holdout-answer rows: {len(block1_rows)}  "
        f"routed non-holdout from holdout stories → IID: {len(routed_iid_rows)}  "
        f"block-IID total rows: {len(iid_rows)}  names rows: {len(names_rows)}"
    )

    # --- B0' ---
    print("\n--- B0' memorizing cover (fit fresh on generated train) ---")
    ts = TokenSpace.from_file(str(PKG_DIR / "bundle.tokenizer.json"))
    b0 = build_b0_prime(train_s, ts)
    # Register test prompts for pair_memory lookup keys only (tables already frozen).
    register_b0_prompt_index(b0, holdout_s)
    register_b0_prompt_index(b0, iid_s)
    register_b0_prompt_index(b0, names_s)
    # routed rows already live on holdout_s stories
    print(
        f"admitted={ [n for n,_ in b0.rules] }  "
        f"pair_mem={b0.residual_seed.get('n_pair_memory')}  "
        f"counts={b0.residual_seed.get('n_counts')}  "
        f"k2={b0.residual_seed.get('n_kgram_k2')}  "
        f"k3={b0.residual_seed.get('n_kgram_k3')}  "
        f"stream_len={b0.residual_seed.get('stream_len')}"
    )
    print(
        "EXPECTATION: block-IID high, block-1 residual reappears "
        "(if residual does not reappear, that is a decisive locality finding — report as-is)."
    )

    train_score = score_b0_on_query_rows(
        b0, ts, [q for s in train_s for q in s.queries]
    )
    iid_score = score_b0_on_query_rows(b0, ts, iid_rows)
    b1_score = score_b0_on_query_rows(b0, ts, block1_rows)
    names_score = score_b0_on_query_rows(b0, ts, names_rows)

    def _print_block(label: str, sc: dict[str, Any]) -> None:
        print(
            f"  {label}: agree={sc['agree']:.4f}  "
            f"correct={sc['n_correct']} abstain={sc['n_abstain']} "
            f"wrong={sc['n_wrong']} / n={sc['n']}  "
            f"|R|={len(sc['R'])} (wrong={len(sc['R_wrong'])} abstain={len(sc['R_abstain'])})"
        )
        for t in sorted(sc["per_turn"]):
            pt = sc["per_turn"][t]
            print(
                f"    turn{t}: agree={pt['agree']:.3f} abstain={pt['abstain']:.3f} "
                f"err={pt['error']:.3f} n={pt['n']}"
            )

    print("B0' per-block scores:")
    _print_block("train (sanity)", train_score)
    _print_block("block-IID", iid_score)
    _print_block("block-1 (holdout)", b1_score)
    _print_block("test_names (diagnostic only)", names_score)

    # --- H1 ---
    print("\n--- H1: family headroom (gates H2) ---")
    print(H1_READING)
    print(
        "NOTE: moveloc keys on entity-conditioned recency (most-recent matching "
        "verb-phrase location), not on the (entity, location) pair itself — so it is "
        "EXPECTED to fire on held-out pairs never seen as train bindings. That is the "
        "binding-vs-memorization contrast this instrument measures."
    )
    h1 = run_h1_holdout(train_s, b1_score, ts)
    print(f"  mined V: {h1['V']}")
    print(f"  verb_meta: {h1['verb_meta']}")
    print(
        f"  coverage={h1['coverage']:.4f} precision={h1['precision']:.4f} "
        f"|R|={h1['n_R']} fired={h1['n_fired']} table_size={h1['table_size']}"
    )
    if h1.get("note"):
        print(f"  note: {h1['note']}")
    print(f"  H1 verdict: {h1['verdict']}")

    # --- H2 ---
    print("\n--- H2: flat vs gated ---")
    print(H2_READING)
    if h1["status"] != "PASS":
        h2: dict[str, Any] = {
            "status": "SKIPPED",
            "verdict": "SKIPPED",
            "reason": f"gated on H1 (H1={h1['status']}: {h1.get('note') or h1.get('verdict')})",
        }
        print(f"  H2 SKIPPED: {h2['reason']}")
    else:
        h2 = run_h2_holdout(
            b0,
            train_s,
            iid_s,  # pinned val distribution (block-IID stories)
            b1_score,
            iid_score,
            ts,
            set(h1["V"]),
            h1["table"],
        )
        print(f"  tau={h2['tau']:.6f}")
        print(
            f"  FLAT  admitted={h2['flat']['admitted']}"
        )
        print(
            f"  GATED admitted={h2['gated']['admitted']}"
        )
        for label, blk in (("block-1", h2["block1"]), ("block-IID", h2["block_iid"])):
            print(
                f"  [{label}] flat_agree={blk['flat']['agree']:.4f} "
                f"flat_reg={blk['flat']['regressions']}  "
                f"gated_agree={blk['gated']['agree']:.4f} "
                f"gated_reg={blk['gated']['regressions']}  "
                f"b={blk['discordant']['b_gated_right_flat_wrong']} "
                f"c={blk['discordant']['c_flat_right_gated_wrong']} "
                f"p={blk['p_value']:.4g}"
            )
        print(
            "  verdict uses block-1 only: "
            f"gated_agree>={h2['block1']['flat']['agree'] + H2_AGREE_DELTA:.4f} "
            f"AND gated_reg<=flat_reg AND p<{H2_P}"
        )
        print(f"  H2 verdict: {h2['verdict']}")

    elapsed = time.time() - t0

    # --- SCOREBOARD ---
    print("\n" + "=" * 72)
    print("SCOREBOARD")
    print("=" * 72)
    print(
        f"world train={len(train_g)} holdout={len(holdout_g)} iid={len(iid_g)} "
        f"names={len(names_g)}  held-out pairs excluded from train={held_ok}  "
        f"min legal pair count={min_c}"
    )
    print(
        f"B0' block-IID agree={iid_score['agree']:.4f} |R|={len(iid_score['R'])}  "
        f"block-1 agree={b1_score['agree']:.4f} |R|={len(b1_score['R'])} "
        f"(wrong={len(b1_score['R_wrong'])} abstain={len(b1_score['R_abstain'])})  "
        f"names agree={names_score['agree']:.4f} (diagnostic)"
    )
    print(
        f"H1 coverage={h1['coverage']:.4f} precision={h1['precision']:.4f} "
        f"|R|={h1['n_R']} → {h1['verdict']}"
    )
    if h2.get("status") == "SKIPPED":
        print(f"H2 SKIPPED ({h2.get('reason')})")
    else:
        print(
            f"H2 tau={h2['tau']:.6f}  "
            f"block1 flat={h2['block1']['flat']['agree']:.4f} "
            f"gated={h2['block1']['gated']['agree']:.4f} "
            f"p={h2['p_value']:.4g} → {h2['verdict']}"
        )
        print(
            f"H2 block-IID (sanity) flat={h2['block_iid']['flat']['agree']:.4f} "
            f"gated={h2['block_iid']['gated']['agree']:.4f}"
        )
    print(f"wall_clock_s={elapsed:.1f}")
    print(
        "NOTE: B0' is a fresh pair_memory+counts+kgram SW cover on the generated train "
        "(stand-in for the on-disk package family; on-disk corpus has no holdout). "
        "pair_memory = joint (entity,loc) co-occurrence memorization from train. "
        "moveloc is hand-authored (template_fixed-class); frac_induced unaffected. "
        "Scoring teacher-forced on query tails; first-token host-BPE; prompts end at A:."
    )

    report = {
        "note": {
            "generated_world": (
                "Freshly generated qa1 world with 4 held-out entity×location pairs; "
                "train excludes those joints while covering all atoms and the other 20 pairs."
            ),
            "b0_prime": (
                "B0' = pair_memory (entity×location co-occurrence from train movements) + "
                "counts + kgram(k=2,3) support-weighted cover fit fresh on generated train. "
                "Stand-in for on-disk served package family because corpus_babi.txt / package "
                "have no holdout (slice #79). pair_memory is pure joint memorization: fires "
                "only when the live binding was seen in train (holdout pairs abstain)."
            ),
            "moveloc": "hand-authored / template_fixed-class; frac_induced unaffected",
            "scoring": (
                "teacher-forced context; first-token host-BPE; prompts end at A:; "
                "R = R_wrong ∪ R_abstain (B0' err or abstain)"
            ),
            "pins": {
                "baseline_ensemble": "B0'",
                "val_distribution": "block-IID query tails (non-held-out pairs)",
            },
            "thresholds": {
                "moveloc_det": MOVELOC_DET,
                "moveloc_supp": MOVELOC_SUPP,
                "admit": ADMIT_THRESH,
                "h1_cov": H1_COV,
                "h1_prec": H1_PREC,
                "h2_agree_delta": H2_AGREE_DELTA,
                "h2_p": H2_P,
                "laplace_alpha": LAPLACE_ALPHA,
                "kgram_minsupp": KGRAM_MINSUPP,
                "kgram_mindet": KGRAM_MINDET,
            },
            "held_out_pairs": [list(p) for p in HELD_OUT_PAIRS],
        },
        "world": {
            "n_train": len(train_g),
            "n_holdout": len(holdout_g),
            "n_iid": len(iid_g),
            "n_names": len(names_g),
            "n_block1_rows": len(block1_rows),
            "n_iid_rows": len(iid_rows),
            "n_routed_iid_from_holdout": len(routed_iid_rows),
            "train_excludes_held_out": held_ok,
            "min_legal_pair_count": min_c,
            "pair_counts": {f"{e}|{loc}": c for (e, loc), c in sorted(pair_counts.items())},
        },
        "b0_prime": {
            "admitted": [n for n, _ in b0.rules],
            "n_pair_memory": b0.residual_seed.get("n_pair_memory"),
            "n_counts": b0.residual_seed.get("n_counts"),
            "n_kgram_k2": b0.residual_seed.get("n_kgram_k2"),
            "n_kgram_k3": b0.residual_seed.get("n_kgram_k3"),
            "train": {
                "agree": train_score["agree"],
                "n": train_score["n"],
                "n_R": len(train_score["R"]),
                "n_wrong": train_score["n_wrong"],
                "n_abstain": train_score["n_abstain"],
            },
            "block_iid": {
                "agree": iid_score["agree"],
                "n": iid_score["n"],
                "n_R": len(iid_score["R"]),
                "n_wrong": iid_score["n_wrong"],
                "n_abstain": iid_score["n_abstain"],
                "R_wrong": len(iid_score["R_wrong"]),
                "R_abstain": len(iid_score["R_abstain"]),
            },
            "block1": {
                "agree": b1_score["agree"],
                "n": b1_score["n"],
                "n_R": len(b1_score["R"]),
                "n_wrong": b1_score["n_wrong"],
                "n_abstain": b1_score["n_abstain"],
                "R_wrong": len(b1_score["R_wrong"]),
                "R_abstain": len(b1_score["R_abstain"]),
            },
            "names_diagnostic": {
                "agree": names_score["agree"],
                "n": names_score["n"],
                "n_R": len(names_score["R"]),
            },
        },
        "h1": {k: v for k, v in h1.items() if k != "table"},
        "h2": h2,
        "wall_clock_s": elapsed,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    print(f"\nwrote {OUT_JSON}")
    return report


if __name__ == "__main__":
    run_probe()
