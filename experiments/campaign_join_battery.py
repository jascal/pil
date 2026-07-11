"""Synthetic join battery: can residual admission recover planted joins?

CFQ probes #75/#76 showed join structure is not recoverable by majority /
lexical / mined-type predictors on real CFQ, but CFQ joins are underdetermined
— so those negatives cannot say whether the induction machinery *could* induce
joins when they are genuinely there. This campaign builds planted, provably-
determinate synthetic topologies and admits intrinsic proposers through the
existing ``ResidualFamily.admit`` held-out-marginal judge with a ``join_f1``
gate.

Pre-registered verdict rule (decided BEFORE looking at results):

  "machinery CAN induce planted joins" iff >=1 arm in its MATCHED regime
  (S: majority/typed/SIG; L: lexical/SIGW) reaches test join_f1 >= 0.9 IID
  AND >= 0.7 compositional, AND that arm's N-control is clean (n_admitted
  <= 1 AND N test gain <= 0.02 over empty maps)

All five arms are template_fixed-class mined content — NO battery result
changes ``frac_induced`` (the generality-metric accounting stays untouched by
this slice; schema-/train-mined join atoms must not count toward frac_induced).

Run (from repo root):
  .venv/bin/python experiments/campaign_join_battery.py
Env:
  JB_SEED=0
  JB_REGIMES=S,L,N   (comma-split; JB_REGIMES=S runs only regime S)

The ``typed`` arm is structurally equivalent to ``majority`` in regime S:
self-certification guarantees zero within-key ambiguity, so
``assemble_joins_by_type``'s output equals the per-key majority signature for
every train-seen key, and both arms fire only on exact JOINKEY match (never on
held-out compositional keys). This redundancy is intentional — ``typed`` is
kept as a designed cross-check/verification arm confirming the type-driven
assembler reproduces the majority-vote ceiling under regime S's guarantees,
NOT as an independent compositional-generalization test. ``SIG``
(per-signature incidence-subset firing) is the arm that actually tests
compositional transfer in this battery.
"""
from __future__ import annotations

import ast
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.campaign_cfq_typed_join import (  # noqa: E402
    assemble_joins_by_type,
    mine_slot_tables,
)
from pil.cfq_edges import join_f1, parse_sparql_joins  # noqa: E402
from pil.join_battery import (  # noqa: E402
    BatteryExample,
    BatterySplit,
    generate_regime,
    world_counts,
)
from pil.residual_template import (  # noqa: E402
    DomainAtoms,
    MapDict,
    ResidualCandidate,
    ResidualFamily,
)

ARM_CAP = 300
DOMAIN = "join_battery"

# Matched arms per regime for the pre-registered verdict
MATCHED_ARMS: dict[str, tuple[str, ...]] = {
    "S": ("majority", "typed", "SIG"),
    "L": ("lexical", "SIGW"),
}


def log(m: str = "") -> None:
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Signature serialization (round-trip)
# ---------------------------------------------------------------------------


def canon_sig(sig: tuple) -> str:
    """Repr-based canonical string for a join signature (or any nested tuple)."""
    return repr(sig)


def parse_sig(s: str) -> tuple:
    """Inverse of ``canon_sig`` via ``ast.literal_eval``."""
    val = ast.literal_eval(s)
    if not isinstance(val, tuple):
        raise TypeError(f"expected tuple signature, got {type(val).__name__}")
    return val


# ---------------------------------------------------------------------------
# Gold helpers
# ---------------------------------------------------------------------------


def gold_sigs(ex: BatteryExample) -> list[tuple]:
    return list(parse_sparql_joins(ex.sparql).signatures)


def gold_serialized(ex: BatteryExample) -> list[str]:
    return [canon_sig(s) for s in gold_sigs(ex)]


def _multiset_key(sigs: list[tuple]) -> tuple[str, ...]:
    """Deterministic hashable key for a signature multiset (sorted canon strings)."""
    return tuple(sorted(canon_sig(s) for s in sigs))


def _majority_tgt(rows: list[BatteryExample]) -> tuple[str, ...]:
    """Majority gold signature multiset; ties → lex-smallest serialized multiset."""
    votes: Counter[tuple[str, ...]] = Counter()
    for ex in rows:
        votes[_multiset_key(gold_sigs(ex))] += 1
    max_c = max(votes.values())
    winners = sorted(k for k, c in votes.items() if c == max_c)
    return winners[0]


def _cap_by_support(
    items: list[tuple[int, tuple[str, ...], ResidualCandidate]],
) -> list[ResidualCandidate]:
    """items = (support, src, cand); keep top ARM_CAP by -support, then lex src."""
    items_sorted = sorted(items, key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in items_sorted[:ARM_CAP]]


# ---------------------------------------------------------------------------
# Firing rules (shared by mining docs + expand)
# ---------------------------------------------------------------------------


def _sig_pred_subset(sig: tuple, key: tuple[str, ...]) -> bool:
    """True iff the signature's incidence-predicate multiset ⊆ Counter(key)."""
    incs = sig[1] if len(sig) > 1 else ()
    sig_ctr: Counter[str] = Counter(p for p, _slot in incs)
    row_ctr = Counter(key)
    return all(row_ctr[p] >= sig_ctr[p] for p in sig_ctr)


def atom_fires(
    src: tuple[str, ...],
    question_tokens: set[str],
    key: tuple[str, ...],
) -> bool:
    """Whether an admitted atom with this ``src`` fires on (tokens, key)."""
    kind = src[0]
    if kind == "JOINKEY":
        return "|".join(key) == src[1]
    if kind == "WORD":
        return src[1] in question_tokens
    if kind == "SIG":
        return _sig_pred_subset(parse_sig(src[1]), key)
    if kind == "SIGW":
        return (
            src[1] in question_tokens
            and _sig_pred_subset(parse_sig(src[2]), key)
        )
    return False


# ---------------------------------------------------------------------------
# Expander (load-bearing)
# ---------------------------------------------------------------------------


def expand(
    maps: MapDict,
    question_tokens: set[str],
    key: tuple[str, ...],
) -> list[str]:
    """Combine firing atoms' tgt multisets with multiset-MAX (Counter |=).

    LEAK GUARD: this function receives ONLY (maps, question_tokens, key).
    No gold signatures, no SPARQL, no val/test labels — structurally impossible
    to consult gold because no such parameter exists.
    """
    total: Counter[str] = Counter()
    # Stable iteration order for determinism
    for src in sorted(maps.keys()):
        tgt = maps[src]
        if atom_fires(src, question_tokens, key):
            total |= Counter(tgt)
    return sorted(total.elements())


# ---------------------------------------------------------------------------
# Arm miners (TRAIN only)
# ---------------------------------------------------------------------------


def mine_majority(train: list[BatteryExample]) -> list[ResidualCandidate]:
    by_key: dict[tuple[str, ...], list[BatteryExample]] = defaultdict(list)
    for ex in train:
        by_key[ex.key].append(ex)
    items: list[tuple[int, tuple[str, ...], ResidualCandidate]] = []
    for key, rows in by_key.items():
        src = ("JOINKEY", "|".join(key))
        tgt = _majority_tgt(rows)
        cand = ResidualCandidate(
            src=src,
            tgt=tgt,
            template_id="join_key_majority",
            domain=DOMAIN,
            pattern_agnostic=True,
            meta={"support": len(rows)},
        )
        items.append((len(rows), src, cand))
    return _cap_by_support(items)


def mine_typed(train: list[BatteryExample]) -> list[ResidualCandidate]:
    train_pairs = [(ex.question, ex.sparql) for ex in train]
    tables = mine_slot_tables(train_pairs)
    by_key: dict[tuple[str, ...], int] = Counter(ex.key for ex in train)
    items: list[tuple[int, tuple[str, ...], ResidualCandidate]] = []
    for key, support in by_key.items():
        sigs = assemble_joins_by_type(list(key), tables)
        if not sigs:
            continue
        src = ("JOINKEY", "|".join(key))
        tgt = tuple(canon_sig(s) for s in sigs)
        cand = ResidualCandidate(
            src=src,
            tgt=tgt,
            template_id="join_key_typed",
            domain=DOMAIN,
            pattern_agnostic=True,
            meta={"support": support},
        )
        items.append((support, src, cand))
    return _cap_by_support(items)


def mine_lexical(train: list[BatteryExample]) -> list[ResidualCandidate]:
    word_rows: dict[str, list[BatteryExample]] = defaultdict(list)
    for ex in train:
        for w in set(ex.question.split()):
            word_rows[w].append(ex)
    items: list[tuple[int, tuple[str, ...], ResidualCandidate]] = []
    for w, rows in word_rows.items():
        if len(rows) < 2:
            continue
        src = ("WORD", w)
        tgt = _majority_tgt(rows)
        cand = ResidualCandidate(
            src=src,
            tgt=tgt,
            template_id="join_word",
            domain=DOMAIN,
            pattern_agnostic=True,
            meta={"support": len(rows)},
        )
        items.append((len(rows), src, cand))
    return _cap_by_support(items)


def mine_sig(train: list[BatteryExample]) -> list[ResidualCandidate]:
    sig_support: Counter[tuple] = Counter()
    for ex in train:
        for s in gold_sigs(ex):
            sig_support[s] += 1
    items: list[tuple[int, tuple[str, ...], ResidualCandidate]] = []
    for sig, support in sig_support.items():
        cs = canon_sig(sig)
        src = ("SIG", cs)
        tgt = (cs,)
        cand = ResidualCandidate(
            src=src,
            tgt=tgt,
            template_id="join_sig",
            domain=DOMAIN,
            pattern_agnostic=True,
            meta={"support": support},
        )
        items.append((support, src, cand))
    return _cap_by_support(items)


def mine_sigw(train: list[BatteryExample]) -> list[ResidualCandidate]:
    pair_support: Counter[tuple[str, tuple]] = Counter()
    for ex in train:
        words = set(ex.question.split())
        for s in gold_sigs(ex):
            for w in words:
                pair_support[(w, s)] += 1
    items: list[tuple[int, tuple[str, ...], ResidualCandidate]] = []
    for (w, sig), support in pair_support.items():
        if support < 2:
            continue
        cs = canon_sig(sig)
        src = ("SIGW", w, cs)
        tgt = (cs,)
        cand = ResidualCandidate(
            src=src,
            tgt=tgt,
            template_id="join_sig_word",
            domain=DOMAIN,
            pattern_agnostic=True,
            meta={"support": support},
        )
        items.append((support, src, cand))
    return _cap_by_support(items)


ARM_MINERS = {
    "majority": mine_majority,
    "typed": mine_typed,
    "lexical": mine_lexical,
    "SIG": mine_sig,
    "SIGW": mine_sigw,
}


# ---------------------------------------------------------------------------
# Precomputed val_score + scoring
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def make_val_score(
    val: list[BatteryExample],
    candidates: list[ResidualCandidate],
    gold_override: list[list[str]] | None = None,
):
    """Build a fast val_score closure over precomputed fire/tgt/gold tables.

    PERFORMANCE: parse_sparql_joins and firing derivation run ONCE here.
    Inside admit's O(max_rules * n_candidates) score_fn calls we only do
    Counter arithmetic over precomputed booleans and gold lists.

    ``gold_override`` (optional): when given, use these per-row serialized gold
    lists (same order as ``val``) in place of ``gold_serialized(ex)``. Used by
    the permutation addendum; when omitted (default), behavior is unchanged.
    """
    # Per-row: tokens, key, gold (serialized)
    row_tokens: list[set[str]] = []
    row_keys: list[tuple[str, ...]] = []
    row_gold: list[list[str]] = []
    for i, ex in enumerate(val):
        row_tokens.append(set(ex.question.split()))
        row_keys.append(ex.key)
        if gold_override is not None:
            row_gold.append(gold_override[i])
        else:
            row_gold.append(gold_serialized(ex))

    # Per-candidate: src, tgt Counter, fire mask over val rows
    cand_srcs = [c.src for c in candidates]
    cand_tgt_ctrs = [Counter(c.tgt) for c in candidates]
    # fire[c_idx][r_idx]
    fire: list[list[bool]] = []
    for c in candidates:
        mask = [
            atom_fires(c.src, row_tokens[r], row_keys[r])
            for r in range(len(val))
        ]
        fire.append(mask)

    src_to_idx = {src: i for i, src in enumerate(cand_srcs)}

    def val_score(maps: MapDict) -> float:
        scores: list[float] = []
        for r in range(len(val)):
            total: Counter[str] = Counter()
            for src, tgt in maps.items():
                idx = src_to_idx.get(src)
                if idx is None:
                    # Fallback (should not happen for our admit path)
                    if atom_fires(src, row_tokens[r], row_keys[r]):
                        total |= Counter(tgt)
                    continue
                if fire[idx][r]:
                    # Use maps' tgt (admits list(cand.tgt)); prefer precomputed
                    # Counter when it matches candidate tgt for speed.
                    total |= cand_tgt_ctrs[idx]
            pred = sorted(total.elements())
            scores.append(join_f1(pred, row_gold[r]))
        return _mean(scores)

    return val_score


def score_block(
    maps: MapDict,
    block: list[BatteryExample],
    gold_override: list[list[str]] | None = None,
) -> float:
    """Mean join_f1 of ``maps`` over ``block``.

    ``gold_override`` (optional): per-row serialized gold lists in the same
    order as ``block``. When omitted, uses each row's real gold (unchanged
    default path). Used by the permutation addendum for permuted test_iid.
    """
    if not block:
        return float("nan")
    scores: list[float] = []
    for i, ex in enumerate(block):
        tokens = set(ex.question.split())
        pred = expand(maps, tokens, ex.key)
        gold = (
            gold_override[i]
            if gold_override is not None
            else gold_serialized(ex)
        )
        scores.append(join_f1(pred, gold))
    return _mean(scores)


def empty_baseline(
    block: list[BatteryExample],
    gold_override: list[list[str]] | None = None,
) -> float:
    """Mean join_f1 of the empty map over ``block`` (or overridden gold)."""
    if not block:
        return float("nan")
    if gold_override is not None:
        return _mean([join_f1([], g) for g in gold_override])
    return _mean([join_f1([], gold_serialized(ex)) for ex in block])


# ---------------------------------------------------------------------------
# Permutation addendum helpers
# ---------------------------------------------------------------------------
# A control answering a DIFFERENT question than the regime-N columns —
# "does the judge admit structure that is not there?" — it is NOT a
# corrected N-control.


def permute_gold_lists(
    block: list[BatteryExample],
    seed: int,
) -> list[list[str]]:
    """Shuffle per-row serialized gold lists; features (tokens/key) stay put.

    Returns a new list of gold lists in the same row order as ``block``, but
    with gold multisets reassigned across rows via ``random.Random(seed)``.
    """
    golds = [gold_serialized(ex) for ex in block]
    rng = random.Random(seed)
    rng.shuffle(golds)
    return golds


# ---------------------------------------------------------------------------
# N inventory-ceiling predictor (oracle upper bound on exact-key lookup)
# ---------------------------------------------------------------------------


def build_inventory_ceiling(
    train: list[BatteryExample],
) -> dict[tuple[str, ...], Counter[str]]:
    """Per train key: multiset-MAX (Counter |=) of gold-signature Counters
    across all train rows sharing that key."""
    inv: dict[tuple[str, ...], Counter[str]] = {}
    for ex in train:
        c = Counter(gold_serialized(ex))
        if ex.key not in inv:
            inv[ex.key] = Counter()
        inv[ex.key] |= c
    return inv


def predict_inventory_ceiling(
    inv: dict[tuple[str, ...], Counter[str]],
    key: tuple[str, ...],
) -> list[str]:
    """Exact-key lookup into inv; [] if key unseen."""
    c = inv.get(key, Counter())
    return sorted(c.elements())


def score_inventory_ceiling(
    train: list[BatteryExample],
    test_iid: list[BatteryExample],
) -> float:
    """Mean join_f1 of the inventory-ceiling predictor on test_iid."""
    inv = build_inventory_ceiling(train)
    scores = [
        join_f1(predict_inventory_ceiling(inv, ex.key), gold_serialized(ex))
        for ex in test_iid
    ]
    return _mean(scores)


# ---------------------------------------------------------------------------
# Comp expressibility bound (S, L only)
# ---------------------------------------------------------------------------


def train_signature_set(train: list[BatteryExample]) -> set[str]:
    """SET of ``canon_sig(s)`` for every gold signature s in train.

    Same derivation as ``world_counts``'s distinct-sig cardinality, but keeps
    the set rather than only its size.
    """
    out: set[str] = set()
    for ex in train:
        for s in gold_sigs(ex):
            out.add(canon_sig(s))
    return out


def comp_bound(gold: list[tuple], train_sig_set: set[str]) -> float:
    """bound = 2k / (k + len(gold)); k = count of gold's signatures (with
    multiplicity, i.e. per-element, not per-distinct-value) whose
    canon_sig string is a member of train_sig_set."""
    k = sum(1 for s in gold if canon_sig(s) in train_sig_set)
    denom = k + len(gold)
    if denom == 0:
        return 0.0
    return (2.0 * k) / denom


def compute_comp_expressibility(
    train: list[BatteryExample],
    test_comp: list[BatteryExample],
    arm_maps: dict[str, MapDict],
) -> dict[str, Any]:
    """Overall + per-topology expressibility bound and per-arm comp join_f1.

    Only topologies with >=1 example in ``test_comp`` appear in the
    per-topology dicts (no forced coverage of star/chain/cycle).
    """
    tss = train_signature_set(train)
    bounds: list[float] = []
    by_topo_bounds: dict[str, list[float]] = defaultdict(list)
    for ex in test_comp:
        gold = gold_sigs(ex)
        b = comp_bound(gold, tss)
        bounds.append(b)
        by_topo_bounds[ex.topology].append(b)
    bound_overall = _mean(bounds)
    bound_by_topology = {
        t: _mean(vs) for t, vs in sorted(by_topo_bounds.items())
    }

    arm_f1_by_topology: dict[str, dict[str, float]] = {}
    for arm, maps in arm_maps.items():
        by_topo: dict[str, list[float]] = defaultdict(list)
        for ex in test_comp:
            tokens = set(ex.question.split())
            pred = expand(maps, tokens, ex.key)
            by_topo[ex.topology].append(join_f1(pred, gold_serialized(ex)))
        arm_f1_by_topology[arm] = {
            t: _mean(vs) for t, vs in sorted(by_topo.items())
        }

    return {
        "bound_overall": bound_overall,
        "bound_by_topology": bound_by_topology,
        "arm_f1_by_topology": arm_f1_by_topology,
    }


# ---------------------------------------------------------------------------
# Per-(regime, arm) run
# ---------------------------------------------------------------------------


def run_arm(
    regime: str,
    arm: str,
    split: BatterySplit,
    perm_val_gold: list[list[str]],
    perm_test_gold: list[list[str]],
) -> dict[str, Any]:
    miner = ARM_MINERS[arm]
    atoms = miner(split.train)
    # Dedup src within arm (should already be unique)
    seen_src: set[tuple[str, ...]] = set()
    deduped: list[ResidualCandidate] = []
    for c in atoms:
        if c.src in seen_src:
            continue
        seen_src.add(c.src)
        deduped.append(c)
    atoms = deduped

    fam = ResidualFamily(domain=DomainAtoms(name=DOMAIN))
    val_score = make_val_score(split.val, atoms)

    # -----------------------------------------------------------------
    # LEAK GUARD INVARIANT (mirrors campaign_cfq_typed_join.py):
    # Atoms are mined from TRAIN ONLY (above). Gold signatures of val/test
    # rows flow ONLY into join_f1(...) calls (via val_score / score_block),
    # NEVER into atom mining or into expand(). expand() reads ONLY
    # (maps, question_tokens, key) — no gold signature parameter exists on
    # expand()'s signature, and nothing calls parse_sparql_joins on a
    # val/test row anywhere except to compute that row's OWN gold list for
    # scoring.
    # -----------------------------------------------------------------

    maps_adm, admit_log = fam.admit(
        {},
        val_score,
        thresh=1e-4,
        max_rules=128,
        celf=False,
        candidates=atoms,
    )
    n_admitted = sum(1 for e in admit_log if e.get("event") == "admit")
    val_empty = val_score({})
    val_adm = val_score(maps_adm)
    val_gain = val_adm - val_empty

    test_iid_f1 = score_block(maps_adm, split.test_iid)
    test_comp_f1: float | None
    if split.test_comp:
        test_comp_f1 = score_block(maps_adm, split.test_comp)
    else:
        test_comp_f1 = None

    empty_iid = empty_baseline(split.test_iid)
    empty_comp = empty_baseline(split.test_comp) if split.test_comp else None
    n_test_gain = test_iid_f1 - empty_iid  # used for N control columns

    # -----------------------------------------------------------------
    # Permutation addendum — a control answering a DIFFERENT question
    # than the regime-N columns — "does the judge admit structure that
    # is not there?" — it is NOT a corrected N-control.
    #
    # Re-run the IDENTICAL admission procedure against permuted val gold
    # (features untouched; gold lists shuffled across rows). Score the
    # permuted-admission maps against permuted test_iid gold (not
    # test_comp: test_comp is empty for regime N, and using test_iid
    # keeps this addendum computable uniformly across all three regimes,
    # consistent with how the existing N-control's n_test_gain is already
    # defined against test_iid).
    # -----------------------------------------------------------------
    val_score_perm = make_val_score(
        split.val, atoms, gold_override=perm_val_gold,
    )
    maps_adm_perm, admit_log_perm = fam.admit(
        {},
        val_score_perm,
        thresh=1e-4,
        max_rules=128,
        celf=False,
        candidates=atoms,
    )
    perm_n_admitted = sum(
        1 for e in admit_log_perm if e.get("event") == "admit"
    )
    perm_test_f1 = score_block(
        maps_adm_perm, split.test_iid, gold_override=perm_test_gold,
    )
    perm_empty = empty_baseline(
        split.test_iid, gold_override=perm_test_gold,
    )
    perm_test_gain = perm_test_f1 - perm_empty

    return {
        "regime": regime,
        "arm": arm,
        "n_candidates": len(atoms),
        "n_admitted": n_admitted,
        "val_gain": val_gain,
        "val_score": val_adm,
        "val_empty": val_empty,
        "test_iid": test_iid_f1,
        "test_comp": test_comp_f1,
        "empty_iid": empty_iid,
        "empty_comp": empty_comp,
        "n_test_gain": n_test_gain,
        "perm_n_admitted": perm_n_admitted,
        "perm_test_gain": perm_test_gain,
        # Transient: used by run_regime for comp expressibility; stripped
        # from JSON payload in main().
        "maps_adm": maps_adm,
        "admit_log": [
            e for e in admit_log if e.get("event") in ("admit", "stop")
        ],
    }


def run_regime(
    regime: str, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all arms for one regime.

    Returns ``(arm_results, extras)`` where ``extras`` may contain
    ``n_inventory_ceiling`` (regime N) and/or ``comp_expressibility``
    (regimes S/L).
    """
    log(f"\n=== generating regime {regime} (seed={seed}) ===")
    split = generate_regime(regime, seed=seed)
    wc = world_counts(split)
    log(
        f"  train={len(split.train)} val={len(split.val)} "
        f"test_iid={len(split.test_iid)} test_comp={len(split.test_comp)} "
        f"n_distinct_sigs={wc['n_distinct_sigs']} "
        f"n_distinct_keys={wc['n_distinct_keys']}"
    )

    extras: dict[str, Any] = {}

    # N inventory-ceiling (oracle exact-key multiset-max predictor)
    if regime == "N":
        inv_f1 = score_inventory_ceiling(split.train, split.test_iid)
        extras["n_inventory_ceiling"] = inv_f1
        log(f"  N inventory-ceiling test_iid join_f1={inv_f1:.4f}")

    # Permutation gold: computed ONCE per regime, shared by all 5 arms.
    # Seeds are fixed literals (not env-configurable) for determinism.
    # A control answering a DIFFERENT question than the regime-N columns
    # — "does the judge admit structure that is not there?" — it is NOT a
    # corrected N-control.
    perm_val_gold = permute_gold_lists(split.val, seed=1234)
    # test_iid (not test_comp) is "the test block" for this addendum —
    # test_comp is empty for regime N, and using test_iid keeps this
    # addendum computable uniformly across all three regimes, consistent
    # with how the existing N-control's n_test_gain is already defined
    # against test_iid.
    perm_test_gold = permute_gold_lists(split.test_iid, seed=1235)

    results: list[dict[str, Any]] = []
    arm_maps: dict[str, MapDict] = {}
    for arm in ARM_MINERS:
        log(f"  arm={arm} …")
        r = run_arm(regime, arm, split, perm_val_gold, perm_test_gold)
        arm_maps[arm] = r["maps_adm"]
        comp_s = "None" if r["test_comp"] is None else f"{r['test_comp']:.4f}"
        log(
            f"    n_cand={r['n_candidates']} n_adm={r['n_admitted']} "
            f"val_gain={r['val_gain']:.4f} "
            f"iid={r['test_iid']:.4f} "
            f"comp={comp_s} "
            f"perm_adm={r['perm_n_admitted']} "
            f"perm_gain={r['perm_test_gain']:.4f}"
        )
        results.append(r)

    # Comp expressibility bound + per-topology breakdown (S, L only)
    if regime in ("S", "L") and split.test_comp:
        ce = compute_comp_expressibility(
            split.train, split.test_comp, arm_maps,
        )
        extras["comp_expressibility"] = ce
        topo_b = " ".join(
            f"{t}={v:.4f}" for t, v in ce["bound_by_topology"].items()
        )
        log(
            f"  comp-expressibility bound_overall={ce['bound_overall']:.4f}"
            f"  by_topo: {topo_b}"
        )
        for arm, by_topo in ce["arm_f1_by_topology"].items():
            arm_s = " ".join(f"{t}={v:.4f}" for t, v in by_topo.items())
            log(f"    arm={arm} comp_f1_by_topo: {arm_s}")

    return results, extras


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def evaluate_verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered verdict rule to collected (regime, arm) rows."""
    by = {(r["regime"], r["arm"]): r for r in results}
    n_controls: dict[str, dict[str, Any]] = {}
    for arm in ARM_MINERS:
        if ("N", arm) in by:
            n_controls[arm] = by[("N", arm)]

    winners: list[dict[str, Any]] = []
    for regime, arms in MATCHED_ARMS.items():
        for arm in arms:
            row = by.get((regime, arm))
            if row is None:
                continue
            iid = row["test_iid"]
            comp = row["test_comp"]
            if comp is None:
                continue
            n_row = n_controls.get(arm)
            n_clean = False
            if n_row is not None:
                n_clean = (
                    n_row["n_admitted"] <= 1
                    and n_row["n_test_gain"] <= 0.02
                )
            passed = (
                iid >= 0.9
                and comp >= 0.7
                and n_row is not None
                and n_clean
            )
            detail = {
                "regime": regime,
                "arm": arm,
                "iid": iid,
                "comp": comp,
                "n_admitted_N": None if n_row is None else n_row["n_admitted"],
                "n_test_gain": None if n_row is None else n_row["n_test_gain"],
                "n_clean": n_clean if n_row is not None else False,
                "passed": passed,
            }
            if passed:
                winners.append(detail)
            row["matched_pass"] = passed
            row["matched_detail"] = detail

    can_induce = len(winners) >= 1
    verdict = (
        'machinery CAN induce planted joins'
        if can_induce
        else 'machinery CANNOT induce planted joins (pre-registered rule not met)'
    )
    return {
        "can_induce": can_induce,
        "verdict": verdict,
        "winners": winners,
        "rule": (
            '"machinery CAN induce planted joins" iff >=1 arm in its MATCHED '
            "regime (S: majority/typed/SIG; L: lexical/SIGW) reaches test "
            "join_f1 >= 0.9 IID AND >= 0.7 compositional, AND that arm's "
            "N-control is clean (n_admitted <= 1 AND N test gain <= 0.02 over "
            "empty maps)"
        ),
    }


def print_scoreboard(
    results: list[dict[str, Any]],
    verdict: dict[str, Any],
    n_inventory_ceiling: dict[str, float] | None = None,
    comp_expressibility: dict[str, dict[str, Any]] | None = None,
) -> None:
    log("\n" + "=" * 130)
    log("JOIN BATTERY SCOREBOARD (join_f1; empty-maps baseline computed per block)")
    log("=" * 130)
    hdr = (
        f"{'regime':6} {'arm':18} {'n_cand':>6} {'n_adm':>5} "
        f"{'val_gain':>9} {'empty_iid':>9} {'test_iid':>9} "
        f"{'test_comp':>9} {'N_gain':>8} "
        f"{'perm_adm':>8} {'perm_gain':>9}"
    )
    log(hdr)

    # Group results by regime to insert inventory-ceiling / expressibility
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    regime_order: list[str] = []
    for r in results:
        if r["regime"] not in by_regime:
            regime_order.append(r["regime"])
        by_regime[r["regime"]].append(r)

    for regime in regime_order:
        # N inventory-ceiling row ABOVE per-arm rows (not a 6th arm)
        if (
            n_inventory_ceiling is not None
            and regime in n_inventory_ceiling
        ):
            inv = n_inventory_ceiling[regime]
            log(
                f"{regime:6} {'inventory-ceiling':18} "
                f"{'—':>6} {'—':>5} "
                f"{'—':>9} {'—':>9} {inv:9.4f} "
                f"{'—':>9} {'—':>8} "
                f"{'—':>8} {'—':>9}"
            )

        for r in by_regime[regime]:
            comp_s = (
                f"{r['test_comp']:9.4f}"
                if r["test_comp"] is not None
                else f"{'—':>9}"
            )
            n_gain_s = (
                f"{r['n_test_gain']:8.4f}"
                if r["regime"] == "N"
                else f"{'—':>8}"
            )
            log(
                f"{r['regime']:6} {r['arm']:18} {r['n_candidates']:6d} "
                f"{r['n_admitted']:5d} {r['val_gain']:9.4f} "
                f"{r['empty_iid']:9.4f} {r['test_iid']:9.4f} "
                f"{comp_s} {n_gain_s} "
                f"{r['perm_n_admitted']:8d} {r['perm_test_gain']:9.4f}"
            )
            # Informational only — does NOT gate the pre-registered verdict
            perm_ok = (
                r["perm_n_admitted"] <= 2 and r["perm_test_gain"] <= 0.05
            )
            log(
                f"         perm-addendum (info): "
                f"{'PASS' if perm_ok else 'FAIL'} "
                f"(expect n_adm<=2 AND perm_gain<=0.05)"
            )
            detail = r.get("matched_detail")
            if detail is not None:
                status = "PASS" if detail["passed"] else "FAIL"
                log(
                    f"         matched-arm check: {status} "
                    f"(iid>={0.9}, comp>={0.7}, N_clean="
                    f"{detail['n_clean']})"
                )

        # Comp expressibility block under S/L sections
        if (
            comp_expressibility is not None
            and regime in comp_expressibility
        ):
            ce = comp_expressibility[regime]
            topo_b = " ".join(
                f"{t}={v:.4f}"
                for t, v in ce["bound_by_topology"].items()
            )
            log(
                f"         comp-expressibility bound_overall="
                f"{ce['bound_overall']:.4f}  by_topo: {topo_b}"
            )
            for arm, by_topo in ce["arm_f1_by_topology"].items():
                arm_s = " ".join(
                    f"{t}={v:.4f}" for t, v in by_topo.items()
                )
                log(f"         arm={arm:10} comp_f1_by_topo: {arm_s}")

    log("")
    log(
        "PERMUTATION-ADDENDUM EXPECTATION (informational; does not gate "
        "the verdict): n_admitted <= 2 AND perm test gain <= 0.05 per arm"
    )
    log("")
    log(f"PRE-REGISTERED RULE: {verdict['rule']}")
    log(f"OVERALL VERDICT: {verdict['verdict']}")
    if verdict["winners"]:
        for w in verdict["winners"]:
            log(
                f"  winner: regime={w['regime']} arm={w['arm']} "
                f"iid={w['iid']:.4f} comp={w['comp']:.4f} "
                f"N_adm={w['n_admitted_N']} N_gain={w['n_test_gain']:.4f}"
            )


def main() -> None:
    seed = int(os.environ.get("JB_SEED", "0"))
    want = [
        s.strip()
        for s in os.environ.get("JB_REGIMES", "S,L,N").split(",")
        if s.strip()
    ]
    results: list[dict[str, Any]] = []
    n_inventory_ceiling: dict[str, float] = {}
    comp_expressibility: dict[str, dict[str, Any]] = {}
    for regime in want:
        if regime not in ("S", "L", "N"):
            log(f"skip unknown regime {regime!r}")
            continue
        rows, extras = run_regime(regime, seed)
        results.extend(rows)
        if "n_inventory_ceiling" in extras:
            n_inventory_ceiling[regime] = extras["n_inventory_ceiling"]
        if "comp_expressibility" in extras:
            comp_expressibility[regime] = extras["comp_expressibility"]

    verdict = evaluate_verdict(results)
    print_scoreboard(
        results,
        verdict,
        n_inventory_ceiling=n_inventory_ceiling or None,
        comp_expressibility=comp_expressibility or None,
    )

    out_dir = REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Strip bulky admit logs + transient maps_adm from JSON payload
    slim = []
    for r in results:
        slim.append({
            k: v for k, v in r.items()
            if k not in ("admit_log", "matched_detail", "maps_adm")
        })
    payload = {
        "note": (
            "All five arms are template_fixed-class mined content — NO battery "
            "result changes frac_induced (the generality-metric accounting stays "
            "untouched by this slice; schema-/train-mined join atoms must not "
            "count toward frac_induced, mirroring campaign_cfq_typed_join.py)."
        ),
        "seed": seed,
        "regimes": want,
        "results": slim,
        "verdict": verdict,
        # Additive measurements (present only for regimes that were run)
        "n_inventory_ceiling": n_inventory_ceiling,
        "comp_expressibility": comp_expressibility,
    }
    out = out_dir / "join_battery.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
