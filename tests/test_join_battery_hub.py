"""Focused tests for join-battery HUB PROOF arms: CONSTRUCT, CAGG, LSTRUCT."""
from __future__ import annotations

import inspect
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # experiments/ has no __init__.py

from experiments.campaign_cfq_typed_join import SlotTables  # noqa: E402
from experiments.campaign_join_battery import (  # noqa: E402
    _CONSTRUCT_TABLES,
    _LSTRUCT_TABLES,
    DOMAIN,
    _construct_tgt,
    _lstruct_tgt,
    assemble_by_topo,
    canon_sig,
    gold_serialized,
    gold_sigs,
    make_val_score,
    mine_construct,
    mine_lstruct,
    mine_sig,
    score_block,
)
from pil.join_battery import generate_regime  # noqa: E402
from pil.residual_template import DomainAtoms, ResidualFamily  # noqa: E402

# ---------------------------------------------------------------------------
# (a) Unit: hand-built SlotTables → expected canon_sig for a held-out key
# ---------------------------------------------------------------------------


def test_construct_tgt_hand_built_tables():
    """_construct_tgt on a tiny SlotTables returns the expected canon_sig."""
    # Two predicates sharing a typed hub on subj → one star-shaped join sig.
    tables = SlotTables(
        var_rate={
            ("ns:p_a", "subj"): 1.0,
            ("ns:p_a", "obj"): 0.0,
            ("ns:p_b", "subj"): 1.0,
            ("ns:p_b", "obj"): 0.0,
        },
        req_type={
            ("ns:p_a", "subj"): "ns:person",
            ("ns:p_a", "obj"): "ns:film",
            ("ns:p_b", "subj"): "ns:person",
            ("ns:p_b", "obj"): "ns:film",
        },
        label_maj={
            ("ns:p_a", "subj"): "ns:person",
            ("ns:p_b", "subj"): "ns:person",
        },
        n_parse_fail=0,
    )
    tid = f"hand_{len(_CONSTRUCT_TABLES)}"
    _CONSTRUCT_TABLES[tid] = tables
    src = ("CONSTRUCT", tid)
    # Held-out key never seen at mine time (hand-built tables, no mining).
    key = ("ns:p_a", "ns:p_b")
    got = _construct_tgt(src, key)
    expected_sig = (
        "ns:person",
        (("ns:p_a", "subj"), ("ns:p_b", "subj")),
    )
    assert got == [canon_sig(expected_sig)]


# ---------------------------------------------------------------------------
# (b) Leak guard: construction depends only on key + train tables
# ---------------------------------------------------------------------------


def test_construct_tgt_depends_only_on_key_not_gold():
    """Same key → same constructed tgt; val/test gold never consulted."""
    split = generate_regime("S", seed=0)
    # Tiny train subset is enough to mine non-empty tables.
    train = split.train[:30]
    cands = mine_construct(train)
    assert len(cands) == 1
    src = cands[0].src
    assert src[0] == "CONSTRUCT"

    # Pick a key from val and one from test_comp (held-out).
    key_val = split.val[0].key
    key_comp = split.test_comp[0].key if split.test_comp else split.test_iid[0].key

    # Construct twice for each key; must be deterministic and gold-free.
    t1 = _construct_tgt(src, key_val)
    t2 = _construct_tgt(src, key_val)
    assert t1 == t2

    t3 = _construct_tgt(src, key_comp)
    t4 = _construct_tgt(src, key_comp)
    assert t3 == t4

    # Different keys may differ; same key identity is the leak invariant.
    # Re-call after "seeing" val/test sparql is impossible — _construct_tgt
    # has no sparql/gold parameter. Assert the signature stays (src, key)-only:
    sig = inspect.signature(_construct_tgt)
    assert list(sig.parameters) == ["src", "key"]
    # mine_construct reads train only (question, sparql pairs from train).
    sig_m = inspect.signature(mine_construct)
    assert list(sig_m.parameters) == ["train"]


# ---------------------------------------------------------------------------
# (c) End-to-end: CONSTRUCT admitted maps beat SIG on regime-S test_comp
# ---------------------------------------------------------------------------


def test_construct_beats_sig_on_regime_s_test_comp():
    """On a small regime-S split, CONSTRUCT's admitted maps > SIG on test_comp."""
    split = generate_regime("S", seed=0)
    fam = ResidualFamily(domain=DomainAtoms(name=DOMAIN))

    def admit_arm(atoms):
        val_score = make_val_score(split.val, atoms)
        maps_adm, _log = fam.admit(
            {},
            val_score,
            thresh=1e-4,
            max_rules=128,
            celf=False,
            candidates=atoms,
        )
        return maps_adm

    construct_atoms = mine_construct(split.train)
    sig_atoms = mine_sig(split.train)

    maps_c = admit_arm(construct_atoms)
    maps_s = admit_arm(sig_atoms)

    f1_c = score_block(maps_c, split.test_comp)
    f1_s = score_block(maps_s, split.test_comp)
    assert f1_c > f1_s, (
        f"expected CONSTRUCT test_comp > SIG test_comp, got "
        f"CONSTRUCT={f1_c:.4f} SIG={f1_s:.4f}"
    )


# ---------------------------------------------------------------------------
# LSTRUCT (a) Unit: assemble_by_topo round-trips train gold signatures
# ---------------------------------------------------------------------------


def test_lstruct_assemble_by_topo_round_trip():
    """assemble_by_topo(topo, key) multiset-equals gold_sigs for train rows."""
    split = generate_regime("L", seed=0)
    train = split.train[:40]
    cands = mine_lstruct(train)
    assert len(cands) == 1
    tid = cands[0].src[1]
    templates = _LSTRUCT_TABLES[tid]
    assert "star" in templates or "chain" in templates

    checked = 0
    for ex in train:
        tokens = set(ex.question.split())
        if "topoA" in tokens:
            topo = "star"
        elif "topoB" in tokens:
            topo = "chain"
        else:
            continue
        if topo not in templates:
            continue
        got = assemble_by_topo(topo, list(ex.key), templates)
        gold = [canon_sig(s) for s in gold_sigs(ex)]
        assert Counter(got) == Counter(gold), (
            f"round-trip fail topo={topo} key={ex.key}: "
            f"got={got} gold={gold}"
        )
        # Also via gold_serialized for multiset equality
        assert Counter(got) == Counter(gold_serialized(ex))
        checked += 1
    assert checked > 0, "expected at least one train example with a template"


# ---------------------------------------------------------------------------
# LSTRUCT (b) Leak guard: _lstruct_tgt depends only on (src, tokens, key)
# ---------------------------------------------------------------------------


def test_lstruct_tgt_depends_only_on_tokens_key_not_gold():
    """Same (tokens, key) → same tgt; val/test gold never consulted."""
    split = generate_regime("L", seed=0)
    train = split.train[:30]
    cands = mine_lstruct(train)
    assert len(cands) == 1
    src = cands[0].src
    assert src[0] == "LSTRUCT"

    # Pick a val and a held-out test_comp row.
    ex_val = split.val[0]
    ex_comp = split.test_comp[0] if split.test_comp else split.test_iid[0]
    tokens_val = set(ex_val.question.split())
    tokens_comp = set(ex_comp.question.split())

    t1 = _lstruct_tgt(src, tokens_val, ex_val.key)
    t2 = _lstruct_tgt(src, tokens_val, ex_val.key)
    assert t1 == t2

    t3 = _lstruct_tgt(src, tokens_comp, ex_comp.key)
    t4 = _lstruct_tgt(src, tokens_comp, ex_comp.key)
    assert t3 == t4

    # Signature is (src, question_tokens, key) only — no sparql/gold.
    sig = inspect.signature(_lstruct_tgt)
    assert list(sig.parameters) == ["src", "question_tokens", "key"]
    sig_a = inspect.signature(assemble_by_topo)
    assert "sparql" not in sig_a.parameters and "gold" not in sig_a.parameters
    sig_m = inspect.signature(mine_lstruct)
    assert list(sig_m.parameters) == ["train"]


# ---------------------------------------------------------------------------
# LSTRUCT (c) End-to-end: LSTRUCT beats SIG on regime-L star test_comp
# ---------------------------------------------------------------------------


def test_lstruct_beats_sig_on_regime_l_star_test_comp():
    """On regime L, LSTRUCT admitted maps > SIG on star subset of test_comp."""
    split = generate_regime("L", seed=0)
    star_comp = [ex for ex in split.test_comp if ex.topology == "star"]
    assert star_comp, "expected star rows in regime-L test_comp"

    fam = ResidualFamily(domain=DomainAtoms(name=DOMAIN))

    def admit_arm(atoms):
        val_score = make_val_score(split.val, atoms)
        maps_adm, _log = fam.admit(
            {},
            val_score,
            thresh=1e-4,
            max_rules=128,
            celf=False,
            candidates=atoms,
        )
        return maps_adm

    lstruct_atoms = mine_lstruct(split.train)
    sig_atoms = mine_sig(split.train)

    maps_l = admit_arm(lstruct_atoms)
    maps_s = admit_arm(sig_atoms)

    f1_l = score_block(maps_l, star_comp)
    f1_s = score_block(maps_s, star_comp)
    assert f1_l > f1_s, (
        f"expected LSTRUCT star test_comp > SIG star test_comp, got "
        f"LSTRUCT={f1_l:.4f} SIG={f1_s:.4f}"
    )
