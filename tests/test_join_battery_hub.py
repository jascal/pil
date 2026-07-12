"""Focused tests for join-battery HUB PROOF arms: CONSTRUCT and CAGG."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # experiments/ has no __init__.py

from experiments.campaign_cfq_typed_join import SlotTables  # noqa: E402
from experiments.campaign_join_battery import (  # noqa: E402
    _CONSTRUCT_TABLES,
    DOMAIN,
    _construct_tgt,
    canon_sig,
    make_val_score,
    mine_construct,
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
