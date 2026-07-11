"""Unit tests for qa1 config-holdout battery (generator, round-trip, B0'/H2 helpers)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from random import Random

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from campaign_qa1_cond_headroom import select_gate_tau  # noqa: E402
from campaign_qa1_config_holdout import (  # noqa: E402
    LAPLACE_ALPHA,
    _kgram_table_gated,
    compare_planted_vs_parsed,
    gen_to_story,
    roundtrip_test_via_prompts,
    roundtrip_train_via_loader,
)

from pil import qa1_battery as qb  # noqa: E402
from pil.qa1_battery import (  # noqa: E402
    ENTITIES,
    HELD_OUT_PAIRS,
    HELD_OUT_SET,
    NOVEL_ENTITIES,
    TRAIN_LEGAL_PAIRS,
    GenStory,
    _assert_train_excludes_held_out,
    _assert_train_pair_coverage,
    generate_story,
    generate_world,
    inject_held_out_into_train_text,
    is_holdout_binding,
    join_train_corpus,
    move_sentence,
    parse_story_text,
    story_prompts_bench_style,
)


# ---------------------------------------------------------------------------
# Generator determinism
# ---------------------------------------------------------------------------
def test_generate_world_determinism():
    w1 = generate_world(seed=0)
    w2 = generate_world(seed=0)
    for key in ("train", "test_holdout", "test_iid", "test_names"):
        assert len(w1[key]) == len(w2[key])
        for a, b in zip(w1[key], w2[key], strict=True):
            assert a.text == b.text
            assert a.story_id == b.story_id


def test_generate_world_sizes_and_ids():
    w = generate_world(seed=1)
    assert len(w["train"]) == 2000
    assert len(w["test_holdout"]) == 200
    assert len(w["test_iid"]) == 200
    assert len(w["test_names"]) == 200
    ids = (
        [s.story_id for s in w["train"]]
        + [s.story_id for s in w["test_holdout"]]
        + [s.story_id for s in w["test_iid"]]
        + [s.story_id for s in w["test_names"]]
    )
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Train held-out exclusion assert (real, not a no-op)
# ---------------------------------------------------------------------------
def test_train_excludes_held_out_pairs_passes_on_normal_world():
    w = generate_world(seed=0)
    _assert_train_excludes_held_out(w["train"])
    counts = _assert_train_pair_coverage(w["train"])
    for p in HELD_OUT_PAIRS:
        assert counts[p] == 0
    for p in TRAIN_LEGAL_PAIRS:
        assert counts[p] >= 20


def test_train_held_out_assert_rejects_injected_violation():
    """Deliberately inject a held-out movement; assert must raise."""
    ent, loc = HELD_OUT_PAIRS[0]
    bad_text = (
        f"{move_sentence(ent, 'moved', loc)} "
        f"John went to the hallway. "
        f"Q: Where is {ent}? A: {loc}. "
        f"John moved to the office. Mary journeyed to the bedroom. "
        f"Q: Where is John? A: office. "
        f"Sandra went to the garden. Daniel travelled to the hallway. "
        f"Q: Where is Sandra? A: garden. "
        f"Mary moved to the hallway. John went back to the bedroom. "
        f"Q: Where is Mary? A: hallway. "
        f"Daniel moved to the office. Sandra journeyed to the hallway. "
        f"Q: Where is Daniel? A: office."
    )
    bad = GenStory(text=bad_text, story_id="train:bad")
    raised = False
    try:
        _assert_train_excludes_held_out([bad])
    except AssertionError as e:
        raised = True
        assert "held-out" in str(e).lower() or ent in str(e)
    assert raised, "held-out exclusion assert must reject injected violation"

    # Corpus-text injection helper plants a held-out movement
    w = generate_world(seed=0)
    corpus = join_train_corpus(w["train"][:5])
    poisoned = inject_held_out_into_train_text(corpus, HELD_OUT_PAIRS[1])
    movs, _ = parse_story_text(poisoned)
    assert any((e, loc) in HELD_OUT_SET for e, _v, loc in movs)


def test_force_held_out_movement_in_internal_helper_rejected_by_train_assert():
    """Internal generator with unrestricted pairs can emit holdout; train assert catches it."""
    rng = Random(0)
    s = generate_story(
        rng,
        "holdout:force",
        legal_pairs=[(e, loc) for e in ENTITIES for loc in qb.LOCATIONS],
        force_holdout_turns={0, 1},
        allow_held_out_moves=True,
    )
    movs, qs = parse_story_text(s.text)
    assert any((e, loc) in HELD_OUT_SET for e, _v, loc in movs) or any(
        is_holdout_binding(e, a) for e, a in qs
    )
    if any((e, loc) in HELD_OUT_SET for e, _v, loc in movs):
        try:
            _assert_train_excludes_held_out([GenStory(text=s.text, story_id="train:0")])
            raise AssertionError("expected AssertionError from train exclude assert")
        except AssertionError as e:
            if "expected AssertionError" in str(e):
                raise
            # correctly rejected
            pass


# ---------------------------------------------------------------------------
# Round-trip parse parity
# ---------------------------------------------------------------------------
def test_roundtrip_train_via_load_train_stories():
    w = generate_world(seed=0)
    with tempfile.TemporaryDirectory() as td:
        parsed = roundtrip_train_via_loader(w["train"], Path(td))
    assert len(parsed) == 2000
    for gs, ps in zip(w["train"], parsed, strict=True):
        compare_planted_vs_parsed(gs, ps)


def test_roundtrip_test_blocks_via_bench_prompts():
    w = generate_world(seed=0)
    for key in ("test_holdout", "test_iid", "test_names"):
        parsed = roundtrip_test_via_prompts(w[key])
        assert len(parsed) == 200
        for gs, ps in zip(w[key], parsed, strict=True):
            compare_planted_vs_parsed(gs, ps)
            for q in ps.queries:
                assert q.prompt.rstrip().endswith("A:")


# ---------------------------------------------------------------------------
# Block-1 rows are holdout bindings; names never use original entities
# ---------------------------------------------------------------------------
def test_block1_rows_all_holdout_gold():
    w = generate_world(seed=0)
    for gs in w["test_holdout"]:
        _movs, qs = parse_story_text(gs.text)
        hold = [(e, a) for e, a in qs if is_holdout_binding(e, a)]
        assert len(hold) >= 2
        for e, a in hold:
            assert (e, a) in HELD_OUT_SET


def test_test_names_entities_never_original():
    w = generate_world(seed=0)
    original = set(ENTITIES)
    novel = set(NOVEL_ENTITIES)
    for gs in w["test_names"]:
        movs, qs = parse_story_text(gs.text)
        for e, _v, _loc in movs:
            assert e not in original
            assert e in novel
        for e, _a in qs:
            assert e not in original
            assert e in novel


def test_test_iid_has_no_holdout_bindings():
    w = generate_world(seed=0)
    for gs in w["test_iid"]:
        movs, qs = parse_story_text(gs.text)
        for e, _v, loc in movs:
            assert (e, loc) not in HELD_OUT_SET
        for e, a in qs:
            assert not is_holdout_binding(e, a)


# ---------------------------------------------------------------------------
# kgram-mirror parity (local gate vs raw majority counts)
# ---------------------------------------------------------------------------
def test_kgram_mirror_support_det_gate():
    """Local _kgram_table_gated admits only support>=2 and det>=0.5 keys."""
    # Synthetic stream:
    # key (1,2) → always 3 (det=1, supp=3)
    # key (4,5) → 6 twice, 7 once (det=2/3>=0.5, supp=3)
    # key (8,9) → 10 once only (supp=1 < 2) → dropped
    # key (11,12) → three distinct nexts (det=1/3 < 0.5) → dropped
    stream = (
        [1, 2, 3] * 3
        + [4, 5, 6, 4, 5, 6, 4, 5, 7]
        + [8, 9, 10]
        + [11, 12, 13, 11, 12, 14, 11, 12, 15]
    )
    table = _kgram_table_gated(stream, k=2, minsupp=2, mindet=0.5, alpha=LAPLACE_ALPHA)
    assert (1, 2) in table
    assert table[(1, 2)][0] == 3
    assert (4, 5) in table
    assert table[(4, 5)][0] == 6
    assert (8, 9) not in table  # support 1
    assert (11, 12) not in table  # det 1/3
    y, conf = table[(1, 2)]
    assert y == 3
    assert abs(conf - 3 / (3 + LAPLACE_ALPHA)) < 1e-9


# ---------------------------------------------------------------------------
# H2 tau-selection determinism
# ---------------------------------------------------------------------------
def test_select_gate_tau_deterministic():
    conf = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    correct = torch.tensor(
        [False, False, False, True, True, True, True, True, True, True]
    )
    t1 = select_gate_tau(conf, correct, seed=0)
    t2 = select_gate_tau(conf, correct, seed=0)
    t3 = select_gate_tau(conf, correct, seed=99)
    assert t1 == t2 == t3
    assert isinstance(t1, float)


# ---------------------------------------------------------------------------
# Accumulated prompt shape (bench-style)
# ---------------------------------------------------------------------------
def test_story_prompts_accumulate():
    w = generate_world(seed=0)
    gs = w["test_iid"][0]
    prompts = story_prompts_bench_style(gs)
    assert len(prompts) == 5
    for i in range(1, 5):
        prev_ans = prompts[i - 1][1]
        assert prev_ans in prompts[i][0]
        assert prompts[i][0].endswith("A:")


def test_gen_to_story_movements_match():
    w = generate_world(seed=0)
    gs = w["train"][0]
    st = gen_to_story(gs)
    movs, qs = parse_story_text(gs.text)
    assert len(st.all_movements) == len(movs) == 10
    assert len(st.queries) == 5
    for q, (e, a) in zip(st.queries, qs, strict=True):
        assert q.entity == e and q.answer == a
        live = qb.live_location(
            q.entity, [(m.entity, m.verb, m.loc) for m in q.movements_before]
        )
        assert live == a
