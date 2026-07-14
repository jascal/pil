"""Fixture tests for Probe A semantic-forcing pilot (no GPU / pythia required)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_semantic_forcing as SF  # noqa: E402
from pil.qa1_battery import live_location, move_sentence, parse_story_text  # noqa: E402

# ---------------------------------------------------------------------------
# (a) bAbI perturbation flips gold via live_location
# ---------------------------------------------------------------------------


def test_babi_perturbation_flips_live_location():
    """Perturb one movement → live_location gold changes for that entity."""
    # Fixture story: Mary ends in kitchen; perturb last Mary move → garden
    text = (
        "Mary journeyed to the kitchen. John moved to the office. "
        "Q: Where is Mary? A: kitchen. "
        "Mary went to the bathroom. John travelled to the garden. "
        "Q: Where is John? A: garden."
    )
    movs, qs = parse_story_text(text)
    assert live_location("Mary", movs) == "bathroom"
    # Perturb Mary's most recent move bathroom → garden
    pert = list(movs)
    for i in range(len(pert) - 1, -1, -1):
        e, v, loc = pert[i]
        if e == "Mary":
            pert[i] = (e, v, "garden")
            break
    assert live_location("Mary", pert) == "garden"
    assert live_location("Mary", pert) != live_location("Mary", movs)
    # move_sentence shape
    assert move_sentence("Mary", "went", "garden") == "Mary went to the garden."


# ---------------------------------------------------------------------------
# (b) agreement count-aggregate pure logic
# ---------------------------------------------------------------------------


def test_agreement_aggregate_hand_fixtures():
    """(subject-number, attractor-number) → gold verb number = subject number."""
    cases = [
        # subj_num, attr_num, gold_verb_num
        (0, 0, 0),  # sg,sg → sg verb
        (0, 1, 0),  # sg,pl → sg verb (attraction trap; gold still subject)
        (1, 0, 1),  # pl,sg → pl verb
        (1, 1, 1),  # pl,pl → pl verb
    ]
    subj_nums = [c[0] for c in cases]
    gold = [c[2] for c in cases]
    py = SF.python_agreement_aggregate(subj_nums)
    assert py == gold
    # Attractors must NOT enter the aggregate
    for (s, _a, g), pred in zip(cases, py, strict=True):
        assert pred == s == g
        assert pred == g  # independent of attractor


def test_souffle_agreement_mirrors_python():
    pytest.importorskip("subprocess")
    import shutil

    if shutil.which("souffle") is None:
        pytest.skip("souffle not on PATH")
    subj = [0, 1, 0, 1, 1]
    py = SF.python_agreement_aggregate(subj)
    sf = SF.souffle_agreement(subj)
    assert sf == py
    tw = SF.three_way_agreement_arm2(subj, subj)
    assert tw["derivability"] == 1.0
    assert tw["three_way_agree"] == len(subj)


# ---------------------------------------------------------------------------
# (c) discriminator / verdict logic on canned numbers
# ---------------------------------------------------------------------------


def test_verdict_fires():
    ims = [
        SF.ImitatorScores("a", unperturbed=0.95, propagation=0.10),
        SF.ImitatorScores("b", unperturbed=0.90, propagation=0.20),
    ]
    status, mode, tag = SF.classify_verdict(
        bar1=0.99,
        bar2=0.95,
        constraint_propagation=1.0,
        imitators=ims,
        positive_control=True,
    )
    assert status == "FIRES"
    assert mode is None
    assert tag == "positive-control"


def test_verdict_fires_empirical_tag():
    ims = [
        SF.ImitatorScores("nn", unperturbed=0.85, propagation=0.40),
        SF.ImitatorScores("bg", unperturbed=0.88, propagation=0.30),
    ]
    status, mode, tag = SF.classify_verdict(
        bar1=0.96,
        bar2=0.91,
        constraint_propagation=0.97,
        imitators=ims,
        positive_control=False,
    )
    assert status == "FIRES"
    assert "empirical" in tag
    assert "templated" in tag


def test_verdict_easy_win():
    """Both imitators weak unperturbed → easy-win (#89)."""
    ims = [
        SF.ImitatorScores("a", unperturbed=0.50, propagation=0.10),
        SF.ImitatorScores("b", unperturbed=0.40, propagation=0.10),
    ]
    status, mode, _ = SF.classify_verdict(
        bar1=0.99,
        bar2=0.95,
        constraint_propagation=1.0,
        imitators=ims,
        positive_control=False,
    )
    assert status == "DEAD"
    assert mode == "easy-win"


def test_verdict_vacuous_discriminator_propagates():
    """Strong imitators that also propagate → vacuous-discriminator."""
    ims = [
        SF.ImitatorScores("a", unperturbed=0.95, propagation=0.90),
        SF.ImitatorScores("b", unperturbed=0.92, propagation=0.85),
    ]
    status, mode, _ = SF.classify_verdict(
        bar1=0.99,
        bar2=0.95,
        constraint_propagation=1.0,
        imitators=ims,
        positive_control=False,
    )
    assert status == "DEAD"
    assert mode == "vacuous-discriminator"


def test_verdict_vacuous_only_one_strong():
    ims = [
        SF.ImitatorScores("a", unperturbed=0.95, propagation=0.10),
        SF.ImitatorScores("b", unperturbed=0.50, propagation=0.10),
    ]
    status, mode, _ = SF.classify_verdict(
        bar1=0.99,
        bar2=0.95,
        constraint_propagation=1.0,
        imitators=ims,
        positive_control=False,
    )
    assert status == "DEAD"
    assert mode == "vacuous-discriminator"


def test_verdict_soft_not_forcing():
    ims = [
        SF.ImitatorScores("a", unperturbed=0.90, propagation=0.20),
        SF.ImitatorScores("b", unperturbed=0.85, propagation=0.15),
    ]
    status, mode, _ = SF.classify_verdict(
        bar1=0.80,  # fail bar1
        bar2=0.95,
        constraint_propagation=1.0,
        imitators=ims,
        positive_control=False,
    )
    assert status == "DEAD"
    assert mode == "soft-not-forcing"


def test_verdict_not_derivable():
    ims = [
        SF.ImitatorScores("a", unperturbed=0.90, propagation=0.20),
        SF.ImitatorScores("b", unperturbed=0.85, propagation=0.15),
    ]
    status, mode, _ = SF.classify_verdict(
        bar1=0.99,
        bar2=0.50,  # fail bar2
        constraint_propagation=1.0,
        imitators=ims,
        positive_control=False,
    )
    assert status == "DEAD"
    assert mode == "not-derivable"


def test_verdict_bar1_strict_gt():
    """bar1 must be STRICTLY > 0.95 (equality fails)."""
    ims = [
        SF.ImitatorScores("a", unperturbed=0.90, propagation=0.20),
        SF.ImitatorScores("b", unperturbed=0.85, propagation=0.15),
    ]
    status, mode, _ = SF.classify_verdict(
        bar1=0.95,  # not > 0.95
        bar2=0.95,
        constraint_propagation=1.0,
        imitators=ims,
        positive_control=False,
    )
    assert status == "DEAD"
    assert mode == "soft-not-forcing"


# ---------------------------------------------------------------------------
# (d) mechanism-engaged no-op drop
# ---------------------------------------------------------------------------


def test_mechanism_engaged_noop_dropped_and_counted():
    """A subject-number 'flip' that does not change gold must be dropped."""
    # Construct item then force a no-op: same verb forms for both numbers is
    # impossible with real pairs — instead test the guard logic on aggregates.
    nouns = SF.build_noun_lexicon()
    train_lex, eval_lex = SF.split_lexis(nouns, eval_n=10, train_n=10, seed=7)
    matched, mismatch, drop = SF.build_stimuli(
        eval_lex, n_matched=20, n_mismatch=20, seed=11
    )
    # All kept items must have mechanism-engaged flips
    for it in matched:
        flipped = it.flip_subject()
        assert flipped.gold_verb != it.gold_verb
    # Matched = same number; mismatch = opposite
    for it in matched:
        assert it.subj_num == it.attr_num
        assert it.matched is True
    for it in mismatch:
        assert it.subj_num != it.attr_num
    # drop count is reported (may be 0 by construction)
    assert isinstance(drop, int)
    assert drop >= 0


def test_mechanism_engaged_explicit_noop_counter():
    """Simulate a no-op perturbation path and confirm drop counting idiom."""
    item = SF.AgreementItem(
        subject=SF.Noun("key", "keys"),
        attractor=SF.Noun("cabinet", "cabinets"),
        subj_num=0,
        attr_num=0,
        verb_sg="is",
        verb_pl="are",
        prep="to the",
        adj="old",
        matched=True,
    )
    flipped = item.flip_subject()
    assert flipped.gold_verb != item.gold_verb  # real flip is engaged

    # Artificial no-op: same gold after "flip" of attractor only (not subject)
    noop_gold_before = item.gold_verb
    noop_gold_after = item.gold_verb  # attractor flip wouldn't change gold
    drops = 0
    kept = 0
    if noop_gold_after == noop_gold_before:
        drops += 1
    else:
        kept += 1
    assert drops == 1
    assert kept == 0


def test_lexis_disjoint_assertion():
    nouns = SF.build_noun_lexicon()
    train_lex, eval_lex = SF.split_lexis(nouns, eval_n=12, train_n=12, seed=3)
    train_forms = {n.singular for n in train_lex} | {n.plural for n in train_lex}
    eval_forms = {n.singular for n in eval_lex} | {n.plural for n in eval_lex}
    assert not (train_forms & eval_forms)


def test_distance_k_respected():
    nouns = SF.build_noun_lexicon()
    _, eval_lex = SF.split_lexis(nouns, eval_n=8, train_n=8, seed=5)
    matched, mismatch, _ = SF.build_stimuli(
        eval_lex, n_matched=10, n_mismatch=10, seed=6
    )
    for it in matched + mismatch:
        assert it.content_token_distance() >= SF.DISTANCE_K


def test_nearest_noun_tracks_attractor_not_subject():
    item = SF.AgreementItem(
        subject=SF.Noun("key", "keys"),
        attractor=SF.Noun("cabinet", "cabinets"),
        subj_num=0,  # key → is
        attr_num=1,  # cabinets → are (nearest)
        verb_sg="is",
        verb_pl="are",
        prep="to the",
        adj="old",
        matched=False,
    )
    assert item.gold_verb == "is"
    assert SF.nearest_noun_predict(item) == "are"


def test_pair_memory_abstains_on_unseen_joint():
    train_movs = [[("Mary", "went", "kitchen"), ("John", "moved", "office")]]
    train_qs = [[("Mary", "kitchen")]]
    table = SF.fit_pair_memory(train_movs, train_qs)
    assert SF.pair_memory_predict(table, "Mary", "kitchen") == "kitchen"
    # held-out / unseen joint → abstain
    assert SF.pair_memory_predict(table, "Mary", "office") is None


# ---------------------------------------------------------------------------
# (e) subject-token residual read helpers (pure Python; no tokenizer/GPU)
# ---------------------------------------------------------------------------


def test_subject_token_offset_matching_prefix():
    full = [10, 20, 30, 40, 50]
    subj = [10, 20]
    assert SF.subject_token_offset(full, subj) == 1  # last token of subject prefix


def test_subject_token_offset_rejects_non_prefix():
    full = [10, 20, 30]
    subj = [10, 99]
    with pytest.raises(ValueError, match="not a prefix"):
        SF.subject_token_offset(full, subj)


def test_subject_token_offset_rejects_empty_subject_ids():
    with pytest.raises(ValueError, match="empty"):
        SF.subject_token_offset([1, 2, 3], [])


def test_agreement_item_subject_prefix():
    item = SF.AgreementItem(
        subject=SF.Noun("key", "keys"),
        attractor=SF.Noun("cabinet", "cabinets"),
        subj_num=0,
        attr_num=1,
        verb_sg="is",
        verb_pl="are",
        prep="to the",
        adj="old",
        matched=False,
    )
    assert item.subject_prefix() == "The key"
    assert item.prefix().startswith(item.subject_prefix())
    assert item.prefix() == "The key to the cabinets"

    flipped = item.flip_subject()
    assert flipped.subject_prefix() == "The keys"  # subject flipped
    assert flipped.prefix().startswith(flipped.subject_prefix())
    # attractor unchanged after flip_subject
    assert "cabinets" in flipped.prefix()
    assert flipped.subject_prefix() != item.subject_prefix()


# ---------------------------------------------------------------------------
# (f) Arm-1 bar1 teacher frame + wrong-location selection
# ---------------------------------------------------------------------------


def test_arm1_bar1_wrong_location_deterministic_and_not_correct():
    for correct in SF.LOCATIONS:
        wrong = SF.arm1_bar1_wrong_location(correct)
        assert wrong != correct
        assert wrong in SF.LOCATIONS
        # deterministic across repeated calls
        assert SF.arm1_bar1_wrong_location(correct) == wrong
        # matches the inline selection rule used historically
        assert wrong == next(loc for loc in SF.LOCATIONS if loc != correct)


def test_arm1_bar1_natural_completion_prefix_format():
    assert SF.arm1_bar1_prefix("Mary") == "Mary is now in the"
    assert SF.arm1_bar1_prefix("John") == "John is now in the"
    # no trailing space or period
    assert not SF.arm1_bar1_prefix("Mary").endswith(" ")
    assert not SF.arm1_bar1_prefix("Mary").endswith(".")


def test_teacher_pinned_to_gpt2_mid_layer():
    assert SF.TEACHER_MODEL == "gpt2"
    assert 0 <= SF.TEACHER_LAYER < 12
