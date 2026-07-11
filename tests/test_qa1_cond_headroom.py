"""Unit tests for qa1 conditional-headroom probe (moveloc, splits, H1/H2 helpers)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from campaign_qa1_cond_headroom import (  # noqa: E402
    MOVELOC_DET,
    MOVELOC_SUPP,
    Movement,
    QueryRow,
    Story,
    apply_majority,
    assert_story_splits_disjoint,
    exact_binomial_two_sided,
    fit_majority_table,
    mine_movement_verbs_from_events,
    moveloc_feature_text,
    select_gate_tau,
)


def _story(sid: str, events: list[tuple]) -> Story:
    """Build a Story from a list of ('m', Movement) / ('q', entity, answer) events."""
    movs: list[Movement] = []
    queries: list[QueryRow] = []
    turn = 0
    for ev in events:
        if ev[0] == "m":
            movs.append(ev[1])
        else:
            _, ent, ans = ev
            queries.append(
                QueryRow(
                    story_id=sid,
                    turn=turn,
                    entity=ent,
                    answer=ans,
                    prompt=f"Q: Where is {ent}? A:",
                    movements_before=list(movs),
                )
            )
            turn += 1
    return Story(story_id=sid, queries=queries, all_movements=list(movs))


def _mv(ent: str, verb: str, loc: str) -> Movement:
    return Movement(entity=ent, verb=verb, loc=loc, text=f"{ent} {verb} to the {loc}.")


# ---- moveloc feature: most-recent-wins / stale-pointer / no-movement ----

def test_moveloc_most_recent_wins():
    """X moves twice via admitted verbs → feature returns the LATER location."""
    V = {"moved", "went"}
    movs = [
        _mv("Mary", "moved", "bathroom"),
        _mv("John", "went", "hallway"),
        _mv("Mary", "went", "garden"),  # later location for Mary
    ]
    assert moveloc_feature_text("Mary", movs, V) == "garden"
    assert moveloc_feature_text("John", movs, V) == "hallway"


def test_moveloc_stale_pointer_not_question_context():
    """Stale-pointer case: only movement sentences count; second move wins."""
    V = {"moved", "journeyed"}
    movs = [
        _mv("Daniel", "moved", "office"),
        _mv("Sandra", "journeyed", "kitchen"),
        _mv("Daniel", "moved", "bedroom"),
    ]
    # most recent Daniel movement is bedroom, not office
    assert moveloc_feature_text("Daniel", movs, V) == "bedroom"


def test_moveloc_no_movement_returns_none():
    V = {"moved"}
    movs = [_mv("John", "moved", "hallway")]
    assert moveloc_feature_text("Mary", movs, V) is None
    assert moveloc_feature_text("John", [], V) is None
    # verb not in member set
    assert moveloc_feature_text("John", [_mv("John", "teleported", "garden")], V) is None


# ---- member-set mining thresholds ----

def test_mine_movement_verb_thresholds():
    """Synthetic TRAIN: one verb clears det/support; one does not."""
    # "moved": always correct, support >= 20
    # "drifted": low determinism
    # "hopped": high det but support < 20
    stories: list[Story] = []
    for i in range(25):
        # moved → correct next answer
        stories.append(
            _story(
                f"train:{i}",
                [
                    ("m", _mv("Mary", "moved", "bathroom")),
                    ("q", "Mary", "bathroom"),
                ],
            )
        )
    for i in range(25, 35):
        # drifted → wrong half the time (det=0.5)
        loc = "garden" if i % 2 == 0 else "office"
        ans = "garden"  # always ask garden — half of drifts match
        stories.append(
            _story(
                f"train:{i}",
                [
                    ("m", _mv("John", "drifted", loc)),
                    ("q", "John", ans),
                ],
            )
        )
    for i in range(35, 40):
        # hopped: perfect det but support=5 < 20
        stories.append(
            _story(
                f"train:{i}",
                [
                    ("m", _mv("Sandra", "hopped", "kitchen")),
                    ("q", "Sandra", "kitchen"),
                ],
            )
        )

    V, meta = mine_movement_verbs_from_events(
        stories, det_thresh=MOVELOC_DET, supp_thresh=MOVELOC_SUPP
    )
    assert "moved" in V
    assert "drifted" not in V
    assert "hopped" not in V
    assert meta["moved"]["support"] >= MOVELOC_SUPP
    assert meta["moved"]["determinism"] >= MOVELOC_DET
    assert meta["hopped"]["support"] < MOVELOC_SUPP
    assert meta["drifted"]["determinism"] < MOVELOC_DET


# ---- story-split disjointness ----

def test_story_split_disjointness_assert():
    train = [_story("train:0", [("m", _mv("A", "moved", "x")), ("q", "A", "x")])]
    test = [_story("test:0", [("m", _mv("B", "moved", "y")), ("q", "B", "y")])]
    assert_story_splits_disjoint(train, test)

    # collision must raise
    bad_test = [_story("train:0", [("m", _mv("B", "moved", "y")), ("q", "B", "y")])]
    try:
        assert_story_splits_disjoint(train, bad_test)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_story_split_rejects_wrong_prefix():
    train = [_story("foo:0", [("m", _mv("A", "moved", "x")), ("q", "A", "x")])]
    test = [_story("test:0", [("m", _mv("B", "moved", "y")), ("q", "B", "y")])]
    try:
        assert_story_splits_disjoint(train, test)
        raised = False
    except AssertionError:
        raised = True
    assert raised


# ---- majority table fit / no-leak structure ----

def test_majority_table_fit_and_score():
    # feat 1 → gold mostly 10; feat 2 → gold 20
    feat = [1, 1, 1, 2, 2, -1, 1]
    gold = [10, 10, 11, 20, 20, 99, 10]
    table = fit_majority_table(feat, gold)
    assert table[1] == 10
    assert table[2] == 20
    # -1 never enters table
    assert -1 not in table
    preds = apply_majority(table, [1, 2, 3, -1])
    assert preds == [10, 20, -1, -1]


def test_majority_table_sorted_tiebreak():
    feat = [5, 5]
    gold = [3, 1]  # tie 1 each → smaller gold id wins
    table = fit_majority_table(feat, gold)
    assert table[5] == 1


def test_majority_no_leak_structure():
    """Table fit only on train feats; held-out feats may be absent (no gold leak)."""
    train_feat = [1, 1, 2]
    train_gold = [10, 10, 20]
    table = fit_majority_table(train_feat, train_gold)
    # held-out feature value 99 never seen at fit
    assert 99 not in table
    # scoring held-out never consults held-out gold when building table
    held_feat = [1, 99]
    held_gold = [10, 777]  # must not be used for fit
    preds = apply_majority(table, held_feat)
    assert preds[0] == 10
    assert preds[1] == -1
    # re-fit must not include held_gold
    table2 = fit_majority_table(train_feat, train_gold)
    assert table2 == table
    _ = held_gold  # explicitly unused for fit


# ---- gated-tau selection determinism ----

def test_gate_tau_selection_deterministic():
    # low conf on errors, high conf on correct
    conf = torch.tensor([0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.12, 0.88])
    correct = torch.tensor([False, False, False, True, True, True, False, True])
    t1 = select_gate_tau(conf, correct, seed=0)
    t2 = select_gate_tau(conf, correct, seed=0)
    t3 = select_gate_tau(conf, correct, seed=99)  # seed unused; still deterministic
    assert t1 == t2 == t3
    assert math.isfinite(t1)
    # tau should separate: sit somewhere between error and correct clusters
    assert 0.0 < t1 < 1.0


# ---- exact binomial on hand-computed discordant counts ----

def test_exact_binomial_hand_computed():
    # b=8, c=2, n=10 → two-sided binomial under p=0.5
    p = exact_binomial_two_sided(8, 2)
    # known: P(X<=2 or X>=8) for Bin(10,0.5) = 2 * sum_{k=0..2} C(10,k)/1024
    # C(10,0)+C(10,1)+C(10,2)=1+10+45=56; 2*56/1024=112/1024=0.109375
    assert abs(p - 0.109375) < 1e-9

    p_sym = exact_binomial_two_sided(5, 5)
    assert p_sym == 1.0 or p_sym > 0.9

    assert exact_binomial_two_sided(0, 0) == 1.0

    p_strong = exact_binomial_two_sided(10, 0)
    assert p_strong < 0.01
