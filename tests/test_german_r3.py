"""Tests for German R3 proper labeled-dependency (head_deprel) predictor."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import campaign_german_r3 as r3  # noqa: E402
import campaign_german_r3min as r3min  # noqa: E402
import wyly_lm_v5 as v5  # noqa: E402

# ---------------------------------------------------------------------------
# 1. head_deprel loader + head_offset arithmetic
# ---------------------------------------------------------------------------

def test_load_head_deprel_record_shape_and_arithmetic():
    """jsonl-shaped record → tokens/head_offset/deprel; head_index = i + offset."""
    # Synthetic sentence: DET NOUN VERB PUNCT
    # heads: DET→NOUN(+1), NOUN root(0), VERB→NOUN(-1), PUNCT→NOUN(-2)
    head_offset = [1, 0, -1, -2]
    for i, o in enumerate(head_offset):
        hi = r3.head_index(i, o)
        if o == 0:
            assert hi == i  # root self-loop
        else:
            assert hi == i + o
    assert r3.head_index(1, 0) == 1
    assert r3.head_index(0, 1) == 1
    assert r3.head_index(2, -1) == 1


def test_load_head_deprel_length_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"sent_id":"x","tokens":["a","b"],'
        '"targets":{"head_offset":[0],"deprel":["root","punct"]}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(r3, "TASKS", tmp_path.parent)
    # load_head_deprel_split looks at TASKS / "head_deprel" / f"{split}.jsonl"
    hd_dir = tmp_path / "head_deprel"
    hd_dir.mkdir()
    (hd_dir / "train.jsonl").write_text(
        '{"sent_id":"x","tokens":["a","b"],'
        '"targets":{"head_offset":[0],"deprel":["root","punct"]}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(r3, "TASKS", tmp_path)
    with pytest.raises(RuntimeError, match="length mismatch"):
        r3.load_head_deprel_split("train")


def test_clamp_offset_boundary():
    # i=0, offset=-3 → g=-3 → clamp to 0 → offset=0
    assert r3.clamp_offset(0, -3, 5) == 0
    # i=4, offset=+3 → g=7 → clamp to 4 → offset=0
    assert r3.clamp_offset(4, 3, 5) == 0
    # in-bounds unchanged
    assert r3.clamp_offset(2, -1, 5) == -1
    assert r3.clamp_offset(1, 0, 5) == 0


# ---------------------------------------------------------------------------
# 2. Haiku-tree flattening — both shapes
# ---------------------------------------------------------------------------

def test_flatten_haiku_tree_dict_shape():
    """dict with nested tree: root self-loop; children point to parent."""
    # positions 1-indexed: root@2 "Hund", child@1 "Der" (det), child@3 "." (punct)
    tree = {
        "word": "Hund",
        "dep": "root",
        "position": 2,
        "children": [
            {"word": "Der", "dep": "det", "position": 1, "children": []},
            {"word": ".", "dep": "punct", "position": 3, "children": []},
        ],
    }
    triples = r3.flatten_haiku_parsed({"tree": tree})
    assert triples is not None
    # set of (position, dep, head)
    as_set = set(triples)
    assert (2, "root", 2) in as_set  # root self-loop
    assert (1, "det", 2) in as_set
    assert (3, "punct", 2) in as_set
    arrays = r3.triples_to_arrays(triples)
    assert arrays is not None
    ho, dep = arrays
    # i=0 pos1 "Der": head=2 → offset = (2-1)-0 = 1
    # i=1 pos2 "Hund": head=2 → offset = (2-1)-1 = 0
    # i=2 pos3 ".": head=2 → offset = (2-1)-2 = -1
    assert ho == [1, 0, -1]
    assert dep == ["det", "root", "punct"]


def test_flatten_haiku_forest_list_shape():
    """Bare list of 2+ subtrees — each top-level is local root."""
    forest = [
        {
            "word": "A",
            "dep": "root",
            "position": 1,
            "children": [
                {"word": "B", "dep": "nmod", "position": 2, "children": []},
            ],
        },
        {
            "word": "C",
            "dep": "root",
            "position": 3,
            "children": [],
        },
    ]
    triples = r3.flatten_haiku_parsed(forest)
    assert triples is not None
    as_set = set(triples)
    assert (1, "root", 1) in as_set
    assert (2, "nmod", 1) in as_set
    assert (3, "root", 3) in as_set
    arrays = r3.triples_to_arrays(triples)
    assert arrays is not None
    ho, dep = arrays
    # i=0: head 1 → 0; i=1: head 1 → (1-1)-1 = -1; i=2: head 3 → (3-1)-2 = 0
    assert ho == [0, -1, 0]
    assert dep == ["root", "nmod", "root"]


# ---------------------------------------------------------------------------
# 3. Majority-offset + majority-deprel baseline
# ---------------------------------------------------------------------------

def test_majority_offset_deprel_baseline_fixture():
    """Known majority per R1-predicted-POS key; fallback global majority."""
    # POS ids: 0,0,0,1,1  offsets: 1,1,2, -1,-1  deprels: nsubj,nsubj,obj, amod,amod
    pos_ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    offsets = [1, 1, 2, -1, -1]
    deprels = ["nsubj", "nsubj", "obj", "amod", "amod"]
    dep2i = {"nsubj": 0, "obj": 1, "amod": 2, "root": 3}
    i2dep = ["nsubj", "obj", "amod", "root"]
    bundle = r3.fit_majority_baseline(pos_ids, offsets, deprels, dep2i, i2dep)

    # query pos 0 → majority offset 1 (2/3), majority deprel nsubj
    # query pos 1 → majority offset -1, deprel amod
    # query pos 2 (unseen) → global maj offset 1 (appears twice? 1,1,2,-1,-1 → 1 and -1 both 2;
    # Counter.most_common is stable by first-seen max count — both count 2; first max wins)
    q = torch.tensor([0, 1, 2], dtype=torch.long)
    sent_pos = [0, 1, 2]
    sent_n = [5, 5, 5]
    pred_o, pred_d = r3.predict_majority_baseline(q, bundle, sent_pos, sent_n)
    assert pred_o[0] == 1
    assert pred_d[0] == "nsubj"
    assert pred_o[1] == -1
    assert pred_d[1] == "amod"
    # unseen POS falls back to global majority
    assert pred_d[2] in ("nsubj", "amod")  # both count 2
    assert pred_o[2] in (1, -1)


# ---------------------------------------------------------------------------
# 4. UAS / LAS scorer
# ---------------------------------------------------------------------------

def test_uas_las_scorer_mixed():
    """Mix of correct/incorrect head and deprel."""
    # 4 tokens
    gold_ho = [1, 0, -1, -2]
    gold_dep = ["det", "root", "nsubj", "punct"]
    # pred: token0 head wrong; token1 ok head+dep; token2 ok head wrong dep; token3 ok both
    pred_ho = [0, 0, -1, -2]  # 0→self wrong (gold→1); rest heads ok
    pred_dep = ["det", "root", "obj", "punct"]  # token2 dep wrong
    uas, las = r3.score_uas_las(pred_ho, pred_dep, gold_ho, gold_dep)
    # heads correct: indices 1,2,3 → 3/4
    assert uas == pytest.approx(0.75)
    # LAS: 1 and 3 only (2 has wrong dep) → 2/4
    assert las == pytest.approx(0.5)


def test_fires_within_5_of_haiku():
    assert r3.fires_within_5_of_haiku(0.80, 0.84) is True   # within 5 pts
    assert r3.fires_within_5_of_haiku(0.78, 0.84) is False  # 6 pts below
    assert r3.fires_within_5_of_haiku(0.90, 0.84) is True   # above haiku


# ---------------------------------------------------------------------------
# 5. Serve-honest case-under-predicted-deprels wiring
# ---------------------------------------------------------------------------

def test_serve_honest_cascade_consumes_predicted_not_gold():
    """Predicted (wrong) head_offset/deprel must change cascade candidate sets vs gold.

    Fixture: 'wegen des Wetters' — gold: wegen is case-child of Wetters → PREP→Gen.
    Predicted: attach wegen as root/orphan so Wetters has no ADP/case child → PREP
    does not fire; if we mark Wetters as nsubj instead, SUBJ→Nom. Candidate sets differ.
    """
    tokens = ["wegen", "des", "Wetters", "."]
    # Gold structure (prep government on Wetters)
    gold_upos = ["ADP", "DET", "NOUN", "PUNCT"]
    gold_ho = [2, 1, 0, -1]
    gold_dep = ["case", "det", "root", "punct"]
    strict = {"wegen": "Gen"}
    two_way: dict[str, list[str]] = {}
    verb_idx: list = []

    _pg_g, vg_g, diag_g = r3min.oracle_case_gov_sentence_full(
        tokens, gold_upos, gold_ho, gold_dep, strict, two_way, verb_idx,
    )
    assert diag_g["fired"][2] == "prep"
    assert _pg_g[2] == {"Gen"}

    # Predicted (wrong): Wetters has no ADP/case child; deprel=nsubj → SUBJ→Nom
    # (serve-honest wiring uses predicted attachment + predicted POS)
    pred_upos = ["ADP", "DET", "NOUN", "PUNCT"]  # same POS, wrong structure
    pred_ho = [0, 1, 0, -1]  # wegen is local root; des→Wetters; Wetters root
    pred_dep = ["root", "det", "nsubj", "punct"]

    _pg_p, vg_p, diag_p = r3min.oracle_case_gov_sentence_full(
        tokens, pred_upos, pred_ho, pred_dep, strict, two_way, verb_idx,
    )
    # PREP must NOT fire on Wetters under predicted structure
    assert diag_p["fired"][2] != "prep"
    # SUBJ fires → Nom via verb_gov channel
    assert diag_p["fired"][2] == "subj"
    assert vg_p[2] == {"Nom"}

    # Demonstrably different candidate sets
    gold_cand = _pg_g[2] or vg_g[2]
    pred_cand = _pg_p[2] or vg_p[2]
    assert gold_cand != pred_cand
    assert gold_cand == {"Gen"}
    assert pred_cand == {"Nom"}


def test_unflatten_to_sentences_order():
    split = [
        {"sent_id": "a", "tokens": ["x", "y"]},
        {"sent_id": "b", "tokens": ["z"]},
    ]
    pred_off = [1, 0, 0]
    pred_dep = ["det", "root", "root"]
    pos_by = {"a": ["DET", "NOUN"], "b": ["VERB"]}
    out = r3.unflatten_to_sentences(split, pred_off, pred_dep, pos_by)
    assert out[0]["head_offset"] == [1, 0]
    assert out[0]["deprel"] == ["det", "root"]
    assert out[0]["upos"] == ["DET", "NOUN"]
    assert out[1]["head_offset"] == [0]
    assert out[1]["upos"] == ["VERB"]


def test_offset_class_roundtrip():
    k = 5
    for o in range(-k, k + 1):
        assert r3.class_to_offset(r3.offset_to_class(o, k), k) == o
    # OOR gold clamps to ±k for training class
    assert r3.class_to_offset(r3.offset_to_class(100, k), k) == k
    assert r3.class_to_offset(r3.offset_to_class(-100, k), k) == -k


def test_best_per_key_used_like_memorizer():
    """Smoke: majority baseline path uses v5.best_per_key like r1.memorizer."""
    key = torch.tensor([0, 0, 1], dtype=torch.long)
    val = torch.tensor([2, 2, 3], dtype=torch.long)
    table, n = v5.best_per_key(key, val, minsupp=1, mindet=0.0)
    assert n == 2
    pred = table.lookup(torch.tensor([0, 1, 9], dtype=torch.long))
    assert int(pred[0]) == 2
    assert int(pred[1]) == 3
    assert int(pred[2]) == -1
