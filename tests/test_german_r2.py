"""Tests for German R2 campaign (det_pron + aux_verb disambiguation, SOFT=0)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import campaign_german_r2 as r2  # noqa: E402
import wyly_lm_v5 as v5  # noqa: E402

TASKS = Path("/home/allans/code/germandata/tasks")


# ---------------------------------------------------------------------------
# (a) Loader: indices/labels align; length mismatch raises
# ---------------------------------------------------------------------------

def test_parse_task_record_aligns_indices_labels():
    rec = {
        "sent_id": "x1",
        "text": "Behebung der Probleme.",
        "tokens": ["Behebung", "der", "Probleme", "."],
        "targets": {"indices": [1], "label": ["DET"]},
    }
    parsed = r2.parse_task_record(rec)
    assert len(parsed["indices"]) == len(parsed["labels"])
    assert parsed["indices"] == [1]
    assert parsed["labels"] == ["DET"]
    assert parsed["tokens"][parsed["indices"][0]] == "der"


def test_parse_task_record_multi_target_aligns():
    rec = {
        "sent_id": "x2",
        "tokens": ["Die", "Frau", "hat", "das", "Buch", "."],
        "targets": {"indices": [0, 3], "label": ["DET", "DET"]},
    }
    parsed = r2.parse_task_record(rec)
    assert parsed["indices"] == [0, 3]
    assert parsed["labels"] == ["DET", "DET"]


def test_parse_task_record_mismatch_raises():
    rec = {
        "sent_id": "x",
        "tokens": ["a", "b"],
        "targets": {"indices": [0, 1], "label": ["DET"]},
    }
    with pytest.raises(ValueError, match="mismatch"):
        r2.parse_task_record(rec)


def test_parse_task_record_index_out_of_range_raises():
    rec = {
        "sent_id": "x",
        "tokens": ["a", "b"],
        "targets": {"indices": [5], "label": ["DET"]},
    }
    with pytest.raises(ValueError, match="out of range"):
        r2.parse_task_record(rec)


def test_real_det_pron_train_line_loads_if_present():
    path = TASKS / "det_pron" / "train.jsonl"
    if not path.exists():
        pytest.skip("germandata not available")
    rows = r2.load_jsonl(path)
    assert len(rows) > 0
    parsed = r2.parse_task_record(rows[0])
    assert len(parsed["indices"]) == len(parsed["labels"])
    for i in parsed["indices"]:
        assert 0 <= i < len(parsed["tokens"])


# ---------------------------------------------------------------------------
# (b) Mined next_cap context rule fires on a serve-honest fixture
# ---------------------------------------------------------------------------

def test_next_cap_rule_fires_and_admits_on_synthetic():
    """'die' + Capitalized → DET; 'die' + lowercase → PRON; next_cap recovers DET."""
    # Synthetic train: enough support to clear minsupp/mindet gates
    train = []
    for k in range(12):
        train.append({
            "sent_id": f"tr-det-{k}",
            "tokens": ["die", "Frau", "geht"],
            "indices": [0],
            "labels": ["DET"],
        })
        train.append({
            "sent_id": f"tr-pron-{k}",
            "tokens": ["die", "ist", "hier"],
            "indices": [0],
            "labels": ["PRON"],
        })
    # Dev mirrors the contrast so admission sees positive marginal
    dev = [
        {
            "sent_id": "dv-det",
            "tokens": ["die", "Katze", "schläft"],
            "indices": [0],
            "labels": ["DET"],
        },
        {
            "sent_id": "dv-pron",
            "tokens": ["die", "kommt", "morgen"],
            "indices": [0],
            "labels": ["PRON"],
        },
    ]
    # Held-out probe: die + CapitalizedWord → DET
    probe = [{
        "sent_id": "probe",
        "tokens": ["die", "CapitalizedWord", "x"],
        "indices": [0],
        "labels": ["DET"],
    }]

    vocab = r2.build_vocab([train])
    lab2i, _i2lab = r2.label_map(r2.DET_PRON_LABELS)
    n_classes = 2
    base = len(vocab) + 1

    tr = r2.flatten_instances(train, vocab, lab2i)
    dv = r2.flatten_instances(dev, vocab, lab2i)
    counts = r2.build_counts(tr["form_ids"], tr["y"], len(vocab), n_classes)
    fill_maj = int(counts.sum(0).argmax())

    # Without next_cap, memorizer alone ties 50/50 on "die" — either label
    # With next_cap, should admit and get perfect dev.
    cands = [c for c in r2.candidate_specs(base, "det_pron") if c["name"] == "next_cap"]
    assert len(cands) == 1
    admitted, dev_acc = r2.greedy_admit(
        tr["windows"], tr["y"], dv["windows"], dv["y"],
        counts, n_classes, fill_maj, cands,
    )
    assert any(r["name"] == "next_cap" for r in admitted), "next_cap should be admitted"
    assert dev_acc > 0.5  # positive marginal over chance/memorizer tie

    # Rules-off (counts only) on balanced "die" is ~0.5; rules-on should be 1.0 on dev
    pred_off = r2.predict_with_rules(dv["windows"], dv["y"], counts, n_classes, [], fill_maj)
    pred_on = r2.predict_with_rules(
        dv["windows"], dv["y"], counts, n_classes, admitted, fill_maj,
    )
    acc_off = r2.accuracy(pred_off, dv["y"])
    acc_on = r2.accuracy(pred_on, dv["y"])
    assert acc_on > acc_off, "positive dev marginal for next_cap"
    assert acc_on == pytest.approx(1.0)

    # Probe: die + CapitalizedWord → DET
    pr = r2.flatten_instances(probe, vocab, lab2i)
    pred = r2.predict_with_rules(
        pr["windows"], pr["y"], counts, n_classes, admitted, fill_maj,
    )
    assert int(pred[0]) == lab2i["DET"]


# ---------------------------------------------------------------------------
# (c) Majority-per-form baseline via v5.best_per_key
# ---------------------------------------------------------------------------

def test_best_per_key_majority_per_form():
    # forms: 0,0,0,1,1  labels: A,A,B,B,B  -> form0->A (2/3), form1->B (2/2)
    form = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    lab = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)  # 0=A, 1=B
    table, n = v5.best_per_key(form, lab, minsupp=1, mindet=0.0)
    assert n == 2
    pred = table.lookup(torch.tensor([0, 1, 2], dtype=torch.long))
    assert int(pred[0]) == 0  # form 0 -> A
    assert int(pred[1]) == 1  # form 1 -> B
    assert int(pred[2]) == -1  # OOV


def test_memorizer_predict_fills_global_maj():
    form = torch.tensor([0, 0, 1], dtype=torch.long)
    lab = torch.tensor([0, 0, 1], dtype=torch.long)
    table, _ = v5.best_per_key(form, lab, minsupp=1, mindet=0.0)
    pred = r2.memorizer_predict(table, torch.tensor([0, 1, 9]), global_maj=0)
    assert int(pred[0]) == 0
    assert int(pred[1]) == 1
    assert int(pred[2]) == 0  # OOV -> global maj


# ---------------------------------------------------------------------------
# (d) Scorer scores indexed tokens only
# ---------------------------------------------------------------------------

def test_scorer_scores_indexed_tokens_only():
    """Non-indexed gold/pred mismatch must not flip accuracy if scorer is correct."""
    # Sentence: tokens 0..3; only index 1 is a target.
    # Gold full:  [DET, DET, PRON, DET]  but only index 1 is scored as DET
    # Pred full:  [PRON, DET, DET, PRON]
    # If wrongly scoring all positions: 1/4 correct = 0.25
    # If correctly scoring index 1 only: 1/1 = 1.0
    gold_full = ["DET", "DET", "PRON", "DET"]
    pred_full = ["PRON", "DET", "DET", "PRON"]
    indices = [1]

    acc = r2.score_indexed_tokens(pred_full, gold_full, indices)
    assert acc == pytest.approx(1.0)

    # Full-sentence naive accuracy would be wrong:
    naive = sum(p == g for p, g in zip(pred_full, gold_full, strict=True)) / len(gold_full)
    assert naive == pytest.approx(0.25)
    assert acc != naive

    # Wrong prediction on the indexed token → 0.0
    pred_wrong = ["PRON", "PRON", "DET", "PRON"]
    assert r2.score_indexed_tokens(pred_wrong, gold_full, indices) == pytest.approx(0.0)

    # Two indices: one right, one wrong → 0.5
    indices2 = [1, 2]
    # pred_full[1]==DET==gold, pred_full[2]==DET!=PRON
    assert r2.score_indexed_tokens(pred_full, gold_full, indices2) == pytest.approx(0.5)


def test_flatten_instances_only_emits_target_rows():
    """Pipeline instances are one row per targets.indices entry, not per token."""
    split = [{
        "sent_id": "s",
        "tokens": ["a", "die", "b", "c"],
        "indices": [1],
        "labels": ["DET"],
    }]
    vocab = r2.build_vocab([split])
    lab2i, _ = r2.label_map(r2.DET_PRON_LABELS)
    inst = r2.flatten_instances(split, vocab, lab2i)
    assert inst["n"] == 1
    assert len(inst["y"]) == 1


# ---------------------------------------------------------------------------
# (e) Verdict logic — prereg rule verbatim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "det_acc,aux_acc,catalog_ok,expected",
    [
        (0.97, 0.97, True, "FIRES"),
        (0.99, 0.99, True, "FIRES"),
        (0.97, 0.97, False, "IN-BETWEEN"),  # catalog not recovered
        (0.969, 0.99, True, "IN-BETWEEN"),  # det below bar
        (0.99, 0.969, True, "IN-BETWEEN"),  # aux below bar
        (0.96, 0.96, True, "IN-BETWEEN"),
        (0.50, 0.50, False, "IN-BETWEEN"),
        (1.0, 1.0, False, "IN-BETWEEN"),
        (0.97, 0.96, True, "IN-BETWEEN"),
    ],
)
def test_verdict_branches(det_acc, aux_acc, catalog_ok, expected):
    assert r2.verdict(det_acc, aux_acc, catalog_ok) == expected


# ---------------------------------------------------------------------------
# Feature honesty smoke checks
# ---------------------------------------------------------------------------

def test_participle_shape_heuristic():
    assert r2.participle_shape("gemacht") is True
    assert r2.participle_shape("gesehen") is True
    assert r2.participle_shape("gesagt") is True
    # Too-short stem after ge-
    assert r2.participle_shape("geben") is False  # stem 'b' len 1
    # No ge-
    assert r2.participle_shape("verstanden") is False
    assert r2.participle_shape("Haus") is False


def test_next_cap_surface_only():
    toks = ["die", "Frau", "geht"]
    fb = r2.feature_bundle(toks, 0)
    assert fb["next_cap"] == 1
    toks2 = ["die", "ist", "hier"]
    fb2 = r2.feature_bundle(toks2, 0)
    assert fb2["next_cap"] == 0


def test_starts_upper():
    assert r2.starts_upper("Frau") is True
    assert r2.starts_upper("ist") is False
    assert r2.starts_upper(".") is False
