"""Tests for German R1 campaign (SOFT=0 register arbitration student)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import campaign_german_r1 as r1  # noqa: E402
import wyly_lm_v5 as v5  # noqa: E402

TASKS = Path("/home/allans/code/germandata/tasks")
REGS = Path("/home/allans/code/germandata/registers")


# ---------------------------------------------------------------------------
# (a) Data loader
# ---------------------------------------------------------------------------

def test_parse_task_record_aligns_tokens_targets():
    rec = {
        "sent_id": "x1",
        "text": "Sehr gute Beratung.",
        "tokens": ["Sehr", "gute", "Beratung", "."],
        "targets": {"upos": ["ADV", "ADJ", "NOUN", "PUNCT"]},
    }
    parsed = r1.parse_task_record(rec, "upos")
    assert len(parsed["tokens"]) == len(parsed["targets"])
    assert parsed["tokens"] == ["Sehr", "gute", "Beratung", "."]
    assert parsed["targets"] == ["ADV", "ADJ", "NOUN", "PUNCT"]


def test_parse_task_record_mismatch_raises():
    rec = {
        "sent_id": "x",
        "tokens": ["a", "b"],
        "targets": {"upos": ["NOUN"]},
    }
    with pytest.raises(ValueError, match="mismatch"):
        r1.parse_task_record(rec, "upos")


def test_contraction_expanded_in_real_train_line():
    """GSD JSONL already expands contractions (im -> in dem); consume tokens as given."""
    path = TASKS / "pos" / "train.jsonl"
    if not path.exists():
        pytest.skip("germandata not available")
    found = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text", "")
            if " im " in text or text.startswith("Im ") or " im" in text:
                toks = rec["tokens"]
                # expanded form present as separate tokens
                if "in" in toks and "dem" in toks:
                    found = rec
                    break
            if " zur " in text or " zur" in text:
                toks = rec["tokens"]
                if "zu" in toks and "der" in toks:
                    found = rec
                    break
    assert found is not None, "expected a contraction-expanded train sentence"
    parsed = r1.parse_task_record(found, "upos")
    assert len(parsed["tokens"]) == len(parsed["targets"])
    # text still has the orthographic contraction somewhere
    assert "im" in found["text"].lower() or "zur" in found["text"].lower()
    # tokens are expanded
    assert (
        ("in" in parsed["tokens"] and "dem" in parsed["tokens"])
        or ("zu" in parsed["tokens"] and "der" in parsed["tokens"])
    )


# ---------------------------------------------------------------------------
# (b) Register lookup (article paradigm inversion)
# ---------------------------------------------------------------------------

def test_article_paradigm_inversion_dem_and_die():
    # Prefer real register file; fall back to inline fixture shape
    path = REGS / "article_paradigm.json"
    if path.exists():
        table = json.loads(path.read_text(encoding="utf-8"))["table"]
    else:
        table = {
            "definite": {
                "Nom": {"Masc": "der", "Fem": "die", "Neut": "das", "Plur": "die"},
                "Acc": {"Masc": "den", "Fem": "die", "Neut": "das", "Plur": "die"},
                "Dat": {"Masc": "dem", "Fem": "der", "Neut": "dem", "Plur": "den"},
                "Gen": {"Masc": "des", "Fem": "der", "Neut": "des", "Plur": "der"},
            },
            "indefinite": {
                "Nom": {"Masc": "ein", "Fem": "eine", "Neut": "ein"},
                "Acc": {"Masc": "einen", "Fem": "eine", "Neut": "ein"},
                "Dat": {"Masc": "einem", "Fem": "einer", "Neut": "einem"},
                "Gen": {"Masc": "eines", "Fem": "einer", "Neut": "eines"},
            },
        }
    inv = r1.invert_article_paradigm(table)
    assert inv["dem"] == {"Dat"}
    assert "Nom" in inv["die"] and "Acc" in inv["die"]
    assert len(inv["die"]) > 1  # ambiguous
    assert inv["des"] == {"Gen"}


# ---------------------------------------------------------------------------
# (c) Majority-per-form baseline (v5.best_per_key)
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


# ---------------------------------------------------------------------------
# (d) Scorer: accuracy, null floor, per-class
# ---------------------------------------------------------------------------

def test_scorer_accuracy_null_floor_per_class():
    # gold:  -, Nom, Acc, -, Dat   (null id = 4 for CASE_LABELS order Nom Acc Dat Gen -)
    # pred:  -, Nom, Nom, -, Dat
    # overall acc = 4/5 = 0.8
    # null floor = 2/5 = 0.4
    # Nom: 1/1, Acc: 0/1, Dat: 1/1, Gen: 0/0, -: 2/2
    case2i = {c: i for i, c in enumerate(r1.CASE_LABELS)}
    gold = torch.tensor([
        case2i["-"], case2i["Nom"], case2i["Acc"], case2i["-"], case2i["Dat"],
    ])
    pred = torch.tensor([
        case2i["-"], case2i["Nom"], case2i["Nom"], case2i["-"], case2i["Dat"],
    ])
    assert r1.accuracy(pred, gold) == pytest.approx(0.8)
    assert r1.null_floor(gold, case2i["-"]) == pytest.approx(0.4)
    pc = r1.per_class_accuracy(pred, gold, r1.CASE_LABELS)
    assert pc["Nom"]["acc"] == pytest.approx(1.0)
    assert pc["Nom"]["support"] == 1
    assert pc["Acc"]["acc"] == pytest.approx(0.0)
    assert pc["Acc"]["support"] == 1
    assert pc["Dat"]["acc"] == pytest.approx(1.0)
    assert pc["-"]["acc"] == pytest.approx(1.0)
    assert pc["-"]["support"] == 2
    assert pc["Gen"]["support"] == 0


# ---------------------------------------------------------------------------
# (e) Verdict logic — all three branches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pos_acc,case_acc,expected",
    [
        (0.95, 0.90, "FIRES"),
        (0.99, 0.95, "FIRES"),
        (0.95, 0.89, "IN-BETWEEN"),  # case below bar
        (0.94, 0.91, "IN-BETWEEN"),  # pos below bar but >= 0.85
        (0.90, 0.90, "IN-BETWEEN"),
        (0.849, 0.99, "HALTED"),
        (0.50, 0.50, "HALTED"),
        (0.85, 0.90, "IN-BETWEEN"),  # pos == 0.85 is not < 0.85 and not >= 0.95
    ],
)
def test_verdict_branches(pos_acc, case_acc, expected):
    assert r1.verdict(pos_acc, case_acc) == expected


# ---------------------------------------------------------------------------
# Enrichment tests: narrow-and-arbitrate, government spans, verb stem, GNN
# ---------------------------------------------------------------------------

def test_masked_arbitrate_hard_prune_and_soft_pick_within_set():
    """Hard prune keeps answer in candidate set; soft picks best survivor."""
    # Labels: 0=Nom, 1=Acc, 2=Dat, 3=Gen, 4=-
    n_classes = 5
    # Single position, form id 0. Unconstrained global majority is class 4 ("-")
    # with huge mass, but candidate set is {Nom=0, Acc=1}. Form row prefers Acc.
    form_counts = torch.zeros(3, n_classes, dtype=torch.long)
    form_counts[0, 4] = 100  # unconstrained argmax would be "-"
    form_counts[0, 0] = 3    # Nom
    form_counts[0, 1] = 7    # Acc  <- best within {Nom, Acc}
    form_ids = torch.tensor([0], dtype=torch.long)

    # Empty bigram support so Score A wins when it has support
    bg_keys = torch.empty(0, dtype=torch.long)
    bg_counts = torch.zeros(0, n_classes, dtype=torch.long)
    bg_ids = torch.tensor([99], dtype=torch.long)

    # Fallback would pick global maj outside the set
    fallback = torch.tensor([4], dtype=torch.long)
    cand = [{0, 1}]

    pred = r1.masked_arbitrate(
        cand, form_counts, form_ids, bg_keys, bg_counts, bg_ids, n_classes,
        fallback=fallback,
    )
    assert int(pred[0]) in (0, 1), "hard-prune invariant: must stay in candidate set"
    assert int(pred[0]) == 1, "soft pick within set should choose Acc (7 > 3)"

    # Naive unconstrained argmax on form row would be class 4
    unconstrained = int(form_counts[0].argmax())
    assert unconstrained == 4
    assert int(pred[0]) != unconstrained


def test_masked_arbitrate_intersection_and_empty_degrades_to_unconstrained():
    """Intersect multiple sources; empty intersection does not crash / force a label."""
    n_classes = 5
    form_counts = torch.zeros(2, n_classes, dtype=torch.long)
    form_counts[0, 2] = 5  # Dat majority for form 0
    form_counts[1, 4] = 9
    form_ids = torch.tensor([0, 1], dtype=torch.long)
    bg_keys = torch.empty(0, dtype=torch.long)
    bg_counts = torch.zeros(0, n_classes, dtype=torch.long)
    bg_ids = torch.tensor([0, 1], dtype=torch.long)
    fallback = torch.tensor([4, 4], dtype=torch.long)

    # Position 0: {Nom, Acc} ∩ {Acc, Dat} = {Acc}
    # Position 1: {Nom} ∩ {Dat} = empty -> unconstrained (fallback)
    inter = {0, 1} & {1, 2}
    assert inter == {1}
    empty = {0} & {2}
    assert empty == set()

    cand = [inter, None]  # empty intersection already degraded to None by merge
    # Also exercise merge_candidate_sources empty-intersection path
    sources_a = [[{"Nom", "Acc"}], [{"Nom"}]]
    sources_b = [[{"Acc", "Dat"}], [{"Dat"}]]
    lab2i = {c: i for i, c in enumerate(r1.CASE_LABELS)}
    merged = r1.merge_candidate_sources(sources_a, sources_b, lab2i=lab2i)
    assert merged[0] == {lab2i["Acc"]}
    assert merged[1] is None  # empty intersection -> unconstrained

    pred = r1.masked_arbitrate(
        cand, form_counts, form_ids, bg_keys, bg_counts, bg_ids, n_classes,
        fallback=fallback,
    )
    assert int(pred[0]) == 1  # Acc only survivor; form row has 0 for Acc -> global among allowed
    # With zero form support on Acc, falls to global marginal over {Acc}:
    # global is mostly class 4, but Acc is the only allowed — so Acc
    assert int(pred[0]) in cand[0]
    assert int(pred[1]) == 4  # unconstrained uses fallback


def test_np_span_to_clause_boundary_strict_and_two_way():
    """Prep government spans det+adj+noun and stops at punct / conj / next prep."""
    strict = {"mit": "Dat", "für": "Acc"}
    two_way = {"in": ["Acc", "Dat"], "auf": ["Acc", "Dat"]}
    prep_forms = set(strict) | set(two_way) | {"von", "zu"}
    conj_forms = {"und", "oder", "aber"}

    # mit dem alten Haus , ...  -> Dat on dem, alten, Haus; stop at comma
    tokens = ["mit", "dem", "alten", "Haus", ",", "und", "mehr"]
    sets = r1.prep_gov_case_sets(tokens, strict, two_way, prep_forms, conj_forms)
    assert sets[0] is None  # the prep itself
    assert sets[1] == {"Dat"}
    assert sets[2] == {"Dat"}
    assert sets[3] == {"Dat"}
    assert sets[4] is None  # punctuation boundary (not assigned)
    assert sets[5] is None
    assert sets[6] is None

    # in die schöne Stadt und ... -> two-way {Acc,Dat} on die/schöne/Stadt; stop at und
    tokens2 = ["in", "die", "schöne", "Stadt", "und", "Land"]
    sets2 = r1.prep_gov_case_sets(tokens2, strict, two_way, prep_forms, conj_forms)
    assert sets2[1] == {"Acc", "Dat"}
    assert sets2[2] == {"Acc", "Dat"}
    assert sets2[3] == {"Acc", "Dat"}
    assert sets2[4] is None  # conjunction terminates
    assert sets2[5] is None

    # für den Mann mit ... -> Acc on den/Mann; stop at next prep "mit"
    tokens3 = ["für", "den", "Mann", "mit", "Hut"]
    sets3 = r1.prep_gov_case_sets(tokens3, strict, two_way, prep_forms, conj_forms)
    assert sets3[1] == {"Acc"}
    assert sets3[2] == {"Acc"}
    # "mit" starts a new span; "Hut" gets Dat from mit, not Acc from für
    assert sets3[3] is None  # the prep "mit"
    assert sets3[4] == {"Dat"}

    # np_span_indices unit check
    idx = r1.np_span_indices(
        ["X", "dem", "alten", "Haus", "."], 1, prep_forms, conj_forms,
    )
    assert idx == [1, 2, 3]


def test_verb_stem_prefix_match_beats_exact_only():
    """Inflected surface sharing a lemma stem matches; exact lemma still matches."""
    path = REGS / "verb_government.json"
    if not path.exists():
        pytest.skip("germandata not available")
    table = json.loads(path.read_text(encoding="utf-8"))["table"]
    assert "machen" in table
    stem_index = r1.build_verb_stem_index(table)

    # exact lemma match still works
    m_exact = r1.match_verb_stem("machen", stem_index)
    assert m_exact is not None
    assert m_exact[0] == "machen"

    # inflected form old exact-match would miss (shares stem prefix "mach")
    # note: ge-participle "gemacht" is intentionally NOT matched by startswith-stem
    # (documented ablaut/affix limitation); use finite inflection instead.
    m_infl = r1.match_verb_stem("machte", stem_index)
    assert m_infl is not None
    assert m_infl[0] == "machen"
    case, conf = r1.verb_majority_case(m_infl[1])
    assert case is not None
    assert conf > 0
    # old exact-match path would only hit lemma key
    assert "machte" not in table
    assert "machen" in table

    # another lemma: absolvieren -> absolviert
    assert "absolvieren" in table
    m_abs = r1.match_verb_stem("absolviert", stem_index)
    assert m_abs is not None
    assert m_abs[0] == "absolvieren"

    # stem helper
    assert r1.verb_stem("machen") == "mach"
    assert r1.verb_stem("absolvieren") == "absolvier"

    # government span after stem-matched (inflected) verb
    tokens = ["Sie", "machten", "die", "Arbeit", "."]
    sets = r1.verb_gov_case_sets(tokens, stem_index, prep_forms=set(), conj_forms={"und"})
    # majority for machen is Acc; span covers die + Arbeit, stops at punct
    assert sets[2] is not None and "Acc" in sets[2]
    assert sets[3] is not None and "Acc" in sets[3]
    assert sets[4] is None


def test_gnn_register_tier_inversion_and_nonempty():
    """GNN candidate inversions for known forms; register tier tables non-empty."""
    art_path = REGS / "article_paradigm.json"
    pro_path = REGS / "pronoun_paradigm.json"
    adj_path = REGS / "adjective_endings.json"
    if not (art_path.exists() and pro_path.exists() and adj_path.exists()):
        pytest.skip("germandata not available")

    art_table = json.loads(art_path.read_text(encoding="utf-8"))["table"]
    pro_table = json.loads(pro_path.read_text(encoding="utf-8"))["table"]
    adj_table = json.loads(adj_path.read_text(encoding="utf-8"))["table"]

    art_f2g = r1.invert_article_paradigm_gnn(art_table)
    pro_f2g = r1.invert_pronoun_paradigm_gnn(pro_table)
    end2g = r1.invert_adjective_endings_gnn(adj_table)

    # Article Plur-key form "die" includes -|Plur (and likely Fem|Sing from Fem slots)
    assert "-|Plur" in art_f2g["die"]
    # Singular gendered: "dem" is Masc/Neut Dat -> Masc|Sing and/or Neut|Sing
    assert "Masc|Sing" in art_f2g["dem"] or "Neut|Sing" in art_f2g["dem"]
    assert "Masc|Sing" in art_f2g["der"] or "Fem|Sing" in art_f2g["der"]

    # Pronoun "Sie": 2_Formal -> -|Plur; dual-variant union also pulls Fem|Sing from "sie"
    sie_set = r1.lookup_pronoun_gnn_set("Sie", pro_f2g)
    assert sie_set is not None
    assert "Fem|Sing" in sie_set
    assert "-|Plur" in sie_set

    # Surface "sie" alone also has both (3_Sing_Fem + 3_Plur)
    sie_low = r1.lookup_pronoun_gnn_set("sie", pro_f2g)
    assert sie_low is not None
    assert "Fem|Sing" in sie_low
    assert "-|Plur" in sie_low

    # png mapping sanity
    assert r1.png_key_to_gnn("2_Formal") == "-|Plur"
    assert r1.png_key_to_gnn("3_Sing_Fem") == "Fem|Sing"
    assert r1.png_key_to_gnn("1_Sing") == "-|Sing"
    assert r1.gender_key_to_gnn("Plur") == "-|Plur"
    assert r1.gender_key_to_gnn("Masc") == "Masc|Sing"

    # Register tier non-empty (unlike former gnn_reg_tables = {})
    assert len(art_f2g) > 0
    assert len(pro_f2g) > 0
    assert len(end2g) > 0
    gnn_reg_tables = {
        "article_gnn": art_f2g,
        "pronoun_gnn": pro_f2g,
        "adj_end_gnn": end2g,
    }
    gnn_reg_names = list(gnn_reg_tables.keys())
    assert gnn_reg_names == ["article_gnn", "pronoun_gnn", "adj_end_gnn"]
    assert all(len(v) > 0 for v in gnn_reg_tables.values())
