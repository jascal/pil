"""Tests for German R3-minimal 4-arm case/aux_verb ladder."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import campaign_german_r1 as r1  # noqa: E402
import campaign_german_r2 as r2  # noqa: E402
import campaign_german_r3min as r3  # noqa: E402

TASKS = Path("/home/allans/code/germandata/tasks")
GDATA = Path("/home/allans/code/germandata")


# ---------------------------------------------------------------------------
# 1. sent_id alignment on real germandata
# ---------------------------------------------------------------------------

def test_head_deprel_tokens_match_morph_case_and_aux_verb_test():
    """head_deprel test token lists match morph_case and aux_verb by sent_id."""
    if not (TASKS / "head_deprel" / "test.jsonl").exists():
        pytest.skip("germandata not available")
    if not (TASKS / "morph_case" / "test.jsonl").exists():
        pytest.skip("germandata not available")
    if not (TASKS / "aux_verb" / "test.jsonl").exists():
        pytest.skip("germandata not available")

    hd = r3.load_head_deprel_split("test")
    case = r1.load_jsonl(TASKS / "morph_case" / "test.jsonl")
    aux = r2.load_task_split("aux_verb", "test")

    hd_by = r3.index_by_sent_id(hd)
    case_by = {r["sent_id"]: r for r in case}
    aux_by = {r["sent_id"]: r for r in aux}

    # morph_case ∩ head_deprel
    common_case = set(hd_by) & set(case_by)
    assert len(common_case) == len(case_by) == len(hd_by)
    for sid in common_case:
        assert hd_by[sid]["tokens"] == case_by[sid]["tokens"], sid

    # aux_verb ⊆ head_deprel (aux is indexed subset of sentences)
    for sid, rec in aux_by.items():
        assert sid in hd_by, f"aux_verb sent_id missing from head_deprel: {sid}"
        assert hd_by[sid]["tokens"] == rec["tokens"], sid


# ---------------------------------------------------------------------------
# 2. Governor lookup via head_offset
# ---------------------------------------------------------------------------

def test_governor_index_fixture():
    """Synthetic 4-token sentence: root + known offsets."""
    # tokens: 0=DET, 1=NOUN(root), 2=VERB, 3=PUNCT
    # head_offset: DET->NOUN (+1), NOUN root (0), VERB->NOUN (-1), PUNCT->NOUN (-2)
    # governors: 0→1, 1→None (root), 2→1, 3→1
    head_offset = [1, 0, -1, -2]
    assert r3.governor_index(0, head_offset) == 1
    assert r3.governor_index(1, head_offset) is None  # root
    assert r3.governor_index(2, head_offset) == 1
    assert r3.governor_index(3, head_offset) == 1

    children = r3.build_children(4, head_offset)
    assert children[1] == [0, 2, 3]
    assert children[0] == []
    assert children[2] == []


# ---------------------------------------------------------------------------
# 3. Arm A oracle-case assignment fixtures
# ---------------------------------------------------------------------------

def test_arm_a_branch1_strict_prep_wegen_gen():
    """Nominal with ADP/case child 'wegen' → singleton {Gen}."""
    # "wegen des Wetters" — wegen is case-dependent of Wetters
    tokens = ["wegen", "des", "Wetters", "."]
    upos = ["ADP", "DET", "NOUN", "PUNCT"]
    # wegen → Wetters (+2), des → Wetters (+1), Wetters root (0), . → Wetters (-1)
    head_offset = [2, 1, 0, -1]
    deprel = ["case", "det", "root", "punct"]
    strict = {"wegen": "Gen"}
    two_way: dict[str, list[str]] = {}
    verb_idx: list = []
    prep_gov, verb_gov, diag = r3.oracle_case_gov_sentence(
        tokens, upos, head_offset, deprel, strict, two_way, verb_idx,
    )
    # position 2 = Wetters should get Gen from wegen child
    assert prep_gov[2] == {"Gen"}
    assert verb_gov[2] is None
    assert diag["b1_cands"][2] == {"Gen"}
    # det "des" may or may not get government; branch1 only if it has ADP child
    assert prep_gov[0] is None  # ADP itself


def test_arm_a_branch1_two_way_prep_in():
    """Nominal with ADP/case child 'in' → {Acc, Dat} (not yet arbitrated)."""
    tokens = ["in", "dem", "Garten", "."]
    upos = ["ADP", "DET", "NOUN", "PUNCT"]
    head_offset = [2, 1, 0, -1]
    deprel = ["case", "det", "root", "punct"]
    strict: dict[str, str] = {}
    two_way = {"in": ["Acc", "Dat"]}
    prep_gov, _verb_gov, diag = r3.oracle_case_gov_sentence(
        tokens, upos, head_offset, deprel, strict, two_way, [],
    )
    assert prep_gov[2] == {"Acc", "Dat"}
    assert diag["b1_cands"][2] == {"Acc", "Dat"}


def test_arm_a_branch2_subject_exclusion():
    """Nominal with deprel=nsubj governed by VERB — branch 2 must NOT fire."""
    # "Er geht" — Er = nsubj of geht
    tokens = ["Er", "geht", "."]
    upos = ["PRON", "VERB", "PUNCT"]
    # Er → geht (+1), geht root (0), . → geht (-1)
    head_offset = [1, 0, -1]
    deprel = ["nsubj", "root", "punct"]
    # Provide a verb stem so branch 2 *could* fire if not excluded
    entry = {"counts": {"obj:Acc": 10}, "tot": 10}
    verb_stem_index = [("geh", "gehen", entry)]
    prep_gov, verb_gov, diag = r3.oracle_case_gov_sentence(
        tokens, upos, head_offset, deprel, {}, {}, verb_stem_index,
    )
    assert prep_gov[0] is None
    assert verb_gov[0] is None  # subject exclusion
    assert diag["b2_cands"][0] is None


def test_arm_a_branch2_object_fires():
    """Nominal with deprel=obj governed by VERB — branch 2 fires with majority case."""
    tokens = ["sieht", "den", "Hund", "."]
    upos = ["VERB", "DET", "NOUN", "PUNCT"]
    # sieht root, den→Hund, Hund→sieht, .→sieht
    head_offset = [0, 1, -2, -3]
    deprel = ["root", "det", "obj", "punct"]
    entry = {"counts": {"obj:Acc": 8, "obj:Dat": 2}, "tot": 10}
    verb_stem_index = [("sieh", "sehen", entry)]
    _pg, verb_gov, diag = r3.oracle_case_gov_sentence(
        tokens, upos, head_offset, deprel, {}, {}, verb_stem_index,
    )
    assert verb_gov[2] == {"Acc"}
    assert diag["b2_cands"][2] == {"Acc"}


# ---------------------------------------------------------------------------
# 3b. Arm A FULL cascade — one fixture per new rule + priority / inherit
# ---------------------------------------------------------------------------

def _full(tokens, upos, head_offset, deprel, strict=None, two_way=None, verb_idx=None):
    return r3.oracle_case_gov_sentence_full(
        tokens, upos, head_offset, deprel,
        strict or {}, two_way or {}, verb_idx or [],
    )


def test_full_cop_to_nom():
    """COP reverse-child → candidate {Nom} on the predicate nominal."""
    # "ist Lehrer" — Lehrer is predicate-nominal root with cop child "ist"
    tokens = ["ist", "Lehrer", "."]
    upos = ["AUX", "NOUN", "PUNCT"]
    head_offset = [1, 0, -1]  # ist→Lehrer (cop), Lehrer root
    deprel = ["cop", "root", "punct"]
    _pg, vg, diag = _full(tokens, upos, head_offset, deprel)
    assert vg[1] == {"Nom"}
    assert diag["fired"][1] == "cop"
    assert diag["rule_cands"]["cop"][1] == {"Nom"}


def test_full_cop_does_not_fire_on_predicate_adj():
    """Predicate ADJ hosts a cop child but is uninflected — COP must not fire.

    e.g. 'Das Essen ist lecker.' — lecker is ADJ/root with cop-child 'ist';
    gold case is '-', so forcing {Nom} is always wrong. Falls through cascade.
    """
    tokens = ["Essen", "ist", "lecker", "."]
    upos = ["NOUN", "AUX", "ADJ", "PUNCT"]
    # Essen→lecker (nsubj), ist→lecker (cop), lecker root
    head_offset = [2, 1, 0, -1]
    deprel = ["nsubj", "cop", "root", "punct"]
    _pg, vg, diag = _full(tokens, upos, head_offset, deprel)
    assert diag["rule_cands"]["cop"][2] is None
    assert diag["fired"][2] != "cop"
    # Essen (NOUN nsubj) still gets SUBJ → Nom
    assert diag["fired"][0] == "subj"
    assert vg[0] == {"Nom"}


def test_full_subj_nsubj_to_nom():
    tokens = ["Er", "geht", "."]
    upos = ["PRON", "VERB", "PUNCT"]
    head_offset = [1, 0, -1]
    deprel = ["nsubj", "root", "punct"]
    _pg, vg, diag = _full(tokens, upos, head_offset, deprel)
    assert vg[0] == {"Nom"}
    assert diag["fired"][0] == "subj"


def test_full_subj_nsubj_pass_to_nom():
    tokens = ["Er", "wurde", "gesehen", "."]
    upos = ["PRON", "AUX", "VERB", "PUNCT"]
    head_offset = [2, 1, 0, -1]  # Er→gesehen, wurde→gesehen
    deprel = ["nsubj:pass", "aux:pass", "root", "punct"]
    _pg, vg, diag = _full(tokens, upos, head_offset, deprel)
    assert vg[0] == {"Nom"}
    assert diag["fired"][0] == "subj"


def test_full_subj_csubj_to_nom():
    # Simplified: clausal subject nominal head with deprel=csubj
    tokens = ["Schwimmen", "macht", "Spaß", "."]
    upos = ["NOUN", "VERB", "NOUN", "PUNCT"]
    head_offset = [1, 0, -1, -2]  # Schwimmen→macht as csubj
    deprel = ["csubj", "root", "obj", "punct"]
    _pg, vg, diag = _full(tokens, upos, head_offset, deprel)
    assert vg[0] == {"Nom"}
    assert diag["fired"][0] == "subj"


def test_full_obj_to_acc():
    """OBJ rule assigns Acc directly (bypasses verb_government)."""
    tokens = ["sieht", "den", "Hund", "."]
    upos = ["VERB", "DET", "NOUN", "PUNCT"]
    head_offset = [0, 1, -2, -3]
    deprel = ["root", "det", "obj", "punct"]
    # Empty verb index: OBJ must still fire without verb_government
    _pg, vg, diag = _full(tokens, upos, head_offset, deprel)
    assert vg[2] == {"Acc"}
    assert diag["fired"][2] == "obj"
    assert diag["rule_cands"]["verbgov"][2] is None


def test_full_iobj_to_dat():
    """Literal iobj → Dat (near-zero coverage on GSD, but implement the letter)."""
    tokens = ["gibt", "ihm", "das", "Buch", "."]
    upos = ["VERB", "PRON", "DET", "NOUN", "PUNCT"]
    head_offset = [0, -1, 1, -3, -4]  # ihm→gibt as iobj
    deprel = ["root", "iobj", "det", "obj", "punct"]
    _pg, vg, diag = _full(tokens, upos, head_offset, deprel)
    assert vg[1] == {"Dat"}
    assert diag["fired"][1] == "iobj"


def test_full_obl_arg_to_dat():
    """DATA-DRIVEN: obl:arg → Dat (GSD bare dative-object label)."""
    tokens = ["hilft", "dem", "Mann", "."]
    upos = ["VERB", "DET", "NOUN", "PUNCT"]
    head_offset = [0, 1, -2, -3]
    deprel = ["root", "det", "obl:arg", "punct"]
    _pg, vg, diag = _full(tokens, upos, head_offset, deprel)
    assert vg[2] == {"Dat"}
    assert diag["fired"][2] == "obl_arg"


def test_full_bare_nmod_to_gen():
    """Bare nmod (no ADP/case child) → Gen."""
    tokens = ["das", "Haus", "des", "Mannes", "."]
    upos = ["DET", "NOUN", "DET", "NOUN", "PUNCT"]
    # Mannes → Haus as nmod; des → Mannes
    head_offset = [1, 0, 1, -2, -3]
    deprel = ["det", "root", "det", "nmod", "punct"]
    _pg, vg, diag = _full(tokens, upos, head_offset, deprel)
    assert vg[3] == {"Gen"}
    assert diag["fired"][3] == "nmod_gen"


def test_full_prepositional_nmod_does_not_fire_nmod_gen():
    """nmod with ADP/case child is resolved by PREP, not NMOD_GEN."""
    tokens = ["das", "Haus", "von", "dem", "Mann", "."]
    upos = ["DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"]
    # Mann → Haus (nmod); von → Mann (case); dem → Mann
    head_offset = [1, 0, 2, 1, -3, -4]
    deprel = ["det", "root", "case", "det", "nmod", "punct"]
    strict = {"von": "Dat"}
    pg, vg, diag = _full(tokens, upos, head_offset, deprel, strict=strict)
    assert pg[4] == {"Dat"}
    assert diag["fired"][4] == "prep"
    assert diag["rule_cands"]["nmod_gen"][4] is None
    assert vg[4] is None


def test_full_prep_wins_over_cop_in_ordnung():
    """Priority: PREP before COP — 'in Ordnung' → Dat set, not Nom from cop.

    Worked example: 'Ordnung' has both cop child ('war') and ADP/case child ('in');
    gold case is Dat (idiom). PREP must win.
    """
    # "... war in Ordnung"
    tokens = ["war", "in", "Ordnung", "."]
    upos = ["AUX", "ADP", "NOUN", "PUNCT"]
    head_offset = [2, 1, 0, -1]  # war→Ordnung (cop), in→Ordnung (case)
    deprel = ["cop", "case", "root", "punct"]
    two_way = {"in": ["Acc", "Dat"]}
    pg, vg, diag = _full(tokens, upos, head_offset, deprel, two_way=two_way)
    assert diag["fired"][2] == "prep"
    assert pg[2] == {"Acc", "Dat"}
    assert "Dat" in pg[2]
    assert diag["rule_cands"]["cop"][2] is None
    assert vg[2] is None  # not Nom from COP


def test_full_inherit_fixpoint_two_hop():
    """2-hop inheritance: agreement dependent of an agreement dependent.

    Chain schönen → großen → Hund, with PREP on Hund only. Fixpoint must
    propagate beyond one hop when processing order would miss a single pass.
    """
    # "mit dem schönen großen Hund" — schönen (idx2) → großen (idx3) → Hund (idx4)
    tokens = ["mit", "dem", "schönen", "großen", "Hund"]
    upos = ["ADP", "DET", "ADJ", "ADJ", "NOUN"]
    head_offset = [4, 3, 1, 1, 0]
    # mit→Hund, dem→Hund, schönen→großen(+1), großen→Hund(+1)
    deprel = ["case", "det", "amod", "amod", "root"]
    strict = {"mit": "Dat"}
    pg, _vg, diag = _full(tokens, upos, head_offset, deprel, strict=strict)
    assert diag["fired"][4] == "prep"
    assert pg[4] == {"Dat"}
    # dem inherits from Hund
    assert diag["fired"][1] == "inherit"
    assert pg[1] == {"Dat"}
    # großen inherits from Hund
    assert diag["fired"][3] == "inherit"
    assert pg[3] == {"Dat"}
    # schönen inherits from großen (2nd hop) — fails under pure one-shot L→R if
    # großen were not yet resolved; fixpoint guarantees it.
    assert diag["fired"][2] == "inherit"
    assert pg[2] == {"Dat"}


def test_partial_oracle_subject_still_excluded():
    """Regression: partial path still excludes subjects (unchanged from 0.8262 era).

    Whole-campaign 0.8262 check lives in main(); this pins the subject-exclusion
    behavior of oracle_case_gov_sentence_partial in isolation.
    """
    tokens = ["Er", "geht", "."]
    upos = ["PRON", "VERB", "PUNCT"]
    head_offset = [1, 0, -1]
    deprel = ["nsubj", "root", "punct"]
    entry = {"counts": {"obj:Acc": 10}, "tot": 10}
    verb_stem_index = [("geh", "gehen", entry)]
    pg, vg, diag = r3.oracle_case_gov_sentence_partial(
        tokens, upos, head_offset, deprel, {}, {}, verb_stem_index,
    )
    assert pg[0] is None
    assert vg[0] is None
    assert diag["b2_cands"][0] is None
    # Full cascade would assign Nom via SUBJ — contrast check
    _pg2, vg2, diag2 = r3.oracle_case_gov_sentence_full(
        tokens, upos, head_offset, deprel, {}, {}, verb_stem_index,
    )
    assert vg2[0] == {"Nom"}
    assert diag2["fired"][0] == "subj"


# ---------------------------------------------------------------------------
# 4. Arm C verb-aware span-stop
# ---------------------------------------------------------------------------

def test_arm_c_verb_aware_span_stops_before_verb():
    """Fronted PP: span from prep must stop before the verb (excludes geht/er)."""
    tokens = ["Durch", "den", "Garten", "geht", "er", "."]
    # Small verb_stem_index so "geht" stem-matches "geh" from "gehen"
    entry = {"counts": {"obj:Acc": 5}, "tot": 5}
    verb_stem_index = [("geh", "gehen", entry)]
    prep_forms = {"durch"}
    conj_forms = set(r3.CONJ_FORMS)

    # Naive span (R1): would continue past verb until punct
    naive = r1.np_span_indices(tokens, 1, prep_forms, conj_forms)
    # indices starting after "Durch" (start=1): den, Garten, geht, er — until punct
    assert 3 in naive  # "geht" included in naive
    assert 4 in naive  # "er" included in naive

    # Verb-aware: stops at "geht"
    aware = r3.np_span_indices_verb_aware(
        tokens, 1, prep_forms, conj_forms, verb_stem_index,
    )
    assert aware == [1, 2]  # "den", "Garten" only
    assert 3 not in aware
    assert 4 not in aware

    # Full prep gov: only den/Garten get Acc from Durch
    strict = {"durch": "Acc"}
    two_way: dict[str, list[str]] = {}
    gov = r3.prep_gov_case_sets_verb_aware(
        tokens, strict, two_way, prep_forms, conj_forms, verb_stem_index,
    )
    assert gov[1] == {"Acc"}  # den
    assert gov[2] == {"Acc"}  # Garten
    assert gov[3] is None  # geht — not corrupted
    assert gov[4] is None  # er — not corrupted


# ---------------------------------------------------------------------------
# 5. Arm B aux surface-participle via gold link
# ---------------------------------------------------------------------------

def test_arm_b_fires_when_governor_is_participle():
    """Ambiguous verb's head_offset governor is participle-shaped → fires."""
    # "gelesen hat" inverted? Better: "hat" dependent of "gelesen" (rare but OK for fixture)
    # Or: "worden" is governor of "ist" — use simple:
    # tokens: Er ist gegangen .  — if "ist" heads "gegangen" as child, that's dir 2.
    # Direction 1: governor is participle. Construct: "gegangen ist er"
    # ist's governor = gegangen
    tokens = ["gegangen", "ist", "er", "."]
    # gegangen root (0), ist → gegangen (-1), er → ist (-1)? simpler:
    # head: ist points to gegangen
    head_offset = [0, -1, -1, -3]  # ist.gov=gegangen; er.gov=gegangen
    idx = 1  # "ist"
    assert r2.participle_shape(tokens[0])
    assert r3.aux_oracle_fires(tokens, idx, head_offset) is True


def test_arm_b_fires_when_child_is_participle():
    """Child of ambiguous verb is participle-shaped → fires."""
    tokens = ["Er", "hat", "gelesen", "."]
    # Er→hat(+1), hat root(0), gelesen→hat(-1), .→hat(-2)
    head_offset = [1, 0, -1, -2]
    idx = 1  # hat
    assert r2.participle_shape("gelesen")
    assert r3.aux_oracle_fires(tokens, idx, head_offset) is True


def test_arm_b_fires_on_zu_infinitive_child():
    """Child is zu-infinitive shape → fires."""
    tokens = ["Er", "scheint", "zu", "schlafen", "."]
    # scheint root; zu→schlafen; schlafen→scheint; Er→scheint
    head_offset = [1, 0, 1, -2, -3]
    idx = 1  # scheint
    assert r3.zu_infinitive_shape(tokens, 3) is True
    assert r3.aux_oracle_fires(tokens, idx, head_offset) is True


def test_arm_b_negative_neither_fires():
    """No participle/zu-inf neighbor via gold links → does not fire (fallback)."""
    tokens = ["Er", "ist", "hier", "."]
    head_offset = [1, 0, -1, -2]  # ist root; Er→ist; hier→ist
    idx = 1
    assert r3.aux_oracle_fires(tokens, idx, head_offset) is False
    # Fallback path
    form_maj = {"ist": "AUX"}
    pred = r3.aux_predict_label("ist", False, form_maj, "VERB")
    assert pred == "AUX"  # from form table, not fire


# ---------------------------------------------------------------------------
# 6. Arm D clause-bounded search
# ---------------------------------------------------------------------------

def test_arm_d_fires_on_clause_final_nonadjacent_participle():
    """'Er hat das Buch gelesen.' — participle clause-final, non-adjacent; fires.

    Exact motivating R2 gap case (has_participle_in_sentence can fire but
    clause-bounded search must also fire here).
    """
    tokens = ["Er", "hat", "das", "Buch", "gelesen", "."]
    idx = 1  # hat
    assert r3.aux_heuristic_fires(tokens, idx) is True
    left, right = r3.clause_bounds(tokens, idx)
    assert left == 0
    assert right == 4  # before final punct
    assert r2.participle_shape("gelesen")


def test_arm_d_does_not_fire_across_comma_clause_boundary():
    """Participle in a DIFFERENT punctuation-separated clause must NOT fire.

    Known false-negative risk on parentheticals is documented in HONEST_SCOPE.
    """
    # "Er ist hier , sie hat gelesen ."
    # "ist" in first clause; "gelesen" in second — clause-bound must block
    tokens = ["Er", "ist", "hier", ",", "sie", "hat", "gelesen", "."]
    idx = 1  # ist
    left, right = r3.clause_bounds(tokens, idx)
    assert left == 0
    assert right == 2  # "hier" is last before comma
    assert r3.aux_heuristic_fires(tokens, idx) is False
    # But "hat" in second clause should fire
    assert r3.aux_heuristic_fires(tokens, 5) is True


# ---------------------------------------------------------------------------
# 7. Verdict / recovery-fraction pure unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "acc,expected",
    [
        (0.90, "CONFIRMED"),
        (0.88, "CONFIRMED"),
        (0.87, "PARTIAL"),
        (0.85, "PARTIAL"),
        (0.849, "INCOMPLETE"),
    ],
)
def test_case_r3_gating_thresholds(acc: float, expected: str):
    assert r3.case_r3_gating(acc) == expected


def test_recovery_fraction_normal():
    assert r3.recovery_fraction(0.04, 0.08) == pytest.approx(0.5)


def test_recovery_fraction_oracle_zero_is_na():
    assert r3.recovery_fraction(0.01, 0.0) == "N/A"


def test_recovery_fraction_oracle_negative_is_na():
    assert r3.recovery_fraction(0.01, -0.02) == "N/A"
    # must not raise
    assert r3.recovery_fraction(-0.01, -0.02) == "N/A"


def test_zu_infinitive_shape_requires_zu_prev():
    assert r3.zu_infinitive_shape(["zu", "schlafen"], 1) is True
    assert r3.zu_infinitive_shape(["schlafen"], 0) is False
    assert r3.zu_infinitive_shape(["zu", "en"], 1) is False  # too short
