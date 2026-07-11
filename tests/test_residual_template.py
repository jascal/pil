"""Tests for residual templates: induction, CELF caveats, honesty guards."""
from __future__ import annotations

from pil.residual_template import (
    CFQ_TEMPLATES,
    SYNTH_TEMPLATES,
    DomainAtoms,
    NFoldTemplate,
    PrefixBodyTemplate,
    ResidualCandidate,
    ResidualFamily,
    RewriteSynthesizer,
    StructuralSeedTemplate,
    candidate_provenance,
    cfq_domain_atoms,
    induce_nfold_markers,
    listops_domain_atoms,
    resolve_nfold_markers,
    scan_domain_atoms,
)


def test_induce_nfold_markers_no_hand_markers():
    maps = {
        ("a", "x2"): ["A", "A"],
        ("b", "x2"): ["B", "B"],
        ("c", "x3"): ["C", "C", "C"],
        ("d", "x3"): ["D", "D", "D"],
    }
    m = induce_nfold_markers(maps)
    assert m["x2"] == 2
    assert m["x3"] == 3


def test_induce_prefers_largest_k_for_bare_unit():
    """[A]*4 should induce k=4 / unit [A], not k=2 / unit [A,A]."""
    maps = {
        ("a", "x4"): ["A", "A", "A", "A"],
        ("b", "x4"): ["B", "B", "B", "B"],
    }
    m = induce_nfold_markers(maps, max_k=8)
    assert m["x4"] == 4


def test_induce_withholds_under_inconsistent_votes():
    """Irregular: non-fold + wrong-k under same marker → no marker (min_support/margin)."""
    only_irreg = {
        ("z", "x2"): ["Z", "NOISE"],
        ("y", "x2"): ["Y", "Y", "Y"],  # 3-fold under x2 label
    }
    m = induce_nfold_markers(only_irreg, min_support=2, min_margin=1)
    assert "x2" not in m


def test_nfold_with_empty_markers_auto_induce():
    maps = {
        ("a", "x2"): ["A", "A"],
        ("b", "x2"): ["B", "B"],
    }
    dom = DomainAtoms(name="t", nfold_markers={}, auto_induce_markers=True)
    cands = NFoldTemplate().propose(maps, dom)
    assert any(c.src == ("a",) for c in cands)
    assert any(c.meta.get("marker_induced") for c in cands)


def test_supplied_marker_conflict_keeps_induced():
    """Wrong hand seed must not suppress data-backed induction."""
    maps = {
        ("a", "m"): ["A", "A"],
        ("b", "m"): ["B", "B"],
    }
    dom = DomainAtoms(name="t", nfold_markers={"m": 4}, auto_induce_markers=True)
    markers, conflicts = resolve_nfold_markers(maps, dom)
    assert markers["m"] == 2
    assert conflicts and conflicts[0]["resolution"] == "keep_induced"
    cands = NFoldTemplate().propose(maps, dom)
    assert any(c.src == ("a",) and c.tgt == ("A",) for c in cands)


def test_prefix_body_strips_known_prefix():
    maps = {("run", "left"): ["I_TURN_LEFT", "I_RUN"]}
    dom = DomainAtoms(
        name="t",
        prefix_tokens=frozenset({"I_TURN_LEFT", "I_TURN_RIGHT"}),
        auto_induce_markers=False,
    )
    cands = PrefixBodyTemplate().propose(maps, dom)
    assert len(cands) == 1
    assert cands[0].src == ("run",)
    assert cands[0].tgt == ("I_RUN",)


def test_structural_seeds_domain_specific():
    maps: dict = {}
    dom = DomainAtoms(
        name="t",
        structural_seeds={("turn", "left"): ["I_TURN_LEFT"]},
    )
    cands = StructuralSeedTemplate().propose(maps, dom)
    assert len(cands) == 1
    assert cands[0].pattern_agnostic is False


def test_rewrite_synth_repeat_without_markers():
    maps = {
        ("c", "x2"): ["C", "C"],
        ("d", "x2"): ["D", "D"],
    }
    dom = DomainAtoms(name="t", nfold_markers={}, auto_induce_markers=True)
    cands = RewriteSynthesizer().propose(maps, dom)
    assert any(c.src == ("c",) and c.tgt == ("C",) for c in cands)


def test_scan_pack_induce_only():
    short = {
        ("run", "twice"): ["I_RUN", "I_RUN"],
        ("look", "twice"): ["I_LOOK", "I_LOOK"],
        ("walk",): ["I_WALK"],
        ("walk", "left"): ["I_TURN_LEFT", "I_WALK"],
    }
    fam = ResidualFamily(scan_domain_atoms(induce_only=True))
    prop = fam.propose_map(short)
    assert prop[("run",)] == ["I_RUN"]
    assert ("turn", "left") in prop


def test_listops_induce_only_default():
    short = {
        ("c", "x2"): ["C", "C"],
        ("d", "x2"): ["D", "D"],
        ("a",): ["A"],
    }
    fam = ResidualFamily(listops_domain_atoms(induce_only=True))
    assert fam.domain.nfold_markers == {}
    prop = fam.propose_map(short)
    assert prop[("c",)] == ["C"]
    assert fam.diagnostics(short)["induced_nfold_markers"]["x2"] == 2


def test_admit_selects_helpful_residual():
    short = {
        ("c", "x2"): ["C", "C"],
        ("d", "x2"): ["D", "D"],
        ("a",): ["A"],
    }
    val = [(["c"], ["C"]), (["a"], ["A"])]

    def score(maps):
        ok = 0
        for cmd, gold in val:
            if maps.get(tuple(cmd)) == gold:
                ok += 1
        return ok / len(val)

    fam = ResidualFamily(listops_domain_atoms(induce_only=True))
    maps, log = fam.admit(short, score, thresh=1e-4, celf=False)
    assert maps.get(("c",)) == ["C"]
    admits = [e for e in log if e.get("event") == "admit"]
    assert len(admits) >= 1
    assert all(e.get("event") in ("admit", "stop") for e in log)


def test_admit_default_is_naive_not_celf():
    """Default must be naive (correct); complementary leaves need re-score each round."""
    import inspect
    sig = inspect.signature(ResidualFamily.admit)
    assert sig.parameters["celf"].default is False


def test_complementary_leaves_naive_recovers_both():
    """Non-submodular: leaf x pays only after a is admitted.

    Score: empty=0, {a}=0.3, {x}=0, {a,x}=0.8 — x's marginal rises after a.
    Default (naive) must admit both. CELF is opt-in/lossy and is not asserted equal.
    """
    short: dict = {}
    a = ResidualCandidate(src=("a",), tgt=("A",), template_id="t", domain="t")
    x = ResidualCandidate(src=("x",), tgt=("X",), template_id="t", domain="t")

    def score(maps):
        has_a = ("a",) in maps
        has_x = ("x",) in maps
        if has_a and has_x:
            return 0.8
        if has_a:
            return 0.3
        if has_x:
            return 0.0
        return 0.0

    fam = ResidualFamily(DomainAtoms(name="t"))
    m_naive, _ = fam.admit(short, score, candidates=[a, x], thresh=0.05, celf=False)
    assert ("a",) in m_naive and ("x",) in m_naive
    assert score(m_naive) == 0.8


def test_celf_opt_in_smoke_does_not_claim_global_correctness():
    """CELF is opt-in. This only checks it runs and tags the stop event.

    Do **not** assert score equality with naive — fuzz (~2%) shows CELF can
    underperform on complementary objectives. Correctness path = naive default.
    """
    short: dict = {}
    a = ResidualCandidate(src=("a",), tgt=("A",), template_id="t", domain="t")
    x = ResidualCandidate(src=("x",), tgt=("X",), template_id="t", domain="t")

    def score(maps):
        has_a = ("a",) in maps
        has_x = ("x",) in maps
        if has_a and has_x:
            return 0.8
        if has_a:
            return 0.3
        return 0.0

    fam = ResidualFamily(DomainAtoms(name="t"))
    _maps, log = fam.admit(short, score, candidates=[a, x], thresh=0.05, celf=True)
    assert any(e.get("event") == "stop" and e.get("celf_note") for e in log)


def test_naive_admits_complements_with_mid_thresh_decoy():
    """Naive greedy recovers complementary x even when a low-marg decoy y is present."""
    short: dict = {}
    a = ResidualCandidate(src=("a",), tgt=("A",), template_id="t", domain="t")
    y = ResidualCandidate(src=("y",), tgt=("Y",), template_id="t", domain="t")
    x = ResidualCandidate(src=("x",), tgt=("X",), template_id="t", domain="t")

    def score(maps):
        s = 0.0
        if ("a",) in maps:
            s += 0.3
        if ("y",) in maps:
            s += 0.1  # small always-positive decoy
        if ("a",) in maps and ("x",) in maps:
            s += 0.5  # complementary
        return s

    fam = ResidualFamily(DomainAtoms(name="t"))
    m_naive, _ = fam.admit(
        short, score, candidates=[a, y, x], thresh=0.15, celf=False,
    )
    assert ("a",) in m_naive and ("x",) in m_naive
    assert score(m_naive) >= 0.8


def test_marker_cache_uses_content_not_object_id():
    """Two equal-content MapDicts must share cache; a different map must recompute."""
    fam = ResidualFamily(listops_domain_atoms(induce_only=True))
    m1 = {("a", "x2"): ["A", "A"], ("b", "x2"): ["B", "B"]}
    m2 = {("a", "x2"): ["A", "A"], ("b", "x2"): ["B", "B"]}  # equal content, new object
    m3 = {("a", "x3"): ["A", "A", "A"], ("b", "x3"): ["B", "B", "B"]}
    r1 = fam._cached_markers(m1)
    r2 = fam._cached_markers(m2)
    assert r1[0] == r2[0] and r1[0].get("x2") == 2
    r3 = fam._cached_markers(m3)
    assert r3[0].get("x3") == 3
    assert "x2" not in r3[0]


def test_admit_rejects_bad_residual_negative_control():
    short = {("a",): ["A"], ("b",): ["B"]}
    bad = ResidualCandidate(
        src=("c",), tgt=("WRONG",), template_id="poison", domain="t", meta={},
    )
    good = ResidualCandidate(
        src=("c",), tgt=("C",), template_id="nfold", domain="t", meta={},
    )
    val = [(["a"], ["A"]), (["c"], ["C"])]

    def score(maps):
        return sum(1 for cmd, gold in val if maps.get(tuple(cmd)) == gold) / len(val)

    fam = ResidualFamily(DomainAtoms(name="t"))
    maps, log = fam.admit(short, score, thresh=1e-4, candidates=[bad, good], celf=False)
    assert maps.get(("c",)) == ["C"]
    asserts = [e for e in log if e.get("event") == "admit"]
    assert len(asserts) == 1
    assert maps.get(("c",)) != ["WRONG"]


def test_celf_matches_naive_when_independent_singletons():
    short = {
        ("c", "x2"): ["C", "C"],
        ("e", "x2"): ["E", "E"],
        ("d", "x3"): ["D", "D", "D"],
        ("f", "x3"): ["F", "F", "F"],
        ("a",): ["A"],
    }
    val = [(["c"], ["C"]), (["d"], ["D"]), (["a"], ["A"])]

    def score(maps):
        return sum(1 for cmd, gold in val if maps.get(tuple(cmd)) == gold) / len(val)

    fam = ResidualFamily(listops_domain_atoms(induce_only=True))
    m1, _ = fam.admit(short, score, celf=True)
    m2, _ = fam.admit(short, score, celf=False)
    assert m1.get(("c",)) == ["C"] and m2.get(("c",)) == ["C"]
    assert m1.get(("d",)) == ["D"] and m2.get(("d",)) == ["D"]


def test_enabled_templates_filter():
    short = {
        ("c", "x2"): ["C", "C"],
        ("d", "x2"): ["D", "D"],
        ("run", "left"): ["I_TURN_LEFT", "I_RUN"],
    }
    dom = DomainAtoms(
        name="mixed",
        nfold_markers={"x2": 2},
        prefix_tokens=frozenset({"I_TURN_LEFT"}),
        enabled_templates=("nfold",),
        auto_induce_markers=False,
    )
    fam = ResidualFamily(dom)
    prop = fam.propose_map(short)
    assert ("c",) in prop
    assert ("run",) not in prop


def test_synth_templates_pack_smoke():
    short = {
        ("c", "x2"): ["C", "C"],
        ("d", "x2"): ["D", "D"],
    }
    fam = ResidualFamily(listops_domain_atoms(induce_only=True), templates=SYNTH_TEMPLATES)
    prop = fam.propose_map(short)
    assert ("c",) in prop


def test_relation_atom_from_multi_votes():
    from pil.residual_template import CFQ_TEMPLATES, cfq_domain_atoms

    base = {("marry", "ns:people.person.spouse"): ["ns:people.person.spouse"]}
    multi = {
        ("marry", "ns:people.person.spouse"): 5,
        ("influence", "ns:influence.influence_node.influenced"): 4,
        ("influence", "ns:people.person.spouse"): 1,  # below min_support 3
    }
    fam = ResidualFamily(
        cfq_domain_atoms(multi_word_path_votes=multi, atom_min_support=3),
        templates=CFQ_TEMPLATES,
    )
    cands = fam.propose(base)
    srcs = {c.src for c in cands}
    assert ("influence", "ns:influence.influence_node.influenced") in srcs
    assert ("marry", "ns:people.person.spouse") not in srcs  # already in base
    assert all(c.template_id == "relation_atom" for c in cands)
    assert all(c.pattern_agnostic for c in cands)


# --- honest provenance metrics -----------------------------------------------

def _rc(
    *,
    template_id: str,
    meta: dict | None = None,
    src: tuple[str, ...] = ("x",),
    tgt: tuple[str, ...] = ("X",),
) -> ResidualCandidate:
    return ResidualCandidate(
        src=src, tgt=tgt, template_id=template_id, domain="t", meta=meta or {},
    )


def test_candidate_provenance_nfold_induced():
    assert candidate_provenance(
        _rc(template_id="nfold", meta={"marker_induced": True}),
    ) == "induced"


def test_candidate_provenance_nfold_supplied():
    assert candidate_provenance(
        _rc(template_id="nfold", meta={"marker_induced": False}),
    ) == "supplied"


def test_candidate_provenance_structural():
    assert candidate_provenance(_rc(template_id="structural")) == "supplied"


def test_candidate_provenance_relation_atom():
    assert candidate_provenance(_rc(template_id="relation_atom")) == "template_fixed"


def test_candidate_provenance_prefix_body_induced():
    assert candidate_provenance(
        _rc(template_id="prefix_body", meta={"prefix_induced": True}),
    ) == "induced"


def test_candidate_provenance_prefix_body_supplied():
    assert candidate_provenance(
        _rc(template_id="prefix_body", meta={"prefix_induced": False}),
    ) == "supplied"


def test_candidate_provenance_rewrite_synth():
    assert candidate_provenance(_rc(template_id="rewrite_synth")) == "induced"


def test_cfq_agnostic_overstates_vs_honest_induced():
    """relation_atom is pattern-agnostic but template_fixed — honesty correction."""
    base = {("marry", "ns:people.person.spouse"): ["ns:people.person.spouse"]}
    multi = {
        ("influence", "ns:influence.influence_node.influenced"): 4,
        ("directed", "ns:film.director.film"): 5,
    }
    fam = ResidualFamily(
        cfq_domain_atoms(multi_word_path_votes=multi, atom_min_support=3),
        templates=CFQ_TEMPLATES,
    )
    cands = fam.propose(base)
    assert cands  # at least one relation_atom residual
    diag = fam.diagnostics(base, admitted_src=[c.src for c in cands])
    assert diag["frac_admitted_agnostic"] == 1.0
    assert diag["frac_admitted_induced"] == 0.0
    assert diag["provenance_admitted"].get("template_fixed", 0) == len(cands)


def test_listops_frac_admitted_induced_is_one():
    short = {
        ("c", "x2"): ["C", "C"],
        ("d", "x2"): ["D", "D"],
        ("a",): ["A"],
    }
    fam = ResidualFamily(listops_domain_atoms(induce_only=True))
    cands = fam.propose(short)
    nfold = [c for c in cands if c.template_id == "nfold"]
    assert nfold
    assert all(c.meta.get("marker_induced") for c in nfold)
    diag = fam.diagnostics(short, admitted_src=[c.src for c in nfold])
    assert diag["frac_admitted_induced"] == 1.0


def test_scan_domain_atoms_induce_only_flags():
    ind = scan_domain_atoms(induce_only=True)
    assert ind.prefix_tokens == frozenset()
    assert ind.auto_induce_prefixes is True
    assert ind.nfold_markers == {}

    hand = scan_domain_atoms()  # default: byte-identical hand pack
    assert hand.prefix_tokens == frozenset({"I_TURN_LEFT", "I_TURN_RIGHT"})
    assert hand.auto_induce_prefixes is False
    assert hand.nfold_markers == {"twice": 2, "thrice": 3}


def test_prefix_induced_meta_under_auto_induce():
    """With empty hand prefixes + auto_induce, prefix_induced is True on recovered prefs."""
    maps = {
        ("run", "left"): ["I_TURN_LEFT", "I_RUN"],
        ("walk", "left"): ["I_TURN_LEFT", "I_WALK"],
        ("look", "right"): ["I_TURN_RIGHT", "I_LOOK"],
        ("jump", "right"): ["I_TURN_RIGHT", "I_JUMP"],
    }
    dom = DomainAtoms(
        name="t",
        prefix_tokens=frozenset(),
        auto_induce_prefixes=True,
        auto_induce_markers=False,
    )
    cands = PrefixBodyTemplate().propose(maps, dom)
    assert cands
    assert all(c.meta.get("prefix_induced") is True for c in cands)
    # also recoverable via scan induce_only pack path
    fam = ResidualFamily(scan_domain_atoms(induce_only=True))
    scan_cands = [
        c for c in fam.propose(maps)
        if c.template_id == "prefix_body"
    ]
    assert scan_cands
    assert all(c.meta.get("prefix_induced") is True for c in scan_cands)
