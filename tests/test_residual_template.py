"""Tests for domain-agnostic residual templates + marker induction + CELF."""
from __future__ import annotations

from pil.residual_template import (
    DomainAtoms,
    NFoldTemplate,
    PrefixBodyTemplate,
    ResidualCandidate,
    ResidualFamily,
    RewriteSynthesizer,
    StructuralSeedTemplate,
    induce_nfold_markers,
    listops_domain_atoms,
    scan_domain_atoms,
)


def test_induce_nfold_markers_no_hand_markers():
    maps = {
        ("a", "x2"): ["A", "A"],
        ("b", "x2"): ["B", "B"],
        ("c", "x3"): ["C", "C", "C"],
    }
    m = induce_nfold_markers(maps)
    assert m["x2"] == 2
    assert m["x3"] == 3


def test_nfold_with_empty_markers_auto_induce():
    maps = {("a", "x2"): ["A", "A"]}
    dom = DomainAtoms(name="t", nfold_markers={}, auto_induce_markers=True)
    cands = NFoldTemplate().propose(maps, dom)
    assert len(cands) == 1
    assert cands[0].src == ("a",)
    assert cands[0].meta.get("marker_induced") is True


def test_nfold_supplied_overrides_induction():
    # inconsistent induction would pick k=2; force k=4 via supplied → no exact match → no cand
    maps = {("a", "m"): ["A", "A"]}  # true 2-fold
    dom = DomainAtoms(name="t", nfold_markers={"m": 4}, auto_induce_markers=True)
    cands = NFoldTemplate().propose(maps, dom)
    assert cands == []  # forced k=4 does not fit


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
    maps = {("c", "x2"): ["C", "C"]}
    dom = DomainAtoms(name="t", nfold_markers={}, auto_induce_markers=True)
    cands = RewriteSynthesizer().propose(maps, dom)
    assert any(c.src == ("c",) and c.tgt == ("C",) for c in cands)


def test_scan_pack_induce_only():
    short = {
        ("run", "twice"): ["I_RUN", "I_RUN"],
        ("walk",): ["I_WALK"],
        ("walk", "left"): ["I_TURN_LEFT", "I_WALK"],
    }
    fam = ResidualFamily(scan_domain_atoms(induce_only=True))
    prop = fam.propose_map(short)
    assert prop[("run",)] == ["I_RUN"]
    assert ("turn", "left") in prop


def test_listops_induce_only_default():
    short = {("c", "x2"): ["C", "C"], ("a",): ["A"]}
    fam = ResidualFamily(listops_domain_atoms(induce_only=True))
    assert fam.domain.nfold_markers == {}
    prop = fam.propose_map(short)
    assert prop[("c",)] == ["C"]
    assert fam.diagnostics(short)["induced_nfold_markers"]["x2"] == 2


def test_admit_selects_helpful_residual():
    short = {("c", "x2"): ["C", "C"], ("a",): ["A"]}
    val = [(["c"], ["C"]), (["a"], ["A"])]

    def score(maps):
        ok = 0
        for cmd, gold in val:
            if maps.get(tuple(cmd)) == gold:
                ok += 1
        return ok / len(val)

    fam = ResidualFamily(listops_domain_atoms(induce_only=True))
    maps, log = fam.admit(short, score, thresh=1e-4, celf=True)
    assert maps.get(("c",)) == ["C"]
    admits = [e for e in log if e.get("event") == "admit"]
    assert len(admits) == 1
    # no per-candidate spam: only admit rows + stop
    assert all(e.get("event") in ("admit", "stop") for e in log)


def test_admit_rejects_bad_residual_negative_control():
    """Val-marginal admit must refuse a residual that hurts exact match."""
    short = {("a",): ["A"], ("b",): ["B"]}
    # poisoned new leaf: c → wrong (hurts val that wants C)
    bad = ResidualCandidate(
        src=("c",), tgt=("WRONG",), template_id="poison",
        domain="t", pattern_agnostic=True, meta={},
    )
    good = ResidualCandidate(
        src=("c",), tgt=("C",), template_id="nfold",
        domain="t", pattern_agnostic=True, meta={},
    )
    val = [(["a"], ["A"]), (["c"], ["C"])]

    def score(maps):
        ok = 0
        for cmd, gold in val:
            if maps.get(tuple(cmd)) == gold:
                ok += 1
        return ok / len(val)

    fam = ResidualFamily(DomainAtoms(name="t"))
    # both compete for same src; CELF/naive should pick good (positive marg) not bad
    maps, log = fam.admit(
        short, score, thresh=1e-4, candidates=[bad, good], celf=True,
    )
    assert maps.get(("c",)) == ["C"]
    asserts = [e for e in log if e.get("event") == "admit"]
    assert len(asserts) == 1
    assert asserts[0]["src"] == "c"
    # bad has negative marginal vs base — never admitted
    assert maps.get(("c",)) != ["WRONG"]


def test_celf_matches_naive_on_small():
    short = {
        ("c", "x2"): ["C", "C"],
        ("d", "x3"): ["D", "D", "D"],
        ("a",): ["A"],
    }
    val = [(["c"], ["C"]), (["d"], ["D"]), (["a"], ["A"])]

    def score(maps):
        return sum(
            1 for cmd, gold in val if maps.get(tuple(cmd)) == gold
        ) / len(val)

    fam = ResidualFamily(listops_domain_atoms(induce_only=True))
    m1, _ = fam.admit(short, score, celf=True)
    m2, _ = fam.admit(short, score, celf=False)
    assert m1.get(("c",)) == ["C"] and m2.get(("c",)) == ["C"]
    assert m1.get(("d",)) == ["D"] and m2.get(("d",)) == ["D"]


def test_enabled_templates_filter():
    short = {
        ("c", "x2"): ["C", "C"],
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
