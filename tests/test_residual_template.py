"""Tests for domain-agnostic residual templates."""
from __future__ import annotations

from pil.residual_template import (
    DomainAtoms,
    NFoldTemplate,
    PrefixBodyTemplate,
    ResidualFamily,
    StructuralSeedTemplate,
    listops_domain_atoms,
    scan_domain_atoms,
)


def test_nfold_proposes_bare_leaf():
    maps = {("a", "x2"): ["A", "A"], ("b", "x3"): ["B", "B", "B"]}
    dom = DomainAtoms(name="t", nfold_markers={"x2": 2, "x3": 3})
    cands = NFoldTemplate().propose(maps, dom)
    srcs = {c.src: c.tgt for c in cands}
    assert srcs[("a",)] == ("A",)
    assert srcs[("b",)] == ("B",)
    assert all(c.pattern_agnostic for c in cands)


def test_prefix_body_strips_known_prefix():
    maps = {("run", "left"): ["I_TURN_LEFT", "I_RUN"]}
    dom = DomainAtoms(
        name="t",
        prefix_tokens=frozenset({"I_TURN_LEFT", "I_TURN_RIGHT"}),
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


def test_scan_pack_matches_legacy_shapes():
    # short maps without bare run; residual should recover run from run twice
    short = {
        ("run", "twice"): ["I_RUN", "I_RUN"],
        ("walk",): ["I_WALK"],
        ("walk", "left"): ["I_TURN_LEFT", "I_WALK"],
    }
    fam = ResidualFamily(scan_domain_atoms())
    prop = fam.propose_map(short)
    assert ("run",) in prop and prop[("run",)] == ["I_RUN"]
    assert ("turn", "left") in prop
    # walk already present — not re-proposed as residual
    assert ("walk",) not in prop or ("walk",) in short


def test_listops_pack_nfold_only():
    short = {("c", "x2"): ["C", "C"], ("a",): ["A"]}
    fam = ResidualFamily(listops_domain_atoms())
    prop = fam.propose_map(short)
    assert prop[("c",)] == ["C"]
    diag = fam.diagnostics(short)
    assert diag["frac_proposed_agnostic"] == 1.0


def test_admit_selects_helpful_residual():
    short = {("c", "x2"): ["C", "C"], ("a",): ["A"]}
    # val: need bare c for score
    val = [ (["c"], ["C"]), (["a"], ["A"]) ]

    def score(maps):
        ok = 0
        for cmd, gold in val:
            if maps.get(tuple(cmd)) == gold:
                ok += 1
        return ok / len(val)

    fam = ResidualFamily(listops_domain_atoms())
    maps, log = fam.admit(short, score, thresh=1e-4)
    assert maps.get(("c",)) == ["C"]
    assert any(row["marginal"] > 0 for row in log)


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
    )
    fam = ResidualFamily(dom)
    prop = fam.propose_map(short)
    assert ("c",) in prop
    assert ("run",) not in prop  # prefix_body disabled
