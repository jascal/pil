"""Tests for residual templates: induction, CELF caveats, honesty guards."""
from __future__ import annotations

from pil.residual_template import (
    SYNTH_TEMPLATES,
    DomainAtoms,
    NFoldTemplate,
    PrefixBodyTemplate,
    ResidualCandidate,
    ResidualFamily,
    RewriteSynthesizer,
    StructuralSeedTemplate,
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
    Default (naive) must admit both. CELF is opt-in and may approximate.
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
    # CELF with stale-refresh should also recover both; not Leskovec-guaranteed
    m_celf, _ = fam.admit(short, score, candidates=[a, x], thresh=0.05, celf=True)
    assert score(m_celf) == 0.8


def test_celf_stop_without_stale_refresh_would_miss_x():
    """Document why CELF is opt-in: a mid-thresh 'decoy' can bury complementary leaves
    if stale heap entries are discarded without re-score (guarded in _admit_celf)."""
    short: dict = {}
    a = ResidualCandidate(src=("a",), tgt=("A",), template_id="t", domain="t")
    y = ResidualCandidate(src=("y",), tgt=("Y",), template_id="t", domain="t")
    x = ResidualCandidate(src=("x",), tgt=("X",), template_id="t", domain="t")

    def score(maps):
        s = 0.0
        if ("a",) in maps:
            s += 0.3
        if ("y",) in maps:
            s += 0.1  # small always-positive
        if ("a",) in maps and ("x",) in maps:
            s += 0.5  # complementary
        return s

    fam = ResidualFamily(DomainAtoms(name="t"))
    # thresh between y's marg (0.1) and x's post-a marg (0.5): naive admits a,x (and y)
    m_naive, _ = fam.admit(
        short, score, candidates=[a, y, x], thresh=0.15, celf=False,
    )
    assert ("a",) in m_naive and ("x",) in m_naive
    assert score(m_naive) >= 0.8


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
