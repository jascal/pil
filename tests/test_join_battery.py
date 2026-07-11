"""Unit tests for the synthetic join battery (generator + campaign machinery)."""
# ruff: noqa: E501
from __future__ import annotations

import inspect
import sys
from collections import Counter
from pathlib import Path

from pil.cfq_edges import parse_sparql_joins
from pil.join_battery import (
    BatteryExample,
    build_regime_s_schema,
    generate_regime,
    planted_join_signatures,
    pred_name,
    self_certifies,
    type_merge_signatures,
    type_name,
    world_counts,
)
from pil.residual_template import DomainAtoms, ResidualCandidate, ResidualFamily

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # experiments/ has no __init__.py

from experiments.campaign_join_battery import (  # noqa: E402
    atom_fires,
    build_inventory_ceiling,
    canon_sig,
    comp_bound,
    expand,
    make_val_score,
    parse_sig,
    permute_gold_lists,
    predict_inventory_ceiling,
)

# ---------------------------------------------------------------------------
# (a) Generator determinism + CFQ-format parseability
# ---------------------------------------------------------------------------


def _ex_tuple(ex: BatteryExample) -> tuple:
    return (ex.question, ex.sparql, ex.topology, ex.key)


def _split_tuples(split) -> tuple:
    return (
        tuple(_ex_tuple(e) for e in split.train),
        tuple(_ex_tuple(e) for e in split.val),
        tuple(_ex_tuple(e) for e in split.test_iid),
        tuple(_ex_tuple(e) for e in split.test_comp),
    )


def test_generator_determinism_s_l_n():
    """generate_regime(regime, seed=0) twice yields identical splits for S/L/N."""
    for regime in ("S", "L", "N"):
        a = generate_regime(regime, seed=0)
        b = generate_regime(regime, seed=0)
        assert _split_tuples(a) == _split_tuples(b), f"nondeterministic regime {regime}"


def test_cfq_format_parseable_and_has_join_vars():
    """Sample >=200 examples/regime: parse_sparql_joins succeeds and n_join_vars>=1."""
    for regime in ("S", "L", "N"):
        split = generate_regime(regime, seed=0)
        pool = list(split.train) + list(split.val) + list(split.test_iid) + list(split.test_comp)
        # Take a strided sample of at least 200 (or all if smaller)
        if len(pool) >= 200:
            sample = pool[:: max(1, len(pool) // 200)][:200]
            assert len(sample) >= 200 or len(sample) == 200
            # Ensure we hit 200 when possible
            if len(sample) < 200:
                sample = pool[:200]
            sample = pool[:200] if len(pool) >= 200 else pool
        else:
            sample = pool
        assert len(sample) >= 200
        for ex in sample:
            jq = parse_sparql_joins(ex.sparql)
            assert jq.n_join_vars >= 1, ex.sparql


# ---------------------------------------------------------------------------
# (b) Regime S self-certification + rejection path
# ---------------------------------------------------------------------------


def test_regime_s_self_certification_on_sample():
    """Planted signature multiset equals type-merge closure on a sample of S."""
    split = generate_regime("S", seed=0)
    rs = build_regime_s_schema(0)
    sample = split.train[:50] + split.val[:20] + split.test_iid[:20]
    for ex in sample:
        jq = parse_sparql_joins(ex.sparql)
        # Rebuild edges from SPARQL body for type-merge check against schema
        first = ex.sparql.find("{")
        last = ex.sparql.rfind("}")
        body = ex.sparql[first + 1 : last]
        edges = []
        var_types: dict[str, str] = {}
        for line in body.split("\n"):
            line = line.strip()
            if line.endswith("."):
                line = line[:-1].strip()
            if not line:
                continue
            parts = line.split()
            assert len(parts) == 3
            s, p, o = parts
            if p == "a":
                var_types[s] = o
            else:
                edges.append((s, p, o))
        planted = planted_join_signatures(edges, var_types)
        merged = type_merge_signatures(edges, rs.schema)
        assert Counter(planted) == Counter(merged)
        assert Counter(planted) == Counter(jq.signatures)


def test_regime_s_rejection_sampling_on_type_collision():
    """Deliberate center-type mismatch fails self_certifies / star builder path.

    If rejection-sampling were deleted and colliding queries emitted, this
    would no longer distinguish valid from invalid constructions.
    """
    rs = build_regime_s_schema(0)
    # Find two star-role preds with DIFFERENT subj types, and one more
    star_pools = rs.role_pools["star"]
    # Collect by subj type
    by_type: dict[str, list[tuple[int, str]]] = {}
    for ri, pool in enumerate(star_pools):
        for p in pool:
            st = rs.schema[p][0]
            by_type.setdefault(st, []).append((ri, p))
    types = sorted(by_type.keys())
    assert len(types) >= 2, "schema must mix star center types for rejection"
    # Pick preds from two different types across the three roles if possible
    # Force a colliding triple: take p0,p1 from type A roles, p2 from type B
    t_a, t_b = types[0], types[1]
    # Build a triple with mismatched center types by hand
    p_from: dict[int, str] = {}
    for ri, p in by_type[t_a]:
        if ri not in p_from:
            p_from[ri] = p
    for ri, p in by_type[t_b]:
        if ri not in p_from and len(p_from) < 3:
            p_from[ri] = p
    # Fill remaining roles from any pool
    for ri in range(3):
        if ri not in p_from:
            p_from[ri] = star_pools[ri][0]
    p0, p1, p2 = p_from[0], p_from[1], p_from[2]
    st0, st1, st2 = rs.schema[p0][0], rs.schema[p1][0], rs.schema[p2][0]
    if st0 == st1 == st2:
        # Force collision by overriding schema view for the certify check
        # Construct edges as if they share a center var but types disagree
        bad_schema = dict(rs.schema)
        bad_schema[p0] = (type_name(0), type_name(10))
        bad_schema[p1] = (type_name(1), type_name(10))  # different center type
        bad_schema[p2] = (type_name(0), type_name(10))
        edges = [
            ("?x0", p0, "M0"),
            ("?x0", p1, "M1"),
            ("?x0", p2, "M2"),
        ]
        # Planted says one var of type t0 with 3 incidences
        var_types = {"?x0": type_name(0)}
        # Type-merge groups by required type → (p0,subj)+(p2,subj) vs (p1,subj)
        # → under-merge relative to planted 3-star → must fail certify
        assert not self_certifies(edges, var_types, bad_schema)
        planted = planted_join_signatures(edges, var_types)
        merged = type_merge_signatures(edges, bad_schema)
        assert Counter(planted) != Counter(merged)
    else:
        # Natural mismatch: star builder returns None
        from pil.join_battery import _build_star_s

        assert _build_star_s(p0, p1, p2, rs.schema, 0) is None
        edges = [
            ("?x0", p0, "M0"),
            ("?x0", p1, "M1"),
            ("?x0", p2, "M2"),
        ]
        # Even if we force var_types to one type, schema merge diverges
        var_types = {"?x0": st0}
        assert not self_certifies(edges, var_types, rs.schema)


# ---------------------------------------------------------------------------
# (c) Regime L disambiguation + Regime N control
# ---------------------------------------------------------------------------


def test_regime_l_topo_token_predicts_topology():
    """For a key with both topologies, topoA/topoB deterministically labels topo."""
    split = generate_regime("L", seed=0)
    by_key: dict[tuple[str, ...], list[BatteryExample]] = {}
    for block in (split.train, split.val, split.test_iid, split.test_comp):
        for ex in block:
            by_key.setdefault(ex.key, []).append(ex)
    dual_key = None
    for k, rows in by_key.items():
        tops = {r.topology for r in rows}
        if tops >= {"star", "chain"}:
            dual_key = k
            break
    assert dual_key is not None, "L must plant both topologies for some key"
    for ex in by_key[dual_key]:
        toks = set(ex.question.split())
        if ex.topology == "star":
            assert "topoA" in toks and "topoB" not in toks
        else:
            assert "topoB" in toks and "topoA" not in toks


def test_regime_n_both_topos_and_no_disambig_token():
    """N train has a dual-topology key; no example carries topoA/topoB."""
    split = generate_regime("N", seed=0)
    by_key: dict[tuple[str, ...], set[str]] = {}
    for ex in split.train:
        by_key.setdefault(ex.key, set()).add(ex.topology)
        toks = ex.question.split()
        assert "topoA" not in toks
        assert "topoB" not in toks
    dual = [k for k, tops in by_key.items() if tops >= {"star", "chain"}]
    assert dual, "N train must have at least one key with both topologies"
    for block in (split.val, split.test_iid):
        for ex in block:
            toks = ex.question.split()
            assert "topoA" not in toks
            assert "topoB" not in toks
    assert split.test_comp == []


# ---------------------------------------------------------------------------
# (d) Compositional disjointness + predicate cover
# ---------------------------------------------------------------------------


def test_comp_keys_disjoint_and_pred_cover():
    """test_comp keys disjoint from train; every individual pred appears in train."""
    for regime in ("S", "L"):
        split = generate_regime(regime, seed=0)
        train_keys = {ex.key for ex in split.train}
        comp_keys = {ex.key for ex in split.test_comp}
        assert train_keys.isdisjoint(comp_keys), f"{regime}: train/comp key overlap"
        train_preds = {p for ex in split.train for p in ex.key}
        all_preds = set()
        for block in (split.train, split.val, split.test_iid, split.test_comp):
            for ex in block:
                all_preds.update(ex.key)
        missing = all_preds - train_preds
        assert not missing, f"{regime}: preds missing from train: {sorted(missing)[:5]}"


# ---------------------------------------------------------------------------
# (e) canon_sig / parse_sig round-trip
# ---------------------------------------------------------------------------


def test_canon_sig_parse_sig_roundtrip():
    """parse_sig(canon_sig(x)) == x for nested JoinSignature shapes + empty incs."""
    cases = [
        ("ns:syn.t3", (("ns:syn.p1", "subj"), ("ns:syn.p9", "obj"))),
        ("VAR", ()),
        ("ns:syn.t0", (("ns:syn.p0", "subj"), ("ns:syn.p0", "obj"), ("ns:syn.p1", "subj"))),
        (),  # empty tuple edge case
        ("ns:syn.t1", (("ns:syn.p2", "obj"),)),
    ]
    for x in cases:
        assert parse_sig(canon_sig(x)) == x


# ---------------------------------------------------------------------------
# (f) expand: multiset-MAX, within-atom multiplicity, firing rules
# ---------------------------------------------------------------------------


def test_expand_multiset_max_not_sum():
    """Two firing atoms proposing the same sig → multiplicity 1 (max, not sum)."""
    sig = canon_sig(("t", (("p1", "subj"), ("p2", "subj"))))
    maps = {
        ("JOINKEY", "ns:syn.p1|ns:syn.p2|ns:syn.p3"): [sig],
        ("WORD", "w1"): [sig],
    }
    key = ("ns:syn.p1", "ns:syn.p2", "ns:syn.p3")
    tokens = {"w1", "w2", "w3"}
    out = expand(maps, tokens, key)
    assert out.count(sig) == 1
    assert len(out) == 1


def test_expand_preserves_within_atom_multiplicity():
    """One atom with tgt=(sig,sig) → output multiplicity 2."""
    sig = canon_sig(("t", (("p1", "subj"), ("p2", "obj"))))
    maps = {
        ("JOINKEY", "ns:syn.p1|ns:syn.p2|ns:syn.p3"): [sig, sig],
    }
    key = ("ns:syn.p1", "ns:syn.p2", "ns:syn.p3")
    out = expand(maps, set(), key)
    assert out.count(sig) == 2


def test_expand_firing_joinkey_word_sig_sigw():
    """Per-kind firing: JOINKEY exact, WORD membership, SIG subset, SIGW both."""
    p1, p2, p3 = pred_name(1), pred_name(2), pred_name(3)
    p9 = pred_name(9)
    key = (p1, p2, p3)
    tokens = {"w1", "w2", "topoA"}
    sig_ok = ("ns:syn.t0", ((p1, "subj"), (p2, "subj")))
    sig_bad = ("ns:syn.t0", ((p9, "subj"), (p1, "subj")))  # needs p9 absent from key
    cs_ok = canon_sig(sig_ok)
    cs_bad = canon_sig(sig_bad)

    # JOINKEY exact match
    assert atom_fires(("JOINKEY", f"{p1}|{p2}|{p3}"), tokens, key)
    assert not atom_fires(("JOINKEY", f"{p1}|{p2}|{p9}"), tokens, key)  # near-miss

    # WORD membership
    assert atom_fires(("WORD", "w1"), tokens, key)
    assert not atom_fires(("WORD", "missing"), tokens, key)

    # SIG multiset-subset
    assert atom_fires(("SIG", cs_ok), tokens, key)
    assert not atom_fires(("SIG", cs_bad), tokens, key)

    # SIGW both conditions
    assert atom_fires(("SIGW", "w1", cs_ok), tokens, key)
    assert not atom_fires(("SIGW", "missing", cs_ok), tokens, key)  # word alone fails
    assert not atom_fires(("SIGW", "w1", cs_bad), tokens, key)  # sig alone fails

    # expand integration for JOINKEY near-miss: should not contribute
    maps = {("JOINKEY", f"{p1}|{p2}|{p9}"): [cs_ok]}
    assert expand(maps, tokens, key) == []
    maps_ok = {("JOINKEY", f"{p1}|{p2}|{p3}"): [cs_ok]}
    assert expand(maps_ok, tokens, key) == [cs_ok]


# ---------------------------------------------------------------------------
# (g) Admission smoke test
# ---------------------------------------------------------------------------


def test_admission_smoke_useful_vs_useless_sig_atom():
    """Useful SIG atom is admitted with positive marginal; useless atom is not."""
    # Val rows: two queries that share a join signature involving p1,p2
    p1, p2, p3 = pred_name(1), pred_name(2), pred_name(3)
    p8, p9 = pred_name(8), pred_name(9)

    def mk_star(preds: tuple[str, str, str], q: str) -> BatteryExample:
        a, b, c = preds
        sparql = (
            "SELECT count(*) WHERE {\n"
            f"?x0 {a} ?x1 .\n"
            f"?x0 {b} ?x2 .\n"
            f"?x0 {c} ?x3 .\n"
            f"?x0 a {type_name(0)} .\n"
            f"?x1 a {type_name(0)} .\n"
            f"?x2 a {type_name(0)} .\n"
            f"?x3 a {type_name(0)} .\n"
            "}"
        )
        key = tuple(sorted(preds))
        return BatteryExample(question=q, sparql=sparql, topology="star", key=key)

    # Gold sig for star center: type t0, three subj incidences
    # Useful atom proposes the actual gold signature for val rows with p1,p2,p3
    val = [
        mk_star((p1, p2, p3), "w1 w2 w3"),
        mk_star((p1, p2, p3), "w1 w2 w3"),
    ]
    # Confirm gold
    gold = parse_sparql_joins(val[0].sparql).signatures
    assert len(gold) == 1
    useful_sig = gold[0]
    cs = canon_sig(useful_sig)

    useful = ResidualCandidate(
        src=("SIG", cs),
        tgt=(cs,),
        template_id="join_sig",
        domain="join_battery",
    )
    # Useless: signature needing p8,p9 never present on val keys
    bad_sig = (type_name(0), ((p8, "subj"), (p9, "subj")))
    useless = ResidualCandidate(
        src=("SIG", canon_sig(bad_sig)),
        tgt=(canon_sig(bad_sig),),
        template_id="join_sig",
        domain="join_battery",
    )

    fam = ResidualFamily(domain=DomainAtoms(name="join_battery"))
    val_score = make_val_score(val, [useful, useless])
    # Useful alone should improve score over empty
    assert val_score({useful.src: list(useful.tgt)}) > val_score({})

    maps_adm, admit_log = fam.admit(
        {},
        val_score,
        thresh=1e-4,
        max_rules=64,
        celf=False,
        candidates=[useful, useless],
    )
    assert useful.src in maps_adm
    assert useless.src not in maps_adm
    admit_events = [e for e in admit_log if e.get("event") == "admit"]
    assert any(e.get("marginal", 0) > 0 for e in admit_events)
    # Useful src string appears in an admit event
    useful_src_str = " ".join(useful.src)
    assert any(e.get("src") == useful_src_str for e in admit_events)


# ---------------------------------------------------------------------------
# (h) Leak guard structural test
# ---------------------------------------------------------------------------


def test_expand_leak_guard_no_gold_parameter():
    """expand has no gold parameter; identical (maps,tokens,key) → identical out."""
    sig = inspect.signature(expand)
    param_names = set(sig.parameters)
    assert "gold" not in param_names
    assert "sparql" not in param_names
    assert param_names == {"maps", "question_tokens", "key"}

    p1, p2, p3 = pred_name(1), pred_name(2), pred_name(3)
    cs = canon_sig(("t", ((p1, "subj"), (p2, "subj"))))
    maps = {("JOINKEY", f"{p1}|{p2}|{p3}"): [cs]}
    tokens = {"w1"}
    key = (p1, p2, p3)
    out1 = expand(maps, tokens, key)
    # Different hypothetical gold must not change expand (gold is never an input)
    _hypothetical_gold_a = [cs]
    _hypothetical_gold_b = [canon_sig(("other", ()))]
    out2 = expand(maps, tokens, key)
    assert out1 == out2 == [cs]
    assert _hypothetical_gold_a != _hypothetical_gold_b  # gold differed, output did not


# ---------------------------------------------------------------------------
# (i) Expressibility invariant (budget fit)
# ---------------------------------------------------------------------------


def test_expressibility_invariant_world_counts():
    """n_distinct_sigs <= 100 and n_distinct_keys <= 250 for S/L/N at seed 0.

    Must check the returned counts against the thresholds (not merely that
    generate_regime did not raise) so a future budget-breaking resize fails
    this test loudly.
    """
    for regime in ("S", "L", "N"):
        split = generate_regime(regime, seed=0)
        counts = world_counts(split)
        assert counts["n_distinct_sigs"] <= 100, (
            f"regime {regime}: n_distinct_sigs={counts['n_distinct_sigs']} > 100"
        )
        assert counts["n_distinct_keys"] <= 250, (
            f"regime {regime}: n_distinct_keys={counts['n_distinct_keys']} > 250"
        )


# ---------------------------------------------------------------------------
# (j) Permutation-path admission (distinct keys; gold_override)
# ---------------------------------------------------------------------------


def test_permutation_admission_admits_zero_on_shuffled_gold():
    """Useful SIG atom admitted on real val; 0 admits under permuted gold.

    Uses >=2 distinctly-keyed val rows (different predicate triples / gold
    signatures) so permuting gold across rows is a meaningful control.
    Three rows are used so Random(1234).shuffle is a non-identity perm
    (on n=2 that seed is a no-op). Calls ``make_val_score(...,
    gold_override=...)`` and ``permute_gold_lists`` directly — focused unit
    test of the permutation-aware scoring path, not an end-to-end
    run_arm/run_regime integration.
    """
    p1, p2, p3 = pred_name(1), pred_name(2), pred_name(3)
    p4, p5, p6 = pred_name(4), pred_name(5), pred_name(6)
    p7, p10, p11 = pred_name(7), pred_name(10), pred_name(11)
    p8, p9 = pred_name(8), pred_name(9)

    def mk_star(preds: tuple[str, str, str], q: str) -> BatteryExample:
        a, b, c = preds
        sparql = (
            "SELECT count(*) WHERE {\n"
            f"?x0 {a} ?x1 .\n"
            f"?x0 {b} ?x2 .\n"
            f"?x0 {c} ?x3 .\n"
            f"?x0 a {type_name(0)} .\n"
            f"?x1 a {type_name(0)} .\n"
            f"?x2 a {type_name(0)} .\n"
            f"?x3 a {type_name(0)} .\n"
            "}"
        )
        key = tuple(sorted(preds))
        return BatteryExample(question=q, sparql=sparql, topology="star", key=key)

    # Three DISTINCTLY-keyed val rows → three different gold signatures
    val = [
        mk_star((p1, p2, p3), "w1 w2 w3"),
        mk_star((p4, p5, p6), "w4 w5 w6"),
        mk_star((p7, p10, p11), "w7 w10 w11"),
    ]
    assert len({ex.key for ex in val}) == 3
    golds = [parse_sparql_joins(ex.sparql).signatures for ex in val]
    assert all(len(g) == 1 for g in golds)
    assert len({g[0] for g in golds}) == 3
    css = [canon_sig(g[0]) for g in golds]

    usefuls = [
        ResidualCandidate(
            src=("SIG", cs),
            tgt=(cs,),
            template_id="join_sig",
            domain="join_battery",
        )
        for cs in css
    ]
    bad_sig = (type_name(0), ((p8, "subj"), (p9, "subj")))
    useless = ResidualCandidate(
        src=("SIG", canon_sig(bad_sig)),
        tgt=(canon_sig(bad_sig),),
        template_id="join_sig",
        domain="join_battery",
    )
    cands = usefuls + [useless]

    fam = ResidualFamily(domain=DomainAtoms(name="join_battery"))

    # --- Unpermuted path: useful atoms admitted with positive marginal ---
    val_score = make_val_score(val, cands)
    assert val_score({usefuls[0].src: list(usefuls[0].tgt)}) > val_score({})
    maps_real, log_real = fam.admit(
        {},
        val_score,
        thresh=1e-4,
        max_rules=64,
        celf=False,
        candidates=cands,
    )
    for u in usefuls:
        assert u.src in maps_real
    assert useless.src not in maps_real
    admit_real = [e for e in log_real if e.get("event") == "admit"]
    assert len(admit_real) >= 1
    assert any(e.get("marginal", 0) > 0 for e in admit_real)

    # --- Permuted path: shuffle gold across the distinct rows (seed 1234) ---
    # Features (tokens/key) stay put; gold multisets move.
    from experiments.campaign_join_battery import gold_serialized

    real_gold = [gold_serialized(ex) for ex in val]
    perm_gold = permute_gold_lists(val, seed=1234)
    assert perm_gold != real_gold, "seed 1234 must nontrivially permute n=3"
    val_score_perm = make_val_score(val, cands, gold_override=perm_gold)
    maps_perm, log_perm = fam.admit(
        {},
        val_score_perm,
        thresh=1e-4,
        max_rules=64,
        celf=False,
        candidates=cands,
    )
    admit_perm = [e for e in log_perm if e.get("event") == "admit"]
    # Required assertion: permuted-path admission yields 0 admitted atoms
    assert len(admit_perm) == 0, (
        f"expected 0 admits under permuted gold, got {len(admit_perm)}: "
        f"{admit_perm}"
    )
    assert maps_perm == {}


# ---------------------------------------------------------------------------
# (k) Expressibility bound (comp_bound unit tests)
# ---------------------------------------------------------------------------


def test_comp_bound_all_none_and_middle():
    """comp_bound: all-in ⇒ 1.0; none-in ⇒ 0.0; middle hand-computed case."""
    # Concrete signature tuples (shape doesn't matter to comp_bound beyond
    # canon_sig membership of their canon strings).
    s_a = (type_name(0), ((pred_name(1), "subj"), (pred_name(2), "subj")))
    s_b = (type_name(0), ((pred_name(3), "subj"), (pred_name(4), "subj")))
    s_c = (type_name(1), ((pred_name(5), "obj"), (pred_name(6), "obj")))
    ca, cb, cc = canon_sig(s_a), canon_sig(s_b), canon_sig(s_c)

    # (i) ALL gold signatures' canon_sig strings ARE in train_sig_set ⇒ 1.0
    train_all = {ca, cb, cc}
    gold_all = [s_a, s_b, s_c]
    assert comp_bound(gold_all, train_all) == 1.0

    # (ii) NONE of them are in train_sig_set ⇒ 0.0
    train_none: set[str] = {canon_sig((type_name(9), ()))}
    assert comp_bound(gold_all, train_none) == 0.0

    # (iii) Middle case: gold has 3 elements, 1 of which is in the inventory
    # ⇒ k=1, |gold|=3 ⇒ bound = 2*1/(1+3) = 0.5
    train_mid = {ca}  # only s_a is "seen in train"
    gold_mid = [s_a, s_b, s_c]  # 3 elements, 1 hit
    # arithmetic: k=1, |gold|=3, bound = 2k/(k+|gold|) = 2/4 = 0.5
    assert comp_bound(gold_mid, train_mid) == 0.5


# ---------------------------------------------------------------------------
# (l) Inventory-ceiling predictor
# ---------------------------------------------------------------------------


def test_inventory_ceiling_lookup_unseen_and_max_not_sum():
    """build/predict inventory ceiling: exact key, unseen [], Counter-MAX."""
    from experiments.campaign_join_battery import gold_serialized

    p1, p2, p3 = pred_name(1), pred_name(2), pred_name(3)
    p4, p5, p6 = pred_name(4), pred_name(5), pred_name(6)
    key_a = tuple(sorted((p1, p2, p3)))
    key_b = tuple(sorted((p4, p5, p6)))

    def mk_once(key: tuple[str, ...], q: str) -> BatteryExample:
        """One join var with two incidences → gold multiset {sig: 1}."""
        a, b = key[0], key[1]
        sparql = (
            "SELECT count(*) WHERE {\n"
            f"?x0 {a} ?y0 .\n"
            f"?x0 {b} ?y1 .\n"
            f"?x0 a {type_name(0)} .\n"
            f"?y0 a {type_name(0)} .\n"
            f"?y1 a {type_name(0)} .\n"
            "}"
        )
        return BatteryExample(
            question=q, sparql=sparql, topology="star", key=key,
        )

    def mk_twice(key: tuple[str, ...], q: str) -> BatteryExample:
        """Two join vars with the same incidence pattern → gold {sig: 2}."""
        a, b = key[0], key[1]
        sparql = (
            "SELECT count(*) WHERE {\n"
            f"?x0 {a} ?y0 .\n"
            f"?x0 {b} ?y1 .\n"
            f"?x1 {a} ?y2 .\n"
            f"?x1 {b} ?y3 .\n"
            f"?x0 a {type_name(0)} .\n"
            f"?x1 a {type_name(0)} .\n"
            f"?y0 a {type_name(0)} .\n"
            f"?y1 a {type_name(0)} .\n"
            f"?y2 a {type_name(0)} .\n"
            f"?y3 a {type_name(0)} .\n"
            "}"
        )
        return BatteryExample(
            question=q, sparql=sparql, topology="star", key=key,
        )

    def mk_star3(key: tuple[str, ...], q: str) -> BatteryExample:
        a, b, c = key
        sparql = (
            "SELECT count(*) WHERE {\n"
            f"?x0 {a} ?x1 .\n"
            f"?x0 {b} ?x2 .\n"
            f"?x0 {c} ?x3 .\n"
            f"?x0 a {type_name(0)} .\n"
            f"?x1 a {type_name(0)} .\n"
            f"?x2 a {type_name(0)} .\n"
            f"?x3 a {type_name(0)} .\n"
            "}"
        )
        return BatteryExample(
            question=q, sparql=sparql, topology="star", key=key,
        )

    # Row A once: gold has signature X once; row A twice: X twice.
    # Same key → inventory Counter |= must take max (2), not sum (3).
    row_once = mk_once(key_a, "once")
    row_twice = mk_twice(key_a, "twice")
    g_once = gold_serialized(row_once)
    g_twice = gold_serialized(row_twice)
    assert len(g_once) == 1
    assert len(g_twice) == 2
    assert g_once[0] == g_twice[0] == g_twice[1]
    sig_shared = g_once[0]

    # (iii) Counter-MAX multiplicity semantics
    inv = build_inventory_ceiling([row_once, row_twice])
    assert key_a in inv
    assert inv[key_a][sig_shared] == 2  # max, not sum (would be 3)

    # (i) exact-key lookup returns the expected per-key multiset
    pred = predict_inventory_ceiling(inv, key_a)
    assert pred.count(sig_shared) == 2
    assert Counter(pred) == Counter([sig_shared, sig_shared])

    # Second key with a different gold multiset
    row_b = mk_star3(key_b, "qb")
    inv2 = build_inventory_ceiling([row_once, row_twice, row_b])
    pred_b = predict_inventory_ceiling(inv2, key_b)
    assert Counter(pred_b) == Counter(gold_serialized(row_b))

    # (ii) unseen key returns []
    unseen = tuple(sorted((pred_name(7), pred_name(8), pred_name(9))))
    assert predict_inventory_ceiling(inv2, unseen) == []
