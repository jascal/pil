"""Datalog export: text shape + exact Soufflé agreement with the tensor hard path."""

import shutil

import pytest
import torch

from pil.datalog_export import export_program, verify_export
from pil.rule_learner import PICRuleLearner, RuleLearnerConfig
from pil.rules import PRESENT, RuleProgram, RuleProgramConfig

HAS_SOUFFLE = shutil.which("souffle") is not None


def small_trained_program():
    g = torch.Generator().manual_seed(0)
    bits = torch.randint(0, 2, (1200, 3), generator=g)
    X = bits + 10
    y = bits.sum(1).remainder(2) + 10        # parity-3: needs thresholds or full patterns
    cfg = RuleProgramConfig(vocab_size=50304, window=3, frame_dim=8, candidates=[10, 11], seed=0)
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=4, init_rules=24, births_per_phase=8, epochs_per_phase=15,
        max_rules=64, recency_gamma=1.0, seed=0, verbose=False,
    )
    PICRuleLearner(prog, lc).fit(X, y)
    return prog, X


def test_export_text_contains_program_shape():
    prog, _ = small_trained_program()
    dl = export_program(prog, "t")
    assert ".decl tok(inst:number, pos:number, id:number)" in dl
    assert ".decl fired0(inst:number, k:number)" in dl
    assert ".output decide" in dl
    assert "sum W : { contrib(I,_,V,W) }" in dl
    assert "max S2 : { logit(I,_,S2) }" in dl


def test_negated_and_threshold_clauses_render():
    cfg = RuleProgramConfig(vocab_size=100, window=2, frame_dim=4, candidates=[1, 2], seed=0)
    prog = RuleProgram(cfg)
    anchors = torch.tensor([[7, 9], [7, 9]])
    rel = torch.full((2, 2), PRESENT)
    sgn = torch.tensor([[PRESENT, -PRESENT], [PRESENT, PRESENT]])
    thr = torch.tensor([0.0, 1.5])
    is_thr = torch.tensor([False, True])
    prog.add_rules_stratum1(anchors, rel, sgn, thr, is_thr)
    prog.round_structure()
    dl = export_program(prog)
    assert "fired0(I,0) :- inst(I), tok(I,0,7), !tok(I,1,9)." in dl
    assert "thr0(1,2)." in dl                      # ceil(1.5) = 2
    assert "sat0(I,1,0) :- tok(I,0,7)." in dl
    assert "N = count : { sat0(I,K,_) }, N >= T." in dl


@pytest.mark.skipif(not HAS_SOUFFLE, reason="souffle not on PATH")
def test_souffle_agreement_exact():
    prog, X = small_trained_program()
    res = verify_export(prog, X[:200])
    assert res["undecided"] == 0
    assert res["agreement"] == 1.0


@pytest.mark.skipif(not HAS_SOUFFLE, reason="souffle not on PATH")
def test_schema_rule_exports_arithmetic_clause_and_runs():
    from pil.schemas import SchemaBank, arithmetic_library, propose_schemas

    p = 7
    toks = list(range(100, 100 + p))            # token ids for residues 0..6
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    X = torch.stack([torch.tensor(toks)[a], torch.tensor(toks)[b]], dim=1)
    y_idx = (a + b).remainder(p)                # candidate index == residue

    cfg = RuleProgramConfig(vocab_size=200, window=2, frame_dim=4, candidates=toks, seed=0)
    prog = RuleProgram(cfg)
    values = dict(zip(toks, range(p), strict=True))
    lib = arithmetic_library(0, 1, p)
    accepted, scores = propose_schemas(
        lib, X, y_idx, prog.candidate_ids, 200, values, min_hit_rate=0.5
    )
    assert [s.name for s in accepted] == [f"add_mod{p}[0,1]"]
    assert scores[0] == 1.0

    bank = SchemaBank(accepted, 200, values)
    prog.attach_schemas(bank)
    with torch.no_grad():
        bank.w[0] = 4.0
    # tensor semantics: argmax == a+b mod p everywhere
    L = prog(X, hard=True)["logits"]
    assert (L.argmax(-1) == y_idx).all()
    # datalog semantics: same
    dl = export_program(prog)
    assert "num(A,Na), num(B,Nb)" in dl and "% 7" in dl
    res = verify_export(prog, X)
    assert res["agreement"] == 1.0 and res["undecided"] == 0


@pytest.mark.skipif(not HAS_SOUFFLE, reason="souffle not on PATH")
def test_lookup_family_exports_and_runs_exact():
    cfg = RuleProgramConfig(
        vocab_size=100, window=2, frame_dim=4, candidates=[1, 2, 3],
        lookup_offsets=(1,), seed=0,
    )
    prog = RuleProgram(cfg)
    with torch.no_grad():   # give the lookup family some non-trivial rows
        emb = prog.lookup["1"]
        emb.weight[7] = 2.0 * prog.U[0]
        emb.weight[9] = 2.0 * prog.U[2]
    X = torch.tensor([[5, 7], [5, 9], [5, 8]])
    dl = export_program(prog, lookup_domain=X)
    assert "contrib(I,900000001,V,W) :- tok(I,1,C), lkp1(C,V,W)." in dl
    res = verify_export(prog, X)
    assert res["agreement"] == 1.0 and res["undecided"] == 0


@pytest.mark.skipif(not HAS_SOUFFLE, reason="souffle not on PATH")
def test_direct_lookup_exports_and_runs_exact():
    cfg = RuleProgramConfig(
        vocab_size=100, window=2, frame_dim=4, candidates=[1, 2, 3],
        direct_lookup_offsets=(1,), seed=0,
    )
    prog = RuleProgram(cfg)
    with torch.no_grad():
        prog.direct_lookup["1"].weight[7] = torch.tensor([3.0, 0.0, -1.0])
        prog.direct_lookup["1"].weight[9] = torch.tensor([0.0, 0.0, 2.0])
    X = torch.tensor([[5, 7], [5, 9], [5, 8]])
    dl = export_program(prog, lookup_domain=X)
    assert "contrib(I,910000001,V,W) :- tok(I,1,C), lkpd1(C,V,W)." in dl
    res = verify_export(prog, X)
    assert res["agreement"] == 1.0 and res["undecided"] == 0


@pytest.mark.skipif(not HAS_SOUFFLE, reason="souffle not on PATH")
def test_eq_literal_exports_as_join_and_runs():
    from pil.rules import ABSENT

    cfg = RuleProgramConfig(
        vocab_size=100, window=3, frame_dim=4, candidates=[1, 2], eq_atoms=True, seed=0
    )
    prog = RuleProgram(cfg)
    s1 = prog.strata[0]
    rel = torch.full((2, s1.n_lits), ABSENT)
    sgn = torch.full((2, s1.n_lits), PRESENT)
    rel[0, 3 + 1] = PRESENT                     # x0 == x2
    rel[1, 3 + 1] = PRESENT
    sgn[1, 3 + 1] = -PRESENT                    # x0 != x2
    prog.add_rules_stratum1(torch.zeros(2, 3, dtype=torch.long), rel, sgn)
    prog.round_structure()
    with torch.no_grad():
        prog.heads[0][0] = 3.0 * prog.U[0]
        prog.heads[0][1] = 3.0 * prog.U[1]
    dl = export_program(prog)
    assert "tok(I,0,Eq0), tok(I,2,Eq0)" in dl
    assert "Ea0 != Eb0" in dl
    X = torch.tensor([[7, 1, 7], [7, 1, 8], [4, 4, 4]])
    res = verify_export(prog, X)
    assert res["agreement"] == 1.0 and res["undecided"] == 0
