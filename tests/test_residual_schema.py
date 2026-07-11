"""Residual→schema bridge: parity + selection + Datalog round-trip.

Verifies the step-2 bridge (``docs/notes/residual_as_schema.md``): a residual
leaf becomes a token-presence Schema whose tensor ``predict`` and exported
Datalog clause agree, and which the existing ``propose_schemas`` selector adopts.
"""
from __future__ import annotations

import shutil

import pytest
import torch

from pil.residual_schema import residual_candidates_to_schemas
from pil.residual_template import ResidualCandidate
from pil.rules import RuleProgram, RuleProgramConfig
from pil.schemas import SchemaBank, propose_schemas

HAS_SOUFFLE = shutil.which("souffle") is not None

# shared toy vocabulary: two question words, two ns: paths
STOI = {"marry": 10, "influence": 11, "ns:spouse": 20, "ns:influenced": 21}


def _relation_atom(word: str, path: str) -> ResidualCandidate:
    return ResidualCandidate(
        src=(word, path), tgt=(path,), template_id="relation_atom", domain="cfq",
        meta={"kind": "multi_ns_vote_passthrough"},
    )


def test_bridge_filters_unknown_and_multitoken():
    cands = [
        _relation_atom("marry", "ns:spouse"),
        _relation_atom("influence", "ns:influenced"),
        _relation_atom("unknown_word", "ns:spouse"),          # word not in stoi
        ResidualCandidate(src=("marry",), tgt=("A", "A"),       # multi-token tgt (n-fold)
                          template_id="nfold", domain="scan"),
    ]
    schemas, skipped = residual_candidates_to_schemas(cands, STOI)
    assert [s.name for s in schemas] == [
        "relation_atom/marry|ns:spouse",
        "relation_atom/influence|ns:influenced",
    ]
    reasons = {s["reason"] for s in skipped}
    assert reasons == {"unknown-symbol", "not-single-token-target"}


def test_duplicate_word_path_collapses():
    cands = [_relation_atom("marry", "ns:spouse"), _relation_atom("marry", "ns:spouse")]
    schemas, skipped = residual_candidates_to_schemas(cands, STOI)
    assert len(schemas) == 1 and skipped == []


def test_predict_matches_presence_semantics():
    """SchemaBank.predict reproduces 'fire path token iff word present in window'."""
    schemas, _ = residual_candidates_to_schemas(
        [_relation_atom("marry", "ns:spouse")], STOI)
    bank = SchemaBank(schemas, vocab_size=100, values={})
    # rows: [word present], [absent], [present at pos 1]
    x = torch.tensor([[10, 99], [98, 99], [99, 10]])
    tok = schemas[0].predict(x, bank.values, bank.v2t)
    assert tok.tolist() == [20, -1, 20]        # path token where 'marry' present, else abstain


def test_propose_schemas_selects_relation_atoms():
    """Token-level exact-match selection adopts both relation atoms above threshold."""
    schemas, _ = residual_candidates_to_schemas(
        [_relation_atom("marry", "ns:spouse"),
         _relation_atom("influence", "ns:influenced")], STOI)
    # 2 rows fire 'marry'→ns:spouse, 2 fire 'influence'→ns:influenced
    x = torch.tensor([[10, 99], [10, 98], [11, 99], [11, 98]])
    candidate_ids = torch.tensor([20, 21])
    gold_tok = torch.tensor([20, 20, 21, 21])
    y_cand_idx = torch.searchsorted(candidate_ids, gold_tok)
    accepted, scores = propose_schemas(
        schemas, x, y_cand_idx, candidate_ids, vocab_size=100, values={},
        min_hit_rate=0.4,
    )
    assert {s.name for s in accepted} == {
        "relation_atom/marry|ns:spouse", "relation_atom/influence|ns:influenced"}
    assert all(abs(s - 0.5) < 1e-6 for s in scores)   # each fires on its 2/4 rows


def test_below_threshold_schema_rejected():
    schemas, _ = residual_candidates_to_schemas(
        [_relation_atom("marry", "ns:spouse")], STOI)
    x = torch.tensor([[10, 99], [98, 99], [97, 96], [95, 94]])   # word present in 1/4 rows
    candidate_ids = torch.tensor([20, 21])
    y_cand_idx = torch.searchsorted(candidate_ids, torch.tensor([20, 21, 21, 21]))
    accepted, _ = propose_schemas(
        schemas, x, y_cand_idx, candidate_ids, vocab_size=100, values={},
        min_hit_rate=0.4,
    )
    assert accepted == []   # 0.25 hit-rate < 0.4


@pytest.mark.skipif(not HAS_SOUFFLE, reason="souffle not on PATH")
def test_export_datalog_roundtrips_via_souffle():
    """Exported presence clause decodes identically in Soufflé (proved equivalence)."""
    from pil.datalog_export import export_program, verify_export

    schemas, _ = residual_candidates_to_schemas(
        [_relation_atom("marry", "ns:spouse"),
         _relation_atom("influence", "ns:influenced")], STOI)
    cfg = RuleProgramConfig(vocab_size=100, window=2, frame_dim=4,
                            candidates=[20, 21], seed=0)
    prog = RuleProgram(cfg)
    bank = SchemaBank(schemas, vocab_size=100, values={})
    prog.attach_schemas(bank)
    with torch.no_grad():
        bank.w[:] = 5.0     # dominate bias so the fired path wins argmax
    # each row contains exactly one firing word → one path gets weight
    x = torch.tensor([[10, 99], [11, 99], [99, 10], [98, 11]])
    dl = export_program(prog)
    assert "tok(I,_,10), C=20" in dl and "tok(I,_,11), C=21" in dl
    res = verify_export(prog, x)
    assert res["agreement"] == 1.0 and res["undecided"] == 0
