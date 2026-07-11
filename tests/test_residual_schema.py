"""Residual→schema bridge: parity + selection + Datalog round-trip.

Verifies the step-2 bridge (``docs/notes/residual_as_schema.md``): a residual
leaf becomes a token-presence Schema whose tensor ``predict`` and exported
Datalog clause agree, and which the existing ``propose_schemas`` selector adopts.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import torch

from pil.residual_schema import (
    cfq_stoi_from,
    encode_cfq,
    mean_set_f1_schemas,
    propose_schemas_setf1,
    residual_candidates_to_schemas,
    schema_set_predict,
)
from pil.residual_template import (
    CFQ_TEMPLATES,
    ResidualCandidate,
    ResidualFamily,
    cfq_domain_atoms,
)
from pil.rules import RuleProgram, RuleProgramConfig
from pil.schemas import SchemaBank, propose_schemas

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # experiments/ has no __init__.py; needed for the import below

from experiments.campaign_cfq_residual import (  # noqa: E402
    content_words,
    mean_set_f1,
    predict_paths,
    sparql_ns,
)

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


def test_cfq_stoi_shared_vocab():
    words = ["marry", "influence", "book", "marry"]  # note duplicate
    paths = ["ns:spouse", "ns:influenced", "ns:author"]
    stoi = cfq_stoi_from(words, paths)
    assert stoi == {
        "book": 0, "influence": 1, "marry": 2,
        "ns:author": 3, "ns:influenced": 4, "ns:spouse": 5,
    }
    # order/duplicate-invariance: shuffled input, same sets -> identical stoi
    stoi2 = cfq_stoi_from(list(reversed(words)) + ["book"], list(reversed(paths)))
    assert stoi2 == stoi


def test_schema_set_decode_matches_predict_paths():
    """schema_set_predict (id→path) matches residual predict_paths, row by row."""
    cands = [
        _relation_atom("marry", "ns:spouse"),
        _relation_atom("influence", "ns:influenced"),
    ]
    stoi = cfq_stoi_from(
        ["marry", "influence", "tom"],
        ["ns:spouse", "ns:influenced"],
    )
    schemas, _ = residual_candidates_to_schemas(cands, stoi)
    maps = {c.src: list(c.tgt) for c in cands}
    itos = {i: s for s, i in stoi.items()}
    questions = [
        "who did tom marry",
        "who did tom influence",
        "what did tom do",  # no matching word -> empty on both sides
    ]
    window = max(1, max(len(content_words(q)) for q in questions))
    word_lists = [content_words(q) for q in questions]
    X = encode_cfq(word_lists, stoi, window)
    pred_sets = schema_set_predict(schemas, X)
    for q, pred_ids in zip(questions, pred_sets, strict=True):
        schema_paths = {itos[i] for i in pred_ids}
        residual_paths = predict_paths(q, maps)
        assert schema_paths == residual_paths


def test_setf1_selector_reproduces_admit():
    """propose_schemas_setf1 reproduces ResidualFamily.admit on a synthetic fixture.

    Verified on this fixture only (empirical), not a general proof.
    """
    base_maps = {("marry", "ns:spouse"): ["ns:spouse"]}
    multi_votes = {
        ("influence", "ns:influenced"): 3,
        ("book", "ns:author"): 3,
        ("decoy", "ns:decoy"): 3,  # never appears in any val question -> never selected
    }
    domain = cfq_domain_atoms(multi_word_path_votes=multi_votes, atom_min_support=2)
    fam = ResidualFamily(domain, templates=CFQ_TEMPLATES)
    cands = fam.propose(base_maps)

    val_pairs = [
        ("who did tom marry", "ns:spouse"),
        ("who did tom influence", "ns:influenced"),
        ("who wrote the book", "ns:author"),
    ]

    def val_score(maps):
        return mean_set_f1(val_pairs, maps)

    maps_adm, admit_log = fam.admit(
        base_maps, val_score, thresh=1e-4, max_rules=16, celf=False,
    )

    stoi = cfq_stoi_from(
        ["marry", "influence", "book", "decoy", "tom", "wrote", "who", "did", "what"],
        ["ns:spouse", "ns:influenced", "ns:author", "ns:decoy"],
    )
    base_cands = [
        ResidualCandidate(
            src=("marry", "ns:spouse"), tgt=("ns:spouse",),
            template_id="base_atom", domain="cfq",
        ),
    ]
    base_schemas, _ = residual_candidates_to_schemas(base_cands, stoi)
    cand_schemas, _ = residual_candidates_to_schemas(cands, stoi)
    word_lists = [content_words(q) for q, _ in val_pairs]
    window = max(len(ws) for ws in word_lists)
    X_val = encode_cfq(word_lists, stoi, window)
    gold_val = [{stoi[p] for p in sparql_ns(a)} for _, a in val_pairs]

    selected, sel_log = propose_schemas_setf1(
        base_schemas, cand_schemas, X_val, gold_val, thresh=1e-4, max_rules=16,
    )

    def _name_to_pair(name: str) -> tuple[str, str]:
        _, rest = name.split("/", 1)
        word, path = rest.split("|", 1)
        return word, path

    selected_pairs = {_name_to_pair(s.name) for s in selected}
    admitted_pairs = {k for k in maps_adm if k not in base_maps}
    assert selected_pairs == admitted_pairs
    assert selected_pairs == {("influence", "ns:influenced"), ("book", "ns:author")}
    assert len(selected) == sum(1 for e in admit_log if e.get("event") == "admit")
    assert len(selected) == 2

    f1_residual = mean_set_f1(val_pairs, maps_adm)
    f1_schema = mean_set_f1_schemas(base_schemas + selected, X_val, gold_val)
    assert abs(f1_residual - f1_schema) < 1e-9

    # relation_atom: src == (word, path) always — residual src-dedup and schema
    # (stoi[word], stoi[path])-dedup coincide
    for c in cands:
        assert c.template_id == "relation_atom" and c.src == (c.src[0], c.tgt[0])
    _ = sel_log  # log structure exercised by selector; shape checked via n_selected


def test_setf1_rejects_nonhelpful():
    """Selector admits nothing when residual word never appears in val questions."""
    base_maps = {("marry", "ns:spouse"): ["ns:spouse"]}
    multi_votes = {("neverseen", "ns:unrelated"): 5}  # word never in any val question
    domain = cfq_domain_atoms(multi_word_path_votes=multi_votes, atom_min_support=2)
    fam = ResidualFamily(domain, templates=CFQ_TEMPLATES)
    cands = fam.propose(base_maps)
    val_pairs = [("who did tom marry", "ns:spouse")]

    stoi = cfq_stoi_from(
        ["marry", "neverseen", "tom"],
        ["ns:spouse", "ns:unrelated"],
    )
    base_cands = [
        ResidualCandidate(
            src=("marry", "ns:spouse"), tgt=("ns:spouse",),
            template_id="base_atom", domain="cfq",
        ),
    ]
    base_schemas, _ = residual_candidates_to_schemas(base_cands, stoi)
    cand_schemas, _ = residual_candidates_to_schemas(cands, stoi)
    word_lists = [content_words(q) for q, _ in val_pairs]
    window = max(1, max(len(ws) for ws in word_lists))
    X_val = encode_cfq(word_lists, stoi, window)
    gold_val = [{stoi[p] for p in sparql_ns(a)} for _, a in val_pairs]

    selected, sel_log = propose_schemas_setf1(
        base_schemas, cand_schemas, X_val, gold_val, thresh=1e-4, max_rules=16,
    )
    assert selected == []
    assert sel_log == [{"event": "stop", "n_selected": 0}]
