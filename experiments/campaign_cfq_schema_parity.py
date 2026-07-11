"""CFQ schema parity — does propose_schemas_setf1 reproduce ResidualFamily._admit_naive's
selection EXACTLY on real data/cfq_mcd1.pt?

Mirrors experiments/campaign_cfq_residual.run_split fit/val/test construction and
naive admit, then runs the set-F1 schema selector over the same candidate pool.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from torch import Tensor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.campaign_cfq_residual import (  # noqa: E402
    content_words,
    load_split,
    mean_set_f1,
    mine_base_atoms,
    mine_multi_votes,
    sparql_ns,
)
from pil.residual_schema import (  # noqa: E402
    cfq_stoi_from,
    encode_cfq,
    mean_set_f1_schemas,
    propose_schemas_setf1,
    residual_candidates_to_schemas,
    schema_name_to_pair,
)
from pil.residual_template import (  # noqa: E402
    CFQ_TEMPLATES,
    ResidualCandidate,
    ResidualFamily,
    cfq_domain_atoms,
)

TAG = "mcd1"


def encode_pairs(
    pairs: list[tuple[str, str]],
    stoi: dict[str, int],
) -> tuple[Tensor, list[set[int]]]:
    word_lists = [content_words(q) for q, _ in pairs]
    window = max(1, max((len(w) for w in word_lists), default=1))
    X = encode_cfq(word_lists, stoi, window)
    gold: list[set[int]] = []
    for _, a in pairs:
        paths = sparql_ns(a)
        missing = [p for p in paths if p not in stoi]
        assert not missing, f"gold path(s) missing from stoi: {missing}"
        gold.append({stoi[p] for p in paths})
    return X, gold


def main() -> None:
    data_path = REPO / "data" / f"cfq_{TAG}.pt"
    if not data_path.exists():
        print(f"missing {data_path.name} — skip (do not build)")
        return

    blob = load_split(TAG)
    tr_q, tr_a = blob["train_q"], blob["train_sparql"]
    te_q, te_a = blob["test_q"], blob["test_sparql"]

    # fit / val from train — verbatim from run_split
    idx = list(range(len(tr_q)))
    random.Random(0).shuffle(idx)
    nval = min(int(os.environ.get("CFQ_MAX_VAL", "2000")), max(1, len(idx) // 10))
    val_i, fit_i = idx[:nval], idx[nval:]
    fit_q = [tr_q[i] for i in fit_i]
    fit_a = [tr_a[i] for i in fit_i]
    val_pairs = [(tr_q[i], tr_a[i]) for i in val_i]
    # subsample test for speed if huge — verbatim from run_split
    te_pairs = list(zip(te_q, te_a, strict=True))
    if len(te_pairs) > 3000:
        te_pairs = [te_pairs[i] for i in range(0, len(te_pairs), max(1, len(te_pairs) // 3000))][:3000]

    base_maps = mine_base_atoms(fit_q, fit_a, max_ns=1, min_support=2)
    multi_votes = mine_multi_votes(fit_q, fit_a, min_ns=2)
    multi_votes = {k: v for k, v in multi_votes.items() if k not in base_maps}
    top_k = int(os.environ.get("CFQ_RES_TOPK", "80"))
    if len(multi_votes) > top_k:
        multi_votes = dict(sorted(multi_votes.items(), key=lambda kv: -kv[1])[:top_k])

    domain = cfq_domain_atoms(multi_word_path_votes=multi_votes, atom_min_support=3)
    fam = ResidualFamily(domain, templates=CFQ_TEMPLATES)
    cands = fam.propose(base_maps)

    val_admit = val_pairs
    max_va = int(os.environ.get("CFQ_ADMIT_VAL", "400"))
    if len(val_admit) > max_va:
        val_admit = val_admit[:max_va]

    def val_score(maps):
        return mean_set_f1(val_admit, maps)

    maps_adm, admit_log = fam.admit(
        base_maps, val_score, thresh=1e-4, max_rules=16, celf=False,
    )
    admitted_pairs = {k for k in maps_adm if k not in base_maps}
    _ = admit_log

    # Shared vocab: every word/path that will be looked up via stoi
    all_words: set[str] = set()
    all_paths: set[str] = set()
    for k, v in base_maps.items():
        if k:
            all_words.add(k[0])
        if len(k) >= 2:
            all_paths.add(k[1])
        all_paths.update(v)
    for c in cands:
        if c.src:
            all_words.add(c.src[0])
        if len(c.src) >= 2:
            all_paths.add(c.src[1])
        all_paths.update(c.tgt)
    for q, a in val_admit:
        all_words.update(content_words(q))
        all_paths.update(sparql_ns(a))
    for q, a in te_pairs:
        all_words.update(content_words(q))
        all_paths.update(sparql_ns(a))
    stoi = cfq_stoi_from(all_words, all_paths)

    base_cands = [
        ResidualCandidate(
            src=k, tgt=tuple(v), template_id="base_atom", domain="cfq",
        )
        for k, v in base_maps.items()
    ]
    for c in base_cands:
        assert len(c.tgt) == 1, f"mine_base_atoms must yield singleton tgt, got {c.tgt}"
    base_schemas, _ = residual_candidates_to_schemas(base_cands, stoi)
    cand_schemas, _ = residual_candidates_to_schemas(cands, stoi)  # preserve cands order

    X_val, gold_val = encode_pairs(val_admit, stoi)
    selected, sel_log = propose_schemas_setf1(
        base_schemas, cand_schemas, X_val, gold_val, thresh=1e-4, max_rules=16,
    )
    _ = sel_log
    selected_pairs = {schema_name_to_pair(s.name) for s in selected}

    parity_ok = selected_pairs == admitted_pairs
    val_f1_residual = mean_set_f1(val_admit, maps_adm)
    val_f1_schema = mean_set_f1_schemas(base_schemas + selected, X_val, gold_val)
    X_test, gold_test = encode_pairs(te_pairs, stoi)
    test_f1_residual = mean_set_f1(te_pairs, maps_adm)
    test_f1_schema = mean_set_f1_schemas(base_schemas + selected, X_test, gold_test)

    # Diagnostics first (always), then assert so failures still print numbers
    print(f"PARITY: {'OK' if parity_ok else 'FAIL'}")
    print(f"n_selected={len(selected_pairs)} n_admitted_minus_base={len(admitted_pairs)}")
    if selected_pairs != admitted_pairs:
        print(f"selected - admitted: {selected_pairs - admitted_pairs}")
        print(f"admitted - selected: {admitted_pairs - selected_pairs}")
    print(f"val_f1_residual={val_f1_residual:.6f} val_f1_schema={val_f1_schema:.6f} "
          f"val_match={abs(val_f1_residual - val_f1_schema) < 1e-9}")
    print(f"test_f1_residual={test_f1_residual:.6f} test_f1_schema={test_f1_schema:.6f} "
          f"test_match={abs(test_f1_residual - test_f1_schema) < 1e-9}")
    print(f"mcd1 test set-F1 (admit) = {test_f1_residual:.3f}")
    print(f"n_base={len(base_maps)} n_cands={len(cands)} n_val_admit={len(val_admit)} "
          f"n_test={len(te_pairs)}")

    assert selected_pairs == admitted_pairs
    assert abs(val_f1_residual - val_f1_schema) < 1e-9
    assert abs(test_f1_residual - test_f1_schema) < 1e-9


if __name__ == "__main__":
    main()
