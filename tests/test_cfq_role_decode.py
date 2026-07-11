"""Unit tests for experiments/campaign_cfq_role_decode (synthetic fixtures only)."""
from __future__ import annotations

import sys
from pathlib import Path

from pil.cfq_edges import edge_f1, parse_sparql_edges, role_typed

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # experiments/ has no __init__.py

from experiments.campaign_cfq_residual import content_words  # noqa: E402
from experiments.campaign_cfq_role_decode import (  # noqa: E402
    _content_words_with_pos,
    decode_roles,
    fit_role_stats,
    predict_majority_joint,
)

P = "ns:people.person.spouse_s"
P_UNSEEN = "ns:film.film.directed_by"
P_OTHER = "ns:influence.influence_node.influenced"


def _sparql_var_var(pred: str) -> str:
    return f"""SELECT count(*) WHERE {{
?x0 {pred} ?x1 .
}}"""


def _sparql_var_ent(pred: str, ment: str = "M1") -> str:
    return f"""SELECT count(*) WHERE {{
?x0 {pred} {ment} .
}}"""


def _sparql_ent_ent(pred: str, s: str = "M0", o: str = "M1") -> str:
    return f"""SELECT count(*) WHERE {{
{s} {pred} {o} .
}}"""


def _fit_majority_var_var() -> tuple[list[str], list[str]]:
    """Fit data: P has majority joint (VAR, VAR); only 'marry' co-occurs with P ≥2×.

    Other content words (someone/somebody/anyone) each appear once so they stay
    below mine_edge_pred_atoms min_support=2 — leaving a unique anchor for P.
    """
    fit_q = [
        "Did marry someone",
        "Did marry somebody",
        "Did marry anyone",
        # extra mass so global_default stays (VAR, VAR)
        "Did influence someone",
        "Did influence somebody",
    ]
    fit_a = [
        _sparql_var_var(P),
        _sparql_var_var(P),
        _sparql_var_var(P),
        _sparql_var_var(P_OTHER),
        _sparql_var_var(P_OTHER),
    ]
    return fit_q, fit_a


def test_content_words_raw_window_includes_stopword_a() -> None:
    """'a' is filtered from content_words but must appear in raw windows."""
    q = "Did a person marry M1"
    cw = content_words(q)
    assert "a" not in cw
    assert "marry" in cw
    kept, pos, raw = _content_words_with_pos(q)
    assert "a" in [t.lower() for t in raw]
    mi = kept.index("marry")
    ri = pos[mi]
    obj_window = raw[ri + 1 : ri + 4]
    assert any(t == "M1" for t in obj_window)
    subj_window = raw[max(0, ri - 3) : ri]
    assert any(t.lower() == "a" for t in subj_window)


def test_decoder_beats_majority_on_surface_override() -> None:
    fit_q, fit_a = _fit_majority_var_var()
    stats = fit_role_stats(fit_q, fit_a)
    assert P in stats["maj_role"]
    assert stats["maj_role"][P][1] == "VAR"  # majority object is VAR

    q = "Did a person marry M1"
    gold_sparql = _sparql_var_ent(P, "M1")
    gold_rt = [role_typed(e) for e in parse_sparql_edges(gold_sparql).edges]
    assert gold_rt[0][2] == "ENT"

    dec, _counts = decode_roles([P], q, stats)
    maj = predict_majority_joint([P], stats)
    assert dec[0][2] == "ENT"
    assert maj[0][2] == "VAR"
    assert edge_f1(dec, gold_rt) > edge_f1(maj, gold_rt)


def test_decoder_fallback_on_ambiguous_anchor() -> None:
    fit_q, fit_a = _fit_majority_var_var()
    stats = fit_role_stats(fit_q, fit_a)

    # No M# and no a/an in the object window after 'marry' → AMBIG → maj_role fallback
    q = "Did person marry someone"
    gold_sparql = _sparql_var_var(P)
    gold_rt = [role_typed(e) for e in parse_sparql_edges(gold_sparql).edges]

    dec, counts = decode_roles([P], q, stats)
    assert dec[0] == (stats["maj_role"][P][0], P, stats["maj_role"][P][1])
    assert counts["fallback"] == 1
    assert counts["unseen"] == 0
    assert counts["total"] == 1
    # also double-anchor case
    q2 = "Did a person marry and marry again"
    dec2, counts2 = decode_roles([P], q2, stats)
    assert dec2[0] == (stats["maj_role"][P][0], P, stats["maj_role"][P][1])
    assert counts2["fallback"] == 1
    _ = gold_rt  # fixture available for never_worse


def test_decoder_no_leak_on_unseen_predicate() -> None:
    fit_q, fit_a = _fit_majority_var_var()
    stats = fit_role_stats(fit_q, fit_a)
    assert P_UNSEEN not in stats["maj_role"]
    gd = stats["global_default"]
    assert gd == ("VAR", "VAR")

    q = "Was M0 directed by M1"
    gold_sparql = _sparql_ent_ent(P_UNSEEN, "M0", "M1")
    gold_rt = [role_typed(e) for e in parse_sparql_edges(gold_sparql).edges]
    assert gold_rt[0] == ("ENT", P_UNSEEN, "ENT")
    assert (gold_rt[0][0], gold_rt[0][2]) != gd

    dec, counts = decode_roles([P_UNSEEN], q, stats)
    assert dec[0] == (gd[0], P_UNSEEN, gd[1])
    assert dec[0] != gold_rt[0]
    assert counts["unseen"] == 1
    assert counts["total"] == 1


def test_never_worse_property() -> None:
    fit_q, fit_a = _fit_majority_var_var()
    stats = fit_role_stats(fit_q, fit_a)
    q = "Did person marry someone"
    gold_sparql = _sparql_var_var(P)
    gold_rt = [role_typed(e) for e in parse_sparql_edges(gold_sparql).edges]
    dec, counts = decode_roles([P], q, stats)
    maj = predict_majority_joint([P], stats)
    assert counts["fallback"] == 1
    assert edge_f1(dec, gold_rt) == edge_f1(maj, gold_rt)


def test_maj_nonent_defaults_to_var_when_only_ent() -> None:
    """Predicate with only ENT subjects/objects → maj_nonent_* defaults to VAR."""
    fit_q = [
        "Did M0 marry M1",
        "Did M2 marry M3",
    ]
    fit_a = [
        _sparql_ent_ent(P, "M0", "M1"),
        _sparql_ent_ent(P, "M2", "M3"),
    ]
    stats = fit_role_stats(fit_q, fit_a)
    assert stats["maj_role"][P] == ("ENT", "ENT")
    assert stats["maj_nonent_subj"][P] == "VAR"
    assert stats["maj_nonent_obj"][P] == "VAR"


def test_global_default_is_computed_not_hardcoded() -> None:
    fit_q_var = ["Did a person marry someone", "Did a person marry somebody"]
    fit_a_var = [_sparql_var_var(P), _sparql_var_var(P)]
    stats_var = fit_role_stats(fit_q_var, fit_a_var)
    assert stats_var["global_default"] == ("VAR", "VAR")

    fit_q_ent = ["Did M0 marry M1", "Did M2 marry M3"]
    fit_a_ent = [_sparql_ent_ent(P, "M0", "M1"), _sparql_ent_ent(P, "M2", "M3")]
    stats_ent = fit_role_stats(fit_q_ent, fit_a_ent)
    assert stats_ent["global_default"] == ("ENT", "ENT")
    assert stats_var["global_default"] != stats_ent["global_default"]
