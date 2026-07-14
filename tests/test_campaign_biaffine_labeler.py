"""Unit tests for the axis-B biaffine relation-labeler (grok lane)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.campaign_biaffine_labeler import (  # noqa: E402
    GENERIC_FALLBACK,
    BilinearLabelConfig,
    BilinearRelationLabeler,
    _coarsen,
    apply_preregistered_read,
    apply_through_line_case_read,
    partition_deprel_accuracy,
)
from experiments.german_r1_codex import Sentence  # noqa: E402
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    head_deprel_by_sent,
)


def _sentence(
    sent_id: str,
    tokens: tuple[str, ...],
    *,
    case: tuple[str, ...] | None = None,
) -> Sentence:
    return Sentence(
        sent_id=sent_id,
        text=" ".join(tokens),
        tokens=tokens,
        targets={
            "pos": ("NOUN",) * len(tokens),
            "morph_case": case or ("-",) * len(tokens),
            "morph_gnn": ("-|-",) * len(tokens),
        },
    )


def test_predict_signature_excludes_gold_fields() -> None:
    signature = inspect.signature(BilinearRelationLabeler.predict)
    parameter_names = set(signature.parameters)
    assert "self" in parameter_names
    assert parameter_names == {"self", "tokens", "predicted_pos", "head_offsets"}
    assert "gold" not in " ".join(parameter_names)
    assert "gold_heads" not in parameter_names
    assert "gold_deprels" not in parameter_names
    assert "gold_head_offset" not in parameter_names
    assert "gold_deprel" not in parameter_names


def test_scorer_logit_shapes_and_low_rank_form() -> None:
    vocab = ("dep", "nsubj", "obj", "root")
    labeler = BilinearRelationLabeler(
        BilinearLabelConfig(rank=4, epochs=1, seed=0),
        relation_vocab=vocab,
    )
    tokens = ("Das", "Kind", "sieht", "den", "Hund")
    pos = ("DET", "NOUN", "VERB", "DET", "NOUN")
    # token 1 -> 2 (offset +1), token 2 root (0), etc.
    heads = (1, 1, 0, 1, -2)

    phi = labeler.phi(tokens[0], pos[0], 0, len(tokens))
    scores = labeler.logits(tokens, pos, heads)
    predicted = labeler.predict(tokens, pos, heads)

    assert phi.shape == (labeler.feature_dim,)
    assert labeler.U is not None and labeler.V is not None and labeler.C is not None
    assert labeler.U.shape == (labeler.feature_dim, 4)
    assert labeler.V.shape == (labeler.feature_dim, 4)
    assert labeler.C.shape == (len(vocab), 4, 4)
    assert scores.shape == (len(tokens), len(vocab))
    assert len(predicted) == len(tokens)
    assert all(label in vocab for label in predicted)
    # ROOT governor short-circuit: offset 0 uses root phi path without crash.
    root_phi = labeler.phi("<ROOT>", "<UNK>", 0, len(tokens), root=True)
    assert root_phi.shape == (labeler.feature_dim,)
    assert root_phi[2] == 1.0  # root flag slot in shipped phi


def test_oov_gold_relation_still_predicts_member_of_r() -> None:
    """True gold may be outside R; predict must still return a defined R member."""
    # Train vocab deliberately omits "obl:arg" / exotic labels.
    vocab = ("dep", "nsubj", "obj", "punct")
    labeler = BilinearRelationLabeler(
        BilinearLabelConfig(rank=4, epochs=1, seed=1),
        relation_vocab=vocab,
    )
    tokens = ("Mit", "Freude", "kam", "er")
    pos = ("ADP", "NOUN", "VERB", "PRON")
    heads = (1, 1, 0, -1)
    # Gold would be ("case", "obl:arg", "root", "nsubj") — several OOV w.r.t. R.
    predicted = labeler.predict(tokens, pos, heads)

    assert len(predicted) == len(tokens)
    assert all(label in vocab for label in predicted)
    assert "obl:arg" not in predicted
    assert GENERIC_FALLBACK in vocab  # documented generic available in R


def test_empty_vocab_fallback_is_generic_dep() -> None:
    labeler = BilinearRelationLabeler(BilinearLabelConfig(rank=2, epochs=1, seed=0))
    assert labeler.relation_vocab == ()
    predicted = labeler.predict(("A", "B"), ("NOUN", "VERB"), (1, 0))
    assert predicted == (GENERIC_FALLBACK, GENERIC_FALLBACK)


def test_fit_records_epoch_losses_and_predict_stays_in_vocab() -> None:
    sentences = [
        _sentence("a", ("Kind", "sieht", "Hund")),
        _sentence("b", ("Mann", "mag", "Frau")),
    ]
    heads = head_deprel_by_sent(
        [
            HeadDeprelRecord(
                "a",
                sentences[0].text,
                sentences[0].tokens,
                (1, 0, -1),
                ("nsubj", "root", "obj"),
            ),
            HeadDeprelRecord(
                "b",
                sentences[1].text,
                sentences[1].tokens,
                (1, 0, -1),
                ("nsubj", "root", "obj"),
            ),
        ]
    )
    predicted_pos = {
        "a": ("NOUN", "VERB", "NOUN"),
        "b": ("NOUN", "VERB", "NOUN"),
    }
    labeler = BilinearRelationLabeler(
        BilinearLabelConfig(rank=4, epochs=3, learning_rate=0.1, l2_penalty=0.0, seed=0)
    )
    labeler.fit(sentences, heads, predicted_pos)

    assert len(labeler.epoch_losses) == 3
    assert all(isinstance(loss, float) for loss in labeler.epoch_losses)
    assert set(labeler.relation_vocab) == {"nsubj", "obj", "root"}
    predicted = labeler.predict(
        sentences[0].tokens, predicted_pos["a"], (1, 0, -1)
    )
    assert all(label in labeler.relation_vocab for label in predicted)


def test_partition_metric_known_split() -> None:
    # Gold coarse membership: nsubj,obj,obl in pairwise; det,amod,case in local;
    # root/punct? punct is local; root excluded from both.
    gold = (
        "nsubj",
        "obj",
        "obl:arg",  # coarse obl → pairwise; exact mismatch if pred is obl
        "det",
        "amod",
        "case",
        "root",  # neither partition
        "punct",
    )
    predicted = (
        "nsubj",  # hit pairwise
        "iobj",  # miss pairwise (exact)
        "obl:arg",  # hit pairwise exact
        "det",  # hit local
        "amod",  # hit local
        "nmod",  # miss local
        "root",  # excluded
        "punct",  # hit local
    )
    result = partition_deprel_accuracy(predicted, gold)

    assert result["pairwise_sensitive_n"] == 3
    assert result["pairwise_sensitive_accuracy"] == pytest.approx(2 / 3)
    assert result["local_n"] == 4
    assert result["local_accuracy"] == pytest.approx(3 / 4)


def test_coarsen_strips_subtype() -> None:
    assert _coarsen("nsubj:pass") == "nsubj"
    assert _coarsen("obl:arg") == "obl"
    assert _coarsen("det") == "det"


def test_read_verdict_fires() -> None:
    per_seed = [
        {
            "seed": 0,
            "las": 0.55,
            "las_gain": 0.04,
            "deprel_only_accuracy": 0.70,
            "deprel_only_gain": 0.05,
            "case": 0.80,
            "case_gain": 0.02,
        },
        {
            "seed": 1,
            "las": 0.56,
            "las_gain": 0.05,
            "deprel_only_accuracy": 0.71,
            "deprel_only_gain": 0.06,
            "case": 0.81,
            "case_gain": 0.03,
        },
        {
            "seed": 2,
            "las": 0.54,
            "las_gain": 0.03,
            "deprel_only_accuracy": 0.69,
            "deprel_only_gain": 0.04,
            "case": 0.79,
            "case_gain": 0.01,
        },
    ]
    read = apply_preregistered_read(
        unary_las=0.51,
        unary_case=0.78,
        per_seed=per_seed,
        partition_gains={"pairwise_sensitive": 0.08, "local": 0.02},
    )
    assert read["verdict"] == "FIRES"
    assert read["las_gain_mean"] == pytest.approx(0.04)
    assert "FIRES" in read["text"]


def test_read_verdict_halted() -> None:
    per_seed = [
        {
            "seed": 0,
            "las": 0.48,
            "las_gain": -0.02,
            "deprel_only_accuracy": 0.60,
            "deprel_only_gain": -0.01,
            "case": 0.74,
            "case_gain": -0.01,
        },
        {
            "seed": 1,
            "las": 0.49,
            "las_gain": -0.01,
            "deprel_only_accuracy": 0.61,
            "deprel_only_gain": 0.0,
            "case": 0.75,
            "case_gain": 0.0,
        },
        {
            "seed": 2,
            "las": 0.47,
            "las_gain": -0.03,
            "deprel_only_accuracy": 0.59,
            "deprel_only_gain": -0.02,
            "case": 0.73,
            "case_gain": -0.02,
        },
    ]
    read = apply_preregistered_read(
        unary_las=0.50,
        unary_case=0.75,
        per_seed=per_seed,
        partition_gains={"pairwise_sensitive": -0.01, "local": 0.0},
    )
    assert read["verdict"] == "HALTED"
    assert read["las_gain_mean"] < 0
    assert "HALTED" in read["text"]


def test_read_verdict_in_between_small_gain() -> None:
    per_seed = [
        {
            "seed": 0,
            "las": 0.515,
            "las_gain": 0.015,
            "deprel_only_accuracy": 0.66,
            "deprel_only_gain": 0.01,
            "case": 0.77,
            "case_gain": 0.01,
        },
        {
            "seed": 1,
            "las": 0.512,
            "las_gain": 0.012,
            "deprel_only_accuracy": 0.65,
            "deprel_only_gain": 0.005,
            "case": 0.76,
            "case_gain": 0.0,
        },
        {
            "seed": 2,
            "las": 0.518,
            "las_gain": 0.018,
            "deprel_only_accuracy": 0.67,
            "deprel_only_gain": 0.015,
            "case": 0.78,
            "case_gain": 0.02,
        },
    ]
    read = apply_preregistered_read(
        unary_las=0.50,
        unary_case=0.76,
        per_seed=per_seed,
        partition_gains={"pairwise_sensitive": 0.03, "local": 0.01},
    )
    assert read["verdict"] == "IN-BETWEEN"
    assert 0.0 < read["las_gain_mean"] < 0.03
    assert "IN-BETWEEN" in read["text"]


def test_read_verdict_in_between_diffuse_partition() -> None:
    # Mean LAS gain high and positive every seed, but partition diffuse.
    per_seed = [
        {
            "seed": 0,
            "las": 0.55,
            "las_gain": 0.05,
            "deprel_only_accuracy": 0.70,
            "deprel_only_gain": 0.05,
            "case": 0.80,
            "case_gain": 0.04,
        },
        {
            "seed": 1,
            "las": 0.56,
            "las_gain": 0.06,
            "deprel_only_accuracy": 0.71,
            "deprel_only_gain": 0.06,
            "case": 0.81,
            "case_gain": 0.05,
        },
        {
            "seed": 2,
            "las": 0.54,
            "las_gain": 0.04,
            "deprel_only_accuracy": 0.69,
            "deprel_only_gain": 0.04,
            "case": 0.79,
            "case_gain": 0.03,
        },
    ]
    read = apply_preregistered_read(
        unary_las=0.50,
        unary_case=0.76,
        per_seed=per_seed,
        partition_gains={"pairwise_sensitive": 0.01, "local": 0.04},
    )
    assert read["verdict"] == "IN-BETWEEN"
    assert "diffuse" in read["text"]


def test_through_line_case_plateau_language() -> None:
    deliverable = apply_through_line_case_read(0.91)
    plateau = apply_through_line_case_read(0.80)
    diagnose = apply_through_line_case_read(0.74)

    assert deliverable["verdict"] == "deliverable_met"
    assert plateau["verdict"] == "plateau"
    assert "plateau" in plateau["text"]
    assert "irreducible" not in plateau["text"].lower()
    assert "required" not in plateau["text"].lower()
    assert diagnose["verdict"] == "diagnose"


def test_logits_are_finite_after_root_and_nonroot_mix() -> None:
    labeler = BilinearRelationLabeler(
        BilinearLabelConfig(rank=3, epochs=1, seed=2),
        relation_vocab=("dep", "nsubj", "root"),
    )
    scores = labeler.logits(
        ("A", "B", "C"),
        ("NOUN", "VERB", "NOUN"),
        (1, 0, -1),
    )
    assert scores.shape == (3, 3)
    assert np.all(np.isfinite(scores))
