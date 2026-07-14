from __future__ import annotations

# This repository intentionally keeps experiment drivers outside an import package.
# ruff: noqa: E402, I001

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from semantic_forcing_codex import (  # noqa: E402
    ImitatorScore,
    agreement_aggregate_python,
    drop_mechanism_noops,
    fixed_verdict,
    souffle_agreement,
)
from pil.qa1_battery import live_location, move_sentence, parse_story_text, qa_span


def test_qa1_movement_perturbation_flips_live_location() -> None:
    old_text = " ".join(
        [
            move_sentence("Mary", "moved", "garden"),
            move_sentence("John", "went", "office"),
            qa_span("Mary", "garden"),
        ]
    )
    old_movements, _ = parse_story_text(old_text)
    old_gold = live_location("Mary", old_movements)
    perturbed_text = " ".join(
        [
            move_sentence("Mary", "moved", "garden"),
            move_sentence("John", "went", "office"),
            move_sentence("Mary", "journeyed", "kitchen"),
            qa_span("Mary", "kitchen"),
        ]
    )
    new_movements, _ = parse_story_text(perturbed_text)
    new_gold = live_location("Mary", new_movements)
    assert old_gold == "garden"
    assert new_gold == "kitchen"
    assert new_gold != old_gold


def test_agreement_identity_aggregate_matches_souffle() -> None:
    subject_numbers = [0, 1, 1, 0]
    expected_verb_numbers = [0, 1, 1, 0]
    python_result = agreement_aggregate_python(subject_numbers)
    souffle_result = souffle_agreement(subject_numbers)
    assert python_result == expected_verb_numbers
    assert souffle_result == expected_verb_numbers
    assert python_result == souffle_result


def _imitator(name: str, unperturbed: float, propagation: float) -> ImitatorScore:
    return ImitatorScore(name, unperturbed, propagation)


def test_fixed_discriminator_verdicts() -> None:
    fires = fixed_verdict(
        bar1=0.96,
        bar2=0.90,
        constraint_propagation=1.0,
        imitators=[_imitator("a", 0.90, 0.10), _imitator("b", 0.85, 0.20)],
    )
    assert (fires.verdict, fires.failure) == ("FIRES", None)

    vacuous = fixed_verdict(
        bar1=1.0,
        bar2=1.0,
        constraint_propagation=1.0,
        imitators=[_imitator("a", 0.90, 0.80), _imitator("b", 0.90, 0.10)],
    )
    assert (vacuous.verdict, vacuous.failure) == ("DEAD", "vacuous-discriminator")

    easy = fixed_verdict(
        bar1=1.0,
        bar2=1.0,
        constraint_propagation=1.0,
        imitators=[_imitator("a", 0.70, 0.10), _imitator("b", 0.79, 0.10)],
    )
    assert (easy.verdict, easy.failure) == ("DEAD", "easy-win")

    soft = fixed_verdict(
        bar1=0.95,
        bar2=1.0,
        constraint_propagation=1.0,
        imitators=[_imitator("a", 1.0, 0.0), _imitator("b", 1.0, 0.0)],
    )
    assert (soft.verdict, soft.failure) == ("DEAD", "soft-not-forcing")

    not_derivable = fixed_verdict(
        bar1=0.96,
        bar2=0.89,
        constraint_propagation=1.0,
        imitators=[_imitator("a", 1.0, 0.0), _imitator("b", 1.0, 0.0)],
    )
    assert (not_derivable.verdict, not_derivable.failure) == ("DEAD", "not-derivable")

    one_weak = fixed_verdict(
        bar1=1.0,
        bar2=1.0,
        constraint_propagation=1.0,
        imitators=[_imitator("a", 0.90, 0.10), _imitator("b", 0.79, 0.10)],
    )
    assert (one_weak.verdict, one_weak.failure) == ("DEAD", "vacuous-discriminator")


def test_mechanism_noop_is_dropped_from_denominator() -> None:
    rows = [
        {"id": "engaged", "old": 0, "new": 1},
        {"id": "noop", "old": 1, "new": 1},
    ]
    kept, drop_count = drop_mechanism_noops(
        rows,
        old_gold=lambda row: row["old"],
        new_gold=lambda row: row["new"],
    )
    assert drop_count == 1
    assert [row["id"] for row in kept] == ["engaged"]
    assert len(kept) == 1  # the propagation denominator excludes the no-op
