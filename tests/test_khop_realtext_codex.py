"""Fast mechanism tests for the pre-registered real-text khop probe."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

from khop_realtext_codex import (  # noqa: E402
    GUARD_THRESHOLD,
    bridge_ablation_guard,
    khop_ablated_predict,
    khop_predict,
    lookup_direct_baselines,
    score_predictions,
)


def test_genuine_two_hop_chain_recovers_and_ablation_breaks() -> None:
    # q=4 -> bridge=2 at the rightmost earlier q; the other 2 -> gold=9.
    ctx = [2, 9, 7, 4, 2, 4]
    gold = 9
    result = khop_predict(ctx, {2})
    assert result is not None
    assert result.prediction == gold
    # The bridge gate is semantically on 2, not on query 4.
    assert khop_predict(ctx, {4}) is None

    one_hop, memorizer = lookup_direct_baselines(
        ctx,
        {4: 8},
        {2: {(2, 4): 8}, 3: {(4, 2, 4): 8}},
    )
    assert one_hop != gold
    assert memorizer != gold

    ablated = khop_ablated_predict(
        ctx, p1=result.p1, bridge=result.bridge, window_index=0, seed=0
    )
    assert ablated.fake_bridge != result.bridge
    assert ablated.prediction != gold
    metrics = score_predictions(
        [gold], [result.prediction], [ablated.prediction], [one_hop], [memorizer]
    )
    assert metrics["one_sided"] == 1
    assert metrics["recovery"] == 1.0
    assert metrics["recovery_ablated"] == 0.0
    assert bridge_ablation_guard(float(metrics["recovery_ablated"]))


def test_redundant_phrase_survives_wrong_bridge_and_fails_guard() -> None:
    # True bridge 3 and the seed-selected fake bridge 1 both lead to gold 9.
    ctx = [1, 9, 2, 9, 3, 9, 4, 3, 4]
    gold = 9
    result = khop_predict(ctx, {3})
    assert result is not None
    assert result.prediction == gold
    one_hop, memorizer = lookup_direct_baselines(
        ctx,
        {4: 8},
        {2: {(3, 4): 8}, 3: {(4, 3, 4): 8}},
    )

    ablated = khop_ablated_predict(
        ctx, p1=result.p1, bridge=result.bridge, window_index=2, seed=0
    )
    assert ablated.fake_bridge == 1
    assert ablated.prediction == gold
    metrics = score_predictions(
        [gold], [result.prediction], [ablated.prediction], [one_hop], [memorizer]
    )
    assert metrics["recovery"] == 1.0
    assert metrics["recovery_ablated"] == 1.0
    assert float(metrics["recovery_ablated"]) >= GUARD_THRESHOLD
    assert not bridge_ablation_guard(float(metrics["recovery_ablated"]))


def test_counters_and_recovery_identity() -> None:
    metrics = score_predictions(
        gold=[1, 2, 3, 4],
        pred_2hop=[1, 9, 3, 9],
        pred_ablated=[9, 9, 9, 9],
        pred_1hop=[9, 2, 9, 9],
        pred_memorizer=[9, 9, 9, 4],
    )
    assert metrics["n_fired"] == 4
    assert metrics["one_sided"] == 2
    assert metrics["reverse"] == 2
    assert metrics["agree_2hop"] == 0.5
    assert metrics["agree_1hop"] == 0.25
    assert metrics["agree_memorizer"] == 0.25
    assert metrics["agree_baseline"] == 0.5
    assert metrics["recovery"] == pytest.approx(
        (int(metrics["one_sided"]) - int(metrics["reverse"]))
        / int(metrics["n_fired"]),
        abs=1e-9,
    )
