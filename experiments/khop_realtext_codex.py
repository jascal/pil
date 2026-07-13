"""Pre-registered B1 probe of certified 2-hop chains on real text.

Run from the repository root with ``.venv/bin/python
experiments/khop_realtext_codex.py``. The full v5 ``tr`` split is used as the
FIT pool for both bridge mining and baseline fitting; all metrics use ``te``.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

# wyly_lm_v5 computes its dataset paths and tags at import time.
os.environ.setdefault("WYLY_TAG", "pythia70m")
os.environ.setdefault("WYLY_DS", "wikitext")

import torch
import wyly_lm_v5 as v5

EDGE_MINSUPP = 2
MIN_IN = 2
MIN_OUT = 2
RECOVERY_THRESHOLD = 0.005
GUARD_THRESHOLD = 0.005
MIN_FIRED = 30
KGRAMS = (2, 3)
SEED = int(os.environ.get("WYLY_SEED", "0"))
REPORT = REPO / "data" / "khop_realtext_codex.json"


@dataclass(frozen=True)
class KhopResult:
    """A firing khop result, including the hop-1 state reused by ablation."""

    prediction: int
    p1: int
    bridge: int


@dataclass(frozen=True)
class AblatedResult:
    """The deterministic wrong-bridge result; prediction is None on abstention."""

    prediction: int | None
    fake_bridge: int | None


def khop_predict(ctx: Sequence[int], bridge_set: set[int]) -> KhopResult | None:
    """Certified 2-hop chain with the real-text gate moved to the hop-1 bridge."""
    width = len(ctx)
    if width < 2:
        return None
    query = ctx[-1]
    cand1 = [j for j in range(width - 1) if ctx[j] == query]
    if not cand1:
        return None
    p1 = max(cand1)
    bridge = ctx[min(p1 + 1, width - 1)]
    if bridge not in bridge_set:
        return None
    cand2 = [
        j
        for j in range(width - 1)
        if ctx[j] == bridge and j != p1 + 1
    ]
    if not cand2:
        return None
    p2 = max(cand2)
    return KhopResult(int(ctx[min(p2 + 1, width - 1)]), p1, int(bridge))


def khop_ablated_predict(
    ctx: Sequence[int],
    *,
    p1: int,
    bridge: int,
    window_index: int,
    seed: int = SEED,
) -> AblatedResult:
    """Re-run only hop 2 with the registered deterministic wrong bridge."""
    width = len(ctx)
    alternatives = sorted(
        {int(ctx[j]) for j in range(max(width - 1, 0)) if ctx[j] != bridge}
    )
    if not alternatives:
        return AblatedResult(None, None)
    fake_bridge = alternatives[
        random.Random(seed + window_index).randrange(len(alternatives))
    ]
    cand2 = [
        j
        for j in range(width - 1)
        if ctx[j] == fake_bridge and j != p1 + 1
    ]
    if not cand2:
        return AblatedResult(None, fake_bridge)
    p2 = max(cand2)
    return AblatedResult(int(ctx[min(p2 + 1, width - 1)]), fake_bridge)


def lookup_direct_baselines(
    ctx: Sequence[int],
    one_hop_table: Mapping[int, int],
    memorizer_tables: Mapping[int, Mapping[tuple[int, ...], int]],
) -> tuple[int | None, int | None]:
    """Small-table mirror used by unit tests; memorizer is longest-suffix-wins."""
    if not ctx:
        return None, None
    one_hop = one_hop_table.get(int(ctx[-1]))
    memorizer = None
    for k in sorted(memorizer_tables, reverse=True):
        if len(ctx) < k:
            continue
        memorizer = memorizer_tables[k].get(tuple(int(x) for x in ctx[-k:]))
        if memorizer is not None:
            break
    return one_hop, memorizer


def score_predictions(
    gold: Sequence[int],
    pred_2hop: Sequence[int | None],
    pred_ablated: Sequence[int | None],
    pred_1hop: Sequence[int | None],
    pred_memorizer: Sequence[int | None],
) -> dict[str, int | float]:
    """Compute registered fired-subset metrics and the one-sided identity."""
    lengths = {
        len(gold),
        len(pred_2hop),
        len(pred_ablated),
        len(pred_1hop),
        len(pred_memorizer),
    }
    if len(lengths) != 1:
        raise ValueError("all prediction and gold sequences must have equal length")
    n = len(gold)
    correct_2hop = [pred == want for pred, want in zip(pred_2hop, gold, strict=True)]
    correct_ablated = [
        pred == want for pred, want in zip(pred_ablated, gold, strict=True)
    ]
    correct_1hop = [pred == want for pred, want in zip(pred_1hop, gold, strict=True)]
    correct_mem = [
        pred == want for pred, want in zip(pred_memorizer, gold, strict=True)
    ]
    correct_baseline = [
        one or mem for one, mem in zip(correct_1hop, correct_mem, strict=True)
    ]
    one_sided = sum(
        two and not baseline
        for two, baseline in zip(correct_2hop, correct_baseline, strict=True)
    )
    reverse = sum(
        not two and baseline
        for two, baseline in zip(correct_2hop, correct_baseline, strict=True)
    )

    def mean(values: Sequence[bool]) -> float:
        return sum(values) / n if n else 0.0

    agree_2hop = mean(correct_2hop)
    agree_ablated = mean(correct_ablated)
    agree_baseline = mean(correct_baseline)
    recovery = agree_2hop - agree_baseline
    recovery_ablated = agree_ablated - agree_baseline
    identity = (one_sided - reverse) / n if n else 0.0
    if abs(recovery - identity) > 1e-9:
        raise AssertionError(
            f"recovery identity failed: {recovery=} vs {identity=}"
        )
    return {
        "recovery": recovery,
        "recovery_ablated": recovery_ablated,
        "agree_2hop": agree_2hop,
        "agree_2hop_ablated": agree_ablated,
        "agree_baseline": agree_baseline,
        "agree_1hop": mean(correct_1hop),
        "agree_memorizer": mean(correct_mem),
        "one_sided": one_sided,
        "reverse": reverse,
        "n_fired": n,
    }


def bridge_ablation_guard(recovery_ablated: float) -> bool:
    """The registered guard holds only when wrong-bridge recovery misses the bar."""
    return recovery_ablated < GUARD_THRESHOLD


def registered_verdict(
    *, recovery: float, recovery_ablated: float, n_fired: int, bridge_set_size: int
) -> str:
    if n_fired < MIN_FIRED or bridge_set_size == 0:
        return "INCONCLUSIVE_TINY_SUBSET"
    if recovery >= RECOVERY_THRESHOLD and bridge_ablation_guard(recovery_ablated):
        return "FIRES"
    return "DEAD"


def mine_bridge_set(
    ids: torch.Tensor,
    yv: torch.Tensor,
    tr: torch.Tensor,
    vocab: int,
) -> set[int]:
    """Mine registered distinct in/out-supported bridge tokens from the full FIT pool."""
    windows = ids[tr]
    source = torch.cat([windows[:, :-1].reshape(-1), windows[:, -1]])
    target = torch.cat([windows[:, 1:].reshape(-1), yv[tr]])
    encoded = source * vocab + target
    edges, counts = encoded.unique(return_counts=True)
    kept = edges[counts >= EDGE_MINSUPP]
    kept_source = kept // vocab
    kept_target = kept % vocab
    in_support = torch.bincount(kept_target, minlength=vocab)
    out_support = torch.bincount(kept_source, minlength=vocab)
    bridges = torch.where((in_support >= MIN_IN) & (out_support >= MIN_OUT))[0]
    return set(int(value) for value in bridges.tolist())


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _empty_metrics() -> dict[str, int | float]:
    return score_predictions([], [], [], [], [])


def run_pilot() -> dict[str, Any]:
    """Run the fixed pilot once and return its JSON-serializable scorecard."""
    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    vocab = len(uv)
    bridge_set = mine_bridge_set(ids, yv, tr, vocab)

    fired: list[tuple[int, int, list[int], KhopResult]] = []
    for test_offset, raw_index in enumerate(te.tolist()):
        ctx = [int(value) for value in ids[raw_index].tolist()]
        result = khop_predict(ctx, bridge_set)
        if result is not None:
            fired.append((test_offset, int(raw_index), ctx, result))

    metrics = _empty_metrics()
    if fired:
        table1, _ = v5.fit_kgram(ids, yv, tr, 1, vocab)
        pred1 = v5.mir_kgram(table1, 1, vocab)(ids[te])
        mem_predictions: dict[int, torch.Tensor] = {}
        for k in KGRAMS:
            table, _ = v5.fit_kgram(ids, yv, tr, k, vocab)
            mem_predictions[k] = v5.mir_kgram(table, k, vocab)(ids[te])
        pred_mem = torch.where(
            mem_predictions[3] >= 0, mem_predictions[3], mem_predictions[2]
        )

        gold: list[int] = []
        pred_2hop: list[int] = []
        pred_ablated: list[int | None] = []
        pred_1hop: list[int] = []
        pred_memorizer: list[int] = []
        for test_offset, raw_index, ctx, result in fired:
            gold.append(int(yv[raw_index]))
            pred_2hop.append(result.prediction)
            pred_ablated.append(
                khop_ablated_predict(
                    ctx,
                    p1=result.p1,
                    bridge=result.bridge,
                    window_index=raw_index,
                    seed=SEED,
                ).prediction
            )
            pred_1hop.append(int(pred1[test_offset]))
            pred_memorizer.append(int(pred_mem[test_offset]))
        metrics = score_predictions(
            gold, pred_2hop, pred_ablated, pred_1hop, pred_memorizer
        )

    verdict = registered_verdict(
        recovery=float(metrics["recovery"]),
        recovery_ablated=float(metrics["recovery_ablated"]),
        n_fired=int(metrics["n_fired"]),
        bridge_set_size=len(bridge_set),
    )
    return {
        "verdict": verdict,
        **metrics,
        "n_te": len(te),
        "bridge_set_size": len(bridge_set),
        "bridge_mining_criterion": {
            "edge_minsupp": EDGE_MINSUPP,
            "min_in_distinct_predecessors": MIN_IN,
            "min_out_distinct_successors": MIN_OUT,
            "edges": (
                "all adjacent in-window pairs plus each teacher-decision tail pair; "
                "supports count distinct kept directed edges"
            ),
        },
        "bar": {
            "recovery_threshold": RECOVERY_THRESHOLD,
            "guard_threshold": GUARD_THRESHOLD,
        },
        "provenance": {
            "WYLY_SEED": SEED,
            "WYLY_TAG": v5.TAG,
            "WYLY_DS": v5.DS,
            "git_commit": _git_commit(),
            "n_fit": len(tr),
            "fit_pool": (
                "full v5 tr split (85%): used for bridge-set mining and all "
                "baseline-table fitting; no admission/threshold-tuning val split"
            ),
            "eval_pool": "v5 te split (15%) only",
        },
    }


def _print_scoreboard(report: Mapping[str, Any]) -> None:
    print("\n=== khop-in-real-text · Codex lane ===")
    for key in (
        "verdict",
        "recovery",
        "recovery_ablated",
        "agree_2hop",
        "agree_2hop_ablated",
        "agree_baseline",
        "agree_1hop",
        "agree_memorizer",
        "one_sided",
        "reverse",
        "n_fired",
        "n_te",
        "bridge_set_size",
    ):
        print(f"{key:>23}: {report[key]}")
    guard = bridge_ablation_guard(float(report["recovery_ablated"]))
    print(f"{'bridge-ablation guard':>23}: {'HOLDS' if guard else 'FAILS'}")
    criterion = report["bridge_mining_criterion"]
    print(
        f"{'bridge thresholds':>23}: edge>={criterion['edge_minsupp']}, "
        f"in>={criterion['min_in_distinct_predecessors']}, "
        f"out>={criterion['min_out_distinct_successors']}"
    )
    bar = report["bar"]
    print(
        f"{'registered bars':>23}: recovery>={bar['recovery_threshold']}, "
        f"ablated<{bar['guard_threshold']}"
    )
    provenance = report["provenance"]
    print(
        f"{'provenance':>23}: seed={provenance['WYLY_SEED']}, "
        f"tag={provenance['WYLY_TAG']}, ds={provenance['WYLY_DS']}, "
        f"git={provenance['git_commit']}"
    )
    print(f"{'FIT pool':>23}: {provenance['fit_pool']}")
    print(f"{'eval pool':>23}: {provenance['eval_pool']}")


def main() -> None:
    report = run_pilot()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _print_scoreboard(report)
    print(f"{'report':>23}: {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
