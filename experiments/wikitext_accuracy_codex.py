"""Pre-registered Slice 6 wikitext agreement pilot (Codex lane).

Run from the repository root with ``.venv/bin/python
experiments/wikitext_accuracy_codex.py``. The driver pins the registered build
environment before importing ``wyly_lm_v5``, builds the energy package once,
checks that real firing candidates carry discriminating counts, measures all
held-out windows, and writes ``data/wikitext_accuracy_codex.json``.

Pass ``--reuse-package`` only when rerunning measurements against an energy
package already built by this driver.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ROSETTA_PY = REPO.parent / "rosetta" / "py"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(ROSETTA_PY))

BUILD_ENV = {
    "WYLY_TAG": "pythia70m",
    "WYLY_LIB": "mined",
    "WYLY_JUDGE": "cover",
    "WYLY_ONLINE": "1",
    "WYLY_COVER": "sw",
    "WYLY_DS": "wikitext",
    "WYLY_EMIT": "1",
    "WYLY_ENERGY_MODE": "1",
    "WYLY_PKG_SUFFIX": "_energy_codex",
}
PACKAGE = REPO / "data" / "wyly_expert_package_v5_energy_codex" / "manifest.json"
REPORT = REPO / "data" / "wikitext_accuracy_codex.json"
M_SWEEP = (2, 3, 4, 5)
BEAM_WIDTH = 8
VERDICT_DELTA = 0.01
MIN_COUNTS_FRACTION = 0.01


def agreement(served: Sequence[int | None], gold: Sequence[int]) -> float:
    """All-window agreement: an abstention (None) counts as incorrect."""
    if len(served) != len(gold):
        raise ValueError("served and gold must have equal lengths")
    if not gold:
        return 0.0
    return sum(pred == want for pred, want in zip(served, gold, strict=True)) / len(gold)


def delta_agreement(
    beam_served: Sequence[int | None],
    corner_served: Sequence[int | None],
    gold: Sequence[int],
) -> float:
    return agreement(beam_served, gold) - agreement(corner_served, gold)


def _coverage(served: Sequence[int | None]) -> float:
    return sum(pred is not None for pred in served) / len(served) if served else 0.0


def _set_build_environment() -> None:
    for key, value in BUILD_ENV.items():
        os.environ[key] = value


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _build_package() -> Any:
    _set_build_environment()
    import wyly_lm_v5 as v5

    v5.main()
    if not PACKAGE.exists():
        raise RuntimeError(f"energy package was not emitted at {PACKAGE}")
    return v5


def _load_v5_without_build() -> Any:
    _set_build_environment()
    import wyly_lm_v5 as v5

    if not PACKAGE.exists():
        raise FileNotFoundError(f"--reuse-package requested but {PACKAGE} does not exist")
    return v5


def _contexts_and_gold(v5: Any) -> tuple[list[list[int]], list[int], int]:
    ids, y, cls, uv, _tr, te = v5.load_ds()
    vocabulary = uv.tolist()
    windows = te.tolist()
    contexts = [[vocabulary[c] for c in ids[wi].tolist()] for wi in windows]
    gold = [vocabulary[int(cls[int(y[wi])])] for wi in windows]
    return contexts, gold, len(windows)


def _counts_fraction(raw_manifest: dict[str, Any]) -> float:
    rules = raw_manifest.get("rules", [])
    if not rules:
        return 0.0
    with_counts = sum(rule.get("counts") is not None for rule in rules)
    return with_counts / len(rules)


def _real_differentiating_examples(
    contexts: Sequence[Sequence[int]],
    idioms: Any,
    ngrams: Any,
    manifest: dict[str, Any],
    limit: int = 2,
) -> list[dict[str, Any]]:
    from beam_engine import det_rank_compare
    from serve_package import enumerate_candidates

    examples: list[dict[str, Any]] = []
    for window_offset, ctx in enumerate(contexts):
        candidates = enumerate_candidates(
            ctx,
            idioms,
            ngrams,
            manifest.get("W", 1),
            m_derived=manifest.get("_derived"),
            cmap=manifest.get("_cmap"),
        )
        counted = [candidate for candidate in candidates if candidate.counts is not None]
        for i, first in enumerate(counted):
            for second in counted[i + 1 :]:
                comparison = det_rank_compare(first.counts, second.counts)
                if comparison == 0:
                    continue
                examples.append(
                    {
                        "test_window_offset": window_offset,
                        "first": {
                            "answer": first.answer,
                            "counts": list(first.counts),
                            "rule": first.meta.get("rule", first.meta.get("k")),
                        },
                        "second": {
                            "answer": second.answer,
                            "counts": list(second.counts),
                            "rule": second.meta.get("rule", second.meta.get("k")),
                        },
                        "det_rank_compare_first_vs_second": comparison,
                    }
                )
                if len(examples) == limit:
                    return examples
    return examples


@contextmanager
def _candidate_counter() -> Iterator[dict[str, int]]:
    """Count candidate returns and enumeration calls without editing rosetta."""
    import serve_package
    import text_oracle

    original_serve = serve_package.enumerate_candidates
    original_oracle = text_oracle.enumerate_candidates
    counter = {"candidate_evals": 0, "node_expansions": 0}

    def counted(*args: Any, **kwargs: Any) -> Any:
        candidates = original_serve(*args, **kwargs)
        counter["candidate_evals"] += len(candidates)
        counter["node_expansions"] += 1
        return candidates

    serve_package.enumerate_candidates = counted
    text_oracle.enumerate_candidates = counted
    try:
        yield counter
    finally:
        serve_package.enumerate_candidates = original_serve
        text_oracle.enumerate_candidates = original_oracle


def _measure_corner(
    contexts: Sequence[Sequence[int]], idioms: Any, ngrams: Any, manifest: dict[str, Any]
) -> tuple[list[int | None], float, dict[str, int]]:
    from serve_package import decide

    corner_manifest = dict(manifest, cover="energy-beam", M=1, beam_width=1)
    with _candidate_counter() as counter:
        started = time.perf_counter_ns()
        results = [decide(ctx, idioms, ngrams, corner_manifest) for ctx in contexts]
        elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    served = [None if result is None else int(result["answer"]) for result in results]
    return served, elapsed_seconds, dict(counter)


def _beam_result_to_answer(result: dict[str, Any], oracle: Any) -> int | None:
    commit = result["committed_value"]
    if commit is None:
        fallback = oracle.fallback()
        return None if fallback is None else int(fallback["answer"])
    if isinstance(commit, dict):
        return int(commit["answer"])
    token, _meta = commit
    return int(token)


def _measure_beam(
    contexts: Sequence[Sequence[int]],
    idioms: Any,
    ngrams: Any,
    manifest: dict[str, Any],
    M: int,
) -> tuple[list[int | None], float, dict[str, int], int, int]:
    from beam_engine import beam_decode
    from text_oracle import TextOracle

    served: list[int | None] = []
    prune_events = 0
    n_steps = 0
    with _candidate_counter() as counter:
        started = time.perf_counter_ns()
        for ctx in contexts:
            oracle = TextOracle(
                ctx,
                idioms,
                ngrams,
                manifest.get("W", 1),
                m_derived=manifest.get("_derived"),
                cmap=manifest.get("_cmap"),
                m_tau=manifest.get("_tau"),
            )
            result = beam_decode(oracle, M, BEAM_WIDTH, "decide_energy_v1", 0)
            served.append(_beam_result_to_answer(result, oracle))
            prune_events += int(result["prune_events"])
            n_steps += int(result["n_steps"])
        elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    return served, elapsed_seconds, dict(counter), prune_events, n_steps


def _verdict(per_m: dict[str, dict[str, Any]]) -> tuple[str, list[int]]:
    regressions = [int(M) for M, row in per_m.items() if row["delta_agree"] <= -VERDICT_DELTA]
    if regressions:
        return "REGRESSION", regressions
    gains = [int(M) for M, row in per_m.items() if row["delta_agree"] >= VERDICT_DELTA]
    if gains:
        return "GAIN", gains
    return "NULL", []


def _write_report(report: dict[str, Any]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _print_scoreboard(report: dict[str, Any]) -> None:
    print("\n=== Slice 6 wikitext accuracy — CODEX lane ===")
    print(
        f"windows={report['provenance']['n_test_windows']} "
        f"rules={report['provenance']['n_rules']} "
        f"counts_fraction={report['non_degeneracy']['fraction_rules_with_counts']:.6f}"
    )
    if report.get("verdict") is None:
        print(f"STOP: {report['status']}")
        return
    corner = report["corner"]
    print(
        f"M=1  agree={corner['agree_corner']:.6f} "
        f"cover={corner['cover_corner']:.6f} latency=1.000x"
    )
    print("  M      agree      delta      cover    prune_bite  latency  candidate  nodes")
    for M in M_SWEEP:
        row = report["per_m"][str(M)]
        print(
            f"  {M}  {row['agree_beam']:.6f}  {row['delta_agree']:+.6f}  "
            f"{row['cover_beam']:.6f}  {row['prune_bite']:.6f}  "
            f"{row['latency_ratio']:.3f}x  {row['candidate_eval_ratio']:.3f}x  "
            f"{row['node_expansion_ratio']:.3f}x"
        )
    deciding = report["deciding_M"] or "none"
    print(f"VERDICT: {report['verdict']} (deciding M: {deciding})")
    print(f"report: {REPORT}")


def run(*, reuse_package: bool = False) -> dict[str, Any]:
    v5 = _load_v5_without_build() if reuse_package else _build_package()

    from serve_package import load_package

    raw_manifest = json.loads(PACKAGE.read_text())
    idioms, ngrams, manifest = load_package(PACKAGE)
    contexts, gold, n_test_windows = _contexts_and_gold(v5)
    fraction = _counts_fraction(raw_manifest)
    examples = _real_differentiating_examples(contexts, idioms, ngrams, manifest)
    non_degeneracy = {
        "fraction_rules_with_counts": fraction,
        "minimum_material_fraction": MIN_COUNTS_FRACTION,
        "det_rank_differentiates": bool(examples),
        "differentiating_examples": examples,
    }
    provenance = {
        "package_manifest": str(PACKAGE.relative_to(REPO)),
        "n_rules": int(raw_manifest.get("n_rules", len(raw_manifest.get("rules", [])))),
        "env_config": BUILD_ENV,
        "git_commit": _git_commit(),
        "n_test_windows": n_test_windows,
    }
    if fraction < MIN_COUNTS_FRACTION or not examples:
        report = {
            "metric_definition": "all held-out windows; abstentions count as incorrect",
            "provenance": provenance,
            "non_degeneracy": non_degeneracy,
            "status": "DEGENERATE_SUBSTRATE",
            "verdict": None,
            "corner": None,
            "per_m": {},
            "deciding_M": [],
        }
        _write_report(report)
        _print_scoreboard(report)
        return report

    corner_served, corner_seconds, corner_work = _measure_corner(
        contexts, idioms, ngrams, manifest
    )
    agree_corner = agreement(corner_served, gold)
    cover_corner = _coverage(corner_served)
    per_m: dict[str, dict[str, Any]] = {}
    for M in M_SWEEP:
        beam_served, beam_seconds, beam_work, prune_events, n_steps = _measure_beam(
            contexts, idioms, ngrams, manifest, M
        )
        per_m[str(M)] = {
            "agree_beam": agreement(beam_served, gold),
            "delta_agree": delta_agreement(beam_served, corner_served, gold),
            "cover_beam": _coverage(beam_served),
            "prune_bite": prune_events / n_steps if n_steps else 0.0,
            "prune_events": prune_events,
            "n_steps": n_steps,
            "latency_seconds_per_token": beam_seconds / n_test_windows,
            "latency_ratio": beam_seconds / corner_seconds,
            "candidate_evals": beam_work["candidate_evals"],
            "candidate_eval_ratio": (
                beam_work["candidate_evals"] / corner_work["candidate_evals"]
                if corner_work["candidate_evals"]
                else None
            ),
            "node_expansions": beam_work["node_expansions"],
            "node_expansion_ratio": (
                beam_work["node_expansions"] / corner_work["node_expansions"]
                if corner_work["node_expansions"]
                else None
            ),
        }
    verdict, deciding_M = _verdict(per_m)
    report = {
        "metric_definition": "all held-out windows; abstentions count as incorrect",
        "prune_bite_definition": "sum(prune_events) / sum(n_steps) across windows",
        "provenance": provenance,
        "non_degeneracy": non_degeneracy,
        "corner": {
            "agree_corner": agree_corner,
            "cover_corner": cover_corner,
            "latency_seconds_per_token": corner_seconds / n_test_windows,
            "candidate_evals": corner_work["candidate_evals"],
            "node_expansions": corner_work["node_expansions"],
        },
        "per_m": per_m,
        "verdict": verdict,
        "deciding_M": deciding_M,
        "status": "COMPLETE",
    }
    _write_report(report)
    _print_scoreboard(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-package",
        action="store_true",
        help="skip mining and reuse this driver's existing energy package",
    )
    args = parser.parse_args()
    run(reuse_package=args.reuse_package)


if __name__ == "__main__":
    main()
