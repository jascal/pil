"""Reproduce the frozen v5 admission composition under the native WYLY_SEED knob.

Default usage runs seed 0 twice (unset, then explicit) and checks both against the frozen seed-0
record. ``--seed N`` runs once and checks a matching B3 record when one exists. Add
``--emit-manifest PATH`` to emit from the captured seed-0 model without another regeneration.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

CORE_SW_TOLERANCE = 1e-4


def _recipe() -> dict[str, str]:
    from experiments.campaign_seed_sweep import _BOOTSTRAP_RECIPE, _RECIPE

    assert _RECIPE == _BOOTSTRAP_RECIPE
    return dict(_RECIPE)


def _assert_matches(actual: dict, reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text())
    actual_core = actual["core_sw"]
    reference_core = reference["core_sw"]
    assert actual_core.keys() == reference_core.keys(), reference_path
    drift = {
        key: actual_core[key] - reference_core[key]
        for key in ("agree", "cover", "agree_fired")
    }
    print(
        f"core_sw drift vs {reference_path}: "
        + " ".join(f"{key}={value:+.2e}" for key, value in drift.items())
        + f" (tolerance {CORE_SW_TOLERANCE:g})"
    )
    assert actual["admitted_names"] == reference["admitted_names"], reference_path
    if any(abs(value) > CORE_SW_TOLERANCE for value in drift.values()):
        raise AssertionError(f"{reference_path}: core_sw drift exceeds tolerance")


def _worker(result_path: Path, emit_manifest: Path | None) -> int:
    import random

    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    import wyly_lm_v5 as v5

    original = v5.core_cover_sw
    captured = {}

    def capture(model, rules, ids, yv, cls, idxs, **kwargs):
        captured["model"] = model
        captured["rules"] = list(rules)
        return original(model, rules, ids, yv, cls, idxs, **kwargs)

    temporary_state = result_path.with_suffix(".state.pt")
    v5.STATE = temporary_state
    v5.core_cover_sw = capture
    try:
        v5.main()
    finally:
        v5.core_cover_sw = original
        temporary_state.unlink(missing_ok=True)
    if not captured:
        raise RuntimeError("core_cover_sw capture wrapper was never called")

    model = captured["model"]
    rules = list(model.rules)
    ids, y, cls, uv, _tr, te = v5.load_ds()
    core_sw = original(model, rules, ids, cls[y], cls, te)
    result = {
        "admitted_names": [name for name, _fn in rules],
        "core_sw": core_sw,
        "internal_seed": v5.WYLY_SEED,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if emit_manifest is not None:
        manifest, _skipped = v5.emit_full(model, cls, uv, v5.load_codec(), len(uv))
        emit_manifest.parent.mkdir(parents=True, exist_ok=True)
        emit_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


def _run_once(seed: int | None, result_path: Path, emit_manifest: Path | None = None) -> dict:
    env = {**os.environ, **_recipe(), "PYTHONHASHSEED": "0"}
    env.pop("WYLY_EMIT", None)
    if seed is None:
        env.pop("WYLY_SEED", None)
    else:
        env["WYLY_SEED"] = str(seed)
    command = [sys.executable, __file__, "--_worker", str(result_path)]
    if emit_manifest is not None:
        command.extend(["--emit-manifest", str(emit_manifest)])
    subprocess.run(command, cwd=REPO, env=env, check=True)
    return json.loads(result_path.read_text())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="run once with WYLY_SEED=N")
    parser.add_argument(
        "--emit-manifest", type=Path,
        help="write a manifest from the captured WYLY_SEED=0 model (default mode only)",
    )
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.emit_manifest is not None and args.seed is not None:
        parser.error("--emit-manifest is supported only in default seed-0 mode")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args._worker is not None:
        return _worker(args._worker, args.emit_manifest)

    with tempfile.TemporaryDirectory(prefix="wyly-repro-") as tmp:
        tmpdir = Path(tmp)
        if args.seed is not None:
            actual = _run_once(args.seed, tmpdir / "result.json")
            matching = REPO / "data" / "seed_sweep" / f"run_b3_seed{args.seed}_rep0.json"
            if matching.exists():
                _assert_matches(actual, matching)
                print(f"matched {matching.relative_to(REPO)}")
            print(json.dumps(actual, indent=2, sort_keys=True))
            return 0

        reference = REPO / "data" / "seed_sweep" / "run_seed0_rep0.json"
        unset = _run_once(None, tmpdir / "unset.json")
        explicit = _run_once(0, tmpdir / "explicit.json", args.emit_manifest)
        _assert_matches(unset, reference)
        _assert_matches(explicit, reference)
        assert unset["admitted_names"] == explicit["admitted_names"]
        assert unset["core_sw"] == explicit["core_sw"]
        print(f"unset and explicit WYLY_SEED=0 matched {reference.relative_to(REPO)}")
        if args.emit_manifest is not None:
            print(f"manifest: {args.emit_manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
