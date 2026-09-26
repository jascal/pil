"""Preregistered directional calibration: develop, lock selection, then confirm.

Run ``dev`` before creating EVAL2. ``confirm`` reads the saved dev selection,
refits on the 450 seen decisions, and evaluates the fresh EVAL2 dump once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.campaign_certified_early_exit import evaluate_split  # noqa: E402
from experiments.campaign_restricted_early_exit_scale import process  # noqa: E402
from pil.directional_calibration import CANDIDATES, bounds, candidate, fit, verdict, wilson  # noqa: E402
from pil.early_exit import Bundle  # noqa: E402


def model_data(bundle: Path, dumps: Path, eval2: Path | None = None):
    b = Bundle(bundle)
    theta = b.f32("norm")
    meta = json.loads((dumps / "meta.json").read_text())
    cal = process(dumps / "cal_canonical.dump.jsonl", meta, b, theta)
    ev = process(dumps / "eval_canonical.dump.jsonl", meta, b, theta)
    fresh = None
    if eval2 is not None:
        fresh = process(eval2 / "eval2_canonical.dump.jsonl",
                        json.loads((eval2 / "meta.json").read_text()), b, theta)
    layer = b.d * (b.d + 2 * b.n_kv * b.head_dim + b.d) + 3 * b.d * b.d_ff
    c_chk = 6 * b.d / layer
    all_records = cal + ev + (fresh or [])
    sr = max(r.s_resid for r, _ in all_records)
    if sr >= 1e-3:
        raise RuntimeError(f"s-recovery failed: {sr}")
    return cal, ev, fresh, c_chk, sr, b.n_layer


def score(recs, fitted, c_chk):
    return evaluate_split([r for r, _ in recs], bounds(recs, fitted), c_chk)


def cmd_dev(a):
    result = {"stage": "development", "prereg": "SIGNED 2026-09-26 14:29 UTC", "models": {}}
    for tag, bundle, dumps in (("0.5b", a.bundle_05, a.dumps_05), ("3b", a.bundle_3b, a.dumps_3b)):
        cal, ev, _, cost, sr, layers = model_data(bundle, dumps)
        if len(cal) != 150 or len(ev) != 300:
            raise RuntimeError(f"{tag}: expected CAL 150/EVAL 300, got {len(cal)}/{len(ev)}")
        arms = {kind: score(ev, fit(cal, kind), cost) for kind in CANDIDATES}
        result["models"][tag] = {"layers": layers, "s_resid_max": sr, "check_cost": cost,
                                 "candidates": arms}
    result["selected"] = candidate(result["models"]["3b"]["candidates"])
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    for tag, row in result["models"].items():
        print(f"[{tag}] s-recovery max={row['s_resid_max']:.3g}")
        for kind, m in row["candidates"].items():
            print(f"  {kind}: coverage={m['coverage']:.3f} certified={round(m['N']*m['coverage'])} "
                  f"violations={m['violations']} net={m['net_every']:+.3f}")
    print(f"SELECTED: {result['selected'] or 'NO CANDIDATE'}")


def cmd_confirm(a):
    dev = json.loads(Path(a.dev).read_text())
    selected = dev["selected"]
    if selected not in CANDIDATES:
        raise RuntimeError("no selected candidate; EVAL2 must not be evaluated")
    result = {"stage": "confirmation", "selected": selected, "models": {}}
    for tag, bundle, dumps, fresh_dir in (("0.5b", a.bundle_05, a.dumps_05, a.eval2_05),
                                          ("3b", a.bundle_3b, a.dumps_3b, a.eval2_3b)):
        cal, ev, fresh, cost, sr, layers = model_data(bundle, dumps, fresh_dir)
        if len(cal) + len(ev) != 450 or fresh is None or len(fresh) != 150:
            raise RuntimeError(f"{tag}: expected seen 450/fresh 150")
        fitted = fit(cal + ev, selected)
        measured = score(fresh, fitted, cost)
        oracle = evaluate_split([r for r, _ in fresh], np.stack([push[:-1] for _, push in fresh]), cost)
        sound = oracle["violations"] == 0 and sr < 1e-3
        cert_count = round(measured["N"] * measured["coverage"])
        report = {"layers": layers, "s_resid_max": sr, "check_cost": cost, "oracle_D": oracle,
                  "calibrated_D": measured, "soundness_gate": sound, "verdict": verdict(measured, sound),
                  "coverage_wilson95": wilson(cert_count, measured["N"]),
                  "violation_wilson95": wilson(measured["violations"], cert_count)}
        lib = fresh_dir / "library_decisions.json"
        if lib.exists():
            libd = json.loads(lib.read_text())
            meta = json.loads((fresh_dir / "meta.json").read_text())
            agree = [libd[r.sid]["value"] == meta[r.sid]["options"][r.pred] for r, _ in fresh]
            report["library_fieldrun_agreement"] = float(np.mean(agree))
        result["models"][tag] = report
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    for tag, row in result["models"].items():
        m = row["calibrated_D"]
        print(f"[{tag}] {row['verdict']}: coverage={m['coverage']:.3f}, "
              f"violations={m['violations']}/{round(m['N']*m['coverage'])}, "
              f"net={m['net_every']:+.3f}; oracle violations={row['oracle_D']['violations']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("dev", "confirm"):
        s = sub.add_parser(name)
        s.add_argument("--bundle-05", type=Path, required=True)
        s.add_argument("--bundle-3b", type=Path, required=True)
        s.add_argument("--dumps-05", type=Path, default=Path("runs/ree"))
        s.add_argument("--dumps-3b", type=Path, default=Path("runs/ree3b"))
        s.add_argument("--out", type=Path, required=True)
        if name == "confirm":
            s.add_argument("--dev", type=Path, required=True)
            s.add_argument("--eval2-05", type=Path, required=True)
            s.add_argument("--eval2-3b", type=Path, required=True)
    a = p.parse_args()
    if a.command == "dev":
        cmd_dev(a)
    else:
        cmd_confirm(a)


if __name__ == "__main__":
    main()
