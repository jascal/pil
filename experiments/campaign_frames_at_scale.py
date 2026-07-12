"""Registered mined-frames-at-scale campaign (slice #86).

The import-safe helpers are fixture tested.  The real entry point deliberately regenerates each
scale in its own process because ``wyly_lm_v5`` binds its teacher tag at import time.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# v5 reads its recipe at import time.  A child receives WYLY_TAG from the parent; an ordinary
# fixture import uses the scale of interest but never loads its dataset.
os.environ.setdefault("WYLY_TAG", "pythia2.8b")
os.environ.setdefault("WYLY_DS", "wikitext")
os.environ.setdefault("WYLY_LIB", "mined")
os.environ.setdefault("WYLY_JUDGE", "cover")
os.environ.setdefault("WYLY_ONLINE", "1")
os.environ.setdefault("WYLY_COVER", "sw")
os.environ.setdefault("WYLY_SEED", "0")
for _disabled in ("WYLY_CONCEPTS", "WYLY_POINTER", "WYLY_TPOINTER", "WYLY_DX", "WYLY_CX", "WYLY_FOLDS"):
    os.environ.pop(_disabled, None)

import wyly_lm_v5 as v5  # noqa: E402

from experiments.campaign_wikitext_gated_rescue import (  # noqa: E402
    ADMIT_THRESH,
    _counts_row_fn,
    exact_discordant_p,
    split_te,
)
from pil.wyly_block import WylyBlock  # noqa: E402

OUTPUT = REPO / "data" / "frames_at_scale.json"
SCALES = {
    "410m": {"tag": "pythia410m", "reference": 0.28832247853279114},
    "2.8b": {"tag": "pythia2.8b", "reference": 0.27006158232688904},
}
BAND_TOLERANCE = 0.005
SUPPORT_GRID = (4, 10, 15)
INTERACTION_GRID = (0.0625, 0.109375, 0.15625, 0.203125, 0.25)
GRID = tuple((support, interaction) for support in SUPPORT_GRID for interaction in INTERACTION_GRID)
PAIR_SUPPORT_GATE = 10
PAIR_INTERACTION_GATE = 0.3
SAMPLE_N = 6000
HONESTY_NOTE = (
    "miner thresholds swept, judge bar untouched; single test_te evaluation; frames are "
    "template_fixed-class (not learned end-to-end); frac_induced unaffected."
)


def log(message: str = "") -> None:
    print(message, flush=True)


def registered_grid() -> tuple[tuple[int, float], ...]:
    """Return the immutable singleton support × interaction registration."""
    return GRID


def within_registered_band(scale: str, agree: float) -> bool:
    """The registered ±0.005 band is inclusive."""
    reference = SCALES[scale]["reference"]
    return reference - BAND_TOLERANCE <= float(agree) <= reference + BAND_TOLERANCE


def _deciles(values: torch.Tensor) -> list[float]:
    if not len(values):
        return []
    q = torch.arange(1, 11, dtype=torch.float64, device=values.device) / 10
    return [float(x) for x in torch.quantile(values.double(), q).cpu()]


def _cascade_row(
    label: str,
    candidates: torch.Tensor,
    cnt: torch.Tensor,
    score: torch.Tensor,
    support_gate: float,
    interaction_gate: float,
    accepted: int,
) -> dict[str, Any]:
    support_ok = cnt >= support_gate
    interaction_ok = score >= interaction_gate
    return {
        "offset": label,
        "pre_gate_candidates": int(len(candidates)),
        "support_deciles": _deciles(cnt),
        "interaction_deciles": _deciles(score),
        "post_support": int(support_ok.sum()),
        "post_interaction_after_support": int((support_ok & interaction_ok).sum()),
        "final_accepted": int(accepted),
    }


class TunableMinedGates(v5.MinedGates):
    """Exact MinedGates algorithm with only the singleton gate parameterized.

    Pair conjunctions retain the source algorithm's fixed support=10, interaction=0.3 gate.
    ``diagnostics`` records both gate cascades without changing their decisions.
    """

    def __init__(self, vocab, dev, *, support_gate=15, interaction_gate=0.25, **kwargs):
        super().__init__(vocab, dev, **kwargs)
        self.support_gate = support_gate
        self.interaction_gate = interaction_gate
        self.diagnostics: dict[str, Any] = {}

    def mine(self, model, ids, yv, cls, fitset, g, log_rows, cycle, sample_n=SAMPLE_N):
        smp = fitset[torch.randint(len(fitset), (sample_n,), generator=g)]
        _, pred = v5.core_cover_sw(
            model, model.rules, ids, yv, cls, smp, return_pred=True
        )
        err = smp[pred != yv[smp]]
        self.diagnostics = {
            "sample_n": int(len(smp)),
            "error_pool_size": int(len(err)),
            "error_pool_rate": float(len(err) / len(smp)),
            "singletons": [],
            "pairs": [],
        }
        if len(err) < 50:
            return

        tails, ytail = ids[fitset], yv[fitset]
        new1 = new2 = 0
        for o in self.offs:
            tab, _ = v5.best_per_key(
                self.A(tails[:, -o]) * self.B + tails[:, -1], ytail, 3, 0.4
            )
            v, _ = tab.lookup_conf(
                self.A(ids[err][:, -o]) * self.B + ids[err][:, -1]
            )
            good = (v == yv[err]).float()
            ua, inv = self.A(ids[err][:, -o]).unique(return_inverse=True)
            rec = torch.zeros(len(ua), device=self.dev).index_add_(0, inv, good)
            cnt = torch.zeros(len(ua), device=self.dev).index_add_(
                0, inv, torch.ones_like(good)
            )
            score = rec / cnt.clamp(min=1)
            accepted_here = 0
            for a in ua[(cnt >= self.support_gate) & (score >= self.interaction_gate)].tolist():
                if (o, a) in self.mined1:
                    continue
                self.mined1.add((o, a))
                sl = (tab.k // self.B) == a
                self.t1 = self._merge(
                    self.t1,
                    (o * self.B + a) * self.B + tab.k[sl] % self.B,
                    tab.v[sl],
                    tab.c[sl],
                )
                new1 += 1
                accepted_here += 1
            self.diagnostics["singletons"].append(
                _cascade_row(
                    str(o), ua, cnt, score, self.support_gate, self.interaction_gate,
                    accepted_here,
                )
            )

        v1, _ = self.lookup_conf_all(ids[err])
        res = err[v1 != yv[err]]
        self.diagnostics["post_singleton_residual_errors"] = int(len(res))
        if len(res) >= 50:
            for o1, o2 in self.pair_offs:
                key_fit = (
                    (self.A(tails[:, -o1]) * self.B + self.A(tails[:, -o2]))
                    * self.B
                    + tails[:, -1]
                )
                tab, _ = v5.best_per_key(key_fit, ytail, 3, 0.5)
                ke = (
                    (self.A(ids[res][:, -o1]) * self.B + self.A(ids[res][:, -o2]))
                    * self.B
                    + ids[res][:, -1]
                )
                v, _ = tab.lookup_conf(ke)
                good = (v == yv[res]).float()
                pk = (
                    self.A(ids[res][:, -o1]) * self.B
                    + self.A(ids[res][:, -o2])
                )
                ua, inv = pk.unique(return_inverse=True)
                rec = torch.zeros(len(ua), device=self.dev).index_add_(0, inv, good)
                cnt = torch.zeros(len(ua), device=self.dev).index_add_(
                    0, inv, torch.ones_like(good)
                )
                score = rec / cnt.clamp(min=1)
                oid = o1 * 16 + o2
                accepted_here = 0
                for ab in ua[
                    (cnt >= PAIR_SUPPORT_GATE) & (score >= PAIR_INTERACTION_GATE)
                ].tolist():
                    if (oid, ab) in self.mined2:
                        continue
                    self.mined2.add((oid, ab))
                    sl = (tab.k // self.B) == ab
                    self.t2 = self._merge(
                        self.t2,
                        (oid * self.B**2 + ab) * self.B + tab.k[sl] % self.B,
                        tab.v[sl],
                        tab.c[sl],
                    )
                    new2 += 1
                    accepted_here += 1
                self.diagnostics["pairs"].append(
                    _cascade_row(
                        f"{o1},{o2}", ua, cnt, score, PAIR_SUPPORT_GATE,
                        PAIR_INTERACTION_GATE, accepted_here,
                    )
                )

        self.n_frames = len(self.mined1) + len(self.mined2)
        if new1 or new2:
            log_rows.append(
                f"    sleep {cycle}: MINED {new1} singleton + {new2} conjunction frames "
                f"(total {self.n_frames}; from {len(err)} errors)"
            )


def select_grid_point(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select by the registered filter; ties prefer lower support then interaction."""
    eligible = [
        row
        for row in rows
        if row["marginal"] >= ADMIT_THRESH and row["val_regressions"] == 0
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (row["marginal"], -row["support_gate"], -row["interaction_gate"]),
    )


def resolve_selection(
    rows: list[dict[str, Any]],
    evaluate_test: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Make the no-admission branch structurally incapable of touching test_te."""
    any_admitted = any(row["marginal"] >= ADMIT_THRESH for row in rows)
    if not any_admitted:
        return {
            "outcome": "SCALE FACT",
            "reason": "no grid point admitted",
            "selected": None,
            "test": None,
        }
    selected = select_grid_point(rows)
    if selected is None:
        return {
            "outcome": "NO ZERO-REGRESSION SELECTION",
            "reason": "points admitted, but none passed the registered zero-regression filter",
            "selected": None,
            "test": None,
        }
    return {"outcome": "selected", "selected": selected, "test": evaluate_test(selected)}


def verdict(test_marginal: float, pvalue: float, test_regressions: int) -> str:
    """Resolve the registered verdict, including its documented regression-clause gap.

    The literal rules omit significant >=.003 effects with regressions.  We conservatively call
    that branch MIDDLE: significant recovery exists, but it is not a clean MINER FACT.
    """
    if pvalue >= 0.05:
        return "SCALE FACT"
    if test_marginal >= 0.003 and test_regressions <= 0:
        return "MINER FACT"
    return "MIDDLE"


def _fit_region(tr: torch.Tensor) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    val = tr[torch.randperm(len(tr), generator=g)[: v5.v2.NVAL]]
    return tr[~torch.isin(tr, val)]


def _conf_fns(rules) -> dict[str, Any]:
    out = {name: v5.CONF_FNS[name] for name, _ in rules if name in v5.CONF_FNS}
    for name, fn in rules:
        if name not in out:
            scalar = float(v5.RULE_CONF.get(name, 0.0))
            out[name] = lambda w, fn=fn, scalar=scalar: (
                fn(w), torch.full((len(w),), scalar, device=w.device)
            )
    return out


def _score_candidate(model, rules, mined, ids, yv, cls, idxs) -> dict[str, Any]:
    conf = _conf_fns(rules)
    block = WylyBlock(0, "frames-at-scale")
    block.rules = list(rules)
    block.conf_fns = dict(conf)
    counts_fn = _counts_row_fn(model, cls)
    baseline, _ = block.predict_cover(ids[idxs], counts_row_fn=counts_fn)
    block.rules.append(("mined frames (threshold sweep)", mined.mirror()))
    block.conf_fns["mined frames (threshold sweep)"] = mined.lookup_conf_all
    candidate, _ = block.predict_cover(ids[idxs], counts_row_fn=counts_fn)
    target = yv[idxs]
    base_correct = baseline == target
    cand_correct = candidate == target
    return {
        "baseline_agree": float(base_correct.float().mean()),
        "candidate_agree": float(cand_correct.float().mean()),
        "marginal": float(cand_correct.float().mean() - base_correct.float().mean()),
        "regressions": int((base_correct & ~cand_correct).sum()),
        "baseline_pred": baseline,
        "candidate_pred": candidate,
        "base_correct": base_correct,
        "candidate_correct": cand_correct,
    }


def _historical_mined() -> dict[str, Any] | None:
    spec = v5.EMIT_INFO.get("mined frames (online)")
    if not spec or spec[0] != "mined":
        return None
    mined = spec[1]
    return {
        "n_frames": int(mined.n_frames),
        "singletons": sorted([list(x) for x in mined.mined1]),
        "pairs": sorted([list(x) for x in mined.mined2]),
    }


def _regenerate():
    original = v5.core_cover_sw
    captured: dict[str, Any] = {}

    def capture(model, rules, ids, yv, cls, idxs, **kwargs):
        captured["model"] = model
        captured["rules"] = list(rules)
        return original(model, rules, ids, yv, cls, idxs, **kwargs)

    v5.core_cover_sw = capture
    try:
        v5.main()
    finally:
        v5.core_cover_sw = original
    if not captured:
        raise RuntimeError("core_cover_sw capture wrapper was never called")
    return captured["model"], captured["rules"], original


def _mine_once(model, rules, ids, yv, cls, uv, fit, support, interaction):
    model.rules = list(rules)
    mined = TunableMinedGates(
        len(uv), v5.DEV,
        support_gate=support, interaction_gate=interaction,
    )
    mining_log: list[str] = []
    mined.mine(
        model, ids, yv, cls, fit, torch.Generator().manual_seed(0), mining_log, 0,
        sample_n=SAMPLE_N,
    )
    return mined, mining_log


def _child_run(scale: str, destination: Path) -> int:
    model, captured_rules, original_cover = _regenerate()
    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    core = original_cover(model, captured_rules, ids, yv, cls, te)
    report: dict[str, Any] = {
        "scale": scale,
        "core_sw": core,
        "band_pass": within_registered_band(scale, core["agree"]),
        "rules": [name for name, _ in captured_rules],
        "historical_default_mined": _historical_mined(),
    }
    if not report["band_pass"]:
        report["stopped"] = "registered per-scale band check failed"
        destination.write_text(json.dumps(report, indent=2, sort_keys=True))
        return 2

    val_te, test_te = split_te(te)
    fit = _fit_region(tr)
    # Part A is explicitly descriptive of each captured model, including 410m's admitted family.
    part_a_mined, _ = _mine_once(
        model, captured_rules, ids, yv, cls, uv, fit, 15, 0.25
    )
    report["part_a"] = part_a_mined.diagnostics

    # Remove 410m's admitted mined family only for its separate admission-reproduction row.
    baseline_rules = [
        rule for rule in captured_rules if not rule[0].startswith("mined frames")
    ]
    default_mined, default_log = _mine_once(
        model, baseline_rules, ids, yv, cls, uv, fit, 15, 0.25
    )
    default_val = _score_candidate(
        model, baseline_rules, default_mined, ids, yv, cls, val_te
    )
    report["split"] = {"val_te": len(val_te), "test_te": len(test_te), "seed": 0}
    report["default_remining"] = {
        "n_frames": default_mined.n_frames,
        "singletons": sorted([list(x) for x in default_mined.mined1]),
        "pairs": sorted([list(x) for x in default_mined.mined2]),
        "val_marginal": default_val["marginal"],
        "val_regressions": default_val["regressions"],
        "clears_judge_bar": default_val["marginal"] >= ADMIT_THRESH,
        "log": default_log,
    }

    if scale == "2.8b":
        grid_rows: list[dict[str, Any]] = []
        mined_by_point: dict[tuple[int, float], TunableMinedGates] = {}
        for support, interaction in registered_grid():
            mined, _ = _mine_once(
                model, baseline_rules, ids, yv, cls, uv, fit, support, interaction
            )
            scored = _score_candidate(model, baseline_rules, mined, ids, yv, cls, val_te)
            row = {
                "support_gate": support,
                "interaction_gate": interaction,
                "baseline_val_agree": scored["baseline_agree"],
                "candidate_val_agree": scored["candidate_agree"],
                "marginal": scored["marginal"],
                "val_regressions": scored["regressions"],
                "n_frames": mined.n_frames,
                "singletons": sorted([list(x) for x in mined.mined1]),
                "pairs": sorted([list(x) for x in mined.mined2]),
            }
            grid_rows.append(row)
            mined_by_point[(support, interaction)] = mined

        def evaluate_test(point):
            mined = mined_by_point[(point["support_gate"], point["interaction_gate"])]
            scored = _score_candidate(model, baseline_rules, mined, ids, yv, cls, test_te)
            b = int((scored["base_correct"] & ~scored["candidate_correct"]).sum())
            c = int((scored["candidate_correct"] & ~scored["base_correct"]).sum())
            pvalue = exact_discordant_p(b, c)
            result = {
                "baseline_test_agree": scored["baseline_agree"],
                "candidate_test_agree": scored["candidate_agree"],
                "test_marginal": scored["marginal"],
                "test_regressions": scored["regressions"],
                "discordant_convention": (
                    "b=baseline-correct/candidate-wrong; "
                    "c=candidate-correct/baseline-wrong"
                ),
                "b": b,
                "c": c,
                "two_sided_exact_binomial_p": pvalue,
                "verdict": verdict(scored["marginal"], pvalue, scored["regressions"]),
            }
            result["val_to_test_shrinkage"] = {
                "val_marginal": point["marginal"],
                "test_marginal": scored["marginal"],
                "marginal_change": scored["marginal"] - point["marginal"],
                "val_regressions": point["val_regressions"],
                "test_regressions": scored["regressions"],
            }
            return result

        report["grid"] = grid_rows
        report["resolution"] = resolve_selection(grid_rows, evaluate_test)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _recipe_env(tag: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "WYLY_TAG": tag,
        "WYLY_DS": "wikitext",
        "WYLY_LIB": "mined",
        "WYLY_JUDGE": "cover",
        "WYLY_ONLINE": "1",
        "WYLY_COVER": "sw",
        "WYLY_SEED": "0",
    })
    for key in ("WYLY_CONCEPTS", "WYLY_POINTER", "WYLY_TPOINTER", "WYLY_DX", "WYLY_CX", "WYLY_FOLDS"):
        env.pop(key, None)
    return env


def _print_scoreboard(report: dict[str, Any]) -> None:
    log("=== PART A: 410m vs 2.8b MINING CASCADE ===")
    for scale in ("410m", "2.8b"):
        row = report["scales"][scale]
        log(json.dumps({
            "scale": scale,
            "core_sw_agree": row["core_sw"]["agree"],
            "band_pass": row["band_pass"],
            "part_a": row.get("part_a"),
        }, sort_keys=True))
    if "grid" in report["scales"]["2.8b"]:
        log("=== PART B: FULL 2.8b VALIDATION GRID ===")
        for row in report["scales"]["2.8b"]["grid"]:
            log(json.dumps(row, sort_keys=True))
        log("=== SELECTION / ONE-SHOT VERDICT ===")
        log(json.dumps(report["scales"]["2.8b"]["resolution"], sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-scale", choices=tuple(SCALES))
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args(argv)
    if args.child_scale:
        if args.child_output is None:
            parser.error("--child-output is required with --child-scale")
        return _child_run(args.child_scale, args.child_output)

    log("REGISTERED SINGLETON GATE GRID (printed before any mining):")
    log(json.dumps([
        {"support_gate": support, "interaction_gate": interaction}
        for support, interaction in registered_grid()
    ]))
    report: dict[str, Any] = {
        "honesty_note": HONESTY_NOTE,
        "gate_sweep": "singleton-only; conjunction gate fixed at support=10, interaction=0.3",
        "grid": [list(point) for point in registered_grid()],
        "tie_break": "max marginal, then smallest support, then smallest interaction",
        "verdict_gap_resolution": (
            "The registration omits p<0.05, marginal>=0.003, test_regressions>0. "
            "That significant-but-harmful branch is conservatively labeled MIDDLE."
        ),
        "selection_gap_resolution": (
            "The registration does not name the branch where one or more points clear the "
            "judge bar but every such point introduces validation regressions. That branch "
            "stops without touching test_te and is labeled NO ZERO-REGRESSION SELECTION."
        ),
        "scales": {},
    }
    with tempfile.TemporaryDirectory(prefix="frames-at-scale-") as temp:
        for scale, config in SCALES.items():
            child_output = Path(temp) / f"{scale}.json"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--child-scale", scale, "--child-output", str(child_output),
            ]
            completed = subprocess.run(command, env=_recipe_env(config["tag"]), check=False)
            if child_output.exists():
                report["scales"][scale] = json.loads(child_output.read_text())
            if completed.returncode != 0:
                report["stopped"] = f"{scale} child failed or stopped (exit {completed.returncode})"
                OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))
                log(report["stopped"])
                return completed.returncode

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    _print_scoreboard(report)
    log(f"honesty note: {HONESTY_NOTE}")
    log(f"JSON: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
