"""J-lens early exit — campaign for ``docs/notes/jlens_early_exit_prereg.md`` (SIGNED 2026-09-25).

Re-reads the #126 source dumps through the shrunk J-lens ``ŷ_k = ((1−λ)I + λJ_k) y_k`` and bounds only the
leftover ``e_k = y_final − ŷ_k``. Needs the dumps from ``campaign_certified_early_exit.py`` and the J export:

    fieldrun --jlens-export runs/cee/jlens.npz \
        --jlens-in bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.jlens
    python experiments/campaign_jlens_early_exit.py --bundle $FR/bundles/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.campaign_certified_early_exit import (  # noqa: E402
    ALPHA,
    SPLITS,
    bound_matrix,
    check_cost,
    evaluate_split,
    kstar,
    load_records,
    single_check,
)
from pil.early_exit import Bundle, Readout, conformal_quantile, jlens_predictor  # noqa: E402

LAMBDAS = (0.0, 0.25, 0.5, 1.0)       # PIN C
MECH_K = (12, 20, 21, 22)             # mechanism metric exits
EV = ("eval_prose", "eval_code")


def qhat_for(cal, n_layer):
    rho = np.stack([r.suffix / np.maximum(r.ynorm, 1e-30) for r in cal])
    return np.array([conformal_quantile(rho[:, k], ALPHA) for k in range(n_layer)])


def verdict(res, lam_star) -> str:
    at = res["lambda"][str(lam_star)]
    fires = any(
        at[s]["calibrated"]["coverage"] >= 0.25
        and at[s]["calibrated"]["violation_rate_certified"] <= 2 * ALPHA
        and at[s]["calibrated_single"]["net"] > 0
        for s in EV
    )
    if fires:
        return "FIRES-EMPIRICAL"
    if any(at[s]["oracle"]["coverage"] >= 0.25 for s in EV):
        return "IN-BETWEEN"
    best = max(res["lambda"][str(lam)][s]["oracle"]["coverage"] for lam in LAMBDAS if lam > 0 for s in EV)
    return "WEAK" if best >= 0.01 else "HALTED"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True)
    p.add_argument("--dumps", default="runs/cee")
    p.add_argument("--jlens", default="runs/cee/jlens.npz")
    p.add_argument("--out", default="results/jlens_early_exit")
    a = p.parse_args()

    b = Bundle(a.bundle)
    emb = b.f32("embed")
    readout = Readout(emb, b.f32("norm"))
    z = np.load(a.jlens)
    J, fitted = z["J"], z["fitted"]
    L = b.n_layer - 1
    c_chk = check_cost(b)
    c_j = b.d * b.d / (b.d * (b.d + 2 * b.n_kv * b.head_dim + b.d) + 3 * b.d * b.d_ff)
    cost = c_chk + c_j
    dumps = Path(a.dumps)
    base = {s: load_records(dumps / f"{s}.dump.jsonl", readout, emb, b.n_layer, dumps / f"{s}.records.npz")
            for s in SPLITS}

    lines = []
    say = lines.append
    res = {"c_chk": c_chk, "c_j": c_j, "fitted_layers": int(fitted.sum()), "lambda": {}}
    say(f"J-lens early exit — Qwen2.5-0.5B-Instruct; J fitted on {int(fitted.sum())} layers; "
        f"check = {c_chk:.2f} + J {c_j:.3f} layer-equivalents")

    recs = {}
    for lam in LAMBDAS:
        pred = jlens_predictor(J, lam)
        recs[lam] = {s: load_records(dumps / f"{s}.dump.jsonl", readout, emb, b.n_layer,
                                     dumps / f"jl{lam:g}.{s}.records.npz", predict=pred) for s in SPLITS}

    # sanity gate: λ = 0 reproduces #126
    worst_R, worst_suf, same_t = 0.0, 0.0, True
    for s in SPLITS:
        for r0, r in zip(base[s], recs[0.0][s], strict=True):
            same_t &= bool(np.array_equal(r0.t, r.t))
            dR = np.abs(r0.R - r.R) / np.maximum(np.abs(r0.R), 1e-12)
            dS = np.abs(r0.suffix - r.suffix) / np.maximum(r0.suffix, 1e-12)
            worst_R, worst_suf = max(worst_R, float(dR.max())), max(worst_suf, float(dS.max()))
    repro = same_t and worst_R <= 1e-9 and worst_suf <= 1e-9
    res["sanity_lambda0"] = {"same_t": same_t, "max_rel_dR": worst_R, "max_rel_dsuffix": worst_suf}
    say(f"[sanity] λ=0 vs #126: same argmax {same_t}, max rel ΔR {worst_R:.1e}, "
        f"max rel Δsuffix {worst_suf:.1e}")

    # per-λ evaluation
    zeros = np.zeros(b.n_layer)
    cal_cov = {}
    for lam in LAMBDAS:
        q = qhat_for(recs[lam]["cal"], b.n_layer)
        Bc = bound_matrix(recs[lam]["cal"], "calibrated", zeros, q)
        cal_cov[lam] = evaluate_split(recs[lam]["cal"], Bc, cost)["coverage"]
        k0 = int(np.argmax([single_check(recs[lam]["cal"], Bc, k, cost)["net"] for k in range(L)]))
        out = {"qhat": q.tolist(), "cal_calibrated_coverage": cal_cov[lam], "k0": k0}
        for s in EV:
            rs = recs[lam][s]
            ent = {}
            for kind in ("oracle", "calibrated"):
                B = bound_matrix(rs, kind, zeros, q)
                ent[kind] = evaluate_split(rs, B, cost)
                ent[f"{kind}_single"] = single_check(rs, B, k0, cost)
            ks = kstar(rs)
            ent["kstar_mean_skippable"] = float(np.mean(np.maximum(L - ks, 0)))
            ent["mech_leftover_over_suffix"] = {
                str(k): float(np.median([r.suffix[k] / max(r0.suffix[k], 1e-30)
                                         for r, r0 in zip(rs, base[s], strict=True)]))
                for k in MECH_K}
            out[s] = ent
        res["lambda"][str(lam)] = out
        say(f"\n--- λ={lam:g}  (CAL calibrated coverage {cal_cov[lam]:.3f}, k0={k0})")
        for s in EV:
            e = out[s]
            mech = "  ".join(f"k{k}:{v:.2f}" for k, v in e["mech_leftover_over_suffix"].items())
            o = e["oracle"]
            say(f"  {s:10s} oracle cov {o['coverage']:.3f} (≥2 skip {o['coverage_skip2']:.3f}, "
                f"viol {o['violations']})  calibrated cov {e['calibrated']['coverage']:.3f} "
                f"(viol rate {e['calibrated']['violation_rate_certified']:.3f})  P-single net "
                f"{e['calibrated_single']['net']:+.2f}  k★ skippable {e['kstar_mean_skippable']:.2f}  "
                f"‖e‖/‖suffix‖ {mech}")

    lam_star = min((lam for lam in LAMBDAS if lam > 0), key=lambda lam: (-cal_cov[lam], lam))
    res["lambda_star"] = lam_star
    oracle_clean = all(res["lambda"][str(lam)][s]["oracle"]["violations"] == 0 for lam in LAMBDAS for s in EV)
    zero_at_0 = all(res["lambda"][str(0.0)][s]["oracle"]["coverage"] == 0.0 for s in EV)
    gate = repro and oracle_clean and zero_at_0
    res["sanity_gate"] = gate
    res["verdict"] = verdict(res, lam_star) if gate else "VOID (sanity gate failed)"
    say(f"\nλ* (CAL) = {lam_star:g}")
    say(f"sanity gate (λ=0 reproduces #126, oracle 0 violations at every λ): {'PASS' if gate else 'FAIL'}")
    say(f"VERDICT: {res['verdict']}")
    Path(f"{a.out}.json").write_text(json.dumps(res, indent=1) + "\n")
    Path(f"{a.out}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
