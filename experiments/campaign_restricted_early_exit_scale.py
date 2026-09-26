"""Restricted-decision early exit at scale, with a directional bound — for
``docs/notes/restricted_early_exit_scale_prereg.md`` (SIGNED 2026-09-25).

Per model (Arm A = norm bounds as #129; Arm D = the adverse push along each rival's separating direction):

    python experiments/campaign_restricted_early_exit_scale.py model \\
        --bundle $FR/bundles/Qwen2.5-3B-Instruct --dumps runs/ree3b \\
        --weight-bound results/certified_early_exit_weight_bound_3b.json --tag 3b
    python experiments/campaign_restricted_early_exit_scale.py trend --tags 0.5b 3b 7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.campaign_certified_early_exit import ALPHA, evaluate_split, kstar  # noqa: E402
from pil.early_exit import (  # noqa: E402
    Bundle,
    ExitRecord,
    Readout,
    adverse_push,
    conformal_quantile,
    fit_scale,
    iter_dump,
    layer_prefix_index,
)

K = 6
TASKS = ("qa1", "qa2", "qa3")


def process(path: Path, meta: dict, b: Bundle, theta: np.ndarray):
    """ExitRecords (t/pred index the options) plus the adverse push ``a[k]`` per record."""
    out = []
    for rec in iter_dump(path):
        m = meta[rec["sid"]]
        ro = Readout(b.rows(b.readout_name, m["option_ids"]), theta)
        D = np.asarray(rec["d"], dtype=np.float64)
        last = layer_prefix_index(rec["blocks"], b.n_layer)
        s, s_resid = fit_scale(D[0], b.rows("embed", [rec["cur"]])[0], theta)
        pre = np.cumsum(D, axis=0)[last]
        y = pre / theta.astype(np.float64)[None, :]
        t = np.zeros(b.n_layer, dtype=np.int64)
        R = np.zeros(b.n_layer)
        a = np.zeros(b.n_layer)
        W = ro.W.astype(np.float64)
        for k in range(b.n_layer):
            t[k], R[k] = ro.radius(ro.U @ pre[k].astype(np.float32))
            a[k] = adverse_push(W, int(t[k]), y[-1] - y[k])
        r = ExitRecord(sid=rec["sid"], pos=int(rec["pos"]), pred=int(t[-1]), s=s, s_resid=s_resid, t=t, R=R,
                       ynorm=np.linalg.norm(y, axis=1), suffix=np.linalg.norm(y[-1][None, :] - y, axis=1))
        out.append((r, a))
    return out


def arm_a_verdict(m) -> str:
    if m["weight"]["coverage_skip2"] >= 0.10:
        return "FIRES"
    c = m["calibrated"]
    if c["coverage"] >= 0.25 and c["violation_rate_certified"] <= 2 * ALPHA and c["net_every"] >= 1.0:
        return "FIRES-EMPIRICAL"
    return "IN-BETWEEN" if m["oracle"]["coverage"] >= 0.25 else "HALTED"


def arm_d_verdict(m) -> str:
    c = m["calibrated_D"]
    if c["coverage"] >= 0.25 and c["violation_rate_certified"] <= 2 * ALPHA and c["net_every"] >= 1.0:
        return "D-FIRES"
    return "D-IN-BETWEEN" if m["oracle_D"]["coverage"] >= 0.25 else "D-HALTED"


def cmd_model(a):
    b = Bundle(a.bundle)
    theta = b.f32("norm")
    meta = json.loads(Path(a.meta).read_text())
    S_raw = np.array(json.loads(Path(a.weight_bound).read_text())["suffix_bound_raw"])
    layer = b.d * (b.d + 2 * b.n_kv * b.head_dim + b.d) + 3 * b.d * b.d_ff
    c_chk = K * b.d / layer
    L = b.n_layer - 1
    d = Path(a.dumps)
    cal = process(d / "cal_canonical.dump.jsonl", meta, b, theta)
    ev = process(d / "eval_canonical.dump.jsonl", meta, b, theta)
    lines, res = [], {"tag": a.tag, "n_layer": b.n_layer, "d": b.d, "c_chk": c_chk}
    say = lines.append
    say(f"[{a.tag}] Qwen2.5 n_layer={b.n_layer} d={b.d} readout={b.readout_name}; restricted check = "
        f"{c_chk:.5f} layer-equivalents")
    sr = max(r.s_resid for r, _ in cal + ev)
    res["s_resid_max"] = sr
    say(f"[control] s-fit residual max {sr:.2e} (bar 1e-3)")
    if sr >= 1e-3:
        say("ABORT (s-recovery)")
        Path(f"{a.out}.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    lib = Path(a.dumps) / "library_decisions.json"
    if lib.exists():
        libd = json.loads(lib.read_text())
        agree = [libd[r.sid]["value"] == meta[r.sid]["options"][r.pred] for r, _ in ev if r.sid in libd]
        res["engine_agreement"] = float(np.mean(agree))
        say(f"[control] library fp16 vs fieldrun int8 agreement {res['engine_agreement']:.3f} "
            f"(n={len(agree)})")

    cal_r = [r for r, _ in cal]
    ev_r = [r for r, _ in ev]
    ev_a = np.stack([x for _, x in ev])
    q_norm = np.array([conformal_quantile(np.array([r.suffix[k] / max(r.ynorm[k], 1e-30) for r in cal_r]),
                                          ALPHA) for k in range(b.n_layer)])
    q_dir = np.array([conformal_quantile(np.array([x[k] / max(r.ynorm[k], 1e-30) for r, x in cal]), ALPHA)
                      for k in range(b.n_layer)])
    yn = np.stack([r.ynorm for r in ev_r])
    bounds = {
        "oracle": np.stack([r.suffix[:L] for r in ev_r]),
        "weight": np.stack([r.s * S_raw[:L] for r in ev_r]),
        "calibrated": q_norm[None, :L] * yn[:, :L],
        "oracle_D": ev_a[:, :L],
        "calibrated_D": q_dir[None, :L] * yn[:, :L],
    }
    res["arms"] = {k: evaluate_split(ev_r, B, c_chk) for k, B in bounds.items()}
    for k, m in res["arms"].items():
        say(f"  [{k:12s}] coverage {m['coverage']:.3f} (≥2 skip {m['coverage_skip2']:.3f})  k_c med "
            f"{m['kc_median_certified']}  skipped {m['mean_layers_skipped']:.2f}  "
            f"violations {m['violations']} "
            f"({m['violation_rate_certified']:.3f})  net P-every {m['net_every']:+.2f}")
    ks = kstar(ev_r)
    res["kstar_mean_skippable"] = float(np.mean(np.maximum(L - ks, 0)))
    res["accuracy"] = {t: float(np.mean([meta[r.sid]["options"][r.pred] == meta[r.sid]["gold"] for r in ev_r
                                        if meta[r.sid]["task"] == t])) for t in TASKS}
    R = np.stack([r.R for r in ev_r])
    suf = np.stack([r.suffix for r in ev_r])
    tt = np.stack([r.t for r in ev_r])
    fin = np.array([r.pred for r in ev_r])
    trend = {}
    for k in range(L - 3, L):
        ok = tt[:, k] == fin
        trend[str(k - L)] = {"suffix_over_R": float(np.median(suf[ok, k] / np.maximum(R[ok, k], 1e-12))),
                             "push_over_R": float(np.median(ev_a[ok, k] / np.maximum(R[ok, k], 1e-12))),
                             "frac_final": float(ok.mean())}
    res["trend"] = trend
    say(f"  uncertified headroom {res['kstar_mean_skippable']:.2f} layers;  accuracy "
        + "  ".join(f"{t} {v:.3f}" for t, v in res["accuracy"].items()))
    say("  last exits (offset from final): " + "  ".join(
        f"{off}: suffix/R {v['suffix_over_R']:.1f}, push/R {v['push_over_R']:.2f}, "
        f"t_k=final {v['frac_final']:.2f}"
        for off, v in trend.items()))
    arms = res["arms"]
    gate = all(arms[k]["violations"] == 0 for k in ("oracle", "weight", "oracle_D"))
    res["soundness_gate"] = gate
    res["verdict_A"] = arm_a_verdict(arms) if gate else "VOID"
    res["verdict_D"] = arm_d_verdict(arms) if gate else "VOID"
    say(f"  soundness gate {'PASS' if gate else 'FAIL'};  Arm A: {res['verdict_A']};  "
        f"Arm D: {res['verdict_D']}")
    Path(f"{a.out}.json").write_text(json.dumps(res, indent=1) + "\n")
    Path(f"{a.out}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_trend(a):
    rows = [json.loads(Path(f"{a.prefix}_{t}.json").read_text()) for t in a.tags]
    lines = ["scale trend (monotone across all models or 'not monotone'):"]

    def mono(vals):
        inc = all(x < y for x, y in zip(vals, vals[1:], strict=False))
        dec = all(x > y for x, y in zip(vals, vals[1:], strict=False))
        return "increasing" if inc else ("decreasing" if dec else "not monotone")

    series = {
        "suffix/R at last exit": [r["trend"]["-1"]["suffix_over_R"] for r in rows],
        "push/R at last exit": [r["trend"]["-1"]["push_over_R"] for r in rows],
        "qa1 accuracy": [r["accuracy"]["qa1"] for r in rows],
        "mean accuracy": [float(np.mean(list(r["accuracy"].values()))) for r in rows],
        "headroom (layers)": [r["kstar_mean_skippable"] for r in rows],
        "Oracle-D coverage": [r["arms"]["oracle_D"]["coverage"] for r in rows],
        "Calibrated-D coverage": [r["arms"]["calibrated_D"]["coverage"] for r in rows],
    }
    out = {}
    for name, vals in series.items():
        out[name] = {"values": dict(zip(a.tags, vals, strict=True)), "trend": mono(vals)}
        lines.append(f"  {name:24s} " + "  ".join(f"{t}={v:.3f}" for t, v in zip(a.tags, vals, strict=True))
                     + f"   → {mono(vals)}")
    lines.append("verdicts: " + "  ".join(f"{r['tag']}: A={r['verdict_A']}, D={r['verdict_D']}"
                                         for r in rows))
    Path(f"{a.prefix}_trend.json").write_text(json.dumps(out, indent=1) + "\n")
    Path(f"{a.prefix}_trend.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("--bundle", required=True)
    m.add_argument("--dumps", required=True)
    m.add_argument("--meta", default="runs/ree/meta.json")
    m.add_argument("--weight-bound", required=True)
    m.add_argument("--tag", required=True)
    m.add_argument("--out", default=None)
    t = sub.add_parser("trend")
    t.add_argument("--tags", nargs="+", required=True)
    t.add_argument("--prefix", default="results/restricted_scale")
    a = p.parse_args()
    if a.cmd == "model":
        a.out = a.out or f"results/restricted_scale_{a.tag}"
        cmd_model(a)
    else:
        cmd_trend(a)


if __name__ == "__main__":
    main()
