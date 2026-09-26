"""Certified early exit on restricted decisions — campaign for ``docs/notes/restricted_early_exit_prereg.md``.

Reads the ``--tail 1`` source dumps of the decision prompts built by ``restricted_decisions_build.py`` and
applies the #126 certificate restricted to each field's option tokens.

    python experiments/campaign_restricted_early_exit.py \
        --bundle $FR/bundles/Qwen2.5-0.5B-Instruct --dir runs/ree
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
    bound_matrix,
    evaluate_split,
    kstar,
)
from pil.early_exit import (  # noqa: E402
    Bundle,
    ExitRecord,
    Readout,
    conformal_quantile,
    fit_scale,
    iter_dump,
    layer_prefix_index,
)

EV = ("qa1", "qa2", "qa3")


def restricted_record(rec, meta, U, theta, emb, n_layer):
    """One decision → an ExitRecord whose ``t``/``pred`` index the field's options (radius over options)."""
    m = meta[rec["sid"]]
    opts = np.array(m["option_ids"])
    ro = Readout(U[opts], theta)
    D = np.asarray(rec["d"], dtype=np.float64)
    last = layer_prefix_index(rec["blocks"], n_layer)
    s, s_resid = fit_scale(D[0], emb[rec["cur"]].astype(np.float32), theta)
    pre = np.cumsum(D, axis=0)[last]
    y = pre / theta.astype(np.float64)[None, :]
    t = np.zeros(n_layer, dtype=np.int64)
    R = np.zeros(n_layer)
    for k in range(n_layer):
        t[k], R[k] = ro.radius(ro.U @ pre[k].astype(np.float32))
    r = ExitRecord(sid=rec["sid"], pos=int(rec["pos"]), pred=int(t[-1]), s=s, s_resid=s_resid, t=t, R=R,
                   ynorm=np.linalg.norm(y, axis=1), suffix=np.linalg.norm(y[-1][None, :] - y, axis=1))
    return r, D, opts


def load(path, meta, U, theta, emb, n_layer):
    out = []
    for rec in iter_dump(path):
        out.append(restricted_record(rec, meta, U, theta, emb, n_layer))
    return out


def order_bias(first, last, meta, U, blocks):
    """Accuracy per order and the per-block order effect on the gold token's centred option incidence."""
    by_key = {}
    for (r, D, opts), tag in [(x, "first") for x in first] + [(x, "last") for x in last]:
        m = meta[r.sid]
        key = (m["task"], r.sid.rsplit("-", 1)[1])
        g = m["options"].index(m["gold"])
        C = D @ U[opts].T.astype(np.float64)                         # (nb, K) option incidences
        cc = C[:, g] - C.mean(axis=1)
        by_key.setdefault(key, {})[tag] = (cc, m["options"][r.pred] == m["gold"])
    deltas, acc_first, acc_last, shares = [], [], [], []
    for pair in by_key.values():
        if len(pair) != 2:
            continue
        d = pair["first"][0] - pair["last"][0]
        deltas.append(d)
        acc_first.append(pair["first"][1])
        acc_last.append(pair["last"][1])
        a = np.abs(d)
        shares.append(float(np.sort(a)[-5:].sum() / a.sum()) if a.sum() > 0 else float("nan"))
    mean_d = np.mean(deltas, axis=0)
    top = np.argsort(-np.abs(mean_d))[:5]
    return {"n": len(deltas), "acc_gold_first": float(np.mean(acc_first)),
            "acc_gold_last": float(np.mean(acc_last)),
            "median_top5_share": float(np.nanmedian(shares)),
            "top5_blocks_mean_effect": [(blocks[j], float(mean_d[j])) for j in top],
            "top5_share_of_mean_effect": float(np.abs(mean_d[top]).sum() / np.abs(mean_d).sum())}


def verdict(res) -> str:
    ev = res["eval"]
    if ev["weight"]["coverage_skip2"] >= 0.10:
        return "FIRES"
    c = ev["calibrated"]
    if c["coverage"] >= 0.25 and c["violation_rate_certified"] <= 2 * ALPHA and c["net_every"] >= 1.0:
        return "FIRES-EMPIRICAL"
    if ev["oracle"]["coverage"] >= 0.25:
        return "IN-BETWEEN"
    return "HALTED"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True)
    p.add_argument("--dir", default="runs/ree")
    p.add_argument("--weight-bound", default="results/certified_early_exit_weight_bound.json")
    p.add_argument("--out", default="results/restricted_early_exit")
    a = p.parse_args()
    d = Path(a.dir)
    b = Bundle(a.bundle)
    emb = b.f32("embed")
    theta = b.f32("norm")
    meta = json.loads((d / "meta.json").read_text())
    S_raw = np.array(json.loads(Path(a.weight_bound).read_text())["suffix_bound_raw"])
    K = 6
    layer = b.d * (b.d + 2 * b.n_kv * b.head_dim + b.d) + 3 * b.d * b.d_ff
    c_chk = K * b.d / layer
    L = b.n_layer - 1

    data = {s: load(d / f"{s}.dump.jsonl", meta, emb, theta, emb, b.n_layer)
            for s in ("cal_canonical", "eval_canonical", "eval_gold_first", "eval_gold_last")}
    lines, res = [], {"c_chk": c_chk, "K": K}
    say = lines.append
    say(f"restricted early exit — Qwen2.5-0.5B-Instruct, K={K} options; one restricted check = {c_chk:.5f} "
        f"layer-equivalents")

    blocks = next(iter_dump(d / "eval_canonical.dump.jsonl"))["blocks"]
    abort = []
    for s, rows in data.items():
        sr = max(r.s_resid for r, _, _ in rows)
        say(f"[control] {s}: N={len(rows)}  s-fit residual max {sr:.2e}")
        if sr >= 1e-3:
            abort.append(s)
    lib_path = d / "library_decisions.json"
    if lib_path.exists():
        lib = json.loads(lib_path.read_text())
        agree = [lib[r.sid]["value"] == meta[r.sid]["options"][r.pred] for r, _, _ in data["eval_canonical"]
                 if r.sid in lib]
        res["engine_agreement"] = float(np.mean(agree))
        say(f"[control] library fp16 vs fieldrun int8 restricted decision agreement: "
            f"{res['engine_agreement']:.3f} (n={len(agree)})")
    if abort:
        say("ABORT (PIN C self-test): " + ", ".join(abort))
        Path(f"{a.out}.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return

    cal = [r for r, _, _ in data["cal_canonical"]]
    rho = np.stack([r.suffix / np.maximum(r.ynorm, 1e-30) for r in cal])
    qhat = np.array([conformal_quantile(rho[:, k], ALPHA) for k in range(b.n_layer)])
    ev = [r for r, _, _ in data["eval_canonical"]]
    res["eval"] = {}
    ks = kstar(ev)
    res["kstar_mean_skippable"] = float(np.mean(np.maximum(L - ks, 0)))
    say(f"\n=== EVAL (N={len(ev)}) ===  uncertified k★: median {np.median(ks):.0f}, "
        f"mean skippable {res['kstar_mean_skippable']:.2f} layers")
    for kind in ("oracle", "weight", "calibrated"):
        m = evaluate_split(ev, bound_matrix(ev, kind, S_raw, qhat), c_chk)
        res["eval"][kind] = m
        say(f"  [{kind:10s}] coverage {m['coverage']:.3f} (≥2 skipped {m['coverage_skip2']:.3f})  "
            f"k_c med {m['kc_median_certified']}  skipped {m['mean_layers_skipped']:.2f}  violations "
            f"{m['violations']} ({m['violation_rate_certified']:.3f})  net P-every {m['net_every']:+.2f}")
    acc = {t: float(np.mean([meta[r.sid]["options"][r.pred] == meta[r.sid]["gold"] for r in ev
                             if meta[r.sid]["task"] == t])) for t in EV}
    res["accuracy"] = acc
    say("  accuracy vs gold: " + "  ".join(f"{t} {v:.3f}" for t, v in acc.items()))

    ob = order_bias(data["eval_gold_first"], data["eval_gold_last"], meta, emb, blocks)
    res["order_bias"] = ob
    say(f"\n=== option-order bias (descriptive, n={ob['n']}) ===")
    say(f"  accuracy gold-first {ob['acc_gold_first']:.3f}  gold-last {ob['acc_gold_last']:.3f}")
    say(f"  top-5 blocks carry {ob['top5_share_of_mean_effect']:.2f} of the mean order effect: "
        + ", ".join(f"{name} {v:+.3f}" for name, v in ob["top5_blocks_mean_effect"]))
    say(f"  per-decision median top-5 share {ob['median_top5_share']:.2f}")

    sound = all(res["eval"][k]["violations"] == 0 for k in ("oracle", "weight"))
    res["soundness_gate"] = sound
    res["verdict"] = verdict(res) if sound else "VOID (soundness gate failed)"
    say(f"\nsoundness gate: {'PASS' if sound else 'FAIL'}")
    say(f"VERDICT: {res['verdict']}")
    Path(f"{a.out}.json").write_text(json.dumps(res, indent=1) + "\n")
    Path(f"{a.out}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
