"""Certified early exit — campaign for ``docs/notes/certified_early_exit_prereg.md`` (SIGNED 2026-09-25).

Three steps, in the order the prereg fixes:

    python experiments/campaign_certified_early_exit.py splits  --out runs/cee
    python experiments/campaign_certified_early_exit.py weightbound \
        --bundle $FR/bundles/Qwen2.5-0.5B-Instruct \
        --out results/certified_early_exit_weight_bound.json      # committed BEFORE any dump is read
    # dumps (from the fieldrun checkout; one per split):
    #   ./target/release/fieldrun --bundle Qwen2.5-0.5B-Instruct --recursion-explain \
    #       --source-dump runs/cee/<split>.dump.jsonl --texts runs/cee/<split>.texts.jsonl --n 16 --kcand 2
    python experiments/campaign_certified_early_exit.py evaluate --bundle ... --dumps runs/cee \
        --weight-bound results/certified_early_exit_weight_bound.json --out results/certified_early_exit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pil.early_exit import (
    Bundle,
    ExitRecord,
    Readout,
    conformal_quantile,
    iter_dump,
    process_records,
    suffix_weight_bound,
    weight_bounds,
)

ALPHA = 0.05                  # PIN E.3
UNITS_PER_SPLIT = 20          # PIN B
MIN_CHARS, TRUNC = 400, 600   # PIN B
CODE_CHUNK = 40               # PIN B
S_RESID_MAX = 1e-3            # PIN C self-test
RECON_MIN = 0.99              # control
SPLITS = ("cal", "eval_prose", "eval_code")


# ── step 1: splits ──────────────────────────────────────────────────────────────────────────────────────────


def prose_units(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if len(s) >= MIN_CHARS and not (s.startswith("=") and s.endswith("=")):
            out.append(s[:TRUNC])
    return out


def code_units(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    out = []
    for i in range(0, len(lines) - CODE_CHUNK + 1, CODE_CHUNK):
        chunk = "\n".join(lines[i : i + CODE_CHUNK])
        if len(chunk) >= MIN_CHARS:
            out.append(chunk[:TRUNC])
    return out


def make_splits(data: Path) -> dict[str, list[dict]]:
    rng = np.random.default_rng(0)
    prose, code = prose_units(data / "wikitext2_train.txt"), code_units(data / "code_train.txt")
    prose = [prose[i] for i in rng.permutation(len(prose))]
    code = [code[i] for i in rng.permutation(len(code))]
    n = UNITS_PER_SPLIT
    units = {
        "cal": [("prose", u) for u in prose[:n]] + [("code", u) for u in code[:n]],
        "eval_prose": [("prose", u) for u in prose[n : 2 * n]],
        "eval_code": [("code", u) for u in code[n : 2 * n]],
    }
    return {
        split: [{"sid": f"{split}-{kind}-{i}", "text": text} for i, (kind, text) in enumerate(us)]
        for split, us in units.items()
    }


def cmd_splits(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in make_splits(Path(a.data)).items():
        p = out / f"{split}.texts.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[splits] {p}: {len(rows)} units")


# ── step 2: the weight-derived bound (before any dump is read) ──────────────────────────────────────────────


def cmd_weightbound(a):
    b = Bundle(a.bundle)
    A, M = weight_bounds(b)
    S = suffix_weight_bound(A, M)
    rec = {"bundle": str(a.bundle), "A": A.tolist(), "M": M.tolist(), "suffix_bound_raw": S.tolist(),
           "note": "raw units; B_w(k) = s * suffix_bound_raw[k] in y coordinates "
                   "(prereg PIN E.2 + Addendum A)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[weightbound] wrote {a.out}")


# ── step 3: evaluate ────────────────────────────────────────────────────────────────────────────────────────


def load_records(
    dump: Path, readout: Readout, emb: np.ndarray, n_layer: int, cache: Path
) -> list[ExitRecord]:
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return [ExitRecord(sid=str(z["sid"][i]), pos=int(z["pos"][i]), pred=int(z["pred"][i]),
                           s=float(z["s"][i]), s_resid=float(z["s_resid"][i]), t=z["t"][i], R=z["R"][i],
                           ynorm=z["ynorm"][i], suffix=z["suffix"][i]) for i in range(len(z["pos"]))]
    recs = list(process_records(iter_dump(dump), readout, emb, n_layer))
    np.savez(cache, sid=np.array([r.sid for r in recs]), pos=np.array([r.pos for r in recs]),
             pred=np.array([r.pred for r in recs]), s=np.array([r.s for r in recs]),
             s_resid=np.array([r.s_resid for r in recs]), t=np.stack([r.t for r in recs]),
             R=np.stack([r.R for r in recs]), ynorm=np.stack([r.ynorm for r in recs]),
             suffix=np.stack([r.suffix for r in recs]))
    return recs


def check_cost(b: Bundle) -> float:
    """PIN F: one full-vocabulary check in layer-equivalents (attention projections + MLP per layer)."""
    kv = b.n_kv * b.head_dim
    layer = b.d * (b.d + 2 * kv + b.d) + 3 * b.d * b.d_ff
    return b.vocab * b.d / layer


def bound_matrix(recs, kind, S_raw, qhat):
    """``B[n, k]`` in y coordinates for the exits ``k = 0..n_layer-2``."""
    L = len(S_raw) - 1
    if kind == "oracle":
        return np.stack([r.suffix[:L] for r in recs])
    if kind == "weight":
        return np.stack([r.s * S_raw[:L] for r in recs])
    if kind == "calibrated":
        return np.stack([qhat[:L] * r.ynorm[:L] for r in recs])
    raise ValueError(kind)


def evaluate_split(recs, B, c_chk):
    """P-every metrics + per-k certification table for one split and one bound."""
    L = B.shape[1]                                          # exits 0..L-1; L = n_layer - 1 = full model index
    R = np.stack([r.R[:L] for r in recs])
    t = np.stack([r.t[:L] for r in recs])
    pred = np.array([r.pred for r in recs])
    cert = B < R                                            # (N, L)
    any_c = cert.any(axis=1)
    kc = np.where(any_c, cert.argmax(axis=1), L)            # L = no exit
    skipped = np.where(any_c, L - kc, 0)
    checks = np.where(any_c, kc + 1, L)
    wrong = any_c & (t[np.arange(len(recs)), np.minimum(kc, L - 1)] != pred)
    return {
        "N": int(len(recs)),
        "coverage": float(any_c.mean()),
        "coverage_skip2": float((kc <= L - 2).mean()),
        "kc_median_certified": float(np.median(kc[any_c])) if any_c.any() else None,
        "mean_layers_skipped": float(skipped.mean()),
        "violations": int(wrong.sum()),
        "violation_rate_certified": float(wrong.sum() / max(any_c.sum(), 1)),
        "net_every": float((skipped - checks * c_chk).mean()),
        "cert_rate_by_k": cert.mean(axis=0).tolist(),
        "wrong_at_k": ((cert & (t != pred[:, None])).sum(axis=0)).tolist(),
    }


def single_check(recs, B, k0, c_chk):
    L = B.shape[1]
    R = np.array([r.R[k0] for r in recs])
    t = np.array([r.t[k0] for r in recs])
    pred = np.array([r.pred for r in recs])
    cert = B[:, k0] < R
    return {"k0": int(k0), "coverage": float(cert.mean()), "net": float(((L - k0) * cert).mean() - c_chk),
            "violations": int((cert & (t != pred)).sum())}


def kstar(recs):
    """Earliest exit after which the prefix argmax stays equal to the model's decision (uncertified)."""
    out = []
    for r in recs:
        ok = r.t == r.pred
        k = len(ok)
        while k > 0 and ok[k - 1]:
            k -= 1
        out.append(k)
    return np.array(out)


def verdict(res) -> str:
    ev = ("eval_prose", "eval_code")
    fires = any(res[s]["weight"]["coverage_skip2"] >= 0.10 for s in ev)
    fires_emp = any(
        res[s]["calibrated"]["coverage"] >= 0.25
        and res[s]["calibrated"]["violation_rate_certified"] <= 2 * ALPHA
        and res[s]["calibrated_single"]["net"] > 0
        for s in ev
    )
    oracle = any(res[s]["oracle"]["coverage"] >= 0.25 for s in ev)
    if fires:
        return "FIRES"
    if fires_emp:
        return "FIRES-EMPIRICAL"
    if oracle:
        return "IN-BETWEEN"
    return "HALTED"


def cmd_evaluate(a):
    b = Bundle(a.bundle)
    emb = b.f32("embed")
    readout = Readout(emb, b.f32("norm"))
    wb = json.loads(Path(a.weight_bound).read_text())
    S_raw = np.array(wb["suffix_bound_raw"])
    c_chk = check_cost(b)
    dumps = Path(a.dumps)
    recs = {s: load_records(dumps / f"{s}.dump.jsonl", readout, emb, b.n_layer, dumps / f"{s}.records.npz")
            for s in SPLITS}

    lines, res = [], {"c_chk": c_chk, "alpha": ALPHA}
    say = lines.append
    say(f"certified early exit — Qwen2.5-0.5B-Instruct (n_layer={b.n_layer}, d={b.d}, V={b.vocab}); "
        f"one full-vocab check = {c_chk:.2f} layer-equivalents")

    # controls
    abort = []
    for s, rs in recs.items():
        sr = np.array([r.s_resid for r in rs])
        recon = float(np.mean([r.t[-1] == r.pred for r in rs]))
        res.setdefault("controls", {})[s] = {"N": len(rs), "s_resid_max": float(sr.max()), "recon": recon}
        say(f"[control] {s}: N={len(rs)}  s-fit residual max {sr.max():.2e} (bar {S_RESID_MAX:g})  "
            f"full-vocab recon {recon:.3f} (bar {RECON_MIN})")
        if sr.max() >= S_RESID_MAX:
            abort.append(f"{s}: s-fit residual {sr.max():.2e}")
        if recon < RECON_MIN:
            abort.append(f"{s}: recon {recon:.3f}")
    if abort:
        say("ABORT (PIN C / control): " + "; ".join(abort))
        Path(f"{a.out}.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return

    # calibration (CAL only)
    L = b.n_layer - 1
    rho = np.stack([r.suffix / np.maximum(r.ynorm, 1e-30) for r in recs["cal"]])      # (n_cal, n_layer)
    qhat = np.array([conformal_quantile(rho[:, k], ALPHA) for k in range(b.n_layer)])
    res["qhat"] = qhat.tolist()

    kinds = ("oracle", "weight", "calibrated")
    k0 = {}
    for kind in kinds:                                        # P-single: k0 chosen on CAL
        Bc = bound_matrix(recs["cal"], kind, S_raw, qhat)
        nets = [single_check(recs["cal"], Bc, k, c_chk)["net"] for k in range(L)]
        k0[kind] = int(np.argmax(nets))
    res["k0"] = k0

    for s in ("eval_prose", "eval_code"):
        rs = recs[s]
        ks = kstar(rs)
        res[s] = {"kstar_median": float(np.median(ks)), "kstar_exitable": float((ks <= L - 1).mean()),
                  "kstar_mean_skipped": float(np.mean(np.maximum(L - ks, 0)))}
        say(f"\n=== {s} (N={len(rs)}) ===")
        say(f"  uncertified k★: median {np.median(ks):.0f}, exitable (k★ ≤ {L-1}) {(ks <= L-1).mean():.3f}, "
            f"mean layers skippable {np.mean(np.maximum(L - ks, 0)):.2f}")
        for kind in kinds:
            B = bound_matrix(rs, kind, S_raw, qhat)
            m = evaluate_split(rs, B, c_chk)
            sc = single_check(rs, B, k0[kind], c_chk)
            res[s][kind] = m
            res[s][f"{kind}_single"] = sc
            say(f"  [{kind:10s}] coverage {m['coverage']:.3f} (≥2 skipped {m['coverage_skip2']:.3f})  "
                f"k_c med {m['kc_median_certified']}  skipped {m['mean_layers_skipped']:.2f}  "
                f"violations {m['violations']} ({m['violation_rate_certified']:.3f} of certified)  "
                f"net P-every {m['net_every']:+.2f}  P-single(k0={sc['k0']}) cov {sc['coverage']:.3f} "
                f"net {sc['net']:+.2f} viol {sc['violations']}")

    sound = all(res[s][k]["violations"] == 0
                for s in ("eval_prose", "eval_code") for k in ("oracle", "weight"))
    res["soundness_gate"] = sound
    res["verdict"] = verdict(res) if sound else "VOID (soundness gate failed)"
    say(f"\nsoundness gate (0 violations for oracle + weight): {'PASS' if sound else 'FAIL'}")
    say(f"VERDICT: {res['verdict']}")
    Path(f"{a.out}.json").write_text(json.dumps(res, indent=1) + "\n")
    Path(f"{a.out}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("splits")
    s.add_argument("--data", default="data")
    s.add_argument("--out", default="runs/cee")
    w = sub.add_parser("weightbound")
    w.add_argument("--bundle", required=True)
    w.add_argument("--out", default="results/certified_early_exit_weight_bound.json")
    e = sub.add_parser("evaluate")
    e.add_argument("--bundle", required=True)
    e.add_argument("--dumps", default="runs/cee")
    e.add_argument("--weight-bound", default="results/certified_early_exit_weight_bound.json")
    e.add_argument("--out", default="results/certified_early_exit")
    a = p.parse_args()
    {"splits": cmd_splits, "weightbound": cmd_weightbound, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
