"""Does participation ratio predict anything? — campaign for ``docs/notes/pr_validation_prereg.md`` (SIGNED).

    python experiments/campaign_pr_validation.py --bundle $FR/bundles/Qwen2.5-0.5B-Instruct \\
        --dumps runs/cee --replication $FR/experiments/certified_prune_step0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pil.early_exit import Bundle, iter_dump
from pil.pr_validation import (
    bootstrap_ci,
    centred_target,
    erank,
    k_suf,
    merge_layers,
    partial_spearman,
    pr,
    pr_head_decides,
    spearman,
    split_ratio,
)

KCAND = 24                         # PIN A: candidate set for centring = top-24 by full-model logit
RHO_BAR = 0.3                      # Claim A bar
REPS = 2000                        # PIN E
REPLICATION = ("qwen05_science", "qwen05_code", "coder05_science", "qwen7b_science")


def per_position(C: np.ndarray, t: int, full_vocab: bool, cands: np.ndarray | None = None) -> dict:
    """All per-position quantities for one incidence matrix ``C`` (rows = sources, columns = tokens)."""
    L = C.sum(axis=0)
    order = np.argsort(-L)
    v2 = int(order[1] if order[0] == t else order[0])
    if cands is None:
        cands = order[:KCAND]
    cc = centred_target(C, t, cands)
    r_eff = pr(cc)
    out = {"r_eff": r_eff, "r_raw": pr(C[:, t]), "k_suf": k_suf(C, t, v2), "margin": float(L[t] - L[v2]),
           "split": split_ratio(cc)}
    if full_vocab:
        out["head"] = pr_head_decides(C, t, cc, r_eff)
    return out


def primary(bundle: Bundle, dumps: Path):
    U = bundle.f32("embed")
    rows = []
    for split in ("cal", "eval_prose", "eval_code"):
        for rec in iter_dump(dumps / f"{split}.dump.jsonl"):
            D = np.asarray(rec["d"], dtype=np.float32)
            C = (D @ U.T).astype(np.float64)                   # (nb, V) exact logit contributions
            t = int(np.argmax(C.sum(axis=0)))
            row = {"sid": rec["sid"], "corpus": "prose" if "-prose-" in rec["sid"] else "code",
                   "recon": t == int(rec["pred"]), "erank": erank(D)}
            blk = per_position(C, t, full_vocab=True)
            lay = per_position(merge_layers(C), t, full_vocab=True, cands=np.argsort(-C.sum(axis=0))[:KCAND])
            row.update({f"blk_{k}": v for k, v in blk.items()})
            row.update({f"lay_{k}": v for k, v in lay.items()})
            rows.append(row)
    return rows


def replication(rep_dir: Path):
    out = {}
    for name in REPLICATION:
        path = rep_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = []
        for rec in iter_dump(path):
            C = np.asarray(rec["contrib"], dtype=np.float64)       # (nb, K); column 0 = the decode
            if int(np.argmax(C.sum(axis=0))) != 0:
                continue
            cands = np.arange(C.shape[1])
            blk = per_position(C, 0, full_vocab=False, cands=cands)
            lay = per_position(merge_layers(C), 0, full_vocab=False, cands=cands)
            rows.append({**{f"blk_{k}": v for k, v in blk.items()},
                         **{f"lay_{k}": v for k, v in lay.items()}})
        out[name] = rows
    return out


def claim_a_cell(rows, gran, clusters):
    x = np.array([r[f"{gran}_r_eff"] for r in rows])
    y = np.array([r[f"{gran}_k_suf"] for r in rows], dtype=float)
    z = np.array([r[f"{gran}_margin"] for r in rows])
    part = partial_spearman(x, y, z)
    lo, hi = bootstrap_ci(partial_spearman, (x, y, z), clusters, reps=REPS)
    return {"n": len(rows), "partial": part, "ci": [lo, hi], "rho_reff_ksuf": spearman(x, y),
            "rho_margin_ksuf": spearman(z, y), "rho_raw_ksuf": spearman(
                np.array([r[f"{gran}_r_raw"] for r in rows]), y),
            "median_r_eff": float(np.median(x)), "median_k_suf": float(np.median(y)),
            "median_split_ratio": float(np.median([r[f"{gran}_split"] for r in rows])),
            "passes": bool(part >= RHO_BAR and lo > 0)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True)
    p.add_argument("--dumps", default="runs/cee")
    p.add_argument("--replication", required=True)
    p.add_argument("--out", default="results/pr_validation")
    a = p.parse_args()

    rows = primary(Bundle(a.bundle), Path(a.dumps))
    lines, res = [], {"primary": {}, "replication": {}}
    say = lines.append
    recon = float(np.mean([r["recon"] for r in rows]))
    res["recon"] = recon
    say(f"PR validation — Qwen2.5-0.5B-Instruct, {len(rows)} positions; full-vocab recon {recon:.3f}")

    cells = []
    for corpus in ("prose", "code"):
        rs = [r for r in rows if r["corpus"] == corpus]
        clusters = np.array([r["sid"] for r in rs])
        head = float(np.mean([r["blk_head"] for r in rs]))
        er = spearman(np.array([r["blk_r_eff"] for r in rs]), np.array([r["erank"] for r in rs]))
        res["primary"][corpus] = {"head_sufficiency": head, "rho_reff_erank": er,
                                  "median_erank": float(np.median([r["erank"] for r in rs]))}
        say(f"\n=== {corpus} (n={len(rs)}, {len(set(clusters))} units) ===")
        say(f"  Claim B: PR-head sufficiency {head:.3f}   ρ(r_eff, erank) {er:+.2f}   "
            f"median erank {res['primary'][corpus]['median_erank']:.1f}")
        for gran in ("blk", "lay"):
            c = claim_a_cell(rs, gran, clusters)
            res["primary"][corpus][gran] = c
            cells.append(c["passes"])
            say(f"  Claim A [{gran}]: partial ρ(r_eff,k_suf|margin) {c['partial']:+.3f} "
                f"CI [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] {'PASS' if c['passes'] else '----'}   "
                f"ρ(r_eff,k_suf) {c['rho_reff_ksuf']:+.2f}  ρ(r_raw,k_suf) {c['rho_raw_ksuf']:+.2f}  "
                f"ρ(margin,k_suf) {c['rho_margin_ksuf']:+.2f}  med r_eff {c['median_r_eff']:.1f}  "
                f"med k_suf {c['median_k_suf']:.0f}  split×4 ratio {c['median_split_ratio']:.2f}")

    say("\n=== replication (candidate-restricted step0 dumps; not a gate) ===")
    for name, rs in replication(Path(a.replication)).items():
        res["replication"][name] = {}
        for gran in ("blk", "lay"):
            c = claim_a_cell(rs, gran, None)
            res["replication"][name][gran] = c
            say(f"  {name:16s} [{gran}] n={c['n']}  partial {c['partial']:+.3f} "
                f"CI [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]  ρ(r_eff,k_suf) {c['rho_reff_ksuf']:+.2f}  "
                f"ρ(margin,k_suf) {c['rho_margin_ksuf']:+.2f}")

    a_verdict = "SUPPORTED" if all(cells) else ("FRAGILE" if any(cells) else "NOT SUPPORTED")
    heads = [res["primary"][c]["head_sufficiency"] for c in ("prose", "code")]
    b_verdict = "HOLDS" if min(heads) >= 0.8 else ("FAILS" if max(heads) < 0.5 else "PARTIAL")
    res["claim_a"], res["claim_b"] = a_verdict, b_verdict
    say(f"\nClaim A (r_eff tracks coalition size beyond margin): {a_verdict}")
    say(f"Claim B (top round(r_eff) blocks decide): {b_verdict}")
    Path(f"{a.out}.json").write_text(json.dumps(res, indent=1) + "\n")
    Path(f"{a.out}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
