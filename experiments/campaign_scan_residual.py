"""SCAN residual templates on simple failure modes (standalone).

Not depth theatre: block-private **residual leaves** recovered from short composite
maps, admitted by val marginal under the full combinator grammar.

  B0 base     — short maps (len ≤ 2) from fit train
  B0 residual — induce bare leaves from composites, e.g. ``run twice`` → ``run``;
                ``turn left/right`` structural templates
  B1+B2       — joint unary∪binary combinators (as in multi-block campaign)

Also reports prim_compose with/without residual for the grammar ceiling.

Standalone: WordCodec train-only, SOFT=0.

Run:
  cd pil && .venv/bin/python -u experiments/campaign_scan_residual.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.wyly_block import BlockStack, scan_stack_spec  # noqa: E402

sys.path.insert(0, str(REPO / "experiments"))
import campaign_scan_learned as learned  # noqa: E402
import campaign_scan_prims as prims_camp  # noqa: E402
import campaign_scan_standalone as base  # noqa: E402

SPLITS = ["length", "addprim_jump", "simple"]
LEAF_ALWAYS = frozenset({"dir"})
UNARY_POOL = ("twice", "thrice", "around", "opposite")
BINARY_POOL = ("and", "after")


def log(m=""):
    print(m, flush=True)


def ensure_data():
    missing = [s for s in SPLITS if not (REPO / "data" / f"scan_{s}.pt").exists()]
    if missing:
        log(f"building SCAN datasets for {missing}...")
        subprocess.check_call(
            [sys.executable, str(REPO / "experiments" / "build_scan.py")], cwd=str(REPO))


def score_exact(pairs, codec, prims, enabled):
    return learned.score_exact(pairs, codec, prims, set(enabled) | set(LEAF_ALWAYS))


def admit_residual_leaves(
    base_prims: dict,
    residual_cands: dict,
    val_pairs,
    codec,
    enabled_comb: set[str],
    thresh: float = 1e-4,
) -> tuple[dict, list[dict]]:
    """Greedy admit residual leaf maps by val marginal under fixed combinators."""
    admitted: dict = dict(base_prims)
    log_rows: list[dict] = []
    remaining = list(residual_cands.items())
    base_sc = score_exact(val_pairs, codec, admitted, enabled_comb)
    while remaining:
        best = (thresh, None, None)
        for key, acts in remaining:
            trial = dict(admitted)
            trial[key] = acts
            sc = score_exact(val_pairs, codec, trial, enabled_comb)
            marg = sc - base_sc
            log_rows.append({
                "residual": " ".join(key), "marginal": marg, "score": sc,
            })
            if marg > best[0]:
                best = (marg, key, acts)
        if best[1] is None:
            break
        admitted[best[1]] = best[2]
        remaining = [(k, a) for k, a in remaining if k != best[1]]
        base_sc = score_exact(val_pairs, codec, admitted, enabled_comb)
        log(f"    ADMITTED residual leaf {' '.join(best[1])!r} "
            f"(+{best[0]:.4f} → {base_sc:.3f})")
    return admitted, log_rows


def failure_modes(pairs, codec, prims, enabled, k=8):
    en = set(enabled) | set(LEAF_ALWAYS)
    fails = []
    for ci, ao in pairs:
        ct = base.decode_toks(codec, ci)
        gold = base.decode_toks(codec, ao)
        pred = learned.expand_gated(ct, prims, en)
        if pred is None or pred != gold:
            fails.append({
                "cmd": ct,
                "mode": "none" if pred is None else "wrong",
                "gold_head": gold[:10],
            })
        if len(fails) >= k:
            break
    return fails


def run_split(tag: str) -> dict:
    blob, codec = base.load_split(tag)
    tr_in, tr_out = blob["train_in"], blob["train_out"]
    te_in, te_out = blob["test_in"], blob["test_out"]
    fit, val = prims_camp.split_fit_val(tr_in, tr_out, val_frac=0.1, seed=0)
    fit_in = [c for c, _ in fit]
    fit_out = [a for _, a in fit]
    test_pairs = list(zip(te_in, te_out, strict=True))
    full_comb = set(UNARY_POOL) | set(BINARY_POOL)

    # Baselines
    m = base.exact_map(tr_in, tr_out)
    acc_exact = base.eval_exact(m, te_in, te_out)
    # raw short maps only (no residual)
    prims_raw = base.mine_scan_grammar(tr_in, tr_out, codec, residual_leaves=False)
    acc_pc_raw, _ = base.eval_prim_comp(prims_raw, te_in, te_out, codec)
    # full train with residual leaves (grammar ceiling)
    prims_full = base.mine_scan_grammar(tr_in, tr_out, codec, residual_leaves=True)
    acc_pc, cov_pc = base.eval_prim_comp(prims_full, te_in, te_out, codec)
    residual_full = {
        k: v for k, v in base.induce_residual_leaves(prims_raw).items()
        if k not in prims_raw
    }

    # Fit short maps + residual candidates
    base_fit = base.mine_scan_grammar(fit_in, fit_out, codec, residual_leaves=False)
    res_cands = base.induce_residual_leaves(base_fit)
    res_cands = {k: v for k, v in res_cands.items() if k not in base_fit}

    log(f"\n=== SCAN/{tag} residual templates ===")
    log(f"  train={len(tr_in)} fit={len(fit)} val={len(val)} test={len(te_in)}")
    log(f"  short prims_fit={len(base_fit)} residual_cands={len(res_cands)} "
        f"{sorted(' '.join(k) for k in res_cands)}")
    log(f"  prim_compose raw (no residual) {acc_pc_raw:.3f}")
    log(f"  prim_compose + residual leaves {acc_pc:.3f}")

    # Admit residual under full combinators
    log("  -- B0 residual leaf admit --")
    prims_adm, res_log = admit_residual_leaves(
        base_fit, res_cands, val, codec, full_comb)
    n_res_adm = sum(1 for k in prims_adm if k not in base_fit)

    # Joint combinators
    log("  -- B1+B2 joint combinators --")
    en_joint, comb_log = prims_camp.admit_combinators(
        prims_adm, val, codec, UNARY_POOL + BINARY_POOL, already=set())

    # re-admit combos on base_fit only for fair no-residual stack
    en_no_res, _ = prims_camp.admit_combinators(
        base_fit, val, codec, UNARY_POOL + BINARY_POOL, already=set())
    acc_stack_no_res = score_exact(test_pairs, codec, base_fit, en_no_res)
    acc_stack = score_exact(test_pairs, codec, prims_adm, en_joint)
    acc_b0_only = score_exact(test_pairs, codec, prims_adm, set())

    # WylyBlock provenance: residual on B0 family
    stack = BlockStack.from_spec(scan_stack_spec())
    b0, b1, b2 = stack.blocks

    def _dummy(_ids):
        return torch.full((len(_ids),), -1, dtype=torch.long)

    for key in base_fit:
        b0.add_candidate("base:" + " ".join(key), _dummy)
    for key in res_cands:
        b0.add_candidate("residual:" + " ".join(key), _dummy)
    for key in prims_adm:
        prefix = "residual:" if key not in base_fit else "base:"
        b0.rules.append((b0._qualify(prefix + " ".join(key)), _dummy))
    for c in en_joint & set(UNARY_POOL):
        b1.rules.append((b1._qualify(c), _dummy))
    for c in en_joint & set(BINARY_POOL):
        b2.rules.append((b2._qualify(c), _dummy))
    b0.set_residual(
        residual_leaves=[" ".join(k) for k in sorted(prims_adm) if k not in base_fit],
        n_base=len(base_fit),
        n_residual_admitted=n_res_adm,
    )

    fails = failure_modes(test_pairs, codec, prims_adm, en_joint, k=6)
    gap = acc_stack - acc_exact

    log(f"  exact                 {acc_exact:.3f}")
    log(f"  stack without residual{acc_stack_no_res:.3f}")
    log(f"  B0 residual+dir only  {acc_b0_only:.3f}")
    log(f"  stack + residual      {acc_stack:.3f}  enabled={sorted(en_joint)}")
    log(f"  residual leaves adm   {[ ' '.join(k) for k in sorted(prims_adm) if k not in base_fit]}")
    log(f"  gap vs exact          {gap:+.3f}")
    log(f"  failures sample       {len(fails)}")

    return {
        "split": tag,
        "n_train": len(tr_in),
        "n_fit": len(fit),
        "n_val": len(val),
        "n_test": len(te_in),
        "exact": acc_exact,
        "prim_compose_raw": acc_pc_raw,
        "prim_compose": acc_pc,
        "prim_cover": cov_pc,
        "stack_no_residual": acc_stack_no_res,
        "block0_residual_dir": acc_b0_only,
        "block_stack": acc_stack,
        "residual_candidates": [" ".join(k) for k in sorted(res_cands)],
        "residual_admitted": [
            " ".join(k) for k in sorted(prims_adm) if k not in base_fit
        ],
        "full_train_residual_leaves": [" ".join(k) for k in sorted(residual_full)],
        "combinators": sorted(en_joint),
        "residual_admit_log": res_log[:20],
        "comb_admit_log": comb_log[:20],
        "failure_samples": fails,
        "blocks": stack.summary(),
        "learning_gap": gap,
        "origin": "standalone",
        "alphabet": "word",
        "method": "residual_leaf_templates",
    }


def main():
    ensure_data()
    want = [s.strip() for s in os.environ.get("SCAN_SPLITS", ",".join(SPLITS)).split(",")
            if s.strip()]
    results = []
    for tag in want:
        if not (REPO / "data" / f"scan_{tag}.pt").exists():
            log(f"missing scan_{tag}.pt — skip")
            continue
        results.append(run_split(tag))
    log("\n" + "=" * 78)
    log("SCAN RESIDUAL TEMPLATES SCOREBOARD")
    log("=" * 78)
    log(f"{'split':14} {'exact':>7} {'pcRaw':>7} {'pcRes':>7} {'noRes':>7} "
        f"{'stack':>7} {'resLeaves':>28}")
    for r in results:
        leaves = ",".join(r["residual_admitted"])[:28]
        log(f"{r['split']:14} {r['exact']:7.3f} {r['prim_compose_raw']:7.3f} "
            f"{r['prim_compose']:7.3f} {r['stack_no_residual']:7.3f} "
            f"{r['block_stack']:7.3f} {leaves:>28}")
    out = REPO / "data" / "scan_residual_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
