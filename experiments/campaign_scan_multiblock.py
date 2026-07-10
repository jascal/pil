"""SCAN multi-block learned admission with stack diagnostics (standalone).

Uses ``pil.wyly_block.BlockStack`` / ``scan_stack_spec`` for configurable 3-block
families (prims → unary → binary), full learned admission, and per-block marginals.

Also reports:
  - exact / prim_compose baselines (with turn-around expand fix)
  - joint vs staged combinator admit (synergy diagnostic)
  - per-block cumulative test accuracy
  - failure-mode samples (unparsed under stack)

Standalone: WordCodec train-only, no host LLM, SOFT=0.

Run:
  cd pil && .venv/bin/python -u experiments/campaign_scan_multiblock.py
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
    en = set(enabled) | set(LEAF_ALWAYS)
    return learned.score_exact(pairs, codec, prims, en)


def failure_samples(pairs, codec, prims, enabled, k=5):
    """Commands that fail to parse or mismatch under current enable set."""
    en = set(enabled) | set(LEAF_ALWAYS)
    fails = []
    for ci, ao in pairs:
        ct = base.decode_toks(codec, ci)
        gold = base.decode_toks(codec, ao)
        pred = learned.expand_gated(ct, prims, en)
        if pred is None:
            fails.append({"cmd": ct, "gold": gold[:12], "mode": "none"})
        elif pred != gold:
            fails.append({"cmd": ct, "gold": gold[:12], "pred": pred[:12], "mode": "wrong"})
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
    val_pairs = val

    # --- baselines ---
    m = base.exact_map(tr_in, tr_out)
    acc_exact = base.eval_exact(m, te_in, te_out)
    prims_all = base.mine_scan_grammar(tr_in, tr_out, codec)
    acc_pc, cov_pc = base.eval_prim_comp(prims_all, te_in, te_out, codec)

    full_comb = set(UNARY_POOL) | set(BINARY_POOL)
    cand = prims_camp.true_leaf_candidates(fit_in, fit_out, codec)

    log(f"\n=== SCAN/{tag} multi-block ===")
    log(f"  train={len(tr_in)} fit={len(fit)} val={len(val)} test={len(te_in)} "
        f"leaf_cands={len(cand)} prim_compose={acc_pc:.3f}")

    def _dummy(_ids):
        return torch.full((len(_ids),), -1, dtype=torch.long)

    # Build stack from canonical SCAN spec
    stack = BlockStack.from_spec(scan_stack_spec())
    b0, b1, b2 = stack.blocks
    for key in cand:
        b0.add_candidate(" ".join(key), _dummy)
    for c in UNARY_POOL:
        b1.add_candidate(c, _dummy)
    for c in BINARY_POOL:
        b2.add_candidate(c, _dummy)

    # --- B0: admit prims under full combinator grammar ---
    log("  -- B0 prims (stack marginal under full comb) --")
    prims_adm, prim_log = prims_camp.admit_prims(
        cand, val_pairs, codec, enabled_comb=full_comb)
    for key in prims_adm:
        name = b0._qualify(" ".join(key))
        b0.rules.append((name, _dummy))
    b0.set_residual(prims=[" ".join(k) for k in sorted(prims_adm)], leaf_always=list(LEAF_ALWAYS))

    # Score helpers bound to admitted prims
    def score_enabled(enabled: set[str]) -> float:
        return score_exact(val_pairs, codec, prims_adm, enabled)

    # --- Joint combinators (synergistic operators) via freeze-upstream score ---
    log("  -- B1+B2 joint combinators (then partition families) --")
    joint_pool = UNARY_POOL + BINARY_POOL
    en_joint, comb_log = prims_camp.admit_combinators(
        prims_adm, val_pairs, codec, joint_pool, already=set())
    unary_adm = en_joint & set(UNARY_POOL)
    binary_adm = en_joint & set(BINARY_POOL)
    for c in unary_adm:
        b1.rules.append((b1._qualify(c), _dummy))
    for c in binary_adm:
        b2.rules.append((b2._qualify(c), _dummy))
    b1.set_residual(unary=sorted(unary_adm))
    b2.set_residual(binary=sorted(binary_adm), enabled=sorted(en_joint))

    # Staged diagnostic (expected weak when synergy)
    en_u, _ = prims_camp.admit_combinators(
        prims_adm, val_pairs, codec, UNARY_POOL, already=set())
    en_st, _ = prims_camp.admit_combinators(
        prims_adm, val_pairs, codec, BINARY_POOL, already=en_u)
    acc_staged = score_exact(test_pairs, codec, prims_adm, en_st)

    # Per-block cumulative test accuracy
    acc_b0 = score_exact(test_pairs, codec, prims_adm, set())
    acc_b0_u = score_exact(test_pairs, codec, prims_adm, unary_adm)
    acc_stack = score_exact(test_pairs, codec, prims_adm, en_joint)

    # Val prefix marginals for diagnostics
    def score_rules(rules):
        # map qualified rule names back to combinator/prim enable set
        en: set[str] = set()
        prims_local = dict(prims_adm)
        for n, _ in rules:
            short = n.split("/", 1)[-1]
            if short in UNARY_POOL or short in BINARY_POOL:
                en.add(short)
            # prims already in prims_adm; rule names are spaces
        return score_exact(val_pairs, codec, prims_local, en)

    # Build synthetic rule lists for marginal report
    r0 = list(b0.rules)
    r1 = r0 + list(b1.rules)
    r2 = r1 + list(b2.rules)
    marg_rows = [
        {"block_id": 0, "family": "prims", "score": score_rules(r0),
         "marginal": score_rules(r0) - score_rules([]), "admitted": [n for n, _ in b0.rules]},
        {"block_id": 1, "family": "unary", "score": score_rules(r1),
         "marginal": score_rules(r1) - score_rules(r0), "admitted": [n for n, _ in b1.rules]},
        {"block_id": 2, "family": "binary", "score": score_rules(r2),
         "marginal": score_rules(r2) - score_rules(r1), "admitted": [n for n, _ in b2.rules]},
    ]

    # Use BlockStack.per_block_marginals with a score on combinators only (prims fixed)
    def score_fn_comb_rules(rules):
        en = set()
        for n, _ in rules:
            short = n.split("/", 1)[-1]
            if short in UNARY_POOL or short in BINARY_POOL:
                en.add(short)
        return score_exact(val_pairs, codec, prims_adm, en)

    # only B1+B2 rules in this helper stack
    sub = BlockStack([b1, b2], carry="merge", name="comb_only")
    comb_marginals = sub.per_block_marginals(score_fn_comb_rules)

    fails = failure_samples(test_pairs, codec, prims_adm, en_joint, k=6)
    gap_to_pc = acc_stack - acc_pc

    log(f"  exact              {acc_exact:.3f}")
    log(f"  prim_compose       {acc_pc:.3f}  cover={cov_pc:.3f}")
    log(f"  B0 prims+dir       {acc_b0:.3f}  prims={[' '.join(k) for k in sorted(prims_adm)]}")
    log(f"  B0+B1 unary        {acc_b0_u:.3f}  unary={sorted(unary_adm)}")
    log(f"  B0+B1+B2 stack     {acc_stack:.3f}  binary={sorted(binary_adm)}")
    log(f"  staged B1→B2       {acc_staged:.3f}")
    log(f"  gap vs prim_comp   {gap_to_pc:+.3f}")
    for row in marg_rows:
        log(f"  val marg B{row['block_id']} ({row['family']}): "
            f"score={row['score']:.3f} Δ={row['marginal']:+.3f}")

    # Gated carry smoke: forward dummy ids, ensure residual merges
    dummy_ids = torch.zeros(4, 3, dtype=torch.long)
    stack.forward(dummy_ids)
    carried = stack.last_carried[-1].residual if stack.last_carried else {}

    return {
        "split": tag,
        "n_train": len(tr_in),
        "n_fit": len(fit),
        "n_val": len(val),
        "n_test": len(te_in),
        "exact": acc_exact,
        "prim_compose": acc_pc,
        "prim_cover": cov_pc,
        "block0_prims": acc_b0,
        "block0_plus_unary": acc_b0_u,
        "block_stack": acc_stack,
        "staged_unary_then_binary": acc_staged,
        "gap_vs_prim_compose": gap_to_pc,
        "admitted_prims": [" ".join(k) for k in sorted(prims_adm)],
        "unary_combinators": sorted(unary_adm),
        "binary_combinators": sorted(binary_adm),
        "per_block_val_marginals": marg_rows,
        "comb_prefix_marginals": comb_marginals,
        "failure_samples": fails,
        "n_failures_sample": len(fails),
        "stack_summary": stack.summary(),
        "carried_residual_keys": sorted(carried.keys()),
        "leaf_always": sorted(LEAF_ALWAYS),
        "origin": "standalone",
        "alphabet": "word",
        "expand_note": "turn around L/R = 4 turns (grammar residual closed)",
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
    log("SCAN MULTI-BLOCK LEARNED ADMISSION SCOREBOARD")
    log("=" * 78)
    log(f"{'split':14} {'exact':>7} {'prim':>7} {'B0':>7} {'B0+U':>7} {'stack':>7} "
        f"{'staged':>7} {'gap':>7}")
    for r in results:
        log(f"{r['split']:14} {r['exact']:7.3f} {r['prim_compose']:7.3f} "
            f"{r['block0_prims']:7.3f} {r['block0_plus_unary']:7.3f} "
            f"{r['block_stack']:7.3f} {r['staged_unary_then_binary']:7.3f} "
            f"{r['gap_vs_prim_compose']:+7.3f}")
        log(f"  prims={r['admitted_prims']}  U={r['unary_combinators']}  "
            f"B={r['binary_combinators']}")
    out = REPO / "data" / "scan_multiblock_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
