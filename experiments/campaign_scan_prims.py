"""SCAN prim admit + 3-block stack (standalone).

Successor to campaign_scan_learned.py:

  1. **Leaf phrase formation always on** — ``dir`` (and bare ``turn L/R``) is lexicon
     syntax, not a combinator. Val never sees held-out systematic uses of a new prim
     with direction (addprim_jump: ``jump left``), so gating ``dir`` kills systematicity.
  2. **B0 prim admit** — greedy val-marginal over fit-mined true leaves
     (unigram actions + ``turn left/right``), scored under the full combinator
     grammar (isolated leaves have ~0 marginal on composed val commands).
  3. **B1+B2 combinators** — freeze prims; **joint** greedy admit over unary∪binary
     (staged unary-then-binary fails when operators are synergistic: e.g. almost every
     ``around`` val command also needs ``and``/``after``, so unary alone has ~0 marginal).
     Admitted names are partitioned into B1 (unary) / B2 (binary) for provenance.

Reports vs exact / prim_compose / prior flat learned_admit (combinators only, bulk prims).

Standalone: WordCodec train-only, no host LLM, SOFT=0.

Run:
  cd pil && .venv/bin/python experiments/build_scan.py   # if needed
  cd pil && .venv/bin/python -u experiments/campaign_scan_prims.py
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.alphabet import WordCodec  # noqa: E402
from pil.wyly_block import WylyBlock  # noqa: E402

sys.path.insert(0, str(REPO / "experiments"))
import campaign_scan_learned as learned  # noqa: E402
import campaign_scan_standalone as base  # noqa: E402

SPLITS = ["length", "addprim_jump", "simple"]
# Phrase formation — always enabled (not admitted).
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


def true_leaf_candidates(
    train_in, train_out, codec: WordCodec,
) -> dict[tuple[str, ...], list[str]]:
    """Unigram actions + turn left/right only (compositional leaves)."""
    prims: dict[tuple[str, ...], list[str]] = {}
    for ci, ao in zip(train_in, train_out, strict=True):
        ct = tuple(base.decode_toks(codec, ci))
        at = base.decode_toks(codec, ao)
        if len(ct) == 1 or (len(ct) == 2 and ct[0] == "turn" and ct[1] in ("left", "right")):
            prims[ct] = at
    return prims


def score_exact(
    pairs: list[tuple[list[int], list[int]]],
    codec: WordCodec,
    prims: dict,
    enabled: set[str],
) -> float:
    """Exact-match with LEAF_ALWAYS (dir) unioned into enabled."""
    en = set(enabled) | set(LEAF_ALWAYS)
    return learned.score_exact(pairs, codec, prims, en)


def admit_prims(
    candidates: dict[tuple[str, ...], list[str]],
    val_pairs,
    codec: WordCodec,
    enabled_comb: set[str],
    thresh: float = 1e-4,
) -> tuple[dict[tuple[str, ...], list[str]], list[dict]]:
    """Greedy forward selection of leaf prim maps by val exact-match marginal."""
    admitted: dict[tuple[str, ...], list[str]] = {}
    log_rows: list[dict] = []
    remaining = list(candidates.items())
    base_sc = score_exact(val_pairs, codec, admitted, enabled_comb)
    while remaining:
        best = (thresh, None, None)  # marg, key, actions
        for key, acts in remaining:
            trial = dict(admitted)
            trial[key] = acts
            sc = score_exact(val_pairs, codec, trial, enabled_comb)
            marg = sc - base_sc
            log_rows.append({
                "prim": " ".join(key), "marginal": marg, "score": sc,
                "n_admitted": len(admitted),
            })
            if marg > best[0]:
                best = (marg, key, acts)
        if best[1] is None:
            break
        admitted[best[1]] = best[2]
        remaining = [(k, a) for k, a in remaining if k != best[1]]
        base_sc = score_exact(val_pairs, codec, admitted, enabled_comb)
        log(f"    ADMITTED prim {' '.join(best[1])!r} (+{best[0]:.4f} → {base_sc:.3f})")
    return admitted, log_rows


def admit_combinators(
    prims: dict,
    val_pairs,
    codec: WordCodec,
    pool: tuple[str, ...],
    already: set[str],
    thresh: float = 1e-4,
) -> tuple[set[str], list[dict]]:
    """Greedy admit from pool; ``already`` stays enabled."""
    enabled = set(already)
    log_rows: list[dict] = []
    base_sc = score_exact(val_pairs, codec, prims, enabled)
    remaining = [c for c in pool if c not in enabled]
    while remaining:
        best = (thresh, None)
        for c in remaining:
            trial = enabled | {c}
            sc = score_exact(val_pairs, codec, prims, trial)
            marg = sc - base_sc
            log_rows.append({
                "combinator": c, "marginal": marg, "score": sc,
                "enabled_so_far": sorted(enabled),
            })
            if marg > best[0]:
                best = (marg, c)
        if best[1] is None:
            break
        enabled.add(best[1])
        remaining.remove(best[1])
        base_sc = score_exact(val_pairs, codec, prims, enabled)
        log(f"    ADMITTED combinator {best[1]!r} (+{best[0]:.4f} → {base_sc:.3f})")
    return enabled, log_rows


def split_fit_val(train_in, train_out, val_frac=0.1, seed=0):
    n = len(train_in)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    nval = max(1, int(n * val_frac))
    val_i, fit_i = idx[:nval], idx[nval:]
    fit = [(train_in[i], train_out[i]) for i in fit_i]
    val = [(train_in[i], train_out[i]) for i in val_i]
    return fit, val


def run_split(tag: str) -> dict:
    blob, codec = base.load_split(tag)
    tr_in, tr_out = blob["train_in"], blob["train_out"]
    te_in, te_out = blob["test_in"], blob["test_out"]
    fit, val = split_fit_val(tr_in, tr_out, val_frac=0.1, seed=0)
    fit_in = [c for c, _ in fit]
    fit_out = [a for _, a in fit]
    test_pairs = list(zip(te_in, te_out, strict=True))

    # Baselines
    m = base.exact_map(tr_in, tr_out)
    acc_exact = base.eval_exact(m, te_in, te_out)
    prims_all = base.mine_scan_grammar(tr_in, tr_out, codec)
    acc_pc, cov_pc = base.eval_prim_comp(prims_all, te_in, te_out, codec)

    full_comb = set(UNARY_POOL) | set(BINARY_POOL)

    # Prior flat path: bulk short prims + combinator admit, dir gated (PR #60)
    prims_bulk = base.mine_scan_grammar(fit_in, fit_out, codec)
    en_flat, _ = learned.admit_combinators(prims_bulk, val, codec)
    acc_flat = learned.score_exact(test_pairs, codec, prims_bulk, en_flat)
    # Same flat path but with leaf dir always-on (systematicity fix)
    acc_flat_dir = score_exact(test_pairs, codec, prims_bulk, en_flat)

    cand = true_leaf_candidates(fit_in, fit_out, codec)
    log(f"\n=== SCAN/{tag} prim admit + 3-block ===")
    log(f"  train={len(tr_in)} fit={len(fit)} val={len(val)} test={len(te_in)} "
        f"leaf_cands={len(cand)} {sorted(' '.join(k) for k in cand)}")

    def _dummy_fn(_ids):
        return torch.full((len(_ids),), -1, dtype=torch.long)

    # --- B0: admit prims under full combinator grammar (dir always-on) ---
    # Isolated prims (no combinators) have near-zero val marginal on composed
    # commands; leaves must be scored under the composition operators they unlock.
    log("  -- B0 prims (scored under full unary+binary) --")
    b0 = WylyBlock(0, "prim_lexicon", depends_on=[])
    for key in cand:
        b0.add_candidate(" ".join(key), _dummy_fn)
    prims_adm, prim_log = admit_prims(cand, val, codec, enabled_comb=full_comb)
    for key in prims_adm:
        b0.rules.append((b0._qualify(" ".join(key)), _dummy_fn))
    acc_b0 = score_exact(test_pairs, codec, prims_adm, set())  # prims + dir only
    acc_b0_full = score_exact(test_pairs, codec, prims_adm, full_comb)

    # --- B1+B2: joint combinator admit (prims frozen), then partition by kind ---
    log("  -- B1+B2 joint combinators (unary∪binary) --")
    joint_pool = UNARY_POOL + BINARY_POOL
    en_joint, comb_log = admit_combinators(
        prims_adm, val, codec, joint_pool, already=set())
    unary_only = en_joint & set(UNARY_POOL)
    binary_only = en_joint & set(BINARY_POOL)
    b1 = WylyBlock(1, "unary", depends_on=[0])
    for c in UNARY_POOL:
        b1.add_candidate(c, _dummy_fn)
    for c in unary_only:
        b1.rules.append((b1._qualify(c), _dummy_fn))
    b2 = WylyBlock(2, "binary", depends_on=[0, 1])
    for c in BINARY_POOL:
        b2.add_candidate(c, _dummy_fn)
    for c in binary_only:
        b2.rules.append((b2._qualify(c), _dummy_fn))
    acc_b1 = score_exact(test_pairs, codec, prims_adm, unary_only)
    acc_stack = score_exact(test_pairs, codec, prims_adm, en_joint)

    # Diagnostic: staged unary→binary (expected weak when operators synergize)
    log("  -- diagnostic staged unary then binary --")
    en_u, _ = admit_combinators(prims_adm, val, codec, UNARY_POOL, already=set())
    en_st, _ = admit_combinators(prims_adm, val, codec, BINARY_POOL, already=en_u)
    acc_staged = score_exact(test_pairs, codec, prims_adm, en_st)

    # Bulk true-leaves + full comb (no prim admit; ceiling for fit leaves)
    leaves_fit = true_leaf_candidates(fit_in, fit_out, codec)
    acc_leaves_full = score_exact(test_pairs, codec, leaves_fit, full_comb)

    log(f"  exact              {acc_exact:.3f}")
    log(f"  prim_compose       {acc_pc:.3f}")
    log(f"  flat learned (old) {acc_flat:.3f}  comb={sorted(en_flat)}")
    log(f"  flat + leaf dir    {acc_flat_dir:.3f}  (dir always-on)")
    log(f"  B0 prims+dir only  {acc_b0:.3f}  prims={[' '.join(k) for k in sorted(prims_adm)]}")
    log(f"  B0 + full comb     {acc_b0_full:.3f}")
    log(f"  B0+B1 unary-only   {acc_b1:.3f}  unary={sorted(unary_only)}")
    log(f"  B0+B1+B2 joint     {acc_stack:.3f}  enabled={sorted(en_joint)}")
    log(f"  staged B1→B2 diag  {acc_staged:.3f}  staged={sorted(en_st)}")
    log(f"  fit-leaves+full    {acc_leaves_full:.3f}  (bulk true leaves, no admit)")

    return {
        "split": tag,
        "n_train": len(tr_in),
        "n_fit": len(fit),
        "n_val": len(val),
        "n_test": len(te_in),
        "n_leaf_cands": len(cand),
        "exact": acc_exact,
        "prim_compose": acc_pc,
        "prim_cover": cov_pc,
        "flat_learned_old": acc_flat,
        "flat_learned_dir": acc_flat_dir,
        "flat_combinators": sorted(en_flat),
        "block0_prims_dir_only": acc_b0,
        "block0_prims_full_comb": acc_b0_full,
        "admitted_prims": [" ".join(k) for k in sorted(prims_adm)],
        "block1_unary": acc_b1,
        "unary_combinators": sorted(unary_only),
        "block_stack": acc_stack,
        "binary_combinators": sorted(binary_only),
        "enabled_all": sorted(en_joint),
        "staged_unary_then_binary": acc_staged,
        "staged_enabled": sorted(en_st),
        "fit_leaves_full": acc_leaves_full,
        "prim_admit_log": prim_log[:30],
        "comb_admit_log": comb_log[:40],
        "blocks": [b0.summary(), b1.summary(), b2.summary()],
        "leaf_always": sorted(LEAF_ALWAYS),
        "learning_gap": acc_stack - acc_exact,
        "origin": "standalone",
        "alphabet": "word",
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
    log("SCAN PRIM ADMIT + 3-BLOCK SCOREBOARD")
    log("=" * 78)
    log(f"{'split':14} {'exact':>7} {'prim':>7} {'flat':>7} {'+dir':>7} "
        f"{'B0f':>7} {'stack':>7} {'staged':>7} {'leaves':>7}")
    for r in results:
        log(f"{r['split']:14} {r['exact']:7.3f} {r['prim_compose']:7.3f} "
            f"{r['flat_learned_old']:7.3f} {r['flat_learned_dir']:7.3f} "
            f"{r['block0_prims_full_comb']:7.3f} {r['block_stack']:7.3f} "
            f"{r['staged_unary_then_binary']:7.3f} {r['fit_leaves_full']:7.3f}")
        log(f"  prims={r['admitted_prims']}  unary={r['unary_combinators']}  "
            f"binary={r['binary_combinators']}")
    out = REPO / "data" / "scan_prims_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
