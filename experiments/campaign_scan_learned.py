"""SCAN learned admit + 2-block stack (standalone).

Fills the ``learned_admit`` slot left by campaign_scan_standalone.py:

  1. Mine prims from short train commands (fit region).
  2. **Flat learned admit**: greedy cover-marginal admission of SCAN combinators
     (twice, thrice, around, opposite, and, after, dir) on a held-out val slice of train.
  3. **2-block stack**: B0 = prims only; B1 admits combinators by stack marginal
     (same combinator pool, depth-structured provenance).
  4. Report exact-match action-sequence accuracy on official test splits vs exact / prim_compose.

Standalone: WordCodec train-only alphabet, no host LLM, SOFT=0 (symbolic rules only).

Run:
  cd pil && .venv/bin/python experiments/build_scan.py   # if needed
  cd pil && .venv/bin/python -u experiments/campaign_scan_learned.py
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

# Reuse expand / mine / decode from the flat campaign
sys.path.insert(0, str(REPO / "experiments"))
import campaign_scan_standalone as base  # noqa: E402

SPLITS = ["length", "addprim_jump", "simple"]
# Combinators that can be admitted (fixed operational semantics; admission is data-driven).
# ``dir`` (P left/right phrase formation) is leaf syntax, not a combinator — val never
# sees held-out systematic uses (addprim_jump: jump left), so gating dir kills
# systematicity. See campaign_scan_prims.py / docs/notes/scan_standalone.md.
COMBINATOR_POOL = ("and", "after", "twice", "thrice", "around", "opposite")
LEAF_ALWAYS = frozenset({"dir"})


def log(m=""):
    print(m, flush=True)


def ensure_data():
    missing = [s for s in SPLITS if not (REPO / "data" / f"scan_{s}.pt").exists()]
    if missing:
        log(f"building SCAN datasets for {missing}...")
        subprocess.check_call(
            [sys.executable, str(REPO / "experiments" / "build_scan.py")], cwd=str(REPO))


def expand_gated(
    tokens: list[str],
    prims: dict[tuple[str, ...], list[str]],
    enabled: set[str],
) -> list[str] | None:
    """Like base.expand but combinators only fire if present in ``enabled``."""
    if not tokens:
        return []
    if "and" in enabled:
        for i, t in enumerate(tokens):
            if t == "and":
                left = expand_gated(tokens[:i], prims, enabled)
                right = expand_gated(tokens[i + 1 :], prims, enabled)
                if left is None or right is None:
                    return None
                return left + right
    if "after" in enabled:
        for i, t in enumerate(tokens):
            if t == "after":
                left = expand_gated(tokens[:i], prims, enabled)
                right = expand_gated(tokens[i + 1 :], prims, enabled)
                if left is None or right is None:
                    return None
                return right + left
    if "twice" in enabled and len(tokens) >= 2 and tokens[-1] == "twice":
        base_e = expand_gated(tokens[:-1], prims, enabled)
        return None if base_e is None else base_e + base_e
    if "thrice" in enabled and len(tokens) >= 2 and tokens[-1] == "thrice":
        base_e = expand_gated(tokens[:-1], prims, enabled)
        return None if base_e is None else base_e + base_e + base_e
    if (
        "around" in enabled
        and len(tokens) >= 3
        and tokens[-2] == "around"
        and tokens[-1] in ("left", "right")
    ):
        body = expand_gated(tokens[:-2], prims, enabled)
        if body is None:
            return None
        td = base._turn(tokens[-1])
        return (td + body) * 4
    if (
        "opposite" in enabled
        and len(tokens) >= 3
        and tokens[-2] == "opposite"
        and tokens[-1] in ("left", "right")
    ):
        if tokens[0] == "turn" and len(tokens) == 3:
            td = base._turn(tokens[-1])
            return td + td
        body = expand_gated(tokens[:-2], prims, enabled)
        if body is None:
            return None
        td = base._turn(tokens[-1])
        return td + td + body
    # turn left / turn right always available with prims (leaf)
    if len(tokens) == 2 and tokens[0] == "turn" and tokens[1] in ("left", "right"):
        key = tuple(tokens)
        if key in prims:
            return list(prims[key])
        return base._turn(tokens[1])
    if (
        "dir" in enabled
        and len(tokens) >= 2
        and tokens[-1] in ("left", "right")
        and tokens[-2] not in ("around", "opposite", "turn")
    ):
        body = expand_gated(tokens[:-1], prims, enabled)
        if body is None:
            return None
        return base._turn(tokens[-1]) + body
    key = tuple(tokens)
    if key in prims:
        return list(prims[key])
    if len(tokens) == 1 and (tokens[0],) in prims:
        return list(prims[(tokens[0],)])
    return None


def score_exact(
    pairs: list[tuple[list[int], list[int]]],
    codec: WordCodec,
    prims: dict,
    enabled: set[str],
) -> float:
    en = set(enabled) | set(LEAF_ALWAYS)
    ok = tot = 0
    for ci, ao in pairs:
        ct = base.decode_toks(codec, ci)
        gold = base.decode_toks(codec, ao)
        pred = expand_gated(ct, prims, en)
        if pred is not None and pred == gold:
            ok += 1
        tot += 1
    return ok / max(tot, 1)


def admit_combinators(
    prims: dict,
    val_pairs: list[tuple[list[int], list[int]]],
    codec: WordCodec,
    pool: tuple[str, ...] = COMBINATOR_POOL,
    thresh: float = 1e-4,
) -> tuple[set[str], list[dict]]:
    """Greedy forward selection of combinators by val exact-match marginal."""
    enabled: set[str] = set()
    log_rows: list[dict] = []
    base = score_exact(val_pairs, codec, prims, enabled)
    remaining = list(pool)
    while remaining:
        best = (thresh, None)
        for c in remaining:
            trial = enabled | {c}
            sc = score_exact(val_pairs, codec, prims, trial)
            marg = sc - base
            log_rows.append({"combinator": c, "marginal": marg, "score": sc,
                             "enabled_so_far": sorted(enabled)})
            if marg > best[0]:
                best = (marg, c)
        if best[1] is None:
            break
        enabled.add(best[1])
        remaining.remove(best[1])
        base = score_exact(val_pairs, codec, prims, enabled)
        log(f"    ADMITTED combinator {best[1]!r} (+{best[0]:.4f} → {base:.3f})")
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

    # Baselines (full train prims for prim_compose upper bound on same test)
    m = base.exact_map(tr_in, tr_out)
    acc_exact = base.eval_exact(m, te_in, te_out)
    prims_all = base.mine_scan_grammar(tr_in, tr_out, codec)
    acc_pc, cov_pc = base.eval_prim_comp(prims_all, te_in, te_out, codec)

    # Learned: prims from FIT only; combinators admitted on VAL
    prims_fit = base.mine_scan_grammar(fit_in, fit_out, codec)
    log(f"\n=== SCAN/{tag} learned admit ===")
    log(f"  train={len(tr_in)} fit={len(fit)} val={len(val)} test={len(te_in)} "
        f"prims_fit={len(prims_fit)}")
    val_base = score_exact(val, codec, prims_fit, set())
    log(f"  val prims-only: {val_base:.3f}")
    enabled, admit_log = admit_combinators(prims_fit, val, codec)
    acc_learned = score_exact(list(zip(te_in, te_out, strict=True)), codec, prims_fit, enabled)
    cov_learned = sum(
        1 for ci, _ in zip(te_in, te_out, strict=True)
        if expand_gated(base.decode_toks(codec, ci), prims_fit, enabled) is not None
    ) / max(len(te_in), 1)

    # 2-block: B0 = prims-only score; B1 admits combinators (same algorithm, structured log)
    b0 = WylyBlock(0, "prim_lexicon", depends_on=[])
    b1 = WylyBlock(1, "combinators", depends_on=[0])
    # Represent combinators as candidates; score_fn uses expand_gated with B0 prims + B1 set
    def dummy_fn(ids):
        return torch.full((len(ids),), -1, dtype=torch.long)

    for c in COMBINATOR_POOL:
        b1.add_candidate(c, dummy_fn)

    def score_b1(rules):
        en = {n.split("/", 1)[-1] if "/" in n else n for n, _ in rules}
        # strip b1/ prefix from qualify
        en = {x[3:] if x.startswith("b1/") else x for x in en}
        return score_exact(val, codec, prims_fit, en)

    # start empty B1
    b1_admitted = b1.admit_greedy(score_b1, thresh=1e-4, max_rules=len(COMBINATOR_POOL))
    en_block = set()
    for n in b1_admitted:
        # names are b1/and etc.
        en_block.add(n.split("/", 1)[-1])
    acc_block = score_exact(
        list(zip(te_in, te_out, strict=True)), codec, prims_fit, en_block)
    # B0-only test
    acc_b0 = score_exact(
        list(zip(te_in, te_out, strict=True)), codec, prims_fit, set())

    log(f"  exact           {acc_exact:.3f}")
    log(f"  prim_compose    {acc_pc:.3f}  (all combinators + all-train prims)")
    log(f"  learned_admit   {acc_learned:.3f}  enabled={sorted(enabled)}")
    log(f"  block0 (prims)  {acc_b0:.3f}")
    log(f"  block_stack     {acc_block:.3f}  B1={sorted(en_block)}")
    log(f"  learning_gap    {acc_learned - acc_exact:+.3f}")

    return {
        "split": tag,
        "n_train": len(tr_in),
        "n_fit": len(fit),
        "n_val": len(val),
        "n_test": len(te_in),
        "n_prims_fit": len(prims_fit),
        "exact": acc_exact,
        "prim_compose": acc_pc,
        "prim_cover": cov_pc,
        "learned_admit": acc_learned,
        "learned_cover": cov_learned,
        "learned_combinators": sorted(enabled),
        "admit_log": admit_log[:40],
        "block0_prims_only": acc_b0,
        "block_stack": acc_block,
        "block1_combinators": sorted(en_block),
        "blocks": [b0.summary(), b1.summary()],
        "learning_gap": acc_learned - acc_exact,
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
    log("\n" + "=" * 72)
    log("SCAN LEARNED ADMIT + 2-BLOCK SCOREBOARD")
    log("=" * 72)
    log(f"{'split':14} {'exact':>7} {'prim':>7} {'learned':>8} {'B0':>7} {'stack':>7} "
        f"{'combinators':>28}")
    for r in results:
        comb = ",".join(r["learned_combinators"])[:28]
        log(f"{r['split']:14} {r['exact']:7.3f} {r['prim_compose']:7.3f} "
            f"{r['learned_admit']:8.3f} {r['block0_prims_only']:7.3f} "
            f"{r['block_stack']:7.3f} {comb:>28}")
    out = REPO / "data" / "scan_learned_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
