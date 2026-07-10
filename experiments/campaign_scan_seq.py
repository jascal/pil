"""SCAN next-action tables — flat neural-free seq2seq baseline (standalone).

Complements compositional admit (campaign_scan_prims.py) with pure lookup tables:

  1. **exact** — full command → full action sequence (always ~0: unique commands).
  2. **step** — key=(cmd, t) → action[t]; AR for t=0,1,... until gold length or miss.
  3. **hist** — key=(cmd, last-W actions) → next action (incl. EOS); AR until EOS.
  4. **bag_hist** — key=(frozenset(cmd tokens), last-W) → next; shares across cmd orderings.
  5. **suf_hist** — key=(cmd suffix of length K, last-W) → next; local template transfer.
  6. **hist_only** — key=last-W actions → next (action LM, ignores command).

Also reports teacher-forced next-action accuracy (oracle prefix) vs exact-match AR sequences.

Standalone: WordCodec train-only, no host LLM, SOFT=0. Uses Python dict majority tables
(int64 KeyTable packing overflows for full commands; same exact-lookup semantics).

Run:
  cd pil && .venv/bin/python -u experiments/campaign_scan_seq.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.alphabet import WordCodec  # noqa: E402

sys.path.insert(0, str(REPO / "experiments"))
import campaign_scan_standalone as base  # noqa: E402

SPLITS = ["length", "addprim_jump", "simple"]
EOS = "__EOS__"
MAX_AR = 64  # safety cap (SCAN test actions up to ~48 on length)
HIST_W = 3
CMD_SUFFIX_K = 3


def log(m=""):
    print(m, flush=True)


def ensure_data():
    missing = [s for s in SPLITS if not (REPO / "data" / f"scan_{s}.pt").exists()]
    if missing:
        log(f"building SCAN datasets for {missing}...")
        subprocess.check_call(
            [sys.executable, str(REPO / "experiments" / "build_scan.py")], cwd=str(REPO))


def majority_table(pairs: list[tuple]) -> dict:
    """pairs: (key, val) → keep majority val per key (ties: first seen max count)."""
    buckets: dict = defaultdict(Counter)
    for k, v in pairs:
        buckets[k][v] += 1
    out = {}
    for k, ctr in buckets.items():
        out[k] = ctr.most_common(1)[0][0]
    return out


def fit_tables(
    train_in, train_out, codec: WordCodec, w: int = HIST_W, sk: int = CMD_SUFFIX_K,
):
    step_pairs = []
    hist_pairs = []
    bag_pairs = []
    suf_pairs = []
    hist_only_pairs = []
    exact = {}
    for ci, ao in zip(train_in, train_out, strict=True):
        cmd = tuple(base.decode_toks(codec, ci))
        acts = base.decode_toks(codec, ao)
        exact[cmd] = tuple(acts)
        bag = frozenset(cmd)
        suf = cmd[-sk:] if len(cmd) >= sk else cmd
        prev: list[str] = []
        for t, a in enumerate(acts):
            step_pairs.append(((cmd, t), a))
            h = tuple(prev[-w:])
            hist_pairs.append(((cmd, h), a))
            bag_pairs.append(((bag, h), a))
            suf_pairs.append(((suf, h), a))
            hist_only_pairs.append((h, a))
            prev.append(a)
        # EOS after last action
        h = tuple(prev[-w:])
        hist_pairs.append(((cmd, h), EOS))
        bag_pairs.append(((bag, h), EOS))
        suf_pairs.append(((suf, h), EOS))
        hist_only_pairs.append((h, EOS))
    return {
        "exact": exact,
        "step": majority_table(step_pairs),
        "hist": majority_table(hist_pairs),
        "bag_hist": majority_table(bag_pairs),
        "suf_hist": majority_table(suf_pairs),
        "hist_only": majority_table(hist_only_pairs),
        "n_step": len(step_pairs),
        "n_hist_keys": len({k for k, _ in hist_pairs}),
    }


def ar_hist(
    cmd: tuple[str, ...],
    table: dict,
    key_fn,
    w: int = HIST_W,
    max_len: int = MAX_AR,
) -> list[str] | None:
    """Autoregressive decode with history table; None if first step misses."""
    prev: list[str] = []
    out: list[str] = []
    for _ in range(max_len):
        h = tuple(prev[-w:])
        nxt = table.get(key_fn(cmd, h))
        if nxt is None:
            return None if not out else out  # partial; treat as incomplete
        if nxt == EOS:
            return out
        out.append(nxt)
        prev.append(nxt)
    return out


def ar_step(
    cmd: tuple[str, ...],
    table: dict,
    max_len: int = MAX_AR,
) -> list[str] | None:
    out: list[str] = []
    for t in range(max_len):
        a = table.get((cmd, t))
        if a is None:
            break
        out.append(a)
    return out if out else None


def teacher_forced_acc(
    pairs,
    codec: WordCodec,
    table: dict,
    kind: str,
    w: int = HIST_W,
) -> float:
    """Fraction of (cmd,t) or (cmd,hist) next-action queries that match gold."""
    ok = tot = 0
    for ci, ao in pairs:
        cmd = tuple(base.decode_toks(codec, ci))
        acts = base.decode_toks(codec, ao)
        prev: list[str] = []
        for t, a in enumerate(acts):
            if kind == "step":
                pred = table.get((cmd, t))
            elif kind == "hist":
                pred = table.get((cmd, tuple(prev[-w:])))
            elif kind == "bag_hist":
                pred = table.get((frozenset(cmd), tuple(prev[-w:])))
            elif kind == "suf_hist":
                suf = cmd[-CMD_SUFFIX_K:] if len(cmd) >= CMD_SUFFIX_K else cmd
                pred = table.get((suf, tuple(prev[-w:])))
            elif kind == "hist_only":
                pred = table.get(tuple(prev[-w:]))
            else:
                raise ValueError(kind)
            if pred == a:
                ok += 1
            tot += 1
            prev.append(a)
    return ok / max(tot, 1)


def eval_exact_seq(pred_fn, test_in, test_out, codec) -> tuple[float, float]:
    """pred_fn(cmd_tuple) -> list[str]|None. Returns (exact_match, coverage)."""
    ok = cov = 0
    for ci, ao in zip(test_in, test_out, strict=True):
        cmd = tuple(base.decode_toks(codec, ci))
        gold = base.decode_toks(codec, ao)
        pred = pred_fn(cmd)
        if pred is None:
            continue
        cov += 1
        if pred == gold:
            ok += 1
    n = max(len(test_in), 1)
    return ok / n, cov / n


def run_split(tag: str) -> dict:
    blob, codec = base.load_split(tag)
    tr_in, tr_out = blob["train_in"], blob["train_out"]
    te_in, te_out = blob["test_in"], blob["test_out"]
    test_pairs = list(zip(te_in, te_out, strict=True))

    tables = fit_tables(tr_in, tr_out, codec, w=HIST_W)
    prims = base.mine_scan_grammar(tr_in, tr_out, codec)
    acc_pc, cov_pc = base.eval_prim_comp(prims, te_in, te_out, codec)
    m = base.exact_map(tr_in, tr_out)
    acc_exact = base.eval_exact(m, te_in, te_out)

    # AR exact-match
    acc_step, cov_step = eval_exact_seq(
        lambda c: ar_step(c, tables["step"]), te_in, te_out, codec)
    acc_hist, cov_hist = eval_exact_seq(
        lambda c: ar_hist(c, tables["hist"], lambda cmd, h: (cmd, h)),
        te_in, te_out, codec)
    acc_bag, cov_bag = eval_exact_seq(
        lambda c: ar_hist(c, tables["bag_hist"], lambda cmd, h: (frozenset(cmd), h)),
        te_in, te_out, codec)

    def _suf_key(cmd, h):
        suf = cmd[-CMD_SUFFIX_K:] if len(cmd) >= CMD_SUFFIX_K else cmd
        return (suf, h)

    acc_suf, cov_suf = eval_exact_seq(
        lambda c: ar_hist(c, tables["suf_hist"], _suf_key), te_in, te_out, codec)
    acc_ho, cov_ho = eval_exact_seq(
        lambda c: ar_hist(c, tables["hist_only"], lambda cmd, h: h),
        te_in, te_out, codec)

    # Teacher-forced next-action
    tf_step = teacher_forced_acc(test_pairs, codec, tables["step"], "step")
    tf_hist = teacher_forced_acc(test_pairs, codec, tables["hist"], "hist")
    tf_bag = teacher_forced_acc(test_pairs, codec, tables["bag_hist"], "bag_hist")
    tf_suf = teacher_forced_acc(test_pairs, codec, tables["suf_hist"], "suf_hist")
    tf_ho = teacher_forced_acc(test_pairs, codec, tables["hist_only"], "hist_only")

    log(f"\n=== SCAN/{tag} next-action tables ===")
    log(f"  train={len(tr_in)} test={len(te_in)} hist_w={HIST_W} suf_k={CMD_SUFFIX_K} "
        f"step_keys={len(tables['step'])} hist_keys={len(tables['hist'])}")
    log(f"  exact (full map)     {acc_exact:.3f}")
    log(f"  prim_compose         {acc_pc:.3f}")
    log(f"  AR step (cmd,t)      {acc_step:.3f}  cover={cov_step:.3f}  tf={tf_step:.3f}")
    log(f"  AR hist (cmd,prevW)  {acc_hist:.3f}  cover={cov_hist:.3f}  tf={tf_hist:.3f}")
    log(f"  AR bag+hist          {acc_bag:.3f}  cover={cov_bag:.3f}  tf={tf_bag:.3f}")
    log(f"  AR suf+hist          {acc_suf:.3f}  cover={cov_suf:.3f}  tf={tf_suf:.3f}")
    log(f"  AR hist-only (no cmd){acc_ho:.3f}  cover={cov_ho:.3f}  tf={tf_ho:.3f}")

    return {
        "split": tag,
        "n_train": len(tr_in),
        "n_test": len(te_in),
        "hist_w": HIST_W,
        "cmd_suffix_k": CMD_SUFFIX_K,
        "exact": acc_exact,
        "prim_compose": acc_pc,
        "prim_cover": cov_pc,
        "ar_step": acc_step,
        "ar_step_cover": cov_step,
        "tf_step": tf_step,
        "ar_hist": acc_hist,
        "ar_hist_cover": cov_hist,
        "tf_hist": tf_hist,
        "ar_bag_hist": acc_bag,
        "ar_bag_hist_cover": cov_bag,
        "tf_bag_hist": tf_bag,
        "ar_suf_hist": acc_suf,
        "ar_suf_hist_cover": cov_suf,
        "tf_suf_hist": tf_suf,
        "ar_hist_only": acc_ho,
        "ar_hist_only_cover": cov_ho,
        "tf_hist_only": tf_ho,
        "n_step_keys": len(tables["step"]),
        "n_hist_keys": len(tables["hist"]),
        "n_bag_keys": len(tables["bag_hist"]),
        "n_suf_keys": len(tables["suf_hist"]),
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
    log("SCAN NEXT-ACTION TABLE SCOREBOARD (AR exact-match | teacher-forced)")
    log("=" * 78)
    log(f"{'split':14} {'exact':>7} {'prim':>7} {'step':>7} {'hist':>7} {'bag':>7} "
        f"{'suf':>7} {'hOnly':>7}  {'tfB':>6} {'tfSuf':>6}")
    for r in results:
        log(f"{r['split']:14} {r['exact']:7.3f} {r['prim_compose']:7.3f} "
            f"{r['ar_step']:7.3f} {r['ar_hist']:7.3f} {r['ar_bag_hist']:7.3f} "
            f"{r['ar_suf_hist']:7.3f} {r['ar_hist_only']:7.3f}  "
            f"{r['tf_bag_hist']:6.3f} {r['tf_suf_hist']:6.3f}")
    out = REPO / "data" / "scan_seq_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {out}")
    log("\nReading: full-cmd next-action tables → ~0 AR exact (unique commands). "
        "Shared keys (bag/suffix) can get high teacher-forced token acc but still ~0 "
        "full-sequence AR. prim_compose remains the compositional ceiling.")


if __name__ == "__main__":
    main()
