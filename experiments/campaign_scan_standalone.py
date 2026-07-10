"""SCAN standalone campaign (flat baselines) — compositional generalization stress.

No host LLM / soft SGD. WordCodec alphabet fit on train only. Reports exact-match
action-sequence accuracy on official SCAN splits:

  length       — train short, test long (productivity / length)
  addprim_jump — "jump" held out as a primitive at train (systematicity)
  simple       — i.i.d. control (unique commands; exact still ~0)

Baselines:
  exact     — full-command dictionary from train (always ~0: unique commands)
  lcp       — longest train command that is a contiguous subseq of test cmd
  prim_comp — prims + unary/binary composition induced from train (tiny SCAN grammar)
  gap       — prim_comp − exact  (learning room for Wyly admit / multi-block)

Multi-layer path (not yet learned admit): ``pil/wyly_block.py`` +
``docs/notes/scan_standalone.md``. Next: admit combinators under cover-marginal on
hard splits, then BlockStack B0→B1→B2 on SCAN.

Run:
  cd pil && .venv/bin/python experiments/build_scan.py
  cd pil && .venv/bin/python -u experiments/campaign_scan_standalone.py
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
from pil.alphabet import WordCodec  # noqa: E402

SPLITS = ["length", "addprim_jump", "simple"]


def log(m=""):
    print(m, flush=True)


def ensure_data():
    missing = [s for s in SPLITS if not (REPO / "data" / f"scan_{s}.pt").exists()]
    if missing:
        log(f"building SCAN datasets for {missing}...")
        subprocess.check_call(
            [sys.executable, str(REPO / "experiments" / "build_scan.py")], cwd=str(REPO))


def load_split(tag: str):
    blob = torch.load(REPO / "data" / f"scan_{tag}.pt", map_location="cpu", weights_only=False)
    codec = WordCodec.from_file(REPO / "data" / blob["alphabet"])
    return blob, codec


def ids_to_tuple(xs: list[int]) -> tuple[int, ...]:
    return tuple(xs)


def exact_map(train_in, train_out) -> dict[tuple[int, ...], tuple[int, ...]]:
    m: dict[tuple[int, ...], tuple[int, ...]] = {}
    for ci, ao in zip(train_in, train_out, strict=True):
        m[ids_to_tuple(ci)] = ids_to_tuple(ao)
    return m


def eval_exact(m, test_in, test_out) -> float:
    ok = 0
    for ci, ao in zip(test_in, test_out, strict=True):
        pred = m.get(ids_to_tuple(ci))
        if pred is not None and pred == ids_to_tuple(ao):
            ok += 1
    return ok / max(len(test_in), 1)


def longest_contiguous_match(m: dict[tuple[int, ...], tuple[int, ...]], cmd: list[int]):
    """Longest train command that appears as a contiguous sub-sequence of cmd."""
    best = None
    best_len = 0
    n = len(cmd)
    # prefer longer keys; scan by decreasing length among train keys is expensive —
    # try all subarrays of cmd and look up
    for i in range(n):
        for j in range(i + 1, n + 1):
            key = tuple(cmd[i:j])
            if key in m and (j - i) > best_len:
                best_len = j - i
                best = m[key]
    return best


def eval_lcp(m, test_in, test_out) -> float:
    ok = 0
    for ci, ao in zip(test_in, test_out, strict=True):
        pred = longest_contiguous_match(m, ci)
        if pred is not None and pred == ids_to_tuple(ao):
            ok += 1
    return ok / max(len(test_in), 1)


# --- tiny SCAN-like composition induced from train ---------------------------------

def decode_toks(codec: WordCodec, ids: list[int]) -> list[str]:
    return [codec.token_str(i) for i in ids]


def encode_toks(codec: WordCodec, toks: list[str]) -> list[int] | None:
    out = []
    for t in toks:
        if t not in codec.stoi:
            return None
        out.append(codec.stoi[t])
    return out


def mine_scan_grammar(train_in, train_out, codec: WordCodec):
    """Induce prims + verify unary/binary composition from train pairs.

    Grammar (classic SCAN fragment):
      C := U | C 'and' C | C 'after' C
      U := P | U 'twice' | U 'thrice' | U 'around' D | 'turn' D | ...
      P := walk|jump|look|run | turn left|turn right | P 'left'|P 'right'|P 'opposite' D
    We learn leaf prim maps from unigram/bigram train commands and implement the standard
    combinators so addprim/length probe *systematic* use of known leaves.
    """
    prims: dict[tuple[str, ...], list[str]] = {}
    for ci, ao in zip(train_in, train_out, strict=True):
        ct = tuple(decode_toks(codec, ci))
        at = decode_toks(codec, ao)
        if len(ct) <= 2:
            prims[ct] = at
    return prims


def _turn(d: str) -> list[str]:
    if d == "left":
        return ["I_TURN_LEFT"]
    if d == "right":
        return ["I_TURN_RIGHT"]
    return []


def expand(tokens: list[str], prims: dict[tuple[str, ...], list[str]]) -> list[str] | None:
    """Recursive expansion of a SCAN-like command; None if stuck.

    Combinators bind tighter than and/after; we split and/after at the top level first
    (leftmost), then peel unary suffixes, then direction-modified primitives.
    """
    if not tokens:
        return []
    # top-level binary (leftmost and/after)
    depth = 0  # unused — flat language; just leftmost
    _ = depth
    for i, t in enumerate(tokens):
        if t == "and":
            left = expand(tokens[:i], prims)
            right = expand(tokens[i + 1 :], prims)
            if left is None or right is None:
                return None
            return left + right
    for i, t in enumerate(tokens):
        if t == "after":
            left = expand(tokens[:i], prims)
            right = expand(tokens[i + 1 :], prims)
            if left is None or right is None:
                return None
            return right + left
    # unary suffixes (twice/thrice bind after direction phrases)
    if len(tokens) >= 2 and tokens[-1] == "twice":
        base = expand(tokens[:-1], prims)
        return None if base is None else base + base
    if len(tokens) >= 2 and tokens[-1] == "thrice":
        base = expand(tokens[:-1], prims)
        return None if base is None else base + base + base
    # around / opposite D
    if len(tokens) >= 3 and tokens[-2] == "around" and tokens[-1] in ("left", "right"):
        body = expand(tokens[:-2], prims)
        if body is None:
            return None
        td = _turn(tokens[-1])
        unit = td + body
        return unit * 4
    if len(tokens) >= 3 and tokens[-2] == "opposite" and tokens[-1] in ("left", "right"):
        # "turn opposite left" = two turns (no body action)
        if tokens[0] == "turn" and len(tokens) == 3:
            td = _turn(tokens[-1])
            return td + td
        body = expand(tokens[:-2], prims)
        if body is None:
            return None
        td = _turn(tokens[-1])
        return td + td + body
    # "turn left" / "turn right"
    if len(tokens) == 2 and tokens[0] == "turn" and tokens[1] in ("left", "right"):
        key = tuple(tokens)
        if key in prims:
            return list(prims[key])
        return _turn(tokens[1])
    # direction after a primitive phrase: "walk left" / "jump right"
    if len(tokens) >= 2 and tokens[-1] in ("left", "right") and tokens[-2] not in (
            "around", "opposite", "turn"):
        body = expand(tokens[:-1], prims)
        if body is None:
            return None
        return _turn(tokens[-1]) + body
    # leaves (including mined prims)
    key = tuple(tokens)
    if key in prims:
        return list(prims[key])
    if len(tokens) == 1 and (tokens[0],) in prims:
        return list(prims[(tokens[0],)])
    # bare primitive actions if present as single-token prims under alternate naming
    if len(tokens) == 1:
        for k, v in prims.items():
            if k == (tokens[0],):
                return list(v)
    return None


def eval_prim_comp(prims, test_in, test_out, codec) -> tuple[float, float]:
    """Returns (exact_match, coverage) where coverage = fraction that parse."""
    ok = cov = 0
    for ci, ao in zip(test_in, test_out, strict=True):
        ct = decode_toks(codec, ci)
        pred = expand(ct, prims)
        if pred is None:
            continue
        cov += 1
        gold = decode_toks(codec, ao)
        if pred == gold:
            ok += 1
    n = max(len(test_in), 1)
    return ok / n, cov / n


def run_split(tag: str) -> dict:
    blob, codec = load_split(tag)
    tr_in, tr_out = blob["train_in"], blob["train_out"]
    te_in, te_out = blob["test_in"], blob["test_out"]
    m = exact_map(tr_in, tr_out)
    acc_exact = eval_exact(m, te_in, te_out)
    # lcp is O(n^2 * test); subsample test if huge
    te_in_s, te_out_s = te_in, te_out
    if len(te_in) > 1500:
        idx = list(range(0, len(te_in), max(1, len(te_in) // 1500)))[:1500]
        te_in_s = [te_in[i] for i in idx]
        te_out_s = [te_out[i] for i in idx]
    acc_lcp = eval_lcp(m, te_in_s, te_out_s)
    prims = mine_scan_grammar(tr_in, tr_out, codec)
    acc_pc, cov_pc = eval_prim_comp(prims, te_in, te_out, codec)
    gap = acc_pc - acc_exact
    log(f"\n=== SCAN/{tag} ===")
    log(f"  train={len(tr_in)} test={len(te_in)} vocab={codec.vocab_size} prims={len(prims)}")
    log(f"  exact_lookup     {acc_exact:.3f}")
    log(f"  longest_subcmd   {acc_lcp:.3f}  (on {len(te_in_s)} test)")
    log(f"  prim_compose     {acc_pc:.3f}  (parse cover {cov_pc:.3f})")
    log(f"  learning_gap     {gap:+.3f}  (prim_compose − exact; target for Wyly admit)")
    # Stub: learned flat admit not wired yet — keep a slot in the scoreboard for the next PR.
    learned_admit = None
    return {
        "split": tag,
        "n_train": len(tr_in),
        "n_test": len(te_in),
        "vocab": codec.vocab_size,
        "n_prims": len(prims),
        "exact": acc_exact,
        "lcp": acc_lcp,
        "prim_compose": acc_pc,
        "prim_cover": cov_pc,
        "learning_gap": gap,
        "learned_admit": learned_admit,  # filled when cover-marginal admit is wired on SCAN
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
    log("\n" + "=" * 60)
    log("SCAN STANDALONE SCOREBOARD (exact-match action sequences)")
    log("=" * 60)
    log(f"{'split':14} {'exact':>8} {'lcp':>8} {'prim_comp':>10} {'gap':>8} "
        f"{'learned':>8} {'n_test':>8}")
    for r in results:
        learned = r.get("learned_admit")
        learned_s = f"{learned:.3f}" if learned is not None else "  n/a"
        log(f"{r['split']:14} {r['exact']:8.3f} {r['lcp']:8.3f} {r['prim_compose']:10.3f} "
            f"{r['learning_gap']:+8.3f} {learned_s:>8} {r['n_test']:8d}")
    out = REPO / "data" / "scan_standalone_scoreboard.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")
    log("\nNote: SCAN commands are unique — exact lookup is ~0 even on simple (no repeated "
        "full commands). prim_compose uses train-mined prims + SCAN combinators; length/jump "
        "splits probe systematic use. learned_admit is a stub until Wyly cover-marginal admit "
        "is wired (see pil/wyly_block.py, docs/notes/scan_standalone.md).")


if __name__ == "__main__":
    main()
