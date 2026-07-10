"""CFQ standalone campaign — compositional Freebase Q→SPARQL yardstick.

No host LLM / soft SGD. WordCodec fit on train only. Reports exact-match SPARQL
accuracy on official CFQ splits (MCD1–3 + random):

  exact       — full question string → SPARQL (always ~0: unique questions)
  bag_query   — frozenset(question tokens) → majority SPARQL on train
  path_f1     — set-F1 of ns: predicates predicted from question-word co-occurrence
                (soft relational structure; not exact SPARQL)
  token_f1    — token multiset F1 of bag_query prediction vs gold SPARQL
  gap         — room for multi-layer / relational admit (1 − best exact-ish)

MCD splits maximize compound divergence (hard composition). Random is the
i.i.d. control — still ~0 exact because questions are unique strings.

Run:
  cd pil && .venv/bin/python experiments/build_cfq.py
  cd pil && .venv/bin/python -u experiments/campaign_cfq_standalone.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.alphabet import WordCodec  # noqa: E402

SPLITS = ["mcd1", "mcd2", "mcd3", "random"]
# Freebase predicate atoms (ns:domain.type.property); ignore vars/M* for structure F1
_NS_RE = re.compile(r"ns:[\w.]+")


def log(m=""):
    print(m, flush=True)


def ensure_data():
    missing = [s for s in SPLITS if not (REPO / "data" / f"cfq_{s}.pt").exists()]
    if missing:
        log(f"building CFQ datasets for {missing}...")
        env = os.environ.copy()
        # map random -> random_split for builder
        cfgs = []
        for s in missing:
            cfgs.append("random_split" if s == "random" else s)
        env["CFQ_CONFIGS"] = ",".join(cfgs)
        subprocess.check_call(
            [sys.executable, str(REPO / "experiments" / "build_cfq.py")],
            cwd=str(REPO), env=env)


def load_split(tag: str):
    blob = torch.load(REPO / "data" / f"cfq_{tag}.pt", map_location="cpu", weights_only=False)
    codec = WordCodec.from_file(REPO / "data" / blob["alphabet"])
    return blob, codec


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def multiset_f1(pred: Counter, gold: Counter) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = sum((pred & gold).values())
    if overlap == 0:
        return 0.0
    p = overlap / sum(pred.values())
    r = overlap / sum(gold.values())
    return 2 * p * r / (p + r)


def set_f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    p = inter / len(pred)
    r = inter / len(gold)
    return 2 * p * r / (p + r)


def sparql_ns(s: str) -> set[str]:
    return set(_NS_RE.findall(s))


def q_tokens(s: str) -> frozenset[str]:
    return frozenset(WordCodec.tokenize(s.lower()))


def fit_bag_query(train_q, train_sparql) -> dict[frozenset[str], str]:
    """Majority SPARQL string per question-token bag."""
    buckets: dict[frozenset[str], Counter] = defaultdict(Counter)
    for q, a in zip(train_q, train_sparql, strict=True):
        buckets[q_tokens(q)][norm_ws(a)] += 1
    return {k: ctr.most_common(1)[0][0] for k, ctr in buckets.items()}


def fit_word_to_ns(train_q, train_sparql) -> dict[str, Counter]:
    """For each content question word, count ns: predicates co-occurring in that pair."""
    word_ns: dict[str, Counter] = defaultdict(Counter)
    for q, a in zip(train_q, train_sparql, strict=True):
        paths = sparql_ns(a)
        if not paths:
            continue
        for w in q_tokens(q):
            if len(w) < 3:
                continue
            for p in paths:
                word_ns[w][p] += 1
    return word_ns


def predict_ns_set(q: str, word_ns: dict[str, Counter], per_word: int = 3, top_k: int = 10) -> set[str]:
    """Union of top ns: predicates associated with content words (set structure)."""
    scored: Counter = Counter()
    for w in q_tokens(q):
        if w not in word_ns:
            continue
        for path, c in word_ns[w].most_common(per_word):
            scored[path] += c
    return {p for p, _ in scored.most_common(top_k)}


def run_split(tag: str) -> dict:
    blob, codec = load_split(tag)
    tr_q, tr_a = blob["train_q"], blob["train_sparql"]
    te_q, te_a = blob["test_q"], blob["test_sparql"]

    exact_map = {q: norm_ws(a) for q, a in zip(tr_q, tr_a, strict=True)}
    bag_map = fit_bag_query(tr_q, tr_a)
    word_ns = fit_word_to_ns(tr_q, tr_a)

    n = max(len(te_q), 1)
    exact_ok = bag_ok = 0
    tok_f1_sum = path_f1_sum = 0.0
    bag_cover = 0

    for q, gold in zip(te_q, te_a, strict=True):
        g = norm_ws(gold)
        # exact
        if exact_map.get(q) == g:
            exact_ok += 1
        # bag → full SPARQL
        pred = bag_map.get(q_tokens(q))
        if pred is not None:
            bag_cover += 1
            if pred == g:
                bag_ok += 1
            tok_f1_sum += multiset_f1(
                Counter(WordCodec.tokenize(pred)),
                Counter(WordCodec.tokenize(g)),
            )
        else:
            tok_f1_sum += 0.0
        # soft structure: predicted ns: set vs gold ns: set
        path_f1_sum += set_f1(predict_ns_set(q, word_ns), sparql_ns(gold))

    acc_exact = exact_ok / n
    acc_bag = bag_ok / n
    mean_tok_f1 = tok_f1_sum / n
    mean_path_f1 = path_f1_sum / n
    # "best hard" exact-ish is bag (still near 0 on MCD)
    gap = 1.0 - acc_bag  # room until perfect exact SPARQL

    log(f"\n=== CFQ/{tag} ===")
    log(f"  train={len(tr_q)} test={len(te_q)} vocab={codec.vocab_size}")
    log(f"  exact              {acc_exact:.4f}")
    log(f"  bag→SPARQL exact   {acc_bag:.4f}  cover={bag_cover / n:.3f}")
    log(f"  bag token-F1       {mean_tok_f1:.3f}")
    log(f"  word→path F1       {mean_path_f1:.3f}  (soft structure)")
    log(f"  learning_gap       {gap:+.3f}  (1 − bag exact; multi-layer target)")

    return {
        "split": tag,
        "n_train": len(tr_q),
        "n_test": len(te_q),
        "vocab": codec.vocab_size,
        "exact": acc_exact,
        "bag_exact": acc_bag,
        "bag_cover": bag_cover / n,
        "bag_token_f1": mean_tok_f1,
        "path_f1": mean_path_f1,
        "learning_gap": gap,
        "origin": "standalone",
        "alphabet": "word",
    }


def main():
    ensure_data()
    want = [s.strip() for s in os.environ.get("CFQ_SPLITS", ",".join(SPLITS)).split(",")
            if s.strip()]
    results = []
    for tag in want:
        if not (REPO / "data" / f"cfq_{tag}.pt").exists():
            log(f"missing cfq_{tag}.pt — skip")
            continue
        results.append(run_split(tag))
    log("\n" + "=" * 72)
    log("CFQ STANDALONE SCOREBOARD")
    log("=" * 72)
    log(f"{'split':10} {'exact':>8} {'bagEx':>8} {'tokF1':>8} {'pathF1':>8} "
        f"{'gap':>8} {'n_test':>8}")
    for r in results:
        log(f"{r['split']:10} {r['exact']:8.4f} {r['bag_exact']:8.4f} "
            f"{r['bag_token_f1']:8.3f} {r['path_f1']:8.3f} "
            f"{r['learning_gap']:+8.3f} {r['n_test']:8d}")
    out = REPO / "data" / "cfq_standalone_scoreboard.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")
    log("\nNote: CFQ questions are unique — exact lookup is ~0 on random and MCD. "
        "MCD maximizes compound divergence (relational composition stress). "
        "path_f1 is a soft structure signal; hard exact SPARQL is the multi-layer target.")


if __name__ == "__main__":
    main()
