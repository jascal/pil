"""CFQ residual atoms — API reuse + honest ceiling diagnosis (standalone).

Uses the same ``ResidualFamily.propose`` / ``admit`` (naive, not CELF) as SCAN
and listops. This measures **admit-loop portability**, not induced join structure:

  - Base: single-``ns:`` co-occurrence → ``(word, path) → [path]``
  - Residual: ``RelationAtomTemplate`` is a thin pass-through over multi-ns
    word×path votes computed by the campaign (no structural pattern detection)
  - Predict: bag-of-words set-union of atoms — **not** compositional join

Scoring is predicate set-F1 (structure bag). Exact SPARQL = 0 without a generator.

Baselines (required anchors):
  - frequency prior: always predict top-K train paths (question-blind)
  - base / hardcode residual / certified admit

Honest findings (mcd1): freq prior can beat base; admit often **underperforms**
hardcode (val misaligned); set-F1 plateaus ≪ 1 — symbolic word→path is at a ceiling.

Holdouts:
  - weakened_base: drop a path from base+residual (other atoms may still help)
  - recovery: drop path from base only; residual votes still carry it (true recovery test)
  - deep queries (≥6 ns)

Run:
  cd pil && .venv/bin/python -u experiments/campaign_cfq_residual.py
Env: CFQ_SPLITS=mcd1  CFQ_MAX_VAL=2000  CFQ_FREQ_K=20
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.residual_template import (  # noqa: E402
    CFQ_TEMPLATES,
    MapDict,
    ResidualFamily,
    cfq_domain_atoms,
)

SPLITS = ["mcd1", "mcd2", "mcd3"]
_NS_RE = re.compile(r"ns:[\w.]+")
# light stopword list for question tokens (domain content filter, not a template)
_STOP = frozenset({
    "a", "an", "the", "of", "to", "and", "or", "was", "is", "did", "do", "does",
    "were", "are", "by", "with", "from", "in", "on", "for", "s", "that", "which",
    "who", "what", "how", "many", "whose", "'s", ",", "?", ".",
})


def log(m=""):
    print(m, flush=True)


def ensure_data(tags: list[str]):
    missing = [s for s in tags if not (REPO / "data" / f"cfq_{s}.pt").exists()]
    if not missing:
        return
    log(f"building CFQ for {missing}...")
    env = os.environ.copy()
    env["CFQ_CONFIGS"] = ",".join(missing)
    subprocess.check_call(
        [sys.executable, str(REPO / "experiments" / "build_cfq.py")],
        cwd=str(REPO), env=env,
    )


def load_split(tag: str):
    return torch.load(REPO / "data" / f"cfq_{tag}.pt", map_location="cpu", weights_only=False)


def sparql_ns(s: str) -> set[str]:
    return set(_NS_RE.findall(s))


def content_words(q: str) -> list[str]:
    # simple tokenize on non-alnum
    toks = re.findall(r"[A-Za-z0-9_]+", q.lower())
    return [t for t in toks if t not in _STOP and len(t) > 1]


def set_f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    p, r = inter / len(pred), inter / len(gold)
    return 2 * p * r / (p + r)


def mine_base_atoms(
    questions: list[str],
    sparqls: list[str],
    *,
    max_ns: int = 1,
    min_support: int = 2,
) -> MapDict:
    """Single-ns (or ≤max_ns) queries → (word, path) atoms by majority co-occurrence."""
    votes: dict[tuple[str, str], int] = Counter()
    for q, a in zip(questions, sparqls, strict=True):
        paths = sparql_ns(a)
        if not paths or len(paths) > max_ns:
            continue
        words = content_words(q)
        for w in words:
            for p in paths:
                votes[(w, p)] += 1
    # keep pairs with support; one path per (word,path) key
    maps: MapDict = {}
    for (w, p), c in votes.items():
        if c >= min_support:
            maps[(w, p)] = [p]
    return maps


def mine_multi_votes(
    questions: list[str],
    sparqls: list[str],
    *,
    min_ns: int = 2,
) -> dict[tuple[str, str], int]:
    """Multi-ns co-occurrence votes for RelationAtomTemplate residuals."""
    votes: dict[tuple[str, str], int] = Counter()
    for q, a in zip(questions, sparqls, strict=True):
        paths = sparql_ns(a)
        if len(paths) < min_ns:
            continue
        for w in content_words(q):
            for p in paths:
                votes[(w, p)] += 1
    return dict(votes)


def predict_paths(q: str, maps: MapDict) -> set[str]:
    """Join = set-union of relation atoms for content words (complementary)."""
    pred: set[str] = set()
    words = set(content_words(q))
    for key, tgt in maps.items():
        if len(key) >= 1 and key[0] in words:
            pred.update(tgt)
    return pred


def mean_set_f1(
    pairs: list[tuple[str, str]],
    maps: MapDict,
) -> float:
    if not pairs:
        return 0.0
    total = 0.0
    for q, a in pairs:
        total += set_f1(predict_paths(q, maps), sparql_ns(a))
    return total / len(pairs)


def exact_sparql_rate(pairs: list[tuple[str, str]], maps: MapDict) -> float:
    """Exact full-string SPARQL is out of scope for atom maps — always ~0; reported honestly."""
    _ = maps
    return 0.0  # atom residual does not emit SPARQL strings


def holdout_path_maps(maps: MapDict, path: str) -> MapDict:
    return {k: v for k, v in maps.items() if path not in v and (len(k) < 2 or k[1] != path)}


def frequency_prior_paths(sparqls: list[str], k: int = 20) -> set[str]:
    """Question-blind prior: top-k most common ns: paths on train."""
    counts: Counter[str] = Counter()
    for a in sparqls:
        counts.update(sparql_ns(a))
    return {p for p, _ in counts.most_common(k)}


def mean_set_f1_constant(pairs: list[tuple[str, str]], const: set[str]) -> float:
    if not pairs:
        return 0.0
    return sum(set_f1(const, sparql_ns(a)) for _, a in pairs) / len(pairs)


def run_split(tag: str) -> dict:
    blob = load_split(tag)
    tr_q, tr_a = blob["train_q"], blob["train_sparql"]
    te_q, te_a = blob["test_q"], blob["test_sparql"]

    # fit / val from train
    idx = list(range(len(tr_q)))
    random.Random(0).shuffle(idx)
    nval = min(int(os.environ.get("CFQ_MAX_VAL", "2000")), max(1, len(idx) // 10))
    val_i, fit_i = idx[:nval], idx[nval:]
    fit_q = [tr_q[i] for i in fit_i]
    fit_a = [tr_a[i] for i in fit_i]
    val_pairs = [(tr_q[i], tr_a[i]) for i in val_i]
    # subsample test for speed if huge
    te_pairs = list(zip(te_q, te_a, strict=True))
    if len(te_pairs) > 3000:
        te_pairs = [te_pairs[i] for i in range(0, len(te_pairs), max(1, len(te_pairs) // 3000))][:3000]

    base_maps = mine_base_atoms(fit_q, fit_a, max_ns=1, min_support=2)
    multi_votes = mine_multi_votes(fit_q, fit_a, min_ns=2)
    # strip multi votes already in base; keep top-K by support for admit cost
    multi_votes = {k: v for k, v in multi_votes.items() if k not in base_maps}
    top_k = int(os.environ.get("CFQ_RES_TOPK", "80"))
    if len(multi_votes) > top_k:
        multi_votes = dict(sorted(multi_votes.items(), key=lambda kv: -kv[1])[:top_k])

    domain = cfq_domain_atoms(multi_word_path_votes=multi_votes, atom_min_support=3)
    fam = ResidualFamily(domain, templates=CFQ_TEMPLATES)
    cands = fam.propose(base_maps)

    # subsample val for admit scoring (full val used only for final report metrics)
    val_admit = val_pairs
    max_va = int(os.environ.get("CFQ_ADMIT_VAL", "400"))
    if len(val_admit) > max_va:
        val_admit = val_admit[:max_va]

    def val_score(maps: MapDict) -> float:
        return mean_set_f1(val_admit, maps)

    # naive admit only (joins complementary)
    maps_adm, admit_log = fam.admit(
        base_maps, val_score, thresh=1e-4, max_rules=16, celf=False,
    )
    maps_hard = {**base_maps, **fam.propose_map(base_maps)}

    f1_base = mean_set_f1(te_pairs, base_maps)
    f1_hard = mean_set_f1(te_pairs, maps_hard)
    f1_adm = mean_set_f1(te_pairs, maps_adm)
    exact = exact_sparql_rate(te_pairs, maps_adm)
    freq_k = int(os.environ.get("CFQ_FREQ_K", "20"))
    freq_paths = frequency_prior_paths(fit_a, k=freq_k)
    f1_freq = mean_set_f1_constant(te_pairs, freq_paths)
    admit_vs_hard = f1_adm - f1_hard
    admit_vs_freq = f1_adm - f1_freq
    admit_vs_base = f1_adm - f1_base

    path_counts = Counter()
    for v in base_maps.values():
        path_counts.update(v)
    hold_path = path_counts.most_common(1)[0][0] if path_counts else ""

    # --- weakened_base holdout: path removed from base AND residual votes ---
    # Other atoms may still help; this is NOT "recover the dropped relation".
    base_weak = holdout_path_maps(base_maps, hold_path) if hold_path else dict(base_maps)
    multi_weak = {k: v for k, v in multi_votes.items() if k[1] != hold_path}
    if len(multi_weak) > top_k:
        multi_weak = dict(sorted(multi_weak.items(), key=lambda kv: -kv[1])[:top_k])
    fam_weak = ResidualFamily(
        cfq_domain_atoms(multi_word_path_votes=multi_weak, atom_min_support=3),
        templates=CFQ_TEMPLATES,
    )
    f1_weak_base = mean_set_f1(te_pairs, base_weak)
    maps_weak_adm, _ = fam_weak.admit(
        base_weak, val_score, thresh=1e-4, max_rules=16, celf=False)
    f1_weak_adm = mean_set_f1(te_pairs, maps_weak_adm)

    # --- recovery holdout: path removed from base only; multi votes still carry it ---
    multi_rec = dict(multi_votes)  # may still propose held path
    if len(multi_rec) > top_k:
        multi_rec = dict(sorted(multi_rec.items(), key=lambda kv: -kv[1])[:top_k])
    fam_rec = ResidualFamily(
        cfq_domain_atoms(multi_word_path_votes=multi_rec, atom_min_support=3),
        templates=CFQ_TEMPLATES,
    )
    maps_rec_hard = {**base_weak, **fam_rec.propose_map(base_weak)}
    f1_rec_hard = mean_set_f1(te_pairs, maps_rec_hard)
    # did hardcode residual re-introduce the held path into any atom?
    recovered_in_hard = any(
        hold_path in v or (len(k) > 1 and k[1] == hold_path)
        for k, v in maps_rec_hard.items()
    )
    maps_rec_adm, _ = fam_rec.admit(
        base_weak, val_score, thresh=1e-4, max_rules=16, celf=False)
    f1_rec_adm = mean_set_f1(te_pairs, maps_rec_adm)
    recovered_in_adm = any(
        hold_path in v or (len(k) > 1 and k[1] == hold_path)
        for k, v in maps_rec_adm.items()
    )

    # --- depth: deep multi-ns queries (join stress / ceiling) ---
    deep = [(q, a) for q, a in te_pairs if len(sparql_ns(a)) >= 6]
    f1_deep_base = mean_set_f1(deep, base_maps) if deep else 0.0
    f1_deep_adm = mean_set_f1(deep, maps_adm) if deep else 0.0
    f1_deep_hard = mean_set_f1(deep, maps_hard) if deep else 0.0
    f1_deep_freq = mean_set_f1_constant(deep, freq_paths) if deep else 0.0

    n_adm = sum(1 for e in admit_log if e.get("event") == "admit")
    diag = fam.diagnostics(
        base_maps,
        admitted_src=[c.src for c in cands if c.src in maps_adm and c.src not in base_maps],
    )

    log(f"\n=== CFQ/{tag} residual atoms (API portability / ceiling diagnosis) ===")
    log(f"  fit={len(fit_q)} val={len(val_pairs)} test={len(te_pairs)} "
        f"base_atoms={len(base_maps)} residual_cands={len(cands)}")
    log(f"  set-F1 freq prior@{freq_k} {f1_freq:.3f}  (question-blind)")
    log(f"  set-F1 base          {f1_base:.3f}  (vs freq {f1_base - f1_freq:+.3f})")
    log(f"  set-F1 hardcode res  {f1_hard:.3f}")
    log(f"  set-F1 certified adm {f1_adm:.3f}  n_adm={n_adm}  "
        f"vs hard {admit_vs_hard:+.3f}  vs freq {admit_vs_freq:+.3f}")
    log(f"  exact SPARQL         {exact:.3f}")
    log(f"  weakened_base (path {hold_path!r} removed from base+res): "
        f"{f1_weak_base:.3f}→{f1_weak_adm:.3f}")
    log(f"  recovery (path only out of base; res may re-emit): "
        f"hard={f1_rec_hard:.3f} adm={f1_rec_adm:.3f} "
        f"recovered_hard={recovered_in_hard} recovered_adm={recovered_in_adm}")
    log(f"  deep (≥6 ns) n={len(deep)}: freq={f1_deep_freq:.3f} base={f1_deep_base:.3f} "
        f"hard={f1_deep_hard:.3f} adm={f1_deep_adm:.3f}")
    log("  ceiling note: bag set-F1 plateaus ≪1; admit may lose to hardcode "
        "(val/test misalign)")

    return {
        "split": tag,
        "n_fit": len(fit_q),
        "n_val": len(val_pairs),
        "n_test": len(te_pairs),
        "n_base_atoms": len(base_maps),
        "n_residual_cands": len(cands),
        "n_admitted": n_adm,
        "freq_k": freq_k,
        "set_f1_freq_prior": f1_freq,
        "set_f1_base": f1_base,
        "set_f1_hardcode": f1_hard,
        "set_f1_admit": f1_adm,
        "admit_minus_hardcode": admit_vs_hard,
        "admit_minus_freq": admit_vs_freq,
        "admit_minus_base": admit_vs_base,
        "exact_sparql": exact,
        "holdout_path": hold_path,
        "weakened_base_f1": f1_weak_base,
        "weakened_admit_f1": f1_weak_adm,
        "recovery_hard_f1": f1_rec_hard,
        "recovery_admit_f1": f1_rec_adm,
        "recovery_path_in_hardcode": recovered_in_hard,
        "recovery_path_in_admit": recovered_in_adm,
        "deep_n": len(deep),
        "deep_freq_f1": f1_deep_freq,
        "deep_base_f1": f1_deep_base,
        "deep_hard_f1": f1_deep_hard,
        "deep_admit_f1": f1_deep_adm,
        "diagnostics": diag,
        "admit_log": admit_log[:30],
        "sample_admitted": [e for e in admit_log if e.get("event") == "admit"][:10],
        "origin": "standalone",
        "admit": "naive",
        "claim_scope": (
            "ResidualFamily API reuse on CFQ; RelationAtomTemplate is a vote pass-through, "
            "not induced join structure. Predict is bag-union set-F1."
        ),
        "negative_result": (
            "Symbolic word→path set-F1 is near a frequency-prior ceiling (~0.25–0.28); "
            "certified admit can underperform hardcode; exact SPARQL remains 0. "
            "Pivot: residual_as_schema / compositional decode, not more atoms."
        ),
    }


def main():
    want = [s.strip() for s in os.environ.get("CFQ_SPLITS", "mcd1").split(",") if s.strip()]
    ensure_data(want)
    results = []
    for tag in want:
        if not (REPO / "data" / f"cfq_{tag}.pt").exists():
            log(f"missing cfq_{tag}.pt — skip")
            continue
        results.append(run_split(tag))

    log("\n" + "=" * 78)
    log("CFQ RESIDUAL SCOREBOARD (set-F1 bag / exact SPARQL)")
    log("=" * 78)
    log(f"{'split':8} {'freq':>7} {'base':>7} {'hard':>7} {'admit':>7} "
        f"{'ad-hd':>7} {'exact':>7} {'recHd':>7} {'deepH':>7}")
    for r in results:
        log(f"{r['split']:8} {r['set_f1_freq_prior']:7.3f} {r['set_f1_base']:7.3f} "
            f"{r['set_f1_hardcode']:7.3f} {r['set_f1_admit']:7.3f} "
            f"{r['admit_minus_hardcode']:+7.3f} {r['exact_sparql']:7.3f} "
            f"{r['recovery_hard_f1']:7.3f} {r['deep_hard_f1']:7.3f}")
        log(f"  {r['negative_result']}")
    out_dir = REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cfq_residual_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {out}")
    log("\nReading: freq prior anchors the board (base may lose to it). "
        "admit−hardcode < 0 = certified selection costs F1. "
        "exact=0 + deep plateau = symbolic word→path ceiling → residual_as_schema.")


if __name__ == "__main__":
    main()
