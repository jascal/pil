"""Distillability vs scale: distill each pythia rung into a PIC program, same budget.

For every pythia size, the same 120k shared wikitext windows are labeled by that model
(argmax + top-8 logits); a PIC rule program is trained by *distributional* distillation
(soft CE against the teacher's top-K) under an identical config, and scored on held-out
windows. The curve of interest: **behavioral distillability vs model scale** -- does a
fixed-budget rule program capture less of a bigger model's decision function (the
composed/forge-tax fraction growing), and how does that track the tau*/r_eff scaling
measurements in pil/results?

Per rung: agreement with the teacher's argmax, teacher-top5 hit rate, the teacher's own
bigram floor (mode of argmax given last token -- the retrievable-by-lookup fraction),
lift over that floor, certified fraction, program size.

Usage:
  python experiments/pythia_ladder_distill.py --labels-dir <dir> \
      [--sizes 14m,70m,160m,410m,1b,1.4b,2.8b] [--out results/pythia_ladder_distill.jsonl]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pil.rule_learner import (
    PICRuleLearner,
    RuleLearnerConfig,
    init_direct_lookup_from_counts,
)
from pil.rules import RuleProgram, RuleProgramConfig, certified_fraction, program_summary

W = 8


def bench_rung(labels: Path, size: str, n_train: int, top_v: int, seed: int = 0) -> dict:
    d = np.load(labels)
    X = torch.from_numpy(d["X"])
    y = torch.from_numpy(d["y"])
    tk_ids = torch.from_numpy(d["topk_ids"])
    tk_log = torch.from_numpy(d["topk_logits"])
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]

    vals, counts = ytr.unique(return_counts=True)
    cands = vals[counts.argsort(descending=True)[:top_v]].sort().values
    tr_in = torch.isin(ytr, cands)
    coverage = float(torch.isin(yte, cands).float().mean())

    # teacher's bigram floor on the held-out windows
    mode: dict[int, int] = {}
    last, ytr_l = Xtr[:, -1], ytr
    for c in last.unique().tolist():
        mode[c] = int(ytr_l[last == c].mode().values)
    gmode = int(ytr.mode().values)
    big = torch.tensor([mode.get(int(c), gmode) for c in Xte[:, -1]])
    bigram = float((big == yte).float().mean())

    cfg = RuleProgramConfig(
        vocab_size=50304, window=W, frame_dim=64, candidates=cands.tolist(),
        lookup_offsets=(W - 3, W - 2), direct_lookup_offsets=(W - 1,), seed=seed,
    )
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=5, init_rules=768, births_per_phase=384, epochs_per_phase=3,
        max_rules=1536, batch_size=512, probe_size=1024,
        ambiguity_split_threshold=0.4, max_splits_per_phase=32,
        hard_target_weight=0.5, death_on_val=True, seed=seed, verbose=False,
    )
    learner = PICRuleLearner(prog, lc)
    init_direct_lookup_from_counts(prog, Xtr[tr_in], prog.target_index(ytr[tr_in]))
    for emb in prog.direct_lookup.values():
        emb.weight.requires_grad_(False)   # the count estimate is already right; SGD hurt it
    # reference: the untrained count-table program (gram mining alone, no learning)
    with torch.no_grad():
        L0 = learner._logits_in_chunks(Xte, hard=True)
        count_agree = float((prog.candidate_ids[L0.argmax(-1)] == yte).float().mean())
    soft = learner.make_soft_targets(tk_ids[:n_train][tr_in], tk_log[:n_train][tr_in])
    t0 = time.time()
    learner.fit(Xtr[tr_in], ytr[tr_in], soft=soft)
    t_fit = time.time() - t0

    with torch.no_grad():
        L = learner._logits_in_chunks(Xte, hard=True)
        pred = prog.candidate_ids[L.argmax(-1)]
    agree = float((pred == yte).float().mean())
    top5 = float(
        (pred.unsqueeze(1) == tk_ids[n_train:, :5]).any(dim=1).float().mean()
    )
    return {
        "size": size,
        "n_train": int(tr_in.sum()), "n_heldout": len(yte),
        "cand_coverage": round(coverage, 4),
        "teacher_bigram_floor": round(bigram, 4),
        "count_table_agreement": round(count_agree, 4),
        "pil_agreement": round(agree, 4),
        "lift_over_bigram": round(agree - bigram, 4),
        "lift_over_count_table": round(agree - count_agree, 4),
        "pil_in_teacher_top5": round(top5, 4),
        "certified_frac_delta_0.25": round(certified_fraction(prog, Xte[:4000], 0.25), 3),
        "fit_s": round(t_fit),
        "program": program_summary(prog),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--sizes", default="14m,70m,160m,410m,1b,1.4b,2.8b")
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--top-v", type=int, default=1024)
    ap.add_argument("--out", default="results/pythia_ladder_distill.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes.split(","):
        labels = Path(args.labels_dir) / f"pythia-{size}_labels.npz"
        if not labels.exists():
            print(f"=== {size}: labels missing, skipping", flush=True)
            continue
        print(f"=== pythia-{size} ===", flush=True)
        row = bench_rung(labels, size, args.n_train, args.top_v)
        for k, v in row.items():
            print(f"  {k}: {v}", flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
    print(f"appended rows to {out}")


if __name__ == "__main__":
    main()
