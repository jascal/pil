"""IOI (indirect object identification) in real pythia tokens, with unseen-pair eval.

The canonical mechanistic-interpretability task: "When A and B went to the store, B gave
a drink to" -> A. The circuit answer is *relational* -- "the name that is NOT repeated" --
so the benchmark is designed to make memorization fail: **test name pairs never co-occur
in training**. A model that memorizes (pair -> answer) transfers nothing; a model that
learns the relational rule (equality literals: which earlier name matches the second
mention) generalizes.

Everything is real pythia-160m tokens (29 single-token names, single-token template words),
so the learned program is directly comparable to what rosetta's decompiler mines from
pythia itself (its `ioi_pm` rules).

Usage:  python experiments/ioi_bench.py [--out results/ioi_bench.jsonl]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from rule_learner_bench import mlp_acc, rule_param_count, train_mlp  # noqa: E402

from pil.datalog_export import verify_export
from pil.rule_learner import PICRuleLearner, RuleLearnerConfig
from pil.rules import RuleProgram, RuleProgramConfig, certified_fraction, program_summary
from pil.tokens import TokenSpace

ROSETTA_PKG = Path("/home/allans/code/rosetta/models/pythia160m")

NAMES = [
    "Mary", "John", "Tom", "Anna", "Peter", "Sarah", "James", "Laura", "David", "Emma",
    "Paul", "Alice", "Mark", "Julia", "Henry", "Grace", "Frank", "Helen", "Oscar", "Ruth",
    "Adam", "Clara", "Leo", "Diana", "Sam", "Max", "Ivy", "Ben", "Rose",
]
# "When A and B went to the store, B gave a drink to" -> A   (positions 1 and 3 hold A, B)
TEMPLATE = ["When", "<A>", " and", "<B>", " went", " to", " the", " store", ",",
            "<S>", " gave", " a", " drink", " to"]
A_POS, B_POS, S_POS = 1, 3, 9


def build_data(seed: int = 0, n: int = 8000):
    ts = TokenSpace.from_rosetta_package(ROSETTA_PKG)
    name_ids = torch.tensor([ts.encode(" " + n_)[0] for n_ in NAMES])
    fixed = {i: ts.encode(w)[0] for i, w in enumerate(TEMPLATE) if not w.startswith("<")}
    W, N_names = len(TEMPLATE), len(NAMES)

    g = torch.Generator().manual_seed(seed)
    # disjoint-pair split: an unordered name pair is train xor test
    pairs = [(a, b) for a in range(N_names) for b in range(N_names) if a < b]
    perm = torch.randperm(len(pairs), generator=g).tolist()
    cut = int(len(pairs) * 0.75)
    pool = {"train": [pairs[i] for i in perm[:cut]], "test": [pairs[i] for i in perm[cut:]]}

    def make(split: str, n_rows: int):
        X = torch.zeros(n_rows, W, dtype=torch.long)
        for i, tok_id in fixed.items():
            X[:, i] = tok_id
        y = torch.zeros(n_rows, dtype=torch.long)
        ps = pool[split]
        for r in range(n_rows):
            a, b = ps[int(torch.randint(0, len(ps), (1,), generator=g))]
            if bool(torch.rand(1, generator=g) < 0.5):
                a, b = b, a                      # both orders of the pair
            subj_is_b = bool(torch.rand(1, generator=g) < 0.5)
            subj, answer = (b, a) if subj_is_b else (a, b)
            X[r, A_POS], X[r, B_POS], X[r, S_POS] = name_ids[a], name_ids[b], name_ids[subj]
            y[r] = name_ids[answer]
        return X, y

    Xtr, ytr = make("train", n)
    Xte, yte = make("test", n // 4)
    return Xtr, ytr, Xte, yte, name_ids, W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Xtr, ytr, Xte, yte, name_ids, W = build_data()
    print(f"train {len(ytr)}, test (disjoint pairs) {len(yte)}, chance {1/len(NAMES):.3f}")

    cfg = RuleProgramConfig(
        vocab_size=50304, window=W, frame_dim=32, candidates=name_ids.tolist(),
        eq_atoms=True, seed=0,
    )
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=10, init_rules=192, births_per_phase=96, epochs_per_phase=20,
        max_rules=768, recency_gamma=1.0, eq_seed_p=0.6,
        ambiguity_split_threshold=0.7, max_splits_per_phase=32,
        # the internal val split shares train pairs, so it cannot see the disjoint-pair
        # shift -- keep structure search alive for the whole run
        solved_val_acc=2.0, seed=0, verbose=True,
    )
    learner = PICRuleLearner(prog, lc)
    t0 = time.time()
    learner.fit(Xtr, ytr)
    t_fit = time.time() - t0

    yte_i = prog.target_index(yte)
    hard = learner._acc(Xte, yte_i, hard=True)

    t0 = time.time()
    mlp = train_mlp(Xtr, prog.target_index(ytr), len(NAMES), 50304, epochs=120, seed=0)
    t_mlp = time.time() - t0

    row = {
        "task": "IOI (pythia tokens, disjoint name pairs)",
        "chance": round(1 / len(NAMES), 4),
        "rule_hard_acc": round(hard, 4),
        "mlp_acc": round(mlp_acc(mlp, Xte, yte_i), 4),
        "certified_frac_delta_0.25": round(certified_fraction(prog, Xte, 0.25), 3),
        "rules": prog.n_rules,
        "rule_params": rule_param_count(prog),
        "rule_s": round(t_fit), "mlp_s": round(t_mlp),
        "summary": program_summary(prog),
    }
    if shutil.which("souffle"):
        row["souffle"] = verify_export(prog, Xte[:300])
    for k, v in row.items():
        print(f"  {k}: {v}", flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(json.dumps(row) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
