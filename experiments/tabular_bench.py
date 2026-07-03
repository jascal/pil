"""Classic tabular ML benchmarks: the PIC rule learner vs gradient-boosted trees.

The honest incumbent for rule learners is not an MLP but GBT (HistGradientBoosting) --
baselines are computed on the *same fixed splits* by a companion script (sklearn venv) and
read from ``<data>/<name>.baselines.json``; raw features go to GBT/LogReg, while PIL and
the MLP consume the same quantile-binned atoms (8 bins/feature; categorical = category id).

Encoding: position ``o`` = feature ``o``; the token at ``o`` is a synthetic atom id
``ATOM_BASE + o*MAX_LEVELS + level`` (tabular rows aren't text, so the pythia binding of
the LM benchmarks doesn't apply -- the program semantics are identical). Classes are the
candidate "tokens". PIC-T3 threshold rules read naturally here ("at least θ of these
feature-level tests hold"), and the export is the same Soufflé program as everywhere else
(verified on the small datasets).

Usage:
  python experiments/tabular_bench.py --data <dir from make_tabular.py> \
      [--task breast_cancer|wine|digits|adult|all] [--out results/tabular_bench.txt]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from rule_learner_bench import mlp_acc, rule_param_count, train_mlp  # noqa: E402

from pil.datalog_export import verify_export
from pil.rule_learner import PICRuleLearner, RuleLearnerConfig
from pil.rules import RuleProgram, RuleProgramConfig, certified_fraction, program_summary

ATOM_BASE = 1000
CLASS_BASE = 100
MAX_LEVELS = 32
N_BINS = 16


def binned_tokens(
    Xtr: np.ndarray, Xte: np.ndarray, cat_mask: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantile-bin numeric features (train quantiles), pass categorical ids through."""
    F = Xtr.shape[1]
    tr = np.zeros_like(Xtr, dtype=np.int64)
    te = np.zeros_like(Xte, dtype=np.int64)
    for f in range(F):
        if cat_mask[f]:
            tr[:, f] = np.clip(Xtr[:, f].astype(np.int64), 0, MAX_LEVELS - 1)
            te[:, f] = np.clip(Xte[:, f].astype(np.int64), 0, MAX_LEVELS - 1)
        else:
            edges = np.unique(np.quantile(Xtr[:, f], np.linspace(0, 1, N_BINS + 1)[1:-1]))
            tr[:, f] = np.searchsorted(edges, Xtr[:, f])
            te[:, f] = np.searchsorted(edges, Xte[:, f])
    off = ATOM_BASE + np.arange(F, dtype=np.int64) * MAX_LEVELS
    return (
        torch.from_numpy(tr + off),
        torch.from_numpy(te + off),
        ATOM_BASE + F * MAX_LEVELS,
    )


CONFIGS: dict[str, dict] = {
    "breast_cancer": dict(n_phases=8, init_rules=128, births_per_phase=64,
                          epochs_per_phase=40, max_rules=512),
    "wine": dict(n_phases=8, init_rules=128, births_per_phase=64,
                 epochs_per_phase=40, max_rules=512),
    "digits": dict(n_phases=8, init_rules=256, births_per_phase=128,
                   epochs_per_phase=25, max_rules=1024, thresh_fraction=0.5),
    "adult": dict(n_phases=6, init_rules=256, births_per_phase=128,
                  epochs_per_phase=10, max_rules=1024, batch_size=512),
}
for _c in CONFIGS.values():
    _c["le_seed_p"] = 0.5        # ordinal literals: bin tokens are ordered within a feature
SOUFFLE_VERIFY = {"breast_cancer", "wine"}


def bench(name: str, data_dir: Path, seed: int = 0) -> dict:
    d = np.load(data_dir / f"{name}.npz")
    base = json.loads((data_dir / f"{name}.baselines.json").read_text())
    Xtr, Xte, vocab = binned_tokens(d["X_train"], d["X_test"], d["cat_mask"])
    n_classes = base["n_classes"]
    cands = [CLASS_BASE + c for c in range(n_classes)]
    ytr = torch.from_numpy(d["y_train"].astype(np.int64)) + CLASS_BASE
    yte = torch.from_numpy(d["y_test"].astype(np.int64)) + CLASS_BASE
    W = Xtr.shape[1]

    cfg = RuleProgramConfig(
        vocab_size=vocab, window=W, frame_dim=max(16, 4 * n_classes),
        candidates=cands, seed=seed,
    )
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(recency_gamma=1.0, seed=seed, verbose=False, **CONFIGS[name])
    learner = PICRuleLearner(prog, lc)
    t0 = time.time()
    learner.fit(Xtr, ytr)
    t_rules = time.time() - t0

    yte_i = prog.target_index(yte)
    hard = learner._acc(Xte, yte_i, hard=True)

    t0 = time.time()
    mlp = train_mlp(Xtr, prog.target_index(ytr), n_classes, vocab, epochs=80, seed=seed)
    t_mlp = time.time() - t0

    row = {
        "task": name,
        "n_train": base["n_train"], "n_test": base["n_test"],
        "n_features": base["n_features"], "n_classes": n_classes,
        "rule_hard_acc": round(hard, 4),
        "gbt_acc": round(base["gbt_acc"], 4),
        "logreg_acc": round(base["logreg_acc"], 4),
        "mlp_acc": round(mlp_acc(mlp, Xte, yte_i), 4),
        "certified_frac_delta_0.25": round(certified_fraction(prog, Xte, 0.25), 3),
        "rules": prog.n_rules,
        "rule_params": rule_param_count(prog),
        "rule_s": round(t_rules), "mlp_s": round(t_mlp),
        "summary": program_summary(prog),
    }
    if name in SOUFFLE_VERIFY and shutil.which("souffle"):
        row["souffle"] = verify_export(prog, Xte[:200])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", default="all", choices=[*CONFIGS, "all"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    names = list(CONFIGS) if args.task == "all" else [args.task]
    for name in names:
        print(f"=== {name} ===", flush=True)
        r = bench(name, Path(args.data))
        rows.append(r)
        for k, v in r.items():
            if k != "summary":
                print(f"  {k}: {v}")
        print(f"  summary: {r['summary']}", flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
