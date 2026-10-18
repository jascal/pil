"""Score negative incidences of the published mod-add program.

The task builder is ``rule_learner_bench.build_modadd`` (p=97, 50% of pairs, seed 0,
pythia residue tokens, the same schema library and learner config). Schema weights are
not rows of ``head_incidences`` and are not scored. The MLP baseline is not trained.

Run:  python experiments/modadd_pole_audit.py
"""

from __future__ import annotations

import os

# Fixed before torch loads so two CPU invocations share one BLAS reduction order.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from rule_learner_bench import build_modadd

from pil.pole import pole_label
from pil.rule_learner import PICRuleLearner

torch.set_num_threads(1)


def main() -> None:
    print("[modadd-pole] device=cpu threads=1 seed=0 schemas_scored=0")
    xtr, ytr, _xte, _yte, prog, lc, _accepted, _scores = build_modadd()
    if prog.candidate_ids.numel() != 97:
        raise RuntimeError(
            f"modadd candidate count is {prog.candidate_ids.numel()}, expected 97 residue tokens"
        )
    rep = PICRuleLearner(prog, lc).fit(xtr, ytr)
    rows = prog.head_incidences()
    counts = {"dead": 0, "thin": 0, "bipolar": 0, "scattered": 0}
    cands = prog.candidate_ids.tolist()
    for k in range(rows.shape[0]):
        neg, leader, share, label = pole_label(rows[k])
        counts[label] += 1
        token = -1 if leader < 0 else int(cands[leader])
        print(
            f"prog=modadd rule={k} neg={neg:.6f} leader={leader} "
            f"share={share:.6f} token={token} label={label}",
            flush=True,
        )
    voters = rows.shape[0] - counts["dead"]
    print(
        f"prog=modadd rules={rows.shape[0]} dead={counts['dead']} thin={counts['thin']} "
        f"bipolar={counts['bipolar']} scattered={counts['scattered']} voters={voters}"
    )
    print(
        f"prog=modadd hard_train_acc={rep.final['hard_train_acc']:.6f} "
        f"hard_val_acc={rep.final['hard_val_acc']:.6f}"
    )
    if rows.shape[0] == 0:
        print("prog=modadd FAILED_FIT no rule rows")
    elif voters == 0:
        print("prog=modadd FAILED_FIT all heads dead")


if __name__ == "__main__":
    main()
