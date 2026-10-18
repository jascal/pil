"""Score the negative incidences of two hardened PIC programs.

Parity-8 uses the published demo config. The 8-way task is generated here: the
target is the token at position 0, so a negative side can land on more than one
candidate. Cuts live in ``pil.pole`` and are not retuned.

Run:  python experiments/pole_audit.py
"""

from __future__ import annotations

import os

# Fixed before torch loads so two CPU invocations share one BLAS reduction order.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

from pil.pole import pole_label
from pil.rule_learner import PICRuleLearner, RuleLearnerConfig
from pil.rules import RuleProgram, RuleProgramConfig

torch.set_num_threads(1)


def _score(name: str, prog: RuleProgram) -> tuple[str, int, int, int]:
    rows = prog.head_incidences()
    counts = {"dead": 0, "thin": 0, "bipolar": 0, "scattered": 0}
    lines = []
    cands = prog.candidate_ids.tolist()
    for k in range(rows.shape[0]):
        neg, leader, share, label = pole_label(rows[k])
        counts[label] += 1
        token = -1 if leader < 0 else int(cands[leader])
        lines.append(
            f"prog={name} rule={k} neg={neg:.6f} leader={leader} "
            f"share={share:.6f} token={token} label={label}"
        )
    voters = rows.shape[0] - counts["dead"]
    lines.append(
        f"prog={name} dead={counts['dead']} thin={counts['thin']} "
        f"bipolar={counts['bipolar']} scattered={counts['scattered']} voters={voters}"
    )
    if voters == 0:
        lines.append(f"prog={name} FAILED_FIT all heads dead")
    return "\n".join(lines), counts["bipolar"], voters, counts["scattered"]


def fit_parity() -> tuple[str, int, int, int]:
    g = torch.Generator().manual_seed(0)
    bits = torch.randint(0, 2, (6000, 8), generator=g)
    X, y = bits + 10, bits.sum(1).remainder(2) + 10
    cfg = RuleProgramConfig(
        vocab_size=50304, window=8, frame_dim=16, candidates=[10, 11], seed=0, device="cpu"
    )
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=8, init_rules=96, births_per_phase=64, epochs_per_phase=25,
        max_rules=512, recency_gamma=1.0, seed=0, verbose=False,
    )
    learner = PICRuleLearner(prog, lc)
    rep = learner.fit(X, y)
    text, bipolar, voters, scattered = _score("parity8", prog)
    acc = (
        f"prog=parity8 hard_train_acc={rep.final['hard_train_acc']:.6f} "
        f"hard_val_acc={rep.final['hard_val_acc']:.6f}"
    )
    return acc + "\n" + text, bipolar, voters, scattered


def fit_eight() -> tuple[str, int, int, int]:
    g = torch.Generator().manual_seed(0)
    X = torch.randint(0, 8, (4000, 4), generator=g)
    y = X[:, 0].clone()
    cfg = RuleProgramConfig(
        vocab_size=8, window=4, frame_dim=16, candidates=list(range(8)), seed=0, device="cpu"
    )
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=4, init_rules=32, births_per_phase=16, epochs_per_phase=10,
        max_rules=128, seed=0, verbose=False,
    )
    learner = PICRuleLearner(prog, lc)
    rep = learner.fit(X, y)
    text, bipolar, voters, scattered = _score("eight", prog)
    acc = (
        f"prog=eight hard_train_acc={rep.final['hard_train_acc']:.6f} "
        f"hard_val_acc={rep.final['hard_val_acc']:.6f}"
    )
    return acc + "\n" + text, bipolar, voters, scattered


def main() -> None:
    print("[pole] device=cpu threads=1 seed=0")
    parity, bp, pv, ps = fit_parity()
    print(parity, flush=True)
    if ps != 0:
        raise RuntimeError("parity8 scattered != 0; the candidate axis was scored wrong")
    eight, eb, ev, _es = fit_eight()
    print(eight, flush=True)
    if pv == 0 or ev == 0:
        decision = "failed_fit"
    elif bp > 0 or eb > 0:
        decision = "bipolar_present"
    else:
        decision = "no_bipolar"
    print(f"DECISION {decision}")


if __name__ == "__main__":
    main()
