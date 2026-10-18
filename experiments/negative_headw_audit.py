"""Delete negative headw rows and record whether the hard argmax moves.

Fits the same parity-8 and 8-way programs as ``pole_audit``. Three edits zero
selected negative incidences and leave positive entries in place. ``changed`` is
the number of validation rows whose hard argmax differs. Mean margin is printed
and does not vote. Parity's ``scattered_neg`` edit selects no rows and must hold.

Run:  python experiments/negative_headw_audit.py
"""

from __future__ import annotations

import os

# Fixed before torch loads so two CPU invocations share one BLAS reduction order.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from pole_audit import eight_task, parity_task

from pil.pole import ABLATION_MODES, ablated_incidence, decide_delta, incidence_logits, pole_label
from pil.rule_learner import PICRuleLearner
from pil.rules import RuleProgram

torch.set_num_threads(1)


def _hard_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean())


def _fit(task):
    x, y, prog, lc = task()
    rep = PICRuleLearner(prog, lc).fit(x, y)
    if rep.val_X is None or rep.val_y is None:
        raise RuntimeError("fit did not stash the validation slice")
    return prog, rep


def _score_program(name: str, prog: RuleProgram, rep) -> tuple[str, str]:
    rows = prog.head_incidences()
    scattered = sum(pole_label(rows[k])[3] == "scattered" for k in range(rows.shape[0]))
    if name == "parity8" and scattered != 0:
        raise RuntimeError("parity8 scattered != 0; the candidate axis was scored wrong")
    base = incidence_logits(prog, rep.val_X, rows)
    acc = _hard_acc(base, rep.val_y)
    reported = float(rep.final["hard_val_acc"])
    if acc != reported:
        raise RuntimeError(
            f"{name} unabated hard accuracy {acc} != hard_val_acc {reported}"
        )
    lines = [f"prog={name} hard_val_acc={reported:.6f}"]
    all_neg = "holds"
    for mode in ABLATION_MODES:
        edited = ablated_incidence(rows, mode)
        after = incidence_logits(prog, rep.val_X, edited)
        changed, margin_before, margin_after = decide_delta(base, after)
        word = "moves" if changed > 0 else "holds"
        if mode == "all_neg":
            all_neg = word
        lines.append(
            f"prog={name} mode={mode} changed={changed} "
            f"hard_acc={_hard_acc(after, rep.val_y):.6f} "
            f"margin_before={margin_before:.6f} margin_after={margin_after:.6f} {word}"
        )
    text = "\n".join(lines)
    control = f"prog={name} mode=scattered_neg"
    if name == "parity8" and any(line.startswith(control) and not line.endswith("holds") for line in lines):
        print(text, flush=True)
        raise RuntimeError("parity8 scattered_neg moved; the edit touched the wrong rows")
    return text, all_neg


def main() -> None:
    print("[headw] device=cpu threads=1 seed=0")
    parity_prog, parity_rep = _fit(parity_task)
    parity_text, parity_all = _score_program("parity8", parity_prog, parity_rep)
    print(parity_text, flush=True)
    eight_prog, eight_rep = _fit(eight_task)
    eight_text, eight_all = _score_program("eight", eight_prog, eight_rep)
    print(eight_text, flush=True)
    if parity_all == "moves" or eight_all == "moves":
        decision = "negative_load_bearing"
    else:
        decision = "net_sum_sufficient"
    print(f"DECISION {decision}")


if __name__ == "__main__":
    main()
