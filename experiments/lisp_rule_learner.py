"""Our PIC rule learner on Lisp primitive redexes — does the symbolic prior separate retrieved/computed?

Each redex is (op a b) with a,b single integer tokens (int-as-token). We split OPERAND PAIRS into
train/held-out (disjoint pairs) and measure held-out accuracy per operator. The key contrast:

  = , <   RELATIONAL — expressible as a symbolic rule over eq/order atoms  -> should GENERALIZE few-shot
  + - *   COMPUTED   — no finite symbolic form; only memorizable as a per-pair table -> should MEMORIZE
                       (train acc high, held-out acc collapses)

So held-out acc is a direct retrieved-vs-computed readout: relational ops = retrieved-by-rule (cheap,
general); arithmetic = the computed core the symbolic prior cannot compress. Compare to the transformer,
which needs thousands of examples to statistically approximate even the relational rules.

Run: cd pil && .venv/bin/python experiments/lisp_rule_learner.py
"""

from __future__ import annotations

import random

import torch

from pil.rule_learner import PICRuleLearner, RuleLearnerConfig
from pil.rules import RuleProgram, RuleProgramConfig, certified_fraction

MAXV = 12                       # operands 0..12
OPS = ["+", "-", "*", "<", "="]
MAXR = 200                      # int tokens are 0..MAXR (token id == the integer)
LP, RP = MAXR + 1, MAXR + 2
OP_TOK = {op: MAXR + 3 + i for i, op in enumerate(OPS)}
VOCAB = MAXR + 3 + len(OPS)


def compute(op, a, b):
    return {"+": a + b, "-": a - b, "*": a * b, "<": int(a < b), "=": int(a == b)}[op]


def make(op, pairs):
    x = [[LP, OP_TOK[op], a, b, RP] for a, b in pairs]
    y = [compute(op, a, b) for a, b in pairs]
    return torch.tensor(x), torch.tensor(y)


def hard_acc(learner, prog, xte, yte):
    with torch.no_grad():
        logits = learner._logits_in_chunks(xte, hard=True)
        pred = prog.candidate_ids[logits.argmax(-1)]
    return float((pred == yte).float().mean())


def run_op(op, seed=0, train_frac=0.7):
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(MAXV + 1) for b in range(MAXV + 1)]
    if op == "-":
        pairs = [(a, b) for a, b in pairs if a >= b]   # keep results non-negative
    rng.shuffle(pairs)
    cut = int(train_frac * len(pairs))
    xtr, ytr = make(op, pairs[:cut])
    xte, yte = make(op, pairs[cut:])
    cands = sorted(set(ytr.tolist()) | set(yte.tolist()))
    cfg = RuleProgramConfig(vocab_size=VOCAB, window=5, frame_dim=32, candidates=cands,
                            eq_atoms=True, lookup_offsets=(2, 3), direct_lookup_offsets=(2, 3), seed=seed)
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(n_phases=8, init_rules=128, births_per_phase=96, max_rules=768,
                           epochs_per_phase=6, max_strata=2, le_seed_p=0.4, seed=seed, verbose=False)
    learner = PICRuleLearner(prog, lc)
    learner.fit(xtr, ytr)
    return {
        "op": op, "ntrain": len(xtr), "ntest": len(xte),
        "train_acc": hard_acc(learner, prog, xtr, ytr),
        "heldout_acc": hard_acc(learner, prog, xte, yte),
        "certified": certified_fraction(prog, xte, 0.25),
        "n_rules": prog.n_rules,
    }


def main():
    print(f"PIC rule learner on primitive Lisp redexes (operands 0..{MAXV}, 70/30 pair split)")
    print(f"{'op':>3}{'ntrain':>8}{'ntest':>7}{'train_acc':>11}{'heldout_acc':>13}{'certified':>11}{'rules':>7}")
    for op in OPS:
        r = run_op(op)
        kind = "relational" if op in ("<", "=") else "arithmetic"
        print(f"{r['op']:>3}{r['ntrain']:>8}{r['ntest']:>7}{r['train_acc']:>11.3f}"
              f"{r['heldout_acc']:>13.3f}{r['certified']:>11.3f}{r['n_rules']:>7}  ({kind})", flush=True)
    print("\nread: held-out acc HIGH on relational ops (=,<) = a symbolic rule generalized (retrieved-by-"
          "rule, cheap); held-out acc COLLAPSES on arithmetic (+,-,*) = only per-pair memorization, no "
          "symbolic form = the computed core. Train_acc high everywhere (memorization is always available).")


if __name__ == "__main__":
    main()
