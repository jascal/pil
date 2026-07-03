"""Demo: learn parity-8 as a PIC-LP program, inspect it, export it, verify it in Soufflé.

Run:  python experiments/rule_program_demo.py [--out results/parity8_learned.dl]

What to look for in the printout: the learner's threshold rules count matches to a
reference pattern and the head signs *alternate with the ones-count parity* -- the
textbook depth-2 threshold-circuit construction of parity, rediscovered from random
data-seeded rules by semiring backprop + structural search.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from pil.datalog_export import export_program, verify_export
from pil.rule_learner import PICRuleLearner, RuleLearnerConfig
from pil.rules import RuleProgram, RuleProgramConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/parity8_learned.dl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    bits = torch.randint(0, 2, (6000, 8), generator=g)
    X, y = bits + 10, bits.sum(1).remainder(2) + 10   # tokens 10/11; parity target

    cfg = RuleProgramConfig(
        vocab_size=50304, window=8, frame_dim=16, candidates=[10, 11], seed=args.seed
    )
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=8, init_rules=96, births_per_phase=64, epochs_per_phase=25,
        max_rules=512, recency_gamma=1.0, seed=args.seed, verbose=True,
    )
    learner = PICRuleLearner(prog, lc)
    rep = learner.fit(X, y)
    print(f"\nfinal: {rep.final}")

    # -- inspect the learned threshold structure --------------------------------
    s1 = prog.strata[0]
    C = prog.head_incidences()
    print(f"\n{prog.n_rules} rules ({int(s1.is_thresh.sum())} threshold); "
          "theta, ones-counting literals, head(P1-P0):")
    rows = []
    for k in range(s1.n_rules):
        if not bool(s1.is_thresh[k]):
            continue
        rel = torch.sigmoid(s1.rel.data[k]) > 0.5
        sgn = s1.sgn.data[k] > 0
        ones = sum(
            1 for o in range(cfg.window)
            if rel[o]
            and ((s1.anchor[k, o] == 11 and sgn[o]) or (s1.anchor[k, o] == 10 and not sgn[o]))
        )
        rows.append((int(torch.ceil(s1.thr.data[k] - 1e-6)), int(rel.sum()), ones,
                     float(C[k, 1] - C[k, 0])))
    for th, n_lit, ones, dm in sorted(rows):
        print(f"  theta>={th:2d}  lits={n_lit}  ones-counting={ones}  head={dm:+.2f}")

    # -- export + verify ---------------------------------------------------------
    dl = export_program(prog, "parity8_learned")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dl)
    print(f"\nwrote {out} ({len(dl.splitlines())} lines)")

    if shutil.which("souffle"):
        res = verify_export(prog, X[:1000])
        print(f"souffle vs tensor hard path: {res}")
    else:
        print("souffle not on PATH; skipped execution check")


if __name__ == "__main__":
    main()
