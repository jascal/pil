"""Structural-complexity probe: does composition DEPTH buy coverage on pythia's decode residual?

The rate axis (tropic F14) showed pythia's computed fraction is data-limited, not a hard floor — but
that measures HOW MUCH is computed, not WHAT the computation is. This probes the structure axis with the
tool the program already has: the PIC rule learner induces an explicit weighted-Datalog program and can
DEEPEN (stack strata = bounded composition). Cap max_strata at 1..4 and read the certified coverage +
accuracy each depth buys, on pythia-70m's (8-token context → argmax) decode data.

- coverage/accuracy climbs with depth ⇒ the residual is symbolic-in-a-richer-class (composition helps);
  the rule learner, extended, models it.
- coverage plateaus at depth 1 ⇒ the residual is a flat lookup + a genuinely distributed remainder no
  bounded composition captures (the forge tax has no compact program in this class).

Data: tropic's m2 pythia-70m windows (scratchpad/m2_train.pt: token ids + pred). CPU.
Usage: python experiments/rule_depth_probe.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from pil.rule_learner import PICRuleLearner, RuleLearnerConfig
from pil.rules import RuleProgram, RuleProgramConfig, certified_fraction

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
W = 8
TOP_V = 512


def build_windows(fname="m2_train.pt"):
    d = torch.load(SP / fname)
    ids, mask, pred = d["ids"], d["mask"], d["pred"]
    lens = mask.long().sum(1)
    keep = lens >= W
    xs = [ids[i, lens[i] - W: lens[i]] for i in range(ids.shape[0]) if keep[i]]
    x = torch.stack(xs)
    y = pred[keep]
    return x, y


def hard_acc(learner, prog, xte, yte):
    with torch.no_grad():
        logits = learner._logits_in_chunks(xte, hard=True)
        pred = prog.candidate_ids[logits.argmax(-1)]
    return float((pred == yte).float().mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="m2_train.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    x, y = build_windows(args.data)
    n = x.shape[0]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    cut = int(0.8 * n)
    xtr, ytr = x[perm[:cut]], y[perm[:cut]]
    xte, yte = x[perm[cut:]], y[perm[cut:]]
    vals, counts = ytr.unique(return_counts=True)
    cands = vals[counts.argsort(descending=True)[:TOP_V]].sort().values
    cand_set = set(cands.tolist())
    cover_te = float(sum(int(t) in cand_set for t in yte.tolist()) / len(yte))
    # fit requires targets in the candidate set -> restrict to in-candidate contexts (same across depths)
    tr_in = torch.tensor([int(t) in cand_set for t in ytr.tolist()])
    te_in = torch.tensor([int(t) in cand_set for t in yte.tolist()])
    xtr, ytr = xtr[tr_in], ytr[tr_in]
    xte, yte = xte[te_in], yte[te_in]
    print(f"windows N={n}  train {cut}  test {n - cut}  W={W}  top-{TOP_V} cover {cover_te:.3f}")
    print(f"in-candidate train {xtr.shape[0]}  test {xte.shape[0]}")
    print(f"\n{'max_strata':>11}{'strata_used':>12}{'n_rules':>9}{'coverage@0.25':>14}{'hard_acc':>9}")
    for ms in (1, 2, 3, 4):
        cfg = RuleProgramConfig(vocab_size=50304, window=W, frame_dim=64, candidates=cands.tolist(),
                                lookup_offsets=(W - 3, W - 2), direct_lookup_offsets=(W - 1,), seed=args.seed)
        prog = RuleProgram(cfg)
        lc = RuleLearnerConfig(n_phases=12, init_rules=768, births_per_phase=384, epochs_per_phase=4,
                               max_rules=2048, batch_size=512, probe_size=1024, max_strata=ms,
                               deepen_patience=1, seed=args.seed, verbose=False)
        learner = PICRuleLearner(prog, lc)
        learner.fit(xtr, ytr)
        cov = certified_fraction(prog, xte, 0.25)
        acc = hard_acc(learner, prog, xte, yte)
        print(f"{ms:>11}{len(prog.strata):>12}{prog.n_rules:>9}{cov:>14.3f}{acc:>9.3f}", flush=True)

    print("\nread: coverage/acc RISING with strata ⇒ composition depth captures structure (residual is "
          "symbolic-in-a-richer-class). FLAT past depth-1 ⇒ a lookup + genuinely distributed remainder "
          "no bounded Datalog composition compacts — the forge tax has no compact program in this class.")


if __name__ == "__main__":
    main()
