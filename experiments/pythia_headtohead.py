"""Head-to-head: a *learned* PIC program vs rosetta's *decompiled* program vs pythia-160m.

The experiment the whole pipeline points at: distill pythia-160m's next-token decisions
(argmax on 8-token windows — the exact semantics of rosetta's logit_cache) into a PIC-LP
rule program, then score three artifacts on identical contexts:

  pythia-160m        the host model (the labels; agreement with itself = 1 by definition)
  circuits.dl        rosetta's decompiled program (mined from weights + its corpus),
                     executed via `souffle run.dl` on the same tok.facts
  PIL program        learned from (context, argmax) pairs alone -- no access to weights

Two arenas:
  wikitext held-out  20k windows pythia labeled, unseen by PIL training
  home turf          the 1526 logit-cache contexts circuits.dl was certified on

Usage:
  python experiments/pythia_headtohead.py --labels <windows.npz> \
      [--rosetta /home/allans/code/rosetta/models/pythia160m] [--out results/...]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pil.rule_learner import PICRuleLearner, RuleLearnerConfig
from pil.rules import RuleProgram, RuleProgramConfig, certified_fraction, program_summary

W = 8


def run_circuits_dl(model_dir: Path, X: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Execute rosetta's circuits.dl on contexts ``X``; returns argmax token id per
    instance (-1 = abstained). Uses the package's own run.dl harness."""
    with tempfile.TemporaryDirectory(prefix="rosetta_run_") as td:
        tdp = Path(td)
        with open(tdp / "tok.facts", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            for i in range(X.shape[0]):
                for o in range(X.shape[1]):
                    w.writerow([i, o, int(X[i, o])])
        (tdp / "temp.facts").write_text(f"{temp}\n")
        subprocess.run(
            ["souffle", str(model_dir / "run.dl"), "-F", str(tdp), "-D", str(tdp)],
            cwd=model_dir, check=True, capture_output=True, text=True,
        )
        best: dict[int, tuple[float, int]] = defaultdict(lambda: (float("-inf"), -1))
        with open(tdp / "cdist.csv") as f:
            for line in f:
                i_s, tk_s, p_s = line.split("\t")
                i, tk, p = int(i_s), int(tk_s), float(p_s)
                if p > best[i][0]:
                    best[i] = (p, tk)
        out = torch.full((X.shape[0],), -1, dtype=torch.long)
        for i, (_, tk) in best.items():
            out[i] = tk
    return out


def agreement(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    fired = pred >= 0
    return {
        "overall": float((pred == ref).float().mean()),
        "fired_frac": float(fired.float().mean()),
        "on_fired": float((pred[fired] == ref[fired]).float().mean()) if fired.any() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--rosetta", default="/home/allans/code/rosetta/models/pythia160m")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--top-v", type=int, default=1024)
    ap.add_argument("--circuits-n", type=int, default=8000)  # souffle budget on held-out
    args = ap.parse_args()

    d = np.load(args.labels)
    X = torch.from_numpy(d["X"])
    y = torch.from_numpy(d["y"])
    Xtr, ytr = X[: args.n_train], y[: args.n_train]
    Xte, yte = X[args.n_train :], y[args.n_train :]

    # candidate set: pythia's most frequent argmax decisions on train
    vals, counts = ytr.unique(return_counts=True)
    cands = vals[counts.argsort(descending=True)[: args.top_v]].sort().values
    tr_in = torch.isin(ytr, cands)
    te_in = torch.isin(yte, cands)
    coverage = float(te_in.float().mean())
    print(f"top-{args.top_v} candidate coverage of pythia argmax: {coverage:.3f}", flush=True)

    # bigram floor on the distillation labels
    mode: dict[int, int] = {}
    last = Xtr[tr_in][:, -1]
    ytr_in = ytr[tr_in]
    for c in last.unique().tolist():
        m = last == c
        mode[c] = int(ytr_in[m].mode().values)
    gmode = int(ytr_in.mode().values)
    big = torch.tensor([mode.get(int(c), gmode) for c in Xte[:, -1]])
    bigram = agreement(big, yte)

    # -- PIL: learn the program from (context, argmax) pairs -------------------
    cfg = RuleProgramConfig(
        vocab_size=50304, window=W, frame_dim=64, candidates=cands.tolist(),
        lookup_offsets=(W - 3, W - 2),
        direct_lookup_offsets=(W - 1,),   # full-rank bigram backbone (rosetta gram2d shape)
        seed=0,
    )
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=6, init_rules=768, births_per_phase=384, epochs_per_phase=4,
        max_rules=2048, batch_size=512, probe_size=1024,
        ambiguity_split_threshold=0.4, max_splits_per_phase=32, seed=0, verbose=True,
    )
    learner = PICRuleLearner(prog, lc)
    t0 = time.time()
    learner.fit(Xtr[tr_in], ytr[tr_in])
    t_fit = time.time() - t0

    with torch.no_grad():
        L = learner._logits_in_chunks(Xte, hard=True)
        pil_pred = prog.candidate_ids[L.argmax(-1)]
    pil = agreement(pil_pred, yte)
    pil["in_cand_only"] = float((pil_pred[te_in] == yte[te_in]).float().mean())

    # -- rosetta circuits.dl on the same held-out windows ----------------------
    n_c = min(args.circuits_n, Xte.shape[0])
    t0 = time.time()
    ros_pred = run_circuits_dl(Path(args.rosetta), Xte[:n_c])
    t_ros = time.time() - t0
    ros = agreement(ros_pred, yte[:n_c])

    # -- home turf: the logit-cache contexts circuits.dl was certified on ------
    lc_json = json.loads((Path(args.rosetta) / "logit_cache.json").read_text())
    Xh = torch.tensor([[int(t) for t in k.split(",")] for k in lc_json])
    yh = torch.tensor([v[0][0] for v in lc_json.values()])
    ros_home = agreement(run_circuits_dl(Path(args.rosetta), Xh), yh)
    with torch.no_grad():
        Lh = learner._logits_in_chunks(Xh, hard=True)
        pil_home_pred = prog.candidate_ids[Lh.argmax(-1)]
    pil_home = agreement(pil_home_pred, yh)

    rows = {
        "labels": str(args.labels),
        "n_train": int(tr_in.sum()), "n_heldout": len(yte), "candidates": args.top_v,
        "cand_coverage_heldout": round(coverage, 4),
        "bigram_floor": {k: round(v, 4) for k, v in bigram.items()},
        "pil_wikitext": {k: round(v, 4) for k, v in pil.items()},
        "rosetta_wikitext": {k: round(v, 4) for k, v in ros.items()},
        "rosetta_wikitext_n": n_c,
        "pil_home_turf": {k: round(v, 4) for k, v in pil_home.items()},
        "rosetta_home_turf": {k: round(v, 4) for k, v in ros_home.items()},
        "home_turf_n": len(yh),
        "pil_fit_s": round(t_fit), "rosetta_souffle_s": round(t_ros),
        "certified_frac_delta_0.25": round(certified_fraction(prog, Xte[:4000], 0.25), 3),
        "program": program_summary(prog),
    }
    for k, v in rows.items():
        print(f"  {k}: {v}", flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
