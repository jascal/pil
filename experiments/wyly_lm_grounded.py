"""GROUNDED concepts for Wyly-LM: does the sec-6 unification change anything the proposer can use?

The endgame review's sec 6 proposes: don't learn Wyly's concepts from scratch -- ground them in a real
model's geometry, and let the proposer operate in that named space. The Fable review (sec 5) split this
claim: grounding fixes the proposal SPACE, but proposal SCORING must be interaction-aware regardless,
and "ground once, run standalone" is teacher-in-training-path, not the adapter. This experiment is the
minimal falsifiable version of the space claim on the wikitext bench:

  concept init : RANDOM (from-scratch, as the whole arc)  vs  GROUNDED = z-scored top-K0 PCA of
                 pythia-70m's embedding rows for this task's input vocab, read directly from the
                 sibling fieldrun bundle (no HF, no GPU model -- the teacher appears ONLY at init).
  concept table: TRAINABLE (grounded init only)  vs  FROZEN (ground once, run standalone -- also a
                 STATIONARY literal space, the review's sec 2.2 intrinsic fix: propose over frozen
                 literals so incidence never rots under concept drift).
  proposer     : interaction-scored (the best intrinsic arm from wyly_lm_bench.py), same harness.

Read: on THIS task the transformer <= bigram (no super-linear headroom), so final top-1 is capped for
everyone; what grounding can still show is (a) a better/worse LINEAR path, (b) a changed RULES-MARGINAL
(the proposer finding more per rule in a meaningful, stationary space). If neither moves, sec 6's space
claim is untestable on wikitext-12tok and belongs on the battery (wyly_rel_battery.py).

Run: cd pil && .venv/bin/python experiments/wyly_lm_grounded.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from wyly_data import load_windows
from wyly_lm_bench import DEV, K0, V, WylyBench, grow_interaction, run_rules

BUNDLE = Path(__file__).resolve().parent.parent.parent / "fieldrun" / "bundles"


def pythia_embed(uv: torch.Tensor) -> torch.Tensor:
    """(len(uv), 512) f32 embedding rows for the compact input vocab, from the fieldrun bundle."""
    meta = json.loads((BUNDLE / "pythia-70m.fieldrun.json").read_text())
    arr = next(a for a in meta["arrays"] if a["name"] == "embed")
    assert arr["dtype"] == "f16"
    emb = np.memmap(BUNDLE / "pythia-70m.fieldrun.bin", dtype=np.float16, mode="r",
                    offset=arr["offset"], shape=tuple(arr["shape"]))
    return torch.from_numpy(np.asarray(emb[np.asarray(uv)], dtype=np.float32))


def grounded_init(uv: torch.Tensor) -> torch.Tensor:
    """z-scored top-K0 PCA coordinates of the teacher embedding = named-ish concept axes."""
    e = pythia_embed(uv)
    e = e - e.mean(0)
    _, _, vt = torch.pca_lowrank(e, q=K0, niter=4)
    c = e @ vt[:, :K0]
    return (c - c.mean(0)) / c.std(0).clamp_min(1e-6)


def main():
    ids, y, vocab, tr, te = load_windows(V, DEV)
    d = torch.load(Path(__file__).resolve().parent.parent / "data" / "wyly_nexttoken_pythia70m.pt")
    kept, target = d["kept_ids"], d["target"]
    keep = torch.isin(target, torch.bincount(target).argsort(descending=True)[:V])
    uv = kept[keep].reshape(-1).unique()                    # same compaction as load_windows
    ground = grounded_init(uv).to(DEV)
    print(f"GROUNDED-concepts study -- vocab {vocab}, teacher = pythia-70m embed via fieldrun bundle")
    print(f"grounded init: {tuple(ground.shape)} (z-scored PCA-{K0})\n")
    print(f"{'arm':>34}{'top-1':>9}{'ablated':>9}{'marginal':>10}{'rules':>7}")
    arms = [
        ("random init, trainable", None, False),
        ("grounded init, trainable", ground, False),
        ("random init, FROZEN C", None, True),
        ("grounded init, FROZEN C", ground, True),
    ]
    for name, init, freeze in arms:
        def patched(vocab_, init_=init, freeze_=freeze):
            m = WylyBench(vocab_)
            if init_ is not None:
                with torch.no_grad():
                    m.C.copy_(init_)
            if freeze_:
                sgd0 = m.sgd

                def sgd_nofreeze(lr, m_=m, sgd0_=sgd0):
                    g = m_.C.grad
                    m_.C.grad = None                        # frozen concept table
                    sgd0_(lr)
                    m_.C.grad = g
                m.sgd = sgd_nofreeze
            return m
        acc, acc0, nr = run_rules(ids, y, tr, te, grow_interaction, gate=False,
                                  make_model=patched)
        print(f"{name:>34}{acc:>9.3f}{acc0:>9.3f}{acc - acc0:>+10.3f}{nr:>7}", flush=True)
    print("\nread: grounded vs random on (a) the ablated column (linear path) and (b) the marginal")
    print("(what the interaction proposer extracts per rule in a meaningful/stationary space).")
    print("No movement = sec 6's space claim is untestable at 12-tok wikitext; test it on the battery.")


if __name__ == "__main__":
    main()
