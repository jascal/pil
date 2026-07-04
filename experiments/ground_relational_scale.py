"""Deep dive: does the cross-token linearity gap SHRINK with scale (compiled) or PERSIST (computed)?

The frontier finding: cross-token relations (induction/repeat, neighbor-reading) are only ~0.78 linearly
decodable vs lexical ~0.94 -- the computed/distributed regime. Sharp question: with model SCALE, does the
induction circuit get COMPILED into a clean linear feature (gap shrinks -> computed core is scale-dependent,
'inline-cache' idea) or is cross-token comparison IRREDUCIBLY computed (gap persists)?

Compares the full-rank balanced linear ceiling for LEXICAL vs CROSS-TOKEN concepts across pythia 70m/410m/1b.

Run: cd pil && .venv/bin/python experiments/ground_relational_scale.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import ground_relational as GR  # noqa: E402

MODELS = [("pythia-70m", "relational_pythia70m.pt"),
          ("pythia-410m", "relational_pythia410m.pt"),
          ("pythia-1b", "relational_pythia1b.pt")]
LEXICAL = ["func", "cap"]
CROSS = ["is_repeat", "local_repeat", "prev_func"]      # cross-token comparison (the distributed ones)
STRUCT = ["inside_paren"]                                # structural counting (clean-linear control)


def ceiling(r, y, seed):
    rk, yk, tr, te = GR.balanced(r, y, seed)
    return GR.probe(rk[tr], yk[tr], rk[te], yk[te])


def mean_ceiling(r, labs, names, group):
    vals = []
    for nm in group:
        y = labs[:, names.index(nm)].float()
        if int(y.sum()) < 300:
            continue
        vals.append(sum(ceiling(r, y, s) for s in (0, 1, 2)) / 3)
    return sum(vals) / max(len(vals), 1)


def main():
    print("cross-token linearity vs SCALE -- does the computed gap shrink (compiled) or persist?\n")
    print(f"{'model':>13}{'lexical':>9}{'cross-tok':>10}{'struct':>8}{'GAP(lex-cross)':>16}")
    for name, path in MODELS:
        f = GR.SP / path
        if not f.exists():
            print(f"{name:>13}   ({path} not found; skipping)")
            continue
        d = torch.load(f)
        r, names_, labs = d["r"].float(), d["names"], d["labels"]
        lx = mean_ceiling(r, labs, names_, LEXICAL)
        cx = mean_ceiling(r, labs, names_, CROSS)
        st = mean_ceiling(r, labs, names_, STRUCT)
        print(f"{name:>13}{lx:>9.3f}{cx:>10.3f}{st:>8.3f}{lx - cx:>16.3f}", flush=True)
    print("\nread: GAP (lexical - cross-token) SHRINKING with scale = the induction/comparison circuit gets\n"
          "COMPILED into a cleaner linear feature at scale (computed core scale-dependent; inline-cache).\n"
          "GAP PERSISTING = cross-token comparison is IRREDUCIBLY computed/distributed at every scale = the\n"
          "stable home of the forge tax. (struct = bracket-counting control, clean-linear throughout.)")


if __name__ == "__main__":
    main()
