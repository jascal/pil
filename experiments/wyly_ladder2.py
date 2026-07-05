"""The DATASET x RULE-LIBRARY matrix: empirical ladders for the library/dataset goal.

Reads whatever (library, teacher, dataset) v5 runs exist (wyly_lm_v5.py STATE files) plus the
teacher/dataset files, and prints:
  - per (dataset, teacher): gold (capability), copy% (teacher-follows-induction incidence),
    v5[base] and v5[ext] rows -- teacher-agreement, rules-marginal, certified-core (package cover),
    admitted rules;
  - the wikitext ext ladder across all 7 pythia sizes vs the sec-8.5 base ladder;
  - the named confounds + design directions live in the companion report
    (docs/notes/wyly_library_dataset_ladders.md).

Run: cd pil && .venv/bin/python experiments/wyly_ladder2.py
"""

from __future__ import annotations

from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
DATASETS = ["wikitext", "code", "wt103"]
TEACHERS = ["pythia14m", "pythia70m", "pythia160m", "pythia410m", "pythia1b", "pythia1.4b",
            "pythia2.8b"]
# sec-8.5 base (induction-only) certified cores, from wyly_ladder.py @ commit 9a3ffeb
BASE85 = {"pythia14m": 0.285, "pythia70m": 0.276, "pythia160m": 0.258, "pythia410m": 0.245,
          "pythia1b": 0.242, "pythia1.4b": 0.231, "pythia2.8b": 0.230}


def dfile(ds):
    return REPO / "data" / ("wyly_nexttoken_wikitext_L256.pt" if ds == "wikitext"
                            else f"wyly_nexttoken_{ds}_L256.pt")


def tfile(tag, ds):
    return REPO / "data" / (f"wyly_teacher_{tag}_L256.pt" if ds == "wikitext"
                            else f"wyly_teacher_{tag}_{ds}_L256.pt")


def sfile(lib, tag, ds):
    return REPO / "data" / f"wyly_v5_{lib}_{tag}_{ds}.pt"


def copy_incidence(ids, teach, bs=8192):
    hits = 0
    for i in range(0, len(ids), bs):
        w, t = ids[i:i + bs], teach[i:i + bs]
        hits += int(((w[:, :-1] == w[:, -1:]) & (w[:, 1:] == t.unsqueeze(1))).any(1).sum())
    return hits / len(ids)


def main():
    print("DATASET x RULE-LIBRARY LADDERS -- 80k L=256 windows per dataset, teacher-decision "
          "agreement\n")
    for ds in DATASETS:
        if not dfile(ds).exists():
            continue
        d = torch.load(dfile(ds), map_location="cpu")
        ids, gold = d["kept_ids"], d["target"]
        print(f"== dataset: {ds} ==")
        print(f"{'teacher':>12}{'gold':>7}{'copy%':>7}{'lib':>6}{'agree':>8}{'r-marg':>8}"
              f"{'core':>7}{'cover':>8}  admitted")
        for tag in TEACHERS:
            if not tfile(tag, ds).exists():
                continue
            teach = torch.load(tfile(tag, ds), map_location="cpu")["teacher"]
            gacc = float((teach == gold).float().mean())
            cpy = copy_incidence(ids, teach)
            shown = False
            for lib in ["base", "ext"]:
                if not sfile(lib, tag, ds).exists():
                    continue
                m = torch.load(sfile(lib, tag, ds), map_location="cpu")
                pre = (f"{tag:>12}{gacc:>7.3f}{cpy:>7.1%}" if not shown else " " * 26)
                shown = True
                rules = ", ".join(r.split(" [")[0] for r in m["rules"])
                print(f"{pre}{lib:>6}{m['full']:>8.3f}{m['full'] - m['norules']:>+8.3f}"
                      f"{m['core']['agree']:>7.3f}{m['core']['cover']:>8.1%}  [{rules}]")
            if not shown and ds == "wikitext" and tag in BASE85:
                print(f"{tag:>12}{gacc:>7.3f}{cpy:>7.1%}{'s8.5':>6}{'':>8}{'':>8}"
                      f"{BASE85[tag]:>7.3f}{'':>8}  [induction 1,2,3 -- v4 run]")
        print()
    print("wikitext ext-vs-base certified core across the pythia ladder:")
    print(f"{'teacher':>12}{'base(s8.5)':>11}{'ext':>7}{'delta':>8}")
    for tag in TEACHERS:
        f = sfile("ext", tag, "wikitext")
        if f.exists() and tag in BASE85:
            c = torch.load(f, map_location="cpu")["core"]["agree"]
            print(f"{tag:>12}{BASE85[tag]:>11.3f}{c:>7.3f}{c - BASE85[tag]:>+8.3f}")


if __name__ == "__main__":
    main()
