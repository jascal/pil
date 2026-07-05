"""The CERTIFIABILITY SCALING LAW: rule-shapedness + student/core metrics across the pythia ladder.

Rungs 2-4 of the scaling goal. For every teacher file present (wyly_make_teacher.py):
  RUNG 2 (no training, full 80k windows, original token space):
    gold    : teacher's own top-1 (the capability axis);
    copy%   : incidence of teacher-follows-induction (some earlier occurrence of the last token whose
              successor IS the teacher's decision) -- the induction bump vs scale;
    big->T  : corpus-bigram (train-region argmax) agreement with the teacher on test windows -- how
              n-gram-shaped the teacher's decisions are;
    d-big   : decisions-bigram ceiling (last token -> teacher decision, fitted on train windows);
    inter-scale decision agreement matrix (test windows).
  RUNG 3 (per v4 state present): student teacher-agreement (soft), certified-core cover agreement,
    admitted rules. RUNG 4: the table IS the curve; verdict printed with the named confounds.

Run: cd pil && .venv/bin/python experiments/wyly_ladder.py
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "wyly_nexttoken_wikitext_L256.pt"
MODELS = [("pythia14m", 14), ("pythia70m", 70), ("pythia160m", 160), ("pythia410m", 410),
          ("pythia1b", 1000), ("pythia1.4b", 1400), ("pythia2.8b", 2800)]


def teacher_file(tag):
    p = REPO / "data" / f"wyly_teacher_{tag}_L256.pt"
    return p if p.exists() else None


def copy_incidence(ids, teach, bs=8192):
    hits = 0
    for i in range(0, len(ids), bs):
        w, t = ids[i:i + bs], teach[i:i + bs]
        m = (w[:, :-1] == w[:, -1:]) & (w[:, 1:] == t.unsqueeze(1))
        hits += int(m.any(1).sum())
    return hits / len(ids)


def main():
    d = torch.load(DATA, map_location="cpu")
    ids, gold = d["kept_ids"], d["target"]
    n = len(ids)
    tr = torch.arange(int(0.85 * n))
    te = torch.arange(int(0.85 * n), n)
    # corpus bigram map from the train-region windows (within-window adjacency, original token space)
    big = defaultdict(Counter)
    for row in ids[tr][::12].tolist():                       # every 12th window ~ dedups stride-1 overlap
        for a, b in zip(row, row[1:], strict=False):
            big[a][b] += 1
    bigmap = {a: c.most_common(1)[0][0] for a, c in big.items()}
    last_te = ids[te, -1].tolist()

    rows, teachers = [], {}
    for tag, mp in MODELS:
        f = teacher_file(tag)
        if not f:
            continue
        teach = torch.load(f, map_location="cpu")["teacher"]
        teachers[tag] = teach
        gacc = float((teach == gold).float().mean())
        cpy = copy_incidence(ids, teach)
        bt = teach[te].tolist()
        bagree = sum(1 for lt, t in zip(last_te, bt, strict=False)
                     if bigmap.get(lt, -1) == t) / len(te)
        dmap = defaultdict(Counter)
        for lt, t in zip(ids[tr, -1].tolist(), teach[tr].tolist(), strict=False):
            dmap[lt][t] += 1
        dmap = {a: c.most_common(1)[0][0] for a, c in dmap.items()}
        dagree = sum(1 for lt, t in zip(last_te, bt, strict=False)
                     if dmap.get(lt, -1) == t) / len(te)
        row = {"tag": tag, "M": mp, "gold": gacc, "copy": cpy, "big": bagree, "dbig": dagree}
        st = REPO / "data" / (f"wyly_v4_state_{tag}.pt" if tag != "pythia70m"
                              else "wyly_v4_state.pt")
        pkg = REPO / "data" / (f"wyly_expert_package_v4_{tag}" if tag != "pythia70m"
                               else "wyly_expert_package_v4")
        if st.exists() and (pkg / "admitted.json").exists():
            os.environ["WYLY_V4_TAG"] = tag                  # (informational; loaders take paths)
            import wyly_lm_v2 as v2
            from wyly_lm_grounded import grounded_init
            from wyly_lm_v3 import WylyV3, mir_induction_L
            from wyly_race_pythia import student_core_cover

            dd = torch.load(DATA, map_location="cpu")
            teach_full = teach
            keep = torch.isin(teach_full,
                              torch.bincount(teach_full).argsort(descending=True)[:v2.V])
            kids, kt = dd["kept_ids"][keep], teach_full[keep]
            nk, ell = kids.shape
            flat = torch.cat([kids.reshape(-1), kt])
            uv, inv = flat.unique(return_inverse=True)
            kids2, yv = inv[:nk * ell].reshape(nk, ell), inv[nk * ell:]
            cls, y = yv.unique(return_inverse=True)
            kids2, y, cls = kids2.to(v2.DEV), y.to(v2.DEV), cls.to(v2.DEV)
            kte = torch.arange(int(0.85 * nk), nk, device=v2.DEV)
            ground = grounded_init(uv)
            ground = ground / ground.shape[1] ** 0.5
            model = WylyV3(ground, cls.cpu(), ell)
            admitted = json.loads((pkg / "admitted.json").read_text())
            for rec in admitted:
                model.install(f"induction L={rec['L']}", mir_induction_L(rec["L"]))
            model.load_state_dict(torch.load(st, map_location="cpu"))
            model = model.to(v2.DEV)
            row["student"] = v2.top1(model, kids2, y, kte)
            row["core"] = student_core_cover(model, admitted, kids2, y, cls, kte, 0.0)["agree"]
            row["rules"] = sorted(r["L"] for r in admitted)
            row["keep"] = float(keep.float().mean())
            del model
            torch.cuda.empty_cache()
        rows.append(row)

    print("THE CERTIFIABILITY SCALING LAW -- pythia ladder, 80k shared L=256 windows\n")
    hdr = (f"{'teacher':>12}{'params':>8}{'gold':>7}{'copy%':>7}{'big->T':>8}{'d-big':>7}"
           f"{'student':>9}{'core':>7}{'core/stu':>9}  rules")
    print(hdr)
    for r in rows:
        s = f"{r['tag']:>12}{r['M']:>7}M{r['gold']:>7.3f}{r['copy']:>7.1%}{r['big']:>8.3f}"
        s += f"{r['dbig']:>7.3f}"
        if "student" in r:
            s += (f"{r['student']:>9.3f}{r['core']:>7.3f}{r['core'] / r['student']:>9.1%}"
                  f"  {r['rules']}")
        print(s)
    if len(teachers) > 1:
        print("\ninter-scale decision agreement (test windows):")
        tags = [t for t, _ in MODELS if t in teachers]
        print(" " * 12 + "".join(f"{t[6:]:>8}" for t in tags))
        for a in tags:
            line = f"{a:>12}"
            for b in tags:
                agree = float((teachers[a][te] == teachers[b][te]).float().mean())
                line += f"{agree:>8.3f}"
            print(line)
    print("\nverdict axes: 'copy%' = the induction bump vs scale; 'big->T' = n-gram-shapedness vs "
          "scale; 'core' = the certified-core agreement (the headline curve); 'core/stu' = the "
          "fraction of what the student captured that crystallized. CONFOUNDS (named): fixed "
          "student capacity + fixed rule library -- a falling core curve means THIS student/library "
          "saturates, not that the model is uncertifiable; a rising or flat curve is the strong "
          "result. Tag: empirical over this ladder and these windows.")


if __name__ == "__main__":
    main()
