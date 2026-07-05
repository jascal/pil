"""THE RACE: analysis route vs learning route, same corpus, same held-out windows, same teacher.

Rung 3+4 of the race goal. Two independent roads to a served expert for pythia-70m:
  ANALYSIS  : rosetta's own builder (abstain_emit: build_tab + confident_rules, W=8, shipped
              defaults minsupp=3/mindet=1.0 + a relaxed point) over the SAME training corpus region
              the student saw -- the observational tier of the decompilation route. (The strong
              causal-idiom tier needs fieldrun probes and is compared qualitatively in the writeup.)
  LEARNING  : wyly-v4's self-compiled package -- online counts + the sleep-judge's admitted
              induction rules (state from wyly_lm_v4.py).
Metric: agreement with pythia-70m's DECISION on held-out windows (abstain counted as miss), plus
coverage / agreement-when-fired / gold accuracy. Rung 4: rule-space overlap -- shared bigram
contexts and their answer agreement, rule counts by suffix length, trusted tiers.

Run: cd pil && .venv/bin/python experiments/wyly_race_pythia.py   (after wyly_lm_v4.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import wyly_lm_v2 as v2
from wyly_lm_grounded import grounded_init
from wyly_lm_v3 import WylyV3, mir_induction_L
from wyly_lm_v4 import PKG, STATE, load_teacher_windows

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))
DEV = v2.DEV
W_ROS = 8


def corpus_and_starts(keep_n=80000, window=256):
    """re-derive the corpus token stream + the kept windows' start positions (extract_nexttoken's
    exact sampling), so the analysis arm trains on the same corpus region the student saw."""
    from pil.tokens import TokenSpace
    ts = TokenSpace.from_file(REPO.parent / "rosetta" / "models" / "pythia70m"
                              / "bundle.tokenizer.json")
    ids = ts.encode((REPO / "data" / "wikitext2_train.txt").read_text())
    n_wins = len(ids) - window
    pick = torch.randperm(n_wins, generator=torch.Generator().manual_seed(0))[:keep_n]
    return ids, pick.sort().values


def analysis_arm(corpus, region_end, xte_orig, teacher_te, gold_te, minsupp, mindet):
    """rosetta's builder on the shared region; evaluate its cover on the held-out windows."""
    from abstain_emit import build_tab, confident_rules, predict
    region = corpus[:region_end]
    wins = [(tuple(region[i - W_ROS:i]), region[i], i) for i in range(W_ROS, len(region))]
    tab, cites = build_tab(wins, W_ROS)
    rules = confident_rules(tab, cites, W_ROS, minsupp, mindet)
    nrules = sum(len(rules[k]) for k in rules)
    hit = agree = gagree = 0
    for i in range(len(xte_orig)):
        a = predict(tuple(xte_orig[i][-W_ROS:]), rules, W_ROS)
        if a is not None:
            hit += 1
            agree += int(a == teacher_te[i])
            gagree += int(a == gold_te[i])
    n = len(xte_orig)
    return {"rules": nrules, "cover": hit / n, "agree": agree / n,
            "agree_fired": agree / max(hit, 1), "gold": gagree / n,
            "k1_rules": {s[0]: r[0] for s, r in rules.get(1, {}).items()}}


def student_core_cover(model, admitted, ids, y, cls, idxs, mindet):
    """the learned package's cover: gated bigram -> admitted inductions (longest L first) -> abstain."""
    w = ids[idxs]
    t = w[:, -1]
    row = model.counts[t]
    mx, am = row.max(1)
    tot = row.sum(1)
    ng_ok = (mx >= 1) & (mx / tot.clamp_min(1) >= mindet)
    pred = torch.where(ng_ok, cls[am], torch.full_like(t, -1))
    for lm in sorted([r["L"] for r in admitted], reverse=True):
        a = mir_induction_L(lm)(w)
        pred = torch.where((pred == -1) & (a >= 0), a, pred)
    yv = cls[y[idxs]]
    fired = pred >= 0
    return {"rules": int((tot > 0).sum()) + len(admitted), "cover": float(fired.float().mean()),
            "agree": float((pred == yv).float().mean()),
            "agree_fired": float((pred[fired] == yv[fired]).float().mean())}


def main():
    ids, y, cls, uv, tr, te, gold = load_teacher_windows()
    ell = ids.shape[1]
    corpus, starts = corpus_and_starts()
    d = torch.load(REPO / "data" / "wyly_nexttoken_wikitext_L256.pt", map_location="cpu")
    teacher_all = torch.load(REPO / "data" / "wyly_teacher_pythia70m_L256.pt",
                             map_location="cpu")["teacher"]
    keep = torch.isin(teacher_all, torch.bincount(teacher_all).argsort(descending=True)[:v2.V])
    starts = starts[keep]
    region_end = int(starts[te[0].item()]) + ell             # everything before the 1st test window's end
    xte_orig = d["kept_ids"][keep][te.cpu()].tolist()        # original pythia token ids
    teacher_te = teacher_all[keep][te.cpu()].tolist()
    gold_te = d["target"][keep][te.cpu()].tolist()
    print(f"RACE -- {len(te)} held-out windows, shared train region = corpus[:{region_end}] "
          f"({region_end / len(corpus):.0%} of the slice)\n")

    print(f"{'arm':>44}{'rules':>8}{'cover':>8}{'agree':>8}{'when-fired':>11}")
    for ms, md in [(3, 1.0), (3, 0.5)]:
        r = analysis_arm(corpus, region_end, xte_orig, teacher_te, gold_te, ms, md)
        print(f"{f'ANALYSIS rosetta W=8 (supp>={ms}, det>={md})':>44}{r['rules']:>8}"
              f"{r['cover']:>8.1%}{r['agree']:>8.3f}{r['agree_fired']:>11.3f}", flush=True)
        if md == 1.0:
            ros_k1 = r["k1_rules"]
            ros_agree = r["agree"]

    ground = grounded_init(uv).to(DEV)
    ground = ground / ground.shape[1] ** 0.5
    model = WylyV3(ground.cpu(), cls.cpu(), ell)
    admitted = json.loads((PKG / "admitted.json").read_text())
    for rec in admitted:                                     # rebuild rule slots before load
        model.install(f"induction L={rec['L']}", mir_induction_L(rec["L"]))
    model.load_state_dict(torch.load(STATE, map_location="cpu"))
    model = model.to(DEV)
    val = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))[:v2.NVAL]]
    best = (0.0, 0.0)
    for md in [0.0, 0.3, 0.5]:                               # core operating point chosen on val
        a = student_core_cover(model, admitted, ids, y, cls, val, md)["agree"]
        if a > best[0]:
            best = (a, md)
    core = student_core_cover(model, admitted, ids, y, cls, te, best[1])
    print(f"{f'LEARNING certified core (det>={best[1]})':>44}{core['rules']:>8}"
          f"{core['cover']:>8.1%}{core['agree']:>8.3f}{core['agree_fired']:>11.3f}")
    full = v2.top1(model, ids, y, te)
    print(f"{'LEARNING full student (soft, orientation)':>44}{'-':>8}{'100%':>8}{full:>8.3f}"
          f"{full:>11.3f}")

    print(f"\nrung 3 VERDICT: learning-route certified core {core['agree']:.3f} vs analysis-route "
          f"{ros_agree:.3f} (teacher-decision agreement, abstain=miss) -> "
          f"{'LEARNING wins' if core['agree'] > ros_agree else 'ANALYSIS wins'}")

    # ---- rung 4: convergence of the two rule spaces ----
    mx, am = model.counts.max(1)
    tot = model.counts.sum(1)
    mine_k1 = {int(uv[t]): int(uv[cls[am[t]]]) for t in torch.where(tot > 0)[0].tolist()}
    shared = [c for c in ros_k1 if c in mine_k1]
    same = sum(1 for c in shared if ros_k1[c] == mine_k1[c])
    print(f"\nrung 4 -- convergence: analysis k=1 rules {len(ros_k1)}, learned bigram contexts "
          f"{len(mine_k1)}; shared contexts {len(shared)}, same answer on "
          f"{same}/{max(len(shared), 1)} ({same / max(len(shared), 1):.1%})")
    print(f"  trusted tiers: analysis (observational build) = none here (the shipped causal tier "
          f"needs fieldrun probes); learning = {[r['L'] for r in admitted]} induction rules, "
          "judge-admitted + family-certified")


if __name__ == "__main__":
    main()
