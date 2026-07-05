"""Wyly-LM v4: the self-compiler IMITATES PYTHIA -- the learning route pointed at a real transformer.

Rung 2 of the race goal. Identical machinery to wyly_lm_v3.py (wake SGD, sleep judge proposing exact
induction programs, growing package) with ONE change: the target is pythia-70m's next-token DECISION
on each window (data/wyly_teacher_pythia70m_L256.pt, produced by running the teacher once), not the
corpus continuation. Questions this answers:
  - does the judge admit DIFFERENT rules when the target is a transformer's behavior? (pythia has an
    induction bump -- copy-pattern windows where the teacher itself follows the earlier occurrence --
    so induction rules should pay MORE than on raw text);
  - how much of a real transformer's decision function can the student capture (teacher-agreement),
    and how much of THAT crystallizes into the certified package (measured in the race script).

Outputs: per-episode teacher-agreement + gold + teacher-copy-subset; the grown package
(data/wyly_expert_package_v4, citations name the teacher and the sleep cycle); saved state for
wyly_race_pythia.py.

Run: cd pil && .venv/bin/python experiments/wyly_lm_v4.py   (after make_teacher has produced the file)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wyly_lm_v2 as v2
from wyly_export_package import build_manifest
from wyly_lm_grounded import grounded_init
from wyly_lm_v3 import WylyV3, mir_induction_L, sleep_admit

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "wyly_nexttoken_wikitext_L256.pt"
_TAG = os.environ.get("WYLY_V4_TAG", "pythia70m")            # ladder runs: per-teacher artifacts
TEACH = Path(os.environ.get("WYLY_TEACHER_FILE",
                            REPO / "data" / f"wyly_teacher_{_TAG}_L256.pt"))
PKG = REPO / "data" / (f"wyly_expert_package_v4_{_TAG}" if _TAG != "pythia70m"
                       else "wyly_expert_package_v4")
STATE = REPO / "data" / (f"wyly_v4_state_{_TAG}.pt" if _TAG != "pythia70m" else "wyly_v4_state.pt")
DEV, V = v2.DEV, v2.V


def load_teacher_windows():
    """like load_windows_tied but y = the TEACHER's decision (top-V teacher classes)."""
    d = torch.load(DATA, map_location="cpu")
    ids, teacher = d["kept_ids"], torch.load(TEACH, map_location="cpu")["teacher"]
    gold = d["target"]
    keep = torch.isin(teacher, torch.bincount(teacher).argsort(descending=True)[:V])
    ids, teacher, gold = ids[keep], teacher[keep], gold[keep]
    n, ell = ids.shape
    flat = torch.cat([ids.reshape(-1), teacher])
    uv, inv = flat.unique(return_inverse=True)
    ids, yv = inv[:n * ell].reshape(n, ell), inv[n * ell:]
    cls, y = yv.unique(return_inverse=True)
    ids, y, cls = ids.to(DEV), y.to(DEV), cls.to(DEV)
    tr = torch.arange(int(0.85 * n), device=DEV)
    te = torch.arange(int(0.85 * n), n, device=DEV)
    return ids, y, cls, uv, tr, te, gold.to(DEV)


def main():
    if not TEACH.exists():
        sys.exit("run the teacher harness first (data/wyly_teacher_pythia70m_L256.pt missing)")
    ids, y, cls, uv, tr, te, gold = load_teacher_windows()
    vocab, ell = len(uv), ids.shape[1]
    cs = v2.copy_subset(ids, y, cls, te)                     # teacher-copy subset: windows where the
    print(f"Wyly-LM v4 (imitate pythia-70m) -- L={ell}, vocab {vocab}, {DEV}")
    print(f"teacher-copy subset (pythia itself follows the earlier occurrence): {len(cs)}/{len(te)} "
          f"({len(cs) / len(te):.0%}; v3's gold-copy subset was 20%)")
    # gold in class space where possible (secondary metric)
    lutv = torch.full((50304,), -1, dtype=torch.long, device=DEV)
    lutv[uv.to(DEV)] = torch.arange(vocab, device=DEV)
    gold_c = lutv[gold]                                      # compact vocab id or -1

    ground = grounded_init(uv).to(DEV)
    ground = ground / ground.shape[1] ** 0.5
    torch.manual_seed(0)
    model = WylyV3(ground, cls.cpu(), ell).to(DEV)
    lut = model.lut
    g = torch.Generator().manual_seed(0)
    val = tr[torch.randperm(len(tr), generator=g)[:v2.NVAL]]
    fit = tr[~torch.isin(tr, val)]
    candidates = [(f"induction L={lm}", mir_induction_L(lm)) for lm in (1, 2, 3)]
    sys.path.insert(0, str(REPO))
    from pil.tokens import TokenSpace
    ts = TokenSpace.from_file(REPO.parent / "rosetta" / "models" / "pythia70m"
                              / "bundle.tokenizer.json")
    admitted, timeline, log = [], [], []
    PKG.mkdir(parents=True, exist_ok=True)
    print(f"\n{'episode':>8}{'teacher-agree':>15}{'gold':>7}{'copy-agree':>12}{'tier':>6}")
    for ep, ch in enumerate(torch.chunk(fit, v2.EPISODES)):
        model.update_counts(ids[ch], y[ch], lut)
        for _ in range(v2.WAKE_STEPS):
            bi = ch[torch.randint(len(ch), (v2.BATCH,), generator=g)]
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(ids[bi]), y[bi]).backward()
            model.sgd(v2.LR)
        rec = sleep_admit(model, ids, y, val, ep, candidates, log)
        if rec:
            rec["citation"] = [rec["citation"][0].replace(
                "admitted by the sleep judge",
                "IMITATING pythia-70m decisions; admitted by the sleep judge")]
            admitted.append(rec)
        man = build_manifest(model.counts, cls, uv, ts, 1, 0.0, induction_rules=admitted,
                             model=f"wyly-v4-imitates-{_TAG}")
        (PKG / "manifest.json").write_text(json.dumps(man))
        timeline.append(len(admitted))
        with torch.no_grad():
            pred = torch.cat([model(ids[te[i:i + 192]]).argmax(1) for i in range(0, len(te), 192)])
        agree = float((pred == y[te]).float().mean())
        gacc = float((cls[pred] == gold_c[te]).float().mean())
        cagree = v2.top1(model, ids, y, cs)
        print(f"{ep:>8}{agree:>15.3f}{gacc:>7.3f}{cagree:>12.3f}{len(admitted):>6}", flush=True)
    shutil.copy(REPO.parent / "rosetta" / "models" / "pythia70m" / "bundle.tokenizer.json",
                PKG / "bundle.tokenizer.json")
    print("\n" + "\n".join(log))
    full = v2.top1(model, ids, y, te)
    norules = v2.top1(model, ids, y, te, use_rules=False)
    cs_full, cs_norules = v2.top1(model, ids, y, cs), v2.top1(model, ids, y, cs, use_rules=False)
    print(f"\nrung 2 -- the student: teacher-agreement {full:.3f} (rules ablated {norules:.3f}, "
          f"marginal {full - norules:+.3f})")
    print(f"  teacher-copy subset {cs_full:.3f} (ablated {cs_norules:.3f}, "
          f"marginal {cs_full - cs_norules:+.3f}; v3-on-gold was +0.058)")
    print(f"  admitted: {[r['L'] for r in admitted]} (tier timeline {timeline})")
    torch.save(model.state_dict(), STATE)
    (PKG / "admitted.json").write_text(json.dumps(admitted))
    print(f"state: {STATE}; package: {PKG}")


if __name__ == "__main__":
    main()
