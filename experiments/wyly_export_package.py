"""Distill Wyly-v2's CERTIFIABLE CORE into a rosetta expert package -- and price the certifiability tax.

The family-3->4 bridge, crossed from the LEARNING side for the first time: every served expert so far
came from decompiling a frozen transformer (fieldrun -> rosetta -> package -> sgiandubh). Here the
package is distilled from a LEARNED model (wyly_lm_v2.py): its online bigram count table becomes the
`ngram` gated tier, and its certified induction circuit (wyly_rel_harden.py: Soufflé-proved on its
domain) becomes the `induction` trusted kind -- both already first-class in the schema
(rosetta/PACKAGE.md + py/serve_package.py). The soft paths (grounded-linear, soft relational heads,
conjunction decode) are NOT exportable -- dropping them is the certifiability tax, measured here.

Cover semantics (mirrors serve_package.py exactly): gated n-gram (fire only if confident: support >=
minsupp AND determinism >= mindet) -> induction L=1 on n-gram miss (successor of the most recent
occurrence of the last token) -> ABSTAIN. The (minsupp, mindet) operating point is chosen on a
held-out slice of TRAIN (never test). Rung (c): certified-core top-1 on the L=256 test set, with
abstentions counted as wrong, must be >= the 0.148 Adam-bigram floor.

Also emitted for rung (d): the package directory (manifest.json + bundle.tokenizer.json), a probe set
(round-trip-stable in-scope prompts with expected answers + out-of-scope prompts that must abstain),
and the reference-decoder answers the live spoke is checked against.

Run: cd pil && .venv/bin/python experiments/wyly_lm_v2.py   (once, to produce the state)
     cd pil && .venv/bin/python experiments/wyly_export_package.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from wyly_data import load_windows_tied

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "wyly_nexttoken_wikitext_L256.pt"
STATE = REPO / "data" / "wyly_v2_state.pt"
PKG = REPO / "data" / "wyly_expert_package"
TOKENIZER = REPO.parent / "rosetta" / "models" / "pythia70m" / "bundle.tokenizer.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
V = 4096
FLOOR = 0.148                                                # Adam-bigram, same protocol (wyly_lm_v2)


def cover_eval(ids, yv, cls, counts, minsupp, mindet, idxs):
    """the package cover, vectorized: gated ngram -> induction L=1 -> abstain. -> (top1, parts)"""
    w = ids[idxs]
    t = w[:, -1]
    row = counts[t]
    mx, am = row.max(1)
    tot = row.sum(1)
    ng_ok = (mx >= minsupp) & (mx / tot.clamp_min(1) >= mindet)
    ng_pred = cls[am]
    m = w[:, :-1] == w[:, -1:]
    has = m.any(1)
    mp = (m.float() * torch.arange(1, w.shape[1], device=w.device)).argmax(1)
    ind_pred = w[torch.arange(len(w), device=w.device), mp + 1]
    pred = torch.where(ng_ok, ng_pred, torch.where(has, ind_pred, torch.full_like(t, -1)))
    correct = pred == yv[idxs]
    top1 = float(correct.float().mean())
    fired_ng = float(ng_ok.float().mean())
    fired_ind = float(((~ng_ok) & has).float().mean())
    abstain = float((pred == -1).float().mean())
    acc_ng = float(correct[ng_ok].float().mean()) if int(ng_ok.sum()) else 0.0
    ind_mask = (~ng_ok) & has
    acc_ind = float(correct[ind_mask].float().mean()) if int(ind_mask.sum()) else 0.0
    return top1, dict(ng=fired_ng, ind=fired_ind, abst=abstain, acc_ng=acc_ng, acc_ind=acc_ind)


DEFAULT_INDUCTION = [{
    "kind": "induction", "tier": "trusted", "basis": "causal", "L": 1,
    "citation": ["learned by plain SGD (pil wyly_rel_battery), hardened + Soufflé-certified "
                 "on its domain (pil wyly_rel_harden), installed in wyly_lm_v2"]}]


def build_manifest(counts, cls, uv, ts, minsupp, mindet, induction_rules=None, model="wyly-v2-learned"):
    rules, rid = [], 0
    mx, am = counts.max(1)
    tot = counts.sum(1)
    keep = torch.where((mx >= minsupp) & (mx / tot.clamp_min(1) >= mindet) & (tot > 0))[0]
    for t in keep.tolist():
        out = int(uv[cls[am[t]]])
        prev = int(uv[t])
        det = float(mx[t] / tot[t])
        rules.append({
            "id": rid, "kind": "ngram", "tier": "gated", "basis": "observational",
            "ctx": [prev], "out": out, "support": int(mx[t]), "determinism": round(det, 4),
            "citation": [f"wyly-v2 online counts: {ts.token_str(prev)!r} -> "
                         f"{ts.token_str(out)!r} (n={int(mx[t])}/{int(tot[t])})"]})
        rid += 1
    n_ng = len(rules)
    for r in (DEFAULT_INDUCTION if induction_rules is None else induction_rules):
        rules.append({"id": rid, **r})
        rid += 1
    return {"model": model, "W": 1, "trusted_idioms": 0,
            "gated_ngrams": n_ng, "induction_ood": len(rules) - n_ng,
            "minsupp": int(minsupp), "mindet": float(mindet), "n_rules": len(rules),
            "rules": rules}


def make_probes(ids, yv, cls, counts, uv, ts, te, minsupp, mindet, n_in=12, n_out=3):
    """round-trip-stable text probes + the reference-cover expected answer for each."""
    probes, seen = [], 0
    inv = {int(v): i for i, v in enumerate(uv.tolist())}
    for wi in te.tolist():
        if len(probes) >= n_in or seen > 4000:
            break
        seen += 1
        w = ids[wi]
        orig = [int(uv[t]) for t in w[-40:].tolist()]
        text = ts.decode(orig)
        if ts.encode(text) != orig:                          # BPE round-trip must be exact
            continue
        comp = torch.tensor([inv[t] for t in orig], device=ids.device).unsqueeze(0)
        top1, _ = cover_eval(comp, yv[wi:wi + 1], cls, counts, minsupp, mindet,
                             torch.tensor([0], device=ids.device))
        # recompute the cover's raw answer for the probe context
        t = comp[0, -1]
        row = counts[t]
        mx, am = row.max(0)
        ok = (mx >= minsupp) and (mx / max(float(row.sum()), 1) >= mindet)
        if ok:
            ans = int(uv[cls[am]])
        else:
            m = comp[0, :-1] == comp[0, -1]
            if bool(m.any()):
                mp = int((m.float() * torch.arange(1, comp.shape[1], device=comp.device)).argmax())
                ans = int(uv[comp[0, mp + 1]])
            else:
                continue                                     # would abstain -- not an in-scope probe
        probes.append({"text": text, "expect": ts.token_str(ans), "expect_id": ans,
                       "gold": ts.token_str(int(uv[yv[wi]]))})
    # out-of-scope = genuinely outside the expert's material: tokens NEVER seen as a predecessor in
    # training (BPE decomposes arbitrary gibberish into known subwords, so hand-typed junk is NOT oos)
    oos = []
    for t in torch.where(counts.sum(1) == 0)[0].tolist():
        s = ts.decode([int(uv[t])])
        if s.strip() and ts.encode(s) == [int(uv[t])]:
            oos.append(s)
        if len(oos) >= n_out:
            break
    return probes, oos


def main():
    if not STATE.exists():
        sys.exit("run wyly_lm_v2.py first (needs data/wyly_v2_state.pt)")
    sys.path.insert(0, str(REPO))
    from pil.tokens import TokenSpace

    ids, y, cls, uv, tr, te = load_windows_tied(V, DEV, DATA)
    yv = cls[y]                                              # targets as compact-vocab token ids
    counts = torch.load(STATE, map_location=DEV)["counts"]
    ts = TokenSpace.from_file(TOKENIZER)
    val = tr[-len(tr) // 10:]                                # held-out slice of TRAIN (temporal tail)
    print(f"certified-core export -- {len(tr)} train / {len(te)} test, vocab {len(uv)}, {DEV}")

    print(f"\n{'minsupp':>8}{'mindet':>8}{'val top-1':>11}{'ng%':>7}{'ind%':>7}{'abst%':>7}")
    best = (0.0, 1, 0.0)
    for ms in [1, 2, 3, 5]:
        for md in [0.0, 0.2, 0.3, 0.4, 0.5]:
            t1, p = cover_eval(ids, yv, cls, counts, ms, md, val)
            if t1 > best[0]:
                best = (t1, ms, md)
            print(f"{ms:>8}{md:>8.1f}{t1:>11.3f}{p['ng']:>7.1%}{p['ind']:>7.1%}{p['abst']:>7.1%}",
                  flush=True)
    _, ms, md = best
    print(f"\nchosen on val: minsupp={ms}, mindet={md}")

    t1, p = cover_eval(ids, yv, cls, counts, ms, md, te)
    raw, _ = cover_eval(ids, yv, cls, counts, 1, 0.0, te)    # ungated counts, no confidence gate
    print("\nrung (c) -- TEST, abstentions counted wrong:")
    print(f"  certified core (gated ngram + certified induction): {t1:.3f}")
    print(f"    fired: ngram {p['ng']:.1%} (acc {p['acc_ng']:.3f}) | induction {p['ind']:.1%} "
          f"(acc {p['acc_ind']:.3f}) | abstain {p['abst']:.1%}")
    print(f"  ungated counts-only reference: {raw:.3f} | Adam-bigram floor: {FLOOR} | full v2 soft: 0.199")
    print(f"  VERDICT: {'CLEARED' if t1 >= FLOOR else 'not cleared'} "
          f"({t1 - FLOOR:+.3f} vs floor); certifiability tax vs full v2 = {0.199 - t1:.3f}")

    PKG.mkdir(parents=True, exist_ok=True)
    man = build_manifest(counts, cls, uv, ts, ms, md)
    (PKG / "manifest.json").write_text(json.dumps(man))
    shutil.copy(TOKENIZER, PKG / "bundle.tokenizer.json")
    probes, oos = make_probes(ids, yv, cls, counts, uv, ts, te, ms, md)
    (PKG / "probes.json").write_text(json.dumps({"in_scope": probes, "out_of_scope": oos}, indent=1))
    print(f"\npackage written: {PKG}")
    print(f"  {man['gated_ngrams']} gated ngram rules + 1 certified induction rule; "
          f"{len(probes)} in-scope probes + {len(oos)} out-of-scope")
    print("serve:  cd sgiandubh && ./build.sh && ./build/sgiandubh --rosetta-package "
          f"{PKG} <port>   (REPL: type a probe line)")


if __name__ == "__main__":
    main()
