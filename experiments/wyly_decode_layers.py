"""ENDGAME test: does the rule layer earn its keep as the substrate becomes LESS pre-digested?

The user cares only about the endgame -- Wyly REPLACING the transformer, computing the decode from a raw
substrate -- not the decode-adapter (which the move-1 experiments tested, and which is rigged for linear
because the FINAL residual is already optimized for linear readout). This sweeps the substrate LAYER of
pythia-70m (0 = token embedding, pure lookup, NO cross-token computation done ... final = attention has done
it all) and decodes next-token from each layer via linear / memb->V / rules->V / hybrid.

THESIS PREDICTION: rules-minus-memb ~ 0 or negative at the FINAL layer (attention already did the cross-token
work, so rules only re-encode it) but turns POSITIVE at EARLY/embedding layers (rules must actually COMPUTE
the relational structure from the raw substrate via eq_atom + composition). A positive trend toward the
embedding = rules-as-computed-core VINDICATED for the endgame; a flat/negative trend at all layers = the
rule-form limitation is layer-independent (points at a pointer/bilinear reader, not the substrate).

Run: cd pil && .venv/bin/python experiments/wyly_decode_layers.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import wyly_decode as WD  # noqa: E402
import wyly_decode_ablation as AB  # noqa: E402

K, R, STEPS = 256, 512, 4000
DATA = WD.SP / "layers_pythia70m.pt"


def main():
    d = torch.load(DATA)
    rall = d["r"].float()                                          # (n, nlayer, d)
    ids = d["kept_ids"].to(WD.DEV)
    uniq, y = d["target"].unique(return_inverse=True)
    y = y.to(WD.DEV)
    nclass, nlayer = len(uniq), d["nlayer"]
    n = rall.shape[0]
    tr = torch.arange(int(0.85 * n), device=WD.DEV)
    te = torch.arange(int(0.85 * n), n, device=WD.DEV)
    print(f"ENDGAME layer sweep -- pythia-70m {nlayer}L, FULL vocab ({nclass}), {WD.DEV}, {STEPS} steps")
    print(f"(0=embedding=raw substrate .. {nlayer - 1}=final=attention-pre-digested)\n")
    print(f"{'layer':>6}{'linear':>9}{'memb':>8}{'rules':>8}{'hybrid':>8}{'rules-memb':>12}{'memb-linear':>13}")
    trend = []
    for L in range(nlayer):
        r = rall[:, L].to(WD.DEV)
        r = (r - r.mean(0)) / (r.std(0) + 1e-6)
        lin = WD.train_top1(WD.Linear(r.shape[1], nclass), r, None, y, tr, te, steps=STEPS)
        memb = WD.train_top1(AB.MembDecode(r.shape[1], nclass), r, ids, y, tr, te, steps=STEPS)
        rules = WD.train_top1(WD.WylyDecode(r.shape[1], nclass, K, R), r, ids, y, tr, te, steps=STEPS)
        hyb = WD.train_top1(AB.HybridDecode(r.shape[1], nclass), r, ids, y, tr, te, steps=STEPS)
        tag = " (emb)" if L == 0 else (" (final)" if L == nlayer - 1 else "")
        gap1, gap2 = rules - memb, memb - lin
        print(f"{str(L) + tag:>6}{lin:>9.3f}{memb:>8.3f}{rules:>8.3f}{hyb:>8.3f}{gap1:>+12.3f}{gap2:>+13.3f}",
              flush=True)
        trend.append((L, rules - memb))
    print("\nrules-minus-memb by layer:", "  ".join(f"L{L}:{g:+.3f}" for L, g in trend))
    emb_gap, fin_gap = trend[0][1], trend[-1][1]
    print(f"\nendgame read: embedding-layer rules-memb {emb_gap:+.3f}  vs  final-layer {fin_gap:+.3f}")
    if emb_gap > fin_gap + 0.01:
        print("=> rules earn MORE decode value on the RAW (embedding) substrate than on the pre-digested")
        print("   final residual -> rules-as-computed-core VINDICATED for the endgame; move-1 null was the")
        print("   adapter regime (Grok scope caveat confirmed). Next: multi-position relational.")
    else:
        print("=> rules do NOT gain on the raw substrate either -> rule-form limit is LAYER-INDEPENDENT")
        print("   (single-eq_atom + soft-AND + linear-to-vocab), pointing at a pointer/bilinear reader, not")
        print("   the substrate. The endgame needs a richer relational reader, not just a rawer input.")


if __name__ == "__main__":
    main()
