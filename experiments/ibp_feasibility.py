"""EXPLORATORY (not pre-registered): can a model-side bound through GPT-2 certify a nuisance class?

Study 2 after pil #143. Nuisance = one sentence-initial token "Word," (23 single-token options), so all
variants are
position-aligned and differ only at token 0. For 100 sigma-classes (seed 101) and k in {2, 8, 23} options:
  - IBP (experiments/ibp_gpt2.py) propagates a sound box through GPT-2 small to the decode input;
  - sanity: with k = 1 the box equals the exact float64 forward; every actual residual lies in its box
      (else abort);
  - the host-agreement half of #141's certificate is checked on (a) the IBP box, (b) the TIGHTEST box (the
      exact
    coordinate-wise range of the actual residuals -- the best any box method could produce), (c) the hull.

    /path/to/venv-with-transformers/bin/python experiments/ibp_feasibility.py --out runs/ibp/summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_substitutes as cs  # noqa: E402
from ibp_gpt2 import ibp_forward  # noqa: E402

WORDS = [
    "Yesterday",
    "Today",
    "Tonight",
    "Recently",
    "Later",
    "Earlier",
    "Again",
    "Then",
    "Meanwhile",
    "Suddenly",
    "Finally",
    "Once",
    "Now",
    "Still",
    "Soon",
    "Eventually",
    "Instead",
    "Apparently",
    "Clearly",
    "Luckily",
    "Sadly",
    "Indeed",
    "Naturally",
]
SEED, N_SIGMA, KS = 101, 100, (2, 8, 23)


def box_min(D, lo, hi):
    """min over the box [lo, hi] of D @ u, per row of D."""
    c, r = (lo + hi) / 2, (hi - lo) / 2
    return D @ c - D.abs() @ r


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    m64 = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).double().eval()
    assert all(len(tok.encode(w + ", the")) == 3 for w in WORDS), "options must be position-aligned"
    occ = [w for w in cs.OCCUPATIONS if len(tok.encode(" " + w)) == 1][:40]
    verbs = [w for w in cs.VERBS if w.endswith("ed") and len(tok.encode(" " + w)) == 1][:16]
    combos = [(s, v, o) for s in range(40) for o in range(40) if s != o for v in range(16)]
    random.Random(SEED).shuffle(combos)
    sig = combos[:N_SIGMA]
    U = m64.lm_head.weight

    def ids_for(s, v, o, words):
        txt = [
            f"{w}, the {occ[s]} {verbs[v]} the {occ[o]}. The {occ[o]} was {verbs[v]} by the" for w in words
        ]
        return torch.tensor([tok.encode(t) for t in txt], device=dev)

    ids = ids_for(*sig[0], WORDS[:1])
    lo, hi = ibp_forward(model, ids.unsqueeze(0))
    exact = m64.transformer(ids).last_hidden_state[0, -1]
    out = dict(
        tag="exploratory",
        seed=SEED,
        n_sigma=N_SIGMA,
        words=WORDS,
        k1_exactness=dict(
            max_abs_lo_minus_exact=float((lo[0] - exact).abs().max()), max_width=float((hi - lo).abs().max())
        ),
        k={},
    )
    for k in KS:
        track, cont, const, ibp_ok, box_ok, hull_ok = [], 0, 0, 0, 0, 0
        ratio_w, ratio_m = [], []
        for j, (s, v, o) in enumerate(sig):
            ids = ids_for(s, v, o, WORDS[:k])
            u = m64.transformer(ids).last_hidden_state[:, -1]
            lo, hi = ibp_forward(model, ids.unsqueeze(0), track=track if j == 0 else None)
            lo, hi = lo[0], hi[0]
            if not bool(((u >= lo - 1e-6) & (u <= hi + 1e-6)).all()):
                raise AssertionError("IBP box does not contain an actual residual: unsound (bug)")
            cont += 1
            rng_u = (u.max(0).values - u.min(0).values).clamp_min(1e-9)
            ratio_w.append(float(((hi - lo) / rng_u).median()))
            t = (u @ U.T).argmax(1)
            if not (t == t[0]).all():
                continue
            const += 1
            tt = int(t[0])
            D = U[tt] - U
            for name, (blo, bhi) in (("ibp", (lo, hi)), ("box", (u.min(0).values, u.max(0).values))):
                mins = box_min(D, blo, bhi)
                mins[tt] = 1e30
                if name == "ibp":
                    ibp_ok += bool((mins > 0).all())
                else:
                    box_ok += bool((mins > 0).all())
            hm = u @ D.T
            hm[:, tt] = 1e30
            hull_ok += bool((hm > 0).all())
            L = (u @ U.T).mean(0)
            L[tt] = -1e30
            w = U[tt] - U[int(L.argmax())]
            act = u @ w
            ratio_m.append(float(w.abs() @ (hi - lo)) / max(float(act.max() - act.min()), 1e-9))
        out["k"][str(k)] = dict(
            contained=cont,
            decision_constant=const,
            certified_ibp=ibp_ok,
            certified_tightest_box=box_ok,
            certified_hull=hull_ok,
            median_box_width_over_actual_range=float(np.median(ratio_w)),
            median_margin_range_ibp_over_actual=float(np.median(ratio_m)) if ratio_m else None,
            box_width_by_layer_class0=track,
        )
        print(json.dumps({k: out["k"][str(k)] | {"box_width_by_layer_class0": "..."}}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
