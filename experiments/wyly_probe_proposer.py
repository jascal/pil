"""10: THE ANALYSIS ROUTE AS PROPOSAL ENGINE (scoped v1). Score every teacher attention head for
TOKEN-EQUALITY attention -- the diagonal dominance of M_h = (E Wq_h)(E Wk_h)^T over a vocab
sample, the same test the battery's auto-extraction applies to the learned head -- straight from
the fieldrun bundle weights (no forward passes, no GPU). High-equality heads are the mechanism
behind copy behavior; the emitted proposals name the rule families the teacher's own structure
supports (relation / induction / pointer), for the sleep judge to verify on held-out data as
usual: analysis proposes, learning verifies.

Caveat (named): pythia applies rotary to 25% of head dims; this static score ignores position
phases, so it detects equality STRUCTURE, not offset preferences -- offset-specific proposals
need a forward probe (fieldrun-side feature, not built tonight).

Run: .venv/bin/python experiments/wyly_probe_proposer.py [model_dir] (default rosetta pythia70m)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
MD = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO.parent / "rosetta" / "models" / "pythia70m"


def main():
    meta = json.loads((MD / "bundle.fieldrun.json").read_text())
    arrs = {a["name"]: a for a in meta["arrays"]}
    buf = np.memmap(MD / "bundle.fieldrun.bin", dtype=np.uint8, mode="r")

    def load(name):
        a = arrs[name]
        dt = {"f32": np.float32, "f16": np.float16}[a["dtype"]]
        x = np.frombuffer(buf[a["offset"]:a["offset"] + a["bytes"]], dtype=dt)
        return torch.tensor(x.reshape(a["shape"]).astype(np.float32))

    emb = load("embed")
    nlayer = sum(1 for n in arrs if n.endswith(".q_proj"))
    d = emb.shape[1]
    nhead = 8 if d == 512 else max(1, d // 64)
    hd = d // nhead
    g = torch.Generator().manual_seed(0)
    smp = torch.randperm(emb.shape[0], generator=g)[:512]
    ev = emb[smp]
    ev = (ev - ev.mean(0)) / ev.std(0).clamp(min=1e-6)
    rows = []
    for li in range(nlayer):
        wq, wk = load(f"l{li}.q_proj"), load(f"l{li}.k_proj")
        for h in range(nhead):
            qh = ev @ wq[h * hd:(h + 1) * hd, :].T
            kh = ev @ wk[h * hd:(h + 1) * hd, :].T
            m = qh @ kh.T / hd ** 0.5
            diag = m.diagonal()
            off = (m.sum(1) - diag) / (m.shape[1] - 1)
            z = (diag - off).mean() / m.std().clamp(min=1e-6)
            rows.append((float(z), li, h))
    rows.sort(reverse=True)
    print(f"equality-attention head ranking ({MD.name}, {nlayer} layers x {nhead} heads; "
          f"z = diag dominance in std units):")
    for z, li, h in rows[:8]:
        print(f"  L{li}H{h}: z={z:+.2f}")
    strong = [(li, h, round(z, 2)) for z, li, h in rows if z > 1.0]
    props = {"model": MD.name, "equality_heads": strong,
             "proposals": (["relation (eq-guard)", "induction L=1..5",
                            "pointer (l,lc cells)"] if strong else []),
             "note": "static score; offset-specific proposals need a forward probe"}
    out = REPO / "data" / f"probe_proposals_{MD.name}.json"
    out.write_text(json.dumps(props, indent=1))
    print(f"\n{len(strong)} heads above z=1.0 -> proposals: {props['proposals']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
