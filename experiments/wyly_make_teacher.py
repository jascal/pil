"""Teacher harness: pythia-ladder decisions for the L=256 wikitext windows (the race + scaling goals).

Runs under the lm-sae venv (needs transformers):
    ../lm-sae/.venv/bin/python experiments/wyly_make_teacher.py --model pythia-70m
    ../lm-sae/.venv/bin/python experiments/wyly_make_teacher.py --model pythia-1.4b --dtype fp16 --batch 32

Writes data/wyly_teacher_<model>_L256.pt = {"teacher": (N,) argmax decisions, "L": 256} and prints
the teacher's gold top-1 (the capability axis of the certifiability scaling law).

Also (--anchor) the provenance check that produced the 0.244-vs-0.189 erratum: the stored residuals
in the original layers_pythia70m.pt are already post-ln_f, so the true final-residual decode on the
12-tok set is raw @ lm_head = 0.244; the arc's quoted 0.189 was a trained-probe artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO.parent / "fieldrun" / "bundles"
T0 = time.time()


def anchor():
    meta = json.loads((BUNDLE / "pythia-70m.fieldrun.json").read_text())
    arrs = {a["name"]: a for a in meta["arrays"]}

    def arr(name):
        a = arrs[name]
        return torch.from_numpy(np.array(np.memmap(
            BUNDLE / "pythia-70m.fieldrun.bin", dtype=np.float16, mode="r",
            offset=a["offset"], shape=tuple(a["shape"])))).float()

    lm_head, gw, gb = arr("lm_head"), arr("ln_f.weight"), arr("ln_f.bias")
    layers = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/"
                  "scratchpad/layers_pythia70m.pt")
    d = torch.load(layers, map_location="cpu")
    r, tgt = d["r"][:, -1, :].float(), d["target"]
    ln = torch.nn.functional.layer_norm(r, (512,), gw, gb)
    print(f"ANCHOR 12-tok: ln_f-again decode {float((ln @ lm_head.T).argmax(1).eq(tgt).float().mean()):.3f}"
          f" | raw decode {float((r @ lm_head.T).argmax(1).eq(tgt).float().mean()):.3f}"
          " (stored residuals are post-ln_f; the arc's 0.189 was a probe artifact)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-70m")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--anchor", action="store_true")
    a = ap.parse_args()
    if a.anchor:
        anchor()
    from transformers import AutoModelForCausalLM
    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{a.model}", local_files_only=True,
                                             torch_dtype=dt).cuda().eval()
    dd = torch.load(REPO / "data" / "wyly_nexttoken_wikitext_L256.pt", map_location="cpu")
    ids, target = dd["kept_ids"], dd["target"]
    dec = torch.empty(len(ids), dtype=torch.long)
    with torch.no_grad():
        for i in range(0, len(ids), a.batch):
            b = ids[i:i + a.batch].cuda()
            dec[i:i + a.batch] = m(b, logits_to_keep=1).logits[:, -1].float().argmax(1).cpu()
            if i % (a.batch * 100) == 0:
                print(f"[{time.time() - T0:5.0f}s] {a.model} {i}/{len(ids)}", flush=True)
    out = REPO / "data" / f"wyly_teacher_{a.model.replace('-', '')}_L256.pt"
    torch.save({"teacher": dec, "L": int(dd["L"]), "model": a.model, "dtype": a.dtype}, out)
    print(f"[{time.time() - T0:5.0f}s] {a.model}: gold top-1 {float((dec == target).float().mean()):.3f}"
          f" -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
