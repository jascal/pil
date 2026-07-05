"""Teacher harness for the race goal (rung 1).

Runs under the lm-sae venv (needs transformers): ../lm-sae/.venv/bin/python experiments/wyly_make_teacher.py

Rung 1: (a) the 0.189 anchor -- ln_f(final residual) @ lm_head on the old 12-tok dataset;
(b) pythia-70m teacher decisions for the 80k L=256 wikitext windows (runs under lm-sae venv)."""
import json
import time

import numpy as np
import torch

T0 = time.time()
PIL = "/home/allans/code/pil"
BUNDLE = "/home/allans/code/fieldrun/bundles"

# ---- (a) the anchor: decode the stored final residuals with the fieldrun bundle head ----
meta = json.loads(open(f"{BUNDLE}/pythia-70m.fieldrun.json").read())
arrs = {a["name"]: a for a in meta["arrays"]}
def arr(name):
    a = arrs[name]
    return torch.from_numpy(np.array(np.memmap(f"{BUNDLE}/pythia-70m.fieldrun.bin",
        dtype=np.float16, mode="r", offset=a["offset"], shape=tuple(a["shape"])))).float()
lm_head, gw, gb = arr("lm_head"), arr("ln_f.weight"), arr("ln_f.bias")
LAYERS = "/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad"
d = torch.load(f"{LAYERS}/layers_pythia70m.pt", map_location="cpu")
r = d["r"][:, -1, :].float()                                 # final-layer residual, last position
tgt = d["target"]
ln = torch.nn.functional.layer_norm(r, (512,), gw, gb)
acc_ln = float((ln @ lm_head.T).argmax(1).eq(tgt).float().mean())
acc_raw = float((r @ lm_head.T).argmax(1).eq(tgt).float().mean())
print(f"[{time.time()-T0:.0f}s] ANCHOR 12-tok set: ln_f decode {acc_ln:.3f} | raw decode {acc_raw:.3f} "
      "(expected ~0.189)", flush=True)

# ---- (b) teacher decisions on the L=256 wikitext windows ----
from transformers import AutoModelForCausalLM  # noqa: E402  (import after the cheap anchor pass)

m = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m", local_files_only=True,
                                         torch_dtype=torch.float32).cuda().eval()
dd = torch.load(f"{PIL}/data/wyly_nexttoken_wikitext_L256.pt", map_location="cpu")
ids, target = dd["kept_ids"], dd["target"]
dec = torch.empty(len(ids), dtype=torch.long)
with torch.no_grad():
    for i in range(0, len(ids), 128):
        b = ids[i:i + 128].cuda()
        dec[i:i + 128] = m(b, logits_to_keep=1).logits[:, -1].argmax(1).cpu()
        if i % 12800 == 0:
            print(f"[{time.time()-T0:.0f}s] {i}/{len(ids)}", flush=True)
teach_gold = float((dec == target).float().mean())
torch.save({"teacher": dec, "L": int(dd["L"])}, f"{PIL}/data/wyly_teacher_pythia70m_L256.pt")
print(f"[{time.time()-T0:.0f}s] teacher decisions saved; teacher-vs-gold top-1 at L=256: "
      f"{teach_gold:.3f}", flush=True)
