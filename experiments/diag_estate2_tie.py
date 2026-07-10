"""Reproduce: estate2 conf=1.0 loses to earlier conf=1.0 wrong rules (strict > tie)."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
os.environ.setdefault("WYLY_DS", "babi3x")
os.environ.setdefault("WYLY_TAG", "qwen3b")
os.environ.setdefault("WYLY_ESTATE2", str(REPO / "data" / "wyly_estate2_qa3.json"))
os.environ.setdefault("WYLY_QUERIES", str(REPO / "data" / "wyly_queries_babi_qa3.json"))
os.environ.setdefault("WYLY_TOKENIZER", str(REPO / "data" / "qwen3b.tokenizer.json"))
os.environ.setdefault("WYLY_COVER", "sw")
import wyly_lm_v5 as v5
from pil.tokens import TokenSpace

def log(m=""): print(m, flush=True)

DEV = v5.DEV
ids, y, cls, uv, tr, te = v5.load_ds()
vocab = len(uv)
ts = TokenSpace.from_file(Path(v5.TOKJ))
raw = v5.load_queries(ts, uv, cls)
maxL = max(q.shape[1] for q, _ in raw)
q_list, a_list = [], []
for qids, qans in raw:
    B, L = qids.shape
    if L < maxL:
        qids = torch.cat([qids[:, :1].expand(B, maxL - L), qids], 1)
    q_list.append(qids)
    a_list.append(qans[:, :1])
qcat = torch.cat(q_list)
ans = torch.cat(a_list)[:, 0]
nq = len(qcat)
log(f"queries={nq} L={maxL}")

e2 = json.loads(Path(v5.ESTATE2).read_text())
tokmap, inv = {}, {}
for i in range(vocab):
    tokmap.setdefault(ts.token_str(int(uv[i])).strip(), i)
    inv[int(uv[i])] = i

def mkset(names):
    m = torch.zeros(vocab, dtype=torch.bool, device=DEV)
    for nm in names:
        if nm in tokmap:
            m[tokmap[nm]] = True
        else:
            ft = ts.encode(" " + nm)
            if ft and ft[0] in inv:
                m[inv[ft[0]]] = True
    return m

e2sets = {k: mkset(e2[k2]) for k, k2 in [
    ("ent", "entities"), ("loc", "locations"), ("obj", "objects"),
    ("move", "move_verbs"), ("take", "take_verbs"), ("drop", "drop_verbs")]}

log("feature...")
f = v5.estate2_feature(qcat, e2sets, mode="before")
log(f"feature acc={float((f==ans).float().mean()):.3f} fire={int((f>=0).sum())}")
pr = qcat[f == ans][:, -2] * (vocab + 1) + qcat[f == ans][:, -1]
sl = int(pr.bincount().argmax())
sl2 = (sl // (vocab + 1), sl % (vocab + 1))
log(f"slot {[ts.token_str(int(uv[x])) for x in sl2]}")

def e2conf(w):
    ff = v5.estate2_feature(w, e2sets, mode="before")
    oks = (w[:, -2] == sl2[0]) & (w[:, -1] == sl2[1])
    ff = torch.where(oks, ff, torch.full_like(ff, -1))
    c = torch.where(ff >= 0, torch.ones(len(ff), device=DEV),
                    torch.full((len(ff),), -1e9, device=DEV))
    return ff, c

n_cls = len(cls)
counts = torch.zeros(vocab, n_cls, device=DEV)
counts.index_add_(0, ids[tr[:5000], -1], F.one_hot(y[tr[:5000]], n_cls).float())
model = type("M", (), {
    "counts": counts, "rules": [], "rules2": [],
    "counts_calib": torch.zeros(vocab, device=DEV)})()

wrong_loc = int(torch.where(e2sets["loc"])[0][0])
log(f"wrong_loc={ts.token_str(int(uv[wrong_loc]))!r}")

def bog(w):
    return (torch.full((len(w),), wrong_loc, device=DEV), torch.ones(len(w), device=DEV))

v5.CONF_FNS["bogus"] = bog
v5.CONF_FNS["e2"] = e2conf
v5.SWCOVER = True
idxs = torch.arange(nq, device=DEV)

def ag(rules):
    return v5.core_cover_sw(model, rules, qcat, ans, cls, idxs)["agree"]

log(f"e2 alone:           {ag([('e2', lambda w: e2conf(w)[0])]):.3f}")
log(f"bogus then e2:      {ag([('bogus', lambda w: bog(w)[0]), ('e2', lambda w: e2conf(w)[0])]):.3f}")
log(f"e2 then bogus:      {ag([('e2', lambda w: e2conf(w)[0]), ('bogus', lambda w: bog(w)[0])]):.3f}")

# package conf stats
man = json.load(open(REPO / "data/wyly_expert_package_v5_babi3x/manifest.json"))
mxcs = []
for r in man["rules"]:
    confs = r.get("confs")
    if confs:
        mxcs.append(max(float(x) for x in confs.values()))
if mxcs:
    log(f"package gate maxconf: n={len(mxcs)} mean={sum(mxcs)/len(mxcs):.3f} "
        f"frac==1={sum(c>=1-1e-12 for c in mxcs)/len(mxcs):.3f} "
        f"frac>=0.9={sum(c>=0.9 for c in mxcs)/len(mxcs):.3f}")
# pointer cells
for r in man["rules"]:
    if r.get("kind") == "pointer":
        log(f"pointer cells: {r.get('cells')}")
        break
# any estate2?
kinds = {}
for r in man["rules"]:
    kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
log(f"package kinds: {kinds}")
deriv = man.get("derived") or man.get("features") or []
log(f"derived count={len(deriv)} sample={deriv[:2] if deriv else None}")
