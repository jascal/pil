"""Minimal qa3 re-admit: estate2/before only, after QUERY_BATCHES load-order fix.

Pads queries to one batch (fast), admits estate2 under cover-sw, emits package, benches.

Run: cd pil && .venv/bin/python -u experiments/admit_estate2_qa3.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("WYLY_DS", "babi3x")
os.environ.setdefault("WYLY_TAG", "qwen3b")
os.environ.setdefault("WYLY_LIB", "mined")
os.environ.setdefault("WYLY_JUDGE", "cover")
os.environ.setdefault("WYLY_COVER", "sw")
os.environ.setdefault("WYLY_ESTATE2", "data/wyly_estate2_qa3.json")
os.environ.setdefault("WYLY_QUERIES", "data/wyly_queries_babi_qa3.json")
os.environ.setdefault("WYLY_TOKENIZER", "data/qwen3b.tokenizer.json")
os.environ.setdefault("WYLY_EMBED", "data/qwen3b_embed.npy")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import torch
import wyly_lm_v5 as v5
from wyly_lm_grounded import grounded_init
from wyly_lm_v3 import WylyV3

from pil.tokens import TokenSpace


def log(m=""):
    print(m, flush=True)


def pad_queries(batches, dev):
    maxL = max(q.shape[1] for q, _ in batches)
    maxA = max(a.shape[1] for _, a in batches)
    qs, ans = [], []
    for qids, qans in batches:
        B, L = qids.shape
        if L < maxL:
            qids = torch.cat([qids[:, :1].expand(B, maxL - L), qids], 1)
        if qans.shape[1] < maxA:
            qans = torch.cat([qans, torch.full((B, maxA - qans.shape[1]), -1,
                                               device=dev, dtype=qans.dtype)], 1)
        qs.append(qids)
        ans.append(qans)
    return [(torch.cat(qs), torch.cat(ans))]


def main():
    DEV = v5.DEV
    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    vocab, ell = len(uv), ids.shape[1]
    log(f"vocab={vocab} L={ell} n={len(ids)}")

    ts = TokenSpace.from_file(Path(v5.TOKJ))
    raw = v5.load_queries(ts, uv, cls)
    batches = pad_queries(raw, DEV)
    v5.QUERY_BATCHES = batches
    qcat, acat = batches[0]
    ans = acat[:, 0]
    nq = len(qcat)
    log(f"queries padded n={nq} L={qcat.shape[1]}")

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

    # deployment-first slot (the path that was dead before the load-order fix)
    mode = "before"
    f_q = v5.estate2_feature(qcat, e2sets, mode=mode)
    hq = f_q == ans
    pr = qcat[hq][:, -2] * (vocab + 1) + qcat[hq][:, -1]
    t2 = int(pr.bincount().argmax())
    sl2 = (t2 // (vocab + 1), t2 % (vocab + 1))
    log(f"slot={[ts.token_str(int(uv[x])) for x in sl2]} hits={int(hq.sum())} "
        f"raw_acc={float((f_q==ans).float().mean()):.3f}")

    def e2fn(w, sets_=e2sets, mode_=mode, sl_=sl2):
        f_ = v5.estate2_feature(w, sets_, mode=mode_)
        oks = (w[:, -2] == sl_[0]) & (w[:, -1] == sl_[1])
        return torch.where(oks, f_, torch.full_like(f_, -1))

    B2e = vocab + 2
    fg = e2fn(qcat)
    fired = fg >= 0
    vals_e = torch.where(e2sets["loc"])[0]
    keys_l, vals_l, confs_l = [], [], []
    for v_ in vals_e.tolist():
        mv = fired & (fg == v_)
        nv = int(mv.sum())
        acc = (float((ans[mv] == v_).float().mean()) if nv >= 5
               else float((ans[fired] == fg[fired]).float().mean()) if int(fired.sum()) else 0.0)
        keys_l.append((v_ + 1) * B2e + sl2[1])
        vals_l.append(v_)
        confs_l.append(acc)
        log(f"  identity {ts.token_str(int(uv[v_]))!r}: n={nv} conf={acc:.3f}")
    tab_e = v5.KeyTable(torch.tensor(keys_l, device=DEV),
                        torch.tensor(vals_l, device=DEV),
                        torch.tensor(confs_l, device=DEV))

    def e2conf(w, tab_=tab_e, B2_=B2e, featfn=e2fn):
        f_ = featfn(w)
        v_, c_ = tab_.lookup_conf((f_ + 1) * B2_ + w[:, -1])
        miss = f_ < 0
        return (torch.where(miss, torch.full_like(v_, -1), v_),
                torch.where(miss, torch.full_like(c_, -1e9), c_))

    nm = f"estate2/{mode} (world-state fold) [{len(keys_l)}]"
    v5.CONF_FNS[nm] = e2conf
    v5.EMIT_INFO[nm] = ("dfeat", ("estate2",
                        {"sets": e2sets, "mode": mode, "slot": sl2}), tab_e, B2e)
    _nkeys = len(keys_l)
    v5.RULE_SIZE[nm] = lambda n_=_nkeys: n_
    v5.SWCOVER = True
    v5.STRATA = False
    v5.FOLDS = 0

    # model: counts from train
    ground = grounded_init(uv).to(DEV)
    ground = ground / ground.shape[1] ** 0.5
    model = WylyV3(ground, cls.cpu(), ell).to(DEV)
    # fill counts from a train chunk
    ch = tr[:8000]
    model.update_counts(ids[ch], y[ch], model.lut)
    # query-calibrate counts
    ok_ = torch.zeros(vocab, device=DEV)
    n_ = torch.zeros(vocab, device=DEV)
    t_ = qcat[:, -1]
    row_ = model.counts[t_]
    pred_ = cls[row_.argmax(1)]
    good_ = (pred_ == ans).float()
    ok_.index_add_(0, t_, good_)
    n_.index_add_(0, t_, torch.ones(nq, device=DEV))
    model.counts_calib = torch.where(n_ >= 10, ok_ / n_.clamp(min=1),
                                     torch.full_like(ok_, -1.0))
    log(f"counts calib on ':': {float(model.counts_calib[int(t_[0])]):.3f}")

    cands = [(nm, lambda w: e2conf(w)[0])]
    log_lines = []
    v5.sleep_admit_cover(model, ids, yv, cls, y, tr[:3000], 0, cands, log_lines)
    for line in log_lines:
        log(line)
    log(f"admitted: {[r[0] for r in model.rules]}")

    qa0 = v5.query_agree(model, [], cls, batches)
    qa1 = v5.query_agree(model, model.rules, cls, batches)
    log(f"chain query_agree: empty={qa0:.3f} cover={qa1:.3f}")

    # emit package
    man, skipped = v5.emit_full(model, cls, uv, ts, vocab)
    pkg = REPO / "data" / "wyly_expert_package_v5_babi3x_estate2"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "manifest.json").write_text(json.dumps(man))
    shutil.copy(Path(v5.TOKJ), pkg / "bundle.tokenizer.json")
    corp = REPO / "data" / "corpus_babi3x.txt"
    if not corp.exists():
        corp = REPO / "data" / "corpus_babi_qa3.txt"
    if corp.exists():
        shutil.copy(corp, pkg / "grounding.txt")
    kinds = {}
    for r in man["rules"]:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    e2d = [d for d in man.get("derived", []) if d.get("kind") == "estate2"]
    log(f"package -> {pkg} n_rules={man['n_rules']} kinds={kinds}")
    log(f"estate2 derived: {e2d}")
    if skipped:
        log(f"skipped emit: {skipped}")

    # bench via serve_package
    sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))
    from serve_package import decide, load_package
    idi, ngr, mp_ = load_package(pkg / "manifest.json")
    bench = json.loads((REPO / "data" / "babi_qa3_bench.json").read_text())
    ok = tot = 0
    abstain = 0
    culprits = {}
    for item in bench:
        toks = ts.encode(" " + item["prompt"])
        d = decide(toks, idi, ngr, mp_)
        gold = item["answer"].strip().lower()
        if not d or d.get("answer") is None:
            abstain += 1
            tot += 1
            continue
        pred = ts.token_str(int(d["answer"])).strip().lower()
        hit = pred == gold or gold.startswith(pred) or pred.startswith(gold)
        ok += int(hit)
        tot += 1
        if not hit:
            key = d.get("circuit") or d.get("rule") or d.get("kind") or "?"
            culprits[str(key)] = culprits.get(str(key), 0) + 1
    log(f"BENCH babi_qa3: {ok}/{tot} = {ok/max(tot,1):.3f}  abstain={abstain}")
    if culprits:
        top = sorted(culprits.items(), key=lambda x: -x[1])[:8]
        log(f"  top miss sources: {top}")

    # also learner-side cover on bench prompts (tokenized like queries)
    # quick: re-use judge queries as proxy if bench is separate
    log("done.")


if __name__ == "__main__":
    main()
