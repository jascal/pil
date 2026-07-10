"""Phase A: 2-block WylyBlock prototype on standalone bAbI (SOFT=0 + WordCodec + mined queries).

Block 0 — base world fold (estate2) + local SW cover (+ counts fallback).
Block 1 — residual / composition tier: candidates that improve the *stack* cover after B0
           is frozen (kgrams, skips, or a second estate mode). Depends on B0.

Success criteria (foundation):
  - no regression vs single-block estate2 on test bench
  - per-block admit logs + stack summary
  - package-style provenance: block_id / depends_on in a small JSON report

Run:
  cd pil && .venv/bin/python -u experiments/campaign_wyly_blocks.py
  BABI_TASKS=babi2_word .venv/bin/python -u experiments/campaign_wyly_blocks.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# Standalone defaults before importing v5
os.environ.setdefault("WYLY_TAG", "standalone")
os.environ.setdefault("WYLY_LIB", "mined")
os.environ.setdefault("WYLY_SOFT", "0")
os.environ.setdefault("WYLY_ALPHABET", "word")
os.environ.setdefault("WYLY_LABELS", "corpus")
os.environ.setdefault("WYLY_CONCEPT_INIT", "random")
os.environ.setdefault("WYLY_QUERY_SOURCE", "corpus_mined")
os.environ.setdefault("WYLY_ORIGIN", "standalone")
os.environ.setdefault("WYLY_ONLINE", "1")
os.environ.setdefault("WYLY_FOLDS", "0")
os.environ.setdefault("WYLY_CONCEPTS", "0")
os.environ.setdefault("WYLY_COVER", "sw")
os.environ.setdefault("WYLY_JUDGE", "cover")
os.environ.setdefault("WYLY_VAL_REGION", "1")

import wyly_lm_v5 as v5  # noqa: E402

from pil.wyly_block import BlockStack, WylyBlock  # noqa: E402

ALPHA = v5.ALPHA
DEV = v5.DEV


def log(msg=""):
    print(msg, flush=True)


def ensure_data(ds: str):
    q = REPO / "data" / f"wyly_queries_mined_{ds}.json"
    pt = REPO / "data" / f"wyly_nexttoken_{ds}_L256.pt"
    al = REPO / "data" / f"alphabet_{ds}.json"
    if q.exists() and pt.exists() and al.exists():
        return
    log("rebuilding word data + mined queries...")
    import subprocess
    subprocess.check_call(
        [sys.executable, str(REPO / "experiments" / "build_word_babi.py")], cwd=str(REPO))


def load_task(ds: str):
    os.environ["WYLY_DS"] = ds
    # re-bind module paths for this DS (v5 already read DS at import — reload data paths)
    v5.DS = ds
    v5.DATA = REPO / "data" / f"wyly_nexttoken_{ds}_L256.pt"
    v5.ALPHABET = "word"
    v5.ALPHABET_PATH = str(REPO / "data" / f"alphabet_{ds}.json")
    v5.QUERY_SOURCE = "corpus_mined"
    v5.QUERIES = str(REPO / "data" / f"wyly_queries_mined_{ds}.json")
    v5.ESTATE2 = str(
        REPO / "data" / ("wyly_estate2_qa2.json" if "babi2" in ds else "wyly_estate2_qa3.json"))
    v5.LABELS = "corpus"
    v5.SOFT = False
    v5.QUERY_BATCHES = []
    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    codec = v5.load_codec()
    batches = v5.load_queries(codec, uv, cls)
    if v5.QUERY_PACK:
        batches = v5.pack_query_batches(batches)
    v5.QUERY_BATCHES = batches
    return ids, y, yv, cls, uv, tr, te, codec, batches


def build_counts(model_counts, cls, vocab):
    """counts_row_fn: ids -> (pred_class_space, conf) using query-calibrated counts if present."""

    def counts_row_fn(ids):
        t = ids[:, -1]
        row = model_counts[t]
        mx, am = row.max(1)
        tot = row.sum(1)
        a = torch.where(mx >= 1, cls[am], torch.full_like(t, -1))
        c = mx.float() / (tot + ALPHA)
        ccal = getattr(counts_row_fn, "calib", None)
        if ccal is not None:
            c = torch.where(ccal[t] >= 0, ccal[t], c)
        return a, c

    return counts_row_fn


def calibrate_counts(counts, cls, batches, vocab):
    ok = torch.zeros(vocab, device=DEV)
    n = torch.zeros(vocab, device=DEV)
    for qids, qans in batches:
        t = qids[:, -1]
        row = counts[t]
        pred = cls[row.argmax(1)]
        good = (pred == qans[:, 0]).float()
        ok.index_add_(0, t, good)
        n.index_add_(0, t, torch.ones(len(t), device=DEV))
    return torch.where(n >= 10, ok / n.clamp(min=1), torch.full_like(ok, -1.0))


def query_agree_rules(rules, conf_fns, counts_fn, cls, batches, stack_blocks=None):
    """Chain-scored agree on deployment queries using SW cover over given rules."""
    # reuse v5 machinery: temporarily install conf_fns
    saved = dict(v5.CONF_FNS)
    v5.CONF_FNS.clear()
    v5.CONF_FNS.update(conf_fns)
    ok = tot = 0
    for qids, qans in batches:
        w = qids
        pred = torch.full((len(w),), -1, dtype=torch.long, device=DEV)
        conf = torch.full((len(w),), -1e9, device=DEV)
        for name, fn in rules:
            if name in conf_fns:
                a, c = conf_fns[name](w)
            else:
                a = fn(w)
                c = torch.zeros(len(w), device=DEV)
            m = (a >= 0) & (c > conf)
            pred = torch.where(m, a, pred)
            conf = torch.where(m, c, conf)
        if counts_fn is not None:
            a, c = counts_fn(w)
            m = (a >= 0) & (c > conf)
            pred = torch.where(m, a, pred)
            conf = torch.where(m, c, conf)
        # first-token only for speed (answers are single word tokens)
        ok += int((pred == qans[:, 0]).sum())
        tot += len(qans)
    v5.CONF_FNS.clear()
    v5.CONF_FNS.update(saved)
    return ok / max(tot, 1)


def make_estate2_candidate(mode, e2sets, vocab, batches, fit_ids, fit_yv, uv, codec):
    """Build estate2 identity-table candidate (same logic as v5, simplified)."""
    # deployment-first slot
    prs = []
    for qids, qans in batches:
        f = v5.estate2_feature(qids, e2sets, mode=mode)
        hit = f == qans[:, 0]
        if int(hit.sum()):
            prs.append(qids[hit][:, -2] * (vocab + 1) + qids[hit][:, -1])
    if not prs:
        return None
    pr = torch.cat(prs)
    t2 = int(pr.bincount().argmax())
    sl = (t2 // (vocab + 1), t2 % (vocab + 1))

    def e2fn(w, sets=e2sets, mode=mode, sl=sl):
        f = v5.estate2_feature(w, sets, mode=mode)
        ok = (w[:, -2] == sl[0]) & (w[:, -1] == sl[1])
        return torch.where(ok, f, torch.full_like(f, -1))

    B2 = vocab + 2
    # query-calibrated identity confs
    f_all, ans_all = [], []
    for qids, qans in batches:
        f_all.append(e2fn(qids))
        ans_all.append(qans[:, 0])
    f_all = torch.cat(f_all)
    ans_all = torch.cat(ans_all)
    fired = f_all >= 0
    keys, vals, confs = [], [], []
    for v in torch.where(e2sets["loc"])[0].tolist():
        m = fired & (f_all == v)
        nv = int(m.sum())
        if nv >= 5:
            acc = float((ans_all[m] == v).float().mean())
        elif int(fired.sum()):
            acc = float((ans_all[fired] == f_all[fired]).float().mean())
        else:
            acc = 0.0
        keys.append((v + 1) * B2 + sl[1])
        vals.append(v)
        confs.append(acc)
    tab = v5.KeyTable(torch.tensor(keys, device=DEV),
                      torch.tensor(vals, device=DEV),
                      torch.tensor(confs, device=DEV))

    def conf_fn(w, tab=tab, B2=B2, feat=e2fn):
        f = feat(w)
        v, c = tab.lookup_conf((f + 1) * B2 + w[:, -1])
        miss = f < 0
        return (torch.where(miss, torch.full_like(v, -1), v),
                torch.where(miss, torch.full_like(c, -1e9), c))

    nm = f"estate2/{mode} [{len(keys)}]"
    emit = ("dfeat", ("estate2", {"sets": e2sets, "mode": mode, "slot": sl}), tab, B2)
    return nm, (lambda w: conf_fn(w)[0]), conf_fn, emit, sl


def make_kgram_candidate(ids, yv, fit, k, vocab):
    of = v5.OnlineFrame(tuple(range(k, 0, -1)), vocab, DEV, minsupp=40)
    of.update(ids[fit], yv[fit])
    nm = f"kgram k={k} (online s40)"

    def conf(w, of=of):
        return of.table.lookup_conf(of._suffix_key(w, False))

    return nm, of.mirror(), conf, ("kgram", of)


def load_estate_sets(path, vocab, uv, codec):
    e2 = json.loads(Path(path).read_text())
    tokmap, inv = {}, {}
    for i in range(vocab):
        tokmap.setdefault(codec.token_str(int(uv[i])).strip(), i)
        inv[int(uv[i])] = i

    def mkset(names):
        m = torch.zeros(vocab, dtype=torch.bool, device=DEV)
        for nm in names:
            if nm in tokmap:
                m[tokmap[nm]] = True
            else:
                ft = codec.encode(" " + nm)
                if ft and ft[0] in inv:
                    m[inv[ft[0]]] = True
        return m

    return {
        "ent": mkset(e2["entities"]),
        "loc": mkset(e2["locations"]),
        "obj": mkset(e2["objects"]),
        "move": mkset(e2["move_verbs"]),
        "take": mkset(e2["take_verbs"]),
        "drop": mkset(e2["drop_verbs"]),
    }


def bench_pred(codec, alphabet_path, pred_fn, bench_path, uv, cls):
    """pred_fn(ids_compact) -> class-space preds; map to strings via uv."""
    bench = json.loads(Path(bench_path).read_text())
    inv = {int(t): i for i, t in enumerate(uv.tolist())}
    ok = tot = 0
    for item in bench:
        raw = codec.encode(item["prompt"])
        # map to compact class space used by model
        compact = [inv[t] for t in raw if t in inv]
        if not compact:
            tot += 1
            continue
        ids = torch.tensor([compact], device=DEV)
        # pad to at least 24
        if ids.shape[1] < 24:
            pad = ids[:, :1].expand(1, 24 - ids.shape[1])
            ids = torch.cat([pad, ids], 1)
        pred = pred_fn(ids)
        p = int(pred[0])
        if p < 0:
            tot += 1
            continue
        raw_id = int(uv[p])
        pred_s = codec.token_str(raw_id).strip().lower()
        gold = item["answer"].strip().lower()
        hit = pred_s == gold or gold.startswith(pred_s) or pred_s.startswith(gold)
        ok += int(hit)
        tot += 1
    return ok / max(tot, 1), ok, tot


def run_task(ds: str) -> dict:
    ensure_data(ds)
    log(f"\n{'='*60}\n2-BLOCK PROTOTYPE × {ds}\n{'='*60}")
    ids, y, yv, cls, uv, tr, te, codec, batches = load_task(ds)
    vocab = len(uv)
    n_cls = len(cls)
    fit = tr[: max(1000, int(0.85 * len(tr)))]
    # counts from fit
    counts = torch.zeros(vocab, n_cls, device=DEV)
    counts.index_add_(0, ids[fit, -1],
                      torch.nn.functional.one_hot(y[fit], n_cls).float())
    calib = calibrate_counts(counts, cls, batches, vocab)
    counts_fn = build_counts(counts, cls, vocab)
    counts_fn.calib = calib

    e2path = v5.ESTATE2
    e2sets = load_estate_sets(e2path, vocab, uv, codec)
    modes = ["is"] if "babi2" in ds else ["is", "before"]

    # ---- Block 0: estate world fold ----
    b0 = WylyBlock(0, "world_fold", depends_on=[], device=DEV)
    for mode in modes:
        built = make_estate2_candidate(
            mode, e2sets, vocab, batches, ids, yv, uv, codec)
        if built is None:
            log(f"  skip estate2/{mode}: no slot hits")
            continue
        nm, pred_fn, conf_fn, emit, sl = built
        log(f"  B0 candidate {nm} slot={sl}")
        b0.add_feature(f"estate2_{mode}", lambda w, m=mode: v5.estate2_feature(w, e2sets, mode=m))
        b0.add_candidate(nm, pred_fn, conf_fn, emit=emit)

    def score_b0(rules):
        return query_agree_rules(rules, b0.conf_fns, counts_fn, cls, batches)

    adm0 = b0.admit_greedy(score_b0, thresh=5e-4, max_rules=4)
    log(f"  B0 admitted: {adm0}")
    base_b0 = score_b0(b0.rules)
    log(f"  B0 query agree (w/ counts): {base_b0:.3f}")

    # ---- Block 1: residual tier (depends on B0) ----
    b1 = WylyBlock(1, "residual", depends_on=[0], device=DEV)

    # feature: upstream prediction (composition hook)
    def up_pred(ids, upstream):
        if upstream is None or upstream.pred is None:
            return torch.full((len(ids),), -1, dtype=torch.long, device=ids.device)
        return upstream.pred

    b1.add_feature("b0_pred", up_pred, needs_upstream=True)

    # residual candidates: kgrams that can fill where fold is silent
    for k in (2, 3):
        if (vocab + 1) ** (k + 1) >= 2 ** 62:
            continue
        nm, pred_fn, conf_fn, emit = make_kgram_candidate(ids, yv, fit, k, vocab)
        b1.add_candidate(nm, pred_fn, conf_fn, emit=emit)

    # stack score: B0 rules + B1 trial rules, counts once
    def score_stack(b1_rules):
        rules = list(b0.rules) + list(b1_rules)
        confs = {}
        confs.update(b0.conf_fns)
        confs.update(b1.conf_fns)
        return query_agree_rules(rules, confs, counts_fn, cls, batches)

    # freeze B0; admit B1 by stack marginal
    adm1 = b1.admit_greedy(score_stack, thresh=5e-4, max_rules=4)
    log(f"  B1 admitted: {adm1}")
    stack_agree = score_stack(b1.rules)
    log(f"  stack query agree: {stack_agree:.3f} (B0-only {base_b0:.3f}, "
        f"Δ={stack_agree - base_b0:+.4f})")

    stack = BlockStack([b0, b1])

    # bench
    bench_path = REPO / "data" / (
        "babi_qa2_bench.json" if "babi2" in ds else "babi_qa3_bench.json")

    def pred_b0_only(qids):
        st = b0.forward(qids, None, counts_row_fn=counts_fn)
        return st.pred

    def pred_stack(qids):
        states = stack.forward(qids, counts_row_fn=counts_fn)
        p, _ = stack.final_pred(states)
        return p

    a0, o0, t0 = bench_pred(codec, v5.ALPHABET_PATH, pred_b0_only, bench_path, uv, cls)
    aS, oS, tS = bench_pred(codec, v5.ALPHABET_PATH, pred_stack, bench_path, uv, cls)
    log(f"  BENCH B0-only: {o0}/{t0}={a0:.3f}")
    log(f"  BENCH stack:   {oS}/{tS}={aS:.3f}")

    report = {
        "ds": ds,
        "blocks": stack.summary(),
        "b0_admit_detail": b0.last_admit[:20],
        "b1_admit_detail": b1.last_admit[:20],
        "query_agree_b0": base_b0,
        "query_agree_stack": stack_agree,
        "bench_b0": {"agree": a0, "ok": o0, "tot": t0},
        "bench_stack": {"agree": aS, "ok": oS, "tot": tS},
        "standalone": {
            "soft": False,
            "alphabet": "word",
            "labels": "corpus",
            "query_source": "corpus_mined",
            "origin": "standalone",
        },
        "delta_stack_minus_b0": aS - a0,
    }
    out = REPO / "data" / f"wyly_blocks_report_{ds}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log(f"  wrote {out}")
    return report


def main():
    tasks = [t.strip() for t in os.environ.get(
        "BABI_TASKS", "babi2_word,babi3x_word").split(",") if t.strip()]
    # rebuild once if needed
    if any(not (REPO / "data" / f"wyly_queries_mined_{t}.json").exists() for t in tasks):
        ensure_data(tasks[0])
    results = []
    for t in tasks:
        # fresh module state for QUERY_BATCHES
        v5.QUERY_BATCHES = []
        v5.CONF_FNS.clear()
        results.append(run_task(t))
    log("\n" + "=" * 60)
    log("2-BLOCK PHASE A SCOREBOARD")
    log("=" * 60)
    log(f"{'ds':14} {'B0 bench':>10} {'stack':>10} {'Δ':>8} {'B0 rules':>20} {'B1 rules':>20}")
    for r in results:
        b0r = ",".join(r["blocks"][0]["admitted"])[:20]
        b1r = ",".join(r["blocks"][1]["admitted"])[:20] if len(r["blocks"]) > 1 else ""
        log(f"{r['ds']:14} {r['bench_b0']['agree']:10.3f} {r['bench_stack']['agree']:10.3f} "
            f"{r['delta_stack_minus_b0']:+8.4f} {b0r:>20} {b1r:>20}")
    out = REPO / "data" / "wyly_blocks_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
