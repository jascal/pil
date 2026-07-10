"""Band-A diagnostic: why estate2/before has a perfect feature but ~0 cover-marginal on qa3.

Pads all queries to a shared length and runs estate2 once per mode (fast).

Run: cd pil && .venv/bin/python -u experiments/diag_estate2_qa3.py
"""
from __future__ import annotations

import json
import os
import sys
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
os.environ.setdefault("WYLY_JUDGE", "cover")

import wyly_lm_v5 as v5  # noqa: E402
from pil.tokens import TokenSpace  # noqa: E402


def log(msg=""):
    print(msg, flush=True)


def main():
    DEV = v5.DEV
    log(f"device={DEV} DATA={v5.DATA}")
    ids, y, cls, uv, tr, te = v5.load_ds()
    vocab = len(uv)
    log(f"windows={len(ids)} L={ids.shape[1]} vocab={vocab}")

    ts = TokenSpace.from_file(Path(v5.TOKJ))
    raw_batches = v5.load_queries(ts, uv, cls)
    # pad all queries to max length (left-pad with first token, same as load_queries)
    maxL = max(q.shape[1] for q, _ in raw_batches)
    maxA = max(a.shape[1] for _, a in raw_batches)
    q_list, a_list = [], []
    for qids, qans in raw_batches:
        B, L = qids.shape
        if L < maxL:
            pad = qids[:, :1].expand(B, maxL - L)
            qids = torch.cat([pad, qids], 1)
        if qans.shape[1] < maxA:
            pad_a = torch.full((B, maxA - qans.shape[1]), -1, device=DEV, dtype=qans.dtype)
            qans = torch.cat([qans, pad_a], 1)
        q_list.append(qids)
        a_list.append(qans)
    qcat = torch.cat(q_list, 0)
    acat = torch.cat(a_list, 0)
    ans = acat[:, 0]
    nq = len(qcat)
    log(f"queries={nq} padded_L={maxL} ans_len={maxA}")
    # single-batch for query_agree
    batches = [(qcat, acat)]
    v5.QUERY_BATCHES = batches

    e2 = json.loads(Path(v5.ESTATE2).read_text())
    tokmap, inv_tok = {}, {}
    for i in range(vocab):
        tokmap.setdefault(ts.token_str(int(uv[i])).strip(), i)
        inv_tok[int(uv[i])] = i

    def mkset(names):
        m_ = torch.zeros(vocab, dtype=torch.bool, device=DEV)
        for nm_ in names:
            if nm_ in tokmap:
                m_[tokmap[nm_]] = True
            else:
                ft = ts.encode(" " + nm_)
                if ft and ft[0] in inv_tok:
                    m_[inv_tok[ft[0]]] = True
        return m_

    e2sets = {
        "ent": mkset(e2["entities"]), "loc": mkset(e2["locations"]),
        "obj": mkset(e2["objects"]), "move": mkset(e2["move_verbs"]),
        "take": mkset(e2["take_verbs"]), "drop": mkset(e2["drop_verbs"]),
    }
    for k, v in e2sets.items():
        toks = [ts.token_str(int(uv[i])) for i in torch.where(v)[0].tolist()]
        log(f"  set {k} ({len(toks)}): {toks}")

    # counts model
    n_cls = len(cls)
    counts = torch.zeros(vocab, n_cls, device=DEV)
    fit_n = min(20000, len(tr))
    counts.index_add_(0, ids[tr[:fit_n], -1], F.one_hot(y[tr[:fit_n]], n_cls).float())
    model = type("M", (), {"counts": counts, "rules": [], "rules2": []})()
    ok_ = torch.zeros(vocab, device=DEV)
    n_ = torch.zeros(vocab, device=DEV)
    t_ = qcat[:, -1]
    row_ = counts[t_]
    pred_ = cls[row_.argmax(1)]
    good_ = (pred_ == ans).float()
    ok_.index_add_(0, t_, good_)
    n_.index_add_(0, t_, torch.ones(nq, device=DEV))
    calib_ = torch.where(n_ >= 10, ok_ / n_.clamp(min=1), torch.full_like(ok_, -1.0))
    model.counts_calib = calib_
    v5.SWCOVER = True
    v5.STRATA = False
    for t in t_.unique()[:12]:
        log(f"  counts calib tail {ts.token_str(int(uv[int(t)]))!r}: "
            f"{float(calib_[int(t)]):.3f} n={int(n_[int(t)])}")

    idxs = torch.arange(nq, device=DEV)

    for mode in ("is", "before"):
        log(f"\n======== mode={mode} ========")
        log("  running estate2_feature...")
        f = v5.estate2_feature(qcat, e2sets, mode=mode)
        fired = f >= 0
        n_f = int(fired.sum())
        acc_f = float((f[fired] == ans[fired]).float().mean()) if n_f else 0.0
        log(f"RAW feature: fire {n_f}/{nq}={n_f/nq:.3f}  acc|fire {acc_f:.3f}  "
            f"acc_all {float((f == ans).float().mean()):.3f}")

        raw_ok = f == ans
        pr = qcat[raw_ok][:, -2] * (vocab + 1) + qcat[raw_ok][:, -1]
        if len(pr) < 30:
            log(f"  only {len(pr)} raw-correct hits — skip")
            continue
        bc = pr.bincount()
        topk = bc.argsort(descending=True)[:8]
        for ti in topk.tolist():
            if bc[ti] == 0:
                continue
            s0, s1 = ti // (vocab + 1), ti % (vocab + 1)
            log(f"  slot cand n={int(bc[ti])}: "
                f"{ts.token_str(int(uv[s0]))!r}+{ts.token_str(int(uv[s1]))!r}")
        t2 = int(bc.argmax())
        sl2 = (t2 // (vocab + 1), t2 % (vocab + 1))
        log(f"  CHOSEN slot: {ts.token_str(int(uv[sl2[0]]))!r}+"
            f"{ts.token_str(int(uv[sl2[1]]))!r}")

        def make_gate(slots):
            def e2fn(w, sets_=e2sets, mode_=mode, slots_=slots):
                f_ = v5.estate2_feature(w, sets_, mode=mode_)
                oks = torch.zeros(len(w), dtype=torch.bool, device=w.device)
                for s0, s1 in slots_:
                    oks |= (w[:, -2] == s0) & (w[:, -1] == s1)
                return torch.where(oks, f_, torch.full_like(f_, -1))
            return e2fn

        def make_conf(featfn, lasts, conf_by_v):
            B2e = vocab + 2
            keys_l, vals_l, confs_l = [], [], []
            for v_, acc_ in conf_by_v.items():
                for lt in lasts:
                    keys_l.append((v_ + 1) * B2e + lt)
                    vals_l.append(v_)
                    confs_l.append(acc_)
            tab = v5.KeyTable(torch.tensor(keys_l, device=DEV),
                              torch.tensor(vals_l, device=DEV),
                              torch.tensor(confs_l, device=DEV))

            def e2conf(w, tab_=tab, B2_=B2e, featfn_=featfn):
                f_ = featfn_(w)
                v_, c_ = tab_.lookup_conf((f_ + 1) * B2_ + w[:, -1])
                miss = f_ < 0
                return (torch.where(miss, torch.full_like(v_, -1), v_),
                        torch.where(miss, torch.full_like(c_, -1e9), c_))
            return e2conf

        def confs_from(feat, ans_):
            fired_m = feat >= 0
            out = {}
            for v_ in torch.where(e2sets["loc"])[0].tolist():
                mv_ = fired_m & (feat == v_)
                nv_ = int(mv_.sum())
                if nv_ >= 5:
                    out[v_] = float((ans_[mv_] == v_).float().mean())
                elif int(fired_m.sum()):
                    out[v_] = float((ans_[fired_m] == feat[fired_m]).float().mean())
                else:
                    out[v_] = 0.0
            return out

        def report(label, e2conf, nm):
            v5.CONF_FNS[nm] = e2conf
            pred, conf = e2conf(qcat)
            n_t = int((pred >= 0).sum())
            acc_t = float((pred[pred >= 0] == ans[pred >= 0]).float().mean()) if n_t else 0.0
            base = v5.core_cover_sw(model, [], qcat, ans, cls, idxs)
            with_e = v5.core_cover_sw(model, [(nm, lambda w, c=e2conf: c(w)[0])],
                                      qcat, ans, cls, idxs)
            qa0 = v5.query_agree(model, [], cls, batches)
            qa1 = v5.query_agree(model, [(nm, lambda w, c=e2conf: c(w)[0])], cls, batches)
            # arb
            t = qcat[:, -1]
            row = counts[t]
            mx, am = row.max(1)
            a_c = torch.where(mx >= 1, cls[am], torch.full_like(t, -1))
            c_c = torch.where(calib_[t] >= 0, calib_[t],
                              mx.float() / (row.sum(1) + v5.ALPHA))
            e2_wins = (pred >= 0) & (conf > c_c)
            lost = (pred == ans) & (pred >= 0) & (conf <= c_c) & (a_c != ans)
            log(f"{label}:")
            log(f"  table fire {n_t}/{nq}={n_t/nq:.3f} acc|fire={acc_t:.3f} "
                f"conf_mean={float(conf[pred>=0].mean()) if n_t else 0:.3f}")
            log(f"  first-token: counts={base['agree']:.3f} +e2={with_e['agree']:.3f} "
                f"marg={with_e['agree']-base['agree']:+.4f}")
            log(f"  chain: empty={qa0:.3f} +e2={qa1:.3f} marg={qa1-qa0:+.4f}")
            log(f"  arb: e2_wins={int(e2_wins.sum())} e2_right_lost_to_counts={int(lost.sum())}")
            if n_t:
                log(f"  when e2 fires: conf_e2={float(conf[pred>=0].mean()):.3f} "
                    f"conf_c={float(c_c[pred>=0].mean()):.3f}")
            return qa1 - qa0, with_e["agree"] - base["agree"]

        # single-slot (current v5)
        e2fn1 = make_gate([sl2])
        fg = e2fn1(qcat)
        n_g = int((fg >= 0).sum())
        log(f"GATED single: fire {n_g}/{nq} acc_all={float((fg==ans).float().mean()):.3f} "
            f"raw_ok&~slot={int((raw_ok & ~((qcat[:,-2]==sl2[0])&(qcat[:,-1]==sl2[1]))).sum())}")
        cf = confs_from(fg, ans)
        for v_, a_ in cf.items():
            log(f"  id-conf {ts.token_str(int(uv[v_]))!r}={a_:.3f}")
        e2conf1 = make_conf(e2fn1, {sl2[1]}, cf)
        report("SINGLE-SLOT (current)", e2conf1, f"e2/{mode}/1")

        # multi-slot: all slots with n>=30
        multi = []
        for ti in topk.tolist():
            if bc[ti] >= 30:
                multi.append((ti // (vocab + 1), ti % (vocab + 1)))
        if not multi:
            multi = [sl2]
        e2fnM = make_gate(multi)
        fM = e2fnM(qcat)
        log(f"GATED multi({len(multi)}): fire {int((fM>=0).sum())}/{nq} "
            f"acc_all={float((fM==ans).float().mean()):.3f}")
        e2confM = make_conf(e2fnM, {s[1] for s in multi}, confs_from(fM, ans))
        report(f"MULTI-SLOT n={len(multi)}", e2confM, f"e2/{mode}/M")

        # OPEN: no slot gate
        def e2fn_open(w, sets_=e2sets, mode_=mode):
            return v5.estate2_feature(w, sets_, mode=mode_)

        lasts_all = qcat[fired, -1].unique().tolist() if n_f else [sl2[1]]
        e2confO = make_conf(e2fn_open, set(lasts_all), confs_from(f, ans))
        # but open still needs feature to fire — and table key needs last token present
        report("OPEN (no slot)", e2confO, f"e2/{mode}/O")

        # FORCE: ungated feature as prediction with conf=1.0 (bypass table/slot)
        def e2conf_force(w, sets_=e2sets, mode_=mode):
            f_ = v5.estate2_feature(w, sets_, mode=mode_)
            c_ = torch.where(f_ >= 0, torch.ones(len(f_), device=w.device),
                             torch.full((len(f_),), -1e9, device=w.device))
            return f_, c_

        report("FORCE conf=1 no-slot", e2conf_force, f"e2/{mode}/F")


if __name__ == "__main__":
    main()
