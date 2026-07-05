"""Wyly-LM v5: the EXTENDED rule library -- how far does the certified tier reach on a fixed dataset?

Phase 1 of the library/dataset-ladder goal. Same self-compiling loop as wyly_lm_v4.py (wake SGD on
teacher decisions, sleep judge admits exact programs by held-out val marginal), with the candidate
library enlarged from {induction L=1,2,3} to:

  induction L=1..5   : the package runtime's own kind (suffix match -> most recent -> successor);
  kgram k=2,3        : suffix-table rules fit ONCE on the train region (support>=2, det>=0.5
                       pre-gate; stored as sorted-key tensors, mirror = searchsorted) -- the native
                       `ngram` kind at ctx length k;
  skip-bigram o=2,3  : token at offset o -> most frequent teacher decision (the package `gate` kind
                       with an EMPTY frame and slot=o);
  repetition         : if the last two tokens are equal, predict that token again (a relational
                       eq+copy rule; NO package kind exists -- named as a schema design direction).

Env: WYLY_TAG (teacher tag, default pythia70m), WYLY_DS (dataset tag, default wikitext ->
data/wyly_nexttoken_wikitext_L256.pt + wyly_teacher_<tag>_L256.pt legacy names; other tags use
wyly_nexttoken_<ds>_L256.pt + wyly_teacher_<tag>_<ds>_L256.pt), WYLY_LIB (base|ext, default ext).

Run: cd pil && WYLY_TAG=pythia70m .venv/bin/python experiments/wyly_lm_v5.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wyly_lm_v2 as v2
from wyly_lm_grounded import grounded_init
from wyly_lm_v3 import WylyV3, mir_induction_L

REPO = Path(__file__).resolve().parent.parent
TAG = os.environ.get("WYLY_TAG", "pythia70m")
DS = os.environ.get("WYLY_DS", "wikitext")
LIB = os.environ.get("WYLY_LIB", "ext")
JUDGE = os.environ.get("WYLY_JUDGE", "soft")                 # soft = sec-8.6 baseline; cover = design dir 1
ONLINE = os.environ.get("WYLY_ONLINE", "") == "1"            # design dir 5: online k-gram tiers
SWCOVER = os.environ.get("WYLY_COVER", "") == "sw"           # design dir 2: support-weighted cover
EMIT = os.environ.get("WYLY_EMIT", "") == "1"                # emit the rosetta package (relation kind)
ALPHA = 2.0                                                  # Laplace shrinkage for confidence
DATA = REPO / "data" / (f"wyly_nexttoken_{DS}_L256.pt" if DS != "wikitext"
                        else "wyly_nexttoken_wikitext_L256.pt")
TEACH = REPO / "data" / (f"wyly_teacher_{TAG}_{DS}_L256.pt" if DS != "wikitext"
                         else f"wyly_teacher_{TAG}_L256.pt")
STATE = REPO / "data" / (f"wyly_v5_{LIB}_{TAG}_{DS}{'_cov' if JUDGE == 'cover' else ''}"
                         f"{'_ol' if ONLINE else ''}{'_sw' if SWCOVER else ''}.pt")
ADMIT_F = REPO / "data" / f"wyly_v5_{LIB}_{TAG}_{DS}{'_cov' if JUDGE == 'cover' else ''}_admitted.json"
DEV, V = v2.DEV, v2.V


def load_ds():
    d = torch.load(DATA, map_location="cpu")
    ids, teacher = d["kept_ids"], torch.load(TEACH, map_location="cpu")["teacher"]
    keep = torch.isin(teacher, torch.bincount(teacher).argsort(descending=True)[:V])
    ids, teacher = ids[keep], teacher[keep]
    n, ell = ids.shape
    flat = torch.cat([ids.reshape(-1), teacher])
    uv, inv = flat.unique(return_inverse=True)
    ids, yv = inv[:n * ell].reshape(n, ell), inv[n * ell:]
    cls, y = yv.unique(return_inverse=True)
    ids, y, cls = ids.to(DEV), y.to(DEV), cls.to(DEV)
    tr = torch.arange(int(0.85 * n), device=DEV)
    te = torch.arange(int(0.85 * n), n, device=DEV)
    return ids, y, cls, uv, tr, te


class KeyTable:
    """exact suffix/slot table as (sorted keys, values[, confidences]); mirror = searchsorted."""

    def __init__(self, keys, vals, conf=None):
        order = keys.argsort()
        self.k, self.v = keys[order], vals[order]
        self.c = conf[order] if conf is not None else None

    def lookup(self, q):
        if len(self.k) == 0:
            return torch.full_like(q, -1)
        i = torch.searchsorted(self.k, q).clamp(max=len(self.k) - 1)
        hit = self.k[i] == q
        return torch.where(hit, self.v[i], torch.full_like(q, -1))

    def lookup_conf(self, q):
        """(val, conf) -- conf = Laplace-shrunk determinism of the fired key; -inf on miss."""
        if len(self.k) == 0 or self.c is None:
            return torch.full_like(q, -1), torch.full((len(q),), -1e9, device=q.device)
        i = torch.searchsorted(self.k, q).clamp(max=len(self.k) - 1)
        hit = self.k[i] == q
        val = torch.where(hit, self.v[i], torch.full_like(q, -1))
        conf = torch.where(hit, self.c[i], torch.full((len(q),), -1e9, device=q.device))
        return val, conf


def best_per_key(key, val, minsupp, mindet):
    """most frequent val per key with support/determinism gating -> KeyTable."""
    pair, cnt = torch.stack([key, val]).unique(dim=1, return_counts=True)
    uk, kinv = pair[0].unique(return_inverse=True)
    tot = torch.zeros(len(uk), device=key.device).index_add_(0, kinv, cnt.float())
    comp = kinv * (int(cnt.max()) + 1) + cnt                 # sort by key, best count first
    order = comp.argsort(descending=True)
    first = torch.ones(len(order), dtype=torch.bool, device=key.device)
    first[1:] = kinv[order][1:] != kinv[order][:-1]
    sel = order[first]
    keep = (cnt[sel] >= minsupp) & (cnt[sel].float() / tot[kinv[sel]].clamp(min=1) >= mindet)
    conf = cnt[sel].float() / (tot[kinv[sel]] + ALPHA)
    return KeyTable(pair[0][sel][keep], pair[1][sel][keep], conf[keep]), int(keep.sum())


class OnlineFrame:
    """design direction 5: an ONLINE k-gram tier -- the same suffix-frame table as fit_frame, but
    its exact pair counts merge in from every wake chunk (like the bigram counts buffer) and the
    gated lookup table is refreshed at each sleep. The admitted rule is the TIER; its content grows
    with the stream, so fit-once staleness (and the fit-once-vs-online confound) disappears."""

    def __init__(self, offs, vocab, dev, minsupp=40, mindet=0.5):
        # minsupp=40: sliding counts across stride-1-overlapping windows repeat each corpus pair
        # ~21x, so 40 sliding counts ~ 2 real corpus occurrences (the fit-once tables' gate)
        self.offs, self.base = offs, vocab + 1
        self.minsupp, self.mindet = minsupp, mindet
        self.pk = torch.empty(0, dtype=torch.long, device=dev)   # pairkey = suffixkey*base + next
        self.pc = torch.empty(0, dtype=torch.long, device=dev)
        self.table = KeyTable(self.pk.clone(), self.pk.clone())

    def _suffix_key(self, ids, sliding):
        if sliding:
            maxo, ell = max(self.offs), ids.shape[1]
            key = torch.zeros(len(ids) * (ell - maxo), dtype=torch.long, device=ids.device)
            for o in self.offs:
                key = key * self.base + ids[:, maxo - o:ell - o].reshape(-1)
            return key
        key = torch.zeros(len(ids), dtype=torch.long, device=ids.device)
        for o in self.offs:
            key = key * self.base + ids[:, -o]
        return key

    def update(self, w, ytail):
        maxo = max(self.offs)
        key = self._suffix_key(w, True)
        nxt = w[:, maxo:].reshape(-1)
        pair = torch.cat([key * self.base + nxt,
                          self._suffix_key(w, False) * self.base + ytail])
        pk, pc = pair.unique(return_counts=True)
        allk = torch.cat([self.pk, pk])
        allc = torch.cat([self.pc, pc])
        self.pk, inv = allk.unique(return_inverse=True)
        self.pc = torch.zeros(len(self.pk), dtype=torch.long,
                              device=w.device).index_add_(0, inv, allc)

    def refresh(self):
        if not len(self.pk):
            return
        key, val, cnt = self.pk // self.base, self.pk % self.base, self.pc
        uk, kinv = key.unique(return_inverse=True)
        tot = torch.zeros(len(uk), device=key.device).index_add_(0, kinv, cnt.float())
        comp = kinv * (int(cnt.max()) + 1) + cnt
        order = comp.argsort(descending=True)
        first = torch.ones(len(order), dtype=torch.bool, device=key.device)
        first[1:] = kinv[order][1:] != kinv[order][:-1]
        sel = order[first]
        keep = (cnt[sel] >= self.minsupp) & (cnt[sel].float() / tot[kinv[sel]].clamp(min=1)
                                             >= self.mindet)
        conf = cnt[sel].float() / (tot[kinv[sel]] + ALPHA)
        self.table = KeyTable(key[sel][keep], val[sel][keep], conf[keep])

    def mirror(self):
        def fn(ids):
            return self.table.lookup(self._suffix_key(ids, False))
        return fn


def fit_frame(ids, yv, tr, offs, vocab, stride=12, minsupp=2, mindet=0.5):
    """tokens at OFFSETS (1-based from the end, e.g. (5,1) = a gapped frame) -> most frequent
    TEACHER decision; fit from within-window sliding positions + the window tail. offs=(k..1)
    contiguous reproduces the k-gram; anything else is a `gate`-kind frame."""
    w = ids[tr[::stride]]
    ell = w.shape[1]
    base = vocab + 1
    maxo = max(offs)
    key = torch.zeros((len(w) * (ell - maxo),), dtype=torch.long, device=ids.device)
    for o in offs:
        key = key * base + w[:, maxo - o:ell - o].reshape(-1)
    nxt = w[:, maxo:].reshape(-1)
    tail = torch.zeros(len(tr), dtype=torch.long, device=ids.device)
    for o in offs:
        tail = tail * base + ids[tr][:, -o]
    return best_per_key(torch.cat([key, tail]), torch.cat([nxt, yv[tr]]), minsupp, mindet)


def mir_frame(table, offs, vocab):
    base = vocab + 1

    def fn(ids):
        q = torch.zeros(len(ids), dtype=torch.long, device=ids.device)
        for o in offs:
            q = q * base + ids[:, -o]
        return table.lookup(q)
    return fn


def fit_kgram(ids, yv, tr, k, vocab, stride=12, minsupp=2, mindet=0.5):
    return fit_frame(ids, yv, tr, tuple(range(k, 0, -1)), vocab, stride, minsupp, mindet)


def mir_kgram(table, k, vocab):
    base = vocab + 1

    def fn(ids):
        q = torch.zeros(len(ids), dtype=torch.long, device=ids.device)
        for i in range(k):
            q = q * base + ids[:, -k + i]
        return table.lookup(q)
    return fn


def fit_skip(ids, yv, tr, off):
    """token at offset -off -> most frequent teacher decision (package `gate`, empty frame)."""
    return best_per_key(ids[tr][:, -off], yv[tr], 3, 0.3)


def mir_skip(table, off):
    def fn(ids):
        return table.lookup(ids[:, -off])
    return fn


CONF_FNS = {}                                            # name -> (ids)->(val, per-key conf)
RULE_CONF = {}                                           # name -> scalar val fired-accuracy


def register_conf(name, table, keyfn):
    def cf(ids):
        return table_ref[0].lookup_conf(keyfn(ids))
    table_ref = [table]
    CONF_FNS[name] = cf
    return table_ref


def keyfn_offsets(offs, vocab):
    base = vocab + 1

    def kf(ids):
        q = torch.zeros(len(ids), dtype=torch.long, device=ids.device)
        for o in offs:
            q = q * base + ids[:, -o]
        return q
    return kf


def mir_repetition(ids):
    same = ids[:, -1] == ids[:, -2]
    return torch.where(same, ids[:, -1], torch.full_like(ids[:, -1], -1))


def mir_relation(i, j, k):
    """the RELATION kind's mirror (schema: eq-guard + copy): ctx[-i]==ctx[-j] -> ctx[-k]."""
    def fn(ids):
        same = ids[:, -i] == ids[:, -j]
        return torch.where(same, ids[:, -k], torch.full_like(ids[:, -1], -1))
    return fn


RELATION_GRID = [(1, 2, 1), (1, 3, 1), (2, 3, 2)]


def core_cover_sw(model, rules, ids, yv, cls, idxs, extra_conf=None, return_pred=False):
    """SUPPORT-WEIGHTED cover (design dir 2): every applicable rule fires; the answer with the
    highest confidence wins. Table rules use per-key Laplace-shrunk determinism cnt/(tot+ALPHA);
    the counts tier likewise; scalar rules (induction etc.) use their val fired-accuracy
    (RULE_CONF, refreshed each sleep). Host-side and exact -- the arbitration uses only stats the
    manifest already ships (+ one scalar confidence field for non-table kinds)."""
    w = ids[idxs]
    pred = torch.full_like(w[:, -1], -1)
    conf = torch.full((len(w),), -1e9, device=w.device)

    def consider(a, c):
        nonlocal pred, conf
        m = (a >= 0) & (c > conf)
        pred = torch.where(m, a, pred)
        conf = torch.where(m, c, conf)

    ec = extra_conf or {}
    for name, fn in rules:
        if name in CONF_FNS:
            a, c = CONF_FNS[name](w)
        else:
            a = fn(w)
            c = torch.full((len(w),), float(ec.get(name, RULE_CONF.get(name, 0.0))),
                           device=w.device)
        consider(a, c)
    t = w[:, -1]
    row = model.counts[t]
    mx, am = row.max(1)
    tot = row.sum(1)
    a = torch.where(mx >= 1, cls[am], torch.full_like(t, -1))
    consider(a, mx.float() / (tot + ALPHA))
    fired = pred >= 0
    out = {"agree": float((pred == yv[idxs]).float().mean()),
           "cover": float(fired.float().mean()),
           "agree_fired": float((pred[fired] == yv[idxs][fired]).float().mean())}
    return (out, pred) if return_pred else out


def val_fired_acc(fn, ids, yv, val):
    a = fn(ids[val])
    f = a >= 0
    return float((a[f] == yv[val][f]).float().mean()) if int(f.sum()) else 0.0


class MinedGates:
    """the MINED anchored-frame family (design dir 3, second half): {offset: token} frames --
    singleton and 2-anchor CONJUNCTIONS -- discovered from the student's residual errors under the
    arbitrated cover, interaction-scored (a frame is accepted by the error-recovery of its whole
    slot-table slice, not by marginals). Frames are end-relative, so tables fit on window TAILS
    (no sliding-overlap inflation); accepted content merges into one growing confidence-carrying
    store with online-tier semantics. The library stops being hand-designed here: offsets, anchors
    and conjunctions all come from the data."""

    def __init__(self, vocab, dev, offs=(2, 3, 4, 5, 6, 7, 8),
                 pair_offs=((2, 3), (2, 4), (3, 4), (2, 5))):
        self.B, self.dev = vocab + 1, dev
        self.offs, self.pair_offs = list(offs), list(pair_offs)
        self.mined1, self.mined2 = set(), set()
        e = torch.empty(0, dtype=torch.long, device=dev)
        self.t1 = KeyTable(e, e.clone(), torch.empty(0, device=dev))     # (o,a,last) singleton
        self.t2 = KeyTable(e.clone(), e.clone(), torch.empty(0, device=dev))  # 2-anchor frames
        self.n_frames = 0

    def _merge(self, tab, keys, vals, confs):
        k = torch.cat([tab.k, keys])
        v = torch.cat([tab.v, vals])
        c = torch.cat([tab.c, confs])
        return KeyTable(k, v, c)

    def mine(self, model, ids, yv, cls, fitset, g, log, cycle, sample_n=6000):
        smp = fitset[torch.randint(len(fitset), (sample_n,), generator=g)]
        _, pred = core_cover_sw(model, model.rules, ids, yv, cls, smp, return_pred=True)
        err = smp[pred != yv[smp]]
        if len(err) < 50:
            return
        tails, ytail = ids[fitset], yv[fitset]
        new1 = new2 = 0
        for o in self.offs:                                  # singleton frames {o: a}
            tab, _ = best_per_key(tails[:, -o] * self.B + tails[:, -1], ytail, 3, 0.4)
            v, c = tab.lookup_conf(ids[err][:, -o] * self.B + ids[err][:, -1])
            good = (v == yv[err]).float()
            ua, inv = ids[err][:, -o].unique(return_inverse=True)
            rec = torch.zeros(len(ua), device=self.dev).index_add_(0, inv, good)
            cnt = torch.zeros(len(ua), device=self.dev).index_add_(0, inv,
                                                                   torch.ones_like(good))
            for a in ua[(cnt >= 15) & (rec / cnt.clamp(min=1) >= 0.25)].tolist():
                if (o, a) in self.mined1:
                    continue
                self.mined1.add((o, a))
                sl = (tab.k // self.B) == a
                self.t1 = self._merge(self.t1, (o * self.B + a) * self.B + tab.k[sl] % self.B,
                                      tab.v[sl], tab.c[sl])
                new1 += 1
        v1, _ = self.lookup_conf_all(ids[err])               # residual errors after singletons
        res = err[v1 != yv[err]]
        if len(res) >= 50:
            for o1, o2 in self.pair_offs:                    # 2-anchor conjunctions {o1:a1, o2:a2}
                key_fit = ((tails[:, -o1] * self.B + tails[:, -o2]) * self.B + tails[:, -1])
                tab, _ = best_per_key(key_fit, ytail, 3, 0.5)
                ke = (ids[res][:, -o1] * self.B + ids[res][:, -o2]) * self.B + ids[res][:, -1]
                v, c = tab.lookup_conf(ke)
                good = (v == yv[res]).float()
                pk = ids[res][:, -o1] * self.B + ids[res][:, -o2]
                ua, inv = pk.unique(return_inverse=True)
                rec = torch.zeros(len(ua), device=self.dev).index_add_(0, inv, good)
                cnt = torch.zeros(len(ua), device=self.dev).index_add_(0, inv,
                                                                       torch.ones_like(good))
                oid = o1 * 16 + o2
                for ab in ua[(cnt >= 10) & (rec / cnt.clamp(min=1) >= 0.3)].tolist():
                    if (oid, ab) in self.mined2:
                        continue
                    self.mined2.add((oid, ab))
                    sl = (tab.k // self.B) == ab
                    self.t2 = self._merge(self.t2, (oid * self.B ** 2 + ab) * self.B
                                          + tab.k[sl] % self.B, tab.v[sl], tab.c[sl])
                    new2 += 1
        self.n_frames = len(self.mined1) + len(self.mined2)
        if new1 or new2:
            log.append(f"    sleep {cycle}: MINED {new1} singleton + {new2} conjunction frames "
                       f"(total {self.n_frames}; from {len(err)} errors)")

    def lookup_conf_all(self, w):
        best_v = torch.full((len(w),), -1, dtype=torch.long, device=w.device)
        best_c = torch.full((len(w),), -1e9, device=w.device)
        for o in self.offs:
            v, c = self.t1.lookup_conf((o * self.B + w[:, -o]) * self.B + w[:, -1])
            m = (v >= 0) & (c > best_c)
            best_v, best_c = torch.where(m, v, best_v), torch.where(m, c, best_c)
        for o1, o2 in self.pair_offs:
            k = ((o1 * 16 + o2) * self.B ** 2 + w[:, -o1] * self.B + w[:, -o2]) * self.B + w[:, -1]
            v, c = self.t2.lookup_conf(k)
            m = (v >= 0) & (c > best_c)
            best_v, best_c = torch.where(m, v, best_v), torch.where(m, c, best_c)
        return best_v, best_c

    def mirror(self):
        def fn(ids):
            return self.lookup_conf_all(ids)[0]
        return fn


def sleep_admit_v5(model, ids, y, val, cycle, candidates, log, thresh=0.002):
    """the judge, library-agnostic: temp-install each unadmitted candidate; admit the best payer."""
    base = v2.top1(model, ids, y, val)
    best = (thresh, None)
    for name, fn in candidates:
        if any(n == name for n, _ in model.rules):
            continue
        model.install(name, fn)
        marg = v2.top1(model, ids, y, val) - base
        model.rules.pop()
        model.rw = torch.nn.ParameterList(list(model.rw)[:-1])
        log.append(f"    sleep {cycle}: candidate {name} val marginal {marg:+.4f}")
        if marg > best[0]:
            best = (marg, (name, fn))
    if best[1]:
        name, fn = best[1]
        model.install(name, fn)
        log.append(f"    sleep {cycle}: ADMITTED {name} ({best[0]:+.4f})")
        return name
    return None


def sleep_admit_cover(model, ids, yv, cls, y, val, cycle, candidates, log, thresh=5e-4):
    """COVER-AWARE admission (design direction 1): a candidate is scored by its marginal to the
    PACKAGE COVER on held-out val -- the objective the runtime actually realizes -- not by its vote
    in the soft mixture. Rules that merely displace an equally good lower tier score ~0 and are
    rejected; rules still install into the soft model as votes once admitted (training benefit)."""
    cover = core_cover_sw if SWCOVER else core_cover
    base = cover(model, model.rules, ids, yv, cls, val)["agree"]
    best = (thresh, None)
    for name, fn in candidates:
        if any(n == name for n, _ in model.rules):
            continue
        ec = ({name: val_fired_acc(fn, ids, yv, val)}
              if SWCOVER and name not in CONF_FNS and name not in RULE_CONF else None)
        kw = {"extra_conf": ec} if SWCOVER else {}
        marg = cover(model, model.rules + [(name, fn)], ids, yv, cls, val, **kw)["agree"] - base
        log.append(f"    sleep {cycle}: candidate {name} COVER marginal {marg:+.4f}")
        t = 0.002 if name.startswith("gate") else thresh     # gate tables are high-variance
        if marg > t and marg > best[0]:
            best = (marg, (name, fn))
    if best[1]:
        name, fn = best[1]
        model.install(name, fn)
        log.append(f"    sleep {cycle}: ADMITTED {name} (cover {best[0]:+.4f})")
        return name
    return None


def core_cover(model, rules, ids, yv, cls, idxs, return_pred=False):
    """the PACKAGE cover with the admitted library, runtime order: trusted gates (skip) -> ngrams
    longest-suffix (kgram k desc, then the online counts as k=1) -> induction L desc -> abstain."""
    w = ids[idxs]
    pred = torch.full_like(w[:, -1], -1)

    def take(fn):
        nonlocal pred
        a = fn(w)
        pred = torch.where((pred == -1) & (a >= 0), a, pred)

    for name, fn in rules:
        if name.startswith("gate") or name.startswith("skip"):   # TRUSTED tier first
            take(fn)
    for k in (3, 2):
        for name, fn in rules:
            if name.startswith(f"kgram k={k}"):
                take(fn)
    t = w[:, -1]
    row = model.counts[t]
    mx, am = row.max(1)
    cnt_pred = torch.where(mx >= 1, cls[am], torch.full_like(t, -1))
    pred = torch.where(pred == -1, cnt_pred, pred)
    for lm in (5, 4, 3, 2, 1):
        for name, fn in rules:
            if name == f"induction L={lm}":
                take(fn)
    fired = pred >= 0
    out = {"agree": float((pred == yv[idxs]).float().mean()),
           "cover": float(fired.float().mean()),
           "agree_fired": float((pred[fired] == yv[idxs][fired]).float().mean())}
    return (out, pred) if return_pred else out


def propose_gates(model, gate_cands, ids, yv, cls, sample, log, cycle, top=2):
    """the LEARNED FRAME PROPOSER: rank unadmitted gate candidates by how much of the CURRENT
    cover's error set they would correct (interaction-scored selection from data, not enumeration
    order); only the top picks go to the judge."""
    _, pred = core_cover(model, model.rules, ids, yv, cls, sample, return_pred=True)
    err = pred != yv[sample]
    scored = []
    for name, fn in gate_cands:
        if any(n == name for n, _ in model.rules):
            continue
        a = fn(ids[sample])
        rec = float(((a == yv[sample]) & err).float().mean())
        scored.append((rec, name, fn))
    scored.sort(reverse=True)
    if scored:
        log.append(f"    sleep {cycle}: proposer ranks " +
                   ", ".join(f"{n.split(' [')[0]}={r:.4f}" for r, n, _ in scored[:4]))
    return [(n, f) for _, n, f in scored[:top]]


def main():
    ids, y, cls, uv, tr, te = load_ds()
    yv = cls[y]
    vocab, ell = len(uv), ids.shape[1]
    cs = v2.copy_subset(ids, y, cls, te)
    print(f"Wyly-LM v5 [{LIB}/{JUDGE}] -- teacher {TAG}, dataset {DS}, L={ell}, vocab {vocab}, {DEV}; "
          f"copy-subset {len(cs)}/{len(te)} ({len(cs) / len(te):.0%})")

    g = torch.Generator().manual_seed(0)
    val = tr[torch.randperm(len(tr), generator=g)[:v2.NVAL]]
    fit = tr[~torch.isin(tr, val)]                           # tables are fit on FIT ONLY -- fitting on
    online_frames = []
    candidates = [(f"induction L={lm}", mir_induction_L(lm)) for lm in (1, 2, 3)]
    if LIB == "ext":                                         # tr (which contains val) leaks the val
        candidates += [(f"induction L={lm}", mir_induction_L(lm)) for lm in (4, 5)]
        for k in (2, 3):                                     # windows' own suffix pairs into the
            if ONLINE:                                       # judge's val marginals
                for supp in (40,):                           # s40 ~ 2 corpus occurrences under the
                    of = OnlineFrame(tuple(range(k, 0, -1)), vocab, DEV, minsupp=supp)
                    online_frames.append(of)                 # ~21x window overlap; offering s2 too
                    nm = f"kgram k={k} (online s{supp})"     # re-admits val-noise
                    candidates.append((nm, of.mirror()))
                    CONF_FNS[nm] = (lambda ids, of=of:
                                    of.table.lookup_conf(of._suffix_key(ids, False)))
            else:
                tab, nk = fit_kgram(ids, yv, fit, k, vocab)
                nm = f"kgram k={k} [{nk}]"
                candidates.append((nm, mir_kgram(tab, k, vocab)))
                register_conf(nm, tab, keyfn_offsets(tuple(range(k, 0, -1)), vocab))
        for off in (2, 3):
            tab, nk = fit_skip(ids, yv, fit, off)
            nm = f"skip o={off} [{nk}]"
            candidates.append((nm, mir_skip(tab, off)))
            register_conf(nm, tab, keyfn_offsets((off,), vocab))
        candidates.append(("repetition", mir_repetition))
    if LIB == "mined":                                       # design dir 3 second half: mined frames
        candidates += [(f"induction L={lm}", mir_induction_L(lm)) for lm in (4, 5)]
        for k in (2, 3):
            for supp in (40,):
                of = OnlineFrame(tuple(range(k, 0, -1)), vocab, DEV, minsupp=supp)
                online_frames.append(of)
                nm = f"kgram k={k} (online s{supp})"
                candidates.append((nm, of.mirror()))
                CONF_FNS[nm] = (lambda ids, of=of:
                                of.table.lookup_conf(of._suffix_key(ids, False)))
        for off in (2, 3):
            tab, nk = fit_skip(ids, yv, fit, off)
            nm = f"skip o={off} [{nk}]"
            candidates.append((nm, mir_skip(tab, off)))
            register_conf(nm, tab, keyfn_offsets((off,), vocab))
        for i, j, k in RELATION_GRID:
            candidates.append((f"relation eq({i},{j})c{k}", mir_relation(i, j, k)))
        mined = MinedGates(vocab, DEV)
        candidates.append(("mined frames (online)", mined.mirror()))
        CONF_FNS["mined frames (online)"] = mined.lookup_conf_all
    else:
        mined = None
    gate_cands = []
    if LIB == "gates":                                       # learned-frame family (gate kind)
        candidates += [(f"induction L={lm}", mir_induction_L(lm)) for lm in (4, 5)]
        for k in (2, 3):
            if ONLINE:
                for supp in (40,):
                    of = OnlineFrame(tuple(range(k, 0, -1)), vocab, DEV, minsupp=supp)
                    online_frames.append(of)
                    nm = f"kgram k={k} (online s{supp})"
                    candidates.append((nm, of.mirror()))
                    CONF_FNS[nm] = (lambda ids, of=of:
                                    of.table.lookup_conf(of._suffix_key(ids, False)))
            else:
                tab, nk = fit_kgram(ids, yv, fit, k, vocab)
                nm = f"kgram k={k} [{nk}]"
                candidates.append((nm, mir_kgram(tab, k, vocab)))
                register_conf(nm, tab, keyfn_offsets(tuple(range(k, 0, -1)), vocab))
        for off in (2, 3):
            tab, nk = fit_skip(ids, yv, fit, off)
            nm = f"skip o={off} [{nk}]"
            candidates.append((nm, mir_skip(tab, off)))
            register_conf(nm, tab, keyfn_offsets((off,), vocab))
        for offs in [(3, 1), (4, 1), (5, 1), (6, 1), (3, 2, 1), (4, 2, 1)]:
            tab, nk = fit_frame(ids, yv, fit, offs, vocab, minsupp=3, mindet=0.6)
            nm = f"gate {offs} [{nk}]"
            gate_cands.append((nm, mir_frame(tab, offs, vocab)))
            register_conf(nm, tab, keyfn_offsets(offs, vocab))
    print(f"candidate library ({len(candidates)}): {[n for n, _ in candidates]}")
    if gate_cands:
        print(f"gate grid ({len(gate_cands)}, proposer-selected at sleep): "
              f"{[n for n, _ in gate_cands]}")

    ground = grounded_init(uv).to(DEV)
    ground = ground / ground.shape[1] ** 0.5
    torch.manual_seed(0)
    model = WylyV3(ground, cls.cpu(), ell).to(DEV)
    log = []
    print(f"\n{'episode':>8}{'teacher-agree':>15}{'copy-agree':>12}{'tier':>6}")
    for ep, ch in enumerate(torch.chunk(fit, v2.EPISODES)):
        model.update_counts(ids[ch], y[ch], model.lut)
        for of in online_frames:
            of.update(ids[ch], yv[ch])
        for _ in range(v2.WAKE_STEPS):
            bi = ch[torch.randint(len(ch), (v2.BATCH,), generator=g)]
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(ids[bi]), y[bi]).backward()
            model.sgd(v2.LR)
        for of in online_frames:
            of.refresh()                                     # re-gate the online tiers each sleep
        if mined is not None:
            mined.mine(model, ids, yv, cls, fit, g, log, ep)
        for name, fn in model.rules:                         # scalar confidences from val (sw cover)
            if name not in CONF_FNS:
                RULE_CONF[name] = val_fired_acc(fn, ids, yv, val)
        cands = candidates
        if gate_cands:
            s_prop = fit[torch.randint(len(fit), (4000,), generator=g)]
            cands = candidates + propose_gates(model, gate_cands, ids, yv, cls, s_prop, log, ep)
        if JUDGE == "cover":
            sleep_admit_cover(model, ids, yv, cls, y, val, ep, cands, log)
        else:
            sleep_admit_v5(model, ids, y, val, ep, cands, log)
        print(f"{ep:>8}{v2.top1(model, ids, y, te):>15.3f}{v2.top1(model, ids, y, cs):>12.3f}"
              f"{len(model.rules):>6}", flush=True)
    print("\n" + "\n".join(log))
    full = v2.top1(model, ids, y, te)
    norules = v2.top1(model, ids, y, te, use_rules=False)
    cs_full, cs_norules = v2.top1(model, ids, y, cs), v2.top1(model, ids, y, cs, use_rules=False)
    print(f"\nv5[{LIB}/{JUDGE}] {TAG}/{DS}: teacher-agreement {full:.3f} (ablated {norules:.3f}, "
          f"marginal {full - norules:+.3f})")
    print(f"  copy-subset {cs_full:.3f} (ablated {cs_norules:.3f}, marginal "
          f"{cs_full - cs_norules:+.3f})")
    print(f"  admitted: {[r[0] for r in model.rules]}")
    core_fixed = core_cover(model, model.rules, ids, yv, cls, te)
    core_sw = core_cover_sw(model, model.rules, ids, yv, cls, te)
    core = core_sw if SWCOVER else core_fixed
    print(f"  certified core FIXED-ORDER: agree {core_fixed['agree']:.3f} @ "
          f"{core_fixed['cover']:.1%} | SUPPORT-WEIGHTED: agree {core_sw['agree']:.3f} @ "
          f"{core_sw['cover']:.1%}")
    torch.save({"full": full, "norules": norules, "cs_full": cs_full, "cs_norules": cs_norules,
                "core": core, "core_fixed": core_fixed, "core_sw": core_sw,
                "rules": [r[0] for r in model.rules]}, STATE)
    if EMIT:
        import re
        import shutil

        from wyly_export_package import build_manifest
        sys.path.insert(0, str(REPO))
        from pil.tokens import TokenSpace
        tok = REPO.parent / "rosetta" / "models" / "pythia70m" / "bundle.tokenizer.json"
        ts = TokenSpace.from_file(tok)
        trusted, skipped = [], []
        for name, _fn in model.rules:
            conf = round(RULE_CONF.get(name, 0.0), 4)
            cite = [f"admitted by the sleep judge ({LIB}/{JUDGE}{'/sw' if SWCOVER else ''}), "
                    f"teacher {TAG}, dataset {DS}; val fired-accuracy {conf}"]
            if name.startswith("induction L="):
                trusted.append({"kind": "induction", "tier": "trusted", "basis": "causal",
                                "L": int(name.split("L=")[1]), "confidence": conf,
                                "citation": cite})
            elif name.startswith("relation"):
                i, j, k = map(int, re.match(r"relation eq\((\d+),(\d+)\)c(\d+)", name).groups())
                trusted.append({"kind": "relation", "tier": "trusted", "basis": "causal",
                                "eq": [[i, j]], "copy": k, "confidence": conf, "citation": cite})
            else:
                skipped.append(name)
        pkg = REPO / "data" / "wyly_expert_package_v5"
        pkg.mkdir(parents=True, exist_ok=True)
        man = build_manifest(model.counts, cls, uv, ts, 1, 0.0, induction_rules=trusted,
                             model=f"wyly-v5-{LIB}-{TAG}-{DS}")
        (pkg / "manifest.json").write_text(json.dumps(man))
        shutil.copy(tok, pkg / "bundle.tokenizer.json")
        print(f"  package -> {pkg} ({len(trusted)} trusted rules; table-kind rules not emitted: "
              f"{skipped})")
    ADMIT_F.write_text(json.dumps([r[0] for r in model.rules]))


if __name__ == "__main__":
    main()
