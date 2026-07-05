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
DATA = REPO / "data" / (f"wyly_nexttoken_{DS}_L256.pt" if DS != "wikitext"
                        else "wyly_nexttoken_wikitext_L256.pt")
TEACH = REPO / "data" / (f"wyly_teacher_{TAG}_{DS}_L256.pt" if DS != "wikitext"
                         else f"wyly_teacher_{TAG}_L256.pt")
STATE = REPO / "data" / f"wyly_v5_{LIB}_{TAG}_{DS}{'_cov' if JUDGE == 'cover' else ''}.pt"
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
    """exact suffix/slot table as (sorted keys, values); mirror = searchsorted. Fit once on train."""

    def __init__(self, keys, vals):
        order = keys.argsort()
        self.k, self.v = keys[order], vals[order]

    def lookup(self, q):
        if len(self.k) == 0:
            return torch.full_like(q, -1)
        i = torch.searchsorted(self.k, q).clamp(max=len(self.k) - 1)
        hit = self.k[i] == q
        return torch.where(hit, self.v[i], torch.full_like(q, -1))


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
    return KeyTable(pair[0][sel][keep], pair[1][sel][keep]), int(keep.sum())


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


def mir_repetition(ids):
    same = ids[:, -1] == ids[:, -2]
    return torch.where(same, ids[:, -1], torch.full_like(ids[:, -1], -1))


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
    base = core_cover(model, model.rules, ids, yv, cls, val)["agree"]
    best = (thresh, None)
    for name, fn in candidates:
        if any(n == name for n, _ in model.rules):
            continue
        marg = core_cover(model, model.rules + [(name, fn)], ids, yv, cls, val)["agree"] - base
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
    candidates = [(f"induction L={lm}", mir_induction_L(lm)) for lm in (1, 2, 3)]
    if LIB == "ext":                                         # tr (which contains val) leaks the val
        candidates += [(f"induction L={lm}", mir_induction_L(lm)) for lm in (4, 5)]
        for k in (2, 3):                                     # windows' own suffix pairs into the
            tab, nk = fit_kgram(ids, yv, fit, k, vocab)      # judge's val marginals
            candidates.append((f"kgram k={k} [{nk}]", mir_kgram(tab, k, vocab)))
        for off in (2, 3):
            tab, nk = fit_skip(ids, yv, fit, off)
            candidates.append((f"skip o={off} [{nk}]", mir_skip(tab, off)))
        candidates.append(("repetition", mir_repetition))
    gate_cands = []
    if LIB == "gates":                                       # learned-frame family (gate kind)
        candidates += [(f"induction L={lm}", mir_induction_L(lm)) for lm in (4, 5)]
        for k in (2, 3):
            tab, nk = fit_kgram(ids, yv, fit, k, vocab)
            candidates.append((f"kgram k={k} [{nk}]", mir_kgram(tab, k, vocab)))
        for off in (2, 3):
            tab, nk = fit_skip(ids, yv, fit, off)
            candidates.append((f"skip o={off} [{nk}]", mir_skip(tab, off)))
        for offs in [(3, 1), (4, 1), (5, 1), (6, 1), (3, 2, 1), (4, 2, 1)]:
            tab, nk = fit_frame(ids, yv, fit, offs, vocab, minsupp=3, mindet=0.6)
            gate_cands.append((f"gate {offs} [{nk}]", mir_frame(tab, offs, vocab)))
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
        for _ in range(v2.WAKE_STEPS):
            bi = ch[torch.randint(len(ch), (v2.BATCH,), generator=g)]
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(ids[bi]), y[bi]).backward()
            model.sgd(v2.LR)
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
    core = core_cover(model, model.rules, ids, yv, cls, te)
    print(f"  certified core (package cover): agree {core['agree']:.3f} @ cover {core['cover']:.1%}"
          f" (when-fired {core['agree_fired']:.3f})")
    torch.save({"full": full, "norules": norules, "cs_full": cs_full, "cs_norules": cs_norules,
                "core": core, "rules": [r[0] for r in model.rules]}, STATE)
    ADMIT_F.write_text(json.dumps([r[0] for r in model.rules]))


if __name__ == "__main__":
    main()
