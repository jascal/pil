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

Env (selected):
  WYLY_TAG / WYLY_DS / WYLY_LIB  -- teacher tag, dataset, rule library (default ext)
  WYLY_LABELS=teacher|corpus    -- wake/table targets; default *teacher* (host decisions).
                                  *corpus* = stream next-token from the window file (no teacher file)
  WYLY_CONCEPT_INIT=grounded|random -- soft-student geometry; default *grounded* (host embed PCA).
                                  *random* = no host embed (standalone arms)
  WYLY_SOFT=1|0                 -- soft SGD wake; *0* = counts+sleep only (P0)
  WYLY_ALPHABET=host|word       -- host BPE TokenSpace vs corpus WordCodec (P1)
  WYLY_ALPHABET_PATH            -- path to alphabet_*.json when alphabet=word
  WYLY_ORIGIN                   -- package provenance override (e.g. standalone)
  WYLY_PKG_SUFFIX               -- optional package dir suffix (e.g. _standalone)
  WYLY_QUERY_SOURCE=file|corpus_mined -- hand JSON vs mine from corpus (P2)
  bAbI DS tags also auto-default cover-sw, QUERIES/ESTATE2 paths, QUERY_PACK (see module body)

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
# B3: bAbI paths freeze deployment-first defaults (cover-sw + query/estate2 paths when files
# exist). Explicit env still wins. Non-babi keeps historical soft / no-sw defaults.
_BABI = "babi" in DS
LIB = os.environ.get("WYLY_LIB", "ext")
JUDGE = os.environ.get("WYLY_JUDGE", "cover" if _BABI else "soft")  # soft = sec-8.6; cover = dir 1
ONLINE = os.environ.get("WYLY_ONLINE", "") == "1"            # design dir 5: online k-gram tiers
SWCOVER = os.environ.get("WYLY_COVER", "sw" if _BABI else "") == "sw"  # design dir 2
EMIT = os.environ.get("WYLY_EMIT", "") == "1"                # emit the rosetta package (relation kind)
CONCEPTS = os.environ.get("WYLY_CONCEPTS", "") == "1"        # concept induction + class frames
POINTER = os.environ.get("WYLY_POINTER", "1" if _BABI else "") == "1"  # pointer rules
TPOINTER = os.environ.get("WYLY_TPOINTER", "") == "1"        # transform-composed pointers (5 o 2)
GROW = os.environ.get("WYLY_GROW", "") == "1"                # append-only concept growth (13+14)
CX = os.environ.get("WYLY_CX", "1" if _BABI else "") == "1"  # derived gates (estate2 lives here)
DX = os.environ.get("WYLY_DX", "") == "1"                    # detection extensions: MDL judge +
MDL_LAM = float(os.environ.get("WYLY_MDL_LAM", "1e-8"))     # key-level retreat + anti-unification
FOLDS = int(os.environ.get("WYLY_FOLDS", "0"))               # multi-fold admission (0 = off)
WYLY_SEED = int(os.environ.get("WYLY_SEED", "0"))             # internal admission/model seed
TOKJ = os.environ.get("WYLY_TOKENIZER", "")                  # tokenizer override (non-pythia teachers)
KGRAMS = tuple(int(x) for x in os.environ.get("WYLY_KGRAMS", "2,3").split(","))
MINE_OFFS = tuple(int(x) for x in os.environ.get("WYLY_MINE_OFFS", "2,3,4,5,6,7,8").split(","))


def SAFE_KGRAMS(vocab):
    """drop tiers whose PAIR key (suffix*base + next) would overflow int64 -- tiny vocabs
    (bAbI: 38 tokens) overflow at k=11 (39**12 > 2**63), silently corrupting keys."""
    return tuple(k for k in KGRAMS if (vocab + 1) ** (k + 1) < 2 ** 62)
ECHO_SUCCS = tuple(int(x) for x in os.environ.get("WYLY_ECHO_SUCCS", "1").split(","))
VALREG = os.environ.get("WYLY_VAL_REGION", "") == "1"        # judge on a HELD-OUT corpus REGION
TUPLE = os.environ.get("WYLY_TUPLE", "") == "1"              # tuple-keyed tiers: exact, any reach
# Auto-wire query / estate2 files for known bAbI dataset tags when env is unset.
def _babi_paths(ds):
    if ds in ("babi2", "babi2_word"):
        return (REPO / "data" / "wyly_queries_babi_qa2.json",
                REPO / "data" / "wyly_estate2_qa2.json")
    if ds in ("babi3", "babi3x", "babi3x_word"):
        return (REPO / "data" / "wyly_queries_babi_qa3.json",
                REPO / "data" / "wyly_estate2_qa3.json")
    if ds in ("babi",):
        return (REPO / "data" / "wyly_queries_babi.json", None)
    return (None, None)


_q_default, _e2_default = _babi_paths(DS)
# P2: corpus_mined = wyly_queries_mined_{DS}.json (from build_word_babi / query_mine);
# file = hand wyly_queries_*.json (legacy).
QUERY_SOURCE = os.environ.get("WYLY_QUERY_SOURCE", "file")   # file | corpus_mined
QUERIES = os.environ.get("WYLY_QUERIES", "")                 # QUERY-SHAPED judging: val = deployment
if not QUERIES:
    if QUERY_SOURCE == "corpus_mined":
        _qm = REPO / "data" / f"wyly_queries_mined_{DS}.json"
        if _qm.exists():
            QUERIES = str(_qm)
    elif _q_default and _q_default.exists():
        QUERIES = str(_q_default)
QUERY_PAD = 24                                               # queries left-padded to >= max offset
# Pack length-groups into one left-padded batch (default on for bAbI): admission/query_agree
# were O(n_groups) host launches; packing preserves semantics and makes estate2 tractable.
QUERY_PACK = os.environ.get("WYLY_QUERY_PACK", "1" if _BABI else "0") == "1"
STRATA = os.environ.get("WYLY_STRATA", "") == "1"            # stratified admission + fall-through
STRATA_TAU = float(os.environ.get("WYLY_STRATA_TAU", "0.35"))
ESTATE2 = os.environ.get("WYLY_ESTATE2", "")                 # mined world-state sets (json)
if not ESTATE2 and _e2_default and _e2_default.exists():
    ESTATE2 = str(_e2_default)
# Alphabet: host = HF/BPE TokenSpace; word = corpus WordCodec (P1 standalone).
ALPHABET = os.environ.get("WYLY_ALPHABET", "host")           # host | word
ALPHABET_PATH = os.environ.get("WYLY_ALPHABET_PATH", "")
if not ALPHABET_PATH and ALPHABET == "word":
    _ap = REPO / "data" / f"alphabet_{DS}.json"
    if _ap.exists():
        ALPHABET_PATH = str(_ap)
# Soft student SGD (P0): 0 = counts + sleep only (no wake backprop).
SOFT = os.environ.get("WYLY_SOFT", "1") == "1"
# Non-pythia teachers: auto-pick local tokenizer only for *host* alphabet.
if ALPHABET == "host" and not TOKJ and "qwen" in TAG.lower():
    _tok = REPO / "data" / "qwen3b.tokenizer.json"
    if _tok.exists():
        TOKJ = str(_tok)
# Label source for wake / tables: "teacher" = host LM decisions (distillation);
# "corpus" = stream next-token from the window file (standalone: no teacher file).
LABELS = os.environ.get("WYLY_LABELS", "teacher")
# Soft-student concept geometry: "grounded" = host embed PCA; "random" = no host geometry.
CONCEPT_INIT = os.environ.get("WYLY_CONCEPT_INIT", "grounded")
PKG_SUFFIX = os.environ.get("WYLY_PKG_SUFFIX", "")           # e.g. _e0 / _e1 package dirs
ORIGIN = os.environ.get("WYLY_ORIGIN", "")                   # optional provenance override
# Only auto-wire host embeds when grounded init is requested (standalone arms use random).
if (ALPHABET == "host" and CONCEPT_INIT == "grounded" and "qwen" in TAG.lower()
        and not os.environ.get("WYLY_EMBED")):
    _emb = REPO / "data" / "qwen3b_embed.npy"
    if _emb.exists():
        os.environ["WYLY_EMBED"] = str(_emb)
RULE_SIZE = {}                                               # name -> callable -> table entries
CONCEPTS_CMAP = [None]                                       # cmap holder for package emission
ALPHA = 2.0                                                  # Laplace shrinkage for confidence
DATA = REPO / "data" / (f"wyly_nexttoken_{DS}_L256.pt" if DS != "wikitext"
                        else "wyly_nexttoken_wikitext_L256.pt")
TEACH = REPO / "data" / (f"wyly_teacher_{TAG}_{DS}_L256.pt" if DS != "wikitext"
                         else f"wyly_teacher_{TAG}_L256.pt")
STATE = REPO / "data" / (f"wyly_v5_{LIB}_{TAG}_{DS}{'_cov' if JUDGE == 'cover' else ''}"
                         f"{'_ol' if ONLINE else ''}{'_sw' if SWCOVER else ''}"
                         f"{'_cn' if CONCEPTS else ''}{'_pt' if POINTER else ''}"
                         f"{'_tp' if TPOINTER else ''}{'_dx' if DX else ''}"
                         f"{'_cx' if CX else ''}{'_gr' if GROW else ''}"
                         f"{'_f' + str(FOLDS) if FOLDS else ''}"
                         f"{'_corp' if LABELS == 'corpus' else ''}"
                         f"{'_rnd' if CONCEPT_INIT == 'random' else ''}.pt")
ADMIT_F = REPO / "data" / f"wyly_v5_{LIB}_{TAG}_{DS}{'_cov' if JUDGE == 'cover' else ''}_admitted.json"
DEV, V = v2.DEV, v2.V


def load_ds():
    d = torch.load(DATA, map_location="cpu")
    ids = d["kept_ids"]
    if LABELS == "corpus":
        labels = d["target"]                                 # gold next-token in the stream
        print(f"LABELS=corpus (stream next-token from {DATA.name}; no teacher file)",
              flush=True)
    else:
        labels = torch.load(TEACH, map_location="cpu")["teacher"]
        print(f"LABELS=teacher from {TEACH.name}", flush=True)
    keep = torch.isin(labels, torch.bincount(labels).argsort(descending=True)[:V])
    ids, labels = ids[keep], labels[keep]
    n, ell = ids.shape
    flat = torch.cat([ids.reshape(-1), labels])
    uv, inv = flat.unique(return_inverse=True)
    ids, yv = inv[:n * ell].reshape(n, ell), inv[n * ell:]
    cls, y = yv.unique(return_inverse=True)
    ids, y, cls = ids.to(DEV), y.to(DEV), cls.to(DEV)
    tr = torch.arange(int(0.85 * n), device=DEV)
    te = torch.arange(int(0.85 * n), n, device=DEV)
    return ids, y, cls, uv, tr, te


def concept_init(uv: torch.Tensor) -> torch.Tensor:
    """Soft-student concept rows: grounded (host embed PCA) or random (standalone)."""
    from wyly_lm_bench import K0
    if CONCEPT_INIT == "random" or ALPHABET == "word":
        # Word alphabet has no host embed table; always random geometry.
        g = torch.Generator().manual_seed(0)
        c = torch.randn(len(uv), K0, generator=g)
        c = (c - c.mean(0)) / c.std(0).clamp_min(1e-6)
        print(f"CONCEPT_INIT=random ({tuple(c.shape)}; alphabet={ALPHABET})", flush=True)
        return c
    print("CONCEPT_INIT=grounded (host embed PCA)", flush=True)
    return grounded_init(uv)


def load_codec():
    """TokenSpace (host BPE) or WordCodec (corpus word alphabet)."""
    sys.path.insert(0, str(REPO))
    if ALPHABET == "word":
        from pil.alphabet import WordCodec
        path = ALPHABET_PATH or str(REPO / "data" / f"alphabet_{DS}.json")
        print(f"ALPHABET=word codec={path}", flush=True)
        return WordCodec.from_file(path)
    from pil.tokens import TokenSpace
    path = TOKJ if TOKJ else str(REPO.parent / "rosetta" / "models" / "pythia70m"
                                 / "bundle.tokenizer.json")
    print(f"ALPHABET=host codec={path}", flush=True)
    return TokenSpace.from_file(path)


def provenance_origin() -> str:
    if ORIGIN:
        return ORIGIN
    if LABELS == "corpus" and (CONCEPT_INIT == "random" or ALPHABET == "word") and not SOFT:
        return "standalone"
    if LABELS == "corpus":
        return "document"
    return "teacher"


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
        k2, v2, c2 = key[sel][keep], val[sel][keep], conf[keep]
        if hasattr(self, "banned") and len(self.banned):
            kp = ~torch.isin(k2, self.banned)
            k2, v2, c2 = k2[kp], v2[kp], c2[kp]
        self.table = KeyTable(k2, v2, c2)

    def mirror(self):
        def fn(ids):
            return self.table.lookup(self._suffix_key(ids, False))
        return fn

    def retreat(self, ids, yv, val, floor=0.15, minn=8):
        """8: CEGIS-style retreat -- a rule keeps only the keys it is RIGHT on. Per-key val
        fired-accuracy; keys with n >= minn and acc < floor are banned (and stay banned through
        refreshes). Raises per-rule confidence, which the sw cover immediately monetizes."""
        if not hasattr(self, "banned"):
            self.banned = torch.empty(0, dtype=torch.long, device=self.pk.device)
        q = self._suffix_key(ids[val], False)
        v, _ = self.table.lookup_conf(q)
        f = v >= 0
        if not int(f.sum()):
            return 0
        good = (v[f] == yv[val][f]).float()
        uk, inv = q[f].unique(return_inverse=True)
        acc = torch.zeros(len(uk), device=q.device).index_add_(0, inv, good)
        nn = torch.zeros(len(uk), device=q.device).index_add_(0, inv, torch.ones_like(good))
        bad = uk[(nn >= minn) & (acc / nn.clamp(min=1) < floor)]
        if len(bad):
            self.banned = torch.cat([self.banned, bad]).unique()
        keep = ~torch.isin(self.table.k, self.banned)
        self.table = KeyTable(self.table.k[keep], self.table.v[keep],
                              self.table.c[keep] if self.table.c is not None else None)
        return int(len(bad))


class TupleFrame:
    """TUPLE-KEYED online tier (the key-reach fix): rows are stored as 2D (suffix tokens, next)
    with counts -- no base**k packing, so NO int64 overflow at any k and any vocab: reach is
    unbounded and keys are EXACT (the packed tiers silently WRAPPED at base**(k+1) >= 2**63,
    functioning as an accidental hash table). Lookup is a tensorized trie: per level, (prev-id,
    token) pairs compact to unique ids -- ids are bounded by ROW COUNT, never by vocab**k."""

    def __init__(self, offs, vocab, dev, minsupp=40, mindet=0.5):
        self.offs, self.vocab = offs, vocab
        self.k = len(offs)
        self.minsupp, self.mindet = minsupp, mindet
        self.dev = dev
        self.rows = torch.empty(0, self.k + 1, dtype=torch.long, device=dev)
        self.cnts = torch.empty(0, dtype=torch.long, device=dev)
        self.levels, self.leaf_val, self.leaf_conf = [], None, None
        self.banned = torch.empty(0, dtype=torch.long, device=dev)

    def _tails(self, ids, sliding):
        if sliding:
            maxo, ell = max(self.offs), ids.shape[1]
            cols = [ids[:, maxo - o:ell - o].reshape(-1) for o in self.offs]
            return torch.stack(cols, 1)
        return torch.stack([ids[:, -o] for o in self.offs], 1)

    def update(self, w, ytail):
        maxo = max(self.offs)
        r1 = torch.cat([self._tails(w, True), w[:, maxo:].reshape(-1, 1)], 1)
        r2 = torch.cat([self._tails(w, False), ytail.reshape(-1, 1)], 1)
        allr = torch.cat([self.rows.repeat_interleave(
            torch.ones(len(self.rows), dtype=torch.long, device=self.dev), 0), r1, r2])
        allc = torch.cat([self.cnts,
                          torch.ones(len(r1) + len(r2), dtype=torch.long, device=self.dev)])
        uniq, inv = allr.unique(dim=0, return_inverse=True)
        self.rows = uniq
        self.cnts = torch.zeros(len(uniq), dtype=torch.long,
                                device=self.dev).index_add_(0, inv, allc)

    def refresh(self):
        if not len(self.rows):
            return
        suf, val, cnt = self.rows[:, :-1], self.rows[:, -1], self.cnts
        usuf, kinv = suf.unique(dim=0, return_inverse=True)
        tot = torch.zeros(len(usuf), device=self.dev).index_add_(0, kinv, cnt.float())
        comp = kinv * (int(cnt.max()) + 1) + cnt
        order = comp.argsort(descending=True)
        first = torch.ones(len(order), dtype=torch.bool, device=self.dev)
        first[1:] = kinv[order][1:] != kinv[order][:-1]
        sel = order[first]
        keep = (cnt[sel] >= self.minsupp) & (cnt[sel].float()
                                             / tot[kinv[sel]].clamp(min=1) >= self.mindet)
        keys2d, vals = suf[sel][keep], val[sel][keep]
        confs = (cnt[sel].float() / (tot[kinv[sel]] + ALPHA))[keep]
        if not len(keys2d):
            self.levels, self.leaf_val, self.leaf_conf = [], None, None
            return
        self.keys2d, self.vals2d, self.confs2d = keys2d, vals, confs
        # trie build: per level, (prev-id, token) pairs -> compact unique ids
        self.levels = []
        ids_ = torch.zeros(len(keys2d), dtype=torch.long, device=self.dev)
        nprev = 1
        for i in range(self.k):
            pair = ids_ * (self.vocab + 1) + keys2d[:, i]    # ids_ < N rows -> never overflows
            upair, inv = pair.unique(return_inverse=True)
            self.levels.append(upair)
            ids_ = inv
            nprev = len(upair)
        self.leaf_val = torch.full((nprev,), -1, dtype=torch.long, device=self.dev)
        self.leaf_conf = torch.zeros(nprev, device=self.dev)
        self.leaf_val[ids_] = vals
        self.leaf_conf[ids_] = confs

    def _walk(self, q2d):
        ids_ = torch.zeros(len(q2d), dtype=torch.long, device=self.dev)
        alive = torch.ones(len(q2d), dtype=torch.bool, device=self.dev)
        for i, upair in enumerate(self.levels):
            pair = ids_ * (self.vocab + 1) + q2d[:, i]
            j = torch.searchsorted(upair, pair).clamp(max=max(len(upair) - 1, 0))
            hit = (upair[j] == pair) if len(upair) else torch.zeros_like(alive)
            alive &= hit
            ids_ = torch.where(alive, j, ids_)
        return ids_, alive

    def lookup_conf(self, ids):
        q2d = self._tails(ids, False)
        if not self.levels or self.leaf_val is None:
            return (torch.full((len(ids),), -1, dtype=torch.long, device=self.dev),
                    torch.full((len(ids),), -1e9, device=self.dev))
        ids_, alive = self._walk(q2d)
        v = torch.where(alive, self.leaf_val[ids_], torch.full_like(ids_, -1))
        c = torch.where(alive & (v >= 0), self.leaf_conf[ids_],
                        torch.full((len(ids),), -1e9, device=self.dev))
        return v, c

    def mirror(self):
        def fn(ids):
            return self.lookup_conf(ids)[0]
        return fn

    def retreat(self, ids, yv, val, floor=0.15, minn=8):
        v, _ = self.lookup_conf(ids[val])
        f = v >= 0
        if not int(f.sum()):
            return 0
        good = (v[f] == yv[val][f]).float()
        q2d = self._tails(ids[val], False)[f]
        uq, inv = q2d.unique(dim=0, return_inverse=True)
        acc = torch.zeros(len(uq), device=self.dev).index_add_(0, inv, good)
        nn = torch.zeros(len(uq), device=self.dev).index_add_(0, inv, torch.ones_like(good))
        badk = uq[(nn >= minn) & (acc / nn.clamp(min=1) < floor)]
        if not len(badk):
            return 0
        ids2, alive = self._walk(badk)
        self.leaf_val[ids2[alive]] = -1
        return int(alive.sum())


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


class ConceptSpace:
    """TABLE-DRIVEN CONCEPT INDUCTION (build-order 12): two token classes belong to one concept
    when they are interchangeable in the certified tier -- their next-decision count rows agree.
    Refreshed each sleep: support-gated rows, cosine similarity, union-find merge; cmap maps every
    class to its concept REPRESENTATIVE (min member id), identity where nothing merged. Discrete,
    extensional, and certifiable by the members' pooled stats -- no new trust surface."""

    def __init__(self, vocab, dev, minsupp=30, tau=0.85):
        self.vocab, self.dev = vocab, dev
        self.minsupp, self.tau = minsupp, tau
        self.cmap = torch.arange(vocab, device=dev)
        self.n_concepts = self.n_merged = 0

    def refresh(self, counts, extra_pairs=None):
        tot = counts.sum(1)
        strong = torch.where(tot >= self.minsupp)[0]
        if len(strong) < 10:
            return
        rows = counts[strong].float()
        rn = rows / rows.norm(dim=1, keepdim=True).clamp(min=1e-9)
        sim = rn @ rn.T
        sim.fill_diagonal_(0)
        ii, jj = torch.nonzero(sim >= self.tau, as_tuple=True)
        keep = ii < jj
        pairs = torch.stack([strong[ii[keep]], strong[jj[keep]]], 1).tolist()
        if extra_pairs:
            pairs = pairs + [list(p) for p in extra_pairs]
        parent = list(range(self.vocab))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        cmap = torch.tensor([find(i) for i in range(self.vocab)], device=self.dev)
        self.n_merged = int((cmap != torch.arange(self.vocab, device=self.dev)).sum())
        self.n_concepts = int(cmap.unique().numel())
        self.cmap = cmap


def pooled_counts(counts, cmap, vocab):
    """counts pooled by concept: rows index_add'd over cmap -> (argmax, Laplace conf) per concept
    representative. Fires through cmap[last]; sw arbitration decides when pooling beats the exact
    row (exact wins where it has support; the pool wins on rare members -- the de/morphology fix)."""
    pooled = torch.zeros_like(counts).index_add_(0, cmap, counts)
    mx, am = pooled.max(1)
    tot = pooled.sum(1)
    conf = mx.float() / (tot + ALPHA)
    return am, conf, mx


class PointerRule:
    """The POINTER kind (build-order 2): generalized copy. Score every in-window source position p
    by discrete match features -- exact-suffix length l (longest k with ids[p-k:p] == tail k) and
    CONCEPT-suffix length lc (same under cmap: the channel verb-final word order needs -- the
    source clause matches by class even when inflection differs) -- and predict ids[p], the
    successor of the best match, argmax lexicographic (l, lc, recency). Induction L is the
    lc-blind, fixed-l special case. Confidence is a per-(l,lc)-CELL val fired-accuracy, refreshed
    each sleep and support-gated -- the sw cover arbitrates each answer at its cell's measured
    reliability (C10's calibration premise, applied cell-wise)."""

    def __init__(self, vocab, dev, lmax=6, cmap_ref=None):
        self.lmax, self.dev = lmax, dev
        self.cmap_ref = cmap_ref                             # [cmap] holder (concept space) or None
        self.cells = {}                                      # (l, lc) -> conf

    def _feats(self, ids):
        """per window: (pred, l*, lc*) of the argmax source position; pred -1 where nothing fires
        (l == 0 and lc < 2). Positions p in [1, n-1] predict ids[:, p]."""
        n = ids.shape[1]
        amap = self.cmap_ref[0] if self.cmap_ref is not None else None
        cds = amap[ids] if amap is not None else ids
        npos = n - 1
        el = torch.zeros(len(ids), npos, dtype=torch.long, device=ids.device)
        ec = torch.zeros_like(el)
        okl = torch.ones(len(ids), npos, dtype=torch.bool, device=ids.device)
        okc = okl.clone()
        for k in range(1, self.lmax + 1):
            # source positions p = 1..n-1; compare ids[:, p-k] vs ids[:, n-k]; p-k >= 0 required
            cmp_l = torch.zeros(len(ids), npos, dtype=torch.bool, device=ids.device)
            cmp_c = cmp_l.clone()
            cmp_l[:, k - 1:] = ids[:, :n - k] == ids[:, n - k:n - k + 1]
            cmp_c[:, k - 1:] = cds[:, :n - k] == cds[:, n - k:n - k + 1]
            okl &= cmp_l
            okc &= cmp_c
            el += okl.long()
            ec += okc.long()
        fire = (el >= 1) | (ec >= 2)
        score = (el * (self.lmax + 2) + ec) * (npos + 1) + torch.arange(
            npos, device=ids.device)
        score = torch.where(fire, score, torch.full_like(score, -1))
        best = score.argmax(1)
        rows = torch.arange(len(ids), device=ids.device)
        has = score[rows, best] >= 0
        pred = torch.where(has, ids[rows, best + 1], torch.full_like(best, -1))
        return pred, el[rows, best], ec[rows, best], has

    def refresh(self, ids, yv, val, minsupp=30):
        pred, l, lc, has = self._feats(ids[val])
        good = (pred == yv[val]) & has
        self.cells = {}
        for li in range(self.lmax + 1):
            for ci in range(self.lmax + 1):
                m = has & (l == li) & (lc == ci)
                nn = int(m.sum())
                if nn >= minsupp:
                    self.cells[(li, ci)] = float(good[m].float().mean())

    def lookup_conf(self, ids):
        pred, l, lc, has = self._feats(ids)
        conf = torch.full((len(ids),), -1e9, device=ids.device)
        for (li, ci), c in self.cells.items():
            conf = torch.where(has & (l == li) & (lc == ci),
                               torch.full_like(conf, c), conf)
        pred = torch.where(conf > -1e9, pred, torch.full_like(pred, -1))
        return pred, conf

    def mirror(self):
        def fn(ids):
            return self.lookup_conf(ids)[0]
        return fn


class TransformPointer(PointerRule):
    """TRANSFORM-COMPOSED pointer (build-order 5 composed with 2): the pointer finds the source
    by class-match; the COUNTS TIER re-inflects. Where the plain pointer copies the source's
    successor verbatim (and gets the wrong surface form on inflecting languages), this rule
    predicts the member of the successor's CONCEPT that the target's local bigram row supports
    most: pred = argmax_{m in concept(succ)} counts[last, m]. A composition of two certified
    structures -- the pointer proposes the class, the counts decode the form -- with its own
    per-(l,lc)-cell val confidences."""

    def __init__(self, vocab, dev, counts_ref, cls, lmax=6, cmap_ref=None):
        super().__init__(vocab, dev, lmax=lmax, cmap_ref=cmap_ref)
        self.counts_ref, self.cls = counts_ref, cls

    def _feats(self, ids):
        pred, l, lc, has = super()._feats(ids)
        if self.cmap_ref is None:
            return pred, l, lc, has
        cmap = self.cmap_ref[0]
        counts = self.counts_ref[0]
        dec_concept = cmap[self.cls]                          # concept of each decision class
        out = torch.full_like(pred, -1)
        ok = torch.zeros_like(has)
        for i in range(0, len(ids), 4096):                    # chunked class->form decode
            sl = slice(i, min(i + 4096, len(ids)))
            m = has[sl] & (pred[sl] >= 0)
            if not int(m.sum()):
                continue
            tgt = cmap[pred[sl]]                              # proposed CLASS (of the successor)
            row = counts[ids[sl, -1]].float()                 # target-local bigram evidence
            row = row * (dec_concept[None, :] == tgt[:, None])
            mx, am = row.max(1)
            good = m & (mx >= 1)
            out[sl] = torch.where(good, self.cls[am], out[sl])
            ok[sl] = good
        return out, l, lc, ok


def bracket_sets(uv, ts):
    """extensional opener/closer token sets from the tokenizer (exact; class = decoded string)."""
    op, cl = [], []
    pairs = {"(": ")", "[": "]", "{": "}"}
    for i, t in enumerate(uv.tolist()):
        d = ts.token_str(t).strip()
        if d in pairs:
            op.append(i)
        elif d in pairs.values():
            cl.append(i)
    return op, cl


def mate_feature(ids, is_open, is_close):
    """15 (v1): the DERIVED FEATURE 'innermost unclosed opener' -- a role, not a token set.
    depth = cumsum(+1 opener / -1 closer); opener at i is unclosed iff min depth over [i, end)
    >= depth at i (reverse cummin); the feature is the CLASS ID of the innermost such opener
    (or -1). Exact and host-side: the first two-layer rule -- derive the predicate, gate on it."""
    sign = is_open[ids].long() - is_close[ids].long()
    pref = sign.cumsum(1)
    revmin = pref.flip(1).cummin(1).values.flip(1)
    cand = is_open[ids] & (revmin >= pref)
    pos = torch.arange(ids.shape[1], device=ids.device)
    best = torch.where(cand, pos, torch.full_like(pos, -1).expand_as(cand)).max(1).values
    rows = torch.arange(len(ids), device=ids.device)
    return torch.where(best >= 0, ids[rows, best.clamp(min=0)],
                       torch.full_like(best, -1))


def recent_member_pos(ids, member, unique=False):
    """position of the most recent member (optionally unique-occurrence); -1 if none."""
    m = member[ids]
    if unique:
        srt, _ = ids.sort(1)
        first = torch.ones_like(srt, dtype=torch.bool)
        first[:, 1:] = srt[:, 1:] != srt[:, :-1]
        last_ = torch.ones_like(srt, dtype=torch.bool)
        last_[:, :-1] = srt[:, :-1] != srt[:, 1:]
        once_sorted = first & last_
        rank = ids.argsort(dim=1, stable=True).argsort(dim=1, stable=True)
        once = once_sorted.gather(1, rank)
        m = m & once
    pos = torch.arange(ids.shape[1], device=ids.device)
    return torch.where(m, pos, torch.full_like(pos, -1).expand_as(m)).max(1).values


def mate_pos(ids, is_open, is_close):
    """position of the innermost unclosed opener; -1 if none."""
    sign = is_open[ids].long() - is_close[ids].long()
    pref = sign.cumsum(1)
    revmin = pref.flip(1).cummin(1).values.flip(1)
    cand = is_open[ids] & (revmin >= pref)
    pos = torch.arange(ids.shape[1], device=ids.device)
    return torch.where(cand, pos, torch.full_like(pos, -1).expand_as(cand)).max(1).values


def prev_occ_pos(ids, pos):
    """CHAINED ROLE: the position of the PREVIOUS occurrence of the token found at `pos` --
    strictly before it. -1 where the base position is absent or the token never occurred
    earlier. Composed with at_pos(succ=1) this is the ENTITY ECHO: what followed this referent
    the last time it appeared."""
    rows = torch.arange(len(ids), device=ids.device)
    base_tok = ids[rows, pos.clamp(min=0)]
    eq = ids == base_tok[:, None]
    idxs = torch.arange(ids.shape[1], device=ids.device)
    eq &= idxs[None, :] < pos[:, None]
    prev = torch.where(eq, idxs, torch.full_like(idxs, -1).expand_as(eq)).max(1).values
    return torch.where(pos >= 0, prev, torch.full_like(prev, -1))


def prev_occ_pos_filtered(ids, pos, avoid, look=3):
    """FILTERED chained role (entity-conditioned MOVEMENT binding): the previous occurrence of
    the token at `pos`, skipping occurrences whose following `look` tokens contain an avoid-set
    member (e.g. question markers -- the stale-location fix the region judge demanded)."""
    rows = torch.arange(len(ids), device=ids.device)
    base_tok = ids[rows, pos.clamp(min=0)]
    eq = ids == base_tok[:, None]
    n = ids.shape[1]
    bad = torch.zeros_like(eq)
    for j in range(1, look + 1):
        bad[:, :n - j] |= avoid[ids[:, j:]]
    idxs = torch.arange(n, device=ids.device)
    eq &= (idxs[None, :] < pos[:, None]) & ~bad
    prev = torch.where(eq, idxs, torch.full_like(idxs, -1).expand_as(eq)).max(1).values
    return torch.where(pos >= 0, prev, torch.full_like(prev, -1))


def estate_feature(ids, ent, val, avoid, within=7, look=3, slot=None):
    """ENTITY-STATE REGISTER (the EAV rule form, one attribute per instance): last-writer-wins
    fold -- at each clean occurrence of an entity-class token followed by a value-class token
    within `within`, register[entity] := value; the feature is register[query entity] (the most
    recent entity-class token). Self-grounded member sets; validated 1000/1000 on bAbI qa1."""
    n = ids.shape[1]
    rows = torch.arange(len(ids), device=ids.device)
    qpos = recent_member_pos(ids, ent)                       # the queried entity
    qtok = ids[rows, qpos.clamp(min=0)]
    eq = (ids == qtok[:, None]) & ent[ids]                   # occurrences of THAT entity
    bad = torch.zeros_like(eq)
    for j in range(1, look + 1):                             # question-context filter
        bad[:, :n - j] |= avoid[ids[:, j:]]
    eq &= ~bad
    idxs = torch.arange(n, device=ids.device)
    eq &= idxs[None, :] < qpos[:, None]
    has_val = torch.zeros_like(eq)                           # entity occ with a value in reach
    vmask = val[ids]
    first_val = torch.full_like(ids, -1)
    for j in range(within, 0, -1):                           # nearest value within `within`
        cand = torch.roll(ids, -j, 1)
        cmask = torch.roll(vmask, -j, 1)
        cmask[:, n - j:] = False
        first_val = torch.where(cmask, cand, first_val)
        has_val |= cmask
    eq &= has_val
    last = torch.where(eq, idxs, torch.full_like(idxs, -1).expand_as(eq)).max(1).values
    feat = torch.where(last >= 0, first_val[rows, last.clamp(min=0)],
                       torch.full_like(last, -1))
    feat = torch.where(qpos >= 0, feat, torch.full_like(feat, -1))
    if slot is not None:                                     # answer-slot signature: the (t-2,
        ok_slot = (ids[:, -2] == slot[0]) & (ids[:, -1] == slot[1])   # t-1) pair where the
        feat = torch.where(ok_slot, feat, torch.full_like(feat, -1))  # register predicts next
    return feat


def estate2_feature(ids, sets, mode="is", hist_slots=8):
    """WORLD-STATE FOLD (estate2): batched sequential simulation -- loc[entity] from movements,
    holder/loc[object] from mined take/drop verbs (drops freeze location), location HISTORY per
    object for "before"-queries. mode="is": feature = current object location; mode="before":
    feature = the history predecessor of the reference location. Probes: qa2 1000/1000,
    qa3 998/1000 (probe_estate2.py)."""
    B, n = ids.shape
    dev = ids.device
    ent, loc, obj = sets["ent"], sets["loc"], sets["obj"]
    mv, tk, dp = sets["move"], sets["take"], sets["drop"]
    nE = int(ent.sum())
    nO = int(obj.sum())
    eidx = torch.full((len(ent),), -1, dtype=torch.long, device=dev)
    eidx[torch.where(ent)[0]] = torch.arange(nE, device=dev)
    oidx = torch.full((len(obj),), -1, dtype=torch.long, device=dev)
    oidx[torch.where(obj)[0]] = torch.arange(nO, device=dev)
    locE = torch.full((B, nE), -1, dtype=torch.long, device=dev)
    holder = torch.full((B, nO), -1, dtype=torch.long, device=dev)
    locO = torch.full((B, nO), -1, dtype=torch.long, device=dev)
    hist = torch.full((B, nO, hist_slots), -1, dtype=torch.long, device=dev)
    rows = torch.arange(B, device=dev)

    def push_hist(omask_rows, osel, newloc):
        # shift the history of (row, obj) pairs where location changes
        changed = omask_rows & (locO[rows, osel] != newloc)
        if not bool(changed.any()):
            return
        r = rows[changed]
        o = osel[changed]
        hist[r, o, 1:] = hist[r, o, :-1].clone()
        hist[r, o, 0] = locO[r, o]
        locO[r, o] = newloc[changed]

    for i in range(n - 1):
        w = ids[:, i]
        v = ids[:, i + 1]
        we = eidx[w]                                          # -1 if not entity
        is_ent = we >= 0
        # movement destination: first location token in i+2..i+6
        dest = torch.full((B,), -1, dtype=torch.long, device=dev)
        for j in range(min(6, n - 1 - i), 1, -1):
            cand = ids[:, i + j]
            dest = torch.where(loc[cand], cand, dest)
        m_move = is_ent & mv[v] & (dest >= 0)
        if bool(m_move.any()):
            locE[rows[m_move], we[m_move]] = dest[m_move]
            for o in range(nO):                               # held objects follow
                held = m_move & (holder[rows, o] == we)
                if bool(held.any()):
                    push_hist(held, torch.full((B,), o, device=dev), dest)
        # grab object: first object token in i+2..i+5
        gobj = torch.full((B,), -1, dtype=torch.long, device=dev)
        for j in range(min(5, n - 1 - i), 1, -1):
            cand = ids[:, i + j]
            gobj = torch.where(obj[cand], cand, gobj)
        wo = torch.where(gobj >= 0, oidx[gobj.clamp(min=0)], torch.full_like(gobj, -1))
        m_take = is_ent & tk[v] & (wo >= 0)
        if bool(m_take.any()):
            holder[rows[m_take], wo[m_take]] = we[m_take]
            eloc = locE[rows, we.clamp(min=0)]
            known = m_take & (eloc >= 0)
            if bool(known.any()):
                push_hist(known, wo, eloc)
        m_drop = is_ent & dp[v] & (wo >= 0)
        if bool(m_drop.any()):
            holder[rows[m_drop], wo[m_drop]] = -1
    # query: the LAST object token = the queried object
    idxs = torch.arange(n, device=dev)
    om = obj[ids]
    qopos = torch.where(om, idxs, torch.full_like(idxs, -1).expand_as(om)).max(1).values
    qo = torch.where(qopos >= 0, oidx[ids[rows, qopos.clamp(min=0)]],
                     torch.full_like(qopos, -1))
    if mode == "is":
        feat = torch.where(qo >= 0, locO[rows, qo.clamp(min=0)], torch.full_like(qo, -1))
        return feat
    # mode "before": reference = the last location token AFTER the queried object
    lm = loc[ids] & (idxs[None, :] > qopos[:, None])
    refpos = torch.where(lm, idxs, torch.full_like(idxs, -1).expand_as(lm)).max(1).values
    ref = torch.where(refpos >= 0, ids[rows, refpos.clamp(min=0)],
                      torch.full_like(refpos, -1))
    full = torch.cat([hist[rows, qo.clamp(min=0)].flip(1),
                      locO[rows, qo.clamp(min=0)].unsqueeze(1)], 1)   # oldest..current
    feat = torch.full((B,), -1, dtype=torch.long, device=dev)
    for k in range(full.shape[1] - 1, 0, -1):                 # last match of ref, predecessor
        hit = (full[:, k] == ref) & (feat < 0) & (ref >= 0)
        feat = torch.where(hit, full[:, k - 1], feat)
    return torch.where(qo >= 0, feat, torch.full_like(feat, -1))


def next_member_after(ids, pos, member):
    """the first member-set token AFTER `pos` (the location slot after a movement mention --
    template-length-invariant, unlike fixed succ shifts). -1 if none."""
    n = ids.shape[1]
    m = member[ids]
    idxs = torch.arange(n, device=ids.device)
    m &= idxs[None, :] > pos[:, None]
    nxt = torch.where(m, idxs, torch.full_like(idxs, n + 1).expand_as(m)).min(1).values
    ok = (pos >= 0) & (nxt <= n - 1)
    rows = torch.arange(len(ids), device=ids.device)
    return torch.where(ok, ids[rows, nxt.clamp(max=n - 1)], torch.full_like(pos, -1))


def at_pos(ids, pos, succ=0):
    """ROLE COMPOSITION: the token at (position + succ) -- features of features ("the token
    after the mate", "the token after the clause opener"). -1 where the position is absent or
    the shift runs off the window."""
    p = pos + succ
    ok = (pos >= 0) & (p >= 0) & (p < ids.shape[1])
    rows = torch.arange(len(ids), device=ids.device)
    return torch.where(ok, ids[rows, p.clamp(min=0, max=ids.shape[1] - 1)],
                       torch.full_like(pos, -1))


def recent_member_feature(ids, member, unique=False):
    """the most recent member token (optionally unique-occurrence) -- at_pos of the pos variant."""
    return at_pos(ids, recent_member_pos(ids, member, unique=unique))


def since_member_feature(ids, member, cap=32):
    """DISCOURSE extractor: tokens SINCE the most recent member occurrence, capped -- with
    members = sentence enders this is position-in-sentence; with discourse connectives it is
    position-in-move. cap+1 where no member occurred. Exact; certifiable (max aggregate +
    arithmetic)."""
    pos = recent_member_pos(ids, member)
    n = ids.shape[1]
    return torch.where(pos >= 0, (n - 1 - pos).clamp(max=cap),
                       torch.full_like(pos, cap + 1))


def member_parity_feature(ids, member):
    """DISCOURSE extractor: the count of member occurrences mod 2 -- with members = quote marks
    this is the inside-quotation indicator (attribution scope). Exact; certifiable (count + mod)."""
    return member[ids].long().sum(1) % 2


DSTATE_EDGES = (2, 6, 14, 32)


def dstate_feature(ids, sent_member, quote_member):
    """RHETORICAL-STATE role (multiple surface roles combined): bucketed position-in-sentence
    (edges 2/6/14/32, +none) x quotation parity -> ~10 discrete states. Low cardinality by
    construction -- the lesson from sentpair's decline: pair registers need a low-card side."""
    since = since_member_feature(ids, sent_member, cap=33)
    bucket = torch.zeros_like(since)
    for e in DSTATE_EDGES:
        bucket += (since > e).long()
    par = member_parity_feature(ids, quote_member)
    return bucket * 2 + par


def depth_feature(ids, is_open, is_close, cap=8):
    """derived extractor (build-order 4, the balance counter): the current bracket DEPTH, clamped
    to [0, cap] -- stack-shaped context no frame can express, now one prefix sum."""
    sign = is_open[ids].long() - is_close[ids].long()
    d = sign.cumsum(1)[:, -1].clamp(min=0, max=cap)
    return d


def fit_dgate_feature(ids, yv, tr, featfn, vocab, minsupp=3, mindet=0.5):
    """generic (feature, last) dgate fitter (fit_mate generalized to any extractor)."""
    B = vocab + 2
    w = ids[tr]
    f = featfn(w)
    m = f >= 0
    if int(m.sum()) < 20:                                    # feature never fires here (e.g. an
        e = torch.empty(0, dtype=torch.long, device=ids.device)  # English member set on German
        return KeyTable(e, e.clone(), torch.empty(0, device=ids.device)), 0   # text) -> no gate
    key = (f[m] + 1) * B + w[m][:, -1]
    tab, nk = best_per_key(key, yv[tr][m], minsupp, mindet)
    return tab, nk


def fit_dgate2(ids, yv, tr, featfnA, featfnB, vocab, minsupp=3, mindet=0.5):
    """PAIR dgate fitter: key = (featureA, featureB) jointly -- the rhetorical span-pair /
    claim-evidence signature. Fires only where BOTH features exist."""
    B = vocab + 2
    w = ids[tr]
    fa, fb = featfnA(w), featfnB(w)
    m = (fa >= 0) & (fb >= 0)
    if int(m.sum()) < 20:
        e = torch.empty(0, dtype=torch.long, device=ids.device)
        return KeyTable(e, e.clone(), torch.empty(0, device=ids.device)), 0
    key = (fa[m] + 1) * B + fb[m]
    tab, nk = best_per_key(key, yv[tr][m], minsupp, mindet)
    return tab, nk


def fit_mate(ids, yv, tr, is_open, is_close, vocab, minsupp=3, mindet=0.5):
    """gate keyed on (mate, last): when the innermost unclosed opener is X and the last token is
    Y, the teacher's decision -- the rule that knows WHICH closer (or continuation) fits."""
    B = vocab + 2
    w = ids[tr]
    f = mate_feature(w, is_open, is_close)
    m = f >= 0
    key = (f[m] + 1) * B + w[m][:, -1]
    tab, nk = best_per_key(key, yv[tr][m], minsupp, mindet)
    return tab, nk


class WylyGrow(WylyV3):
    """13+14: APPEND-ONLY concept growth. C stays a frozen buffer and the tied decode is untouched
    -- the C9-compatible reading of the frozen-C rule: the retention argument forbids MUTATING
    decode rows, not appending input-side ones. Sleep mines high-PMI adjacent class pairs
    (compound tokens -- BPE-above-BPE, chosen by the learner); each compound gets a trainable
    delta vector ADDED to the last-position concept features whenever the pair is present; wake
    trains the new rows like any parameter, and only behavior OUTSIDE the certified cover can
    move."""

    def __init__(self, ground, cls, ell):
        super().__init__(ground, cls, ell)
        self.register_buffer("grow_keys", torch.empty(0, dtype=torch.long,
                                                      device=ground.device))
        self.Cx = torch.nn.Parameter(torch.zeros(0, ground.shape[1], device=ground.device))

    def base(self, ids):
        f = super().base(ids)
        if len(self.grow_keys):
            pair = ids[:, -2] * self.C.shape[0] + ids[:, -1]
            i = torch.searchsorted(self.grow_keys, pair).clamp(max=len(self.grow_keys) - 1)
            hit = self.grow_keys[i] == pair
            kk = self.C.shape[1]
            add = torch.zeros(len(ids), kk, device=ids.device, dtype=f.dtype)
            add[hit] = self.Cx[i[hit]]
            f = torch.cat([f[:, :kk] + add, f[:, kk:]], -1)
        return f

    def grow(self, new_keys):
        if not len(new_keys):
            return 0
        allk = torch.cat([self.grow_keys, new_keys]).unique(sorted=True)
        cx = torch.zeros(len(allk), self.C.shape[1], device=allk.device)
        if len(self.grow_keys):
            pos = torch.searchsorted(allk, self.grow_keys)
            cx[pos] = self.Cx.data
        added = len(allk) - len(self.grow_keys)
        self.grow_keys = allk
        self.Cx = torch.nn.Parameter(cx)
        return added


def mine_compounds(ids, chunk, vocab, topn=40, minn=50, minpmi=3.0):
    """14: high-PMI adjacent pairs -- compound-token concepts. The uniform stride-overlap
    inflation cancels in the PMI ratio."""
    w = ids[chunk]
    a, b = w[:, :-1].reshape(-1), w[:, 1:].reshape(-1)
    pair = a * vocab + b
    pk, pc = pair.unique(return_counts=True)
    tok, tc = w.reshape(-1).unique(return_counts=True)
    freq = torch.zeros(vocab, device=ids.device).index_add_(0, tok, tc.float())
    n = float(len(a))
    pa, pb = freq[pk // vocab] / n, freq[pk % vocab] / n
    pmi = torch.log((pc.float() / n) / (pa * pb).clamp(min=1e-12))
    keep = (pc >= minn) & (pmi >= minpmi)
    pk, pmi = pk[keep], pmi[keep]
    if len(pk) > topn:
        pk = pk[pmi.topk(topn).indices]
    return pk


CONF_FNS = {}                                            # name -> (ids)->(val, per-key conf)
RULE_CONF = {}                                           # name -> scalar val fired-accuracy
EMIT_INFO = {}                                           # name -> (kind, ...) for package emission


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
    ccal = getattr(model, "counts_calib", None)              # QUERY-CALIBRATED counts: window
    cconf = mx.float() / (tot + ALPHA)                       # confidences can be junk-confident
    if ccal is not None:                                     # at deployment tails (babi2 ':'
        c2_ = ccal[t]                                        # row: conf 0.489, acc 0.000 on
        cconf = torch.where(c2_ >= 0, c2_, cconf)            # 1000/1000 queries)
    consider(a, cconf)
    if STRATA and getattr(model, "rules2", None):            # FALL-THROUGH: where stratum-1
        low = conf < STRATA_TAU                              # confidence is below tau, stratum-2
        if bool(low.any()):                                  # candidates may claim the answer
            for name2, fn2 in model.rules2:
                if name2 in CONF_FNS:
                    a2, c2 = CONF_FNS[name2](w)
                else:
                    a2 = fn2(w)
                    c2 = torch.full((len(w),), float(RULE_CONF.get(name2, 0.0)),
                                    device=w.device)
                m2 = low & (a2 >= 0) & (c2 > conf)
                pred = torch.where(m2, a2, pred)
                conf = torch.where(m2, c2, conf)
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

    def __init__(self, vocab, dev, offs=None,
                 pair_offs=((2, 3), (2, 4), (3, 4), (2, 5)), amap=None):
        offs = MINE_OFFS if offs is None else offs
        self.B, self.dev = vocab + 1, dev
        self.amap = amap                                     # anchor-space map (class frames): ids
        # pass through amap[.] on the ANCHOR channel only; table keys stay raw tokens
        self.offs, self.pair_offs = list(offs), list(pair_offs)
        self.mined1, self.mined2 = set(), set()
        e = torch.empty(0, dtype=torch.long, device=dev)
        self.t1 = KeyTable(e, e.clone(), torch.empty(0, device=dev))     # (o,a,last) singleton
        self.t2 = KeyTable(e.clone(), e.clone(), torch.empty(0, device=dev))  # 2-anchor frames
        self.n_frames = 0

    def A(self, x):
        return x if self.amap is None else self.amap[0][x]

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
            tab, _ = best_per_key(self.A(tails[:, -o]) * self.B + tails[:, -1], ytail, 3, 0.4)
            v, c = tab.lookup_conf(self.A(ids[err][:, -o]) * self.B + ids[err][:, -1])
            good = (v == yv[err]).float()
            ua, inv = self.A(ids[err][:, -o]).unique(return_inverse=True)
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
                key_fit = ((self.A(tails[:, -o1]) * self.B + self.A(tails[:, -o2])) * self.B + tails[:, -1])
                tab, _ = best_per_key(key_fit, ytail, 3, 0.5)
                ke = (self.A(ids[res][:, -o1]) * self.B + self.A(ids[res][:, -o2])) * self.B + ids[res][:, -1]
                v, c = tab.lookup_conf(ke)
                good = (v == yv[res]).float()
                pk = self.A(ids[res][:, -o1]) * self.B + self.A(ids[res][:, -o2])
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
            v, c = self.t1.lookup_conf((o * self.B + self.A(w[:, -o])) * self.B + w[:, -1])
            m = (v >= 0) & (c > best_c)
            best_v, best_c = torch.where(m, v, best_v), torch.where(m, c, best_c)
        for o1, o2 in self.pair_offs:
            k = ((o1 * 16 + o2) * self.B ** 2 + self.A(w[:, -o1]) * self.B
                 + self.A(w[:, -o2])) * self.B + w[:, -1]
            v, c = self.t2.lookup_conf(k)
            m = (v >= 0) & (c > best_c)
            best_v, best_c = torch.where(m, v, best_v), torch.where(m, c, best_c)
        return best_v, best_c

    def mirror(self):
        def fn(ids):
            return self.lookup_conf_all(ids)[0]
        return fn

    def retreat(self, ids, yv, val, floor=0.15, minn=8):
        """8: key-level retreat over both mined stores (see OnlineFrame.retreat)."""
        pruned = 0
        w = ids[val]
        for o in self.offs:
            q = (o * self.B + self.A(w[:, -o])) * self.B + w[:, -1]
            pruned += self._retreat_store("t1", q, yv[val], floor, minn)
        for o1, o2 in self.pair_offs:
            q = ((o1 * 16 + o2) * self.B ** 2 + self.A(w[:, -o1]) * self.B
                 + self.A(w[:, -o2])) * self.B + w[:, -1]
            pruned += self._retreat_store("t2", q, yv[val], floor, minn)
        return pruned

    def _retreat_store(self, which, q, y, floor, minn):
        tab = getattr(self, which)
        v, _ = tab.lookup_conf(q)
        f = v >= 0
        if not int(f.sum()):
            return 0
        good = (v[f] == y[f]).float()
        uk, inv = q[f].unique(return_inverse=True)
        acc = torch.zeros(len(uk), device=q.device).index_add_(0, inv, good)
        nn = torch.zeros(len(uk), device=q.device).index_add_(0, inv, torch.ones_like(good))
        bad = uk[(nn >= minn) & (acc / nn.clamp(min=1) < floor)]
        if not len(bad):
            return 0
        keep = ~torch.isin(tab.k, bad)
        setattr(self, which, KeyTable(tab.k[keep], tab.v[keep], tab.c[keep]))
        return int(len(bad))

    def mine_clustered(self, model, ids, yv, cls, fitset, ground, g, log, cycle,
                       kcl=10, sample_n=6000):
        """9: ERROR CLUSTERING IN CONCEPT SPACE -- cluster residual-error contexts by their mean
        concept vector (last 16 positions), then run the anchor search WITHIN each cluster with
        lower floors: scopes the offset grid never contained. Frames found here join the same
        stores (and face the same judge)."""
        smp = fitset[torch.randint(len(fitset), (sample_n,), generator=g)]
        _, pred = core_cover_sw(model, model.rules, ids, yv, cls, smp, return_pred=True)
        err = smp[pred != yv[smp]]
        if len(err) < 200:
            return
        feat = ground[ids[err][:, -16:]].mean(1)
        cent = feat[torch.randperm(len(feat), generator=g)[:kcl].to(feat.device)]
        for _ in range(6):
            d = torch.cdist(feat, cent)
            asg = d.argmin(1)
            for c in range(len(cent)):
                m = asg == c
                if int(m.sum()):
                    cent[c] = feat[m].mean(0)
        tails, ytail = ids[fitset], yv[fitset]
        new1 = 0
        for c in range(kcl):
            sub = err[asg == c]
            if len(sub) < 30:
                continue
            for o in self.offs:
                tab, _ = best_per_key(self.A(tails[:, -o]) * self.B + tails[:, -1], ytail, 3, 0.4)
                v, cc = tab.lookup_conf(self.A(ids[sub][:, -o]) * self.B + ids[sub][:, -1])
                good = (v == yv[sub]).float()
                ua, inv = self.A(ids[sub][:, -o]).unique(return_inverse=True)
                rec = torch.zeros(len(ua), device=self.dev).index_add_(0, inv, good)
                cnt = torch.zeros(len(ua), device=self.dev).index_add_(0, inv,
                                                                       torch.ones_like(good))
                for a in ua[(cnt >= 8) & (rec / cnt.clamp(min=1) >= 0.3)].tolist():
                    if (o, a) in self.mined1:
                        continue
                    self.mined1.add((o, a))
                    sl = (tab.k // self.B) == a
                    self.t1 = self._merge(self.t1, (o * self.B + a) * self.B + tab.k[sl] % self.B,
                                          tab.v[sl], tab.c[sl])
                    new1 += 1
        self.n_frames = len(self.mined1) + len(self.mined2)
        if new1:
            log.append(f"    sleep {cycle}: CLUSTER-MINED {new1} frames (from {len(err)} errors, "
                       f"{kcl} concept-space clusters)")

    def au_pairs(self, min_shared=5, agree=0.8):
        """7: ANTI-UNIFICATION -- anchors at the same offset whose singleton tables agree on
        shared content keys are interchangeable; the pairs feed ConceptSpace as frame-tier
        evidence (least-general-generalization: classes DISCOVERED from admitted structure)."""
        pairs = []
        if not len(self.t1.k):
            return pairs
        oa, last = self.t1.k // self.B, self.t1.k % self.B
        o_all, a_all = oa // self.B, oa % self.B
        for o in self.offs:
            m = o_all == o
            if int(m.sum()) < 2 * min_shared:
                continue
            anchors = a_all[m].unique()
            tabs = {}
            for a in anchors.tolist():
                sl = m & (a_all == a)
                tabs[a] = dict(zip(last[sl].tolist(), self.t1.v[sl].tolist(), strict=True))
            alist = list(tabs)
            for i in range(len(alist)):
                for j in range(i + 1, len(alist)):
                    ti, tj = tabs[alist[i]], tabs[alist[j]]
                    shared = set(ti) & set(tj)
                    if len(shared) >= min_shared:
                        ag = sum(ti[k] == tj[k] for k in shared) / len(shared)
                        if ag >= agree:
                            pairs.append((alist[i], alist[j]))
        return pairs


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


QUERY_BATCHES = []                                           # [(ids [B,L], y [B])] per length


def load_queries(ts, uv, cls):
    """WYLY_QUERIES: deployment-shaped judge set -- {prompt, answer} pairs, tokenized, mapped to
    class space, grouped by length, left-padded to >= the deepest rule offset. The judge then
    scores candidates WHERE DEPLOYMENT QUERIES LIVE (prompt-final positions), not on random
    window slices -- the fix for the window-vs-query blindness seen on bAbI movement rules and
    the elements k-tier admissions."""
    import json as _json
    qs = _json.loads(Path(QUERIES).read_text())
    inv = {int(t): i for i, t in enumerate(uv.tolist())}
    clsset = set(cls.tolist())
    groups = {}
    for q in qs:
        toks = [inv.get(t) for t in ts.encode(" " + q["prompt"])]
        ans = [inv.get(t) for t in ts.encode(" " + q["answer"])][:8]
        if any(t is None for t in toks) or not ans or any(a is None for a in ans) \
                or ans[0] not in clsset:
            continue
        ln = max(len(toks), QUERY_PAD)
        padded = [toks[0]] * (ln - len(toks)) + toks
        groups.setdefault(ln, []).append((padded, ans))
    out = []
    maxa = max((len(a_) for g in groups.values() for _, a_ in g), default=1)
    for _ln, items in sorted(groups.items()):
        out.append((torch.tensor([p_ for p_, _ in items], device=DEV),
                    torch.tensor([a_ + [-1] * (maxa - len(a_)) for _, a_ in items],
                                 device=DEV)))
    n = sum(len(b[1]) for b in out)
    print(f"query judge: {n} deployment-shaped queries in {len(out)} length groups "
          f"(source={QUERY_SOURCE})", flush=True)
    return out


def pack_query_batches(batches):
    """Left-pad all length groups into a single batch. Same chain-scoring semantics; far fewer
    estate2 / cover kernel launches when stories vary widely in length (bAbI qa3: 531 groups)."""
    if not batches or len(batches) == 1:
        return batches
    maxL = max(q.shape[1] for q, _ in batches)
    maxA = max(a.shape[1] for _, a in batches)
    qs, ans = [], []
    for qids, qans in batches:
        B, L = qids.shape
        if L < maxL:
            qids = torch.cat([qids[:, :1].expand(B, maxL - L), qids], 1)
        if qans.shape[1] < maxA:
            pad = torch.full((B, maxA - qans.shape[1]), -1, device=qans.device, dtype=qans.dtype)
            qans = torch.cat([qans, pad], 1)
        qs.append(qids)
        ans.append(qans)
    n = sum(len(a) for a in ans)
    print(f"query pack: {len(batches)} length groups -> 1 batch (n={n}, L={maxL})")
    return [(torch.cat(qs), torch.cat(ans))]


def ensure_query_batches(uv, cls, *, required=False):
    """Idempotent load of QUERY_BATCHES. Call before any candidate that mines slots/confs from
    deployment (estate2, query-calibrated tables, ...). The qa3 Band-A residual was exactly this
    ordering bug: construction ran while QUERY_BATCHES was still empty.

    required=True asserts that WYLY_QUERIES is set and at least one batch is loaded -- use at
    the head of deployment-first candidate builders to fail loud on future regressions.
    """
    global QUERY_BATCHES
    if QUERY_BATCHES:
        return QUERY_BATCHES
    if not QUERIES:
        if required:
            raise RuntimeError(
                "ensure_query_batches(required=True) but WYLY_QUERIES is unset; "
                "deployment-first candidates need query batches before construction")
        return QUERY_BATCHES
    _qts = load_codec()
    loaded = load_queries(_qts, uv, cls)
    if QUERY_PACK:
        loaded = pack_query_batches(loaded)
    QUERY_BATCHES.extend(loaded)
    if required and not QUERY_BATCHES:
        raise RuntimeError(
            f"ensure_query_batches(required=True): loaded 0 batches from {QUERIES}")
    return QUERY_BATCHES


def query_agree(model, rules, cls, batches, fold=None, nfolds=3):
    """CHAIN-scored agreement on deployment queries: greedy-slide through the full answer and
    require every answer token to match -- first-token scoring is blind to generation-time
    value. qans: [B, A] class-space answers padded with -1."""
    ok = tot = 0
    for qids, qans in batches:
        sel = torch.arange(len(qids), device=DEV)
        if fold is not None:
            sel = sel[sel % nfolds == fold]
        if not len(sel):
            continue
        w = qids[sel]
        good = torch.ones(len(sel), dtype=torch.bool, device=DEV)
        for step in range(qans.shape[1]):
            want = qans[sel, step]
            live = want >= 0
            if not bool(live.any()):
                break
            _, pred = core_cover_sw(model, rules, w, want.clamp(min=0), cls,
                                    torch.arange(len(w), device=DEV), return_pred=True)
            good &= (~live) | (pred == want)
            w = torch.cat([w[:, 1:], torch.where(live, want, w[:, -1]).unsqueeze(1)], 1)
        ok += int(good.sum())
        tot += len(sel)
    return ok / max(tot, 1)


def sleep_admit_cover(model, ids, yv, cls, y, val, cycle, candidates, log, thresh=5e-4):
    """COVER-AWARE admission (design direction 1): a candidate is scored by its marginal to the
    PACKAGE COVER on held-out val -- the objective the runtime actually realizes -- not by its vote
    in the soft mixture. Rules that merely displace an equally good lower tier score ~0 and are
    rejected; rules still install into the soft model as votes once admitted (training benefit)."""
    cover = core_cover_sw if SWCOVER else core_cover
    if QUERY_BATCHES and SWCOVER:                            # counts calibration on deployment
        # queries: per-tail fired accuracy
        ok_ = torch.zeros(len(model.counts), device=DEV)     # replaces window conf where
        n_ = torch.zeros(len(model.counts), device=DEV)      # measured (n >= 10)
        for qids_, qans_ in QUERY_BATCHES:
            t_ = qids_[:, -1]
            row_ = model.counts[t_]
            pred_ = cls[row_.argmax(1)]
            good_ = (pred_ == qans_[:, 0]).float()
            ok_.index_add_(0, t_, good_)
            n_.index_add_(0, t_, torch.ones(len(t_), device=DEV))
        calib_ = torch.where(n_ >= 10, ok_ / n_.clamp(min=1),
                             torch.full_like(ok_, -1.0))
        model.counts_calib = calib_
    if QUERY_BATCHES:                                        # QUERY-SHAPED judging: marginals on
        nf = FOLDS if FOLDS else 1                           # deployment-query folds
        qbases = [query_agree(model, model.rules, cls, QUERY_BATCHES,
                              fold=(i if FOLDS else None), nfolds=nf) for i in range(nf)]
    folds = ([val[i::FOLDS] for i in range(FOLDS)] if FOLDS else [val])
    bases = [cover(model, model.rules, ids, yv, cls, f)["agree"] for f in folds]
    best = (thresh, None)
    for name, fn in candidates:
        if any(n == name for n, _ in model.rules):
            continue
        ec = ({name: val_fired_acc(fn, ids, yv, val)}
              if SWCOVER and name not in CONF_FNS and name not in RULE_CONF else None)
        kw = {"extra_conf": ec} if SWCOVER else {}
        t = 0.002 if name.startswith("gate") else thresh     # gate tables are high-variance
        if QUERY_BATCHES:                                    # BLEND: query value counts,
            nf = FOLDS if FOLDS else 1                       # window value keeps broad
            wmargs = [cover(model, model.rules + [(name, fn)], ids, yv, cls, f,
                            **kw)["agree"] - b for f, b in zip(folds, bases, strict=True)]
            qmargs = [query_agree(model, model.rules + [(name, fn)], cls, QUERY_BATCHES,
                                  fold=(i if FOLDS else None), nfolds=nf) - b
                      for i, b in zip(range(nf), qbases, strict=True)]
            margs = [(qm + wm) / 2
                     for qm, wm in zip(qmargs, wmargs, strict=False)] or qmargs
        else:
            margs = [cover(model, model.rules + [(name, fn)], ids, yv, cls, f, **kw)["agree"]
                     - b for f, b in zip(folds, bases, strict=True)]
        marg = sum(margs) / len(margs)
        clears = sum(mm > t for mm in margs)
        pen = MDL_LAM * RULE_SIZE[name]() if DX and name in RULE_SIZE else 0.0
        score = marg - pen                                   # 11: MDL -- marginal per table bit
        log.append(f"    sleep {cycle}: candidate {name} COVER marginal {marg:+.4f}"
                   + (f" [{clears}/{len(folds)} folds]" if FOLDS else "")
                   + (f" (MDL -{pen:.4f})" if pen else ""))
        ok = score > t and (not FOLDS or clears * 2 > len(folds))  # mean over t AND majority of
        if ok and score > best[0]:                           # folds clears -- variance-stable
            best = (score, (name, fn))
    if best[1]:
        name, fn = best[1]
        model.install(name, fn)
        log.append(f"    sleep {cycle}: ADMITTED {name} (cover {best[0]:+.4f})")
    if STRATA:                                               # STRATUM 2: calibrated non-winners
        if not hasattr(model, "rules2"):                     # (fired-accuracy >= 0.5, marginal
            model.rules2 = []                                # not negative) survive as labeled
        for name2, fn2 in candidates:                        # fall-through coverage instead of
            if any(n == name2 for n, _ in model.rules) \
                    or any(n == name2 for n, _ in model.rules2):
                continue
            if QUERY_BATCHES:                            # qualify on DEPLOYMENT-shaped data:
                oks = tots = 0                           # window fired-acc admitted the stale
                for qids, qans in QUERY_BATCHES:         # pointer to stratum 2 (0.71 on val
                    if name2 in CONF_FNS:                # windows) and the fall-through zone
                        a2, _ = CONF_FNS[name2](qids)    # was exactly where it is wrong
                    else:
                        a2 = fn2(qids)
                    f2 = a2 >= 0
                    oks += int((a2[f2] == qans[f2, 0]).sum())
                    tots += int(f2.sum())
                fa = oks / tots if tots >= 20 else 0.0
            else:
                fa = val_fired_acc(fn2, ids, yv, val)
            if fa >= 0.5 and len(model.rules2) < 24:
                model.rules2.append((name2, fn2))
                RULE_CONF.setdefault(name2, fa)
                log.append(f"    sleep {cycle}: stratum-2 {name2} (fired-acc {fa:.2f})")
    if best[1]:
        return best[1][0]
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

def _stability_annotation(name, kind, frequency_table, domain_matches):
    if not domain_matches or frequency_table is None or name is None or kind == "ngram":
        return None
    import re
    base_name = re.sub(r"\s+\[\d+\]$", "", name)
    count = next((row["count"] for row in frequency_table if row["name"] == base_name), None)
    return {"n": count, "N": 8,
            "sweep": "seed_sweep_summary.json@f8ac0574c436976d0ce414400a15ac5300dacdba",
            "domain": "pythia70m/wikitext"}


def emit_full(model, cls, uv, ts, vocab):
    """Package emission, support-weighted (design: C10 realized in serving). Every admitted rule
    family becomes package rules carrying the SAME confidences the learner arbitrates with --
    per-key Laplace-shrunk determinism for tables, val fired-accuracy for scalar kinds -- in the
    learner's arbitration order (admitted rules, then the counts tier). The manifest declares
    cover: support-weighted; a conforming runtime fires every applicable rule and takes the
    argmax confidence (rosetta PACKAGE.md)."""
    import re
    B = vocab + 1
    rules_out, skipped = [], []
    rid = 0
    src = f"wyly-v5-{LIB}-{TAG}-{DS}"
    stability_domain_matches = (TAG, DS) == ("pythia70m", "wikitext")
    stability_path = REPO / "data" / "seed_sweep_summary.json"
    stability_frequency_table = None
    if stability_domain_matches and stability_path.exists():
        stability_frequency_table = json.loads(stability_path.read_text())["b3"]["frequency_table"]

    cur_stratum = 1
    cur_rule_name = None

    def add(r):
        nonlocal rid
        annotation = _stability_annotation(
            cur_rule_name, r["kind"], stability_frequency_table, stability_domain_matches)
        if annotation is not None:
            r["admission_stability"] = annotation
        r["id"] = rid
        if cur_stratum > 1:
            r["stratum"] = cur_stratum
        rules_out.append(r)
        rid += 1

    def decomp(key, n):
        toks = []
        for _ in range(n):
            toks.append(int(key % B))
            key //= B
        toks.reverse()
        return toks

    derived_defs, need_concepts = [], False
    all_rules = [(1, r) for r in model.rules] + \
                [(2, r) for r in getattr(model, "rules2", [])]
    for cur_stratum, (name, _fn) in all_rules:  # noqa: B007 (read by add())
        cur_rule_name = name
        conf = round(RULE_CONF.get(name, 0.0), 4)
        cite = [f"admitted by the sleep judge ({LIB}/{JUDGE}{'/sw' if SWCOVER else ''}), "
                f"teacher {TAG}, dataset {DS}; val fired-accuracy {conf}"]
        if name.startswith("induction L="):
            add({"kind": "induction", "tier": "trusted", "basis": "causal",
                 "L": int(name.split("L=")[1]), "confidence": conf, "citation": cite})
        elif name.startswith("relation"):
            i, j, k = map(int, re.match(r"relation eq\((\d+),(\d+)\)c(\d+)", name).groups())
            add({"kind": "relation", "tier": "trusted", "basis": "causal",
                 "eq": [[i, j]], "copy": k, "confidence": conf, "citation": cite})
        elif name in EMIT_INFO:
            info = EMIT_INFO[name]
            if info[0] == "kgram":
                of = info[1]
                tab, k = of.table, len(of.offs)
                for key, v, c in zip(tab.k.tolist(), tab.v.tolist(), tab.c.tolist(), strict=True):
                    add({"kind": "ngram", "tier": "gated", "basis": "observational",
                         "ctx": [int(uv[t]) for t in decomp(key, k)], "out": int(uv[v]),
                         "confidence": round(c, 4),
                         "citation": [f"{src} online {k}-gram tier (s{of.minsupp})"]})
            elif info[0] == "skip":
                off, tab = info[1], info[2]
                table = {int(uv[k]): int(uv[v]) for k, v in zip(tab.k.tolist(), tab.v.tolist(), strict=True)}
                confs = {int(uv[k]): round(c, 4) for k, c in zip(tab.k.tolist(), tab.c.tolist(), strict=True)}
                add({"kind": "gate", "tier": "gated", "basis": "observational", "frame": {},
                     "slot": off, "table": table, "confs": confs,
                     "citation": [f"{src} skip-bigram offset {off}"]})
            elif info[0] == "mate":
                mtab, opl, cll, B2 = info[1], info[2], info[3], info[4]
                derived_defs.append({"id": "mate0", "kind": "bracket-mate",
                                     "openers": [int(uv[i]) for i in opl],
                                     "closers": [int(uv[i]) for i in cll]})
                table, confs = {}, {}
                for key, v, c in zip(mtab.k.tolist(), mtab.v.tolist(), mtab.c.tolist(),
                                     strict=True):
                    f, last = key // B2 - 1, key % B2
                    ck = f"{int(uv[f])}:{int(uv[last])}"
                    table[ck] = int(uv[v])
                    confs[ck] = round(c, 4)
                add({"kind": "dgate", "tier": "gated", "basis": "observational",
                     "feature": "mate0", "table": table, "confs": confs,
                     "citation": [f"{src} mate gate: innermost unclosed opener (extractor "
                                  "PROVED, wyly_mate_certify)"]})
            elif info[0] == "pointer":
                ptr = info[1]
                need_concepts = ptr.cmap_ref is not None
                add({"kind": "pointer", "tier": "trusted", "basis": "causal",
                     "lmax": ptr.lmax,
                     "cells": {f"{a}:{b}": round(c, 4) for (a, b), c in ptr.cells.items()},
                     "citation": [f"{src} pointer: per-(l,lc)-cell val fired-accuracy"]})
            elif info[0] == "dfeat":
                (kindname, params), tab_, B2_ = info[1], info[2], info[3]
                fid = f"feat{len(derived_defs)}"
                if kindname == "prev-occ":                   # chained: emit the base def first
                    base_id = f"feat{len(derived_defs)}b"
                    derived_defs.append({"id": base_id, "kind": "recent-member",
                                         "members": [int(uv[i]) for i in torch.where(
                                             params["of_members"])[0].tolist()]})
                elif kindname == "estate2":
                    sk = {"ent": "entities", "loc": "locations", "obj": "objects",
                          "move": "move_verbs", "take": "take_verbs", "drop": "drop_verbs"}
                    dd2 = {sk[k2]: [int(uv[i]) for i in torch.where(v2_)[0].tolist()]
                           for k2, v2_ in params["sets"].items()}
                    dd2.update({"id": fid, "kind": "estate2", "mode": params["mode"],
                                "slot": [int(uv[t]) for t in params["slot"]]})
                    derived_defs.append(dd2)
                    params = {}
                elif kindname == "estate":
                    derived_defs.append(
                        {"id": fid, "kind": "estate",
                         "entity_members": [int(uv[i]) for i in
                                            torch.where(params["entity_members"])[0].tolist()],
                         "value_members": [int(uv[i]) for i in
                                           torch.where(params["value_members"])[0].tolist()],
                         "avoid": [int(uv[i]) for i in
                                   torch.where(params["avoid"])[0].tolist()],
                         "within": params.get("within", 7),
                         "slot": [int(uv[t]) for t in params["slot"]]})
                    params = {}
                if kindname == "prev-occ":
                    dd2 = {"id": fid, "kind": "prev-occ", "of": base_id,
                           "succ": params.get("succ", 0)}
                    if params.get("of_shift"):
                        dd2["of_shift"] = params["of_shift"]
                    if params.get("avoid") is not None:
                        dd2["avoid"] = [int(uv[i]) for i in
                                        torch.where(params["avoid"])[0].tolist()]
                    derived_defs.append(dd2)
                    params = {}
                dd = {"id": fid, "kind": kindname}
                for pk, pv in params.items():
                    if pk in ("members", "openers", "closers", "quote_members"):
                        dd[pk] = [int(uv[i]) for i in torch.where(pv)[0].tolist()]
                    else:
                        dd[pk] = pv
                if kindname not in ("prev-occ", "estate", "estate2"):
                    derived_defs.append(dd)
                tokfeat = kindname.startswith("recent") or kindname in ("estate", "estate2")
                table, confs = {}, {}
                for key, v, c in zip(tab_.k.tolist(), tab_.v.tolist(), tab_.c.tolist(),
                                     strict=True):
                    f, last = key // B2_ - 1, key % B2_
                    ck = f"{int(uv[f]) if tokfeat else f}:{int(uv[last])}"
                    table[ck] = int(uv[v])
                    confs[ck] = round(c, 4)
                add({"kind": "dgate", "tier": "gated", "basis": "observational",
                     "feature": fid, "table": table, "confs": confs,
                     "citation": [f"{src} {name}: derived {kindname} gate "
                                  "(extractor family certified, wyly_derived_certify)"]})
            elif info[0] == "dgate2":
                (_pairname, params), tab_, B2_ = info[1], info[2], info[3]
                base_id = f"feat{len(derived_defs)}s"
                head_id = f"feat{len(derived_defs)}h"
                prev_id = f"feat{len(derived_defs)}p"
                mems = [int(uv[i]) for i in torch.where(params["members"])[0].tolist()]
                derived_defs.append({"id": base_id, "kind": "recent-member", "members": mems})
                derived_defs.append({"id": head_id, "kind": "recent-member", "members": mems,
                                     "succ": 1})
                derived_defs.append({"id": prev_id, "kind": "prev-occ", "of": base_id,
                                     "succ": 1})
                table, confs = {}, {}
                for key, v, c in zip(tab_.k.tolist(), tab_.v.tolist(), tab_.c.tolist(),
                                     strict=True):
                    fa, fb = key // B2_ - 1, key % B2_
                    ck = f"{int(uv[fa])}:{int(uv[fb])}"
                    table[ck] = int(uv[v])
                    confs[ck] = round(c, 4)
                add({"kind": "dgate2", "tier": "gated", "basis": "observational",
                     "featureA": prev_id, "featureB": head_id, "table": table, "confs": confs,
                     "citation": [f"{src} {name}: rhetorical span-pair (prev-head, cur-head)"]})
            elif info[0] == "mined":
                mined = info[1]
                by_frame = {}
                for key, v, c in zip(mined.t1.k.tolist(), mined.t1.v.tolist(),
                                     mined.t1.c.tolist(), strict=True):
                    oa, last = key // B, key % B
                    o, a = oa // B, oa % B
                    by_frame.setdefault((("f1", o, a)), []).append((last, v, c))
                for key, v, c in zip(mined.t2.k.tolist(), mined.t2.v.tolist(),
                                     mined.t2.c.tolist(), strict=True):
                    oid_ab, last = key // B, key % B
                    oid, ab = oid_ab // (B * B), oid_ab % (B * B)
                    o1, o2 = oid // 16, oid % 16
                    a1, a2 = ab // B, ab % B
                    by_frame.setdefault((("f2", o1, a1, o2, a2)), []).append((last, v, c))
                for fr, entries in sorted(by_frame.items()):
                    frame = ({str(fr[1]): int(uv[fr[2]])} if fr[0] == "f1"
                             else {str(fr[1]): int(uv[fr[2]]), str(fr[3]): int(uv[fr[4]])})
                    add({"kind": "gate", "tier": "gated", "basis": "observational",
                         "frame": frame, "slot": 1,
                         "table": {int(uv[last]): int(uv[v]) for last, v, _ in entries},
                         "confs": {int(uv[last]): round(c, 4) for last, _, c in entries},
                         "citation": [f"{src} mined frame {frame} (error-driven, online tier)"]})
        else:
            skipped.append(name)
    mx, am = model.counts.max(1)
    tot = model.counts.sum(1)
    ccal_e = getattr(model, "counts_calib", None)
    for t in torch.where(mx >= 1)[0].tolist():
        conf_e = float(mx[t] / (tot[t] + ALPHA))
        if ccal_e is not None and float(ccal_e[t]) >= 0:
            conf_e = float(ccal_e[t])                        # query-calibrated (deployment)
        add({"kind": "ngram", "tier": "gated", "basis": "observational",
             "ctx": [int(uv[t])], "out": int(uv[cls[am[t]]]),
             "support": int(mx[t]), "determinism": round(float(mx[t] / tot[t]), 4),
             "confidence": round(conf_e, 4),
             "citation": [f"{src} online counts: {ts.token_str(int(uv[t]))!r} -> "
                          f"{ts.token_str(int(uv[cls[am[t]]]))!r} (n={int(mx[t])}/{int(tot[t])})"]})
    W = max((len(r["ctx"]) for r in rules_out if r.get("kind") == "ngram"), default=1)
    man = {"model": src, "cover": "support-weighted", "W": W, "n_rules": len(rules_out),
           "origin": provenance_origin(),
           "origin_model": "corpus" if LABELS == "corpus" else TAG,
           "labels": LABELS, "concept_init": CONCEPT_INIT if SOFT else "none",
           "soft_student": SOFT, "alphabet": ALPHABET,
           "query_source": QUERY_SOURCE,
           "grounding": "grounding.txt",
           "strata_tau": STRATA_TAU if STRATA else None, "rules": rules_out}
    if ALPHABET == "word" and ALPHABET_PATH:
        man["alphabet_file"] = Path(ALPHABET_PATH).name
        try:
            from pil.alphabet import WordCodec as _WC
            man["alphabet_hash"] = _WC.from_file(ALPHABET_PATH).hash()
        except Exception:
            pass
    if derived_defs:
        man["derived"] = derived_defs
    if need_concepts and CONCEPTS_CMAP[0] is not None:
        cm = CONCEPTS_CMAP[0].cpu()
        groups = {}
        for i, rep in enumerate(cm.tolist()):
            if rep != i:
                groups.setdefault(int(uv[rep]), []).append(int(uv[i]))
        man["concepts"] = {str(rep): mem for rep, mem in groups.items()}
    return man, skipped

def main():
    ids, y, cls, uv, tr, te = load_ds()
    yv = cls[y]
    vocab, ell = len(uv), ids.shape[1]
    cs = v2.copy_subset(ids, y, cls, te)
    print(f"Wyly-LM v5 [{LIB}/{JUDGE}] -- teacher {TAG}, dataset {DS}, L={ell}, vocab {vocab}, {DEV}; "
          f"copy-subset {len(cs)}/{len(te)} ({len(cs) / len(te):.0%})")

    g = torch.Generator().manual_seed(WYLY_SEED)
    if VALREG:
        # STORY-LEVEL judging: val = a contiguous held-out region; fit excludes every window
        # overlapping its text (margin = window length) -- the tables and the judge never share
        # a story, so memorization shows ZERO val marginal and generalizing rules can win.
        val = tr[-v2.NVAL:]
        fit = tr[:max(0, len(tr) - v2.NVAL - (ids.shape[1] + 4))]   # margin >= window length
    else:
        val = tr[torch.randperm(len(tr), generator=g)[:v2.NVAL]]
        fit = tr[~torch.isin(tr, val)]                       # tables are fit on FIT ONLY -- fitting on
    # QUERY-SHAPED batches MUST be loaded BEFORE any candidate that mines slots / confs from
    # deployment (estate2 deployment-first slot, query-calibrated identity tables, etc.).
    # Loading them after candidate construction left QUERY_BATCHES empty at estate2 build time,
    # so "deployment-first" fell through to fit-window slots (' to'/' the') that fire on 0/1000
    # query tails -- perfect feature, ~0 cover-marginal (qa3 Band-A residual).
    ensure_query_batches(uv, cls)
    online_frames = []
    candidates = [(f"induction L={lm}", mir_induction_L(lm)) for lm in (1, 2, 3)]
    if LIB == "ext":                                         # tr (which contains val) leaks the val
        candidates += [(f"induction L={lm}", mir_induction_L(lm)) for lm in (4, 5)]
        for k in SAFE_KGRAMS(vocab):                         # windows' own suffix pairs into the
            if ONLINE:                                       # judge's val marginals
                for supp in (40,):                           # s40 ~ 2 corpus occurrences under the
                    FrameCls = TupleFrame if TUPLE else OnlineFrame
                    of = FrameCls(tuple(range(k, 0, -1)), vocab, DEV, minsupp=supp)
                    online_frames.append(of)                 # ~21x window overlap; offering s2 too
                    nm = f"kgram k={k} (online s{supp})"     # re-admits val-noise
                    candidates.append((nm, of.mirror()))
                    CONF_FNS[nm] = (lambda ids, of=of:
                                    of.table.lookup_conf(of._suffix_key(ids, False)))
                    EMIT_INFO[nm] = ("kgram", of)
                    RULE_SIZE[nm] = lambda of=of: len(of.table.k)
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
        for k in SAFE_KGRAMS(vocab):
            for supp in (40,):
                of = OnlineFrame(tuple(range(k, 0, -1)), vocab, DEV, minsupp=supp)
                online_frames.append(of)
                nm = f"kgram k={k} (online s{supp})"
                candidates.append((nm, of.mirror()))
                CONF_FNS[nm] = (lambda ids, of=of:
                                of.table.lookup_conf(of._suffix_key(ids, False)))
                EMIT_INFO[nm] = ("kgram", of)
                RULE_SIZE[nm] = lambda of=of: len(of.table.k)
        for off in (2, 3):
            tab, nk = fit_skip(ids, yv, fit, off)
            nm = f"skip o={off} [{nk}]"
            candidates.append((nm, mir_skip(tab, off)))
            register_conf(nm, tab, keyfn_offsets((off,), vocab))
            EMIT_INFO[nm] = ("skip", off, tab)
            RULE_SIZE[nm] = lambda nk=nk: nk
        for i, j, k in RELATION_GRID:
            candidates.append((f"relation eq({i},{j})c{k}", mir_relation(i, j, k)))
        mined = MinedGates(vocab, DEV)
        candidates.append(("mined frames (online)", mined.mirror()))
        CONF_FNS["mined frames (online)"] = mined.lookup_conf_all
        EMIT_INFO["mined frames (online)"] = ("mined", mined)
        RULE_SIZE["mined frames (online)"] = lambda: len(mined.t1.k) + len(mined.t2.k)
        if CONCEPTS:                                         # build-order 12+1: concept induction
            cspace = ConceptSpace(vocab, DEV)                # feeding class-anchored frames + a
            pooled_ref = {}                                  # pooled concept-counts tier

            def pooled_conf(w):
                if "v" not in pooled_ref:
                    return (torch.full((len(w),), -1, dtype=torch.long, device=w.device),
                            torch.full((len(w),), -1e9, device=w.device))
                am, conf, mx = pooled_ref["v"]
                t = cspace.cmap[w[:, -1]]
                a = torch.where(mx[t] >= 1, cls[am[t]], torch.full_like(w[:, -1], -1))
                c = torch.where(mx[t] >= 1, conf[t],
                                torch.full((len(w),), -1e9, device=w.device))
                return a, c

            CONF_FNS["concept counts (pooled)"] = pooled_conf
            candidates.append(("concept counts (pooled)", lambda w: pooled_conf(w)[0]))
            RULE_SIZE["concept counts (pooled)"] = lambda: cspace.n_concepts
            minedc = MinedGates(vocab, DEV, amap=[cspace.cmap])
            candidates.append(("mined cframes (online)", minedc.mirror()))
            CONF_FNS["mined cframes (online)"] = minedc.lookup_conf_all
            RULE_SIZE["mined cframes (online)"] = lambda: len(minedc.t1.k) + len(minedc.t2.k)
        else:
            cspace = minedc = None
        if POINTER:
            pointer = PointerRule(vocab, DEV,
                                  cmap_ref=[cspace.cmap] if cspace is not None else None)
            candidates.append(("pointer (l,lc cells)", pointer.mirror()))
            CONF_FNS["pointer (l,lc cells)"] = pointer.lookup_conf
            EMIT_INFO["pointer (l,lc cells)"] = ("pointer", pointer)
        else:
            pointer = None
        if TPOINTER and cspace is not None:
            tpointer = TransformPointer(vocab, DEV, counts_ref=[None], cls=cls,
                                        cmap_ref=[cspace.cmap])
            candidates.append(("tpointer (class->form)", tpointer.mirror()))
            CONF_FNS["tpointer (class->form)"] = tpointer.lookup_conf
        else:
            tpointer = None
        if CX:                                               # 15 v1: derived-feature gate (mate)
            _ts = load_codec()
            opl, cll = bracket_sets(uv, _ts)
            is_open = torch.zeros(vocab, dtype=torch.bool, device=DEV)
            is_close = torch.zeros(vocab, dtype=torch.bool, device=DEV)
            if opl:
                is_open[torch.tensor(opl, device=DEV)] = True
            if cll:
                is_close[torch.tensor(cll, device=DEV)] = True
            def add_dgate(nm_base, featfn, spec, min_keys=20):
                tab_, nk_ = fit_dgate_feature(ids, yv, fit, featfn, vocab)
                if nk_ < min_keys:                           # estate-class rules are SURGICAL:
                    return                                   # ~6 values x 1 query position
                nm_ = f"{nm_base} [{nk_}]"
                B2_ = vocab + 2

                def dconf(w, tab_=tab_, B2_=B2_, featfn=featfn):
                    f_ = featfn(w)
                    v_, c_ = tab_.lookup_conf((f_ + 1) * B2_ + w[:, -1])
                    miss = f_ < 0
                    return (torch.where(miss, torch.full_like(v_, -1), v_),
                            torch.where(miss, torch.full_like(c_, -1e9), c_))

                candidates.append((nm_, lambda w, dconf=dconf: dconf(w)[0]))
                CONF_FNS[nm_] = dconf
                RULE_SIZE[nm_] = lambda nk_=nk_: nk_
                EMIT_INFO[nm_] = ("dfeat", spec, tab_, B2_)
                print(f"CX: {nm_}")

            cap_set = torch.zeros(vocab, dtype=torch.bool, device=DEV)
            claus_set = torch.zeros(vocab, dtype=torch.bool, device=DEV)
            conn_set = torch.zeros(vocab, dtype=torch.bool, device=DEV)
            temp_set = torch.zeros(vocab, dtype=torch.bool, device=DEV)
            attr_set = torch.zeros(vocab, dtype=torch.bool, device=DEV)
            sent_set = torch.zeros(vocab, dtype=torch.bool, device=DEV)
            quot_set = torch.zeros(vocab, dtype=torch.bool, device=DEV)
            SUBS = {"dass", "weil", "ob", "wenn", "da", "obwohl", "damit", "denn",
                    "sondern", "als", "wie", "nachdem", "bevor", "sobald", "falls",
                    "that", "because", "although", "whether", "since", "unless", "whereas"}
            CONN = {"however", "therefore", "thus", "hence", "moreover", "furthermore",
                    "consequently", "nevertheless", "accordingly", "conversely", "indeed",
                    "namely", "instead", "otherwise", "meanwhile", "aber", "jedoch", "also",
                    "daher", "trotzdem", "deshalb", "dennoch"}
            TEMPO = {"subsequently", "afterwards", "thereafter", "meanwhile", "until",
                     "previously", "eventually", "finally", "initially", "thereupon",
                     "later", "earlier", "immediately", "danach", "zuvor", "schliesslich"}
            ATTR = {"argues", "claims", "says", "suggests", "believes", "maintains",
                    "asserts", "contends", "holds", "denies", "observes", "notes",
                    "concludes", "insists", "argued", "claimed", "said", "wrote"}
            for i in range(vocab):
                d = _ts.token_str(int(uv[i])).strip()
                if len(d) > 1 and d[0].isupper() and d.isalpha():
                    cap_set[i] = True
                dl = d.lower()
                if dl in SUBS:
                    claus_set[i] = True
                if dl in CONN:
                    conn_set[i] = True
                if dl in TEMPO:
                    temp_set[i] = True
                if dl in ATTR:
                    attr_set[i] = True
                if d in {".", "!", "?"}:
                    sent_set[i] = True
                if d in {'"', "``", "''"} or "\u201c" in repr(d) or "\u201d" in repr(d):
                    quot_set[i] = True
            if int(cap_set.sum()) >= 20:
                add_dgate("runiq gate",
                          lambda w, cs=cap_set: recent_member_feature(w, cs, unique=True),
                          ("recent-unique", {"members": cap_set}))
            if int(claus_set.sum()) >= 5:
                add_dgate("clause gate",
                          lambda w, cs=claus_set: recent_member_feature(w, cs),
                          ("recent-member", {"members": claus_set}))
            for nmb, mset in [("conn", conn_set), ("attrib", attr_set), ("tempo", temp_set)]:
                if int(mset.sum()) >= 5:
                    add_dgate(f"{nmb} gate",
                              lambda w, ms=mset: recent_member_feature(w, ms),
                              ("recent-member", {"members": mset}))
                    add_dgate(f"{nmb}-succ gate",
                              lambda w, ms=mset: at_pos(w, recent_member_pos(w, ms), succ=1),
                              ("recent-member", {"members": mset, "succ": 1}))
            if int(sent_set.sum()) >= 1:
                add_dgate("sincedot gate",
                          lambda w, ms=sent_set: since_member_feature(w, ms),
                          ("since-member", {"members": sent_set, "cap": 32}))
            if int(quot_set.sum()) >= 1:
                add_dgate("quoteparity gate",
                          lambda w, ms=quot_set: member_parity_feature(w, ms),
                          ("member-parity", {"members": quot_set}))
            if int(sent_set.sum()) >= 1:                     # DEEP DISCOURSE: rhetorical spans
                def senthead_pos(w, ms=sent_set):
                    return recent_member_pos(w, ms)

                add_dgate("senthead gate",
                          lambda w: at_pos(w, senthead_pos(w), succ=1),
                          ("recent-member", {"members": sent_set, "succ": 1}))
                add_dgate("prevsent-head gate",
                          lambda w: at_pos(w, prev_occ_pos(w, senthead_pos(w)), succ=1),
                          ("prev-occ", {"of_members": sent_set, "succ": 1}))

                def pair_key(w):                             # span-PAIR: (prev-head, cur-head)
                    return (at_pos(w, prev_occ_pos(w, senthead_pos(w)), succ=1),
                            at_pos(w, senthead_pos(w), succ=1))

                ptab, pnk = fit_dgate2(ids, yv, fit,
                                       lambda w: pair_key(w)[0], lambda w: pair_key(w)[1],
                                       vocab)
                if pnk >= 20:
                    nmp = f"sentpair gate [{pnk}]"
                    B2p = vocab + 2

                    def pconf(w, ptab=ptab, B2p=B2p):
                        fa, fb = pair_key(w)
                        m_ = (fa >= 0) & (fb >= 0)
                        v_, c_ = ptab.lookup_conf((fa + 1) * B2p + fb)
                        return (torch.where(m_, v_, torch.full_like(v_, -1)),
                                torch.where(m_, c_, torch.full_like(c_, -1e9)))

                    candidates.append((nmp, lambda w, pconf=pconf: pconf(w)[0]))
                    CONF_FNS[nmp] = pconf
                    RULE_SIZE[nmp] = lambda pnk=pnk: pnk
                    EMIT_INFO[nmp] = ("dgate2", ("sent-pair", {"members": sent_set}), ptab, B2p)
                    print(f"CX: {nmp}")
            if int(sent_set.sum()) >= 1 and int(quot_set.sum()) >= 1:
                add_dgate("dstate gate",
                          lambda w: dstate_feature(w, sent_set, quot_set),
                          ("dstate", {"members": sent_set, "quote_members": quot_set}))

                def dst(w):
                    return dstate_feature(w, sent_set, quot_set)

                def shead(w):
                    return at_pos(w, recent_member_pos(w, sent_set), succ=1)

                def pshead(w):
                    return at_pos(w, prev_occ_pos(w, recent_member_pos(w, sent_set)), succ=1)

                for nm2, fa, fb in [
                        ("dstate*head", dst, shead),
                        ("conn*head", (lambda w: recent_member_feature(w, conn_set)), shead),
                        ("conn*prevhead",
                         (lambda w: recent_member_feature(w, conn_set)), pshead)]:
                    t2, n2 = fit_dgate2(ids, yv, fit, fa, fb, vocab)
                    if n2 < 20:
                        continue
                    nmf = f"{nm2} gate [{n2}]"
                    B2q = vocab + 2

                    def qconf(w, t2=t2, B2q=B2q, fa=fa, fb=fb):
                        va, vb = fa(w), fb(w)
                        m_ = (va >= 0) & (vb >= 0)
                        v_, c_ = t2.lookup_conf((va + 1) * B2q + vb)
                        return (torch.where(m_, v_, torch.full_like(v_, -1)),
                                torch.where(m_, c_, torch.full_like(c_, -1e9)))

                    candidates.append((nmf, lambda w, qconf=qconf: qconf(w)[0]))
                    CONF_FNS[nmf] = qconf
                    RULE_SIZE[nmf] = lambda n2=n2: n2
                    print(f"CX: {nmf}")
            if int(attr_set.sum()) >= 5:                     # CLAIM/EVIDENCE: the claimant roles
                add_dgate("attrib-subj gate",
                          lambda w, ms=attr_set: at_pos(w, recent_member_pos(w, ms), succ=-1),
                          ("recent-member", {"members": attr_set, "succ": -1}))
                add_dgate("claim-echo gate",
                          lambda w, ms=attr_set: at_pos(
                              w, prev_occ_pos(w, (recent_member_pos(w, ms) - 1).clamp(min=-1)),
                              succ=1),
                          ("prev-occ", {"of_members": attr_set, "of_shift": -1, "succ": 1}))
            avoid_set = torch.zeros(vocab, dtype=torch.bool, device=DEV)
            for i in range(vocab):
                if _ts.token_str(int(uv[i])).strip() in {"?", "Q", "A", "Q:", "A:"}:
                    avoid_set[i] = True
            if int(cap_set.sum()) >= 5 and int(avoid_set.sum()) >= 1 \
                    and "estate" not in CONF_FNS:
                flat = ids[fit].reshape(-1)                  # SELF-GROUNDING (estate): entities
                nf = len(flat)                               # = cap tokens NOT predominantly in
                capocc = torch.bincount(flat[cap_set[flat]], minlength=vocab).float()
                dirty_pos = torch.zeros(nf, dtype=torch.bool, device=DEV)
                for jj in range(1, 4):                       # OR over shifts -- summing
                    dirty_pos[:nf - jj] |= avoid_set[flat[jj:]]   # bincounts double-counts
                mrows = cap_set[flat] & dirty_pos
                dirty = torch.bincount(flat[mrows], minlength=vocab).float()
                ent_set = (capocc >= 10) & (dirty / capocc.clamp(min=1) < 0.5)
                the_ids = [i for i in range(vocab)
                           if _ts.token_str(int(uv[i])).strip() == "the"]
                val_set = torch.zeros(vocab, dtype=torch.bool, device=DEV)
                if the_ids and int(ent_set.sum()) >= 2:
                    the_id = the_ids[0]                      # values = the transition slot:
                    ent_near = torch.zeros(nf, dtype=torch.bool, device=DEV)
                    epos = ent_set[flat]
                    for jj in range(1, 8):
                        ent_near[jj:] |= epos[:nf - jj]
                    vh = flat[1:][(flat[:-1] == the_id) & ent_near[1:]]
                    if len(vh) >= 50:
                        hist = torch.bincount(vh, minlength=vocab).float()
                        srt, order = hist.sort(descending=True)
                        cum = srt.cumsum(0) / hist.sum()
                        keep_n = int((cum < 0.95).sum()) + 1
                        val_set[order[:keep_n]] = True
                if int(val_set.sum()) >= 2:
                    samp = fit[:20000]                       # slot mining: where does the
                    f_s = estate_feature(ids[samp], ent_set, val_set, avoid_set)
                    hit = f_s == yv[samp]                    # register PREDICT next? -> the
                    if int(hit.sum()) >= 30:                 # dominant (t-2, t-1) = answer slot
                        pair = ids[samp][hit][:, -2] * (vocab + 1) + ids[samp][hit][:, -1]
                        top = pair.bincount().argmax()
                        slot = (int(top) // (vocab + 1), int(top) % (vocab + 1))
                        add_dgate("estate (entity-state register)",
                                  lambda w, es=ent_set, vs=val_set, av=avoid_set, sl=slot:
                                  estate_feature(w, es, vs, av, slot=sl),
                                  ("estate", {"entity_members": ent_set,
                                              "value_members": val_set,
                                              "avoid": avoid_set, "within": 7,
                                              "slot": slot}), min_keys=4)
            if ESTATE2 and "estate2" not in " ".join(n_ for n_, _ in candidates):
                # Fail loud if WYLY_ESTATE2 is set but queries never loaded -- the exact
                # regression that produced qa3 0.429 (fit-window slots on 0/1000 tails).
                if QUERIES:
                    ensure_query_batches(uv, cls, required=True)
                import json as _json
                e2 = _json.loads(Path(ESTATE2).read_text())
                tokmap = {}
                inv_tok = {}
                for i in range(vocab):
                    tokmap.setdefault(_ts.token_str(int(uv[i])).strip(), i)
                    inv_tok[int(uv[i])] = i
                def mkset(names):
                    m_ = torch.zeros(vocab, dtype=torch.bool, device=DEV)
                    for nm_ in names:
                        if nm_ in tokmap:
                            m_[tokmap[nm_]] = True
                        else:                                # multi-token word ('journeyed' =
                            ft = _ts.encode(" " + nm_)       # ' journey'+'ed'): its FIRST token
                            if ft and ft[0] in inv_tok:      # is the class-space signature
                                m_[inv_tok[ft[0]]] = True
                    return m_
                e2sets = {"ent": mkset(e2["entities"]), "loc": mkset(e2["locations"]),
                          "obj": mkset(e2["objects"]), "move": mkset(e2["move_verbs"]),
                          "take": mkset(e2["take_verbs"]), "drop": mkset(e2["drop_verbs"])}
                if all(int(v_.sum()) >= 2 for v_ in e2sets.values()):
                    for mode in ("is", "before"):
                        yv_q = None
                        if QUERY_BATCHES:                    # DEPLOYMENT-FIRST slot mining:
                            prs = []                         # the slot's job is to gate to
                            for qids_, qans_ in QUERY_BATCHES:   # answer positions -- mine it
                                f_q = estate2_feature(qids_, e2sets, mode=mode)
                                hq = f_q == qans_[:, 0]      # from the queries' own tails
                                if int(hq.sum()):            # (fit-window hits concentrate at
                                    prs.append(qids_[hq][:, -2] * (vocab + 1)
                                               + qids_[hq][:, -1])   # 'to the', gating to 0)
                            if not prs:
                                continue
                            pr2 = torch.cat(prs)
                            if len(pr2) < 30:
                                continue
                        else:
                            samp2 = fit[:20000]
                            f_s2 = estate2_feature(ids[samp2], e2sets, mode=mode)
                            hit2 = f_s2 == yv[samp2]
                            if int(hit2.sum()) < 30:
                                continue
                            pr2 = ids[samp2][hit2][:, -2] * (vocab + 1) + ids[samp2][hit2][:, -1]
                        t2 = pr2.bincount().argmax()
                        sl2 = (int(t2) // (vocab + 1), int(t2) % (vocab + 1))
                        _slot_src = "query" if QUERY_BATCHES else "fit-window"
                        print(f"ESTATE2 mode={mode}: slot=({_ts.token_str(int(uv[sl2[0]]))!r},"
                              f"{_ts.token_str(int(uv[sl2[1]]))!r}) from {_slot_src} "
                              f"({len(pr2)} hits)", flush=True)
                        def e2fn(w, sets_=e2sets, mode_=mode, sl_=sl2):
                            f_ = estate2_feature(w, sets_, mode=mode_)
                            oks = (w[:, -2] == sl_[0]) & (w[:, -1] == sl_[1])
                            return torch.where(oks, f_, torch.full_like(f_, -1))
                        # IDENTITY table, synthesized analytically: the fold's answer IS the
                        # prediction. A fit-statistics table only learns identity per-key --
                        # values whose keys miss the support/det filters fall through to counts
                        # and lose (qa2: a 4-of-6-key table capped the gate at ~0.52 while the
                        # feature was 0.836). Confidence = per-VALUE fired accuracy (C10).
                        B2e = vocab + 2
                        samp2c = fit[:20000]
                        f_all = e2fn(ids[samp2c])
                        fired_m = f_all >= 0
                        if QUERY_BATCHES and int(fired_m.sum()) < 50:
                            # slot gates to answer positions -- fit windows rarely end there;
                            # calibrate per-value confs on the queries instead
                            fq_all, fq_ans = [], []
                            for qids_, qans_ in QUERY_BATCHES:
                                f_q = e2fn(qids_)
                                fq_all.append(f_q)
                                fq_ans.append(qans_[:, 0])
                            f_all = torch.cat(fq_all)
                            yv_q = torch.cat(fq_ans)
                            fired_m = f_all >= 0
                        vals_e = torch.where(e2sets["loc"])[0]
                        keys_l, vals_l, confs_l = [], [], []
                        tgt_e = yv_q if yv_q is not None else yv[samp2c]
                        for v_ in vals_e.tolist():
                            mv_ = fired_m & (f_all == v_)
                            nv_ = int(mv_.sum())
                            acc_ = (float((tgt_e[mv_] == v_).float().mean())
                                    if nv_ >= 5 else
                                    float((tgt_e[fired_m]
                                           == f_all[fired_m]).float().mean()))
                            keys_l.append((v_ + 1) * B2e + sl2[1])
                            vals_l.append(v_)
                            confs_l.append(acc_)
                        tab_e = KeyTable(torch.tensor(keys_l, device=DEV),
                                         torch.tensor(vals_l, device=DEV),
                                         torch.tensor(confs_l, device=DEV))
                        nm_e = f"estate2/{mode} (world-state fold) [{len(keys_l)}]"

                        def e2conf(w, tab_=tab_e, B2_=B2e, featfn=e2fn):
                            f_ = featfn(w)
                            v_, c_ = tab_.lookup_conf((f_ + 1) * B2_ + w[:, -1])
                            miss = f_ < 0
                            return (torch.where(miss, torch.full_like(v_, -1), v_),
                                    torch.where(miss, torch.full_like(c_, -1e9), c_))

                        nkeys_e = len(keys_l)
                        candidates.append((nm_e, lambda w, c_=e2conf: c_(w)[0]))
                        CONF_FNS[nm_e] = e2conf
                        RULE_SIZE[nm_e] = lambda n_=nkeys_e: n_
                        EMIT_INFO[nm_e] = ("dfeat", ("estate2",
                                           {"sets": e2sets, "mode": mode, "slot": sl2}),
                                           tab_e, B2e)
            if int(cap_set.sum()) >= 20 and int(avoid_set.sum()) >= 1:
                for sc in (3, 4, 5):
                    add_dgate(f"move-echo s{sc} gate",
                              lambda w, cs=cap_set, av=avoid_set, sc=sc: at_pos(
                                  w, prev_occ_pos_filtered(
                                      w, recent_member_pos(w, cs), av), succ=sc),
                              ("prev-occ", {"of_members": cap_set, "avoid": avoid_set,
                                            "succ": sc}))
            if int(cap_set.sum()) >= 20:
                for sc in ECHO_SUCCS:
                    add_dgate(f"cap-echo s{sc} gate",
                              lambda w, cs=cap_set, sc=sc: at_pos(
                                  w, prev_occ_pos(w, recent_member_pos(w, cs)), succ=sc),
                              ("prev-occ", {"of_members": cap_set, "succ": sc}))
            if int(claus_set.sum()) >= 5:
                add_dgate("clause-succ gate",
                          lambda w, cs=claus_set: at_pos(w, recent_member_pos(w, cs), succ=1),
                          ("recent-member", {"members": claus_set, "succ": 1}))
            if opl and cll:
                add_dgate("mate-succ gate",
                          lambda w, o=is_open, c=is_close: at_pos(w, mate_pos(w, o, c), succ=1),
                          ("bracket-mate", {"openers": is_open, "closers": is_close, "succ": 1}))
                add_dgate("depth gate",
                          lambda w, o=is_open, c=is_close: depth_feature(w, o, c),
                          ("bracket-depth", {"openers": is_open, "closers": is_close, "cap": 8}))
                mtab, mnk = fit_mate(ids, yv, fit, is_open, is_close, vocab)
                nm = f"mate gate [{mnk}]"
                B2 = vocab + 2

                def mate_conf(w, mtab=mtab, B2=B2):
                    f = mate_feature(w, is_open, is_close)
                    q = (f + 1) * B2 + w[:, -1]
                    v, c = mtab.lookup_conf(q)
                    miss = f < 0
                    return (torch.where(miss, torch.full_like(v, -1), v),
                            torch.where(miss, torch.full_like(c, -1e9), c))

                candidates.append((nm, lambda w: mate_conf(w)[0]))
                CONF_FNS[nm] = mate_conf
                RULE_SIZE[nm] = lambda mnk=mnk: mnk
                EMIT_INFO[nm] = ("mate", mtab, opl, cll, B2)
                print(f"CX: mate gate over {len(opl)} openers/{len(cll)} closers, {mnk} keys")
    else:
        mined = None
    gate_cands = []
    if LIB == "gates":                                       # learned-frame family (gate kind)
        candidates += [(f"induction L={lm}", mir_induction_L(lm)) for lm in (4, 5)]
        for k in SAFE_KGRAMS(vocab):
            if ONLINE:
                for supp in (40,):
                    FrameCls = TupleFrame if TUPLE else OnlineFrame
                    of = FrameCls(tuple(range(k, 0, -1)), vocab, DEV, minsupp=supp)
                    online_frames.append(of)
                    nm = f"kgram k={k} (online s{supp})"
                    candidates.append((nm, of.mirror()))
                    CONF_FNS[nm] = (lambda ids, of=of:
                                    of.table.lookup_conf(of._suffix_key(ids, False)))
                    EMIT_INFO[nm] = ("kgram", of)
                    RULE_SIZE[nm] = lambda of=of: len(of.table.k)
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

    ground = concept_init(uv).to(DEV)
    ground = ground / ground.shape[1] ** 0.5
    torch.manual_seed(WYLY_SEED)
    model = (WylyGrow if GROW else WylyV3)(ground, cls.cpu(), ell).to(DEV)
    log = []
    wake_steps = v2.WAKE_STEPS if SOFT else 0
    print(f"SOFT={int(SOFT)} wake_steps={wake_steps} alphabet={ALPHABET} "
          f"origin={provenance_origin()}", flush=True)
    print(f"\n{'episode':>8}{'label-agree':>15}{'copy-agree':>12}{'tier':>6}")
    for ep, ch in enumerate(torch.chunk(fit, v2.EPISODES)):
        model.update_counts(ids[ch], y[ch], model.lut)
        for of in online_frames:
            of.update(ids[ch], yv[ch])
        for _ in range(wake_steps):                          # P0: SOFT=0 → counts-only wake
            bi = ch[torch.randint(len(ch), (v2.BATCH,), generator=g)]
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(ids[bi]), y[bi]).backward()
            model.sgd(v2.LR)
        for of in online_frames:
            of.refresh()                                     # re-gate the online tiers each sleep
        if mined is not None:
            if CONCEPTS and cspace is not None:              # concepts refresh BEFORE mining so
                au = mined.au_pairs() if DX else None        # 7: frame-tier interchangeability
                cspace.refresh(model.counts, extra_pairs=au)  # evidence joins the bigram rows
                if au:
                    log.append(f"    sleep {ep}: AU {len(au)} anchor pairs -> concept evidence")
                pooled_ref["v"] = pooled_counts(model.counts, cspace.cmap, vocab)
                CONCEPTS_CMAP[0] = cspace.cmap
                minedc.amap[0] = cspace.cmap                 # freshest concept space
                log.append(f"    sleep {ep}: CONCEPTS {cspace.n_concepts} "
                           f"({cspace.n_merged} classes merged)")
            mined.mine(model, ids, yv, cls, fit, g, log, ep)
            if CONCEPTS and minedc is not None:
                minedc.mine(model, ids, yv, cls, fit, g, log, ep)
            if GROW:                                         # 13+14: append compound rows
                nk = model.grow(mine_compounds(ids, ch, vocab))
                if nk:
                    log.append(f"    sleep {ep}: GROW +{nk} compound concepts "
                               f"(total {len(model.grow_keys)})")
            if CX and CONCEPTS and cspace is not None and ep >= 1 and "avoid_set" in dir():
                cm2 = cspace.cmap
                reps2, cnts2 = cm2.unique(return_counts=True)
                for rep in reps2[cnts2.argsort(descending=True)[:4]].tolist():
                    nm3 = f"moveloc [{rep}]"
                    if any(nm3 in n_ for n_, _ in model.rules) or nm3 in CONF_FNS:
                        continue
                    grp = torch.where(cm2 == rep)[0]
                    if len(grp) < 3:
                        continue
                    mset2 = torch.zeros(vocab, dtype=torch.bool, device=DEV)
                    mset2[grp] = True

                    def mfeat(w, ms=mset2):
                        return next_member_after(
                            w, prev_occ_pos_filtered(
                                w, recent_member_pos(w, cap_set), avoid_set), ms)

                    tab3, nk3 = fit_dgate_feature(ids, yv, fit, mfeat, vocab)
                    if nk3 < 20:
                        continue
                    B23 = vocab + 2

                    def mconf(w, tab3=tab3, B23=B23, mfeat=mfeat):
                        f_ = mfeat(w)
                        v_, c_ = tab3.lookup_conf((f_ + 1) * B23 + w[:, -1])
                        miss = f_ < 0
                        return (torch.where(miss, torch.full_like(v_, -1), v_),
                                torch.where(miss, torch.full_like(c_, -1e9), c_))

                    candidates.append((nm3, lambda w, mconf=mconf: mconf(w)[0]))
                    CONF_FNS[nm3] = mconf
                    RULE_SIZE[nm3] = lambda nk3=nk3: nk3
            if CX and CONCEPTS and cspace is not None and ep >= 1:
                cm = cspace.cmap                             # LEARNED member sets: top concept
                reps, cnts = cm.unique(return_counts=True)   # groups offered as recent-member
                for rep in reps[cnts.argsort(descending=True)[:6]].tolist():  # gates (sets
                    nm2 = f"cmember [{rep}]"                 # DISCOVERED, not seeded)
                    if any(nm2 in n for n, _ in model.rules) or nm2 in CONF_FNS:
                        continue
                    grp = torch.where(cm == rep)[0]
                    if len(grp) < 3:
                        continue
                    mset = torch.zeros(vocab, dtype=torch.bool, device=DEV)
                    mset[grp] = True
                    tab2, nk2 = fit_dgate_feature(
                        ids, yv, fit, lambda w, ms=mset: recent_member_feature(w, ms), vocab)
                    if nk2 < 20:
                        continue
                    B2_ = vocab + 2

                    def cconf(w, tab2=tab2, B2_=B2_, mset=mset):
                        f_ = recent_member_feature(w, mset)
                        v_, c_ = tab2.lookup_conf((f_ + 1) * B2_ + w[:, -1])
                        miss = f_ < 0
                        return (torch.where(miss, torch.full_like(v_, -1), v_),
                                torch.where(miss, torch.full_like(c_, -1e9), c_))

                    candidates.append((nm2, lambda w, cconf=cconf: cconf(w)[0]))
                    CONF_FNS[nm2] = cconf
                    RULE_SIZE[nm2] = lambda nk2=nk2: nk2
                    EMIT_INFO[nm2] = ("dfeat", ("recent-member", {"members": mset}), tab2,
                                      B2_)
            if CX and CONCEPTS:                              # 9: cluster-scoped anchor mining
                mined.mine_clustered(model, ids, yv, cls, fit, ground, g, log, ep)
            if DX:                                           # 8: rules retreat to where they are
                pr = mined.retreat(ids, yv, val)             # right -- key-level val pruning
                if CONCEPTS and minedc is not None:
                    pr += minedc.retreat(ids, yv, val)
                for of in online_frames:
                    pr += of.retreat(ids, yv, val)
                if pr:
                    log.append(f"    sleep {ep}: RETREAT pruned {pr} failing keys")
        if POINTER and pointer is not None:
            if pointer.cmap_ref is not None and cspace is not None:
                pointer.cmap_ref[0] = cspace.cmap
            pointer.refresh(ids, yv, val)
        if TPOINTER and tpointer is not None:
            tpointer.cmap_ref[0] = cspace.cmap
            tpointer.counts_ref[0] = model.counts
            tpointer.refresh(ids, yv, val)
            if ep in (0, 7):
                cells = sorted(tpointer.cells.items())
                log.append(f"    sleep {ep}: TPOINTER {len(tpointer.cells)} cells: "
                           + ", ".join(f"l={a},lc={b}:{c:.2f}" for (a, b), c in cells))
            if ep in (0, 7):
                cells = sorted(pointer.cells.items())
                log.append(f"    sleep {ep}: POINTER {len(pointer.cells)} cells: "
                           + ", ".join(f"l={a},lc={b}:{c:.2f}" for (a, b), c in cells))
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
    if QUERY_BATCHES and SWCOVER:
        # BACKWARD ELIMINATION on deployment queries: greedy forward admission can install a
        # locally-better heuristic that permanently SHADOWS a globally-better rule (qa2: the
        # pointer's stale-copy beat estate2 +0.301 vs +0.234 at sleep 0, then held its slot by
        # arbitration confidence). Stepwise selection: drop incumbents that the query folds say
        # the cover is better off without, then re-run admission for the unshadowed candidates.
        # OPTIMIZE THE EMITTED COVER: rules without EMIT_INFO never reach the package, so
        # selection over the in-learner cover optimizes an artifact we don't ship (babi2:
        # moveloc [0] held the in-learner numbers up while the served package sat at 0.498).
        dropped = [n_ for n_, _ in model.rules if n_ not in EMIT_INFO]
        if dropped:
            model.rules = [r_ for r_ in model.rules if r_[0] in EMIT_INFO]
            log.append(f"    dropped unemittable from selection: {dropped}")
        # QUERY-MISCALIBRATED incumbents: window-fit confidences can be junk-confident at
        # query tails (the Q-colon lesson, now inside stratum 1) -- a wrong incumbent that
        # outranks every challenger makes swap trials read flat. Same medicine as stratum-2
        # qualification: measure each rule's fired-accuracy ON QUERIES; evict below 0.5.
        for nm_, fn_ in list(model.rules):
            oks_ = tots_ = 0
            for qids_, qans_ in QUERY_BATCHES:
                if nm_ in CONF_FNS:
                    a_, _c = CONF_FNS[nm_](qids_)
                else:
                    a_ = fn_(qids_)
                f_ = a_ >= 0
                oks_ += int((a_[f_] == qans_[f_, 0]).sum())
                tots_ += int(f_.sum())
            if tots_ >= 20 and oks_ / tots_ < 0.5:
                model.rules = [r_ for r_ in model.rules if r_[0] != nm_]
                log.append(f"    evicted query-miscalibrated {nm_} "
                           f"(query fired-acc {oks_ / tots_:.2f} on {tots_})")
        # rank unadmitted EMITTABLE candidates by STANDALONE query value (cheap pre-filter)
        inn = {n_ for n_, _ in model.rules}
        pool = []
        for cn_, cf_ in cands:
            if cn_ in inn or cn_ not in EMIT_INFO:
                continue
            qa_ = query_agree(model, [(cn_, cf_)], cls, QUERY_BATCHES)
            pool.append((qa_, cn_, cf_))
        pool.sort(reverse=True)
        pool = pool[:5]
        base_q0 = query_agree(model, model.rules, cls, QUERY_BATCHES)
        if pool and pool[0][0] > base_q0 + 0.02:
            # RESTART FROM CHAMPION: a single candidate beats the entire incumbent cover on
            # deployment queries -- multi-incumbent local optima resist one-swap-at-a-time
            # escape (several window-calibrated rules each cap the trial), so rebuild greedily
            # on top of the champion instead.
            qa0, cn0, cf0 = pool[0]
            log.append(f"    RESTART from champion {cn0} "
                       f"(standalone {qa0:.3f} > cover {base_q0:.3f})")
            model.rules = [(cn0, cf0)]
            for sweep_ in range(6):
                if not sleep_admit_cover(model, ids, yv, cls, y, val, 80 + sweep_,
                                         cands, log):
                    break
        changed = True
        sweeps = 0
        while changed and sweeps < 6:
            changed = False
            sweeps += 1
            base_q = query_agree(model, model.rules, cls, QUERY_BATCHES)
            for nm_, _ in list(model.rules):                 # single-rule removals
                if len(model.rules) <= 1:
                    break
                trial = [r_ for r_ in model.rules if r_[0] != nm_]
                q2_ = query_agree(model, trial, cls, QUERY_BATCHES)
                if q2_ > base_q + 1e-9:
                    model.rules = trial
                    log.append(f"    backward-eliminated {nm_} "
                               f"(queries {base_q:.3f} -> {q2_:.3f})")
                    changed = True
                    break
            if not changed:                                  # SWAP moves: remove an incumbent
                done = False                                 # AND add a candidate jointly --
                for nm_, _ in list(model.rules):             # add-only and remove-only steps
                    for _qa, cn_, cf_ in pool:               # cannot escape the tie-shadowing
                        if cn_ in {n_ for n_, _ in model.rules}:
                            continue                         # local optimum (qa2 pointer vs
                        trial = [r_ for r_ in model.rules    # estate2 at conf 1.0)
                                 if r_[0] != nm_] + [(cn_, cf_)]
                        q2_ = query_agree(model, trial, cls, QUERY_BATCHES)
                        if q2_ > base_q + 0.005:
                            model.rules = trial
                            log.append(f"    swapped {nm_} -> {cn_} "
                                       f"(queries {base_q:.3f} -> {q2_:.3f})")
                            changed, done = True, True
                            break
                    if done:
                        break
            if changed:
                readd = sleep_admit_cover(model, ids, yv, cls, y, val, 90 + sweeps,
                                          cands, log)
                changed = changed or bool(readd)
    print("\n" + "\n".join(log))
    full = v2.top1(model, ids, y, te)
    norules = v2.top1(model, ids, y, te, use_rules=False)
    cs_full, cs_norules = v2.top1(model, ids, y, cs), v2.top1(model, ids, y, cs, use_rules=False)
    print(f"\nv5[{LIB}/{JUDGE}] {TAG}/{DS}: label-agreement[{LABELS}] {full:.3f} "
          f"(ablated {norules:.3f}, marginal {full - norules:+.3f})")
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
        import shutil
        sys.path.insert(0, str(REPO))
        ts = load_codec()
        man, skipped = emit_full(model, cls, uv, ts, vocab)
        pkg = REPO / "data" / ("wyly_expert_package_v5" if DS == "wikitext"
                              else f"wyly_expert_package_v5_{DS}{PKG_SUFFIX}")
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "manifest.json").write_text(json.dumps(man))
        if ALPHABET == "word" and ALPHABET_PATH:
            shutil.copy(ALPHABET_PATH, pkg / "alphabet.json")
            # also as bundle.tokenizer.json name for tools that only look for that file name
            shutil.copy(ALPHABET_PATH, pkg / "bundle.tokenizer.json")
        else:
            tok = Path(TOKJ) if TOKJ else (REPO.parent / "rosetta" / "models" / "pythia70m"
                                           / "bundle.tokenizer.json")
            shutil.copy(tok, pkg / "bundle.tokenizer.json")
        for corp_name in (f"corpus_{DS}.txt",
                          "corpus_babi_qa2.txt" if "babi2" in DS else "",
                          "corpus_babi_qa3.txt" if "babi3" in DS else ""):
            if not corp_name:
                continue
            corp = REPO / "data" / corp_name
            if corp.exists():
                shutil.copy(corp, pkg / "grounding.txt")
                break
        kinds = {}
        for r in man["rules"]:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"  package -> {pkg} (support-weighted; {man['n_rules']} rules: {kinds}"
              f"{'; NOT emitted: ' + str(skipped) if skipped else ''})")
        if os.environ.get("WYLY_PARITY") and QUERY_BATCHES:
            # PARITY DUMP: the live cover vs the just-emitted package on identical queries --
            # names the culprit when the served artifact underperforms the learner.
            from collections import Counter as _C
            sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))
            from serve_package import decide as _pdec
            from serve_package import load_package as _pload
            idi, ngr, mp_ = _pload(Path(pkg) / "manifest.json")
            inv_tok = {int(t): i for i, t in enumerate(uv.tolist())}
            l_ok = p_ok = both = n_ = 0
            culprit, l_win_p_lose = _C(), _C()
            for qids, qans in QUERY_BATCHES:
                _, predq = core_cover_sw(model, model.rules, qids, qans[:, 0], cls,
                                         torch.arange(len(qids), device=DEV),
                                         return_pred=True)
                for r in range(len(qids)):
                    gold_ = int(qans[r, 0])
                    lr = int(predq[r]) == gold_
                    ctxt = [int(uv[c]) for c in qids[r].tolist()]
                    d_ = _pdec(ctxt, idi, ngr, mp_)
                    pans = inv_tok.get(d_["answer"], -1) if d_ else -1
                    pr = pans == gold_
                    l_ok += lr
                    p_ok += pr
                    both += lr and pr
                    n_ += 1
                    if lr and not pr:
                        key_ = (d_.get("circuit") or d_.get("rule")
                                or f"k={d_.get('k')}" if d_ else "ABSTAIN")
                        cite_ = (d_.get("citation", "") if d_ else "")
                        cite_ = cite_ if isinstance(cite_, str) else "; ".join(cite_)
                        culprit[str(key_)] += 1
                        l_win_p_lose[cite_[:60]] += 1
            print(f"  PARITY: learner {l_ok}/{n_} = {l_ok / n_:.3f} | package "
                  f"{p_ok}/{n_} = {p_ok / n_:.3f} | both {both}")
            for k_, v_ in culprit.most_common(6):
                print(f"    learner-right/package-wrong via {k_}: {v_}")
            for k_, v_ in l_win_p_lose.most_common(4):
                print(f"      cite: {k_}: {v_}")


if __name__ == "__main__":
    main()
