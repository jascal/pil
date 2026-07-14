"""German R1 — per-token German transduction student + register hard-layer.

Prereg (PREREG_GERMAN_EXPERT.md §R1): GSD train gold only; tune on dev; score test
exactly once. SOFT=0 — pure core_cover_sw arbitration over count tables + registers.
No wake/SGD. Government coverage (#4 prep, #5 verb) is PARTIAL (span / stem heuristics,
no dependency parse) — the morph_case register marginal is therefore a LOWER BOUND on
what a full-government register layer (R3 attachment) would deliver, not the ceiling.

Enrichments (fairer register marginal, still lower-bound):
  narrow-and-arbitrate on ambiguous declension forms; NP-span government incl. two-way
  preps + verb stem/prefix match; morph_gnn paradigm candidate sets.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
import wyly_lm_v5 as v5  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (read-only inputs outside repo; write only data/german_r1.json)
# ---------------------------------------------------------------------------
GDATA = Path("/home/allans/code/germandata")
TASKS = GDATA / "tasks"
REGS = GDATA / "registers"
OUTPUT = REPO / "data" / "german_r1.json"

UPOS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
CASE_LABELS = ["Nom", "Acc", "Dat", "Gen", "-"]
# Engineering safety valve for NP-span government scans (not a linguistic claim).
GOV_SPAN_SAFETY_CAP = 20
# Legacy alias kept so older callers/tests referring to a window bound still resolve.
GOV_WINDOW_N = GOV_SPAN_SAFETY_CAP
# Minimum verb stem length after stripping infinitive suffix (guard against over-match).
VERB_STEM_MIN_LEN = 3
# Adjective-ending gate: apply suffix case only when the form's TRAIN majority
# UPOS is ADJ with support >= ADJ_GATE_MINSUPP (gold-independent at test time).
ADJ_GATE_MINSUPP = 2
# Context-tier (prev1,cur) support/determinism candidates — tuned on DEV only.
CTX_GRID = [
    (2, 0.5), (3, 0.5), (5, 0.5), (5, 0.6),
    (8, 0.5), (8, 0.6), (10, 0.5), (10, 0.7),
]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

HONEST_SCOPE = (
    "Government coverage (prep-case + verb-government) in R1 is PARTIAL and parse-heuristic: "
    f"NP-span-to-clause-boundary is a right-edge/left-edge scan (stop at punct / next prep / "
    f"coordinating conj; safety cap N={GOV_SPAN_SAFETY_CAP} is an engineering valve, not a "
    "linguistic claim), not a dependency parse. Verb matching is a surface stem/prefix "
    "heuristic (lemma minus trailing -en/-n, longest stem wins) and still misses "
    "ablaut-changed strong-verb inflections (e.g. geben/gibt/gab/gegeben). "
    "The morph_case register marginal is a LOWER BOUND on what a full parse-based "
    "government layer (R3 attachment) would deliver, not the ceiling."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_aligned_split(split: str) -> list[dict]:
    """Load pos + morph_case + morph_gnn for a split; verify sent_id/token alignment."""
    pos = load_jsonl(TASKS / "pos" / f"{split}.jsonl")
    case = load_jsonl(TASKS / "morph_case" / f"{split}.jsonl")
    gnn = load_jsonl(TASKS / "morph_gnn" / f"{split}.jsonl")
    if not (len(pos) == len(case) == len(gnn)):
        raise RuntimeError(
            f"{split}: length mismatch pos={len(pos)} case={len(case)} gnn={len(gnn)}"
        )
    out = []
    for p, c, g in zip(pos, case, gnn, strict=True):
        if p["sent_id"] != c["sent_id"] or p["sent_id"] != g["sent_id"]:
            raise RuntimeError(
                f"{split}: sent_id mismatch {p['sent_id']!r} / {c['sent_id']!r} / {g['sent_id']!r}"
            )
        if p["tokens"] != c["tokens"] or p["tokens"] != g["tokens"]:
            raise RuntimeError(f"{split}: token mismatch on {p['sent_id']}")
        n = len(p["tokens"])
        if not (len(p["targets"]["upos"]) == len(c["targets"]["case"])
                == len(g["targets"]["gnn"]) == n):
            raise RuntimeError(f"{split}: target length mismatch on {p['sent_id']}")
        out.append({
            "sent_id": p["sent_id"],
            "text": p.get("text", ""),
            "tokens": p["tokens"],
            "upos": p["targets"]["upos"],
            "case": c["targets"]["case"],
            "gnn": g["targets"]["gnn"],
        })
    return out


def data_hash() -> str:
    """sha256 over concatenation of the 9 task jsonl files (pos/case/gnn × train/dev/test)."""
    h = hashlib.sha256()
    for task in ("pos", "morph_case", "morph_gnn"):
        for split in ("train", "dev", "test"):
            path = TASKS / task / f"{split}.jsonl"
            h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Register loading + inversion
# ---------------------------------------------------------------------------

def load_register(name: str) -> dict:
    return json.loads((REGS / f"{name}.json").read_text(encoding="utf-8"))


def invert_article_paradigm(table: dict) -> dict[str, set[str]]:
    """form -> set of case values realizable by that form."""
    form2case: dict[str, set[str]] = defaultdict(set)
    for _det_type, cases in table.items():
        for case, gmap in cases.items():
            for _g, form in gmap.items():
                form2case[form].add(case)
    return dict(form2case)


def invert_pronoun_paradigm(table: dict) -> dict[str, set[str]]:
    form2case: dict[str, set[str]] = defaultdict(set)
    for _png, cases in table.items():
        for case, form in cases.items():
            form2case[form].add(case)
    return dict(form2case)


def invert_adjective_endings(table: dict) -> dict[str, set[str]]:
    end2case: dict[str, set[str]] = defaultdict(set)
    for _decl, cases in table.items():
        for case, gmap in cases.items():
            for _g, ending in gmap.items():
                end2case[ending].add(case)
    return dict(end2case)


def gender_key_to_gnn(gender_key: str) -> str:
    """Map paradigm gender_key (Masc/Fem/Neut/Plur) to morph_gnn label."""
    if gender_key == "Plur":
        return "-|Plur"
    if gender_key in ("Masc", "Fem", "Neut"):
        return f"{gender_key}|Sing"
    return "-|-"


def invert_article_paradigm_gnn(table: dict) -> dict[str, set[str]]:
    """form -> set of GNN labels reachable across (declension, case, gender_key)."""
    form2gnn: dict[str, set[str]] = defaultdict(set)
    for _det_type, cases in table.items():
        for _case, gmap in cases.items():
            for gkey, form in gmap.items():
                form2gnn[form].add(gender_key_to_gnn(gkey))
    return dict(form2gnn)


def invert_adjective_endings_gnn(table: dict) -> dict[str, set[str]]:
    end2gnn: dict[str, set[str]] = defaultdict(set)
    for _decl, cases in table.items():
        for _case, gmap in cases.items():
            for gkey, ending in gmap.items():
                end2gnn[ending].add(gender_key_to_gnn(gkey))
    return dict(end2gnn)


def png_key_to_gnn(png_key: str) -> str:
    """Map pronoun person_number_gender key to morph_gnn label."""
    if "Masc" in png_key:
        gender = "Masc"
    elif "Fem" in png_key:
        gender = "Fem"
    elif "Neut" in png_key:
        gender = "Neut"
    else:
        gender = "-"
    if "_Plur" in png_key or png_key == "2_Formal":
        number = "Plur"
    else:
        number = "Sing"
    return f"{gender}|{number}"


def invert_pronoun_paradigm_gnn(table: dict) -> dict[str, set[str]]:
    """form -> set of GNN labels (surface-keyed; dual-variant union at lookup)."""
    form2gnn: dict[str, set[str]] = defaultdict(set)
    for png, cases in table.items():
        gnn = png_key_to_gnn(png)
        for _case, form in cases.items():
            form2gnn[form].add(gnn)
    return dict(form2gnn)


def strict_prepositions(table: dict) -> dict[str, str]:
    """Single-case prepositions only."""
    core = table["core"]
    return {prep: cases[0] for prep, cases in core.items() if len(cases) == 1}


def two_way_prepositions(table: dict) -> dict[str, list[str]]:
    """Two-way (typically Acc/Dat) prepositions -> full case list."""
    core = table["core"]
    return {prep: list(cases) for prep, cases in core.items() if len(cases) == 2}


def all_preposition_forms(table: dict) -> set[str]:
    return set(table["core"].keys())


def verb_majority_case(entry: dict) -> tuple[str | None, float]:
    """Majority obj:Case with Laplace conf cnt/(tot+ALPHA)."""
    counts = entry.get("counts", {})
    tot = int(entry.get("tot", 0))
    case_counts: dict[str, int] = defaultdict(int)
    for k, n in counts.items():
        if k.startswith("obj:") and len(k) > 4:
            case_counts[k[4:]] += int(n)
    if not case_counts:
        return None, 0.0
    best = max(case_counts.items(), key=lambda kv: kv[1])
    conf = best[1] / (tot + v5.ALPHA)
    return best[0], conf


def verb_stem(lemma: str, min_len: int = VERB_STEM_MIN_LEN) -> str | None:
    """Strip trailing infinitive -en / -n; guard minimum stem length."""
    if lemma.endswith("en") and len(lemma) - 2 >= min_len:
        return lemma[:-2]
    if lemma.endswith("n") and len(lemma) - 1 >= min_len:
        return lemma[:-1]
    if len(lemma) >= min_len:
        return lemma
    return None


def build_verb_stem_index(verb_table: dict) -> list[tuple[str, str, dict]]:
    """(stem, lemma, entry) list; longest stem preferred at match time."""
    items: list[tuple[str, str, dict]] = []
    for lemma, entry in verb_table.items():
        stem = verb_stem(lemma)
        if stem is not None:
            items.append((stem, lemma, entry))
    items.sort(key=lambda x: len(x[0]), reverse=True)
    return items


def match_verb_stem(
    token: str, stem_index: list[tuple[str, str, dict]],
) -> tuple[str, dict] | None:
    """Longest stem prefix match on token.lower(); returns (lemma, entry) or None."""
    low = token.lower()
    best: tuple[str, dict] | None = None
    best_len = -1
    for stem, lemma, entry in stem_index:
        if low.startswith(stem) and len(stem) > best_len:
            best = (lemma, entry)
            best_len = len(stem)
    return best


# ---------------------------------------------------------------------------
# Vocab / encoding
# ---------------------------------------------------------------------------

class Vocab:
    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: list[str] = []

    def add(self, tok: str) -> int:
        if tok not in self.stoi:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
        return self.stoi[tok]

    def get(self, tok: str, default: int | None = None) -> int:
        if tok in self.stoi:
            return self.stoi[tok]
        if default is not None:
            return default
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)


def build_vocab(train: list[dict], extra_forms: list[str]) -> Vocab:
    v = Vocab()
    v.add(PAD_TOKEN)
    v.add(UNK_TOKEN)
    for s in train:
        for t in s["tokens"]:
            v.add(t)
            v.add(t.lower())
    for f in extra_forms:
        v.add(f)
        v.add(f.lower())
    return v


def label_maps(train: list[dict]) -> dict[str, Any]:
    pos2i = {p: i for i, p in enumerate(UPOS)}
    case2i = {c: i for i, c in enumerate(CASE_LABELS)}
    gnn_set: set[str] = set()
    for s in train:
        gnn_set.update(s["gnn"])
    # ensure null present
    gnn_set.add("-|-")
    gnn_labels = sorted(gnn_set)
    gnn2i = {g: i for i, g in enumerate(gnn_labels)}
    return {
        "pos2i": pos2i, "i2pos": UPOS,
        "case2i": case2i, "i2case": CASE_LABELS,
        "gnn2i": gnn2i, "i2gnn": gnn_labels,
    }


# ---------------------------------------------------------------------------
# Flatten sentences → token stream + government spans
# ---------------------------------------------------------------------------

def flatten_labels(split: list[dict], key: str, lab2i: dict[str, int]) -> torch.Tensor:
    labs = []
    for s in split:
        for lab in s[key]:
            labs.append(lab2i[lab])
    return torch.tensor(labs, dtype=torch.long)


def is_punct_token(tok: str) -> bool:
    if not tok:
        return True
    return all(not ch.isalnum() for ch in tok)


def np_span_indices(
    tokens: list[str],
    start: int,
    prep_forms: set[str],
    conj_forms: set[str],
    safety_cap: int = GOV_SPAN_SAFETY_CAP,
) -> list[int]:
    """Eligible token indices in an NP-ish run from `start`, stopping at clause boundary.

    Boundary = punctuation OR prep-register form OR coordinating-conjunction form.
    Conjunctions *terminate* the span (not skipped). Safety cap is an engineering valve.
    """
    indices: list[int] = []
    end = min(start + safety_cap, len(tokens))
    for j in range(start, end):
        t = tokens[j]
        if is_punct_token(t):
            break
        lj = t.lower()
        if lj in prep_forms or lj in conj_forms:
            break
        indices.append(j)
    return indices


def prep_gov_case_sets(
    tokens: list[str],
    strict_prep: dict[str, str],
    two_way_prep: dict[str, list[str]],
    prep_forms: set[str],
    conj_forms: set[str],
    safety_cap: int = GOV_SPAN_SAFETY_CAP,
) -> list[set[str] | None]:
    """Per-token candidate case sets from prep government over NP spans.

    Strict preps contribute a singleton set; two-way preps contribute their full
    case list (typically {Acc, Dat}). Multiple governors on one token are collected
    for later intersection by the caller (here: last-write per governor is unioned
    into a list of source sets via multi-source collection at stream level).
    """
    # Collect list of source sets per position, then intersect.
    sources: list[list[set[str]]] = [[] for _ in tokens]
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in strict_prep:
            case_set = {strict_prep[low]}
        elif low in two_way_prep:
            case_set = set(two_way_prep[low])
        else:
            continue
        for j in np_span_indices(tokens, i + 1, prep_forms, conj_forms, safety_cap):
            sources[j].append(set(case_set))
    out: list[set[str] | None] = []
    for srcs in sources:
        out.append(_intersect_str_sets(srcs))
    return out


def verb_gov_case_sets(
    tokens: list[str],
    stem_index: list[tuple[str, str, dict]],
    prep_forms: set[str],
    conj_forms: set[str],
    safety_cap: int = GOV_SPAN_SAFETY_CAP,
) -> list[set[str] | None]:
    """Per-token candidate case sets from verb government (stem match + NP span)."""
    sources: list[list[set[str]]] = [[] for _ in tokens]
    for i, tok in enumerate(tokens):
        matched = match_verb_stem(tok, stem_index)
        if matched is None:
            continue
        _lemma, entry = matched
        case, _conf = verb_majority_case(entry)
        if case is None:
            continue
        case_set = {case}
        for j in np_span_indices(tokens, i + 1, prep_forms, conj_forms, safety_cap):
            sources[j].append(set(case_set))
    out: list[set[str] | None] = []
    for srcs in sources:
        out.append(_intersect_str_sets(srcs))
    return out


def _intersect_str_sets(srcs: list[set[str]]) -> set[str] | None:
    if not srcs:
        return None
    inter = set(srcs[0])
    for s in srcs[1:]:
        inter &= s
    if not inter:
        return None  # conflicting sources -> unconstrained
    return inter


def _intersect_int_sets(srcs: list[set[int]]) -> set[int] | None:
    if not srcs:
        return None
    inter = set(srcs[0])
    for s in srcs[1:]:
        inter &= s
    if not inter:
        return None
    return inter


# ---------------------------------------------------------------------------
# Candidate-set sources (declension forms + endings)
# ---------------------------------------------------------------------------

def lookup_pronoun_case_set(tok: str, pro_f2c: dict[str, set[str]]) -> set[str] | None:
    """Surface-first pronoun lookup; do not conflate Sie vs sie."""
    if tok in pro_f2c:
        return set(pro_f2c[tok])
    low = tok.lower()
    if low != tok and low in pro_f2c:
        return set(pro_f2c[low])
    return None


def lookup_pronoun_gnn_set(tok: str, pro_f2g: dict[str, set[str]]) -> set[str] | None:
    """Pronoun GNN lookup: union surface + lowercase when they differ (Sie ambiguity)."""
    parts: list[set[str]] = []
    if tok in pro_f2g:
        parts.append(pro_f2g[tok])
    low = tok.lower()
    if low in pro_f2g:
        parts.append(pro_f2g[low])
    if not parts:
        return None
    out: set[str] = set()
    for p in parts:
        out |= p
    return out


def match_adj_ending(
    form: str,
    endings_sorted: list[str],
    end2labels: dict[str, set[str]],
    form_majority_pos: dict[str, str],
    form_pos_supp: dict[str, int],
) -> set[str] | None:
    """ADJ-gated ending match; returns full label set for the matched ending."""
    if form_majority_pos.get(form) != "ADJ":
        return None
    if form_pos_supp.get(form, 0) < ADJ_GATE_MINSUPP:
        return None
    low = form.lower()
    for e in endings_sorted:
        if low.endswith(e) and len(low) > len(e):
            return set(end2labels[e])
    return None


def form_case_candidate_sets(
    tokens: list[str],
    art_f2c: dict[str, set[str]],
    pro_f2c: dict[str, set[str]],
    end2c: dict[str, set[str]],
    form_majority_pos: dict[str, str],
    form_pos_supp: dict[str, int],
) -> list[list[set[str]]]:
    """Per position, list of contributing case candidate sets from declension registers."""
    endings_sorted = sorted(end2c.keys(), key=len, reverse=True)
    sources: list[list[set[str]]] = [[] for _ in tokens]
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in art_f2c:
            sources[i].append(set(art_f2c[low]))
        pro = lookup_pronoun_case_set(tok, pro_f2c)
        if pro is not None:
            sources[i].append(pro)
        adj = match_adj_ending(tok, endings_sorted, end2c, form_majority_pos, form_pos_supp)
        if adj is not None:
            sources[i].append(adj)
    return sources


def form_gnn_candidate_sets(
    tokens: list[str],
    art_f2g: dict[str, set[str]],
    pro_f2g: dict[str, set[str]],
    end2g: dict[str, set[str]],
    form_majority_pos: dict[str, str],
    form_pos_supp: dict[str, int],
) -> list[list[set[str]]]:
    """Per position, list of contributing GNN candidate sets from paradigm registers."""
    endings_sorted = sorted(end2g.keys(), key=len, reverse=True)
    sources: list[list[set[str]]] = [[] for _ in tokens]
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in art_f2g:
            sources[i].append(set(art_f2g[low]))
        pro = lookup_pronoun_gnn_set(tok, pro_f2g)
        if pro is not None:
            sources[i].append(pro)
        adj = match_adj_ending(tok, endings_sorted, end2g, form_majority_pos, form_pos_supp)
        if adj is not None:
            sources[i].append(adj)
    return sources


def merge_candidate_sources(
    *source_lists: list[list[set[str]]],
    lab2i: dict[str, int],
) -> list[set[int] | None]:
    """Intersect all hard-prune sources per position; empty intersection -> unconstrained."""
    n = len(source_lists[0])
    out: list[set[int] | None] = []
    for i in range(n):
        str_sets: list[set[str]] = []
        for src in source_lists:
            str_sets.extend(src[i])
        if not str_sets:
            out.append(None)
            continue
        inter = set(str_sets[0])
        for s in str_sets[1:]:
            inter &= s
        if not inter:
            out.append(None)
            continue
        ids = {lab2i[lab] for lab in inter if lab in lab2i}
        out.append(ids if ids else None)
    return out


def build_stream(
    split: list[dict],
    vocab: Vocab,
    strict_prep: dict[str, str],
    two_way_prep: dict[str, list[str]],
    prep_forms: set[str],
    conj_forms: set[str],
    verb_stem_index: list[tuple[str, str, dict]],
) -> dict[str, Any]:
    """Flatten split to per-token tensors + government candidate-set precomputes."""
    surf: list[int] = []
    low: list[int] = []
    tokens_flat: list[str] = []
    prep_gov: list[set[str] | None] = []
    verb_gov: list[set[str] | None] = []

    for s in split:
        toks = s["tokens"]
        pg = prep_gov_case_sets(toks, strict_prep, two_way_prep, prep_forms, conj_forms)
        vg = verb_gov_case_sets(toks, verb_stem_index, prep_forms, conj_forms)
        for t, gcase, vcase in zip(toks, pg, vg, strict=True):
            surf.append(vocab.get(t))
            low.append(vocab.get(t.lower()))
            tokens_flat.append(t)
            prep_gov.append(gcase)
            verb_gov.append(vcase)

    n = len(surf)
    pad = vocab.get(PAD_TOKEN)
    # windows: [stream_idx, cur_low, prev2, prev1, cur_surf]
    windows = torch.zeros(n, 5, dtype=torch.long)
    for i in range(n):
        windows[i, 0] = i
        windows[i, 1] = low[i]
        windows[i, 2] = surf[i - 2] if i >= 2 else pad
        windows[i, 3] = surf[i - 1] if i >= 1 else pad
        windows[i, 4] = surf[i]

    return {
        "windows": windows,
        "surf": torch.tensor(surf, dtype=torch.long),
        "low": torch.tensor(low, dtype=torch.long),
        "tokens": tokens_flat,
        "prep_gov": prep_gov,
        "verb_gov": verb_gov,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Count tables / KeyTables
# ---------------------------------------------------------------------------

def build_counts(form_ids: torch.Tensor, labels: torch.Tensor, V: int, C: int) -> torch.Tensor:
    counts = torch.zeros(V, C, dtype=torch.long)
    pair = form_ids * C + labels
    uniq, cnt = pair.unique(return_counts=True)
    counts.view(-1).index_add_(0, uniq, cnt)
    return counts


def build_keyed_counts(
    keys: torch.Tensor, labels: torch.Tensor, n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact per-key × class count matrix for arbitrary integer keys (e.g. bigrams).

    Returns (sorted_unique_keys [K], counts [K, C]).
    """
    if len(keys) == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.zeros(0, n_classes, dtype=torch.long),
        )
    ukeys, inv = keys.unique(return_inverse=True)
    counts = torch.zeros(len(ukeys), n_classes, dtype=torch.long)
    pair = inv * n_classes + labels
    uniq, cnt = pair.unique(return_counts=True)
    counts.view(-1).index_add_(0, uniq, cnt)
    # ukeys from unique() are sorted
    return ukeys, counts


def lookup_keyed_rows(
    sorted_keys: torch.Tensor,
    counts: torch.Tensor,
    query_keys: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """Map query keys to count rows; unknown keys -> zero row."""
    n = len(query_keys)
    if len(sorted_keys) == 0:
        return torch.zeros(n, n_classes, dtype=torch.long)
    idx = torch.searchsorted(sorted_keys, query_keys).clamp(max=len(sorted_keys) - 1)
    hit = sorted_keys[idx] == query_keys
    rows = counts[idx]
    rows = torch.where(hit.unsqueeze(1), rows, torch.zeros_like(rows))
    return rows


def keytable_unambiguous(
    form2cases: dict[str, set[str]],
    vocab: Vocab,
    lab2i: dict[str, int],
    conf: float = 0.93,
) -> v5.KeyTable:
    """Only forms that map to a single case fire. (Legacy helper; case path no longer uses it.)"""
    keys, vals, confs = [], [], []
    for form, cases in form2cases.items():
        if len(cases) != 1:
            continue
        case = next(iter(cases))
        if case not in lab2i:
            continue
        if form.lower() in vocab.stoi:
            fid = vocab.stoi[form.lower()]
        elif form in vocab.stoi:
            fid = vocab.stoi[form]
        else:
            continue
        keys.append(fid)
        vals.append(lab2i[case])
        confs.append(conf)
    if not keys:
        return v5.KeyTable(
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float),
        )
    return v5.KeyTable(
        torch.tensor(keys, dtype=torch.long),
        torch.tensor(vals, dtype=torch.long),
        torch.tensor(confs, dtype=torch.float),
    )


def keytable_det_pron(table: dict, vocab: Vocab, pos2i: dict[str, int]) -> v5.KeyTable:
    keys, vals, confs = [], [], []
    for form, entry in table.items():
        counts = entry.get("counts", {})
        tot = int(entry.get("tot", 0))
        if not counts:
            continue
        best_pos, best_n = max(counts.items(), key=lambda kv: kv[1])
        if best_pos not in pos2i:
            continue
        fid = vocab.stoi.get(form.lower(), vocab.stoi.get(form))
        if fid is None:
            continue
        conf = best_n / (tot + v5.ALPHA)
        keys.append(fid)
        vals.append(pos2i[best_pos])
        confs.append(conf)
    if not keys:
        return v5.KeyTable(
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float),
        )
    return v5.KeyTable(
        torch.tensor(keys, dtype=torch.long),
        torch.tensor(vals, dtype=torch.long),
        torch.tensor(confs, dtype=torch.float),
    )


def keytable_from_position_preds(
    n: int,
    preds: list[str | None] | list[tuple[str | None, float]],
    lab2i: dict[str, int],
    default_conf: float = 0.90,
) -> v5.KeyTable:
    """Legacy position KeyTable builder (kept for tests / compatibility)."""
    keys, vals, confs = [], [], []
    for i, p in enumerate(preds):
        if isinstance(p, tuple):
            case, conf = p
            if case is None or case not in lab2i:
                continue
            keys.append(i)
            vals.append(lab2i[case])
            confs.append(float(conf))
        else:
            if p is None or p not in lab2i:
                continue
            keys.append(i)
            vals.append(lab2i[p])
            confs.append(default_conf)
    if not keys:
        return v5.KeyTable(
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float),
        )
    return v5.KeyTable(
        torch.tensor(keys, dtype=torch.long),
        torch.tensor(vals, dtype=torch.long),
        torch.tensor(confs, dtype=torch.float),
    )


def adj_ending_table(
    end2case: dict[str, set[str]],
    vocab: Vocab,
    case2i: dict[str, int],
    form_majority_pos: dict[str, str],
    form_pos_supp: dict[str, int],
    conf: float = 0.85,
) -> v5.KeyTable:
    """Legacy unique-ending KeyTable (case path now uses candidate sets)."""
    uniq_end = {e: next(iter(cs)) for e, cs in end2case.items() if len(cs) == 1}
    endings = sorted(uniq_end.keys(), key=len, reverse=True)
    keys, vals, confs = [], [], []
    for form, maj in form_majority_pos.items():
        if maj != "ADJ":
            continue
        if form_pos_supp.get(form, 0) < ADJ_GATE_MINSUPP:
            continue
        low = form.lower()
        matched = None
        for e in endings:
            if low.endswith(e) and len(low) > len(e):
                matched = e
                break
        if matched is None:
            continue
        case = uniq_end[matched]
        fid = vocab.stoi.get(form, vocab.stoi.get(low))
        if fid is None:
            continue
        keys.append(fid)
        vals.append(case2i[case])
        confs.append(conf)
    if not keys:
        return v5.KeyTable(
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float),
        )
    return v5.KeyTable(
        torch.tensor(keys, dtype=torch.long),
        torch.tensor(vals, dtype=torch.long),
        torch.tensor(confs, dtype=torch.float),
    )


def train_form_majority_pos(train: list[dict]) -> tuple[dict[str, str], dict[str, int]]:
    """Per surface form, majority UPOS and its support (for ADJ gate)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for s in train:
        for t, u in zip(s["tokens"], s["upos"], strict=True):
            counts[t][u] += 1
    maj: dict[str, str] = {}
    supp: dict[str, int] = {}
    for form, c in counts.items():
        lab, n = c.most_common(1)[0]
        maj[form] = lab
        supp[form] = n
    return maj, supp


# ---------------------------------------------------------------------------
# Narrow-and-arbitrate (hard prune, soft pick)
# ---------------------------------------------------------------------------

def masked_arbitrate(
    candidate_sets: list[set[int] | None],
    form_counts: torch.Tensor,
    form_ids: torch.Tensor,
    bigram_keys: torch.Tensor,
    bigram_counts: torch.Tensor,
    bigram_ids: torch.Tensor,
    n_classes: int,
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard-prune to candidate set; soft-pick within it via form / bigram counts.

    candidate_sets[i] is None => unconstrained (use fallback, or soft over full label space
    if fallback is None). Empty sets are treated as unconstrained (conflict degradation).

    Within a non-empty set: Score A = TRAIN form-count row masked to allowed classes;
    Score B = TRAIN bigram-count row masked likewise; higher Laplace conf wins
    (best_count / (row_total + ALPHA)). If neither has support in the set, fall back to
    the TRAIN-global class marginal restricted to the allowed set.
    """
    n = len(candidate_sets)
    if fallback is not None:
        pred = fallback.clone()
    else:
        pred = torch.full((n,), -1, dtype=torch.long)

    constrained = torch.zeros(n, dtype=torch.bool)
    allow_mask = torch.zeros(n, n_classes, dtype=torch.bool)
    for i, cset in enumerate(candidate_sets):
        if cset is None or len(cset) == 0:
            continue
        constrained[i] = True
        for c in cset:
            if 0 <= c < n_classes:
                allow_mask[i, c] = True

    if not bool(constrained.any()):
        if fallback is None:
            # pure soft over full space
            return _soft_pick_all(
                form_counts, form_ids, bigram_keys, bigram_counts, bigram_ids, n_classes,
            )
        return pred

    # Score A: form counts
    rows_a = form_counts[form_ids].float()  # [N, C]
    tot_a = rows_a.sum(1)
    masked_a = rows_a.clone()
    masked_a[~allow_mask] = 0.0
    masked_a[~constrained] = 0.0
    best_a_val, best_a = masked_a.max(1)
    conf_a = best_a_val / (tot_a + v5.ALPHA)
    conf_a = torch.where(best_a_val > 0, conf_a, torch.full_like(conf_a, -1e9))

    # Score B: bigram counts
    rows_b = lookup_keyed_rows(bigram_keys, bigram_counts, bigram_ids, n_classes).float()
    tot_b = rows_b.sum(1)
    masked_b = rows_b.clone()
    masked_b[~allow_mask] = 0.0
    masked_b[~constrained] = 0.0
    best_b_val, best_b = masked_b.max(1)
    conf_b = best_b_val / (tot_b + v5.ALPHA)
    conf_b = torch.where(best_b_val > 0, conf_b, torch.full_like(conf_b, -1e9))

    use_a = conf_a >= conf_b
    chosen = torch.where(use_a, best_a, best_b)
    has_support = (best_a_val > 0) | (best_b_val > 0)

    # Global marginal restricted to allowed set
    gcounts = form_counts.sum(0).float()  # [C]
    g_exp = gcounts.unsqueeze(0).expand(n, -1).clone()
    g_exp[~allow_mask] = -1.0
    best_g = g_exp.argmax(1)

    final = torch.where(has_support, chosen, best_g)
    pred = torch.where(constrained, final, pred)

    # Safety: never emit out-of-set class on constrained positions
    # (argmax on empty allowed would be wrong; allow_mask guarantees nonempty for constrained)
    return pred


def _soft_pick_all(
    form_counts: torch.Tensor,
    form_ids: torch.Tensor,
    bigram_keys: torch.Tensor,
    bigram_counts: torch.Tensor,
    bigram_ids: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """Unconstrained soft pick: form vs bigram highest-conf; else global maj."""
    n = len(form_ids)
    rows_a = form_counts[form_ids].float()
    tot_a = rows_a.sum(1)
    best_a_val, best_a = rows_a.max(1)
    conf_a = best_a_val / (tot_a + v5.ALPHA)
    conf_a = torch.where(best_a_val > 0, conf_a, torch.full_like(conf_a, -1e9))

    rows_b = lookup_keyed_rows(bigram_keys, bigram_counts, bigram_ids, n_classes).float()
    tot_b = rows_b.sum(1)
    best_b_val, best_b = rows_b.max(1)
    conf_b = best_b_val / (tot_b + v5.ALPHA)
    conf_b = torch.where(best_b_val > 0, conf_b, torch.full_like(conf_b, -1e9))

    use_a = conf_a >= conf_b
    chosen = torch.where(use_a, best_a, best_b)
    has_support = (best_a_val > 0) | (best_b_val > 0)
    gmaj = int(form_counts.sum(0).argmax())
    return torch.where(has_support, chosen, torch.full((n,), gmaj, dtype=torch.long))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def accuracy(pred: torch.Tensor, gold: torch.Tensor) -> float:
    if len(gold) == 0:
        return 0.0
    return float((pred == gold).float().mean())


def null_floor(gold: torch.Tensor, null_id: int) -> float:
    if len(gold) == 0:
        return 0.0
    return float((gold == null_id).float().mean())


def per_class_accuracy(
    pred: torch.Tensor, gold: torch.Tensor, labels: list[str],
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for i, lab in enumerate(labels):
        mask = gold == i
        supp = int(mask.sum())
        if supp == 0:
            out[lab] = {"acc": float("nan"), "support": 0}
        else:
            out[lab] = {
                "acc": float((pred[mask] == gold[mask]).float().mean()),
                "support": supp,
            }
    return out


def verdict(pos_acc: float, case_acc: float) -> str:
    """Verbatim prereg rule — do not soften."""
    if pos_acc >= 0.95 and case_acc >= 0.90:
        return "FIRES"
    if pos_acc < 0.85:
        return "HALTED"
    return "IN-BETWEEN"


def memorizer_predict(
    table: v5.KeyTable, form_ids: torch.Tensor, global_maj: int,
) -> torch.Tensor:
    pred = table.lookup(form_ids)
    return torch.where(pred < 0, torch.full_like(pred, global_maj), pred)


def clear_conf_fns() -> None:
    v5.CONF_FNS.clear()
    v5.RULE_CONF.clear()


# ---------------------------------------------------------------------------
# Per-head fit / predict (POS still uses KeyTable path; case/gnn use narrow-arbitrate)
# ---------------------------------------------------------------------------

def fit_context_tier(
    train_windows: torch.Tensor,
    train_y: torch.Tensor,
    dev_windows: torch.Tensor,
    dev_y: torch.Tensor,
    counts: torch.Tensor,
    n_classes: int,
    base: int,
) -> tuple[v5.KeyTable, tuple[int, float], float]:
    """Tune minsupp/mindet of prev1×cur bigram on DEV (registers-off = counts + ctx)."""
    cls = torch.arange(n_classes)
    model = SimpleNamespace(counts=counts)
    tr_key = train_windows[:, -2] * base + train_windows[:, -1]
    best_acc = -1.0
    best_table: v5.KeyTable | None = None
    best_hp = (5, 0.5)

    for minsupp, mindet in CTX_GRID:
        table, _n = v5.best_per_key(tr_key, train_y, minsupp, mindet)
        clear_conf_fns()
        v5.register_conf(
            "ctx", table,
            keyfn=lambda ids, b=base: ids[:, -2] * b + ids[:, -1],
        )
        rules = [("ctx", lambda w: torch.full_like(w[:, -1], -1))]
        idxs = torch.arange(len(dev_y))
        _out, pred = v5.core_cover_sw(
            model, rules, dev_windows, dev_y, cls, idxs, return_pred=True,
        )
        gmaj = int(counts.sum(0).argmax())
        pred = torch.where(pred < 0, torch.full_like(pred, gmaj), pred)
        acc = accuracy(pred, dev_y)
        if acc > best_acc:
            best_acc = acc
            best_table = table
            best_hp = (minsupp, mindet)

    assert best_table is not None
    clear_conf_fns()
    return best_table, best_hp, best_acc


def predict_head(
    windows: torch.Tensor,
    y: torch.Tensor,
    counts: torch.Tensor,
    n_classes: int,
    rule_names: list[str],
    register_tables: dict[str, v5.KeyTable],
    ctx_table: v5.KeyTable,
    base: int,
    registers_on: bool,
    fill_maj: int,
    key_mode: dict[str, str] | None = None,
) -> torch.Tensor:
    """
    Window layout: [stream_idx, cur_low, prev2, prev1, cur_surf]
    key_mode: name -> 'low' | 'surf' | 'pos' | 'ctx'
      low  : key = cur lowercased form (col 1)
      surf : key = cur surface form (col -1)
      pos  : key = stream index (col 0)
      ctx  : key = prev1*base + cur (cols -2, -1)
    """
    key_mode = key_mode or {}
    clear_conf_fns()

    def keyfn_for(name: str):
        mode = key_mode.get(name, "surf")
        if mode == "low":
            return lambda ids: ids[:, 1]
        if mode == "pos":
            return lambda ids: ids[:, 0]
        if mode == "ctx":
            return lambda ids, b=base: ids[:, -2] * b + ids[:, -1]
        return lambda ids: ids[:, -1]

    names: list[str] = []
    if registers_on:
        for name in rule_names:
            if name == "ctx":
                continue
            tab = register_tables.get(name)
            if tab is None:
                continue
            v5.register_conf(name, tab, keyfn_for(name))
            names.append(name)
    # context always present (registers-off still has form counts + context)
    v5.register_conf("ctx", ctx_table, keyfn_for("ctx"))
    names.append("ctx")

    cls = torch.arange(n_classes)
    model = SimpleNamespace(counts=counts)
    rules = [(n, lambda w: torch.full_like(w[:, -1], -1)) for n in names]
    idxs = torch.arange(len(y))
    _out, pred = v5.core_cover_sw(
        model, rules, windows, y, cls, idxs, return_pred=True,
    )
    pred = torch.where(pred < 0, torch.full_like(pred, fill_maj), pred)
    clear_conf_fns()
    return pred


def predict_case_or_gnn_registers_on(
    stream: dict[str, Any],
    form_sources: list[list[set[str]]],
    lab2i: dict[str, int],
    form_counts: torch.Tensor,
    bigram_keys: torch.Tensor,
    bigram_counts: torch.Tensor,
    bigram_ids: torch.Tensor,
    n_classes: int,
    soft_fallback: torch.Tensor,
    include_gov: bool = True,
) -> torch.Tensor:
    """Build intersected candidate sets and run narrow-and-arbitrate over them."""
    gov_sources: list[list[set[str]]] = []
    if include_gov:
        prep_src = [[s] if s is not None else [] for s in stream["prep_gov"]]
        verb_src = [[s] if s is not None else [] for s in stream["verb_gov"]]
        gov_sources = [prep_src, verb_src]
    cand = merge_candidate_sources(form_sources, *gov_sources, lab2i=lab2i)
    return masked_arbitrate(
        cand,
        form_counts,
        stream["surf"],
        bigram_keys,
        bigram_counts,
        bigram_ids,
        n_classes,
        fallback=soft_fallback,
    )


# ---------------------------------------------------------------------------
# Main campaign
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("German R1 — per-token transduction + register hard-layer (SOFT=0)")
    print("=" * 72)
    print(HONEST_SCOPE)
    print()

    # ---- TRAIN + DEV only (test not touched yet) ----
    print("Loading TRAIN + DEV (gold tasks only; test not read yet)...")
    train = load_aligned_split("train")
    dev = load_aligned_split("dev")
    print(f"  train sentences={len(train)}  dev sentences={len(dev)}")

    # Registers (pre-built; not mined from test/dev)
    art_reg = load_register("article_paradigm")
    adj_reg = load_register("adjective_endings")
    pro_reg = load_register("pronoun_paradigm")
    prep_reg = load_register("preposition_case")
    verb_reg = load_register("verb_government")
    det_reg = load_register("det_pron_ambiguous")

    art_f2c = invert_article_paradigm(art_reg["table"])
    pro_f2c = invert_pronoun_paradigm(pro_reg["table"])
    end2c = invert_adjective_endings(adj_reg["table"])
    art_f2g = invert_article_paradigm_gnn(art_reg["table"])
    pro_f2g = invert_pronoun_paradigm_gnn(pro_reg["table"])
    end2g = invert_adjective_endings_gnn(adj_reg["table"])
    strict_prep = strict_prepositions(prep_reg["table"])
    two_way_prep = two_way_prepositions(prep_reg["table"])
    prep_forms = all_preposition_forms(prep_reg["table"])
    verb_table = verb_reg["table"]
    verb_stem_index = build_verb_stem_index(verb_table)
    # coordinating conjunctions (surface) — terminate government spans
    conj_forms = {"und", "oder", "aber", "denn", "sondern", "bzw", "beziehungsweise"}

    extra_forms = (
        list(art_f2c.keys())
        + list(pro_f2c.keys())
        + list(strict_prep.keys())
        + list(two_way_prep.keys())
        + list(prep_forms)
        + list(det_reg["table"].keys())
        + list(verb_table.keys())
    )
    vocab = build_vocab(train, extra_forms)
    labs = label_maps(train)
    pos2i, case2i, gnn2i = labs["pos2i"], labs["case2i"], labs["gnn2i"]
    n_pos, n_case, n_gnn = len(UPOS), len(CASE_LABELS), len(labs["i2gnn"])
    base = len(vocab) + 1

    form_maj_pos, form_pos_supp = train_form_majority_pos(train)

    print(f"  vocab_size={len(vocab)}  gnn_classes={n_gnn}")
    print(f"  adj-ending gate: train majority UPOS==ADJ with support>={ADJ_GATE_MINSUPP}")
    print(f"  government NP-span safety cap N={GOV_SPAN_SAFETY_CAP}")
    print(
        "  article forms (all, incl. ambiguous):",
        sorted(art_f2c.keys()),
    )
    print(
        "  pronoun forms (all, incl. ambiguous):",
        sorted(pro_f2c.keys()),
    )
    print(
        "  adj endings (all, incl. ambiguous):",
        sorted(end2c.keys()),
    )
    print(
        f"  strict preps: {len(strict_prep)}  two-way preps: {len(two_way_prep)}  "
        f"verb lemmas: {len(verb_table)}  verb stems indexed: {len(verb_stem_index)}"
    )
    print(
        f"  gnn article forms: {len(art_f2g)}  pronoun forms: {len(pro_f2g)}  "
        f"adj endings: {len(end2g)}"
    )

    tr_stream = build_stream(
        train, vocab, strict_prep, two_way_prep, prep_forms, conj_forms, verb_stem_index,
    )
    dv_stream = build_stream(
        dev, vocab, strict_prep, two_way_prep, prep_forms, conj_forms, verb_stem_index,
    )

    y_pos_tr = flatten_labels(train, "upos", pos2i)
    y_case_tr = flatten_labels(train, "case", case2i)
    y_gnn_tr = flatten_labels(train, "gnn", gnn2i)
    y_pos_dv = flatten_labels(dev, "upos", pos2i)
    y_case_dv = flatten_labels(dev, "case", case2i)
    y_gnn_dv = flatten_labels(dev, "gnn", gnn2i)

    # Count tables (form -> label), TRAIN only
    counts_pos = build_counts(tr_stream["surf"], y_pos_tr, len(vocab), n_pos)
    counts_case = build_counts(tr_stream["surf"], y_case_tr, len(vocab), n_case)
    counts_gnn = build_counts(tr_stream["surf"], y_gnn_tr, len(vocab), n_gnn)
    maj_pos = int(counts_pos.sum(0).argmax())
    maj_case = int(counts_case.sum(0).argmax())
    maj_gnn = int(counts_gnn.sum(0).argmax())

    # Raw bigram count matrices (for masked arbitration within candidate sets)
    tr_bigram = tr_stream["windows"][:, -2] * base + tr_stream["windows"][:, -1]
    bg_case_keys, bg_case_counts = build_keyed_counts(tr_bigram, y_case_tr, n_case)
    bg_gnn_keys, bg_gnn_counts = build_keyed_counts(tr_bigram, y_gnn_tr, n_gnn)

    # Control (i): majority-per-form memorizer via v5.best_per_key
    mem_pos_tbl, _ = v5.best_per_key(tr_stream["surf"], y_pos_tr, minsupp=1, mindet=0.0)
    mem_case_tbl, _ = v5.best_per_key(tr_stream["surf"], y_case_tr, minsupp=1, mindet=0.0)
    mem_gnn_tbl, _ = v5.best_per_key(tr_stream["surf"], y_gnn_tr, minsupp=1, mindet=0.0)

    # POS register KeyTable (det/pron ambiguous) — POS path unchanged
    det_tbl = keytable_det_pron(det_reg["table"], vocab, pos2i)

    # GNN register tables are non-empty inversions (measured marginal, not untried +0)
    gnn_reg_tables = {
        "article_gnn": art_f2g,
        "pronoun_gnn": pro_f2g,
        "adj_end_gnn": end2g,
    }
    gnn_reg_names = list(gnn_reg_tables.keys())

    # DEV-tune context tier per head (registers off)
    print("\nTuning context-tier (prev1×cur) thresholds on DEV (registers off)...")
    ctx_pos, hp_pos, acc_pos_dv = fit_context_tier(
        tr_stream["windows"], y_pos_tr, dv_stream["windows"], y_pos_dv,
        counts_pos, n_pos, base,
    )
    ctx_case, hp_case, acc_case_dv = fit_context_tier(
        tr_stream["windows"], y_case_tr, dv_stream["windows"], y_case_dv,
        counts_case, n_case, base,
    )
    ctx_gnn, hp_gnn, acc_gnn_dv = fit_context_tier(
        tr_stream["windows"], y_gnn_tr, dv_stream["windows"], y_gnn_dv,
        counts_gnn, n_gnn, base,
    )
    print(f"  pos  ctx minsupp={hp_pos[0]} mindet={hp_pos[1]}  dev_acc={acc_pos_dv:.4f}")
    print(f"  case ctx minsupp={hp_case[0]} mindet={hp_case[1]}  dev_acc={acc_case_dv:.4f}")
    print(f"  gnn  ctx minsupp={hp_gnn[0]} mindet={hp_gnn[1]}  dev_acc={acc_gnn_dv:.4f}")

    # ---- TEST read exactly once ----
    print("\n*** Reading TEST exactly once (final scoring pass only) ***")
    test = load_aligned_split("test")
    test_read_once = True
    te_stream = build_stream(
        test, vocab, strict_prep, two_way_prep, prep_forms, conj_forms, verb_stem_index,
    )
    y_pos_te = flatten_labels(test, "upos", pos2i)
    y_case_te = flatten_labels(test, "case", case2i)
    y_gnn_te = flatten_labels(test, "gnn", gnn2i)
    print(f"  test sentences={len(test)}  tokens={te_stream['n']}")

    te_bigram = te_stream["windows"][:, -2] * base + te_stream["windows"][:, -1]

    # POS head (unchanged KeyTable path)
    pos_reg_tables = {"det_pron": det_tbl}
    pos_key_mode = {"det_pron": "low", "ctx": "ctx"}
    pos_reg_names = ["det_pron"]

    pred_pos_on = predict_head(
        te_stream["windows"], y_pos_te, counts_pos, n_pos,
        pos_reg_names, pos_reg_tables, ctx_pos, base,
        registers_on=True, fill_maj=maj_pos, key_mode=pos_key_mode,
    )
    pred_pos_off = predict_head(
        te_stream["windows"], y_pos_te, counts_pos, n_pos,
        pos_reg_names, pos_reg_tables, ctx_pos, base,
        registers_on=False, fill_maj=maj_pos, key_mode=pos_key_mode,
    )
    pred_pos_mem = memorizer_predict(mem_pos_tbl, te_stream["surf"], maj_pos)

    # CASE head: registers-off = soft only; registers-on = narrow-and-arbitrate
    pred_case_off = predict_head(
        te_stream["windows"], y_case_te, counts_case, n_case,
        [], {}, ctx_case, base,
        registers_on=False, fill_maj=maj_case, key_mode={"ctx": "ctx"},
    )
    case_form_src = form_case_candidate_sets(
        te_stream["tokens"], art_f2c, pro_f2c, end2c, form_maj_pos, form_pos_supp,
    )
    pred_case_on = predict_case_or_gnn_registers_on(
        te_stream, case_form_src, case2i, counts_case,
        bg_case_keys, bg_case_counts, te_bigram, n_case,
        soft_fallback=pred_case_off, include_gov=True,
    )
    pred_case_mem = memorizer_predict(mem_case_tbl, te_stream["surf"], maj_case)

    # GNN head: paradigm candidate sets via same narrow-and-arbitrate (no gov)
    pred_gnn_off = predict_head(
        te_stream["windows"], y_gnn_te, counts_gnn, n_gnn,
        [], {}, ctx_gnn, base,
        registers_on=False, fill_maj=maj_gnn, key_mode={"ctx": "ctx"},
    )
    gnn_form_src = form_gnn_candidate_sets(
        te_stream["tokens"], art_f2g, pro_f2g, end2g, form_maj_pos, form_pos_supp,
    )
    pred_gnn_on = predict_case_or_gnn_registers_on(
        te_stream, gnn_form_src, gnn2i, counts_gnn,
        bg_gnn_keys, bg_gnn_counts, te_bigram, n_gnn,
        soft_fallback=pred_gnn_off, include_gov=False,
    )
    pred_gnn_mem = memorizer_predict(mem_gnn_tbl, te_stream["surf"], maj_gnn)

    # ---- Metrics ----
    acc_pos_on = accuracy(pred_pos_on, y_pos_te)
    acc_pos_off = accuracy(pred_pos_off, y_pos_te)
    acc_pos_mem = accuracy(pred_pos_mem, y_pos_te)
    acc_case_on = accuracy(pred_case_on, y_case_te)
    acc_case_off = accuracy(pred_case_off, y_case_te)
    acc_case_mem = accuracy(pred_case_mem, y_case_te)
    acc_gnn_on = accuracy(pred_gnn_on, y_gnn_te)
    acc_gnn_off = accuracy(pred_gnn_off, y_gnn_te)
    acc_gnn_mem = accuracy(pred_gnn_mem, y_gnn_te)

    marg_pos = acc_pos_on - acc_pos_off
    marg_case = acc_case_on - acc_case_off
    marg_gnn = acc_gnn_on - acc_gnn_off

    case_null = null_floor(y_case_te, case2i["-"])
    gnn_null = null_floor(y_gnn_te, gnn2i["-|-"])
    case_per = per_class_accuracy(pred_case_on, y_case_te, CASE_LABELS)

    verd = verdict(acc_pos_on, acc_case_on)
    dhash = data_hash()

    # ---- Scoreboard ----
    print()
    print("=" * 72)
    print("SCOREBOARD  (GSD test; registers-on is the gated system)")
    print("=" * 72)
    hdr = f"{'head':<12} {'full(on)':>10} {'regs_off':>10} {'memorizer':>10} {'reg_marg':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, on, off, mem, marg in [
        ("pos", acc_pos_on, acc_pos_off, acc_pos_mem, marg_pos),
        ("morph_case", acc_case_on, acc_case_off, acc_case_mem, marg_case),
        ("morph_gnn", acc_gnn_on, acc_gnn_off, acc_gnn_mem, marg_gnn),
    ]:
        print(f"{name:<12} {on:10.4f} {off:10.4f} {mem:10.4f} {marg:+10.4f}")

    print()
    print(f"morph_case majority-null floor (gold == '-'): {case_null:.4f}")
    print("morph_case per-class accuracy (registers ON):")
    for lab in CASE_LABELS:
        info = case_per[lab]
        print(f"  {lab:>4}: acc={info['acc']:.4f}  support={info['support']}")

    print()
    print(f"morph_gnn null floor (gold == '-|-'): {gnn_null:.4f}")
    print(f"morph_gnn full-system accuracy (reported, NOT gated): {acc_gnn_on:.4f}")

    print()
    print(f"VERDICT [empirical]: {verd}")
    print("  rule: FIRES iff pos>=0.95 AND morph_case>=0.90 (registers ON);")
    print("        HALTED iff pos<0.85; else IN-BETWEEN")
    print(f"  measured: pos={acc_pos_on:.4f}  morph_case={acc_case_on:.4f}  morph_gnn={acc_gnn_on:.4f}")

    print()
    print(f"data-hash (sha256 of 9 task jsonl files): {dhash}")
    print(
        f"test-read-once confirmation: test was read exactly once "
        f"(test_read_once={test_read_once}), only for this final scoring pass; "
        f"no test-set access during fitting or DEV threshold tuning."
    )
    print()
    print(
        f"ADJ ending gate: train-majority UPOS==ADJ (support>={ADJ_GATE_MINSUPP}); "
        "all endings (incl. ambiguous) emit candidate sets for narrow-and-arbitrate."
    )
    print(
        f"GNN register tier: non-empty "
        f"(article={len(art_f2g)}, pronoun={len(pro_f2g)}, adj_end={len(end2g)} forms/endings); "
        f"names={gnn_reg_names}"
    )
    print(HONEST_SCOPE)
    print("=" * 72)

    # Persist artifact
    artifact = {
        "rung": "R1",
        "verdict": verd,
        "verdict_tag": "empirical",
        "data_hash": dhash,
        "test_read_once": test_read_once,
        "honest_scope": HONEST_SCOPE,
        "gov_span_safety_cap": GOV_SPAN_SAFETY_CAP,
        "gov_window_n": GOV_SPAN_SAFETY_CAP,  # alias
        "adj_gate": {
            "rule": "train majority UPOS == ADJ",
            "minsupp": ADJ_GATE_MINSUPP,
            "note": "applies to all endings incl. ambiguous (candidate-set source)",
        },
        "enrichments": {
            "narrow_and_arbitrate": True,
            "np_span_government": True,
            "two_way_preps": True,
            "verb_stem_match": True,
            "gnn_register_tier": True,
        },
        "ctx_hyperparams": {
            "pos": {"minsupp": hp_pos[0], "mindet": hp_pos[1], "dev_acc": acc_pos_dv},
            "morph_case": {"minsupp": hp_case[0], "mindet": hp_case[1], "dev_acc": acc_case_dv},
            "morph_gnn": {"minsupp": hp_gnn[0], "mindet": hp_gnn[1], "dev_acc": acc_gnn_dv},
        },
        "scores": {
            "pos": {
                "full": acc_pos_on,
                "registers_off": acc_pos_off,
                "memorizer": acc_pos_mem,
                "register_marginal": marg_pos,
            },
            "morph_case": {
                "full": acc_case_on,
                "registers_off": acc_case_off,
                "memorizer": acc_case_mem,
                "register_marginal": marg_case,
                "null_floor": case_null,
                "per_class": case_per,
            },
            "morph_gnn": {
                "full": acc_gnn_on,
                "registers_off": acc_gnn_off,
                "memorizer": acc_gnn_mem,
                "register_marginal": marg_gnn,
                "null_floor": gnn_null,
            },
        },
        "n_test_sentences": len(test),
        "n_test_tokens": te_stream["n"],
        "gnn_reg_names": gnn_reg_names,
        "gnn_reg_sizes": {k: len(v) for k, v in gnn_reg_tables.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


# ---------------------------------------------------------------------------
# Export helpers for tests (pure functions; no side effects)
# ---------------------------------------------------------------------------

def parse_task_record(rec: dict, target_key: str) -> dict:
    """Parse one task JSONL record; align tokens/targets 1:1."""
    tokens = rec["tokens"]
    targets = rec["targets"][target_key]
    if len(tokens) != len(targets):
        raise ValueError("tokens/targets length mismatch")
    return {
        "sent_id": rec["sent_id"],
        "text": rec.get("text", ""),
        "tokens": tokens,
        "targets": targets,
    }


if __name__ == "__main__":
    sys.exit(main())
