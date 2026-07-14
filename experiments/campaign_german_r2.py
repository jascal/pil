"""German R2 — det_pron + aux_verb disambiguation (SOFT=0, inspectable rules).

Prereg (PREREG_GERMAN_EXPERT.md §R2): FIRES iff det_pron test accuracy >= 0.97
AND aux_verb test accuracy >= 0.97 AND the admitted rules are inspectable AND
recover the app's rule catalog ("followed-by-NOUN ⇒ DET", "paired-with-participle
⇒ AUX") as judged rules. Else report IN-BETWEEN honestly — never tune to force
FIRES.

Discipline: GSD train gold only (task train.jsonl + train-mined det_pron register
cross-check); tune on dev; test read exactly once for final scoring. SOFT=0 —
pure core_cover_sw arbitration over count tables + DEV-admitted context rules.
No wake/SGD. No teacher/ dumps. No GOLD neighbor tags as features.

Serve-honest features (surface only, available at serve time):
  next_cap, prev_form, next_form, participle_shape (heuristic),
  has_participle_in_sentence, next_is_participle, prev_is_participle, next_ends_en.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
import wyly_lm_v5 as v5  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (read-only inputs outside repo; write only data/german_r2.json)
# ---------------------------------------------------------------------------
GDATA = Path("/home/allans/code/germandata")
TASKS = GDATA / "tasks"
REGS = GDATA / "registers"
OUTPUT = REPO / "data" / "german_r2.json"

DET_PRON_LABELS = ["DET", "PRON"]
AUX_VERB_LABELS = ["AUX", "VERB"]

# Context-tier support/determinism candidates — tuned on DEV only (same grid as R1).
CTX_GRID = [
    (2, 0.5), (3, 0.5), (5, 0.5), (5, 0.6),
    (8, 0.5), (8, 0.6), (10, 0.5), (10, 0.7),
]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Participle surface heuristic: ge- + stem (>= PARTICIPLE_STEM_MIN) + (-t|-en).
# False-positive risk: finite/infinitive verbs that coincidentally match the shape
# (e.g. "gelten" with stem "lt"); this is NOT a morphological analyzer.
PARTICIPLE_STEM_MIN = 2

HONEST_SCOPE = (
    "Serve-honest surface features only (no gold neighbor UPOS/labels): "
    "next_cap (next token starts with uppercase — German noun capitalization proxy "
    "for 'followed-by-NOUN'); prev_form / next_form (neighbor surface-form ids); "
    "participle_shape heuristic (lowercase form starts with 'ge', ends with '-t' or "
    f"'-en', stem length between prefix and suffix >= {PARTICIPLE_STEM_MIN} — not a "
    "morphological analyzer; false-positive risk on short/coincidental 'ge-…-en' "
    "shapes such as 'gelten', and false-negative risk on ge-less participles like "
    "'verstanden'/'bekommen'); has_participle_in_sentence (any OTHER token matches "
    "participle_shape); next_is_participle / prev_is_participle; next_ends_en "
    "(bare-infinitive shape proxy for werden+infinitive). Gold labels of neighboring "
    "tokens are never used as features."
)

# Window columns (last col = target form id for model.counts fallback):
# 0: next_cap, 1: has_participle, 2: next_is_participle, 3: prev_is_participle,
# 4: next_ends_en, 5: prev_form, 6: next_form, 7: form_id
W_NEXT_CAP = 0
W_HAS_PART = 1
W_NEXT_PART = 2
W_PREV_PART = 3
W_NEXT_EN = 4
W_PREV_FORM = 5
W_NEXT_FORM = 6
W_FORM = 7  # last column — required by core_cover_sw counts tier


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


def parse_task_record(rec: dict) -> dict:
    """Parse one det_pron/aux_verb JSONL record; indices and labels aligned 1:1."""
    tokens = rec["tokens"]
    indices = list(rec["targets"]["indices"])
    labels = list(rec["targets"]["label"])
    if len(indices) != len(labels):
        raise ValueError("targets.indices / targets.label length mismatch")
    n = len(tokens)
    for i in indices:
        if not isinstance(i, int) or i < 0 or i >= n:
            raise ValueError(f"target index {i!r} out of range for tokens length {n}")
    return {
        "sent_id": rec.get("sent_id", ""),
        "text": rec.get("text", ""),
        "tokens": tokens,
        "indices": indices,
        "labels": labels,
    }


def load_task_split(task: str, split: str) -> list[dict]:
    path = TASKS / task / f"{split}.jsonl"
    return [parse_task_record(r) for r in load_jsonl(path)]


def data_hash() -> str:
    """sha256 over concatenation of the 6 task jsonl files (fixed order)."""
    h = hashlib.sha256()
    for task in ("det_pron", "aux_verb"):
        for split in ("train", "dev", "test"):
            path = TASKS / task / f"{split}.jsonl"
            h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Serve-honest surface features
# ---------------------------------------------------------------------------

def starts_upper(tok: str) -> bool:
    """True if token has a first alphabetic char that is uppercase."""
    for ch in tok:
        if ch.isalpha():
            return ch.isupper()
        # skip leading non-alpha (quotes, etc.)
        if ch.isalnum():
            return ch.isupper()
    return False


def participle_shape(tok: str, stem_min: int = PARTICIPLE_STEM_MIN) -> bool:
    """Surface heuristic for German past participles (NOT a morphological analyzer).

    Rule: lowercase form starts with 'ge', ends with '-t' or '-en', and the stem
    between 'ge' and the suffix has length >= stem_min (default 2).

    False-positive risk: some non-participles match the shape (e.g. 'gelten').
    False-negative risk: ge-less participles (verstanden, bekommen, …) miss.
    """
    low = tok.lower()
    if not low.startswith("ge") or len(low) < 2 + stem_min + 1:
        return False
    if low.endswith("en"):
        return len(low[2:-2]) >= stem_min
    if low.endswith("t"):
        return len(low[2:-1]) >= stem_min
    return False


def next_ends_en(tok: str) -> bool:
    """Bare-infinitive surface proxy: lowercase form ends with '-en' and len >= 4."""
    low = tok.lower()
    return len(low) >= 4 and low.endswith("en")


def feature_bundle(tokens: list[str], idx: int) -> dict[str, Any]:
    """Serve-honest features for the target token at `idx` (surface only)."""
    n = len(tokens)
    form = tokens[idx]
    nxt = tokens[idx + 1] if idx + 1 < n else None
    prv = tokens[idx - 1] if idx - 1 >= 0 else None
    has_part = any(
        participle_shape(tokens[j]) for j in range(n) if j != idx
    )
    return {
        "form": form,
        "next_cap": 1 if (nxt is not None and starts_upper(nxt)) else 0,
        "has_participle": 1 if has_part else 0,
        "next_is_participle": 1 if (nxt is not None and participle_shape(nxt)) else 0,
        "prev_is_participle": 1 if (prv is not None and participle_shape(prv)) else 0,
        "next_ends_en": 1 if (nxt is not None and next_ends_en(nxt)) else 0,
        "prev_form": prv if prv is not None else PAD_TOKEN,
        "next_form": nxt if nxt is not None else PAD_TOKEN,
    }


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


def build_vocab(train_splits: list[list[dict]]) -> Vocab:
    v = Vocab()
    v.add(PAD_TOKEN)
    v.add(UNK_TOKEN)
    for split in train_splits:
        for s in split:
            for t in s["tokens"]:
                v.add(t)
    return v


def label_map(labels: list[str]) -> tuple[dict[str, int], list[str]]:
    lab2i = {lab: i for i, lab in enumerate(labels)}
    return lab2i, labels


# ---------------------------------------------------------------------------
# Instance stream (one row per target index)
# ---------------------------------------------------------------------------

def flatten_instances(
    split: list[dict],
    vocab: Vocab,
    lab2i: dict[str, int],
) -> dict[str, Any]:
    """Flatten target instances to windows + labels.

    Window layout (last col = form id for core_cover_sw counts fallback):
      [next_cap, has_part, next_is_part, prev_is_part, next_ends_en,
       prev_form, next_form, form_id]
    """
    rows: list[list[int]] = []
    y: list[int] = []
    forms: list[str] = []
    feats: list[dict[str, Any]] = []
    pad = vocab.get(PAD_TOKEN)

    for s in split:
        toks = s["tokens"]
        for idx, lab in zip(s["indices"], s["labels"], strict=True):
            fb = feature_bundle(toks, idx)
            form_id = vocab.get(fb["form"])
            # PAD_TOKEN is in vocab; missing neighbors use PAD
            prev_id = pad if fb["prev_form"] == PAD_TOKEN else vocab.get(fb["prev_form"])
            next_id = pad if fb["next_form"] == PAD_TOKEN else vocab.get(fb["next_form"])
            row = [
                fb["next_cap"],
                fb["has_participle"],
                fb["next_is_participle"],
                fb["prev_is_participle"],
                fb["next_ends_en"],
                prev_id,
                next_id,
                form_id,
            ]
            rows.append(row)
            y.append(lab2i[lab])
            forms.append(fb["form"])
            feats.append(fb)

    if not rows:
        windows = torch.zeros(0, 8, dtype=torch.long)
        y_t = torch.zeros(0, dtype=torch.long)
        form_ids = torch.zeros(0, dtype=torch.long)
    else:
        windows = torch.tensor(rows, dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.long)
        form_ids = windows[:, W_FORM].clone()

    return {
        "windows": windows,
        "y": y_t,
        "form_ids": form_ids,
        "forms": forms,
        "feats": feats,
        "n": len(y),
    }


def build_counts(form_ids: torch.Tensor, labels: torch.Tensor, V: int, C: int) -> torch.Tensor:
    counts = torch.zeros(V, C, dtype=torch.long)
    if len(form_ids) == 0:
        return counts
    pair = form_ids * C + labels
    uniq, cnt = pair.unique(return_counts=True)
    counts.view(-1).index_add_(0, uniq, cnt)
    return counts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def accuracy(pred: torch.Tensor, gold: torch.Tensor) -> float:
    if len(gold) == 0:
        return 0.0
    return float((pred == gold).float().mean())


def score_indexed_tokens(
    pred_full: list[str | None],
    gold_full: list[str | None],
    indices: list[int],
) -> float:
    """Score only positions named in `indices` (non-indexed tokens ignored)."""
    if not indices:
        return 0.0
    correct = 0
    for i in indices:
        if pred_full[i] == gold_full[i]:
            correct += 1
    return correct / len(indices)


def verdict(
    det_pron_acc: float,
    aux_verb_acc: float,
    catalog_recovered: bool,
) -> str:
    """Verbatim prereg rule — do not soften.

    FIRES iff det_pron >= 0.97 AND aux_verb >= 0.97 AND catalog rules recovered
    (inspectable admitted rules recovering the app catalog). Else IN-BETWEEN.
    """
    if det_pron_acc >= 0.97 and aux_verb_acc >= 0.97 and catalog_recovered:
        return "FIRES"
    return "IN-BETWEEN"


def clear_conf_fns() -> None:
    v5.CONF_FNS.clear()
    v5.RULE_CONF.clear()


def memorizer_predict(
    table: v5.KeyTable, form_ids: torch.Tensor, global_maj: int,
) -> torch.Tensor:
    pred = table.lookup(form_ids)
    return torch.where(pred < 0, torch.full_like(pred, global_maj), pred)


# ---------------------------------------------------------------------------
# Context-rule candidates (named, inspectable, serve-honest)
# ---------------------------------------------------------------------------

def _keyfn_form_bin(feat_col: int, n_bins: int = 2) -> Callable:
    """key = form_id * n_bins + binary_feat."""

    def kf(ids: torch.Tensor, col: int = feat_col, nb: int = n_bins) -> torch.Tensor:
        return ids[:, W_FORM] * nb + ids[:, col]

    return kf


def _keyfn_form_neighbor(neigh_col: int, base: int) -> Callable:
    """key = form_id * base + neighbor_form_id."""

    def kf(ids: torch.Tensor, col: int = neigh_col, b: int = base) -> torch.Tensor:
        return ids[:, W_FORM] * b + ids[:, col]

    return kf


def candidate_specs(base: int, task: str) -> list[dict[str, Any]]:
    """Named candidate context rules per task (small inspectable library)."""
    shared = [
        {
            "name": "next_cap",
            "feature": "next_cap (next token starts with uppercase)",
            "keyfn": _keyfn_form_bin(W_NEXT_CAP, 2),
            "train_key": lambda w: w[:, W_FORM] * 2 + w[:, W_NEXT_CAP],
            "decode": "form×2+next_cap",
            "catalog": "followed-by-NOUN ⇒ DET" if task == "det_pron" else None,
        },
        {
            "name": "next_form",
            "feature": "next_form (next token surface form id)",
            "keyfn": _keyfn_form_neighbor(W_NEXT_FORM, base),
            "train_key": lambda w, b=base: w[:, W_FORM] * b + w[:, W_NEXT_FORM],
            "decode": "form×V+next_form",
            "catalog": None,
        },
        {
            "name": "prev_form",
            "feature": "prev_form (previous token surface form id)",
            "keyfn": _keyfn_form_neighbor(W_PREV_FORM, base),
            "train_key": lambda w, b=base: w[:, W_FORM] * b + w[:, W_PREV_FORM],
            "decode": "form×V+prev_form",
            "catalog": None,
        },
    ]
    if task == "aux_verb":
        shared.extend([
            {
                "name": "has_participle",
                "feature": "has_participle_in_sentence (other token matches participle_shape)",
                "keyfn": _keyfn_form_bin(W_HAS_PART, 2),
                "train_key": lambda w: w[:, W_FORM] * 2 + w[:, W_HAS_PART],
                "decode": "form×2+has_participle",
                "catalog": "paired-with-participle ⇒ AUX",
            },
            {
                "name": "next_is_participle",
                "feature": "next_is_participle (next token matches participle_shape)",
                "keyfn": _keyfn_form_bin(W_NEXT_PART, 2),
                "train_key": lambda w: w[:, W_FORM] * 2 + w[:, W_NEXT_PART],
                "decode": "form×2+next_is_participle",
                "catalog": "paired-with-participle ⇒ AUX",
            },
            {
                "name": "prev_is_participle",
                "feature": "prev_is_participle (prev token matches participle_shape)",
                "keyfn": _keyfn_form_bin(W_PREV_PART, 2),
                "train_key": lambda w: w[:, W_FORM] * 2 + w[:, W_PREV_PART],
                "decode": "form×2+prev_is_participle",
                "catalog": "paired-with-participle ⇒ AUX",
            },
            {
                "name": "next_ends_en",
                "feature": "next_ends_en (next token ends with -en; bare-inf proxy)",
                "keyfn": _keyfn_form_bin(W_NEXT_EN, 2),
                "train_key": lambda w: w[:, W_FORM] * 2 + w[:, W_NEXT_EN],
                "decode": "form×2+next_ends_en",
                "catalog": None,
            },
        ])
    return shared


def predict_with_rules(
    windows: torch.Tensor,
    y: torch.Tensor,
    counts: torch.Tensor,
    n_classes: int,
    admitted: list[dict[str, Any]],
    fill_maj: int,
) -> torch.Tensor:
    """Arbitrate admitted rules + model.counts via core_cover_sw; fill residual with maj."""
    clear_conf_fns()
    names: list[str] = []
    for rule in admitted:
        v5.register_conf(rule["name"], rule["table"], rule["keyfn"])
        names.append(rule["name"])
    cls = torch.arange(n_classes)
    model = SimpleNamespace(counts=counts)
    rules = [(n, lambda w: torch.full_like(w[:, -1], -1)) for n in names]
    idxs = torch.arange(len(y)) if len(y) > 0 else torch.zeros(0, dtype=torch.long)
    if len(y) == 0:
        clear_conf_fns()
        return torch.zeros(0, dtype=torch.long)
    _out, pred = v5.core_cover_sw(
        model, rules, windows, y, cls, idxs, return_pred=True,
    )
    pred = torch.where(pred < 0, torch.full_like(pred, fill_maj), pred)
    clear_conf_fns()
    return pred


def fit_candidate_table(
    train_windows: torch.Tensor,
    train_y: torch.Tensor,
    dev_windows: torch.Tensor,
    dev_y: torch.Tensor,
    counts: torch.Tensor,
    n_classes: int,
    fill_maj: int,
    cand: dict[str, Any],
    already: list[dict[str, Any]],
) -> tuple[v5.KeyTable, tuple[int, float], float]:
    """Grid-search minsupp/mindet on DEV for one candidate added to `already`."""
    tr_key = cand["train_key"](train_windows)
    best_acc = -1.0
    best_table: v5.KeyTable | None = None
    best_hp = (5, 0.5)

    for minsupp, mindet in CTX_GRID:
        table, _n = v5.best_per_key(tr_key, train_y, minsupp, mindet)
        trial = already + [{
            "name": cand["name"],
            "table": table,
            "keyfn": cand["keyfn"],
            "feature": cand["feature"],
            "decode": cand["decode"],
            "catalog": cand.get("catalog"),
            "minsupp": minsupp,
            "mindet": mindet,
        }]
        pred = predict_with_rules(
            dev_windows, dev_y, counts, n_classes, trial, fill_maj,
        )
        acc = accuracy(pred, dev_y)
        if acc > best_acc:
            best_acc = acc
            best_table = table
            best_hp = (minsupp, mindet)

    assert best_table is not None
    return best_table, best_hp, best_acc


def greedy_admit(
    train_windows: torch.Tensor,
    train_y: torch.Tensor,
    dev_windows: torch.Tensor,
    dev_y: torch.Tensor,
    counts: torch.Tensor,
    n_classes: int,
    fill_maj: int,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Greedy forward selection: admit candidate if it improves DEV accuracy."""
    admitted: list[dict[str, Any]] = []
    base_pred = predict_with_rules(
        dev_windows, dev_y, counts, n_classes, admitted, fill_maj,
    )
    best_acc = accuracy(base_pred, dev_y)
    remaining = list(candidates)

    while remaining:
        improved = False
        best_cand: dict[str, Any] | None = None
        best_table: v5.KeyTable | None = None
        best_hp: tuple[int, float] = (5, 0.5)
        best_new_acc = best_acc
        best_idx = -1

        for i, cand in enumerate(remaining):
            table, hp, acc = fit_candidate_table(
                train_windows, train_y, dev_windows, dev_y,
                counts, n_classes, fill_maj, cand, admitted,
            )
            if acc > best_new_acc + 1e-12:
                best_new_acc = acc
                best_cand = cand
                best_table = table
                best_hp = hp
                best_idx = i
                improved = True

        if not improved or best_cand is None or best_table is None:
            break

        admitted.append({
            "name": best_cand["name"],
            "table": best_table,
            "keyfn": best_cand["keyfn"],
            "feature": best_cand["feature"],
            "decode": best_cand["decode"],
            "catalog": best_cand.get("catalog"),
            "minsupp": best_hp[0],
            "mindet": best_hp[1],
            "dev_ensemble_acc": best_new_acc,
        })
        best_acc = best_new_acc
        remaining.pop(best_idx)

    return admitted, best_acc


def rule_mapping_human(
    table: v5.KeyTable,
    decode: str,
    vocab: Vocab,
    i2lab: list[str],
    base: int,
    max_entries: int = 40,
) -> list[str]:
    """Human-readable value -> label lines from a KeyTable (capped)."""
    lines: list[str] = []
    if len(table.k) == 0:
        return lines
    for i in range(len(table.k)):
        k = int(table.k[i])
        v = int(table.v[i])
        lab = i2lab[v] if 0 <= v < len(i2lab) else str(v)
        if decode.startswith("form×2+"):
            feat_name = decode.split("+", 1)[1]
            form_id, feat = divmod(k, 2)
            form = vocab.itos[form_id] if 0 <= form_id < len(vocab) else f"id{form_id}"
            lines.append(f"{form}+{feat_name}={feat} → {lab}")
        elif decode.startswith("form×V+"):
            neigh_name = decode.split("+", 1)[1]
            form_id = k // base
            neigh_id = k % base
            form = vocab.itos[form_id] if 0 <= form_id < len(vocab) else f"id{form_id}"
            neigh = vocab.itos[neigh_id] if 0 <= neigh_id < len(vocab) else f"id{neigh_id}"
            lines.append(f"{form}+{neigh_name}={neigh} → {lab}")
        else:
            lines.append(f"key={k} → {lab}")
        if len(lines) >= max_entries:
            lines.append(f"... ({len(table.k) - max_entries} more)")
            break
    return lines


def rule_dev_stats(
    rule: dict[str, Any],
    dev_windows: torch.Tensor,
    dev_y: torch.Tensor,
) -> tuple[int, float]:
    """DEV support (n fired) and fired-accuracy via v5.val_fired_acc."""
    keyfn = rule["keyfn"]
    table = rule["table"]

    def fn(ids: torch.Tensor) -> torch.Tensor:
        return table.lookup(keyfn(ids))

    idxs = torch.arange(len(dev_y))
    if len(dev_y) == 0:
        return 0, 0.0
    pred = fn(dev_windows)
    fired = pred >= 0
    n_supp = int(fired.sum())
    acc = v5.val_fired_acc(fn, dev_windows, dev_y, idxs)
    return n_supp, acc


def catalog_recovery_check(
    task: str,
    admitted: list[dict[str, Any]],
    i2lab: list[str],
) -> dict[str, Any]:
    """Check whether catalog rules were recovered as admitted judged rules."""
    if task == "det_pron":
        catalog_name = "followed-by-NOUN ⇒ DET"
        # Prefer next_cap-gated rule with majority DET when next_cap=1.
        for rule in admitted:
            if rule["name"] != "next_cap":
                continue
            table = rule["table"]
            det_id = i2lab.index("DET") if "DET" in i2lab else 0
            n_cap1 = 0
            n_det = 0
            for i in range(len(table.k)):
                k = int(table.k[i])
                _form_id, feat = divmod(k, 2)
                if feat == 1:
                    n_cap1 += 1
                    if int(table.v[i]) == det_id:
                        n_det += 1
            if n_cap1 > 0 and n_det >= (n_cap1 + 1) // 2:
                status = "RECOVERED"
                # IMPROVED if table has substantial form-conditioned nuance
                if len(table.k) > 2:
                    status = "IMPROVED"
                return {
                    catalog_name: {
                        "status": status,
                        "matched_rule": rule["name"],
                        "reason": (
                            f"admitted rule '{rule['name']}' gates on next_cap; "
                            f"when next_cap=1, {n_det}/{n_cap1} form keys map to DET"
                            + (" (form-conditioned; richer than bare catalog)"
                               if status == "IMPROVED" else "")
                        ),
                    },
                }
        return {
            catalog_name: {
                "status": "MISSED",
                "matched_rule": None,
                "reason": "no admitted next_cap-gated rule with majority DET on next_cap=1",
            },
        }

    # aux_verb
    catalog_name = "paired-with-participle ⇒ AUX"
    part_names = {"has_participle", "next_is_participle", "prev_is_participle"}
    for rule in admitted:
        if rule["name"] not in part_names:
            continue
        table = rule["table"]
        aux_id = i2lab.index("AUX") if "AUX" in i2lab else 0
        n_on = 0
        n_aux = 0
        for i in range(len(table.k)):
            k = int(table.k[i])
            _form_id, feat = divmod(k, 2)
            if feat == 1:
                n_on += 1
                if int(table.v[i]) == aux_id:
                    n_aux += 1
        if n_on > 0 and n_aux >= (n_on + 1) // 2:
            status = "RECOVERED"
            if len(table.k) > 2:
                status = "IMPROVED"
            return {
                catalog_name: {
                    "status": status,
                    "matched_rule": rule["name"],
                    "reason": (
                        f"admitted rule '{rule['name']}' gates on participle context; "
                        f"when feature=1, {n_aux}/{n_on} form keys map to AUX"
                        + (" (form-conditioned; richer than bare catalog)"
                           if status == "IMPROVED" else "")
                    ),
                },
            }
    return {
        catalog_name: {
            "status": "MISSED",
            "matched_rule": None,
            "reason": "no admitted participle-context rule with majority AUX when feature=1",
        },
    }


def crosscheck_det_register(train_forms: list[str], train_y: torch.Tensor, i2lab: list[str]) -> str:
    """Note consistency of train memorizer with registers/det_pron_ambiguous.json."""
    reg_path = REGS / "det_pron_ambiguous.json"
    if not reg_path.exists():
        return "det_pron register file missing; skipped consistency note"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    table = reg.get("table", reg)
    # Build train form majority
    counts: dict[str, Counter] = defaultdict(Counter)
    for form, yi in zip(train_forms, train_y.tolist(), strict=True):
        counts[form.lower()][i2lab[yi]] += 1
    agree = 0
    compared = 0
    for form, entry in table.items():
        if form not in counts:
            continue
        reg_counts = entry.get("counts", {})
        if not reg_counts:
            continue
        reg_maj = max(reg_counts.items(), key=lambda kv: kv[1])[0]
        tr_maj = counts[form].most_common(1)[0][0]
        compared += 1
        if reg_maj == tr_maj:
            agree += 1
    return (
        f"det_pron train-memorizer vs registers/det_pron_ambiguous.json: "
        f"{agree}/{compared} overlapping forms share the same majority label "
        f"(both mined from GSD train; minor form-key / lowercasing differences expected)"
    )


# ---------------------------------------------------------------------------
# Per-task fit + score
# ---------------------------------------------------------------------------

def run_task(
    task: str,
    labels: list[str],
    vocab: Vocab,
    train: list[dict],
    dev: list[dict],
    test: list[dict] | None,
) -> dict[str, Any]:
    lab2i, i2lab = label_map(labels)
    n_classes = len(labels)
    base = len(vocab) + 1

    tr = flatten_instances(train, vocab, lab2i)
    dv = flatten_instances(dev, vocab, lab2i)

    counts = build_counts(tr["form_ids"], tr["y"], len(vocab), n_classes)
    fill_maj = int(counts.sum(0).argmax()) if tr["n"] > 0 else 0

    # Control (i): majority-per-form memorizer
    mem_tbl, _ = v5.best_per_key(tr["form_ids"], tr["y"], minsupp=1, mindet=0.0)

    # Greedy DEV-admit context rules
    cands = candidate_specs(base, task)
    admitted, dev_acc = greedy_admit(
        tr["windows"], tr["y"], dv["windows"], dv["y"],
        counts, n_classes, fill_maj, cands,
    )

    # Enrich admitted with DEV fired stats + human mapping
    admitted_report: list[dict[str, Any]] = []
    for rule in admitted:
        n_supp, fired_acc = rule_dev_stats(rule, dv["windows"], dv["y"])
        mapping = rule_mapping_human(rule["table"], rule["decode"], vocab, i2lab, base)
        admitted_report.append({
            "name": rule["name"],
            "feature": rule["feature"],
            "mapping": mapping,
            "dev_support": n_supp,
            "dev_accuracy": fired_acc,
            "minsupp": rule["minsupp"],
            "mindet": rule["mindet"],
            "dev_ensemble_acc": rule.get("dev_ensemble_acc"),
        })
        rule["dev_support"] = n_supp
        rule["dev_accuracy"] = fired_acc
        rule["mapping"] = mapping

    catalog = catalog_recovery_check(task, admitted, i2lab)

    result: dict[str, Any] = {
        "task": task,
        "n_train": tr["n"],
        "n_dev": dv["n"],
        "fill_maj": fill_maj,
        "fill_maj_label": i2lab[fill_maj],
        "admitted": admitted,
        "admitted_report": admitted_report,
        "catalog_recovery": catalog,
        "dev_acc_rules_on": dev_acc,
        "mem_tbl": mem_tbl,
        "counts": counts,
        "n_classes": n_classes,
        "lab2i": lab2i,
        "i2lab": i2lab,
        "tr": tr,
        "dv": dv,
    }

    if task == "det_pron":
        result["register_note"] = crosscheck_det_register(tr["forms"], tr["y"], i2lab)

    if test is not None:
        te = flatten_instances(test, vocab, lab2i)
        pred_on = predict_with_rules(
            te["windows"], te["y"], counts, n_classes, admitted, fill_maj,
        )
        pred_off = predict_with_rules(
            te["windows"], te["y"], counts, n_classes, [], fill_maj,
        )
        pred_mem = memorizer_predict(mem_tbl, te["form_ids"], fill_maj)
        acc_on = accuracy(pred_on, te["y"])
        acc_off = accuracy(pred_off, te["y"])
        acc_mem = accuracy(pred_mem, te["y"])
        result.update({
            "te": te,
            "acc_full": acc_on,
            "acc_rules_off": acc_off,
            "acc_memorizer": acc_mem,
            "rule_marginal": acc_on - acc_off,
            "n_test": te["n"],
        })

    return result


def score_task_on_split(fit: dict[str, Any], vocab: Vocab, test: list[dict]) -> dict[str, Any]:
    """Score a fitted task on a held-out split (test read once; no refit)."""
    te = flatten_instances(test, vocab, fit["lab2i"])
    pred_on = predict_with_rules(
        te["windows"], te["y"], fit["counts"], fit["n_classes"],
        fit["admitted"], fit["fill_maj"],
    )
    pred_off = predict_with_rules(
        te["windows"], te["y"], fit["counts"], fit["n_classes"],
        [], fit["fill_maj"],
    )
    pred_mem = memorizer_predict(fit["mem_tbl"], te["form_ids"], fit["fill_maj"])
    acc_on = accuracy(pred_on, te["y"])
    acc_off = accuracy(pred_off, te["y"])
    acc_mem = accuracy(pred_mem, te["y"])
    out = dict(fit)
    out.update({
        "te": te,
        "acc_full": acc_on,
        "acc_rules_off": acc_off,
        "acc_memorizer": acc_mem,
        "rule_marginal": acc_on - acc_off,
        "n_test": te["n"],
    })
    return out


# ---------------------------------------------------------------------------
# Main campaign
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("German R2 — det_pron + aux_verb disambiguation (SOFT=0)")
    print("=" * 72)
    print(HONEST_SCOPE)
    print()
    print(
        "Serve-honest features used: next_cap, prev_form, next_form, "
        "participle_shape (heuristic), has_participle_in_sentence, "
        "next_is_participle, prev_is_participle, next_ends_en. "
        "Never gold neighbor tags."
    )
    print()

    # ---- TRAIN + DEV only (test not touched yet) ----
    print("Loading TRAIN + DEV (gold tasks only; test not read yet)...")
    det_train = load_task_split("det_pron", "train")
    det_dev = load_task_split("det_pron", "dev")
    aux_train = load_task_split("aux_verb", "train")
    aux_dev = load_task_split("aux_verb", "dev")
    print(
        f"  det_pron train={len(det_train)} dev={len(det_dev)}  "
        f"aux_verb train={len(aux_train)} dev={len(aux_dev)}"
    )

    vocab = build_vocab([det_train, aux_train])
    print(f"  shared vocab_size={len(vocab)} (PAD/UNK + both tasks' TRAIN tokens)")

    print("\nFitting det_pron (DEV-admitted context rules)...")
    det = run_task(
        "det_pron", DET_PRON_LABELS, vocab, det_train, det_dev, test=None,
    )
    print(f"  train instances={det['n_train']}  dev={det['n_dev']}")
    print(f"  global majority class (train targets): {det['fill_maj_label']}")
    print(f"  admitted rules: {[r['name'] for r in det['admitted']]}")
    print(f"  dev rules-on acc={det['dev_acc_rules_on']:.4f}")
    if "register_note" in det:
        print(f"  {det['register_note']}")

    print("\nFitting aux_verb (DEV-admitted context rules)...")
    aux = run_task(
        "aux_verb", AUX_VERB_LABELS, vocab, aux_train, aux_dev, test=None,
    )
    print(f"  train instances={aux['n_train']}  dev={aux['n_dev']}")
    print(f"  global majority class (train targets): {aux['fill_maj_label']}")
    print(f"  admitted rules: {[r['name'] for r in aux['admitted']]}")
    print(f"  dev rules-on acc={aux['dev_acc_rules_on']:.4f}")

    # ---- TEST read exactly once ----
    print("\n*** Reading TEST exactly once (final scoring pass only) ***")
    det_test = load_task_split("det_pron", "test")
    aux_test = load_task_split("aux_verb", "test")
    test_read_once = True
    print(f"  det_pron test sentences={len(det_test)}")
    print(f"  aux_verb test sentences={len(aux_test)}")

    # Score with already-fit tables (no refit; test used only here)
    det = score_task_on_split(det, vocab, det_test)
    aux = score_task_on_split(aux, vocab, aux_test)

    # Catalog recovery (both must be recovered or improved for verdict gate)
    det_cat = det["catalog_recovery"]
    aux_cat = aux["catalog_recovery"]
    catalog_all: dict[str, Any] = {}
    catalog_all.update(det_cat)
    catalog_all.update(aux_cat)

    def _ok(status: str) -> bool:
        return status in ("RECOVERED", "IMPROVED")

    catalog_recovered = all(
        _ok(v["status"]) for v in catalog_all.values()
    )

    verd = verdict(det["acc_full"], aux["acc_full"], catalog_recovered)
    dhash = data_hash()

    # ---- Scoreboard ----
    print()
    print("=" * 72)
    print("SCOREBOARD  (GSD test; rules-on is the gated system)")
    print("=" * 72)
    hdr = f"{'task':<12} {'full(on)':>10} {'memorizer':>10} {'rules_off':>10} {'rule_marg':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in [("det_pron", det), ("aux_verb", aux)]:
        print(
            f"{name:<12} {r['acc_full']:10.4f} {r['acc_memorizer']:10.4f} "
            f"{r['acc_rules_off']:10.4f} {r['rule_marginal']:+10.4f}"
        )

    for name, r in [("det_pron", det), ("aux_verb", aux)]:
        print()
        print(f"--- {name}: admitted rules (inspectable) ---")
        if not r["admitted_report"]:
            print("  (none admitted — memorizer/counts only)")
        for ar in r["admitted_report"]:
            print(
                f"  rule '{ar['name']}'  feature={ar['feature']}"
            )
            print(
                f"    minsupp={ar['minsupp']} mindet={ar['mindet']}  "
                f"dev_support={ar['dev_support']}  dev_fired_acc={ar['dev_accuracy']:.4f}"
            )
            print("    mapping (sample):")
            for line in ar["mapping"][:12]:
                print(f"      {line}")
            if len(ar["mapping"]) > 12:
                print(f"      ... ({len(ar['mapping']) - 12} more lines)")

    print()
    print("--- catalog-recovery check ---")
    for cname, info in catalog_all.items():
        print(f"  {cname}: {info['status']}")
        print(f"    matched_rule={info['matched_rule']}")
        print(f"    reason: {info['reason']}")

    print()
    print(f"VERDICT [empirical]: {verd}")
    print(
        "  rule: FIRES iff det_pron>=0.97 AND aux_verb>=0.97 AND catalog recovered "
        "(RECOVERED/IMPROVED for both catalog items); else IN-BETWEEN"
    )
    print(
        f"  measured: det_pron={det['acc_full']:.4f}  aux_verb={aux['acc_full']:.4f}  "
        f"catalog_recovered={catalog_recovered}"
    )

    print()
    print(f"data-hash (sha256 of 6 task jsonl files): {dhash}")
    print(
        f"test-read-once confirmation: test was read exactly once "
        f"(test_read_once={test_read_once}), only for this final scoring pass; "
        f"no test-set access during fitting or DEV threshold / admission tuning."
    )
    print()
    print(HONEST_SCOPE)
    print("=" * 72)

    # Persist artifact
    artifact = {
        "rung": "R2",
        "verdict": verd,
        "verdict_tag": "empirical",
        "data_hash": dhash,
        "test_read_once": test_read_once,
        "honest_scope": HONEST_SCOPE,
        "scores": {
            "det_pron": {
                "full": det["acc_full"],
                "memorizer": det["acc_memorizer"],
                "rules_off": det["acc_rules_off"],
                "register_marginal": det["rule_marginal"],
                "n_test": det["n_test"],
            },
            "aux_verb": {
                "full": aux["acc_full"],
                "memorizer": aux["acc_memorizer"],
                "rules_off": aux["acc_rules_off"],
                "register_marginal": aux["rule_marginal"],
                "n_test": aux["n_test"],
            },
        },
        "admitted_rules": {
            "det_pron": det["admitted_report"],
            "aux_verb": aux["admitted_report"],
        },
        "catalog_recovery": catalog_all,
        "catalog_recovered": catalog_recovered,
        "n_train_instances": {
            "det_pron": det["n_train"],
            "aux_verb": aux["n_train"],
        },
        "register_note": det.get("register_note", ""),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
