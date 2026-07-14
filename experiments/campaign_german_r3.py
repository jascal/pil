"""German R3 proper — labeled-dependency (head_deprel) predictor (SOFT=0).

Deliverable A: greedy per-token head_offset + deprel (serve-honest features only).
Deliverable B: feed student PREDICTED attachment + R1-PREDICTED POS into r3min's
full deprel→case cascade → serve-honest morph_case under predicted deprels.

Prereg (PREREG_GERMAN_EXPERT.md §R3). Gold train only; DEV-tune; TEST once.
Does NOT import or reference any *_codex* files (parallel RACED lane).
Does NOT import campaign_german_r2 (aux_verb settled); r3min may pull r2 transitively.
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
import campaign_german_r1 as r1  # noqa: E402
import campaign_german_r3min as r3min  # noqa: E402
import wyly_lm_v5 as v5  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GDATA = Path("/home/allans/code/germandata")
TASKS = GDATA / "tasks"
REGS = GDATA / "registers"
TEACHER_UD = GDATA / "teacher" / "ud_served.jsonl"
OUTPUT = REPO / "data" / "german_r3.json"
BASELINE_R1 = REPO / "data" / "german_r1.json"
BASELINE_R3MIN = REPO / "data" / "german_r3min.json"

# head_offset window grid (DEV-tuned by UAS)
K_GRID = (5, 8, 10, 15, 20)
# position buckets for relative-position feature
N_POS_BUCKETS = 5
# capitalization / punctuation binary features encoded into side keys
N_CAP = 2
N_PUNCT = 2
# ctx grid inherited from r1
CTX_GRID = r1.CTX_GRID
CONJ_FORMS = frozenset(r3min.CONJ_FORMS)

HONEST_SCOPE = (
    "R3 proper labeled-dependency predictor is GREEDY PER-TOKEN — no tree decoder, "
    "MST, projectivity, or global structure constraint in this first cut "
    "(a tree constraint could raise UAS later; not required to answer R3). "
    "Features are SERVE-HONEST only: surface form, R1-predicted POS (never gold UPOS), "
    "neighbor forms + neighbor R1-predicted POS, relative position bucket, capitalization, "
    "punctuation. NEVER gold head_offset/deprel/UPOS of any token. "
    "head_offset is framed as classification over a bounded relative window [-k,+k] "
    "(k DEV-tuned) including ROOT (offset 0). Gold governors outside that window are "
    "stated in-scope misses. Predictions whose governor index would fall outside the "
    "sentence are clamped to the nearest boundary token: "
    "g' = 0 if g < 0 else (n-1 if g >= n else g); offset' = g' - i. "
    "deprel is flat classification over TRAIN-observed labels (~41–46 UD labels). "
    "Deliverable B feeds STUDENT predicted head_offset+deprel and R1-predicted POS "
    "into r3min's full deprel→case cascade (oracle_case_gov_sentence_full) — this is "
    "the serve-honest morph_case number the #116 gold-attachment oracle could not give. "
    "verb_government residual precision (~0.64 from r3min per-rule diagnostics) remains "
    "a residual limiter on deliverable B's ceiling. SOFT=0; gold GSD train only; "
    "TEST read exactly once. [empirical]"
)

HONEST_SCOPE_OFFSET = (
    "Fallback rule for out-of-bounds predicted governors: if g = i + pred_offset "
    "is outside [0, n-1], clamp g to nearest boundary (0 or n-1) and set "
    "pred_offset = g_clamped - i. ROOT (offset 0) is never rewritten by clamp. "
    "No tree-validity repair after clamping. [empirical]"
)

R4_IMPLICATION = (
    "< 0.90 serve-honest → the expert scopes to pos+morph+registers+local-structure "
    "+ a one-small-LLM-call hybrid for attachment (no re-litigating — this number decides)."
)


# ---------------------------------------------------------------------------
# Data loading / hash
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_head_deprel_split(split: str) -> list[dict]:
    """Load head_deprel jsonl → {sent_id, text, tokens, head_offset, deprel}.

    Raises RuntimeError on tokens/targets length mismatch (mirrors r3min).
    """
    path = TASKS / "head_deprel" / f"{split}.jsonl"
    out: list[dict] = []
    for rec in load_jsonl(path):
        tokens = rec["tokens"]
        ho = list(rec["targets"]["head_offset"])
        deprel = list(rec["targets"]["deprel"])
        n = len(tokens)
        if len(ho) != n or len(deprel) != n:
            raise RuntimeError(
                f"head_deprel/{split}: length mismatch on {rec.get('sent_id')}"
            )
        out.append({
            "sent_id": rec["sent_id"],
            "text": rec.get("text", ""),
            "tokens": tokens,
            "head_offset": ho,
            "deprel": deprel,
        })
    return out


def data_hash() -> str:
    """sha256 over {pos, morph_case, head_deprel} × {train, dev, test} (fixed order)."""
    h = hashlib.sha256()
    for task in ("pos", "morph_case", "head_deprel"):
        for split in ("train", "dev", "test"):
            path = TASKS / task / f"{split}.jsonl"
            h.update(path.read_bytes())
    return h.hexdigest()


def head_index(i: int, head_offset: int) -> int:
    """Governor index = i + head_offset; root self-loop when offset == 0."""
    return i + head_offset


def clamp_offset(i: int, offset: int, n: int) -> int:
    """Clamp predicted governor to sentence bounds; return adjusted offset.

    Rule: g = i + offset; if g < 0 → g' = 0; if g >= n → g' = n - 1; else g' = g.
    offset' = g' - i. For n == 0 (empty), returns 0.
    """
    if n <= 0:
        return 0
    g = i + offset
    if g < 0:
        g = 0
    elif g >= n:
        g = n - 1
    return g - i


# ---------------------------------------------------------------------------
# UAS / LAS scorer (pure; unit-testable)
# ---------------------------------------------------------------------------

def score_uas_las(
    pred_head_offset: list[int] | torch.Tensor,
    pred_deprel: list[str],
    gold_head_offset: list[int] | torch.Tensor,
    gold_deprel: list[str],
) -> tuple[float, float]:
    """UAS = mean(pred_head_index == gold_head_index);
    LAS = mean(same AND pred_deprel == gold_deprel).
    """
    if isinstance(pred_head_offset, torch.Tensor):
        pred_ho = [int(x) for x in pred_head_offset.tolist()]
    else:
        pred_ho = list(pred_head_offset)
    if isinstance(gold_head_offset, torch.Tensor):
        gold_ho = [int(x) for x in gold_head_offset.tolist()]
    else:
        gold_ho = list(gold_head_offset)
    n = len(gold_ho)
    if n == 0:
        return 0.0, 0.0
    if not (len(pred_ho) == len(pred_deprel) == len(gold_deprel) == n):
        raise ValueError(
            f"length mismatch pred_ho={len(pred_ho)} pred_dep={len(pred_deprel)} "
            f"gold_ho={len(gold_ho)} gold_dep={len(gold_deprel)}"
        )
    uas_hits = 0
    las_hits = 0
    for i in range(n):
        ph = head_index(i, pred_ho[i])
        gh = head_index(i, gold_ho[i])
        if ph == gh:
            uas_hits += 1
            if pred_deprel[i] == gold_deprel[i]:
                las_hits += 1
    return uas_hits / n, las_hits / n


def fires_within_5_of_haiku(student_las: float, haiku_las: float) -> bool:
    """FIRES iff student LAS within 5 points of Haiku LAS (student >= haiku - 0.05)."""
    return student_las >= (haiku_las - 0.05)


# ---------------------------------------------------------------------------
# Haiku tree / forest flattening
# ---------------------------------------------------------------------------

def _parse_haiku_position(raw: Any) -> int | None:
    """Parse Haiku position to int; reject multiword ranges (e.g. '12-13')."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw == int(raw):
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        # multiword token spans like "12-13" are not single-token positions
        return None
    return None


def flatten_haiku_node(
    node: dict,
    parent_position: int | None = None,
) -> list[tuple[int, str, int]]:
    """Flatten one nested Haiku node → list of (position, dep, head_position) 1-indexed.

    Top-level / local-root nodes have head = their own position (self-loop).
    Children have head = parent's position. Recurses into children.
    Nodes with non-integer positions (multiword spans) are skipped as tokens but
    their children still attach to the nearest integer ancestor when possible.

    Note: Haiku sometimes emits duplicate position fields; for scoring we prefer
    `flatten_haiku_to_arrays` (parent-link identity) over position-keyed triples.
    """
    pos = _parse_haiku_position(node.get("position"))
    out: list[tuple[int, str, int]] = []
    child_parent = pos if pos is not None else parent_position
    if pos is not None:
        dep = str(node.get("dep", "dep"))
        head = pos if parent_position is None else int(parent_position)
        out.append((pos, dep, head))
    for child in node.get("children") or []:
        if isinstance(child, dict) and "position" in child:
            out.extend(flatten_haiku_node(child, parent_position=child_parent))
    return out


def _walk_haiku_roots(parsed: Any) -> list[dict]:
    """Yield top-level root node dicts from either Haiku shape (+ sentences wrapper)."""
    if parsed is None:
        return []
    if isinstance(parsed, dict) and "tree" in parsed and isinstance(parsed["tree"], dict):
        return [parsed["tree"]]
    if isinstance(parsed, list):
        return [n for n in parsed if isinstance(n, dict) and "position" in n]
    if isinstance(parsed, dict) and "sentences" in parsed:
        roots: list[dict] = []
        for sent in parsed["sentences"] or []:
            if not isinstance(sent, dict):
                continue
            if "tree" in sent and isinstance(sent["tree"], dict):
                roots.append(sent["tree"])
            elif "position" in sent:
                roots.append(sent)
        return roots
    return []


def flatten_haiku_parsed(parsed: Any) -> list[tuple[int, str, int]] | None:
    """Handle both validated Haiku shapes (+ rare sentences wrapper).

    1. dict with "tree" key — single nested tree (common).
    2. bare list of nodes — multi-root forest; each top-level is local root.
    3. dict with "sentences" (rare) — each sentence.tree treated as local root.
    Returns None if unhandled / empty.
    """
    roots = _walk_haiku_roots(parsed)
    if not roots:
        return None
    triples: list[tuple[int, str, int]] = []
    for root in roots:
        triples.extend(flatten_haiku_node(root, parent_position=None))
    return triples if triples else None


def triples_to_arrays(
    triples: list[tuple[int, str, int]],
) -> tuple[list[int], list[str]] | None:
    """Convert 1-indexed (position, dep, head) triples → 0-indexed head_offset + deprel.

    Spec: dict position → (dep, head); duplicate positions last-write-wins.
    Sorted by position into a 0-indexed array:
      - contiguous 1..n: head_offset[i] = (head_position_1idx - 1) - i
      - gapped (multiword spans dropped): rank remap head_offset[i] = rank(head) - i
        (equivalent to position-1 arithmetic when positions are exactly 1..n)
    Root self-loop → head_offset 0. Head pointing at a dropped span → self (0).
    """
    by_pos: dict[int, tuple[str, int]] = {}
    for pos, dep, head in triples:
        by_pos[pos] = (dep, head)  # last-write-wins on duplicate positions
    if not by_pos:
        return None
    positions = sorted(by_pos)
    pos_to_i = {p: i for i, p in enumerate(positions)}
    head_offset: list[int] = []
    deprel: list[str] = []
    contiguous = positions == list(range(1, len(positions) + 1))
    for i, pos in enumerate(positions):
        dep, head = by_pos[pos]
        if contiguous:
            # Spec formula: head_offset[i] = (head_position_1idx - 1) - i
            ho = (head - 1) - i
        elif head not in pos_to_i:
            ho = 0
        else:
            ho = pos_to_i[head] - i
        head_offset.append(ho)
        deprel.append(dep)
    return head_offset, deprel


def flatten_haiku_to_arrays(
    parsed: Any,
) -> tuple[list[int], list[str], list[str]] | None:
    """Tree-identity flatten → (head_offset, deprel, words) sorted by position.

    Each integer-position node is one token (parent link defines governor). This
    survives Haiku's occasional duplicate `position` fields (data-quality artifact):
    nodes remain distinct via tree identity; sort key is (position, encounter).
    Multiword span positions are skipped; children attach to nearest integer ancestor.
    Forest: each top-level root is a local root (head_offset 0).
    """
    roots = _walk_haiku_roots(parsed)
    if not roots:
        return None

    nodes: list[dict[str, Any]] = []

    def walk(node: dict, parent_idx: int | None) -> None:
        pos = _parse_haiku_position(node.get("position"))
        if pos is not None:
            idx = len(nodes)
            nodes.append({
                "pos": pos,
                "word": str(node.get("word", "")),
                "dep": str(node.get("dep", "dep")),
                "parent_idx": parent_idx,
            })
            my_idx: int | None = idx
        else:
            my_idx = parent_idx
        for child in node.get("children") or []:
            if isinstance(child, dict) and "position" in child:
                walk(child, my_idx)

    for root in roots:
        walk(root, None)
    if not nodes:
        return None

    order = sorted(range(len(nodes)), key=lambda i: (nodes[i]["pos"], i))
    old_to_new = {old: new for new, old in enumerate(order)}
    head_offset: list[int] = []
    deprel: list[str] = []
    words: list[str] = []
    for new_i, old_i in enumerate(order):
        node = nodes[old_i]
        words.append(node["word"])
        deprel.append(node["dep"])
        p = node["parent_idx"]
        if p is None:
            head_offset.append(0)
        else:
            head_offset.append(old_to_new[p] - new_i)
    return head_offset, deprel, words


def load_haiku_validated(path: Path = TEACHER_UD) -> list[dict]:
    """Load ud_served.jsonl; keep validated (parsed is not null) records with attachments.

    Each returned dict: {sentence, head_offset, deprel, n_tokens, raw_shape}.
    Flatten via position-keyed dict (last-write-wins on Haiku duplicate positions);
    convert to arrays sorted by position. Token-count equality is enforced later at
    GSD alignment (do not force-align different tokenizations).
    """
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            parsed = rec.get("parsed")
            if parsed is None:
                continue
            if isinstance(parsed, dict) and "tree" in parsed:
                shape = "tree"
            elif isinstance(parsed, list):
                shape = "list"
            elif isinstance(parsed, dict) and "sentences" in parsed:
                shape = "sentences"
            else:
                shape = "other"
            triples = flatten_haiku_parsed(parsed)
            if triples is None:
                continue
            arrays = triples_to_arrays(triples)
            if arrays is None:
                continue
            ho, dep = arrays
            out.append({
                "sentence": rec.get("sentence", ""),
                "head_offset": ho,
                "deprel": dep,
                "n_tokens": len(ho),
                "raw_shape": shape,
            })
    return out


def norm_ws(text: str) -> str:
    return " ".join(text.split())


def align_haiku_to_gsd_test(
    haiku_rows: list[dict],
    gsd_hd_test: list[dict],
) -> list[dict]:
    """Align Haiku records to GSD head_deprel test by normalized-whitespace text match
    + exact token-count equality. Returns overlap records with gold + haiku attachments.
    """
    by_text: dict[str, list[dict]] = defaultdict(list)
    for s in gsd_hd_test:
        by_text[norm_ws(s.get("text", ""))].append(s)

    overlap: list[dict] = []
    used_sids: set[str] = set()
    for h in haiku_rows:
        key = norm_ws(h["sentence"])
        cands = by_text.get(key, [])
        matched = None
        for g in cands:
            if g["sent_id"] in used_sids:
                continue
            if len(g["tokens"]) != h["n_tokens"]:
                continue
            matched = g
            break
        if matched is None:
            continue
        used_sids.add(matched["sent_id"])
        overlap.append({
            "sent_id": matched["sent_id"],
            "tokens": matched["tokens"],
            "gold_head_offset": matched["head_offset"],
            "gold_deprel": matched["deprel"],
            "haiku_head_offset": h["head_offset"],
            "haiku_deprel": h["deprel"],
        })
    return overlap


# ---------------------------------------------------------------------------
# Offset class encoding
# ---------------------------------------------------------------------------

def n_offset_classes(k: int) -> int:
    """Classes for offsets in [-k, +k] inclusive (includes ROOT=0)."""
    return 2 * k + 1


def offset_to_class(offset: int, k: int) -> int:
    """Map offset in [-k, k] → class id in [0, 2k]. Clamp gold OOR to ±k for training."""
    o = max(-k, min(k, offset))
    return o + k


def class_to_offset(cls: int, k: int) -> int:
    return int(cls) - k


# ---------------------------------------------------------------------------
# Serve-honest feature helpers
# ---------------------------------------------------------------------------

def pos_bucket(i: int, n: int, n_buckets: int = N_POS_BUCKETS) -> int:
    if n <= 1:
        return 0
    return min(n_buckets - 1, int(i * n_buckets / n))


def is_capitalized(tok: str) -> int:
    return 1 if tok and tok[0].isupper() else 0


def is_punct_feat(tok: str) -> int:
    return 1 if r1.is_punct_token(tok) else 0


def build_deprel_maps(train_hd: list[dict]) -> tuple[dict[str, int], list[str]]:
    labels: set[str] = set()
    for s in train_hd:
        labels.update(s["deprel"])
    i2lab = sorted(labels)
    lab2i = {lab: i for i, lab in enumerate(i2lab)}
    return lab2i, i2lab


# ---------------------------------------------------------------------------
# Flatten joined sentences → stream with serve-honest features
# ---------------------------------------------------------------------------

def flatten_hd_stream(
    joined: list[dict],
    vocab: r1.Vocab,
    pred_pos_by_sid: dict[str, list[str]],
    *,
    k: int,
    dep2i: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Flatten sentences to per-token tensors for head_offset / deprel heads.

    joined items must have: sent_id, tokens, head_offset, deprel (gold — train/dev only
    for labels; at test gold is for scoring only).
    pred_pos_by_sid[sent_id] = R1-predicted POS strings (same length as tokens).

    Window layout (matches r1 / predict_head):
      [stream_idx, cur_low, prev2_surf, prev1_surf, cur_surf]
    Side feature tensors (serve-honest):
      pred_pos_ids, pos_buckets, cap, punct, sent_len_bucket-less pos_bucket already,
      prev_pos_ids, next_pos_ids (pad=0 for OOB neighbors — UPOS[0] is ADJ; we use
      a dedicated PAD id = len(UPOS) for neighbor POS).
    """
    pad = vocab.get(r1.PAD_TOKEN)
    unk_pos_pad = len(r1.UPOS)  # neighbor POS pad (not a real UPOS)
    pos2i = {p: i for i, p in enumerate(r1.UPOS)}

    surf: list[int] = []
    low: list[int] = []
    tokens_flat: list[str] = []
    pred_pos_ids: list[int] = []
    prev_pos_ids: list[int] = []
    next_pos_ids: list[int] = []
    buckets: list[int] = []
    caps: list[int] = []
    puncts: list[int] = []
    y_off: list[int] = []
    y_dep: list[int] = []
    gold_off_raw: list[int] = []
    gold_dep_str: list[str] = []
    sent_ids_tok: list[str] = []
    sent_pos: list[int] = []  # index within sentence
    sent_n: list[int] = []

    for s in joined:
        sid = s["sent_id"]
        toks = s["tokens"]
        n = len(toks)
        pp = pred_pos_by_sid[sid]
        if len(pp) != n:
            raise RuntimeError(
                f"pred_pos length mismatch on {sid}: {len(pp)} vs tokens {n}"
            )
        for i, tok in enumerate(toks):
            surf.append(vocab.get(tok))
            low.append(vocab.get(tok.lower()))
            tokens_flat.append(tok)
            pid = pos2i.get(pp[i], pos2i.get("X", 0))
            pred_pos_ids.append(pid)
            prev_pos_ids.append(
                pos2i.get(pp[i - 1], unk_pos_pad) if i > 0 else unk_pos_pad
            )
            next_pos_ids.append(
                pos2i.get(pp[i + 1], unk_pos_pad) if i + 1 < n else unk_pos_pad
            )
            buckets.append(pos_bucket(i, n))
            caps.append(is_capitalized(tok))
            puncts.append(is_punct_feat(tok))
            raw_o = int(s["head_offset"][i])
            gold_off_raw.append(raw_o)
            y_off.append(offset_to_class(raw_o, k))
            dlab = s["deprel"][i]
            gold_dep_str.append(dlab)
            if dep2i is not None:
                y_dep.append(dep2i.get(dlab, -1))
            sent_ids_tok.append(sid)
            sent_pos.append(i)
            sent_n.append(n)

    n_tok = len(surf)
    windows = torch.zeros(n_tok, 5, dtype=torch.long)
    for i in range(n_tok):
        windows[i, 0] = i
        windows[i, 1] = low[i]
        windows[i, 2] = surf[i - 2] if i >= 2 else pad
        windows[i, 3] = surf[i - 1] if i >= 1 else pad
        windows[i, 4] = surf[i]

    out: dict[str, Any] = {
        "windows": windows,
        "surf": torch.tensor(surf, dtype=torch.long),
        "low": torch.tensor(low, dtype=torch.long),
        "tokens": tokens_flat,
        "pred_pos_ids": torch.tensor(pred_pos_ids, dtype=torch.long),
        "prev_pos_ids": torch.tensor(prev_pos_ids, dtype=torch.long),
        "next_pos_ids": torch.tensor(next_pos_ids, dtype=torch.long),
        "buckets": torch.tensor(buckets, dtype=torch.long),
        "caps": torch.tensor(caps, dtype=torch.long),
        "puncts": torch.tensor(puncts, dtype=torch.long),
        "y_offset_cls": torch.tensor(y_off, dtype=torch.long),
        "gold_offset_raw": gold_off_raw,
        "gold_deprel": gold_dep_str,
        "sent_ids": sent_ids_tok,
        "sent_pos": sent_pos,
        "sent_n": sent_n,
        "n": n_tok,
        "k": k,
    }
    if dep2i is not None:
        # map unknown to 0 temporarily; fit uses only train where all known
        y_dep_t = torch.tensor(
            [d if d >= 0 else 0 for d in y_dep], dtype=torch.long,
        )
        out["y_deprel"] = y_dep_t
        out["y_deprel_known"] = torch.tensor(
            [d >= 0 for d in y_dep], dtype=torch.bool,
        )
    return out


def composite_form_pos_key(
    form_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    base: int,
) -> torch.Tensor:
    return pos_ids * base + form_ids


def composite_pos_bucket_key(
    pos_ids: torch.Tensor,
    buckets: torch.Tensor,
    n_buckets: int = N_POS_BUCKETS,
) -> torch.Tensor:
    return pos_ids * n_buckets + buckets


def composite_rich_key(
    form_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    buckets: torch.Tensor,
    caps: torch.Tensor,
    puncts: torch.Tensor,
    base_form: int,
    n_pos: int = len(r1.UPOS),
    n_buckets: int = N_POS_BUCKETS,
) -> torch.Tensor:
    """Serve-honest composite: (((pos*nb+bucket)*n_cap+cap)*n_punct+punct)*base + form."""
    k = pos_ids * n_buckets + buckets
    k = k * N_CAP + caps
    k = k * N_PUNCT + puncts
    k = k * base_form + form_ids
    return k


# ---------------------------------------------------------------------------
# Fit / predict head_offset and deprel (SOFT=0 KeyTable arbitration)
# ---------------------------------------------------------------------------

def fit_context_tier_custom(
    train_keys: torch.Tensor,
    train_y: torch.Tensor,
    dev_keys: torch.Tensor,
    dev_y: torch.Tensor,
    counts: torch.Tensor,
    form_ids_dev: torch.Tensor,
    n_classes: int,
) -> tuple[v5.KeyTable, tuple[int, float], float]:
    """DEV-tune minsupp/mindet for an arbitrary integer-key context table.

    Counts tier uses form_ids_dev → counts[form] arbitration baseline, same as
    core_cover_sw's form path. Context key is train_keys/dev_keys.
    """
    cls = torch.arange(n_classes)
    model = SimpleNamespace(counts=counts)
    best_acc = -1.0
    best_table: v5.KeyTable | None = None
    best_hp = (5, 0.5)

    # Build a fake window so register keyfn can pull keys: we pass keys via col 0
    # and form via col -1 for counts.
    dev_windows = torch.stack([dev_keys, form_ids_dev], dim=1)

    for minsupp, mindet in CTX_GRID:
        table, _n = v5.best_per_key(train_keys, train_y, minsupp, mindet)
        r1.clear_conf_fns()
        v5.register_conf("ctx", table, keyfn=lambda ids: ids[:, 0])
        rules = [("ctx", lambda w: torch.full_like(w[:, -1], -1))]
        idxs = torch.arange(len(dev_y))
        _out, pred = v5.core_cover_sw(
            model, rules, dev_windows, dev_y, cls, idxs, return_pred=True,
        )
        gmaj = int(counts.sum(0).argmax())
        pred = torch.where(pred < 0, torch.full_like(pred, gmaj), pred)
        acc = r1.accuracy(pred, dev_y)
        if acc > best_acc:
            best_acc = acc
            best_table = table
            best_hp = (minsupp, mindet)

    assert best_table is not None
    r1.clear_conf_fns()
    return best_table, best_hp, best_acc


def predict_with_registers(
    form_ids: torch.Tensor,
    counts: torch.Tensor,
    n_classes: int,
    fill_maj: int,
    y_dummy: torch.Tensor,
    registers: list[tuple[str, v5.KeyTable, torch.Tensor]],
) -> torch.Tensor:
    """Arbitrate form counts + named KeyTable registers via core_cover_sw.

    registers: list of (name, table, keys_tensor aligned to tokens).
    """
    r1.clear_conf_fns()
    # windows: col0..col{R-1} = register keys; last col = form for counts
    parts = [reg_keys for _n, _t, reg_keys in registers] + [form_ids]
    windows = torch.stack(parts, dim=1)
    for i, (name, table, _keys) in enumerate(registers):
        v5.register_conf(name, table, keyfn=lambda ids, col=i: ids[:, col])
    names = [name for name, _t, _k in registers]
    cls = torch.arange(n_classes)
    model = SimpleNamespace(counts=counts)
    rules = [(n, lambda w: torch.full_like(w[:, -1], -1)) for n in names]
    idxs = torch.arange(len(y_dummy))
    _out, pred = v5.core_cover_sw(
        model, rules, windows, y_dummy, cls, idxs, return_pred=True,
    )
    pred = torch.where(pred < 0, torch.full_like(pred, fill_maj), pred)
    r1.clear_conf_fns()
    return pred


def fit_offset_head(
    tr_stream: dict[str, Any],
    dv_stream: dict[str, Any],
    k: int,
    vocab_size: int,
) -> dict[str, Any]:
    """Fit head_offset classifier for fixed k (counts + pos + form_pos + ctx)."""
    n_cls = n_offset_classes(k)
    base = vocab_size + 1
    y_tr = tr_stream["y_offset_cls"]
    y_dv = dv_stream["y_offset_cls"]

    # Expand counts matrix to cover form ids
    counts = r1.build_counts(tr_stream["surf"], y_tr, vocab_size, n_cls)
    fill_maj = int(counts.sum(0).argmax())

    # pos-keyed table
    pos_tbl, _ = v5.best_per_key(
        tr_stream["pred_pos_ids"], y_tr, minsupp=1, mindet=0.0,
    )
    # form×pos composite
    tr_fp = composite_form_pos_key(tr_stream["surf"], tr_stream["pred_pos_ids"], base)
    fp_tbl, _ = v5.best_per_key(tr_fp, y_tr, minsupp=2, mindet=0.5)

    # rich serve-honest key (pos, bucket, cap, punct, form)
    tr_rich = composite_rich_key(
        tr_stream["surf"], tr_stream["pred_pos_ids"], tr_stream["buckets"],
        tr_stream["caps"], tr_stream["puncts"], base,
    )
    rich_tbl, _ = v5.best_per_key(tr_rich, y_tr, minsupp=2, mindet=0.5)

    # (prev_surf, cur_surf) context — DEV-tuned
    tr_ctx = tr_stream["windows"][:, -2] * base + tr_stream["windows"][:, -1]
    dv_ctx = dv_stream["windows"][:, -2] * base + dv_stream["windows"][:, -1]
    ctx_tbl, hp_ctx, dev_acc_cls = fit_context_tier_custom(
        tr_ctx, y_tr, dv_ctx, y_dv, counts, dv_stream["surf"], n_cls,
    )

    # (prev_pos, cur_pos) context
    tr_pctx = tr_stream["prev_pos_ids"] * (len(r1.UPOS) + 2) + tr_stream["pred_pos_ids"]
    pctx_tbl, _ = v5.best_per_key(tr_pctx, y_tr, minsupp=2, mindet=0.5)

    bundle = {
        "k": k,
        "n_classes": n_cls,
        "counts": counts,
        "fill_maj": fill_maj,
        "pos_tbl": pos_tbl,
        "fp_tbl": fp_tbl,
        "rich_tbl": rich_tbl,
        "ctx_tbl": ctx_tbl,
        "pctx_tbl": pctx_tbl,
        "hp_ctx": hp_ctx,
        "dev_acc_cls": dev_acc_cls,
        "base": base,
    }
    return bundle


def predict_offset_head(
    stream: dict[str, Any],
    bundle: dict[str, Any],
) -> list[int]:
    """Predict head_offset ints (clamped to sentence bounds)."""
    k = bundle["k"]
    base = bundle["base"]
    y_dummy = stream["y_offset_cls"]
    form = stream["surf"]
    tr_fp = composite_form_pos_key(form, stream["pred_pos_ids"], base)
    rich = composite_rich_key(
        form, stream["pred_pos_ids"], stream["buckets"],
        stream["caps"], stream["puncts"], base,
    )
    ctx = stream["windows"][:, -2] * base + stream["windows"][:, -1]
    pctx = stream["prev_pos_ids"] * (len(r1.UPOS) + 2) + stream["pred_pos_ids"]

    registers = [
        ("rich", bundle["rich_tbl"], rich),
        ("fp", bundle["fp_tbl"], tr_fp),
        ("pctx", bundle["pctx_tbl"], pctx),
        ("ctx", bundle["ctx_tbl"], ctx),
        ("pos", bundle["pos_tbl"], stream["pred_pos_ids"]),
    ]
    pred_cls = predict_with_registers(
        form, bundle["counts"], bundle["n_classes"], bundle["fill_maj"],
        y_dummy, registers,
    )
    # Decode + per-token clamp using sentence lengths
    offsets: list[int] = []
    for i in range(stream["n"]):
        o = class_to_offset(int(pred_cls[i]), k)
        n_sent = int(stream["sent_n"][i])
        pos = int(stream["sent_pos"][i])
        offsets.append(clamp_offset(pos, o, n_sent))
    return offsets


def fit_deprel_head(
    tr_stream: dict[str, Any],
    dv_stream: dict[str, Any],
    n_dep: int,
    vocab_size: int,
    tr_pred_offset: list[int],
    dv_pred_offset: list[int],
    k: int,
) -> dict[str, Any]:
    """Fit deprel classifier; may use predicted head_offset as serve-honest feature."""
    base = vocab_size + 1
    y_tr = tr_stream["y_deprel"]
    y_dv = dv_stream["y_deprel"]

    counts = r1.build_counts(tr_stream["surf"], y_tr, vocab_size, n_dep)
    fill_maj = int(counts.sum(0).argmax())

    pos_tbl, _ = v5.best_per_key(
        tr_stream["pred_pos_ids"], y_tr, minsupp=1, mindet=0.0,
    )
    tr_fp = composite_form_pos_key(tr_stream["surf"], tr_stream["pred_pos_ids"], base)
    fp_tbl, _ = v5.best_per_key(tr_fp, y_tr, minsupp=2, mindet=0.5)

    tr_rich = composite_rich_key(
        tr_stream["surf"], tr_stream["pred_pos_ids"], tr_stream["buckets"],
        tr_stream["caps"], tr_stream["puncts"], base,
    )
    rich_tbl, _ = v5.best_per_key(tr_rich, y_tr, minsupp=2, mindet=0.5)

    tr_ctx = tr_stream["windows"][:, -2] * base + tr_stream["windows"][:, -1]
    dv_ctx = dv_stream["windows"][:, -2] * base + dv_stream["windows"][:, -1]
    ctx_tbl, hp_ctx, dev_acc_cls = fit_context_tier_custom(
        tr_ctx, y_tr, dv_ctx, y_dv, counts, dv_stream["surf"], n_dep,
    )

    # predicted offset direction/magnitude as feature (never gold)
    def off_feat(offsets: list[int], kk: int) -> torch.Tensor:
        # root=0, left magnitude buckets, right magnitude buckets → class in [0, 2*kk]
        return torch.tensor(
            [offset_to_class(o, kk) for o in offsets], dtype=torch.long,
        )

    tr_of = off_feat(tr_pred_offset, k)
    # key: pred_pos * n_off + offset_class
    n_off = n_offset_classes(k)
    tr_po = tr_stream["pred_pos_ids"] * n_off + tr_of
    po_tbl, _ = v5.best_per_key(tr_po, y_tr, minsupp=2, mindet=0.5)

    tr_pctx = tr_stream["prev_pos_ids"] * (len(r1.UPOS) + 2) + tr_stream["pred_pos_ids"]
    pctx_tbl, _ = v5.best_per_key(tr_pctx, y_tr, minsupp=2, mindet=0.5)

    return {
        "n_classes": n_dep,
        "counts": counts,
        "fill_maj": fill_maj,
        "pos_tbl": pos_tbl,
        "fp_tbl": fp_tbl,
        "rich_tbl": rich_tbl,
        "ctx_tbl": ctx_tbl,
        "po_tbl": po_tbl,
        "pctx_tbl": pctx_tbl,
        "hp_ctx": hp_ctx,
        "dev_acc_cls": dev_acc_cls,
        "base": base,
        "k": k,
        "n_off": n_off,
        "dv_pred_offset": dv_pred_offset,  # unused at predict; kept for debug
    }


def predict_deprel_head(
    stream: dict[str, Any],
    bundle: dict[str, Any],
    pred_offset: list[int],
    i2dep: list[str],
) -> list[str]:
    base = bundle["base"]
    k = bundle["k"]
    n_off = bundle["n_off"]
    form = stream["surf"]
    y_dummy = stream["y_deprel"] if "y_deprel" in stream else torch.zeros(
        stream["n"], dtype=torch.long,
    )
    tr_fp = composite_form_pos_key(form, stream["pred_pos_ids"], base)
    rich = composite_rich_key(
        form, stream["pred_pos_ids"], stream["buckets"],
        stream["caps"], stream["puncts"], base,
    )
    ctx = stream["windows"][:, -2] * base + stream["windows"][:, -1]
    of = torch.tensor(
        [offset_to_class(o, k) for o in pred_offset], dtype=torch.long,
    )
    po = stream["pred_pos_ids"] * n_off + of
    pctx = stream["prev_pos_ids"] * (len(r1.UPOS) + 2) + stream["pred_pos_ids"]

    registers = [
        ("rich", bundle["rich_tbl"], rich),
        ("fp", bundle["fp_tbl"], tr_fp),
        ("po", bundle["po_tbl"], po),
        ("pctx", bundle["pctx_tbl"], pctx),
        ("ctx", bundle["ctx_tbl"], ctx),
        ("pos", bundle["pos_tbl"], stream["pred_pos_ids"]),
    ]
    pred_cls = predict_with_registers(
        form, bundle["counts"], bundle["n_classes"], bundle["fill_maj"],
        y_dummy, registers,
    )
    return [i2dep[int(c)] if 0 <= int(c) < len(i2dep) else i2dep[bundle["fill_maj"]]
            for c in pred_cls.tolist()]


# ---------------------------------------------------------------------------
# Majority-offset + majority-deprel baseline (memorizer)
# ---------------------------------------------------------------------------

def fit_majority_baseline(
    tr_pred_pos_ids: torch.Tensor,
    tr_gold_offset: list[int],
    tr_gold_deprel: list[str],
    dep2i: dict[str, int],
    i2dep: list[str],
) -> dict[str, Any]:
    """TRAIN-fit majority head_offset / deprel per R1-predicted-POS.

    Laplace-shrunk via v5.best_per_key(minsupp=1, mindet=0.0). Fallback = global maj.
    Offset is stored as raw int encoded by shifting with OFFSET_SHIFT so all are >= 0
    for KeyTable values (torch long labels must be non-negative class ids for counts,
    but best_per_key accepts any long vals — negative offsets are fine as values).
    """
    # best_per_key uses vals as class labels; negative offsets are OK as table values.
    y_off = torch.tensor(tr_gold_offset, dtype=torch.long)
    y_dep = torch.tensor([dep2i[d] for d in tr_gold_deprel], dtype=torch.long)

    off_tbl, _ = v5.best_per_key(
        tr_pred_pos_ids, y_off, minsupp=1, mindet=0.0,
    )
    dep_tbl, _ = v5.best_per_key(
        tr_pred_pos_ids, y_dep, minsupp=1, mindet=0.0,
    )
    # global majorities
    g_off = int(Counter(tr_gold_offset).most_common(1)[0][0])
    g_dep_i = int(Counter(y_dep.tolist()).most_common(1)[0][0])
    return {
        "off_tbl": off_tbl,
        "dep_tbl": dep_tbl,
        "global_offset": g_off,
        "global_deprel_i": g_dep_i,
        "i2dep": i2dep,
    }


def predict_majority_baseline(
    pred_pos_ids: torch.Tensor,
    bundle: dict[str, Any],
    sent_pos: list[int],
    sent_n: list[int],
) -> tuple[list[int], list[str]]:
    """Predict majority offset/deprel; use lookup_conf so miss ≠ valid offset -1."""
    g_off = bundle["global_offset"]
    g_dep = bundle["global_deprel_i"]
    i2dep = bundle["i2dep"]

    off_val, off_conf = bundle["off_tbl"].lookup_conf(pred_pos_ids)
    dep_val, dep_conf = bundle["dep_tbl"].lookup_conf(pred_pos_ids)

    offsets: list[int] = []
    deprels: list[str] = []
    for i in range(len(pred_pos_ids)):
        if float(off_conf[i]) < -1e8:
            o = g_off
        else:
            o = int(off_val[i])
        o = clamp_offset(int(sent_pos[i]), o, int(sent_n[i]))
        offsets.append(o)

        if float(dep_conf[i]) < -1e8:
            d = g_dep
        else:
            d = int(dep_val[i])
        deprels.append(i2dep[d] if 0 <= d < len(i2dep) else i2dep[g_dep])
    return offsets, deprels


# ---------------------------------------------------------------------------
# R1-predicted POS (reused everywhere; compute once per split)
# ---------------------------------------------------------------------------

def fit_r1_pos(
    train: list[dict],
    dev: list[dict],
) -> dict[str, Any]:
    """Mirror r1.main() POS fitting (registers-ON path pieces only)."""
    art_reg = r1.load_register("article_paradigm")
    adj_reg = r1.load_register("adjective_endings")
    pro_reg = r1.load_register("pronoun_paradigm")
    prep_reg = r1.load_register("preposition_case")
    verb_reg = r1.load_register("verb_government")
    det_reg = r1.load_register("det_pron_ambiguous")

    art_f2c = r1.invert_article_paradigm(art_reg["table"])
    pro_f2c = r1.invert_pronoun_paradigm(pro_reg["table"])
    end2c = r1.invert_adjective_endings(adj_reg["table"])
    strict_prep = r1.strict_prepositions(prep_reg["table"])
    two_way_prep = r1.two_way_prepositions(prep_reg["table"])
    prep_forms = r1.all_preposition_forms(prep_reg["table"])
    verb_table = verb_reg["table"]
    verb_stem_index = r1.build_verb_stem_index(verb_table)
    conj_forms = set(CONJ_FORMS)

    extra_forms = (
        list(art_f2c.keys())
        + list(pro_f2c.keys())
        + list(strict_prep.keys())
        + list(two_way_prep.keys())
        + list(prep_forms)
        + list(det_reg["table"].keys())
        + list(verb_table.keys())
    )
    vocab = r1.build_vocab(train, extra_forms)
    labs = r1.label_maps(train)
    pos2i = labs["pos2i"]
    case2i = labs["case2i"]
    n_pos = len(r1.UPOS)
    n_case = len(r1.CASE_LABELS)
    base = len(vocab) + 1
    form_maj_pos, form_pos_supp = r1.train_form_majority_pos(train)

    tr_stream = r1.build_stream(
        train, vocab, strict_prep, two_way_prep, prep_forms, conj_forms, verb_stem_index,
    )
    dv_stream = r1.build_stream(
        dev, vocab, strict_prep, two_way_prep, prep_forms, conj_forms, verb_stem_index,
    )
    y_pos_tr = r1.flatten_labels(train, "upos", pos2i)
    y_pos_dv = r1.flatten_labels(dev, "upos", pos2i)
    y_case_tr = r1.flatten_labels(train, "case", case2i)
    y_case_dv = r1.flatten_labels(dev, "case", case2i)

    counts_pos = r1.build_counts(tr_stream["surf"], y_pos_tr, len(vocab), n_pos)
    maj_pos = int(counts_pos.sum(0).argmax())
    counts_case = r1.build_counts(tr_stream["surf"], y_case_tr, len(vocab), n_case)
    maj_case = int(counts_case.sum(0).argmax())

    tr_bigram = tr_stream["windows"][:, -2] * base + tr_stream["windows"][:, -1]
    bg_case_keys, bg_case_counts = r1.build_keyed_counts(tr_bigram, y_case_tr, n_case)

    det_tbl = r1.keytable_det_pron(det_reg["table"], vocab, pos2i)
    ctx_pos, hp_pos, acc_pos_dv = r1.fit_context_tier(
        tr_stream["windows"], y_pos_tr, dv_stream["windows"], y_pos_dv,
        counts_pos, n_pos, base,
    )
    ctx_case, hp_case, acc_case_dv = r1.fit_context_tier(
        tr_stream["windows"], y_case_tr, dv_stream["windows"], y_case_dv,
        counts_case, n_case, base,
    )

    return {
        "vocab": vocab,
        "labs": labs,
        "pos2i": pos2i,
        "case2i": case2i,
        "n_pos": n_pos,
        "n_case": n_case,
        "base": base,
        "form_maj_pos": form_maj_pos,
        "form_pos_supp": form_pos_supp,
        "strict_prep": strict_prep,
        "two_way_prep": two_way_prep,
        "prep_forms": prep_forms,
        "conj_forms": conj_forms,
        "verb_stem_index": verb_stem_index,
        "art_f2c": art_f2c,
        "pro_f2c": pro_f2c,
        "end2c": end2c,
        "counts_pos": counts_pos,
        "maj_pos": maj_pos,
        "counts_case": counts_case,
        "maj_case": maj_case,
        "bg_case_keys": bg_case_keys,
        "bg_case_counts": bg_case_counts,
        "det_tbl": det_tbl,
        "ctx_pos": ctx_pos,
        "hp_pos": hp_pos,
        "acc_pos_dv": acc_pos_dv,
        "ctx_case": ctx_case,
        "hp_case": hp_case,
        "acc_case_dv": acc_case_dv,
        "tr_stream": tr_stream,
        "dv_stream": dv_stream,
        "y_pos_tr": y_pos_tr,
        "y_pos_dv": y_pos_dv,
        "y_case_tr": y_case_tr,
        "y_case_dv": y_case_dv,
    }


def predict_r1_pos(
    split: list[dict],
    pack: dict[str, Any],
    y_pos: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, list[str]], dict[str, Any]]:
    """R1 registers-ON POS for a split. Returns (flat pred ids, by_sid strings, stream)."""
    stream = r1.build_stream(
        split, pack["vocab"], pack["strict_prep"], pack["two_way_prep"],
        pack["prep_forms"], pack["conj_forms"], pack["verb_stem_index"],
    )
    pred = r1.predict_head(
        stream["windows"], y_pos, pack["counts_pos"], pack["n_pos"],
        ["det_pron"], {"det_pron": pack["det_tbl"]}, pack["ctx_pos"], pack["base"],
        registers_on=True, fill_maj=pack["maj_pos"],
        key_mode={"det_pron": "low", "ctx": "ctx"},
    )
    # unflatten to by_sid
    by_sid: dict[str, list[str]] = {}
    idx = 0
    for s in split:
        n = len(s["tokens"])
        labs = [r1.UPOS[int(pred[idx + j])] for j in range(n)]
        by_sid[s["sent_id"]] = labs
        idx += n
    return pred, by_sid, stream


# ---------------------------------------------------------------------------
# Unflatten stream-level preds back to sentence list for cascade
# ---------------------------------------------------------------------------

def unflatten_to_sentences(
    split: list[dict],
    pred_offset: list[int],
    pred_deprel: list[str],
    pred_pos_by_sid: dict[str, list[str]],
) -> list[dict]:
    """Build r3min-cascade input: tokens, upos (R1-pred), head_offset, deprel (student)."""
    out: list[dict] = []
    idx = 0
    for s in split:
        n = len(s["tokens"])
        sid = s["sent_id"]
        out.append({
            "sent_id": sid,
            "tokens": s["tokens"],
            "upos": pred_pos_by_sid[sid],
            "head_offset": pred_offset[idx: idx + n],
            "deprel": pred_deprel[idx: idx + n],
        })
        idx += n
    if idx != len(pred_offset):
        raise RuntimeError(
            f"unflatten length mismatch: consumed {idx} vs pred {len(pred_offset)}"
        )
    return out


# ---------------------------------------------------------------------------
# Main campaign
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("German R3 proper — labeled-dependency (head_deprel) predictor [empirical]")
    print("=" * 72)
    print(HONEST_SCOPE)
    print(HONEST_SCOPE_OFFSET)
    print()
    print(
        "SCOPE LIMIT [empirical]: greedy per-token head_offset/deprel only — "
        "NO tree decoder / MST / global structure constraint."
    )

    # ---- TRAIN + DEV only ----
    print("\nLoading TRAIN + DEV (gold tasks only; test not read yet)...")
    train = r1.load_aligned_split("train")
    dev = r1.load_aligned_split("dev")
    hd_train = load_head_deprel_split("train")
    hd_dev = load_head_deprel_split("dev")
    train_j = r3min.join_head_deprel(train, hd_train, label="train")
    dev_j = r3min.join_head_deprel(dev, hd_dev, label="dev")
    print(
        f"  train sentences={len(train)}  dev sentences={len(dev)}  "
        f"train tokens≈{sum(len(s['tokens']) for s in train)}"
    )

    print("\nFitting R1 POS predictor (registers ON; DEV-tuned ctx)...")
    pack = fit_r1_pos(train, dev)
    print(
        f"  vocab_size={len(pack['vocab'])}  pos ctx minsupp={pack['hp_pos'][0]} "
        f"mindet={pack['hp_pos'][1]}  dev_pos_acc={pack['acc_pos_dv']:.4f}"
    )
    print(
        f"  case ctx minsupp={pack['hp_case'][0]} mindet={pack['hp_case'][1]}  "
        f"dev_case_acc={pack['acc_case_dv']:.4f}"
    )

    y_pos_tr = pack["y_pos_tr"]
    y_pos_dv = pack["y_pos_dv"]
    pred_pos_tr, pos_by_sid_tr, _ = predict_r1_pos(train, pack, y_pos_tr)
    pred_pos_dv, pos_by_sid_dv, _ = predict_r1_pos(dev, pack, y_pos_dv)
    acc_pos_tr = r1.accuracy(pred_pos_tr, y_pos_tr)
    acc_pos_dv = r1.accuracy(pred_pos_dv, y_pos_dv)
    print(
        f"  R1-pred POS train_acc={acc_pos_tr:.4f}  dev_acc={acc_pos_dv:.4f}  "
        f"(shipped test full ≈0.888 — sanity on test later) [empirical]"
    )

    dep2i, i2dep = build_deprel_maps(hd_train)
    n_dep = len(i2dep)
    print(f"  deprel vocabulary (TRAIN): n={n_dep}")

    # ---- DEV-tune k by UAS ----
    print(f"\nTuning head_offset window k on DEV (grid={list(K_GRID)}) by UAS...")
    best_k = K_GRID[0]
    best_dev_uas = -1.0
    best_off_bundle: dict[str, Any] | None = None
    best_dep_bundle: dict[str, Any] | None = None
    best_tr_stream: dict[str, Any] | None = None
    best_dv_stream: dict[str, Any] | None = None
    k_results: dict[int, dict[str, float]] = {}

    for k in K_GRID:
        tr_s = flatten_hd_stream(
            train_j, pack["vocab"], pos_by_sid_tr, k=k, dep2i=dep2i,
        )
        dv_s = flatten_hd_stream(
            dev_j, pack["vocab"], pos_by_sid_dv, k=k, dep2i=dep2i,
        )
        off_b = fit_offset_head(tr_s, dv_s, k, len(pack["vocab"]))
        pred_off_tr = predict_offset_head(tr_s, off_b)
        pred_off_dv = predict_offset_head(dv_s, off_b)
        dep_b = fit_deprel_head(
            tr_s, dv_s, n_dep, len(pack["vocab"]),
            pred_off_tr, pred_off_dv, k,
        )
        pred_dep_dv = predict_deprel_head(dv_s, dep_b, pred_off_dv, i2dep)
        uas, las = score_uas_las(
            pred_off_dv, pred_dep_dv,
            dv_s["gold_offset_raw"], dv_s["gold_deprel"],
        )
        k_results[k] = {"dev_uas": uas, "dev_las": las}
        print(f"  k={k:>2}: DEV UAS={uas:.4f}  LAS={las:.4f} [empirical]")
        if uas > best_dev_uas:
            best_dev_uas = uas
            best_k = k
            best_off_bundle = off_b
            best_dep_bundle = dep_b
            best_tr_stream = tr_s
            best_dv_stream = dv_s

    assert best_off_bundle is not None and best_dep_bundle is not None
    assert best_tr_stream is not None and best_dv_stream is not None
    print(f"  selected k={best_k} (DEV UAS={best_dev_uas:.4f}) [empirical]")

    # Majority baseline (TRAIN-fit, DEV-untuned)
    maj_bundle = fit_majority_baseline(
        best_tr_stream["pred_pos_ids"],
        best_tr_stream["gold_offset_raw"],
        best_tr_stream["gold_deprel"],
        dep2i,
        i2dep,
    )
    maj_off_dv, maj_dep_dv = predict_majority_baseline(
        best_dv_stream["pred_pos_ids"], maj_bundle,
        best_dv_stream["sent_pos"], best_dv_stream["sent_n"],
    )
    maj_uas_dv, maj_las_dv = score_uas_las(
        maj_off_dv, maj_dep_dv,
        best_dv_stream["gold_offset_raw"], best_dv_stream["gold_deprel"],
    )
    print(
        f"  majority baseline DEV UAS={maj_uas_dv:.4f} LAS={maj_las_dv:.4f} [empirical]"
    )

    # Load oracle / R1 baselines from disk (for deliverable B comparison)
    r1_art = json.loads(BASELINE_R1.read_text(encoding="utf-8"))
    r3min_art = json.loads(BASELINE_R3MIN.read_text(encoding="utf-8"))
    case_r1_baseline = float(r1_art["scores"]["morph_case"]["registers_off"])
    case_oracle_ceiling = float(r3min_art["scores"]["morph_case"]["arm_a_oracle_full"])
    print(
        f"\nBaselines (disk): R1 morph_case.registers_off={case_r1_baseline:.4f}  "
        f"r3min arm_a_oracle_full={case_oracle_ceiling:.4f}"
    )

    # ---- TEST read exactly once ----
    print("\n*** Reading TEST exactly once (final scoring pass only) ***")
    test = r1.load_aligned_split("test")
    hd_test = load_head_deprel_split("test")
    test_j = r3min.join_head_deprel(test, hd_test, label="test")
    test_read_once = True

    y_pos_te = r1.flatten_labels(test, "upos", pack["pos2i"])
    y_case_te = r1.flatten_labels(test, "case", pack["case2i"])
    pred_pos_te, pos_by_sid_te, te_stream_r1 = predict_r1_pos(test, pack, y_pos_te)
    acc_pos_te = r1.accuracy(pred_pos_te, y_pos_te)
    print(
        f"  test sentences={len(test)}  tokens={te_stream_r1['n']}  "
        f"R1-pred POS test_acc={acc_pos_te:.4f} "
        f"(shipped ≈0.888; expect within ~0.01) [empirical]"
    )

    te_s = flatten_hd_stream(
        test_j, pack["vocab"], pos_by_sid_te, k=best_k, dep2i=dep2i,
    )
    # Deliverable A — student
    pred_off_te = predict_offset_head(te_s, best_off_bundle)
    pred_dep_te = predict_deprel_head(te_s, best_dep_bundle, pred_off_te, i2dep)
    student_uas, student_las = score_uas_las(
        pred_off_te, pred_dep_te,
        te_s["gold_offset_raw"], te_s["gold_deprel"],
    )

    # Majority baseline on test
    maj_off_te, maj_dep_te = predict_majority_baseline(
        te_s["pred_pos_ids"], maj_bundle, te_s["sent_pos"], te_s["sent_n"],
    )
    maj_uas, maj_las = score_uas_las(
        maj_off_te, maj_dep_te,
        te_s["gold_offset_raw"], te_s["gold_deprel"],
    )

    # Haiku UAS/LAS on overlap
    print("\nExtracting Haiku attachments from teacher/ud_served.jsonl...")
    haiku_rows = load_haiku_validated(TEACHER_UD)
    print(
        f"  validated Haiku records with flattenable attachments: {len(haiku_rows)} "
        f"[empirical]"
    )
    overlap = align_haiku_to_gsd_test(haiku_rows, hd_test)
    n_overlap = len(overlap)
    if n_overlap == 0:
        haiku_uas = haiku_las = float("nan")
    else:
        h_off: list[int] = []
        h_dep: list[str] = []
        g_off: list[int] = []
        g_dep: list[str] = []
        for o in overlap:
            h_off.extend(o["haiku_head_offset"])
            h_dep.extend(o["haiku_deprel"])
            g_off.extend(o["gold_head_offset"])
            g_dep.extend(o["gold_deprel"])
        haiku_uas, haiku_las = score_uas_las(h_off, h_dep, g_off, g_dep)
    print(
        f"  Haiku↔GSD-test overlap n={n_overlap}  "
        f"UAS={haiku_uas:.4f}  LAS={haiku_las:.4f} [empirical]"
    )

    within5 = (
        fires_within_5_of_haiku(student_las, haiku_las)
        if n_overlap > 0 and haiku_las == haiku_las  # not NaN
        else False
    )
    fires_str = "FIRES" if within5 else "miss"

    # ---- Deliverable B: serve-honest case under PREDICTED deprels ----
    print("\n--- Deliverable B: serve-honest case under PREDICTED attachment ---")
    pred_joined = unflatten_to_sentences(
        test, pred_off_te, pred_dep_te, pos_by_sid_te,
    )
    prep_gov_pred, verb_gov_pred, _diag = r3min.flatten_oracle_case_gov_full(
        pred_joined,
        pack["strict_prep"],
        pack["two_way_prep"],
        pack["verb_stem_index"],
    )
    # case soft fallback (registers off) + form candidate sets — mirror r3min
    te_bigram = (
        te_stream_r1["windows"][:, -2] * pack["base"] + te_stream_r1["windows"][:, -1]
    )
    pred_case_off = r1.predict_head(
        te_stream_r1["windows"], y_case_te, pack["counts_case"], pack["n_case"],
        [], {}, pack["ctx_case"], pack["base"],
        registers_on=False, fill_maj=pack["maj_case"], key_mode={"ctx": "ctx"},
    )
    case_form_src = r1.form_case_candidate_sets(
        te_stream_r1["tokens"],
        pack["art_f2c"], pack["pro_f2c"], pack["end2c"],
        pack["form_maj_pos"], pack["form_pos_supp"],
    )
    stream_b = r3min.stream_with_gov(te_stream_r1, prep_gov_pred, verb_gov_pred)
    pred_case_b = r1.predict_case_or_gnn_registers_on(
        stream_b, case_form_src, pack["case2i"], pack["counts_case"],
        pack["bg_case_keys"], pack["bg_case_counts"], te_bigram, pack["n_case"],
        soft_fallback=pred_case_off, include_gov=True,
    )
    acc_case_serve_honest = r1.accuracy(pred_case_b, y_case_te)
    acc_case_off = r1.accuracy(pred_case_off, y_case_te)

    print(
        f"  serve-honest morph_case (predicted attachment) = "
        f"{acc_case_serve_honest:.4f} [empirical]"
    )
    print(
        f"  oracle ceiling (r3min arm_a_oracle_full) = {case_oracle_ceiling:.4f}  "
        f"R1 registers_off baseline = {case_r1_baseline:.4f}  "
        f"recomputed regs_off = {acc_case_off:.4f} [empirical]"
    )
    print(
        "  note: verb_government residual precision ~0.64 (r3min per-rule diagnostics) "
        "is a residual limiter on this ceiling — not re-derived here. [empirical]"
    )
    below_90 = acc_case_serve_honest < 0.90
    if below_90:
        print(f"  R4 IMPLICATION: {R4_IMPLICATION}")
    else:
        print(
            f"  R4 IMPLICATION (gated; serve-honest >= 0.90 so hybrid may not be "
            f"required by the prereg gate): measured={acc_case_serve_honest:.4f} "
            f"[empirical]"
        )
        # Still print the verbatim line for the artifact contract when < 0.90 only;
        # when >= 0.90 state the decision gate result clearly.

    # ---- Scoreboard ----
    print()
    print("=" * 72)
    print("SCOREBOARD  (GSD test; deliverable A)  [empirical]")
    print("=" * 72)
    hdr = (
        f"{'system':<28} {'UAS':>10} {'LAS':>10} {'notes':>24}"
    )
    print(hdr)
    print("-" * len(hdr))
    print(
        f"{'student (greedy token)':<28} {student_uas:10.4f} {student_las:10.4f} "
        f"{'k=' + str(best_k):>24}"
    )
    print(
        f"{'majority-offset baseline':<28} {maj_uas:10.4f} {maj_las:10.4f} "
        f"{'per R1-pred POS':>24}"
    )
    print(
        f"{'Haiku (overlap only)':<28} {haiku_uas:10.4f} {haiku_las:10.4f} "
        f"{'n=' + str(n_overlap):>24}"
    )
    print()
    print(
        f"student-LAS-within-5-of-Haiku-LAS?  {fires_str}  "
        f"(student_LAS={student_las:.4f}  haiku_LAS={haiku_las:.4f}  "
        f"threshold={haiku_las - 0.05:.4f}) [empirical]"
    )
    print()
    print("DELIVERABLE B  [empirical]")
    print(
        f"  morph_case serve-honest (pred deprel) = {acc_case_serve_honest:.4f}"
    )
    print(f"  oracle ceiling (gold attachment)     = {case_oracle_ceiling:.4f}")
    print(f"  R1 parse-free baseline               = {case_r1_baseline:.4f}")
    if below_90:
        print(f"  R4: {R4_IMPLICATION}")
    print()

    dhash = data_hash()
    print(f"data-hash (sha256 of pos/morph_case/head_deprel × train/dev/test): {dhash}")
    print(
        f"test-read-once confirmation: test was read exactly once "
        f"(test_read_once={test_read_once}), only for this final scoring pass; "
        f"no test-set access during fitting or DEV threshold tuning. "
        f"Deliverable A scoring and deliverable B cascade both used that single TEST read."
    )
    print(HONEST_SCOPE)
    print("=" * 72)

    artifact = {
        "rung": "R3",
        "verdict_tag": "empirical",
        "fires_within_5_of_haiku": within5,
        "fires_str": fires_str,
        "data_hash": dhash,
        "test_read_once": test_read_once,
        "honest_scope": HONEST_SCOPE,
        "honest_scope_offset": HONEST_SCOPE_OFFSET,
        "scope_limit": (
            "greedy per-token head_offset/deprel; no tree decoder / MST / "
            "global structure constraint"
        ),
        "hyperparams": {
            "k": best_k,
            "k_grid": list(K_GRID),
            "k_dev_results": k_results,
            "offset_ctx": {
                "minsupp": best_off_bundle["hp_ctx"][0],
                "mindet": best_off_bundle["hp_ctx"][1],
                "dev_acc_cls": best_off_bundle["dev_acc_cls"],
            },
            "deprel_ctx": {
                "minsupp": best_dep_bundle["hp_ctx"][0],
                "mindet": best_dep_bundle["hp_ctx"][1],
                "dev_acc_cls": best_dep_bundle["dev_acc_cls"],
            },
            "pos_ctx": {
                "minsupp": pack["hp_pos"][0],
                "mindet": pack["hp_pos"][1],
                "dev_acc": pack["acc_pos_dv"],
            },
            "case_ctx": {
                "minsupp": pack["hp_case"][0],
                "mindet": pack["hp_case"][1],
                "dev_acc": pack["acc_case_dv"],
            },
            "n_deprel_labels": n_dep,
            "deprel_labels": i2dep,
        },
        "scores": {
            "pos_r1_pred": {
                "test_acc": acc_pos_te,
                "dev_acc": acc_pos_dv,
                "shipped_r1_full_approx": 0.888,
            },
            "head_deprel": {
                "student_uas": student_uas,
                "student_las": student_las,
                "majority_uas": maj_uas,
                "majority_las": maj_las,
                "haiku_uas": haiku_uas,
                "haiku_las": haiku_las,
                "haiku_overlap_n": n_overlap,
                "within_5_of_haiku_las": within5,
                "dev_uas": best_dev_uas,
                "dev_majority_uas": maj_uas_dv,
                "dev_majority_las": maj_las_dv,
            },
            "morph_case": {
                "serve_honest_pred_attachment": acc_case_serve_honest,
                "oracle_ceiling_arm_a_full": case_oracle_ceiling,
                "r1_registers_off_baseline": case_r1_baseline,
                "recomputed_registers_off": acc_case_off,
                "below_0_90": below_90,
                "verb_government_precision_note": (
                    "~0.64 from r3min per-rule diagnostics; residual limiter"
                ),
            },
        },
        "r4_implication": R4_IMPLICATION if below_90 else (
            f"serve-honest={acc_case_serve_honest:.4f} >= 0.90; "
            "prereg hybrid-attachment gate not triggered by the <0.90 rule."
        ),
        "r4_implication_verbatim_if_below_0_90": R4_IMPLICATION,
        "n_test_sentences": len(test),
        "n_test_tokens": te_s["n"],
        "n_haiku_validated": len(haiku_rows),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
