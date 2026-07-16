"""German residual dump + subword-to-word alignment pipeline.

Drives fieldrun's batch source-dump CLI over GSD sentences, aligns per-subword
decode positions to GSD word tokens by character spans, pools residuals
(last-subword / mean-pool) at cumulative-layer checkpoints, and emits a
per-word grounded dataset keyed by (sent_id, token_index).

Text-only grounding (serve-honest): never reads gold targets/head_offset/deprel.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

GDATA = Path("/home/allans/code/germandata")
TASKS = GDATA / "tasks"

DEFAULT_FIELDRUN_DIR = Path("/home/allans/code/fieldrun")
DEFAULT_BUNDLE = "Qwen2.5-0.5B-Instruct"
DEFAULT_N = 64
DEFAULT_KCAND = 24
DEFAULT_CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)

_LAYER_RE = re.compile(r"^L(\d+)\.(attn|mlp)$")


# ---------------------------------------------------------------------------
# Data loading (text-only; never touch targets)
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gsd_text_split(split: str, limit: int | None = None) -> list[dict]:
    """Load head_deprel/{split}.jsonl keeping only sent_id, text, tokens.

    Deterministic order; first ``limit`` records if limit is set. Never reads targets.
    """
    path = TASKS / "head_deprel" / f"{split}.jsonl"
    out: list[dict] = []
    for rec in load_jsonl(path):
        out.append({
            "sent_id": rec["sent_id"],
            "text": rec.get("text", ""),
            "tokens": list(rec["tokens"]),
        })
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# GSD word char spans
# ---------------------------------------------------------------------------


def gsd_word_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Cursor walk: skip whitespace, match each token by len(tok).

    Raises AssertionError if text[span] != tok (caller should catch and drop sentence).
    """
    cursor = 0
    spans: list[tuple[int, int]] = []
    for tok in tokens:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        span = (cursor, cursor + len(tok))
        if text[span[0] : span[1]] != tok:
            raise AssertionError(
                f"GSD word/text mismatch: {tok!r} at {span} "
                f"(got {text[span[0]:span[1]]!r})"
            )
        spans.append(span)
        cursor += len(tok)
    return spans


# ---------------------------------------------------------------------------
# Subword content spans + assignment
# ---------------------------------------------------------------------------


def trim_content_span(text: str, a: int, b: int) -> tuple[int, int] | None:
    """Trim leading/trailing whitespace from char span; None if empty after trim."""
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    if a >= b:
        return None
    return (a, b)


def assign_subwords_to_words(
    subword_content_spans: list[tuple[int, int] | None],
    word_spans: list[tuple[int, int]],
) -> tuple[list[list[int]], list[int]]:
    """Assign each subword index to a GSD word by full containment of content span.

    Returns (word_to_subword_indices, straddling_subword_indices).
    Whitespace-only subwords (None content span) are skipped (not assigned, not straddling).
    """
    n_words = len(word_spans)
    word_subs: list[list[int]] = [[] for _ in range(n_words)]
    straddling: list[int] = []
    for si, cspan in enumerate(subword_content_spans):
        if cspan is None:
            continue
        a, b = cspan
        assigned = False
        for wi, (ws, we) in enumerate(word_spans):
            if ws <= a and b <= we:
                word_subs[wi].append(si)
                assigned = True
                break
        if not assigned:
            straddling.append(si)
    return word_subs, straddling


# ---------------------------------------------------------------------------
# Checkpoints / cumulative residuals
# ---------------------------------------------------------------------------


def parse_n_layer(blocks: list[str]) -> int:
    """n_layer = 1 + max matched layer index from L{{i}}.(attn|mlp) labels."""
    max_i = -1
    for lab in blocks:
        m = _LAYER_RE.match(lab)
        if m:
            max_i = max(max_i, int(m.group(1)))
    if max_i < 0:
        raise ValueError(f"no L*.attn/mlp blocks in {blocks[:5]}...")
    return max_i + 1


def checkpoint_block_indices(
    blocks: list[str], fracs: list[float]
) -> tuple[list[int], list[str], int]:
    """For each frac, resolve block_idx of L{{layer_idx(f)}}.mlp and its label."""
    n_layer = parse_n_layer(blocks)
    label_to_idx = {lab: i for i, lab in enumerate(blocks)}
    idxs: list[int] = []
    labels: list[str] = []
    for f in fracs:
        layer_idx = min(n_layer - 1, int(f * n_layer))
        lab = f"L{layer_idx}.mlp"
        if lab not in label_to_idx:
            raise KeyError(f"checkpoint block {lab!r} not in blocks")
        idxs.append(label_to_idx[lab])
        labels.append(lab)
    return idxs, labels, n_layer


def cumulative_residuals(
    D: np.ndarray, block_idxs: list[int]
) -> np.ndarray:
    """D: (N, nb, dim) -> (N, C, dim) cumulative sum over blocks 0..block_idx inclusive.

    float32 intermediate; caller casts to float16 for storage.
    """
    n, _nb, dim = D.shape
    c = len(block_idxs)
    out = np.zeros((n, c, dim), dtype=np.float32)
    # prefix sums for efficiency
    prefix = np.cumsum(D, axis=1)  # (N, nb, dim)
    for ci, bi in enumerate(block_idxs):
        out[:, ci, :] = prefix[:, bi, :]
    return out


# ---------------------------------------------------------------------------
# Batch dump parse (multi-sentence; sid-aware)
# ---------------------------------------------------------------------------


@dataclass
class SentenceDump:
    sid: str
    positions: list[int]  # dumped pos values (expected 1..L-2)
    curs: list[int]
    targets: list[int]
    D: np.ndarray  # (n_dumped, nb, dim) float32
    blocks: list[str]


def parse_batch_source_dump(path: Path) -> dict[str, SentenceDump]:
    """Group contiguous records by sid change. Returns sid -> SentenceDump."""
    by_sid: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["sid"]
            if sid not in by_sid:
                order.append(sid)
            by_sid[sid].append(rec)

    out: dict[str, SentenceDump] = {}
    for sid in order:
        recs = by_sid[sid]
        blocks = list(recs[0]["blocks"])
        D = np.array([r["d"] for r in recs], dtype=np.float32)
        out[sid] = SentenceDump(
            sid=sid,
            positions=[int(r["pos"]) for r in recs],
            curs=[int(r["cur"]) for r in recs],
            targets=[int(r["target"]) for r in recs],
            D=D,
            blocks=blocks,
        )
    return out


# ---------------------------------------------------------------------------
# Fieldrun CLI
# ---------------------------------------------------------------------------


def tokenizer_path_for_bundle(fieldrun_dir: Path, bundle: str) -> Path:
    return fieldrun_dir / "bundles" / bundle / f"{bundle}.tokenizer.json"


def bundle_stem(bundle: str) -> str:
    return f"bundles/{bundle}/{bundle}"


def run_fieldrun_batch(
    fieldrun_dir: Path,
    bundle: str,
    texts: list[dict],
    out_jsonl: Path,
    n: int,
    kcand: int,
) -> None:
    """Invoke fieldrun batch source-dump. texts: [{sid, text}, ...]."""
    binary = fieldrun_dir / "target" / "release" / "fieldrun"
    if not binary.is_file():
        raise FileNotFoundError(f"fieldrun binary not found: {binary}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", prefix="ground_dump_in_", delete=False, encoding="utf-8"
    ) as tf:
        in_path = Path(tf.name)
        for rec in texts:
            tf.write(json.dumps({"sid": rec["sid"], "text": rec["text"]}, ensure_ascii=False))
            tf.write("\n")

    try:
        cmd = [
            str(binary),
            "--bundle",
            bundle_stem(bundle),
            "--recursion-explain",
            "--source-dump",
            str(out_jsonl),
            "--texts",
            str(in_path),
            "--n",
            str(n),
            "--kcand",
            str(kcand),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(fieldrun_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"fieldrun failed (rc={proc.returncode}):\n"
                f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
            )
    finally:
        in_path.unlink(missing_ok=True)


def fieldrun_git_commit(fieldrun_dir: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(fieldrun_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


# ---------------------------------------------------------------------------
# Alignment + pooling for one sentence
# ---------------------------------------------------------------------------


@dataclass
class WordRow:
    sent_id: str
    token_index: int
    word: str
    covered: bool
    n_subwords: int
    last_pos: int
    resid_last: np.ndarray  # (C, dim) float16
    resid_mean: np.ndarray  # (C, dim) float16
    # debug / report helpers (not always stored)
    subword_indices: list[int] = field(default_factory=list)
    dumped_positions: list[int] = field(default_factory=list)


@dataclass
class SentenceAlignResult:
    sid: str
    rows: list[WordRow]
    n_straddling: int
    sanity_ok: bool
    drop_reason: str | None = None  # if set, rows empty / excluded
    first_mismatch_pos: int | None = None
    n_dumped: int = 0
    expected_dumped: int = 0
    py_ids: list[int] = field(default_factory=list)
    subword_tokens: list[str] = field(default_factory=list)


def align_sentence(
    rec: dict,
    dump: SentenceDump | None,
    tok: Tokenizer,
    block_idxs: list[int],
    dim: int,
    n_ck: int,
) -> SentenceAlignResult:
    """Align one GSD sentence to its dump; return word rows or drop reason."""
    sid = rec["sent_id"]
    text = rec["text"]
    tokens = rec["tokens"]
    zeros = np.zeros((n_ck, dim), dtype=np.float16)

    # GSD spans
    try:
        word_spans = gsd_word_spans(text, tokens)
    except AssertionError as e:
        return SentenceAlignResult(
            sid=sid,
            rows=[],
            n_straddling=0,
            sanity_ok=False,
            drop_reason=f"gsd_span_mismatch: {e}",
        )

    enc = tok.encode(text, add_special_tokens=False)
    py_ids = list(enc.ids)
    py_tokens = list(enc.tokens)
    py_offsets = list(enc.offsets)
    L = len(py_ids)
    expected_dumped = max(0, L - 2)

    if dump is None:
        # fieldrun skips very short tokenizations (e.g. L=3 → "too few ids")
        reason = (
            f"missing_dump (L={L}, expected_dumped={expected_dumped}; "
            "fieldrun may skip sentences with too few token ids)"
        )
        return SentenceAlignResult(
            sid=sid,
            rows=[],
            n_straddling=0,
            sanity_ok=False,
            drop_reason=reason,
            expected_dumped=expected_dumped,
            py_ids=py_ids,
            subword_tokens=py_tokens,
        )

    n_dumped = len(dump.positions)
    if n_dumped != expected_dumped:
        return SentenceAlignResult(
            sid=sid,
            rows=[],
            n_straddling=0,
            sanity_ok=False,
            drop_reason=(
                f"dump_count_mismatch: got {n_dumped} positions, expected L-2={expected_dumped} (L={L})"
            ),
            n_dumped=n_dumped,
            expected_dumped=expected_dumped,
            py_ids=py_ids,
            subword_tokens=py_tokens,
        )

    # Sanity gate: py_ids[p]==cur and py_ids[p+1]==target for every dumped p
    first_mismatch: int | None = None
    for i, p in enumerate(dump.positions):
        if p < 0 or p + 1 >= L:
            first_mismatch = p
            break
        if py_ids[p] != dump.curs[i] or py_ids[p + 1] != dump.targets[i]:
            first_mismatch = p
            break
    if first_mismatch is not None:
        return SentenceAlignResult(
            sid=sid,
            rows=[],
            n_straddling=0,
            sanity_ok=False,
            drop_reason=f"sanity_gate_fail at pos={first_mismatch}",
            first_mismatch_pos=first_mismatch,
            n_dumped=n_dumped,
            expected_dumped=expected_dumped,
            py_ids=py_ids,
            subword_tokens=py_tokens,
        )

    # Map dumped pos -> row index in D
    pos_to_row = {p: i for i, p in enumerate(dump.positions)}
    # Cumulative residuals at checkpoints for all dumped positions
    cum = cumulative_residuals(dump.D, block_idxs)  # (n_dumped, C, dim) float32

    # Content spans
    content_spans: list[tuple[int, int] | None] = []
    for a, b in py_offsets:
        content_spans.append(trim_content_span(text, a, b))

    word_subs, straddling = assign_subwords_to_words(content_spans, word_spans)

    rows: list[WordRow] = []
    for wi, tok_str in enumerate(tokens):
        sub_idxs = word_subs[wi]
        dumped_for_word = [si for si in sub_idxs if si in pos_to_row]
        n_sub = len(sub_idxs)

        if not dumped_for_word:
            rows.append(
                WordRow(
                    sent_id=sid,
                    token_index=wi,
                    word=tok_str,
                    covered=False,
                    n_subwords=n_sub,
                    last_pos=-1,
                    resid_last=zeros.copy(),
                    resid_mean=zeros.copy(),
                    subword_indices=sub_idxs,
                    dumped_positions=[],
                )
            )
            continue

        # last-subword: residual at the word's LAST subword, if dumped
        last_sub = sub_idxs[-1]
        if last_sub in pos_to_row:
            last_pos = last_sub
            resid_last = cum[pos_to_row[last_pos]].astype(np.float16)
            covered = True
        else:
            last_pos = -1
            resid_last = zeros.copy()
            covered = False

        # mean-pool over all dumped positions for this word
        mats = [cum[pos_to_row[p]] for p in dumped_for_word]
        resid_mean = np.mean(np.stack(mats, axis=0), axis=0).astype(np.float16)

        rows.append(
            WordRow(
                sent_id=sid,
                token_index=wi,
                word=tok_str,
                covered=covered,
                n_subwords=n_sub,
                last_pos=last_pos,
                resid_last=resid_last,
                resid_mean=resid_mean,
                subword_indices=sub_idxs,
                dumped_positions=dumped_for_word,
            )
        )

    # Internal consistency: f=1.0 cumulative == full residual
    if block_idxs:
        full_r = dump.D.sum(axis=1)  # (n_dumped, dim)
        last_ck = cum[:, -1, :]
        if not np.allclose(last_ck, full_r, rtol=1e-4, atol=1e-4):
            # soft warn via drop? Spec says assert — raise so caller logs
            max_diff = float(np.max(np.abs(last_ck - full_r)))
            raise AssertionError(
                f"{sid}: f=1.0 cumulative residual != full D.sum (max_diff={max_diff})"
            )

    return SentenceAlignResult(
        sid=sid,
        rows=rows,
        n_straddling=len(straddling),
        sanity_ok=True,
        n_dumped=n_dumped,
        expected_dumped=expected_dumped,
        py_ids=py_ids,
        subword_tokens=py_tokens,
    )


# ---------------------------------------------------------------------------
# NPZ I/O + merge (resumable)
# ---------------------------------------------------------------------------


def load_existing_npz(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = dict(np.load(path, allow_pickle=True))
    return data


def sents_present_in_npz(data: dict[str, Any]) -> set[str]:
    sids = data["sent_id"]
    if sids.dtype == object or sids.dtype.kind in ("U", "S"):
        return set(str(s) for s in sids.tolist())
    return set(str(s) for s in sids)


def rows_to_arrays(
    rows: list[WordRow],
    checkpoint_fracs: list[float],
    checkpoint_blocks: list[str],
    provenance: dict,
) -> dict[str, np.ndarray]:
    m = len(rows)
    if m == 0:
        dim = 0
        c = len(checkpoint_fracs)
        return {
            "sent_id": np.array([], dtype=object),
            "token_index": np.array([], dtype=np.int32),
            "word": np.array([], dtype=object),
            "covered": np.array([], dtype=bool),
            "n_subwords": np.array([], dtype=np.int32),
            "last_pos": np.array([], dtype=np.int32),
            "resid_last": np.zeros((0, c, dim), dtype=np.float16),
            "resid_mean": np.zeros((0, c, dim), dtype=np.float16),
            "checkpoint_fracs": np.array(checkpoint_fracs, dtype=np.float32),
            "checkpoint_blocks": np.array(checkpoint_blocks, dtype=object),
            "provenance": np.array(json.dumps(provenance), dtype=object),
        }
    dim = rows[0].resid_last.shape[1]
    c = rows[0].resid_last.shape[0]
    return {
        "sent_id": np.array([r.sent_id for r in rows], dtype=object),
        "token_index": np.array([r.token_index for r in rows], dtype=np.int32),
        "word": np.array([r.word for r in rows], dtype=object),
        "covered": np.array([r.covered for r in rows], dtype=bool),
        "n_subwords": np.array([r.n_subwords for r in rows], dtype=np.int32),
        "last_pos": np.array([r.last_pos for r in rows], dtype=np.int32),
        "resid_last": np.stack([r.resid_last for r in rows], axis=0).astype(np.float16),
        "resid_mean": np.stack([r.resid_mean for r in rows], axis=0).astype(np.float16),
        "checkpoint_fracs": np.array(checkpoint_fracs, dtype=np.float32),
        "checkpoint_blocks": np.array(checkpoint_blocks, dtype=object),
        "provenance": np.array(json.dumps(provenance), dtype=object),
    }


def merge_npz_rows(
    existing: dict[str, Any] | None, new_rows: list[WordRow],
    checkpoint_fracs: list[float],
    checkpoint_blocks: list[str],
    provenance: dict,
) -> dict[str, np.ndarray]:
    new_arr = rows_to_arrays(new_rows, checkpoint_fracs, checkpoint_blocks, provenance)
    if existing is None or len(existing.get("sent_id", [])) == 0:
        return new_arr
    # Keep existing rows for sids not in new; append new
    new_sids = set(r.sent_id for r in new_rows)
    keep = np.array([str(s) not in new_sids for s in existing["sent_id"]], dtype=bool)
    if not keep.any() and not new_rows:
        return new_arr
    parts: dict[str, list] = {}
    keys_row = [
        "sent_id", "token_index", "word", "covered", "n_subwords", "last_pos",
        "resid_last", "resid_mean",
    ]
    for k in keys_row:
        old = existing[k][keep] if keep.any() else existing[k][:0]
        if new_rows:
            parts[k] = [old, new_arr[k]]
        else:
            parts[k] = [old]
    out: dict[str, np.ndarray] = {}
    for k, arrs in parts.items():
        out[k] = np.concatenate(arrs, axis=0)
    out["checkpoint_fracs"] = np.array(checkpoint_fracs, dtype=np.float32)
    out["checkpoint_blocks"] = np.array(checkpoint_blocks, dtype=object)
    out["provenance"] = np.array(json.dumps(provenance), dtype=object)
    return out


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def parse_checkpoints(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    fieldrun_dir = Path(args.fieldrun_dir)
    bundle = args.bundle
    tok_path = tokenizer_path_for_bundle(fieldrun_dir, bundle)
    if not tok_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tok_path}")
    tok = Tokenizer.from_file(str(tok_path))

    if isinstance(args.checkpoints, str):
        fracs = parse_checkpoints(args.checkpoints)
    else:
        fracs = list(args.checkpoints)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path

    records = load_gsd_text_split(args.split, limit=args.limit)
    n_requested = len(records)
    print(f"loaded {n_requested} sentences from head_deprel/{args.split} (limit={args.limit})")

    existing = load_existing_npz(out_path)
    done_sids: set[str] = sents_present_in_npz(existing) if existing else set()
    todo = [r for r in records if r["sent_id"] not in done_sids]
    print(f"resumable: {len(done_sids)} sids already in output; {len(todo)} remaining")

    t0 = time.perf_counter()
    dumps: dict[str, SentenceDump] = {}
    if todo:
        with tempfile.TemporaryDirectory(prefix="ground_dump_") as td:
            dump_path = Path(td) / "batch_source.jsonl"
            print(f"running fieldrun on {len(todo)} sentences → {dump_path}")
            run_fieldrun_batch(
                fieldrun_dir=fieldrun_dir,
                bundle=bundle,
                texts=[{"sid": r["sent_id"], "text": r["text"]} for r in todo],
                out_jsonl=dump_path,
                n=args.n,
                kcand=args.kcand,
            )
            dumps = parse_batch_source_dump(dump_path)
            print(f"parsed dumps for {len(dumps)} sids")

    # Resolve checkpoints from first available dump's blocks (or re-infer)
    blocks: list[str] | None = None
    if dumps:
        blocks = next(iter(dumps.values())).blocks
    if todo and blocks is None:
        raise RuntimeError("no dump blocks available")

    if blocks is not None:
        block_idxs, ck_labels, n_layer = checkpoint_block_indices(blocks, fracs)
        dim = int(next(iter(dumps.values())).D.shape[-1])
        nb = len(blocks)
    else:
        # all sentences already present; nothing new to align
        assert existing is not None
        block_idxs = []
        n_layer = 0
        dim = int(existing["resid_last"].shape[-1])
        nb = 0
        fracs = [float(x) for x in existing["checkpoint_fracs"].tolist()]
        ck_labels = [str(x) for x in existing["checkpoint_blocks"].tolist()]

    # Align
    all_new_rows: list[WordRow] = []
    dropped: list[dict] = []
    n_straddling_total = 0
    sanity_sent_ok = 0
    sanity_sent_total = 0
    sanity_pos_ok = 0
    sanity_pos_total = 0
    align_examples: dict[str, SentenceAlignResult] = {}

    for rec in todo:
        sid = rec["sent_id"]
        dump = dumps.get(sid)
        # position-level sanity accounting needs dump even if we later drop
        if dump is not None and blocks is not None:
            # provisional block idxs already set
            pass
        try:
            result = align_sentence(
                rec, dump, tok, block_idxs, dim=dim, n_ck=len(fracs),
            )
        except AssertionError as e:
            dropped.append({"sid": sid, "reason": str(e)})
            print(f"DROP {sid}: {e}")
            continue

        if result.drop_reason:
            dropped.append({
                "sid": sid,
                "reason": result.drop_reason,
                "first_mismatch_pos": result.first_mismatch_pos,
            })
            print(f"DROP {sid}: {result.drop_reason}")
            # still count sanity fails
            if result.drop_reason.startswith("sanity_gate"):
                sanity_sent_total += 1
                # count matching positions up to first fail if dump present
                if dump is not None:
                    enc = tok.encode(rec["text"], add_special_tokens=False)
                    py_ids = list(enc.ids)
                    for i, p in enumerate(dump.positions):
                        sanity_pos_total += 1
                        ok = (
                            p + 1 < len(py_ids)
                            and py_ids[p] == dump.curs[i]
                            and py_ids[p + 1] == dump.targets[i]
                        )
                        if ok:
                            sanity_pos_ok += 1
                        else:
                            break
            continue

        sanity_sent_total += 1
        sanity_sent_ok += 1
        sanity_pos_total += result.n_dumped
        sanity_pos_ok += result.n_dumped
        n_straddling_total += result.n_straddling
        all_new_rows.extend(result.rows)
        align_examples[sid] = result
        if result.n_straddling:
            print(f"  {sid}: {result.n_straddling} boundary-straddling subword(s)")

    elapsed = time.perf_counter() - t0
    n_proc = max(1, len(todo))
    per_sent = elapsed / n_proc

    # Coverage on NEW rows: last-subword arm. Uncovered is almost always because the
    # word's last subword is position 0 or L-1 (never dumped by fieldrun).
    n_words = len(all_new_rows)
    n_covered = sum(1 for r in all_new_rows if r.covered)
    uncover_boundary = 0  # last subword at undumped endpoint (pos 0 or L-1)
    uncover_straddle_empty = 0  # no assigned subwords (all straddling / whitespace)
    uncover_other = 0
    for r in all_new_rows:
        if r.covered:
            continue
        if r.n_subwords == 0:
            uncover_straddle_empty += 1
        elif r.last_pos < 0:
            uncover_boundary += 1
        else:
            uncover_other += 1

    git_sha = fieldrun_git_commit(fieldrun_dir)
    provenance = {
        "model_bundle": bundle,
        "bundle_stem": bundle_stem(bundle),
        "tokenizer_path": str(tok_path),
        "fieldrun_dir": str(fieldrun_dir),
        "fieldrun_git": git_sha,
        "n": args.n,
        "kcand": args.kcand,
        "n_layer": n_layer,
        "nb": nb,
        "dim": dim,
        "sample_split": args.split,
        "n_sentences_requested": n_requested,
        "n_sentences_dumped": len(dumps),
        "n_sentences_already_present": len(done_sids),
        "n_sentences_aligned": sanity_sent_ok,
        "n_sentences_dropped": len(dropped),
        "dropped": dropped,
        "n_straddling_subwords": n_straddling_total,
        "pooling": {
            "last_subword": (
                "residual at the word's last subword position if dumped; else uncovered"
            ),
            "mean_pool": (
                "mean of residuals at all of the word's dumped subword positions; "
                "uncovered if zero dumped positions"
            ),
        },
        "checkpoint_fracs": fracs,
        "checkpoint_blocks": ck_labels,
        "sanity_gate": {
            "sentence_pass_rate": (sanity_sent_ok / sanity_sent_total) if sanity_sent_total else None,
            "position_pass_rate": (sanity_pos_ok / sanity_pos_total) if sanity_pos_total else None,
            "sentences_ok": sanity_sent_ok,
            "sentences_checked": sanity_sent_total,
            "positions_ok": sanity_pos_ok,
            "positions_checked": sanity_pos_total,
            "definition": "py_ids[p]==cur and py_ids[p+1]==target for every dumped p",
        },
        "coverage_new_rows": {
            "n_words": n_words,
            "n_covered_last_subword": n_covered,
            "frac_covered": (n_covered / n_words) if n_words else None,
            "n_uncovered": n_words - n_covered,
            "n_uncovered_sentence_boundary": uncover_boundary,
            "n_uncovered_no_assigned_subwords": uncover_straddle_empty,
            "n_uncovered_other": uncover_other,
            "n_straddling_subwords": n_straddling_total,
            "n_dropped_sentences": len(dropped),
            "drop_reasons": dropped,
        },
        "timing": {
            "wall_sec_total": elapsed,
            "wall_sec_per_sentence": per_sent,
            "n_sentences_processed_this_run": len(todo),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    arrays = merge_npz_rows(existing, all_new_rows, fracs, ck_labels, provenance)
    save_npz(out_path, arrays)
    print(f"wrote {out_path}  rows={len(arrays['sent_id'])}")

    # Coverage report
    covered_all = arrays["covered"]
    n_all = len(covered_all)
    n_cov_all = int(covered_all.sum()) if n_all else 0
    print("\n=== COVERAGE REPORT ===")
    print(f"words total (npz): {n_all}")
    print(f"covered (last-subword): {n_cov_all} / {n_all} = {n_cov_all / n_all if n_all else 0:.4f}")
    print(f"uncovered: {n_all - n_cov_all}")
    print(f"  (this run) sentence-boundary (last subword undumped): {uncover_boundary}")
    print(f"  (this run) no assigned subwords (straddle/whitespace): {uncover_straddle_empty}")
    print(f"  (this run) other uncovered: {uncover_other}")
    print(f"  dropped sentences (no word rows): {len(dropped)}")
    for d in dropped:
        print(f"    - {d}")
    print(f"  straddling subwords (this run): {n_straddling_total}")
    print("\n=== SANITY GATE ===")
    print(
        f"sentence pass rate: {sanity_sent_ok}/{sanity_sent_total} = "
        f"{(sanity_sent_ok / sanity_sent_total) if sanity_sent_total else 'n/a'}"
    )
    print(
        f"position pass rate: {sanity_pos_ok}/{sanity_pos_total} = "
        f"{(sanity_pos_ok / sanity_pos_total) if sanity_pos_total else 'n/a'}"
    )
    print("\n=== TIMING ===")
    print(f"total wall: {elapsed:.2f}s  per sentence: {per_sent:.2f}s  (n={len(todo)})")
    print("\n=== SHAPES ===")
    for k, v in arrays.items():
        if k == "provenance":
            continue
        print(f"  {k}: {getattr(v, 'shape', None)} {v.dtype}")
    print("\n=== PROVENANCE ===")
    print(json.dumps(provenance, indent=2, default=str))

    # Worked example: einzigartiger from dev-s1
    _print_worked_example(arrays, align_examples, fracs, ck_labels)

    return {
        "out_path": str(out_path),
        "provenance": provenance,
        "arrays": arrays,
        "align_examples": align_examples,
    }


def _print_worked_example(
    arrays: dict[str, np.ndarray],
    align_examples: dict[str, SentenceAlignResult],
    fracs: list[float],
    ck_labels: list[str],
) -> None:
    print("\n=== WORKED EXAMPLE (einzigartiger / first multi-subword word) ===")
    sids = arrays["sent_id"]
    words = arrays["word"]
    idxs = arrays["token_index"]
    last_pos = arrays["last_pos"]
    covered = arrays["covered"]
    resid_last = arrays["resid_last"]
    resid_mean = arrays["resid_mean"]
    n_sub = arrays["n_subwords"]

    # Prefer dev-s1 "einzigartiger"
    pick = None
    for i in range(len(sids)):
        if str(sids[i]) == "dev-s1" and str(words[i]) == "einzigartiger":
            pick = i
            break
    if pick is None:
        for i in range(len(sids)):
            if int(n_sub[i]) >= 3 and bool(covered[i]):
                pick = i
                break
    if pick is None:
        print("(no suitable example row)")
        return

    sid = str(sids[pick])
    word = str(words[pick])
    ti = int(idxs[pick])
    print(f"sent_id={sid} token_index={ti} word={word!r}")
    print(f"covered={bool(covered[pick])} n_subwords={int(n_sub[pick])} last_pos={int(last_pos[pick])}")
    ar = align_examples.get(sid)
    if ar is not None:
        row = next((r for r in ar.rows if r.token_index == ti), None)
        if row is not None:
            sw = [ar.subword_tokens[j] for j in row.subword_indices]
            print(f"subwords: {sw} indices={row.subword_indices}")
            print(f"dumped positions used (mean): {row.dumped_positions}")
    print("checkpoint residual norms (last-subword / mean-pool):")
    for ci, (f, lab) in enumerate(zip(fracs, ck_labels, strict=True)):
        nl = float(np.linalg.norm(resid_last[pick, ci].astype(np.float32)))
        nm = float(np.linalg.norm(resid_mean[pick, ci].astype(np.float32)))
        print(f"  f={f} ({lab}): ||last||={nl:.4f}  ||mean||={nm:.4f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GSD word-grounded residual dump pipeline")
    p.add_argument("--split", default="dev", help="head_deprel split name")
    p.add_argument("--limit", type=int, default=20, help="max sentences (first N, deterministic)")
    p.add_argument("--bundle", default=DEFAULT_BUNDLE, help="fieldrun bundle name")
    p.add_argument(
        "--out",
        default="data/grounded/qwen05b_gsd_sample20.npz",
        help="output npz path (relative to repo root if not absolute)",
    )
    p.add_argument("--n", type=int, default=DEFAULT_N, help="fieldrun --n (max positions)")
    p.add_argument("--kcand", type=int, default=DEFAULT_KCAND, help="fieldrun --kcand")
    p.add_argument(
        "--checkpoints",
        default=",".join(str(x) for x in DEFAULT_CHECKPOINTS),
        help="comma-separated cumulative layer fractions",
    )
    p.add_argument(
        "--fieldrun-dir",
        default=str(DEFAULT_FIELDRUN_DIR),
        help="path to fieldrun repo (binary + bundles)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
