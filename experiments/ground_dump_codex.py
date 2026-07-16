"""Build word-keyed grounded residuals from a batched fieldrun source dump.

The correctness boundary in this driver is character-span alignment.  Raw fieldrun
records are retained with their ``sid``, ``pos``, and observed ``cur`` token id;
multi-sentence source dumps must never pass through the flattening source loader.

The output NPZ has one row per clean GSD word.  ``sent_ids`` and ``token_index`` are
the row key, while ``last_residual`` and ``mean_residual`` have shape
``(n_words, n_checkpoints, dim)`` and dtype float16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

FIELDRun = Path("/home/allans/code/fieldrun/target/release/fieldrun")
BUNDLE_ROOT = Path("/home/allans/code/fieldrun/bundles")
GSD_ROOT = Path("/home/allans/code/germandata/tasks/head_deprel")
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "grounded"
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
POOLING_ARMS = ("last-subword", "mean-pool")


@dataclass(frozen=True)
class GsdSentence:
    sent_id: str
    text: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class SentenceDump:
    """One sentence's sid-preserving raw source records."""

    sid: str
    positions: np.ndarray  # (P,), subword indices (fieldrun positions)
    cur: np.ndarray  # (P,), observed input token ids
    D: np.ndarray  # (P, nb, dim), per-block contributions
    r: np.ndarray  # (P, dim), D.sum(axis=1)
    blocks: tuple[str, ...]


@dataclass(frozen=True)
class UnalignedSubword:
    sent_id: str
    subword_index: int
    span: tuple[int, int]
    trimmed_span: tuple[int, int] | None
    text: str


@dataclass(frozen=True)
class CoverageCounts:
    total_words: int
    clean_words: int
    boundary_straddling_subwords: int
    leading_single_subword_word_gaps: int
    last_word_gaps: int

    @property
    def fraction(self) -> float:
        return self.clean_words / self.total_words if self.total_words else 0.0


@dataclass(frozen=True)
class AlignmentResult:
    word_spans: tuple[tuple[int, int], ...]
    word_to_subwords: tuple[tuple[int, ...], ...]
    clean_words: tuple[bool, ...]
    unaligned_subwords: tuple[UnalignedSubword, ...]
    coverage: CoverageCounts


@dataclass(frozen=True)
class EncodedSentence:
    sentence: GsdSentence
    original_ids: tuple[int, ...]
    original_offsets: tuple[tuple[int, int], ...]
    dump_text: str
    dump_ids: tuple[int, ...]
    pad_safety_passed: bool


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    positions_complete: bool
    cur_matches: bool
    expected_positions: tuple[int, ...]


@dataclass(frozen=True)
class ReducedWords:
    token_indices: np.ndarray
    last_subword_indices: np.ndarray
    last_residual: np.ndarray
    mean_residual: np.ndarray


def load_gsd_sample(path: str | Path, limit: int) -> list[GsdSentence]:
    """Read only surface fields from the deterministic first-``limit`` sample."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    sentences: list[GsdSentence] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if len(sentences) >= limit:
                break
            if not line.strip():
                continue
            raw = json.loads(line)
            sentences.append(
                GsdSentence(
                    sent_id=str(raw["sent_id"]),
                    text=str(raw["text"]),
                    tokens=tuple(str(token) for token in raw["tokens"]),
                )
            )
    if len(sentences) != limit:
        raise ValueError(f"{path}: requested {limit} sentences, found {len(sentences)}")
    if len({sentence.sent_id for sentence in sentences}) != len(sentences):
        raise ValueError("sample contains duplicate sent_id values")
    return sentences


def load_batched_source_dump(path: str | Path) -> dict[str, SentenceDump]:
    """Parse and group a raw source dump without discarding sid/pos/cur.

    This intentionally does not call :func:`pil.fieldrun_io.load_source_dump`, whose
    flat return type cannot preserve sentence identity in a batch.
    """
    grouped: dict[str, list[tuple[int, int, np.ndarray, tuple[str, ...]]]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            missing = {"sid", "pos", "cur", "d", "blocks"} - raw.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            contribution = np.asarray(raw["d"], dtype=np.float32)
            if contribution.ndim != 2:
                raise ValueError(f"{path}:{line_number}: d must be (nb, dim), got {contribution.shape}")
            blocks = tuple(str(block) for block in raw["blocks"])
            if len(blocks) != contribution.shape[0]:
                raise ValueError(f"{path}:{line_number}: block count does not match d")
            grouped[str(raw["sid"])].append(
                (int(raw["pos"]), int(raw["cur"]), contribution, blocks)
            )
    if not grouped:
        raise ValueError(f"{path}: no records")

    result: dict[str, SentenceDump] = {}
    for sid, rows in grouped.items():
        rows.sort(key=lambda row: row[0])
        blocks = rows[0][3]
        shape = rows[0][2].shape
        if any(row[3] != blocks or row[2].shape != shape for row in rows):
            raise ValueError(f"{path}: inconsistent blocks or d shape for sid={sid}")
        D = np.stack([row[2] for row in rows], axis=0)
        result[sid] = SentenceDump(
            sid=sid,
            positions=np.asarray([row[0] for row in rows], dtype=np.int64),
            cur=np.asarray([row[1] for row in rows], dtype=np.int64),
            D=D,
            r=D.sum(axis=1),
            blocks=blocks,
        )
    return result


def compute_word_spans(text: str, tokens: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Match GSD surfaces left-to-right, skipping only inter-token whitespace.

    GSD represents punctuation as separate tokens even when it is adjacent in the
    sentence string, so matching whole whitespace-delimited runs would incorrectly
    combine a final word and its full stop.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token_index, token in enumerate(tokens):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        start = cursor
        end = start + len(token)
        surface = text[start:end]
        if surface != token:
            raise ValueError(
                f"token {token_index} mismatch: expected {token!r}, found {surface!r} "
                f"in {text!r}"
            )
        cursor = end
        spans.append((start, end))
    if any(not char.isspace() for char in text[cursor:]):
        raise ValueError(f"unmatched non-whitespace suffix {text[cursor:]!r} in {text!r}")
    return tuple(spans)


def _trimmed_span(text: str, span: tuple[int, int]) -> tuple[int, int] | None:
    start, end = span
    if start < 0 or end < start or end > len(text):
        raise ValueError(f"invalid tokenizer offset {span} for text of length {len(text)}")
    while start < end and text[start].isspace():
        start += 1
    return None if start >= end else (start, end)


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def align_subwords_to_words(
    sent_id: str,
    text: str,
    tokens: Sequence[str],
    subword_offsets: Sequence[tuple[int, int]],
    available_positions: set[int] | frozenset[int],
) -> AlignmentResult:
    """Align tokenizer offsets to GSD words using trimmed-span containment only."""
    word_spans = compute_word_spans(text, tokens)
    assignments: list[list[int]] = [[] for _ in word_spans]
    trimmed: list[tuple[int, int] | None] = []
    unaligned: list[UnalignedSubword] = []

    for subword_index, raw_span in enumerate(subword_offsets):
        span = (int(raw_span[0]), int(raw_span[1]))
        clean_span = _trimmed_span(text, span)
        trimmed.append(clean_span)
        owners = [] if clean_span is None else [
            index
            for index, word_span in enumerate(word_spans)
            if word_span[0] <= clean_span[0] and clean_span[1] <= word_span[1]
        ]
        if len(owners) == 1:
            assignments[owners[0]].append(subword_index)
        else:
            unaligned.append(
                UnalignedSubword(sent_id, subword_index, span, clean_span, text)
            )

    unaligned_indices = {item.subword_index for item in unaligned}
    clean_words: list[bool] = []
    leading_gaps = 0
    last_gaps = 0
    final_subword = len(subword_offsets) - 1
    final_word = len(word_spans) - 1
    for word_index, word_span in enumerate(word_spans):
        assigned = assignments[word_index]
        has_straddler = any(
            index in unaligned_indices and span is not None and _overlaps(span, word_span)
            for index, span in enumerate(trimmed)
        )
        last_available = bool(assigned) and assigned[-1] in available_positions
        leading_gap = word_index == 0 and assigned == [0] and not last_available
        last_gap = (
            word_index == final_word
            and bool(assigned)
            and assigned[-1] == final_subword
            and not last_available
        )
        leading_gaps += int(leading_gap)
        last_gaps += int(last_gap)
        clean_words.append(bool(assigned) and not has_straddler and last_available)

    coverage = CoverageCounts(
        total_words=len(word_spans),
        clean_words=sum(clean_words),
        boundary_straddling_subwords=len(unaligned),
        leading_single_subword_word_gaps=leading_gaps,
        last_word_gaps=last_gaps,
    )
    return AlignmentResult(
        word_spans=word_spans,
        word_to_subwords=tuple(tuple(indices) for indices in assignments),
        clean_words=tuple(clean_words),
        unaligned_subwords=tuple(unaligned),
        coverage=coverage,
    )


def choose_checkpoint_layers(blocks: Sequence[str]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Choose quarter/half/three-quarter/final layer checkpoints."""
    layer_count = (len(blocks) - 1) // 2
    if layer_count <= 0:
        raise ValueError(f"cannot infer transformer layers from {len(blocks)} blocks")
    chosen: list[int] = []
    fractions: list[float] = []
    for fraction in CHECKPOINT_FRACTIONS:
        layer = max(0, round(layer_count * fraction) - 1)
        if layer not in chosen:
            marker = f"L{layer}.mlp"
            if marker not in blocks:
                raise ValueError(f"checkpoint block {marker!r} absent from blocks")
            chosen.append(layer)
            fractions.append(fraction)
    return tuple(chosen), tuple(fractions)


def reduce_clean_words(
    dump: SentenceDump,
    alignment: AlignmentResult,
    checkpoint_layers: Sequence[int],
) -> ReducedWords:
    """Reduce clean words to last-subword and available-subword-mean residuals."""
    if len(set(dump.positions.tolist())) != len(dump.positions):
        raise ValueError(f"sid={dump.sid}: duplicate fieldrun positions")
    position_to_row = {int(position): row for row, position in enumerate(dump.positions)}
    block_ends = []
    for layer in checkpoint_layers:
        marker = f"L{layer}.mlp"
        try:
            block_ends.append(dump.blocks.index(marker))
        except ValueError as error:
            raise ValueError(f"sid={dump.sid}: missing checkpoint block {marker}") from error
    cumulative = np.cumsum(dump.D, axis=1, dtype=np.float32)[:, block_ends, :]

    token_indices: list[int] = []
    last_indices: list[int] = []
    last_rows: list[np.ndarray] = []
    mean_rows: list[np.ndarray] = []
    for token_index, (subwords, is_clean) in enumerate(
        zip(alignment.word_to_subwords, alignment.clean_words, strict=True)
    ):
        if not is_clean:
            continue
        available = [index for index in subwords if index in position_to_row]
        if not available or subwords[-1] not in position_to_row:
            raise AssertionError("clean-word inclusion gate admitted a missing last position")
        token_indices.append(token_index)
        last_indices.append(subwords[-1])
        last_rows.append(cumulative[position_to_row[subwords[-1]]])
        mean_rows.append(np.mean([cumulative[position_to_row[index]] for index in available], axis=0))

    shape = (0, len(checkpoint_layers), dump.D.shape[2])
    last_array = np.stack(last_rows) if last_rows else np.empty(shape, dtype=np.float32)
    mean_array = np.stack(mean_rows) if mean_rows else np.empty(shape, dtype=np.float32)
    return ReducedWords(
        token_indices=np.asarray(token_indices, dtype=np.int64),
        last_subword_indices=np.asarray(last_indices, dtype=np.int64),
        last_residual=last_array.astype(np.float16),
        mean_residual=mean_array.astype(np.float16),
    )


def encode_with_safe_pad(
    tokenizer: Tokenizer,
    sentence: GsdSentence,
    pad: str,
) -> EncodedSentence:
    original = tokenizer.encode(sentence.text, add_special_tokens=False)
    padded_text = sentence.text + pad
    padded = tokenizer.encode(padded_text, add_special_tokens=False)
    original_ids = tuple(original.ids)
    padded_ids = tuple(padded.ids)
    pad_safe = padded_ids[: len(original_ids)] == original_ids and len(padded_ids) > len(original_ids)
    return EncodedSentence(
        sentence=sentence,
        original_ids=original_ids,
        original_offsets=tuple((int(start), int(end)) for start, end in original.offsets),
        dump_text=padded_text if pad_safe else sentence.text,
        dump_ids=padded_ids if pad_safe else original_ids,
        pad_safety_passed=pad_safe,
    )


def validate_sentence_dump(dump: SentenceDump, dump_ids: Sequence[int], n: int) -> ValidationResult:
    expected = tuple(range(1, min(len(dump_ids) - 2, n) + 1))
    actual = tuple(int(position) for position in dump.positions)
    complete = actual == expected
    cur_matches = len(actual) == len(dump.cur) and all(
        0 <= position < len(dump_ids) and int(cur) == int(dump_ids[position])
        for position, cur in zip(actual, dump.cur, strict=True)
    )
    return ValidationResult(complete and cur_matches, complete, cur_matches, expected)


def _write_text_batch(path: Path, encoded: Sequence[EncodedSentence]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in encoded:
            record = {"sid": item.sentence.sent_id, "text": item.dump_text}
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def run_fieldrun_batch(
    fieldrun: Path,
    bundle_path: Path,
    encoded: Sequence[EncodedSentence],
    n: int,
    kcand: int,
    work_dir: Path,
) -> tuple[Path, float]:
    input_path = work_dir / "texts.jsonl"
    output_path = work_dir / "source.jsonl"
    _write_text_batch(input_path, encoded)
    command = [
        str(fieldrun),
        "--bundle",
        str(bundle_path),
        "--recursion-explain",
        "--source-dump",
        str(output_path),
        "--texts",
        str(input_path),
        "--n",
        str(n),
        "--kcand",
        str(kcand),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"fieldrun failed with exit {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not output_path.exists():
        raise RuntimeError("fieldrun succeeded without creating its source dump")
    return output_path, elapsed


def _sum_coverage(counts: Sequence[CoverageCounts]) -> CoverageCounts:
    return CoverageCounts(
        total_words=sum(item.total_words for item in counts),
        clean_words=sum(item.clean_words for item in counts),
        boundary_straddling_subwords=sum(item.boundary_straddling_subwords for item in counts),
        leading_single_subword_word_gaps=sum(item.leading_single_subword_word_gaps for item in counts),
        last_word_gaps=sum(item.last_word_gaps for item in counts),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bundle(bundle_stem: str) -> tuple[Path, Path]:
    supplied = Path(bundle_stem)
    bundle_path = supplied if supplied.is_absolute() else BUNDLE_ROOT / bundle_stem / bundle_stem
    tokenizer_path = Path(f"{bundle_path}.tokenizer.json")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    return bundle_path, tokenizer_path


def _safe_path_component(value: str) -> str:
    component = "".join(char if char.isalnum() or char in "-_." else "-" for char in value)
    return component.strip("-.") or "unnamed"


def _encoded_sha256(encoded: Sequence[EncodedSentence]) -> str:
    payload = [
        {
            "sent_id": item.sentence.sent_id,
            "text": item.sentence.text,
            "tokens": item.sentence.tokens,
            "original_ids": item.original_ids,
            "original_offsets": item.original_offsets,
            "dump_text": item.dump_text,
            "dump_ids": item.dump_ids,
            "pad_safety_passed": item.pad_safety_passed,
        }
        for item in encoded
    ]
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _partial_directory(output_root: Path, config: dict[str, Any]) -> Path:
    serialized = json.dumps(config, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    config_hash = hashlib.sha256(serialized.encode()).hexdigest()[:12]
    prefix = (
        f"{_safe_path_component(str(config['model_tag']))}_gsd_"
        f"{_safe_path_component(str(config['split']))}_n{config['limit']}_"
        f"chunk{config['chunk_size']}_"
        f"{_safe_path_component(Path(str(config['bundle_path'])).name)}"
    )
    return output_root / f"{prefix}_{config_hash}_partials"


def _load_chunk_metadata(path: Path, expected_config: dict[str, Any]) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as partial:
            raw = partial["chunk_provenance"]
            metadata = json.loads(str(raw.item()))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid chunk partial: {error}") from error

    actual_config = metadata.get("config")
    if actual_config != expected_config:
        actual = actual_config if isinstance(actual_config, dict) else {}
        mismatches = [
            f"{key} (expected {value!r}, found {actual.get(key)!r})"
            for key, value in expected_config.items()
            if actual.get(key) != value
        ]
        extra = sorted(set(actual) - set(expected_config))
        if extra:
            mismatches.append(f"unexpected keys {extra}")
        detail = "; ".join(mismatches) or "configuration blob differs"
        raise ValueError(f"{path}: chunk partial provenance mismatch: {detail}")
    return metadata


def _write_chunk_partial(
    *,
    partial_path: Path,
    config: dict[str, Any],
    encoded: Sequence[EncodedSentence],
    tokenizer: Tokenizer,
    fieldrun_path: Path,
    bundle_path: Path,
    n: int,
    kcand: int,
    expected_checkpoint_layers: tuple[int, ...] | None,
    expected_blocks: tuple[str, ...] | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ground-dump-codex-") as temp_name:
        dump_path, fieldrun_seconds = run_fieldrun_batch(
            fieldrun_path, bundle_path, encoded, n, kcand, Path(temp_name)
        )
        dumps = load_batched_source_dump(dump_path)

    unknown_sids = set(dumps) - {item.sentence.sent_id for item in encoded}
    if unknown_sids:
        raise ValueError(f"fieldrun returned unknown sid values: {sorted(unknown_sids)}")

    sent_ids: list[str] = []
    token_indices: list[int] = []
    word_text: list[str] = []
    last_subword_indices: list[int] = []
    last_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []
    per_sentence: list[dict[str, Any]] = []
    aggregate_coverages: list[CoverageCounts] = []
    sanity_passes = 0
    dropped_sentences: list[dict[str, Any]] = []
    checkpoint_layers: tuple[int, ...] | None = None
    checkpoint_fractions: tuple[float, ...] | None = None
    blocks_reference: tuple[str, ...] | None = None
    worked_example: dict[str, Any] | None = None
    reduced_sentence_count = 0

    for item in encoded:
        sid = item.sentence.sent_id
        dump = dumps.get(sid)
        if dump is None:
            dropped_sentences.append({"sent_id": sid, "reason": "no fieldrun records"})
            zero_coverage = CoverageCounts(len(item.sentence.tokens), 0, 0, 0, 0)
            aggregate_coverages.append(zero_coverage)
            per_sentence.append(
                {
                    "sent_id": sid,
                    "sanity_passed": False,
                    "dropped": True,
                    "drop_reason": "no fieldrun records",
                    "coverage": {**asdict(zero_coverage), "fraction": 0.0},
                }
            )
            continue
        validation = validate_sentence_dump(dump, item.dump_ids, n)
        sanity_passes += int(validation.passed)
        if not validation.passed:
            dropped_sentences.append(
                {
                    "sent_id": sid,
                    "reason": "position completeness or cur mismatch",
                    "positions_complete": validation.positions_complete,
                    "cur_matches": validation.cur_matches,
                }
            )
            zero_coverage = CoverageCounts(len(item.sentence.tokens), 0, 0, 0, 0)
            aggregate_coverages.append(zero_coverage)
            per_sentence.append(
                {
                    "sent_id": sid,
                    "sanity_passed": False,
                    "dropped": True,
                    "drop_reason": "position completeness or cur mismatch",
                    "coverage": {**asdict(zero_coverage), "fraction": 0.0},
                }
            )
            continue

        layers, fractions = choose_checkpoint_layers(dump.blocks)
        if expected_checkpoint_layers is not None and (
            layers != expected_checkpoint_layers or dump.blocks != expected_blocks
        ):
            raise ValueError(f"sid={sid}: block/checkpoint schema changed within batch")
        if checkpoint_layers is None:
            checkpoint_layers, checkpoint_fractions = layers, fractions
            blocks_reference = dump.blocks
        elif layers != checkpoint_layers or dump.blocks != blocks_reference:
            raise ValueError(f"sid={sid}: block/checkpoint schema changed within batch")

        try:
            alignment = align_subwords_to_words(
                sid,
                item.sentence.text,
                item.sentence.tokens,
                item.original_offsets,
                set(int(position) for position in dump.positions),
            )
        except ValueError as error:
            reason = f"GSD surface alignment error: {error}"
            dropped_sentences.append({"sent_id": sid, "reason": reason})
            zero_coverage = CoverageCounts(len(item.sentence.tokens), 0, 0, 0, 0)
            aggregate_coverages.append(zero_coverage)
            per_sentence.append(
                {
                    "sent_id": sid,
                    "sanity_passed": True,
                    "dropped": True,
                    "drop_reason": reason,
                    "pad_safety_passed": item.pad_safety_passed,
                    "coverage": {**asdict(zero_coverage), "fraction": 0.0},
                }
            )
            continue
        reduced = reduce_clean_words(dump, alignment, layers)
        reduced_sentence_count += 1
        count = len(reduced.token_indices)
        sent_ids.extend([sid] * count)
        token_indices.extend(int(index) for index in reduced.token_indices)
        word_text.extend(item.sentence.tokens[int(index)] for index in reduced.token_indices)
        last_subword_indices.extend(int(index) for index in reduced.last_subword_indices)
        last_parts.append(reduced.last_residual)
        mean_parts.append(reduced.mean_residual)
        aggregate_coverages.append(alignment.coverage)
        per_sentence.append(
            {
                "sent_id": sid,
                "sanity_passed": True,
                "dropped": False,
                "pad_safety_passed": item.pad_safety_passed,
                "coverage": {**asdict(alignment.coverage), "fraction": alignment.coverage.fraction},
                "unaligned_subwords": [asdict(detail) for detail in alignment.unaligned_subwords],
            }
        )
        if worked_example is None:
            candidates = [
                word_index
                for word_index, subwords in enumerate(alignment.word_to_subwords)
                if alignment.clean_words[word_index] and len(subwords) > 1
            ]
            if candidates:
                word_index = candidates[0]
                output_row = int(np.flatnonzero(reduced.token_indices == word_index)[0])
                subword_indices = alignment.word_to_subwords[word_index]
                encoding = tokenizer.encode(item.sentence.text, add_special_tokens=False)
                worked_example = {
                    "sent_id": sid,
                    "token_index": word_index,
                    "word": item.sentence.tokens[word_index],
                    "subword_indices": list(subword_indices),
                    "subword_tokens": [encoding.tokens[index] for index in subword_indices],
                    "fieldrun_position": subword_indices[-1],
                    "checkpoint_layer": layers[0],
                    "last_residual_norm": float(
                        np.linalg.norm(reduced.last_residual[output_row, 0].astype(np.float32))
                    ),
                }

    aggregate = _sum_coverage(aggregate_coverages)
    pad_passed = sum(item.pad_safety_passed for item in encoded)
    if last_parts:
        last_residual = np.concatenate(last_parts, axis=0)
        mean_residual = np.concatenate(mean_parts, axis=0)
    elif checkpoint_layers is not None:
        first_dump = next(dump for dump in dumps.values() if dump.blocks == blocks_reference)
        shape = (0, len(checkpoint_layers), first_dump.D.shape[2])
        last_residual = np.empty(shape, dtype=np.float16)
        mean_residual = np.empty(shape, dtype=np.float16)
    else:
        last_residual = np.empty((0, 0, 0), dtype=np.float16)
        mean_residual = np.empty((0, 0, 0), dtype=np.float16)

    metadata: dict[str, Any] = {
        "config": config,
        "checkpoint_layers": None if checkpoint_layers is None else list(checkpoint_layers),
        "checkpoint_fractions": (
            None if checkpoint_fractions is None else list(checkpoint_fractions)
        ),
        "blocks": None if blocks_reference is None else list(blocks_reference),
        "per_sentence": per_sentence,
        "dropped_sentences": dropped_sentences,
        "sanity_passed_sentences": sanity_passes,
        "sentence_count": len(encoded),
        "coverage": asdict(aggregate),
        "pad_safety": {
            "passed_sentences": pad_passed,
            "total_sentences": len(encoded),
            "per_sentence": {
                item.sentence.sent_id: item.pad_safety_passed for item in encoded
            },
        },
        "fieldrun_subprocess_seconds": fieldrun_seconds,
        "worked_example": worked_example,
        "reduced_sentence_count": reduced_sentence_count,
    }

    partial_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=partial_path.parent, prefix=f".{partial_path.stem}-", suffix=".npz", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                sent_ids=np.asarray(sent_ids, dtype=np.str_),
                token_index=np.asarray(token_indices, dtype=np.int64),
                word_text=np.asarray(word_text, dtype=np.str_),
                last_subword_index=np.asarray(last_subword_indices, dtype=np.int64),
                last_residual=last_residual,
                mean_residual=mean_residual,
                chunk_provenance=np.asarray(
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                ),
            )
        temporary_path.replace(partial_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return metadata


def build_dataset(
    *,
    model_tag: str,
    bundle_stem: str,
    split: str,
    limit: int,
    fieldrun_path: Path = FIELDRun,
    data_root: Path = GSD_ROOT,
    output_root: Path = OUTPUT_ROOT,
    kcand: int = 24,
    pad: str = " x",
    chunk_size: int = 40,
    keep_chunk_partials: bool = True,
) -> tuple[Path, dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    bundle_path, tokenizer_path = _resolve_bundle(bundle_stem)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sentences = load_gsd_sample(data_root / f"{split}.jsonl", limit)
    encoded = [encode_with_safe_pad(tokenizer, sentence, pad) for sentence in sentences]
    n = max(len(item.dump_ids) for item in encoded) + 8
    tokenizer_sha256 = _sha256(tokenizer_path)
    common_config: dict[str, Any] = {
        "format_version": 1,
        "model_tag": model_tag,
        "bundle_stem": bundle_stem,
        "bundle_path": str(bundle_path),
        "fieldrun_path": str(fieldrun_path),
        "split": split,
        "limit": limit,
        "chunk_size": chunk_size,
        "kcand": kcand,
        "pad": pad,
        "n": n,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "encoded_sha256": _encoded_sha256(encoded),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    partial_dir = _partial_directory(output_root, common_config)
    partial_dir.mkdir(parents=True, exist_ok=True)
    chunk_count = (len(encoded) + chunk_size - 1) // chunk_size
    partial_paths: list[Path] = []
    checkpoint_layers: tuple[int, ...] | None = None
    checkpoint_fractions: tuple[float, ...] | None = None
    blocks_reference: tuple[str, ...] | None = None

    for chunk_index, start in enumerate(range(0, len(encoded), chunk_size)):
        stop = min(start + chunk_size, len(encoded))
        chunk = encoded[start:stop]
        partial_path = partial_dir / f"chunk{chunk_index:04d}.npz"
        chunk_config = {
            **common_config,
            "chunk_index": chunk_index,
            "chunk_start": start,
            "chunk_stop": stop,
            "sentence_ids": [item.sentence.sent_id for item in chunk],
        }
        if partial_path.exists():
            metadata = _load_chunk_metadata(partial_path, chunk_config)
            print(
                f"chunk {chunk_index + 1}/{chunk_count}: skipping fieldrun; "
                f"partial already exists: {partial_path}"
            )
        else:
            print(
                f"chunk {chunk_index + 1}/{chunk_count}: running fieldrun for "
                f"sentences {start + 1}-{stop}"
            )
            metadata = _write_chunk_partial(
                partial_path=partial_path,
                config=chunk_config,
                encoded=chunk,
                tokenizer=tokenizer,
                fieldrun_path=fieldrun_path,
                bundle_path=bundle_path,
                n=n,
                kcand=kcand,
                expected_checkpoint_layers=checkpoint_layers,
                expected_blocks=blocks_reference,
            )
            print(f"chunk {chunk_index + 1}/{chunk_count}: wrote partial: {partial_path}")
        partial_paths.append(partial_path)

        chunk_layers_raw = metadata["checkpoint_layers"]
        chunk_fractions_raw = metadata["checkpoint_fractions"]
        chunk_blocks_raw = metadata["blocks"]
        if chunk_layers_raw is not None:
            chunk_layers = tuple(int(layer) for layer in chunk_layers_raw)
            chunk_fractions = tuple(float(fraction) for fraction in chunk_fractions_raw)
            chunk_blocks = tuple(str(block) for block in chunk_blocks_raw)
            if checkpoint_layers is None:
                checkpoint_layers = chunk_layers
                checkpoint_fractions = chunk_fractions
                blocks_reference = chunk_blocks
            elif chunk_layers != checkpoint_layers or chunk_blocks != blocks_reference:
                raise ValueError(
                    f"{partial_path}: block/checkpoint schema changed across chunk partials"
                )

    sent_ids: list[str] = []
    token_indices: list[int] = []
    word_text: list[str] = []
    last_subword_indices: list[int] = []
    last_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []
    per_sentence: list[dict[str, Any]] = []
    aggregate_coverages: list[CoverageCounts] = []
    sanity_passes = 0
    dropped_sentences: list[dict[str, Any]] = []
    pad_passed = 0
    pad_total = 0
    pad_per_sentence: dict[str, bool] = {}
    fieldrun_seconds = 0.0
    worked_example: dict[str, Any] | None = None

    for partial_path in partial_paths:
        with np.load(partial_path, allow_pickle=False) as partial:
            metadata = json.loads(str(partial["chunk_provenance"].item()))
            sent_ids.extend(str(value) for value in partial["sent_ids"])
            token_indices.extend(int(value) for value in partial["token_index"])
            word_text.extend(str(value) for value in partial["word_text"])
            last_subword_indices.extend(int(value) for value in partial["last_subword_index"])
            if int(metadata["reduced_sentence_count"]) > 0:
                last_parts.append(partial["last_residual"])
                mean_parts.append(partial["mean_residual"])
        coverage = metadata["coverage"]
        aggregate_coverages.append(
            CoverageCounts(
                total_words=int(coverage["total_words"]),
                clean_words=int(coverage["clean_words"]),
                boundary_straddling_subwords=int(coverage["boundary_straddling_subwords"]),
                leading_single_subword_word_gaps=int(
                    coverage["leading_single_subword_word_gaps"]
                ),
                last_word_gaps=int(coverage["last_word_gaps"]),
            )
        )
        per_sentence.extend(metadata["per_sentence"])
        dropped_sentences.extend(metadata["dropped_sentences"])
        sanity_passes += int(metadata["sanity_passed_sentences"])
        pad_safety = metadata["pad_safety"]
        pad_passed += int(pad_safety["passed_sentences"])
        pad_total += int(pad_safety["total_sentences"])
        pad_per_sentence.update(
            {str(sid): bool(passed) for sid, passed in pad_safety["per_sentence"].items()}
        )
        fieldrun_seconds += float(metadata["fieldrun_subprocess_seconds"])
        if worked_example is None and metadata["worked_example"] is not None:
            worked_example = metadata["worked_example"]

    if checkpoint_layers is None or checkpoint_fractions is None or not last_parts:
        raise RuntimeError("no sanity-valid sentences were available for dataset construction")
    aggregate = _sum_coverage(aggregate_coverages)
    provenance: dict[str, Any] = {
        "model_id": bundle_path.name,
        "model_tag": model_tag,
        "bundle_path": str(bundle_path),
        "fieldrun_path": str(fieldrun_path),
        "split": split,
        "limit": limit,
        "n": n,
        "kcand": kcand,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "checkpoint_layers": list(checkpoint_layers),
        "checkpoint_fractions": list(checkpoint_fractions),
        "pooling_arms": list(POOLING_ARMS),
        "pad_mitigation": {
            "used": True,
            "pad": pad,
            "safety_passed_sentences": pad_passed,
            "safety_total_sentences": pad_total,
            "per_sentence": pad_per_sentence,
        },
        "sanity_gate": {
            "passed_sentences": sanity_passes,
            "total_sentences": len(encoded),
            "pass_rate": sanity_passes / len(encoded),
        },
        "coverage": {**asdict(aggregate), "fraction": aggregate.fraction},
        "per_sentence": per_sentence,
        "dropped_sentences": dropped_sentences,
        "fieldrun_subprocess_seconds": fieldrun_seconds,
        "fieldrun_seconds_per_sentence": fieldrun_seconds / len(encoded),
        "worked_example": worked_example,
    }

    output_path = output_root / f"{model_tag}_gsd_{split}_n{limit}.npz"
    np.savez_compressed(
        output_path,
        sent_ids=np.asarray(sent_ids, dtype=np.str_),
        token_index=np.asarray(token_indices, dtype=np.int64),
        word_text=np.asarray(word_text, dtype=np.str_),
        last_subword_index=np.asarray(last_subword_indices, dtype=np.int64),
        checkpoint_layers=np.asarray(checkpoint_layers, dtype=np.int64),
        checkpoint_fractions=np.asarray(checkpoint_fractions, dtype=np.float32),
        last_residual=np.concatenate(last_parts, axis=0),
        mean_residual=np.concatenate(mean_parts, axis=0),
        provenance=np.asarray(json.dumps(provenance, ensure_ascii=False, sort_keys=True)),
    )
    if not keep_chunk_partials:
        shutil.rmtree(partial_dir)
    return output_path, provenance


def print_coverage_report(output_path: Path, provenance: dict[str, Any]) -> None:
    print(f"output: {output_path}")
    print(
        f"fieldrun: {provenance['fieldrun_subprocess_seconds']:.3f}s total, "
        f"{provenance['fieldrun_seconds_per_sentence']:.3f}s/sentence"
    )
    for row in provenance["per_sentence"]:
        if row["dropped"]:
            print(f"{row['sent_id']}: DROPPED ({row['drop_reason']})")
        else:
            coverage = row["coverage"]
            print(
                f"{row['sent_id']}: {coverage['clean_words']}/{coverage['total_words']} "
                f"clean ({coverage['fraction']:.3%})"
            )
    coverage = provenance["coverage"]
    sanity = provenance["sanity_gate"]
    padding = provenance["pad_mitigation"]
    print(
        "aggregate coverage: "
        f"{coverage['clean_words']}/{coverage['total_words']} ({coverage['fraction']:.3%}); "
        f"boundary/unaligned={coverage['boundary_straddling_subwords']}, "
        f"leading-single-subword-gaps={coverage['leading_single_subword_word_gaps']}, "
        f"last-word-gaps={coverage['last_word_gaps']}"
    )
    print(
        f"pad safety: {padding['safety_passed_sentences']}/{padding['safety_total_sentences']}; "
        f"sanity gate: {sanity['passed_sentences']}/{sanity['total_sentences']} "
        f"({sanity['pass_rate']:.3%})"
    )
    if provenance["worked_example"] is not None:
        print("worked example: " + json.dumps(provenance["worked_example"], ensure_ascii=False))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tag", default="qwen05b")
    parser.add_argument("--bundle-stem", default="Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--kcand", type=int, default=24)
    parser.add_argument("--pad", default=" x")
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument(
        "--keep-chunk-partials", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fieldrun", type=Path, default=FIELDRun)
    parser.add_argument("--data-root", type=Path, default=GSD_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path, provenance = build_dataset(
        model_tag=args.model_tag,
        bundle_stem=args.bundle_stem,
        split=args.split,
        limit=args.limit,
        fieldrun_path=args.fieldrun,
        data_root=args.data_root,
        output_root=args.output_root,
        kcand=args.kcand,
        pad=args.pad,
        chunk_size=args.chunk_size,
        keep_chunk_partials=args.keep_chunk_partials,
    )
    print_coverage_report(output_path, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
