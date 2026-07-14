"""Windowed count-table deprel labeler campaign (feature-ladder rung 1).

Gives the shipped UNARY count-table richer keys (POS/shape windows around the
token and its governor) and measures it serve-honest against the plain unary
table on the SAME L0 predicted heads.  Pure count-table — no gradient, no
seeds, no neural net.

Pre-registered decision rules from ``docs/notes/windowed_labeler_prereg.md``
§5 are applied verbatim.

Independent of the parallel codex lane; do not import ``windowed_labeler_codex``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.attach_levers_codex import (  # noqa: E402
    predict_deprels,
    score_case_bearing_deprels,
    score_prediction_sequences,
)
from experiments.german_r1_codex import (  # noqa: E402
    DATA_ROOT,
    CountTable,
    GermanR1Student,
    RegisterLayer,
    Sentence,
    accuracy,
    fit_count_table,
    load_split,
)
from experiments.german_r3_codex import (  # noqa: E402
    CampaignTestReadGuard,
    GermanR3DependencyStudent,
    _predicted_pos_by_sentence,
    _punctuation,
    _relation_key_levels,
    choose_window,
    load_head_deprel_file,
    load_head_test_once,
    load_shared_test_once,
    resolve_governor_index,
    run_predicted_case_sentence,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    aligned_head_record,
    head_deprel_by_sent,
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "campaign_windowed_labeler.json"
RUN_TAG = "empirical"
SCOPE = "windowed count-table deprel labeler (feature-ladder rung 1)"

# DEV-only (M, T) grid.  Larger M fragments the key space combinatorially;
# {1,2,3} covers local ±1 through ±3 neighbor context without an explosive
# OOV rate on GSD-scale train.
M_GRID: tuple[int, ...] = (1, 2, 3)
T_GRID: tuple[int, ...] = (5, 10, 20, 50)
TIE_BREAK_NOTE = (
    "Select highest DEV deprel-only accuracy; ties prefer larger T "
    "(more conservative / reliable), then smaller M (simpler)."
)
M_GRID_JUSTIFICATION = (
    "M ∈ {1,2,3}: local context radii only.  Larger radii explode the "
    "(pos, shape) window key space and drive windowed support below any "
    "reasonable T; the unary chain still covers residual tokens."
)

CASE_BASELINE = 0.76
CASE_DELIVERABLE = 0.90
LAS_FIRES_THRESHOLD = 0.03

PAIRWISE_SENSITIVE = frozenset({"nsubj", "obj", "iobj", "obl", "nmod", "conj"})
LOCAL_DEPRELS = frozenset({"det", "amod", "case", "punct", "aux", "cop"})

# Shipped unary key hierarchy width (must match _relation_key_levels).
UNARY_LEVEL_COUNT = 6


def _coarsen(deprel: str) -> str:
    """Strip a UD subtype after the first colon (``nsubj:pass`` → ``nsubj``)."""
    return deprel.split(":", 1)[0]


def _token_shape(token: str) -> tuple[bool, bool, bool]:
    """Coarse surface shape: (istitle, isupper, is_punctuation).  Never raw form."""
    return (token.istitle(), token.isupper(), _punctuation(token))


def _pos_shape_or_boundary(
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    index: int,
) -> Any:
    """One window cell: (pos, shape) or ``<BOS>``/``<EOS>`` at sentence edges."""
    if index < 0:
        return "<BOS>"
    if index >= len(tokens):
        return "<EOS>"
    return (predicted_pos[index], _token_shape(tokens[index]))


def _structure_tuple(
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    index: int,
    predicted_offset: int,
) -> tuple[str, int | str, str]:
    """Match ``_relation_key_levels`` direction / distance_bucket / governor_pos."""
    governor = resolve_governor_index(index, predicted_offset, len(tokens))
    if predicted_offset == 0:
        direction = "SELF"
    elif predicted_offset < 0:
        direction = "LEFT"
    else:
        direction = "RIGHT"
    distance = abs(predicted_offset)
    distance_bucket: int | str
    if distance <= 2:
        distance_bucket = distance
    elif distance <= 5:
        distance_bucket = "3-5"
    else:
        distance_bucket = "6+"
    governor_pos = predicted_pos[governor] if governor is not None else "<INVALID>"
    return (direction, distance_bucket, governor_pos)


def _governor_window(
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    index: int,
    predicted_offset: int,
) -> tuple[Any, Any, Any]:
    """Fixed ±1 window around the governor (or ROOT/INVALID markers).

    Spec treats ROOT / invalid as g=None for the *window* (not for structure,
    which follows ``resolve_governor_index`` / unary convention).  Dependent-side
    radius shrinks across levels; governor window stays ±1 at every level.
    """
    if predicted_offset == 0:
        return ("<BOS>", "<ROOT>", "<EOS>")
    governor = resolve_governor_index(index, predicted_offset, len(tokens))
    if governor is None:
        return ("<BOS>", "<INVALID>", "<EOS>")
    return (
        _pos_shape_or_boundary(tokens, predicted_pos, governor - 1),
        _pos_shape_or_boundary(tokens, predicted_pos, governor),
        _pos_shape_or_boundary(tokens, predicted_pos, governor + 1),
    )


def _dependent_window(
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    index: int,
    radius: int,
) -> tuple[Any, ...]:
    """POS+shape for index±radius excluding the dependent itself (left→right)."""
    if radius < 1:
        raise ValueError("dependent window radius must be >= 1")
    cells: list[Any] = []
    for neighbor in range(index - radius, index + radius + 1):
        if neighbor == index:
            continue
        cells.append(_pos_shape_or_boundary(tokens, predicted_pos, neighbor))
    return tuple(cells)


def windowed_level_key(
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    index: int,
    predicted_offset: int,
    radius: int,
) -> tuple[Any, ...]:
    """Single windowed key at dependent-window radius ``radius`` (level m)."""
    if len(tokens) != len(predicted_pos):
        raise ValueError("tokens and predicted POS must align")
    if not (0 <= index < len(tokens)):
        raise IndexError(f"token index outside sentence: {index}")
    own_pos = predicted_pos[index]
    dep_window = _dependent_window(tokens, predicted_pos, index, radius)
    gov_window = _governor_window(tokens, predicted_pos, index, predicted_offset)
    structure = _structure_tuple(tokens, predicted_pos, index, predicted_offset)
    return (own_pos, dep_window, gov_window, structure)


def full_windowed_keys(
    tokens: Sequence[str],
    predicted_pos: Sequence[str],
    index: int,
    predicted_offset: int,
    hierarchy_radius: int,
) -> tuple[Any, ...]:
    """Full backoff key chain: level(M)..level(1) then shipped unary levels.

    ``full_keys[m]`` for ``m < M`` are windowed; ``full_keys[M:]`` are exactly
    the 6 levels from ``_relation_key_levels`` (same objects, same order).
    Keys are always built for token ``index`` in this call — never via a
    sliced offsets array re-enumerated from 0 (bug-1 guardrail).
    """
    if hierarchy_radius < 1:
        raise ValueError("hierarchy_radius M must be >= 1")
    windowed = tuple(
        windowed_level_key(tokens, predicted_pos, index, predicted_offset, radius)
        for radius in range(hierarchy_radius, 0, -1)
    )
    unary = _relation_key_levels(tokens, predicted_pos, index, predicted_offset)
    if len(unary) != UNARY_LEVEL_COUNT:
        raise RuntimeError(
            f"expected {UNARY_LEVEL_COUNT} unary levels, got {len(unary)}"
        )
    return (*windowed, *unary)


def _mode(labels: Sequence[str]) -> str:
    counts = Counter(labels)
    if not counts:
        raise ValueError("cannot take mode of empty labels")
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


@dataclass
class WindowedBackoffTables:
    """First-present level wins; exposes which level served each prediction.

    Windowed levels (indices ``0 .. M-1``) were fit with ``minsupp=T``;
    unary levels (``M .. M+5``) with ``minsupp=1``.  Fallback is global mode.
    """

    tables: tuple[CountTable, ...]
    fallback: str
    window_level_count: int  # M

    @property
    def M(self) -> int:
        return self.window_level_count

    def predict_with_source(self, keys: Sequence[Any]) -> tuple[str, int | None]:
        if len(keys) != len(self.tables):
            raise ValueError("backoff key/table count mismatch")
        for level_index, (table, key) in enumerate(zip(self.tables, keys, strict=True)):
            entry = table.lookup(key)
            if entry is not None:
                return entry.label, level_index
        return self.fallback, None

    def predict(self, keys: Sequence[Any]) -> str:
        label, _source = self.predict_with_source(keys)
        return label


def fit_windowed_backoff(
    rows: Sequence[tuple[tuple[Any, ...], str]],
    *,
    hierarchy_radius: int,
    minsupp_windowed: int,
) -> WindowedBackoffTables:
    """Fit per-level count tables with differential minsupp (bug-2 guardrail).

    Windowed levels use ``minsupp=T``; trailing unary levels use ``minsupp=1``
    so a fully-OOV-for-windows token gets the same unary fallback chain as the
    plain unary arm.  Do **not** use shipped ``_fit_backoff`` (it forces
    minsupp=1 on every level — the over-trust bug).
    """
    if not rows:
        raise ValueError("cannot fit empty windowed backoff tables")
    if hierarchy_radius < 1:
        raise ValueError("hierarchy_radius must be >= 1")
    if minsupp_windowed < 1:
        raise ValueError("minsupp_windowed T must be >= 1")
    width = len(rows[0][0])
    expected = hierarchy_radius + UNARY_LEVEL_COUNT
    if width != expected:
        raise ValueError(f"key width {width} != M+unary = {expected}")
    if any(len(keys) != width for keys, _label in rows):
        raise ValueError("inconsistent backoff key width")

    tables: list[CountTable] = []
    for level in range(width):
        minsupp = minsupp_windowed if level < hierarchy_radius else 1
        tables.append(
            fit_count_table(
                ((keys[level], label) for keys, label in rows),
                minsupp=minsupp,
                mindet=0.0,
            )
        )
    fallback = _mode([label for _keys, label in rows])
    return WindowedBackoffTables(tuple(tables), fallback, hierarchy_radius)


class WindowedDeprelLabeler:
    """Windowed count-table deprel labeler with (M, T) hierarchy.

    Train governors are the L0 student's own predicted heads (never gold).
    Predict takes the FULL head_offsets sequence and indexes per position
    (matching ``predict_deprels`` — bug-1 guardrail).
    """

    def __init__(self, hierarchy_radius: int = 1, minsupp_windowed: int = 5) -> None:
        if hierarchy_radius < 1:
            raise ValueError("hierarchy_radius M must be >= 1")
        if minsupp_windowed < 1:
            raise ValueError("minsupp_windowed T must be >= 1")
        self.hierarchy_radius = hierarchy_radius
        self.minsupp_windowed = minsupp_windowed
        self.tables: WindowedBackoffTables | None = None

    def fit(
        self,
        train: Sequence[Sentence],
        heads: Mapping[str, HeadDeprelRecord],
        predicted_pos: Mapping[str, Sequence[str]],
        l0_student: GermanR3DependencyStudent,
    ) -> None:
        """Fit on gold deprel labels with L0-predicted train head offsets."""
        rows: list[tuple[tuple[Any, ...], str]] = []
        for sentence in train:
            head = aligned_head_record(sentence, heads)
            if not isinstance(head, HeadDeprelRecord):
                raise TypeError("labeled dependency record required")
            pos = predicted_pos[sentence.sent_id]
            # Train-time governors = student's OWN predicted heads, never gold.
            predicted_offsets = l0_student.predict_heads(sentence.tokens, pos)
            for index, label in enumerate(head.deprel):
                keys = full_windowed_keys(
                    sentence.tokens,
                    pos,
                    index,
                    predicted_offsets[index],
                    self.hierarchy_radius,
                )
                rows.append((keys, label))
        self.tables = fit_windowed_backoff(
            rows,
            hierarchy_radius=self.hierarchy_radius,
            minsupp_windowed=self.minsupp_windowed,
        )

    def fit_from_rows(self, rows: Sequence[tuple[tuple[Any, ...], str]]) -> None:
        """Fit directly from (full_keys, label) rows (tests / diagnostics)."""
        self.tables = fit_windowed_backoff(
            rows,
            hierarchy_radius=self.hierarchy_radius,
            minsupp_windowed=self.minsupp_windowed,
        )

    def _require_tables(self) -> WindowedBackoffTables:
        if self.tables is None:
            raise RuntimeError("windowed deprel table used before fit")
        return self.tables

    def predict_with_source(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[tuple[str, ...], tuple[int | None, ...]]:
        """Predict deprels and the level index (or None=fallback) per token.

        ``head_offsets`` is the FULL sequence; each position builds keys for
        ITS OWN index via ``enumerate`` — never a length-1 slice.
        """
        tables = self._require_tables()
        if not (len(tokens) == len(predicted_pos) == len(head_offsets)):
            raise ValueError("tokens, predicted POS, and head offsets must align")
        labels: list[str] = []
        sources: list[int | None] = []
        for index, offset in enumerate(head_offsets):
            keys = full_windowed_keys(
                tokens, predicted_pos, index, int(offset), self.hierarchy_radius
            )
            label, source = tables.predict_with_source(keys)
            labels.append(label)
            sources.append(source)
        return tuple(labels), tuple(sources)

    def predict(
        self,
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[str, ...]:
        """Predict deprels for the full offsets sequence (``predict_deprels`` pattern)."""
        labels, _sources = self.predict_with_source(tokens, predicted_pos, head_offsets)
        return labels


def partition_deprel_accuracy(
    predicted_deprel: Sequence[str],
    gold_deprel: Sequence[str],
) -> dict[str, float | int]:
    """Deprel-only accuracy on pairwise-sensitive vs local gold coarse partitions."""
    if len(predicted_deprel) != len(gold_deprel):
        raise ValueError("predicted and gold deprels must align")
    pairwise_hits = pairwise_n = 0
    local_hits = local_n = 0
    for predicted, gold in zip(predicted_deprel, gold_deprel, strict=True):
        coarse = _coarsen(gold)
        if coarse in PAIRWISE_SENSITIVE:
            pairwise_n += 1
            pairwise_hits += int(predicted == gold)
        elif coarse in LOCAL_DEPRELS:
            local_n += 1
            local_hits += int(predicted == gold)
    return {
        "pairwise_sensitive_accuracy": (
            pairwise_hits / pairwise_n if pairwise_n else float("nan")
        ),
        "pairwise_sensitive_n": pairwise_n,
        "local_accuracy": local_hits / local_n if local_n else float("nan"),
        "local_n": local_n,
    }


def coarse_las(
    predicted_head_offset: Sequence[int],
    predicted_deprel: Sequence[str],
    gold_head_offset: Sequence[int],
    gold_deprel: Sequence[str],
) -> float:
    """LAS with deprels compared after stripping UD subtypes."""
    size = len(gold_head_offset)
    if not size:
        raise ValueError("cannot score empty sequences")
    if not (
        len(predicted_head_offset)
        == len(predicted_deprel)
        == size
        == len(gold_deprel)
    ):
        raise ValueError("prediction/gold lengths differ")
    correct = 0
    for index in range(size):
        if predicted_head_offset[index] != gold_head_offset[index]:
            continue
        if _coarsen(predicted_deprel[index]) == _coarsen(gold_deprel[index]):
            correct += 1
    return correct / size


def score_arm_sequences(
    predicted_head_offset: Sequence[int],
    predicted_deprel: Sequence[str],
    gold_head_offset: Sequence[int],
    gold_deprel: Sequence[str],
    predicted_case: Sequence[str],
    gold_case: Sequence[str],
) -> dict[str, float | int]:
    """Full metric bundle for one labeler arm on concatenated sequences."""
    base = score_prediction_sequences(
        predicted_head_offset,
        predicted_deprel,
        gold_head_offset,
        gold_deprel,
    )
    case_bearing_accuracy, case_bearing_n = score_case_bearing_deprels(
        predicted_deprel, gold_deprel
    )
    partitions = partition_deprel_accuracy(predicted_deprel, gold_deprel)
    return {
        "uas": base["uas"],
        "las": base["las"],
        "las_coarse": coarse_las(
            predicted_head_offset,
            predicted_deprel,
            gold_head_offset,
            gold_deprel,
        ),
        "deprel_only_accuracy": base["deprel_only_accuracy"],
        "case_bearing_deprel_accuracy": case_bearing_accuracy,
        "case_bearing_deprel_n": case_bearing_n,
        "serve_honest_morph_case": accuracy(predicted_case, gold_case),
        "tokens": base["tokens"],
        **partitions,
    }


def evaluate_labeler_arm(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    common_heads_by_sent: Mapping[str, Sequence[int]],
    deprel_fn,
    case_student: GermanR1Student,
    *,
    serve_honest: bool = True,
) -> dict[str, float | int]:
    """Run one labeler arm on supplied heads through the case cascade."""
    pred_head: list[int] = []
    pred_deprel: list[str] = []
    pred_case: list[str] = []
    gold_head: list[int] = []
    gold_deprel: list[str] = []
    gold_case: list[str] = []

    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        offsets = tuple(common_heads_by_sent[sentence.sent_id])
        deprels = tuple(deprel_fn(sentence.tokens, pos, offsets))
        case_baseline = case_student.predict_sentence(sentence)["full"]["morph_case"]
        eligible = [
            index
            for index, label in enumerate(sentence.targets["morph_case"])
            if label != "-"
        ]
        case_result = run_predicted_case_sentence(
            case_student,
            sentence.tokens,
            case_baseline,
            pos,
            offsets,
            deprels,
            eligible,
        )
        pred_head.extend(offsets)
        pred_deprel.extend(deprels)
        pred_case.extend(case_result.predictions)
        gold_head.extend(head.head_offset)
        gold_deprel.extend(head.deprel)
        gold_case.extend(sentence.targets["morph_case"])

    metrics = score_arm_sequences(
        pred_head, pred_deprel, gold_head, gold_deprel, pred_case, gold_case
    )
    metrics["serve_honest"] = serve_honest
    return metrics


def deprel_only_accuracy_on_split(
    sentences: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    head_offsets_by_sent: Mapping[str, Sequence[int]],
    deprel_fn,
) -> float:
    """Deprel-only accuracy on an already-loaded split (DEV sweep helper)."""
    predicted: list[str] = []
    gold: list[str] = []
    for sentence in sentences:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        offsets = tuple(head_offsets_by_sent[sentence.sent_id])
        predicted.extend(deprel_fn(sentence.tokens, pos, offsets))
        gold.extend(head.deprel)
    return accuracy(predicted, gold)


def select_mt_on_dev(
    train: Sequence[Sentence],
    train_heads: Mapping[str, HeadDeprelRecord],
    train_pos: Mapping[str, Sequence[str]],
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    *,
    m_grid: Sequence[int] = M_GRID,
    t_grid: Sequence[int] = T_GRID,
) -> tuple[int, int, list[dict[str, Any]], float]:
    """Sweep (M, T) on DEV only; never accepts or reads TEST.

    Returns ``(best_M, best_T, curve_rows, best_dev_accuracy)``.
    Tie-break: higher DEV accuracy, then larger T, then smaller M.
    """
    # Signature isolation: this function has no test_* parameter.
    sig = inspect.signature(select_mt_on_dev)
    for name in sig.parameters:
        if "test" in name.lower():
            raise RuntimeError(f"DEV sweep must not accept test inputs: {name}")

    print(
        f"DEV SWEEP: fitting windowed tables over M={list(m_grid)} T={list(t_grid)}.",
        flush=True,
    )
    dev_predicted_heads = {
        sentence.sent_id: l0_student.predict_heads(
            sentence.tokens, dev_pos[sentence.sent_id]
        )
        for sentence in dev
    }

    curve: list[dict[str, Any]] = []
    best: tuple[float, int, int] | None = None  # (acc, T, -M) for tie-break

    for m in m_grid:
        for t in t_grid:
            labeler = WindowedDeprelLabeler(hierarchy_radius=m, minsupp_windowed=t)
            labeler.fit(train, train_heads, train_pos, l0_student)

            def deprel_fn(
                tokens: Sequence[str],
                predicted_pos: Sequence[str],
                head_offsets: Sequence[int],
                _labeler: WindowedDeprelLabeler = labeler,
            ) -> tuple[str, ...]:
                return _labeler.predict(tokens, predicted_pos, head_offsets)

            dev_acc = deprel_only_accuracy_on_split(
                dev, dev_heads, dev_pos, dev_predicted_heads, deprel_fn
            )
            row = {"M": m, "T": t, "dev_deprel_only_accuracy": dev_acc}
            curve.append(row)
            print(
                f"  DEV (M={m}, T={t}): deprel-only={dev_acc:.6f}",
                flush=True,
            )
            # Prefer higher acc; then larger T; then smaller M (via -M).
            candidate = (dev_acc, t, -m)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        raise RuntimeError("empty (M, T) grid")
    best_acc, best_t, neg_m = best
    best_m = -neg_m
    return best_m, best_t, curve, best_acc


# ---------------------------------------------------------------------------
# Diagnostics (required; would have caught the first-pass bugs)
# ---------------------------------------------------------------------------


def diagnostic_windowed_hit_rate(
    level_indices: Sequence[int | None],
    hierarchy_radius: int,
) -> dict[str, float | int]:
    """Fraction of tokens served by a windowed level (index < M) vs unary/fallback."""
    if hierarchy_radius < 1:
        raise ValueError("hierarchy_radius must be >= 1")
    total = len(level_indices)
    if total == 0:
        return {
            "tokens": 0,
            "windowed_hits": 0,
            "unary_or_fallback": 0,
            "windowed_hit_rate": float("nan"),
            "unary_or_fallback_rate": float("nan"),
        }
    windowed_hits = sum(
        1 for src in level_indices if src is not None and src < hierarchy_radius
    )
    unary_or_fallback = total - windowed_hits
    return {
        "tokens": total,
        "windowed_hits": windowed_hits,
        "unary_or_fallback": unary_or_fallback,
        "windowed_hit_rate": windowed_hits / total,
        "unary_or_fallback_rate": unary_or_fallback / total,
    }


def diagnostic_dev_accuracy_vs_t_curve(
    sweep_curve: Sequence[Mapping[str, Any]],
    m_grid: Sequence[int] = M_GRID,
    t_grid: Sequence[int] = T_GRID,
    unary_dev_baseline: float | None = None,
) -> dict[str, Any]:
    """Per-M DEV accuracy at each T; flag non-convergence toward unary as T rises."""
    by_m: dict[str, list[dict[str, Any]]] = {}
    suspect_notes: list[str] = []
    for m in m_grid:
        series = [
            {
                "T": int(row["T"]),
                "dev_deprel_only_accuracy": float(row["dev_deprel_only_accuracy"]),
            }
            for row in sweep_curve
            if int(row["M"]) == m
        ]
        series.sort(key=lambda item: item["T"])
        by_m[str(m)] = series
        if len(series) >= 2:
            first_acc = series[0]["dev_deprel_only_accuracy"]
            last_acc = series[-1]["dev_deprel_only_accuracy"]
            # A correct table should not systematically *fall* as T rises
            # (more backoff → converge toward unary).  Mild non-monotonicity
            # is fine; a clear fall from smallest T to largest is suspect.
            if last_acc + 1e-12 < first_acc - 0.01:
                suspect_notes.append(
                    f"M={m}: DEV accuracy FALLS as T rises "
                    f"({first_acc:.4f} @ T={series[0]['T']} → "
                    f"{last_acc:.4f} @ T={series[-1]['T']}); "
                    "treat run as suspect (broken backoff often fails to "
                    "converge toward unary)."
                )
    text_parts = [
        "DEV deprel-only accuracy vs T, per M.  "
        "Correct tables trend toward the unary DEV baseline as T rises."
    ]
    if unary_dev_baseline is not None:
        text_parts.append(f"Unary DEV deprel-only baseline ≈ {unary_dev_baseline:.4f}.")
    if suspect_notes:
        text_parts.append("SUSPECT: " + " ".join(suspect_notes))
    else:
        text_parts.append(
            "No large accuracy drop as T rises (curve not flagged suspect)."
        )
    return {
        "by_M": by_m,
        "t_grid": list(t_grid),
        "m_grid": list(m_grid),
        "unary_dev_baseline": unary_dev_baseline,
        "suspect": bool(suspect_notes),
        "suspect_notes": suspect_notes,
        "text": " ".join(text_parts),
    }


def diagnostic_windowed_fires_vs_unary(
    windowed_labels: Sequence[str],
    unary_labels: Sequence[str],
    gold_labels: Sequence[str],
    level_indices: Sequence[int | None],
    hierarchy_radius: int,
) -> dict[str, float | int]:
    """On subset S of tokens served by a windowed level: windowed vs unary acc."""
    if not (
        len(windowed_labels)
        == len(unary_labels)
        == len(gold_labels)
        == len(level_indices)
    ):
        raise ValueError("diagnostic sequences must align")
    subset_windowed: list[str] = []
    subset_unary: list[str] = []
    subset_gold: list[str] = []
    for w_lab, u_lab, g_lab, src in zip(
        windowed_labels, unary_labels, gold_labels, level_indices, strict=True
    ):
        if src is not None and src < hierarchy_radius:
            subset_windowed.append(w_lab)
            subset_unary.append(u_lab)
            subset_gold.append(g_lab)
    n = len(subset_gold)
    if n == 0:
        return {
            "subset_n": 0,
            "windowed_accuracy_on_S": float("nan"),
            "unary_accuracy_on_S": float("nan"),
        }
    return {
        "subset_n": n,
        "windowed_accuracy_on_S": accuracy(subset_windowed, subset_gold),
        "unary_accuracy_on_S": accuracy(subset_unary, subset_gold),
    }


def apply_preregistered_read(
    unary_las: float,
    windowed_las: float,
    unary_case: float,
    windowed_case: float,
    partition_gains: Mapping[str, float],
    *,
    dev_M: int,
    dev_T: int,
) -> dict[str, Any]:
    """Apply ``windowed_labeler_prereg.md`` §5 decision rules verbatim.

    FIRES: windowed − unary serve-honest deprel-LAS (strict) gain ≥ +0.03
    absolute, AND the gain on the pairwise-sensitive partition is STRICTLY
    GREATER than the gain on the local partition.

    IN-BETWEEN: gain > 0 but < +0.03, OR diffuse (pairwise gain ≤ local gain),
    OR no case-cascade improvement (windowed serve-honest case ≤ unary case).

    HALTED: windowed ≤ unary (LAS gain ≤ 0).
    """
    las_gain = float(windowed_las) - float(unary_las)
    pairwise_gain = float(partition_gains["pairwise_sensitive"])
    local_gain = float(partition_gains["local"])
    partition_concentrated = pairwise_gain > local_gain
    case_improves = float(windowed_case) > float(unary_case)

    if las_gain <= 0.0:
        verdict = "HALTED"
        text = (
            "HALTED: windowed−unary serve-honest deprel-LAS (strict) gain "
            f"({las_gain:+.4f}) ≤ 0 — raw-feature windows do not capture the "
            "needed context; escalate to rung 2 (grounded features)."
        )
    elif (
        las_gain >= LAS_FIRES_THRESHOLD
        and partition_concentrated
        and case_improves
    ):
        verdict = "FIRES"
        text = (
            "FIRES: windowed−unary serve-honest deprel-LAS (strict) gain "
            f"{las_gain:+.4f} ≥ +{LAS_FIRES_THRESHOLD:.2f} absolute; "
            f"pairwise-sensitive partition gain ({pairwise_gain:+.4f}) > "
            f"local gain ({local_gain:+.4f}); case-cascade improves "
            f"(windowed {windowed_case:.4f} > unary {unary_case:.4f})."
        )
    else:
        verdict = "IN-BETWEEN"
        reasons: list[str] = []
        if 0.0 < las_gain < LAS_FIRES_THRESHOLD:
            reasons.append(
                f"gain {las_gain:+.4f} > 0 but < +{LAS_FIRES_THRESHOLD:.2f}"
            )
        if not case_improves:
            reasons.append(
                "no case-cascade improvement "
                f"(windowed case {windowed_case:.4f} ≤ unary {unary_case:.4f})"
            )
        if not partition_concentrated:
            reasons.append(
                "partition gain diffuse "
                f"(pairwise-sensitive {pairwise_gain:+.4f} ≤ local {local_gain:+.4f})"
            )
        if las_gain >= LAS_FIRES_THRESHOLD and not reasons:
            reasons.append("FIRES sub-conditions incomplete")
        text = "IN-BETWEEN: " + "; ".join(reasons) + "."

    return {
        "verdict": verdict,
        "deprel_las_gain": las_gain,
        "partition_gains": {
            "pairwise_sensitive": pairwise_gain,
            "local": local_gain,
        },
        "case_unary": float(unary_case),
        "case_windowed": float(windowed_case),
        "dev_M": int(dev_M),
        "dev_T": int(dev_T),
        "text": text,
        "unary_las": float(unary_las),
        "windowed_las": float(windowed_las),
    }


def apply_through_line_case_read(case_accuracy: float) -> dict[str, Any]:
    """Through-line serve-honest case read (plateau language; never irreducible)."""
    if case_accuracy >= CASE_DELIVERABLE:
        verdict = "deliverable_met"
        text = (
            f"THROUGH-LINE: serve-honest case {case_accuracy:.4f} ≥ {CASE_DELIVERABLE:.2f} "
            "— deliverable met (empirical; certify next)."
        )
    elif case_accuracy > CASE_BASELINE:
        gap_closed = case_accuracy - CASE_BASELINE
        gap_total = CASE_DELIVERABLE - CASE_BASELINE
        fraction = gap_closed / gap_total if gap_total > 0 else 0.0
        verdict = "plateau"
        text = (
            f"THROUGH-LINE: serve-honest case {case_accuracy:.4f} < {CASE_DELIVERABLE:.2f} "
            f"but > ~{CASE_BASELINE:.2f} baseline — the recipe closes "
            f"{fraction:.1%} of the ~{CASE_BASELINE:.2f}→{CASE_DELIVERABLE:.2f} gap; "
            "achievability of 0.90 open (plateau language)."
        )
    else:
        verdict = "diagnose"
        text = (
            f"THROUGH-LINE: serve-honest case {case_accuracy:.4f} ≤ ~{CASE_BASELINE:.2f} "
            "baseline — the labeler gain did not survive the cascade; diagnose "
            "(R3 cascade-corruption pattern)."
        )
    return {"verdict": verdict, "case": float(case_accuracy), "text": text}


def _predict_heads_by_sentence(
    student: GermanR3DependencyStudent,
    sentences: Sequence[Sentence],
    predicted_pos: Mapping[str, Sequence[str]],
) -> dict[str, tuple[int, ...]]:
    return {
        sentence.sent_id: student.predict_heads(
            sentence.tokens, predicted_pos[sentence.sent_id]
        )
        for sentence in sentences
    }


def collect_test_predictions(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    common_heads_by_sent: Mapping[str, Sequence[int]],
    windowed: WindowedDeprelLabeler,
    l0_student: GermanR3DependencyStudent,
) -> dict[str, Any]:
    """Gather aligned windowed/unary/gold labels and level sources for diagnostics."""
    windowed_labels: list[str] = []
    unary_labels: list[str] = []
    gold_labels: list[str] = []
    level_indices: list[int | None] = []
    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        offsets = tuple(common_heads_by_sent[sentence.sent_id])
        w_labs, sources = windowed.predict_with_source(sentence.tokens, pos, offsets)
        u_labs = predict_deprels(l0_student, sentence.tokens, pos, offsets)
        windowed_labels.extend(w_labs)
        unary_labels.extend(u_labs)
        gold_labels.extend(head.deprel)
        level_indices.extend(sources)
    return {
        "windowed_labels": windowed_labels,
        "unary_labels": unary_labels,
        "gold_labels": gold_labels,
        "level_indices": level_indices,
    }


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit on train, sweep (M,T) on DEV, read TEST once, matched comparison."""
    hasher = hashlib.sha256()
    print("FIT: loading GSD train shared tasks and head_deprel once.", flush=True)
    train = load_split(DATA_ROOT, "train", hasher)
    train_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "train.jsonl", hasher
    )
    train_heads = head_deprel_by_sent(train_head_records)

    registers = RegisterLayer.from_directory(DATA_ROOT / "registers")
    r1_student = GermanR1Student(registers)
    r1_student.fit(train)
    train_pos = _predicted_pos_by_sentence(r1_student, train)

    print(
        "DEV: loading GSD dev once for L0 window choice AND (M,T) sweep.",
        flush=True,
    )
    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_head_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    dev_heads = head_deprel_by_sent(dev_head_records)
    window, dev_coverage, coverage_curve = choose_window(dev_head_records)
    # Keep dev loaded — needed for (M, T) sweep (no test-read-once guard on dev).

    l0_student = GermanR3DependencyStudent(window)
    l0_student.fit(train, train_heads, train_pos)
    print(f"FIT: L0 dependency student window={window}.", flush=True)

    dev_pos = _predicted_pos_by_sentence(r1_student, dev)

    # Unary DEV baseline for the accuracy-vs-T sanity curve.
    dev_predicted_heads = _predict_heads_by_sentence(l0_student, dev, dev_pos)

    def unary_deprel_dev(
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[str, ...]:
        return predict_deprels(l0_student, tokens, predicted_pos, head_offsets)

    unary_dev_acc = deprel_only_accuracy_on_split(
        dev, dev_heads, dev_pos, dev_predicted_heads, unary_deprel_dev
    )
    print(f"DEV: unary deprel-only baseline={unary_dev_acc:.6f}.", flush=True)

    best_m, best_t, sweep_curve, best_dev_acc = select_mt_on_dev(
        train,
        train_heads,
        train_pos,
        dev,
        dev_heads,
        dev_pos,
        l0_student,
        m_grid=M_GRID,
        t_grid=T_GRID,
    )
    print(
        f"DEV SELECT: M={best_m} T={best_t} dev_deprel_only={best_dev_acc:.6f} "
        f"({TIE_BREAK_NOTE})",
        flush=True,
    )

    # Refit selected (M, T) on TRAIN for the single guarded TEST read.
    windowed = WindowedDeprelLabeler(
        hierarchy_radius=best_m, minsupp_windowed=best_t
    )
    windowed.fit(train, train_heads, train_pos, l0_student)
    print(f"FIT: windowed labeler fixed at M={best_m} T={best_t}.", flush=True)

    guard = CampaignTestReadGuard()
    print(
        "FINAL EVAL: reading shared test tasks once and head_deprel test once.",
        flush=True,
    )
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_head_records = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads = head_deprel_by_sent(test_head_records)
    test_pos = _predicted_pos_by_sentence(r1_student, test)
    test_predicted_heads = _predict_heads_by_sentence(l0_student, test, test_pos)

    def unary_deprel(
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[str, ...]:
        return predict_deprels(l0_student, tokens, predicted_pos, head_offsets)

    def windowed_deprel(
        tokens: Sequence[str],
        predicted_pos: Sequence[str],
        head_offsets: Sequence[int],
    ) -> tuple[str, ...]:
        return windowed.predict(tokens, predicted_pos, head_offsets)

    print("EVAL: unary arm on predicted heads.", flush=True)
    unary_metrics = evaluate_labeler_arm(
        test,
        test_heads,
        test_pos,
        test_predicted_heads,
        unary_deprel,
        r1_student,
        serve_honest=True,
    )

    print("EVAL: windowed arm on the SAME predicted heads.", flush=True)
    windowed_metrics = evaluate_labeler_arm(
        test,
        test_heads,
        test_pos,
        test_predicted_heads,
        windowed_deprel,
        r1_student,
        serve_honest=True,
    )

    if not math.isclose(
        float(windowed_metrics["uas"]),
        float(unary_metrics["uas"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise AssertionError(
            f"UAS mismatch: windowed {windowed_metrics['uas']} vs unary "
            f"{unary_metrics['uas']} (common heads must be bit-identical)"
        )

    partition_gains = {
        "pairwise_sensitive": (
            float(windowed_metrics["pairwise_sensitive_accuracy"])
            - float(unary_metrics["pairwise_sensitive_accuracy"])
        ),
        "local": (
            float(windowed_metrics["local_accuracy"])
            - float(unary_metrics["local_accuracy"])
        ),
    }

    primary_read = apply_preregistered_read(
        float(unary_metrics["las"]),
        float(windowed_metrics["las"]),
        float(unary_metrics["serve_honest_morph_case"]),
        float(windowed_metrics["serve_honest_morph_case"]),
        partition_gains,
        dev_M=best_m,
        dev_T=best_t,
    )
    through_line = apply_through_line_case_read(
        float(windowed_metrics["serve_honest_morph_case"])
    )

    # Required diagnostics on TEST at selected (M, T).
    print("DIAGNOSTICS: hit-rate, DEV-vs-T curve, fires-vs-unary on S.", flush=True)
    collected = collect_test_predictions(
        test,
        test_heads,
        test_pos,
        test_predicted_heads,
        windowed,
        l0_student,
    )
    hit_rate = diagnostic_windowed_hit_rate(
        collected["level_indices"], best_m
    )
    dev_curve = diagnostic_dev_accuracy_vs_t_curve(
        sweep_curve,
        m_grid=M_GRID,
        t_grid=T_GRID,
        unary_dev_baseline=unary_dev_acc,
    )
    fires_vs_unary = diagnostic_windowed_fires_vs_unary(
        collected["windowed_labels"],
        collected["unary_labels"],
        collected["gold_labels"],
        collected["level_indices"],
        best_m,
    )
    diagnostics = {
        "windowed_hit_rate": hit_rate,
        "dev_accuracy_vs_T_per_M": dev_curve,
        "windowed_fires_vs_unary_on_S": fires_vs_unary,
    }

    guard.assert_complete()

    scoreboard: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "scope": SCOPE,
        "measurement": (
            "matched unary vs windowed count-table deprel labeler on shared L0 "
            "predicted heads; pure count-table feature-ladder rung 1"
        ),
        "serve_honest_features": [
            "R1-predicted POS",
            "coarse surface shape (istitle, isupper, punctuation)",
            "dependent POS+shape window ±m (m=M..1)",
            "governor POS+shape window ±1 (fixed across levels)",
            "direction / distance_bucket / governor_pos structure",
            "shipped unary relation key levels (backoff tail)",
        ],
        "gold_usage": (
            "Gold used only as TRAIN fit targets (deprel labels) and the one "
            "guarded TEST scoring read; train governors are L0's own predicted "
            "heads (never gold heads); no gold at predict time; DEV (M,T) sweep "
            "never reads TEST."
        ),
        "data_hash_sha256": hasher.hexdigest(),
        "test_read_counts": dict(guard.counts),
        "test_read_once": True,
        "l0_window": {
            "k": window,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "dev_sweep": {
            "m_grid": list(M_GRID),
            "t_grid": list(T_GRID),
            "m_grid_justification": M_GRID_JUSTIFICATION,
            "tie_break": TIE_BREAK_NOTE,
            "curve": sweep_curve,
            "selected_M": best_m,
            "selected_T": best_t,
            "selected_dev_deprel_only_accuracy": best_dev_acc,
            "unary_dev_deprel_only_accuracy": unary_dev_acc,
        },
        "unary": unary_metrics,
        "windowed": windowed_metrics,
        "partition_gains": partition_gains,
        "las_gain": float(windowed_metrics["las"]) - float(unary_metrics["las"]),
        "deprel_only_gain": (
            float(windowed_metrics["deprel_only_accuracy"])
            - float(unary_metrics["deprel_only_accuracy"])
        ),
        "case_gain": (
            float(windowed_metrics["serve_honest_morph_case"])
            - float(unary_metrics["serve_honest_morph_case"])
        ),
        "diagnostics": diagnostics,
        "read": primary_read,
        "through_line_case": through_line,
        "predict_signature": str(inspect.signature(WindowedDeprelLabeler.predict)),
    }
    return scoreboard, guard


def _fmt_metric_row(name: str, metric: Mapping[str, Any]) -> str:
    return (
        f"{name:<28}  "
        f"UAS={float(metric['uas']):.6f}  "
        f"LAS={float(metric['las']):.6f}  "
        f"LAS_coarse={float(metric['las_coarse']):.6f}  "
        f"deprel={float(metric['deprel_only_accuracy']):.6f}  "
        f"case_bearing={float(metric['case_bearing_deprel_accuracy']):.6f} "
        f"(n={metric['case_bearing_deprel_n']})  "
        f"case={float(metric['serve_honest_morph_case']):.6f}  "
        f"pair={float(metric['pairwise_sensitive_accuracy']):.6f}  "
        f"local={float(metric['local_accuracy']):.6f}"
    )


def print_scoreboard(scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard) -> None:
    """Print the full comparison, diagnostics, and preregistered read."""
    print("\nWINDOWED COUNT-TABLE LABELER — GSD TEST (feature-ladder rung 1)")
    print(f"run_tag: {scoreboard['run_tag']}")
    print(f"scope: {scoreboard['scope']}")
    print(
        "serve_honesty: R1-predicted POS + POS/shape windows only; "
        "no gold heads/deprels as prediction inputs"
    )
    sweep = scoreboard["dev_sweep"]
    print(
        f"DEV-selected (M,T)=({sweep['selected_M']}, {sweep['selected_T']})  "
        f"dev_deprel_only={sweep['selected_dev_deprel_only_accuracy']:.6f}  "
        f"unary_dev={sweep['unary_dev_deprel_only_accuracy']:.6f}"
    )
    print(f"tie-break: {sweep['tie_break']}")
    print(f"M grid justification: {sweep['m_grid_justification']}")

    guard.assert_complete()
    print("TEST READ ONCE: CONFIRMED")

    unary = scoreboard["unary"]
    windowed = scoreboard["windowed"]
    print("\n--- MATCHED COMPARISON (same L0 predicted heads) ---")
    print(_fmt_metric_row("UNARY", unary))
    print(
        _fmt_metric_row("WINDOWED", windowed)
        + f"  LAS_gain={scoreboard['las_gain']:+.6f}  "
        f"deprel_gain={scoreboard['deprel_only_gain']:+.6f}  "
        f"case_gain={scoreboard['case_gain']:+.6f}"
    )

    print("\n--- PARTITION (windowed−unary deprel-only gains) ---")
    gains = scoreboard["partition_gains"]
    print(
        f"pairwise-sensitive {{nsubj,obj,iobj,obl,nmod,conj}}: "
        f"{gains['pairwise_sensitive']:+.6f}"
    )
    print(
        f"local {{det,amod,case,punct,aux,cop}}: "
        f"{gains['local']:+.6f}"
    )

    diag = scoreboard["diagnostics"]
    print("\n--- DIAGNOSTICS ---")
    hr = diag["windowed_hit_rate"]
    print(
        f"windowed hit-rate: {hr['windowed_hit_rate']:.4f} "
        f"({hr['windowed_hits']}/{hr['tokens']} tokens from windowed levels; "
        f"unary/fallback={hr['unary_or_fallback']})"
    )
    curve = diag["dev_accuracy_vs_T_per_M"]
    print(f"DEV accuracy vs T: {curve['text']}")
    for m_key, series in curve["by_M"].items():
        cells = ", ".join(
            f"T={row['T']}:{row['dev_deprel_only_accuracy']:.4f}" for row in series
        )
        print(f"  M={m_key}: {cells}")
    fires = diag["windowed_fires_vs_unary_on_S"]
    print(
        f"on windowed-served subset S (n={fires['subset_n']}): "
        f"windowed_acc={fires['windowed_accuracy_on_S']}  "
        f"unary_acc={fires['unary_accuracy_on_S']}"
    )

    print("\n--- THE READ (prereg §5) ---")
    print(scoreboard["read"]["text"])
    print(scoreboard["through_line_case"]["text"])


def main() -> None:
    scoreboard, guard = build_scoreboard()
    OUTPUT_PATH.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print_scoreboard(scoreboard, guard)
    print(f"structured_output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
