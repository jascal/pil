"""Anisotropy-removal (whitening) gate: TRAIN-fit whitening + re-run #121/#122.

Strips the rogue dimension from grounded φ with a TRAIN-fit whitening transform
(V1 z-score or V2 top-k PC removal), then re-runs BOTH the grounded LABELER
(#121) and the grounded ATTACH scorer (#122) on whitened φ against raw-φ
predecessors and unary/L0/surface-#119 baselines.  DEV-selects whitening
variant × layer per primitive; TEST is read once under ``CampaignTestReadGuard``.

Signed pre-registration: ``docs/notes/whitening_gate_prereg.md`` §5 + PIN A/B.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.attach_levers_codex import (  # noqa: E402
    BilinearAttachmentScorer,
    BilinearConfig,
    chu_liu_edmonds,
    governor_columns_to_offsets,
    l0_edge_scores,
)
from experiments.biaffine_labeler_codex import (  # noqa: E402
    BilinearRelationLabeler,
    RelationLabelerConfig,
)
from experiments.campaign_grounded_attach import (  # noqa: E402
    LAS_FIRES_THRESHOLD as ATTACH_LAS_FIRES_THRESHOLD,
)
from experiments.campaign_grounded_attach import (  # noqa: E402
    UAS_FIRES_THRESHOLD,
    GroundedAttachmentScorer,
    assert_matched_token_counts,
    attach_doubly_covered_fraction,
    collect_covered_dependent_keys,
    covered_dependent_fraction,
    evaluate_conversion_point,
    score_uas_on_keys,
)
from experiments.campaign_grounded_labeler import (  # noqa: E402
    CASE_BASELINE,
    CASE_DELIVERABLE,
    CHECKPOINT_LAYERS,
    FEATURE_DIM,
    LAS_FIRES_THRESHOLD,
    LAYER_INDICES,
    GroundedFeatureProvider,
    GroundedRelationLabeler,
    evaluate_dev_layer,
    evaluate_matched_three_arms,
)
from experiments.german_r1_codex import (  # noqa: E402
    DATA_ROOT,
    GermanR1Student,
    RegisterLayer,
    Sentence,
    load_split,
)
from experiments.german_r3_codex import (  # noqa: E402
    CampaignTestReadGuard,
    GermanR3DependencyStudent,
    _predicted_pos_by_sentence,
    choose_window,
    load_head_deprel_file,
    load_head_test_once,
    load_shared_test_once,
)
from experiments.german_r3min_codex import (  # noqa: E402
    HeadDeprelRecord,
    aligned_head_record,
    head_deprel_by_sent,
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "campaign_whitening_gate.json"
GROUNDED_ROOT = Path(__file__).resolve().parents[1] / "data" / "grounded"
# Richer MWT φ only (not the pre-MWT qwen3b_gsd_* dump).
TRAIN_NPZ = GROUNDED_ROOT / "qwen3b_mwt_gsd_train_n600.npz"
DEV_NPZ = GROUNDED_ROOT / "qwen3b_mwt_gsd_dev_n799.npz"
TEST_NPZ = GROUNDED_ROOT / "qwen3b_mwt_gsd_test_n977.npz"

RUN_TAG = "empirical"
SCOPE = (
    "whitening gate: TRAIN-fit anisotropy removal on MWT grounded φ; "
    "re-run #121 labeler + #122 attach with PIN-B log-prob decode"
)
SEED = 0
PIN_A_SAMPLE_SIZE = 800
PIN_A_SEED = 0
# σ == 0 guard: replace with 1.0 so a constant column is a no-op under z-score.
SIGMA_ZERO_REPLACEMENT = 1.0

VariantName = Literal["v1", "v2_k1", "v2_k2"]
VARIANTS: tuple[VariantName, ...] = ("v1", "v2_k1", "v2_k2")
# Deterministic total order for DEV tie-breaks: prefer V1 over V2, then smallest k,
# then smallest layer_index.  Lower rank is better when primary scores tie.
VARIANT_TIE_RANK: dict[str, int] = {"v1": 0, "v2_k1": 1, "v2_k2": 2}


def _variant_k(variant: str) -> int | None:
    if variant == "v1":
        return None
    if variant == "v2_k1":
        return 1
    if variant == "v2_k2":
        return 2
    raise ValueError(f"unknown whitening variant: {variant}")


def load_layer_matrix(npz_path: Path, layer_index: int) -> np.ndarray:
    """Load all residual rows for one checkpoint layer as float64 (N, FEATURE_DIM)."""
    if layer_index not in LAYER_INDICES:
        raise ValueError(f"layer_index must be in {LAYER_INDICES}, got {layer_index}")
    data = np.load(npz_path, allow_pickle=False)
    checkpoint_layers = tuple(int(value) for value in data["checkpoint_layers"])
    if checkpoint_layers != CHECKPOINT_LAYERS:
        raise ValueError(
            f"unexpected checkpoint_layers={checkpoint_layers}; expected {CHECKPOINT_LAYERS}"
        )
    residuals = np.asarray(data["last_residual"][:, layer_index, :], dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"last_residual layer slice must be (N, {FEATURE_DIM}), got {residuals.shape}"
        )
    return residuals


@dataclass
class WhiteningTransform:
    """TRAIN-fit fixed linear whitening map for one layer-checkpoint.

    V1 — per-dim standardize: ``φ' = (φ − μ) / σ`` with ``σ == 0`` replaced by
    ``SIGMA_ZERO_REPLACEMENT`` (1.0).

    V2 — top-k PC removal then residual z-score: center with TRAIN μ, subtract
    projection onto the top-k TRAIN principal directions (SVD of centered TRAIN),
    then z-score with μ/σ fit on the residual TRAIN matrix after PC removal.
    """

    variant: str
    layer_index: int
    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray | None  # (k, d) or None for V1
    residual_mean: np.ndarray | None
    residual_std: np.ndarray | None

    @classmethod
    def fit(
        cls,
        train_phi: np.ndarray,
        *,
        variant: str,
        layer_index: int,
    ) -> WhiteningTransform:
        if train_phi.ndim != 2 or train_phi.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"train_phi must be (N, {FEATURE_DIM}), got {train_phi.shape}"
            )
        if train_phi.shape[0] < 1:
            raise ValueError("cannot fit whitening on empty TRAIN φ")
        mean = train_phi.mean(axis=0)
        std = train_phi.std(axis=0)
        std = np.where(std == 0.0, SIGMA_ZERO_REPLACEMENT, std)

        k = _variant_k(variant)
        if k is None:
            return cls(
                variant=variant,
                layer_index=layer_index,
                mean=mean,
                std=std,
                components=None,
                residual_mean=None,
                residual_std=None,
            )

        if k < 1:
            raise ValueError(f"V2 k must be ≥ 1, got {k}")
        if train_phi.shape[0] < k:
            raise ValueError(
                f"need at least k={k} TRAIN rows for V2, got {train_phi.shape[0]}"
            )
        centered = train_phi - mean
        # SVD of centered TRAIN; principal directions = top-k right singular vectors.
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        components = np.asarray(vt[:k], dtype=np.float64)
        residual = centered - (centered @ components.T) @ components
        residual_mean = residual.mean(axis=0)
        residual_std = residual.std(axis=0)
        residual_std = np.where(residual_std == 0.0, SIGMA_ZERO_REPLACEMENT, residual_std)
        return cls(
            variant=variant,
            layer_index=layer_index,
            mean=mean,
            std=std,
            components=components,
            residual_mean=residual_mean,
            residual_std=residual_std,
        )

    def transform(self, vector: np.ndarray) -> np.ndarray:
        """Apply the fixed linear map to one feature vector."""
        x = np.asarray(vector, dtype=np.float64)
        if x.shape != (FEATURE_DIM,):
            raise ValueError(f"expected vector shape ({FEATURE_DIM},), got {x.shape}")
        if self.components is None:
            return (x - self.mean) / self.std
        assert self.residual_mean is not None and self.residual_std is not None
        centered = x - self.mean
        residual = centered - (centered @ self.components.T) @ self.components
        return (residual - self.residual_mean) / self.residual_std

    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the map to an (N, d) matrix row-wise."""
        x = np.asarray(matrix, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
            raise ValueError(f"expected (N, {FEATURE_DIM}), got {x.shape}")
        if self.components is None:
            return (x - self.mean) / self.std
        assert self.residual_mean is not None and self.residual_std is not None
        centered = x - self.mean
        residual = centered - (centered @ self.components.T) @ self.components
        return (residual - self.residual_mean) / self.residual_std


class WhitenedFeatureProvider:
    """Drop-in substitute for ``GroundedFeatureProvider`` with whitened ``.get``.

    Preserves ``.covered`` / ``.get`` / ``.feature_dim`` / ``.layer_index`` /
    ``.layer`` so #121/#122 code that expects a grounded provider can consume it
    unchanged.  Whitening params are TRAIN-fit constants; this provider only
    applies them.
    """

    def __init__(
        self,
        base: GroundedFeatureProvider,
        transform: WhiteningTransform,
    ) -> None:
        if base.layer_index != transform.layer_index:
            raise ValueError(
                f"provider layer_index={base.layer_index} != "
                f"transform layer_index={transform.layer_index}"
            )
        self._base = base
        self._transform = transform
        self.feature_dim = base.feature_dim
        self.layer_index = base.layer_index
        self.layer = base.layer
        self.n_rows = base.n_rows
        self.n_sentences = base.n_sentences
        self.variant = transform.variant

    def get(self, sent_id: str, token_index: int) -> np.ndarray | None:
        raw = self._base.get(sent_id, token_index)
        if raw is None:
            return None
        return self._transform.transform(raw)

    def covered(self, sent_id: str, token_index: int) -> bool:
        return self._base.covered(sent_id, token_index)


def pin_a_effective_rank(
    vectors: np.ndarray,
    *,
    sample_size: int = PIN_A_SAMPLE_SIZE,
    seed: int = PIN_A_SEED,
) -> float:
    """PIN A: centered participation-ratio effrank on a FIXED n=800/seed=0 sample.

    ``(Σσ²)² / Σσ⁴`` where σ are singular values of the CENTERED sample.
    If ``n_rows < sample_size``, uses all rows (no subsample).  Independent of
    #122's DEV-sample ``C7_SAMPLE_SIZE=300`` helper.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"vectors must be 2-d, got shape {matrix.shape}")
    n_rows = matrix.shape[0]
    if n_rows == 0:
        return 0.0
    take = min(sample_size, n_rows)
    if take == n_rows:
        sample = matrix
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_rows, size=take, replace=False)
        sample = matrix[indices]
    centered = sample - sample.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values**2
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    return float((total**2) / np.square(eigenvalues).sum())


def pin_a_effective_rank_whitened(
    raw_vectors: np.ndarray,
    transform: WhiteningTransform,
    *,
    sample_size: int = PIN_A_SAMPLE_SIZE,
    seed: int = PIN_A_SEED,
) -> float:
    """PIN A on whitened TRAIN φ (same fixed sample knobs, then transform)."""
    matrix = np.asarray(raw_vectors, dtype=np.float64)
    n_rows = matrix.shape[0]
    take = min(sample_size, n_rows)
    if take == n_rows:
        sample = matrix
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_rows, size=take, replace=False)
        sample = matrix[indices]
    whitened = transform.transform_matrix(sample)
    return pin_a_effective_rank(whitened, sample_size=whitened.shape[0], seed=seed)


def _log_softmax_finite(row: np.ndarray) -> np.ndarray:
    """Log-softmax over finite entries; non-finite stay ``-inf``."""
    out = np.full_like(row, -np.inf, dtype=np.float64)
    finite = np.isfinite(row)
    if not np.any(finite):
        return out
    vals = row[finite]
    shifted = vals - np.max(vals)
    # logsumexp of finite entries
    log_z = float(np.log(np.exp(shifted).sum()))
    out[finite] = shifted - log_z
    return out


def pin_b_decode(
    scorer: GroundedAttachmentScorer,
    tokens: Sequence[str],
    sent_id: str,
    l0_scores: np.ndarray,
    provider: GroundedFeatureProvider | WhitenedFeatureProvider,
) -> tuple[int, ...]:
    """PIN B: CLE on a pure log-probability arc matrix (no raw/prob mix).

    For each dependent ``i``:
    a. ``l0_logprob_row`` = log-softmax of L0 raw heuristic scores over finite cols.
    b. If ``i`` is grounded-covered and has ≥1 other covered governor candidate:
       bilinear log-softmax over the covered-governor set only; ROOT + uncovered
       governors keep ``l0_logprob_row``; then renormalize the combined finite row
       via logsumexp so ``exp(row).sum() == 1``.
    c. Else pure ``l0_logprob_row`` (already a log-distribution; re-normalize is
       idempotent — we still run it for uniformity).
    d. Never write raw scores or non-log probabilities into the final matrix.
    """
    size = len(tokens)
    matrix_l0 = np.asarray(l0_scores, dtype=np.float64)
    if matrix_l0.shape != (size, size + 1):
        raise ValueError(
            f"l0_scores must have shape ({size}, {size + 1}), got {matrix_l0.shape}"
        )
    final = np.full((size, size + 1), -np.inf, dtype=np.float64)
    covered_words = [i for i in range(size) if provider.covered(sent_id, i)]
    covered_set = set(covered_words)

    for dep in range(size):
        l0_logprob_row = _log_softmax_finite(matrix_l0[dep])
        others = [g for g in covered_words if g != dep]
        if dep in covered_set and others:
            dep_phi = provider.get(sent_id, dep)
            assert dep_phi is not None
            gov_phis = np.stack([provider.get(sent_id, g) for g in others])
            logits = (dep_phi @ scorer.U) @ (gov_phis @ scorer.V).T + scorer.b
            bil_log = _log_softmax_finite(np.asarray(logits, dtype=np.float64))
            combined = l0_logprob_row.copy()
            for governor, log_p in zip(others, bil_log, strict=True):
                combined[governor + 1] = float(log_p)
            # Renormalize mixed bilinear+L0 row to a proper log-distribution.
            final[dep] = _log_softmax_finite(combined)
        else:
            # Pure L0 log-prob row; re-normalize is idempotent (documented choice).
            final[dep] = _log_softmax_finite(l0_logprob_row)

    return governor_columns_to_offsets(chu_liu_edmonds(final))


def select_dev_variant_layer(
    sweep_rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
) -> tuple[str, int]:
    """Pick (variant, layer_index) maximizing ``score_key`` on DEV-only rows.

    Tie-break total order (stated): prefer V1 over V2, then smallest k, then
    smallest layer_index.  Rows must be DEV-tagged evaluation results only —
    callers must never pass TEST metrics into this function.
    """
    if not sweep_rows:
        raise ValueError("cannot select from an empty DEV sweep")
    best = min(
        sweep_rows,
        key=lambda row: (
            -float(row[score_key]),
            VARIANT_TIE_RANK[str(row["variant"])],
            int(row["layer_index"]),
        ),
    )
    return str(best["variant"]), int(best["layer_index"])


def evaluate_dev_attach_layer_pin_b(
    *,
    dev: Sequence[Sentence],
    dev_heads: Mapping[str, HeadDeprelRecord],
    dev_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    scorer: GroundedAttachmentScorer,
    provider: GroundedFeatureProvider | WhitenedFeatureProvider,
) -> dict[str, Any]:
    """DEV UAS on covered-dependent tokens using PIN-B decode (not #122 decode)."""
    keys = collect_covered_dependent_keys(dev, provider)  # type: ignore[arg-type]
    gold_by_key: dict[tuple[str, int], int] = {}
    predicted_by_key: dict[tuple[str, int], int] = {}
    for sentence in dev:
        head = aligned_head_record(sentence, dev_heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = dev_pos[sentence.sent_id]
        l0_scores = l0_edge_scores(l0_student, sentence.tokens, pos)
        offsets = pin_b_decode(
            scorer, sentence.tokens, sentence.sent_id, l0_scores, provider
        )
        for index, offset in enumerate(offsets):
            key = (sentence.sent_id, index)
            if provider.covered(sentence.sent_id, index):
                predicted_by_key[key] = int(offset)
                gold_by_key[key] = int(head.head_offset[index])
    return {
        "layer_index": provider.layer_index,
        "layer": provider.layer,
        "dev_uas": score_uas_on_keys(predicted_by_key, gold_by_key, keys),
        "dev_covered_dependent_fraction": covered_dependent_fraction(
            dev, provider  # type: ignore[arg-type]
        ),
        "dev_doubly_covered_fraction": attach_doubly_covered_fraction(
            dev, dev_heads, provider  # type: ignore[arg-type]
        ),
    }


def evaluate_matched_four_arm_uas_pin_b(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    surface_bilinear: BilinearAttachmentScorer,
    raw_scorer: GroundedAttachmentScorer,
    whitened_scorer: GroundedAttachmentScorer,
    raw_provider: GroundedFeatureProvider,
    whitened_provider: WhitenedFeatureProvider,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
]:
    """Matched 4-arm UAS; raw and whitened both use PIN-B (only φ differs)."""
    # Coverage mask from raw provider (= whitened coverage; same keys).
    keys = collect_covered_dependent_keys(test, raw_provider)
    expected = len(keys)
    gold_by_key: dict[tuple[str, int], int] = {}
    heads_by_arm: dict[str, dict[str, tuple[int, ...]]] = {
        "l0": {},
        "surface119": {},
        "raw122": {},
        "whitened": {},
    }

    for sentence in test:
        head = aligned_head_record(sentence, heads)
        if not isinstance(head, HeadDeprelRecord):
            raise TypeError("labeled dependency record required")
        pos = predicted_pos[sentence.sent_id]
        sent_id = sentence.sent_id
        tokens = sentence.tokens
        l0_scores = l0_edge_scores(l0_student, tokens, pos)

        l0_offsets = l0_student.predict_sentence(tokens, pos).head_offset
        surface_offsets = governor_columns_to_offsets(
            chu_liu_edmonds(surface_bilinear.score_matrix(tokens, pos))
        )
        raw_offsets = pin_b_decode(raw_scorer, tokens, sent_id, l0_scores, raw_provider)
        white_offsets = pin_b_decode(
            whitened_scorer, tokens, sent_id, l0_scores, whitened_provider
        )
        heads_by_arm["l0"][sent_id] = tuple(int(v) for v in l0_offsets)
        heads_by_arm["surface119"][sent_id] = tuple(int(v) for v in surface_offsets)
        heads_by_arm["raw122"][sent_id] = tuple(int(v) for v in raw_offsets)
        heads_by_arm["whitened"][sent_id] = tuple(int(v) for v in white_offsets)

        for index, gold_offset in enumerate(head.head_offset):
            gold_by_key[(sent_id, index)] = int(gold_offset)

    predicted: dict[str, dict[tuple[str, int], int]] = {arm: {} for arm in heads_by_arm}
    for arm, by_sent in heads_by_arm.items():
        for sent_id, index in keys:
            predicted[arm][(sent_id, index)] = int(by_sent[sent_id][index])

    arm_counts = {arm: len(predicted[arm]) for arm in predicted}
    assert_matched_token_counts(arm_counts, expected)

    metrics: dict[str, dict[str, Any]] = {}
    for arm in ("l0", "surface119", "raw122", "whitened"):
        metrics[arm] = {
            "uas": score_uas_on_keys(predicted[arm], gold_by_key, keys),
            "tokens": expected,
        }
    return (
        metrics,
        heads_by_arm["l0"],
        heads_by_arm["raw122"],
        heads_by_arm["whitened"],
    )


def evaluate_matched_four_arm_labeler(
    test: Sequence[Sentence],
    heads: Mapping[str, HeadDeprelRecord],
    predicted_pos: Mapping[str, Sequence[str]],
    l0_student: GermanR3DependencyStudent,
    surface_labeler: BilinearRelationLabeler,
    raw_labeler: GroundedRelationLabeler,
    whitened_labeler: GroundedRelationLabeler,
    raw_provider: GroundedFeatureProvider,
    whitened_provider: WhitenedFeatureProvider,
    case_student: GermanR1Student,
) -> tuple[dict[str, dict[str, Any]], float]:
    """Four-arm labeler eval: unary / surface / raw-121 / whitened on same L0 heads.

    Reuses ``evaluate_matched_three_arms`` for unary/surface/grounded structure by
    running it twice (raw and whitened as the grounded arm) and merging, with an
    explicit UAS identity assert across all four.
    """
    raw_metrics, fraction = evaluate_matched_three_arms(
        test,
        heads,
        predicted_pos,
        l0_student,
        surface_labeler,
        raw_labeler,
        raw_provider,
        case_student,
    )
    white_metrics, white_fraction = evaluate_matched_three_arms(
        test,
        heads,
        predicted_pos,
        l0_student,
        surface_labeler,
        whitened_labeler,
        whitened_provider,  # type: ignore[arg-type]
        case_student,
    )
    if abs(fraction - white_fraction) > 1e-12:
        raise AssertionError(
            f"raw vs whitened doubly-covered fraction mismatch: "
            f"{fraction} vs {white_fraction}"
        )
    # UAS identity: every arm uses the same L0 heads (evaluate_matched_three_arms
    # already asserts within each run); require identical UAS across all four.
    uas_values = {
        "unary": float(raw_metrics["unary"]["uas"]),
        "surface119": float(raw_metrics["surface119"]["uas"]),
        "raw121": float(raw_metrics["grounded"]["uas"]),
        "whitened": float(white_metrics["grounded"]["uas"]),
    }
    ref_uas = uas_values["unary"]
    for name, uas in uas_values.items():
        if uas != ref_uas:
            raise AssertionError(
                f"matched-labeler UAS invariant failed for {name}: "
                f"unary={ref_uas}, {name}={uas}"
            )
    # Token counts must match across grounded arms.
    if int(raw_metrics["grounded"]["tokens"]) != int(white_metrics["grounded"]["tokens"]):
        raise AssertionError(
            "raw121 vs whitened token count mismatch: "
            f"{raw_metrics['grounded']['tokens']} vs {white_metrics['grounded']['tokens']}"
        )
    # Unary/surface should be identical between the two three-arm runs.
    for arm in ("unary", "surface119"):
        if abs(float(raw_metrics[arm]["las_strict"]) - float(white_metrics[arm]["las_strict"])) > 1e-12:
            raise AssertionError(f"baseline arm {arm} drifted between raw/whitened evals")

    return {
        "unary": raw_metrics["unary"],
        "surface119": raw_metrics["surface119"],
        "raw121": raw_metrics["grounded"],
        "whitened": white_metrics["grounded"],
    }, fraction


def preregistered_whitening_read(
    *,
    labeler_unary: Mapping[str, Any],
    labeler_surface: Mapping[str, Any],
    labeler_raw121: Mapping[str, Any],
    labeler_whitened: Mapping[str, Any],
    attach_l0: Mapping[str, Any],
    attach_surface: Mapping[str, Any],
    attach_raw122: Mapping[str, Any],
    attach_whitened: Mapping[str, Any],
    attach_conv_base_las: float,
    attach_conv_full_las: float,
    case_whitened: float,
    case_baseline: float,
    labeler_dev_variant: str,
    labeler_dev_layer: int,
    attach_dev_variant: str,
    attach_dev_layer: int,
    effrank_raw: float,
    effrank_whitened: float,
    labeler_best_variant: str,
    attach_best_variant: str,
) -> dict[str, Any]:
    """Apply signed whitening-gate preregistration section 5 verbatim.

    Cross-vendor robustness cannot be self-certified in a single-vendor run:
    if a bar is cleared on this lane, the verdict is IN-BETWEEN with an
    explicit pending-cross-vendor note (not FIRES), matching the prereg's
    FIRES = bar AND robust requirement and the build-spec honesty rule.

    Summary ``dev_variant`` / ``dev_layer``: when labeler and attach select
    different cells, report attach's selection at the top level (attach was
    the disposition driver for this gate) and spell out both selections in
    ``text`` and the scoreboard.
    """
    lab_unary_las = float(labeler_unary["las_strict"])
    lab_white_las = float(labeler_whitened["las_strict"])
    lab_raw_las = float(labeler_raw121["las_strict"])
    lab_surface_las = float(labeler_surface["las_strict"])
    labeler_delta = lab_white_las - lab_raw_las
    labeler_gain_vs_unary = lab_white_las - lab_unary_las

    uas_l0 = float(attach_l0["uas"])
    uas_surface = float(attach_surface["uas"])
    uas_raw = float(attach_raw122["uas"])
    uas_white = float(attach_whitened["uas"])
    attach_uas_gain = uas_white - uas_l0
    attach_uas_delta_raw = uas_white - uas_raw
    attach_conv_las_gain = float(attach_conv_full_las) - float(attach_conv_base_las)

    labeler_bar = labeler_gain_vs_unary >= LAS_FIRES_THRESHOLD
    attach_bar = (
        attach_uas_gain >= UAS_FIRES_THRESHOLD
        and attach_conv_las_gain >= ATTACH_LAS_FIRES_THRESHOLD
    )
    labeler_improved = lab_white_las > lab_raw_las
    attach_improved = uas_white > uas_raw
    labeler_leq_raw = lab_white_las <= lab_raw_las
    attach_leq_raw = uas_white <= uas_raw

    cross_vendor_note = (
        "cross-vendor robustness pending the codex lane's parallel run and "
        "cannot be self-certified from this single-vendor grok run"
    )

    if labeler_leq_raw and attach_leq_raw:
        verdict = "HALTED"
        fired_primitive: str | None = "none"
        reason = (
            "whitening ≤ raw-φ on BOTH primitives → rogue-dim anisotropy was "
            "NOT the blocker → R4"
        )
    elif labeler_bar or attach_bar:
        # Bar cleared on this lane, but FIRES requires robust — pending.
        if labeler_bar and attach_bar:
            fired_primitive = "both"
        elif labeler_bar:
            fired_primitive = "labeler"
        else:
            fired_primitive = "attach"
        verdict = "IN-BETWEEN"
        reason = (
            f"bar cleared on this lane for {fired_primitive} "
            f"(labeler gain vs unary={labeler_gain_vs_unary:+.4f}, "
            f"attach UAS gain vs L0={attach_uas_gain:+.4f}, "
            f"conv LAS gain={attach_conv_las_gain:+.4f}); {cross_vendor_note} "
            f"— not claiming FIRES"
        )
    elif labeler_improved or attach_improved:
        verdict = "IN-BETWEEN"
        fired_primitive = "none"
        reason = (
            "whitening improves a primitive over its raw-φ predecessor but "
            "does not clear the #121/#122 bar"
        )
    else:
        # Defensive: no improvement and not both ≤ (e.g. equalities already HALTED).
        verdict = "HALTED"
        fired_primitive = "none"
        reason = "whitening does not improve either primitive over raw-φ → R4"

    # Through-line on whitened case (cascade).
    if case_whitened >= CASE_DELIVERABLE:
        through_line = {
            "met": True,
            "plateau": False,
            "text": (
                f"through-line met: serve-honest case={case_whitened:.4f} "
                f"≥ {CASE_DELIVERABLE}"
            ),
        }
    elif case_whitened > CASE_BASELINE:
        through_line = {
            "met": False,
            "plateau": True,
            "text": (
                f"through-line plateau: serve-honest case={case_whitened:.4f} "
                f"< {CASE_DELIVERABLE} but > ~{CASE_BASELINE}"
            ),
        }
    elif case_whitened <= case_baseline:
        through_line = {
            "met": False,
            "plateau": False,
            "text": (
                f"through-line diagnose: serve-honest case={case_whitened:.4f} "
                f"≤ baseline {case_baseline:.4f}"
            ),
        }
    else:
        through_line = {
            "met": False,
            "plateau": False,
            "text": (
                f"through-line below plateau: serve-honest case={case_whitened:.4f} "
                f"above baseline but ≤ ~{CASE_BASELINE}"
            ),
        }

    # MECHANISM: V1-only gain → conditioning; needs V2 → information-removal.
    variants_used = {labeler_best_variant, attach_best_variant}
    if not (labeler_improved or attach_improved):
        v1_vs_v2 = (
            "no accuracy gain over raw-φ; mechanism recorded without gain "
            f"(effrank raw={effrank_raw:.2f} → whitened={effrank_whitened:.2f})"
        )
    elif variants_used <= {"v1"}:
        v1_vs_v2 = "conditioning"
    elif "v1" not in variants_used:
        v1_vs_v2 = "information-removal"
    else:
        v1_vs_v2 = (
            "mixed: one primitive V1 (conditioning), one V2 (information-removal)"
        )

    # Top-level summary selection: attach cell (disposition driver); both in text.
    summary_variant = attach_dev_variant
    summary_layer = attach_dev_layer

    text = (
        f"{verdict}: labeler whitened las_strict={lab_white_las:.4f} "
        f"(unary={lab_unary_las:.4f} surface={lab_surface_las:.4f} "
        f"raw121={lab_raw_las:.4f} Δraw={labeler_delta:+.4f} "
        f"Δunary={labeler_gain_vs_unary:+.4f}); "
        f"attach whitened UAS={uas_white:.4f} "
        f"(l0={uas_l0:.4f} surface={uas_surface:.4f} raw122={uas_raw:.4f} "
        f"Δraw={attach_uas_delta_raw:+.4f} Δl0={attach_uas_gain:+.4f}) "
        f"conv LAS gain={attach_conv_las_gain:+.4f}; "
        f"DEV labeler variant={labeler_dev_variant} layer={labeler_dev_layer}, "
        f"DEV attach variant={attach_dev_variant} layer={attach_dev_layer} "
        f"(top-level dev_variant/dev_layer report attach selection); "
        f"PIN-A effrank raw={effrank_raw:.2f} whitened={effrank_whitened:.2f}; "
        f"mechanism={v1_vs_v2}; fired_primitive={fired_primitive}. "
        f"{reason}. {through_line['text']}"
    )

    return {
        "verdict": verdict,
        "fired_primitive": fired_primitive,
        "labeler_unary": lab_unary_las,
        "labeler_surface": lab_surface_las,
        "labeler_raw121": lab_raw_las,
        "labeler_whitened": lab_white_las,
        "labeler_delta": labeler_delta,
        "attach_l0": uas_l0,
        "attach_surface": uas_surface,
        "attach_raw122": uas_raw,
        "attach_whitened": uas_white,
        "attach_uas_gain": attach_uas_gain,
        "attach_conv_las_gain": attach_conv_las_gain,
        "dev_variant": summary_variant,
        "dev_layer": summary_layer,
        "labeler_dev_variant": labeler_dev_variant,
        "labeler_dev_layer": labeler_dev_layer,
        "attach_dev_variant": attach_dev_variant,
        "attach_dev_layer": attach_dev_layer,
        "effrank_raw": float(effrank_raw),
        "effrank_whitened": float(effrank_whitened),
        "v1_vs_v2": v1_vs_v2,
        "case_whitened": float(case_whitened),
        "through_line": through_line,
        "cross_vendor": "pending_codex_lane",
        "text": text,
    }


def _make_provider(
    npz_path: Path,
    layer_index: int,
    transform: WhiteningTransform | None,
) -> GroundedFeatureProvider | WhitenedFeatureProvider:
    base = GroundedFeatureProvider(npz_path, layer_index)
    if transform is None:
        return base
    return WhitenedFeatureProvider(base, transform)


def _fit_transform(
    train_npz: Path,
    layer_index: int,
    variant: str,
) -> WhiteningTransform:
    train_phi = load_layer_matrix(train_npz, layer_index)
    return WhiteningTransform.fit(
        train_phi, variant=variant, layer_index=layer_index
    )


def build_scoreboard() -> tuple[dict[str, Any], CampaignTestReadGuard]:
    """Fit whitening on TRAIN, DEV-sweep variants×layers, TEST-read once."""
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

    print("DEV: loading GSD dev once for L0 window + whitening grid.", flush=True)
    dev = load_split(DATA_ROOT, "dev", hasher)
    dev_heads_records = load_head_deprel_file(
        DATA_ROOT / "tasks" / "head_deprel" / "dev.jsonl", hasher
    )
    window, dev_coverage, coverage_curve = choose_window(dev_heads_records)
    dev_heads = head_deprel_by_sent(dev_heads_records)
    dev_pos = _predicted_pos_by_sentence(r1_student, dev)

    l0_student = GermanR3DependencyStudent(window)
    l0_student.fit(train, train_heads, train_pos)

    # Surface baselines (unchanged hyperparameters, seed 0).
    surface_label_config = RelationLabelerConfig(seed=0)
    surface_labeler = BilinearRelationLabeler(surface_label_config)
    print(
        f"FIT: surface-#119 labeler rank={surface_label_config.rank} "
        f"epochs={surface_label_config.epochs} lr={surface_label_config.learning_rate} "
        f"seed=0.",
        flush=True,
    )
    surface_labeler.fit(train, train_heads, train_pos)

    attach_config = BilinearConfig()
    surface_bilinear = BilinearAttachmentScorer(attach_config)
    print(
        f"FIT: surface-#119 attach rank={attach_config.rank} "
        f"epochs={attach_config.epochs} lr={attach_config.learning_rate} seed=0.",
        flush=True,
    )
    surface_bilinear.fit(train, train_heads, train_pos)

    # --- PIN A: raw effrank per layer on TRAIN (n=800, seed 0) ---
    print(
        "PIN A: centered participation-ratio effrank on TRAIN φ "
        f"(n={PIN_A_SAMPLE_SIZE}, seed={PIN_A_SEED}).",
        flush=True,
    )
    pin_a_raw_by_layer: dict[int, float] = {}
    train_phi_by_layer: dict[int, np.ndarray] = {}
    for layer_index in LAYER_INDICES:
        train_phi_by_layer[layer_index] = load_layer_matrix(TRAIN_NPZ, layer_index)
        pin_a_raw_by_layer[layer_index] = pin_a_effective_rank(
            train_phi_by_layer[layer_index]
        )
        print(
            f"PIN A raw layer={CHECKPOINT_LAYERS[layer_index]} "
            f"(idx={layer_index}): effrank={pin_a_raw_by_layer[layer_index]:.4f}",
            flush=True,
        )

    # Pre-fit all TRAIN whitening transforms (no DEV/TEST leakage).
    transforms: dict[tuple[str, int], WhiteningTransform] = {}
    pin_a_white: dict[tuple[str, int], float] = {}
    for variant in VARIANTS:
        for layer_index in LAYER_INDICES:
            tr = WhiteningTransform.fit(
                train_phi_by_layer[layer_index],
                variant=variant,
                layer_index=layer_index,
            )
            transforms[(variant, layer_index)] = tr
            pin_a_white[(variant, layer_index)] = pin_a_effective_rank_whitened(
                train_phi_by_layer[layer_index], tr
            )
            print(
                f"PIN A whitened variant={variant} layer={CHECKPOINT_LAYERS[layer_index]}: "
                f"effrank={pin_a_white[(variant, layer_index)]:.4f} "
                f"(raw={pin_a_raw_by_layer[layer_index]:.4f})",
                flush=True,
            )

    # --- DEV labeler sweep: variant × layer ---
    label_config = RelationLabelerConfig(seed=0)
    label_sweep: list[dict[str, Any]] = []
    labeler_by_cell: dict[tuple[str, int], GroundedRelationLabeler] = {}
    for variant in VARIANTS:
        for layer_index in LAYER_INDICES:
            print(
                f"FIT labeler: variant={variant} layer_index={layer_index} "
                f"layer={CHECKPOINT_LAYERS[layer_index]}.",
                flush=True,
            )
            tr = transforms[(variant, layer_index)]
            train_provider = _make_provider(TRAIN_NPZ, layer_index, tr)
            labeler = GroundedRelationLabeler(label_config)
            labeler.fit(train, train_heads, train_pos, train_provider)  # type: ignore[arg-type]
            print(
                f"FIT labeler losses variant={variant} layer={CHECKPOINT_LAYERS[layer_index]}: "
                + ", ".join(f"{loss:.6f}" for loss in labeler.epoch_losses),
                flush=True,
            )
            dev_provider = _make_provider(DEV_NPZ, layer_index, tr)
            row = evaluate_dev_layer(
                dev=dev,
                dev_heads=dev_heads,
                dev_pos=dev_pos,
                l0_student=l0_student,
                grounded_labeler=labeler,
                provider=dev_provider,  # type: ignore[arg-type]
            )
            row = {
                **row,
                "variant": variant,
                "dev_deprel_only_accuracy": row["dev_deprel_only_accuracy"],
            }
            label_sweep.append(row)
            labeler_by_cell[(variant, layer_index)] = labeler
            print(
                f"DEV labeler sweep: variant={variant} layer={row['layer']} "
                f"deprel-only={row['dev_deprel_only_accuracy']:.6f}",
                flush=True,
            )

    sel_lab_variant, sel_lab_layer_idx = select_dev_variant_layer(
        label_sweep, score_key="dev_deprel_only_accuracy"
    )
    sel_lab_layer = CHECKPOINT_LAYERS[sel_lab_layer_idx]
    selected_whitened_labeler = labeler_by_cell[(sel_lab_variant, sel_lab_layer_idx)]
    print(
        f"DEV selected LABELER: variant={sel_lab_variant} "
        f"layer_index={sel_lab_layer_idx} layer={sel_lab_layer}",
        flush=True,
    )

    # Raw-φ #121 labeler comparison arm at the same DEV-selected layer
    # (layer-matched; only φ whitening differs).  Hyperparameters fixed seed=0.
    print(
        f"FIT raw-φ #121 labeler at layer_index={sel_lab_layer_idx} "
        f"layer={sel_lab_layer} (PIN-matched layer for raw vs whitened).",
        flush=True,
    )
    raw_train_label_provider = GroundedFeatureProvider(TRAIN_NPZ, sel_lab_layer_idx)
    raw_labeler = GroundedRelationLabeler(RelationLabelerConfig(seed=0))
    raw_labeler.fit(train, train_heads, train_pos, raw_train_label_provider)

    # --- DEV attach sweep: variant × layer (PIN-B decode) ---
    attach_sweep: list[dict[str, Any]] = []
    attach_by_cell: dict[tuple[str, int], GroundedAttachmentScorer] = {}
    for variant in VARIANTS:
        for layer_index in LAYER_INDICES:
            print(
                f"FIT attach: variant={variant} layer_index={layer_index} "
                f"layer={CHECKPOINT_LAYERS[layer_index]}.",
                flush=True,
            )
            tr = transforms[(variant, layer_index)]
            train_provider = _make_provider(TRAIN_NPZ, layer_index, tr)
            scorer = GroundedAttachmentScorer(BilinearConfig())
            scorer.fit(train, train_heads, train_provider)  # type: ignore[arg-type]
            print(
                f"FIT attach losses variant={variant} layer={CHECKPOINT_LAYERS[layer_index]}: "
                + ", ".join(f"{loss:.6f}" for loss in scorer.epoch_losses),
                flush=True,
            )
            dev_provider = _make_provider(DEV_NPZ, layer_index, tr)
            row = evaluate_dev_attach_layer_pin_b(
                dev=dev,
                dev_heads=dev_heads,
                dev_pos=dev_pos,
                l0_student=l0_student,
                scorer=scorer,
                provider=dev_provider,
            )
            row = {**row, "variant": variant}
            attach_sweep.append(row)
            attach_by_cell[(variant, layer_index)] = scorer
            print(
                f"DEV attach sweep: variant={variant} layer={row['layer']} "
                f"uas={row['dev_uas']:.6f}",
                flush=True,
            )

    sel_att_variant, sel_att_layer_idx = select_dev_variant_layer(
        attach_sweep, score_key="dev_uas"
    )
    sel_att_layer = CHECKPOINT_LAYERS[sel_att_layer_idx]
    selected_whitened_attach = attach_by_cell[(sel_att_variant, sel_att_layer_idx)]
    print(
        f"DEV selected ATTACH: variant={sel_att_variant} "
        f"layer_index={sel_att_layer_idx} layer={sel_att_layer}",
        flush=True,
    )

    # Raw-φ #122 attach comparison arm at same DEV-selected attach layer; PIN-B.
    print(
        f"FIT raw-φ #122 attach at layer_index={sel_att_layer_idx} "
        f"layer={sel_att_layer} with PIN-B decode.",
        flush=True,
    )
    raw_train_attach_provider = GroundedFeatureProvider(TRAIN_NPZ, sel_att_layer_idx)
    raw_attach = GroundedAttachmentScorer(BilinearConfig())
    raw_attach.fit(train, train_heads, raw_train_attach_provider)

    # --- TEST read once ---
    guard = CampaignTestReadGuard()
    print(
        "FINAL EVAL: reading shared test tasks once and head_deprel test once.",
        flush=True,
    )
    test = load_shared_test_once(DATA_ROOT, hasher, guard)
    test_head_records = load_head_test_once(DATA_ROOT, hasher, guard)
    test_heads = head_deprel_by_sent(test_head_records)
    test_pos = _predicted_pos_by_sentence(r1_student, test)

    lab_tr = transforms[(sel_lab_variant, sel_lab_layer_idx)]
    att_tr = transforms[(sel_att_variant, sel_att_layer_idx)]
    test_raw_label_provider = GroundedFeatureProvider(TEST_NPZ, sel_lab_layer_idx)
    test_white_label_provider = WhitenedFeatureProvider(
        GroundedFeatureProvider(TEST_NPZ, sel_lab_layer_idx), lab_tr
    )
    test_raw_attach_provider = GroundedFeatureProvider(TEST_NPZ, sel_att_layer_idx)
    test_white_attach_provider = WhitenedFeatureProvider(
        GroundedFeatureProvider(TEST_NPZ, sel_att_layer_idx), att_tr
    )

    print("TEST: 4-arm labeler (unary / surface / raw121 / whitened).", flush=True)
    label_arm_metrics, label_test_fraction = evaluate_matched_four_arm_labeler(
        test,
        test_heads,
        test_pos,
        l0_student,
        surface_labeler,
        raw_labeler,
        selected_whitened_labeler,
        test_raw_label_provider,
        test_white_label_provider,
        r1_student,
    )

    print("TEST: 4-arm attach UAS (L0 / surface / raw122 / whitened) PIN-B.", flush=True)
    (
        attach_arm_metrics,
        l0_heads_by_sent,
        _raw_heads_by_sent,
        white_heads_by_sent,
    ) = evaluate_matched_four_arm_uas_pin_b(
        test,
        test_heads,
        test_pos,
        l0_student,
        surface_bilinear,
        raw_attach,
        selected_whitened_attach,
        test_raw_attach_provider,
        test_white_attach_provider,
    )

    print("CONVERSION: baseline (L0 heads + unary labels).", flush=True)
    conversion_base = evaluate_conversion_point(
        test,
        test_heads,
        test_pos,
        l0_heads_by_sent,
        use_grounded_labels=False,
        l0_student=l0_student,
        case_student=r1_student,
        grounded_labeler=None,
        label_provider=test_white_label_provider,  # type: ignore[arg-type]
    )
    print("CONVERSION: #121-equivalent (L0 heads + whitened labels).", flush=True)
    conversion_121 = evaluate_conversion_point(
        test,
        test_heads,
        test_pos,
        l0_heads_by_sent,
        use_grounded_labels=True,
        l0_student=l0_student,
        case_student=r1_student,
        grounded_labeler=selected_whitened_labeler,
        label_provider=test_white_label_provider,  # type: ignore[arg-type]
    )
    print("CONVERSION: full (whitened attach heads + whitened labels).", flush=True)
    conversion_full = evaluate_conversion_point(
        test,
        test_heads,
        test_pos,
        white_heads_by_sent,
        use_grounded_labels=True,
        l0_student=l0_student,
        case_student=r1_student,
        grounded_labeler=selected_whitened_labeler,
        label_provider=test_white_label_provider,  # type: ignore[arg-type]
    )
    if conversion_base["tokens"] != conversion_121["tokens"]:
        raise AssertionError(
            "baseline and #121 conversion must share the same doubly-covered n "
            f"(L0 heads): base={conversion_base['tokens']} 121={conversion_121['tokens']}"
        )

    guard.assert_complete()

    # PIN A numbers at the selected cells (report both; mechanism uses attach
    # cell's whitened effrank as the disposition-linked number, plus labeler).
    effrank_raw_lab = pin_a_raw_by_layer[sel_lab_layer_idx]
    effrank_white_lab = pin_a_white[(sel_lab_variant, sel_lab_layer_idx)]
    effrank_raw_att = pin_a_raw_by_layer[sel_att_layer_idx]
    effrank_white_att = pin_a_white[(sel_att_variant, sel_att_layer_idx)]
    # Top-level PIN A for the read: attach-selected cell (disposition driver).
    effrank_raw = effrank_raw_att
    effrank_whitened = effrank_white_att

    case_whitened = float(label_arm_metrics["whitened"]["serve_honest_morph_case"])
    case_baseline = float(label_arm_metrics["unary"]["serve_honest_morph_case"])

    read = preregistered_whitening_read(
        labeler_unary=label_arm_metrics["unary"],
        labeler_surface=label_arm_metrics["surface119"],
        labeler_raw121=label_arm_metrics["raw121"],
        labeler_whitened=label_arm_metrics["whitened"],
        attach_l0=attach_arm_metrics["l0"],
        attach_surface=attach_arm_metrics["surface119"],
        attach_raw122=attach_arm_metrics["raw122"],
        attach_whitened=attach_arm_metrics["whitened"],
        attach_conv_base_las=float(conversion_base["las_strict"]),
        attach_conv_full_las=float(conversion_full["las_strict"]),
        case_whitened=case_whitened,
        case_baseline=case_baseline,
        labeler_dev_variant=sel_lab_variant,
        labeler_dev_layer=sel_lab_layer,
        attach_dev_variant=sel_att_variant,
        attach_dev_layer=sel_att_layer,
        effrank_raw=effrank_raw,
        effrank_whitened=effrank_whitened,
        labeler_best_variant=sel_lab_variant,
        attach_best_variant=sel_att_variant,
    )

    scoreboard: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "scope": SCOPE,
        "measurement": (
            "whitening gate on MWT grounded φ; TRAIN-fit transform; "
            "DEV variant×layer sweep; PIN-B log-prob attach decode; TEST once"
        ),
        "serve_honest_features": [
            "TRAIN-only whitening map (V1 z-score or V2 top-k PC removal)",
            "Qwen2.5-3B MWT last_residual grounded φ",
            "PIN-B log-probability CLE decode for attach",
            "R1-predicted POS for L0/surface; unused by grounded φ",
        ],
        "gold_usage": (
            "GSD train gold heads/deprels are fit-time targets; whitening fit on "
            "TRAIN φ only (no gold dependency fields feed φ); variant+layer "
            "selected on DEV only; test gold is final scoring oracle only"
        ),
        "data_hash_sha256": hasher.hexdigest(),
        "test_read_counts": dict(guard.counts),
        "test_read_once": True,
        "seed": SEED,
        "l0_window": {
            "k": window,
            "dev_coverage": dev_coverage,
            "coverage_curve_through_selected_k": coverage_curve,
        },
        "fixed_hyperparameters": {
            "labeler": "RelationLabelerConfig(seed=0) defaults (#121)",
            "attach": "BilinearConfig() defaults (#122)",
            "seed": SEED,
            "feature_dim": FEATURE_DIM,
            "checkpoint_layers": list(CHECKPOINT_LAYERS),
            "pin_a_sample_size": PIN_A_SAMPLE_SIZE,
            "pin_a_seed": PIN_A_SEED,
            "sigma_zero_replacement": SIGMA_ZERO_REPLACEMENT,
            "mwt_train_npz": str(TRAIN_NPZ.name),
            "mwt_dev_npz": str(DEV_NPZ.name),
            "mwt_test_npz": str(TEST_NPZ.name),
        },
        "pin_a_raw_by_layer": {
            str(layer_index): {
                "layer": CHECKPOINT_LAYERS[layer_index],
                "effrank": pin_a_raw_by_layer[layer_index],
            }
            for layer_index in LAYER_INDICES
        },
        "pin_a_whitened": {
            f"{variant}@layer{layer_index}": {
                "variant": variant,
                "layer_index": layer_index,
                "layer": CHECKPOINT_LAYERS[layer_index],
                "effrank": pin_a_white[(variant, layer_index)],
            }
            for variant in VARIANTS
            for layer_index in LAYER_INDICES
        },
        "dev_labeler_sweep": label_sweep,
        "dev_attach_sweep": attach_sweep,
        "selected_labeler": {
            "variant": sel_lab_variant,
            "layer_index": sel_lab_layer_idx,
            "layer": sel_lab_layer,
            "effrank_raw": effrank_raw_lab,
            "effrank_whitened": effrank_white_lab,
        },
        "selected_attach": {
            "variant": sel_att_variant,
            "layer_index": sel_att_layer_idx,
            "layer": sel_att_layer,
            "effrank_raw": effrank_raw_att,
            "effrank_whitened": effrank_white_att,
        },
        "labeler_arms": label_arm_metrics,
        "attach_arms": attach_arm_metrics,
        "conversion": {
            "baseline": conversion_base,
            "121": conversion_121,
            "full": conversion_full,
        },
        "doubly_covered_fraction_labeler_test": label_test_fraction,
        "read": read,
    }
    return scoreboard, guard


def _print_labeler_row(name: str, metric: Mapping[str, Any]) -> None:
    print(
        f"{name:<15} {metric['uas']:.6f}  {metric['las_strict']:.6f}  "
        f"{metric['deprel_only_accuracy']:.6f}  "
        f"{metric['serve_honest_morph_case']:.6f} ({metric['serve_honest_morph_case_n']})"
    )


def print_scoreboard(scoreboard: Mapping[str, Any], guard: CampaignTestReadGuard) -> None:
    """Print DEV sweeps, PIN A, labeler/attach tables, §5 read."""
    print("\nWHITENING GATE — MWT grounded φ (anisotropy removal)")
    print(f"run_tag: {scoreboard['run_tag']}")
    print(f"scope: {scoreboard['scope']}")
    cfg = scoreboard["fixed_hyperparameters"]
    print(
        f"config: labeler={cfg['labeler']}; attach={cfg['attach']}; "
        f"seed={cfg['seed']} feature_dim={cfg['feature_dim']} "
        f"PIN-A n={cfg['pin_a_sample_size']} seed={cfg['pin_a_seed']} "
        f"(single seed 0 throughout)"
    )
    print(
        "matched control: labeler arms share L0 heads; attach arms scored on "
        "identical covered-dependent keys; raw vs whitened share PIN-B decode"
    )
    print(
        "serve honesty: whitening fit on TRAIN only; grounded φ from text dump; "
        "no gold dependency fields at predict time; TEST read once"
    )

    print("\nPIN A EFFRANK (TRAIN, n=800 seed=0) — raw by layer")
    for layer_index in LAYER_INDICES:
        row = scoreboard["pin_a_raw_by_layer"][str(layer_index)]
        print(
            f"  layer_idx={layer_index} layer={row['layer']} "
            f"effrank_raw={row['effrank']:.4f}"
        )
    print("PIN A EFFRANK — whitened (all variant×layer cells)")
    for _key, row in sorted(
        scoreboard["pin_a_whitened"].items(),
        key=lambda item: (VARIANT_TIE_RANK[item[1]["variant"]], item[1]["layer_index"]),
    ):
        print(
            f"  variant={row['variant']:<6} layer={row['layer']:<4} "
            f"effrank_whitened={row['effrank']:.4f}"
        )

    print("\nDEV LABELER SWEEP (deprel-only on doubly-covered arcs)")
    print(f"{'variant':<8} {'layer_idx':<10} {'layer':<8} {'deprel_only':<14}")
    for row in scoreboard["dev_labeler_sweep"]:
        print(
            f"{row['variant']:<8} {row['layer_index']:<10} {row['layer']:<8} "
            f"{row['dev_deprel_only_accuracy']:<14.6f}"
        )
    sel_l = scoreboard["selected_labeler"]
    print(
        f"selected labeler: variant={sel_l['variant']} "
        f"layer_index={sel_l['layer_index']} layer={sel_l['layer']} "
        f"effrank raw→whitened {sel_l['effrank_raw']:.2f}→{sel_l['effrank_whitened']:.2f}"
    )

    print("\nDEV ATTACH SWEEP (UAS on covered-dependent, PIN-B decode)")
    print(f"{'variant':<8} {'layer_idx':<10} {'layer':<8} {'dev_uas':<12}")
    for row in scoreboard["dev_attach_sweep"]:
        print(
            f"{row['variant']:<8} {row['layer_index']:<10} {row['layer']:<8} "
            f"{row['dev_uas']:<12.6f}"
        )
    sel_a = scoreboard["selected_attach"]
    print(
        f"selected attach: variant={sel_a['variant']} "
        f"layer_index={sel_a['layer_index']} layer={sel_a['layer']} "
        f"effrank raw→whitened {sel_a['effrank_raw']:.2f}→{sel_a['effrank_whitened']:.2f}"
    )

    print(
        "\nLABELER arm      UAS       LAS-strict deprel-only case (n)"
    )
    arms_l = scoreboard["labeler_arms"]
    _print_labeler_row("unary", arms_l["unary"])
    _print_labeler_row("surface-#119", arms_l["surface119"])
    _print_labeler_row("raw-121", arms_l["raw121"])
    _print_labeler_row("whitened", arms_l["whitened"])

    print("\nATTACH arm       UAS        tokens")
    arms_a = scoreboard["attach_arms"]
    for name, key in (
        ("L0", "l0"),
        ("surface-#119", "surface119"),
        ("raw-122", "raw122"),
        ("whitened", "whitened"),
    ):
        m = arms_a[key]
        print(f"{name:<15} {m['uas']:<10.6f} {m['tokens']}")

    print("\nCONVERSION (whitened labels; baseline/#121 share L0 heads)")
    print(f"{'point':<12} {'LAS-strict':<12} {'case':<12} {'tokens':<10}")
    conv = scoreboard["conversion"]
    for name, key in (("baseline", "baseline"), ("#121", "121"), ("full", "full")):
        m = conv[key]
        print(
            f"{name:<12} {m['las_strict']:<12.6f} "
            f"{m['case']:<12.6f} {m['tokens']:<10}"
        )

    print("\nTHE PRE-REGISTERED READ (§5)")
    print(scoreboard["read"]["text"])
    print(f"through-line: {scoreboard['read']['through_line']['text']}")
    print(f"mechanism (v1_vs_v2): {scoreboard['read']['v1_vs_v2']}")
    print(f"cross-vendor: {scoreboard['read']['cross_vendor']}")
    guard.assert_complete()
    print("TEST READ ONCE: CONFIRMED")


def main() -> None:
    scoreboard, guard = build_scoreboard()
    OUTPUT_PATH.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"structured_output: {OUTPUT_PATH}")
    print_scoreboard(scoreboard, guard)


if __name__ == "__main__":
    main()
