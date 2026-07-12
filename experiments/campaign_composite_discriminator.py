"""Composite discriminator over #87's four axes (slice — ONE shot, no retune).

Registered hyperparameters (FIXED before any measurement; do not retune):
  - Model: L2-logistic via torch (sklearn absent from .venv)
  - Loss: mean(binary cross-entropy with logits) + 0.5 * ||w||^2
          (sklearn C=1 convention: mean(log_loss) + (1/(2*C))*||w||^2; intercept unpenalized)
  - Optimizer: Adam, lr=0.1, steps=500, betas=(0.9, 0.999), eps=1e-8
  - C=1, unweighted; torch fit seed=0 for half-A full fit (CV folds seed from their recipe)
  - Stratified 5-fold CV x 20 repeats (seeds 0..19) for Clause 1
  - Permutation null: 1000 label shuffles from a single Generator seeded 42;
    each shuffle yields one pooled-OOF AUC via 5-fold recipe with fold-generation seed=0
    held fixed while only labels are shuffled (NOT 20 repeats x 1000 shuffles)
  - Half-A OOF-score deciles: stratified 5-fold OOF scores within half-A's gain/regression
    rows (seed=0), then 10 decile thresholds from those OOF scores; full half-A model is
    a single fit on all half-A gain/regression rows used for val_te / half-B / #86 scoring
  - WYLY_SEED=0 throughout

CLAUSE 1 — cross-validated discriminability AUC >= 0.70 beating a permutation null
CLAUSE 2 — rescue replay showing the discriminator recovers a zero-regression-admissible
            operating point (half-B four-clause #81 contrast)

Strategy for Stage-1 row reconstruction: (a) call _stage1_and_2() for authoritative
summary AND independently re-run the same sequence for row indices / B0 traces, then
STOP-loud assert counts match stage1 and data/frontier_rows.json (143 gains / 62 regs).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Pin seed before anything that may read it (frames_at_scale pattern).
os.environ.setdefault("WYLY_SEED", "0")

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

OUTPUT = REPO / "data" / "composite_discriminator.json"
FRONTIER_ARTIFACT = REPO / "data" / "frontier_rows.json"
FRAMES_ARTIFACT = REPO / "data" / "frames_at_scale.json"

# --- registered hyperparameters (do not retune after seeing results) ----------
LOGISTIC_LR = 0.1
LOGISTIC_STEPS = 500
LOGISTIC_OPTIM = "adam"
LOGISTIC_C = 1.0  # sklearn convention
# Loss formula (comment required by registration):
#   L(w,b) = mean_i BCEWithLogits(x_i·w + b, y_i) + (1/(2*C)) * ||w||^2
#   with C=1 ⇒ mean BCE + 0.5 * ||w||^2 ; intercept b is unpenalized; no N-average on penalty.
CV_N_FOLDS = 5
CV_N_REPEATS = 20
CV_REPEAT_SEEDS = list(range(20))  # 0..19
PERM_N_SHUFFLES_TARGET = 1000
PERM_SEED = 42
PERM_FOLD_SEED = 0  # fixed fold-generation seed for null shuffles
HALF_SPLIT_SEED = 0
HALF_A_OOF_SEED = 0
TORCH_FIT_SEED = 0
VAL_GAIN_ADMIT = 0.005  # clause (ii) analog on val_te
HALF_B_AGREE_DELTA = 0.005
RESCUE_86_MARGINAL = 0.003
ADMIT_THRESH = 5e-4

REGISTERED_RULES = (
    "CLAUSE 1 -- cross-validated discriminability AUC >= 0.70 beating a permutation null; "
    "CLAUSE 2 -- a rescue replay showing the discriminator recovers a "
    "zero-regression-admissible operating point. Both must pass; otherwise the "
    "pre-written plateau wording applies. The thread ends either way; no iteration -- "
    "ONE shot only, do not tune anything after seeing results."
)
CLAUSE_1_VERBATIM = (
    "CLAUSE 1 -- cross-validated discriminability AUC >= 0.70 beating a permutation null"
)
CLAUSE_2_VERBATIM = (
    "CLAUSE 2 -- a rescue replay showing the discriminator recovers a "
    "zero-regression-admissible operating point"
)
NO_ADMISSIBLE_MSG = "no admissible operating point (discriminator-filtered)"
TWO_TRAININGS_NOTE = (
    "CV models for clause 1 vs the half-A model for clause 2 back separate claims: "
    "the 20x5-fold CV models fit for Clause 1 are NOT the same model object as the "
    "single half-A model fit for Clause 2 / the #86 replay, and neither claim "
    "(Clause 1's AUC, Clause 2's rescue, the #86 replay) transfers evidentially to "
    "the others' model."
)

# Plateau wording constructed following #82/#83/#86/#87 Tags / "What X means" pattern.
# HONESTY: no separately pre-registered plateau text was found in this repo.
PLATEAU_TEMPLATE = (
    "PLATEAU — composite discriminator does not clear the registered dual bar. "
    "Slice #87 left four weakly-separable axes (~0.60) and priced one multivariate "
    "attempt; that attempt is now closed. CLAUSE 1 median pooled-OOF AUC was "
    "{clause1_auc:.6f} (needed >= 0.70; pass={clause1_pass}); permutation two-sided "
    "p={clause1_p:.6g} (needed < 0.05). CLAUSE 2: {clause2_detail}. "
    "The gain/regression coupling remains stage-invariant at the measured size; "
    "no zero-regression-admissible composite operating point is recommended. "
    "Thread ends here (one shot; no iteration)."
)


def log(message: str = "") -> None:
    print(message, flush=True)


# =============================================================================
# L2-logistic (torch) + z-scoring + stratified CV
# =============================================================================


def zscore_fit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std on training rows only (std floored at 1e-8)."""
    mean = x.double().mean(dim=0)
    std = x.double().std(dim=0, unbiased=False).clamp_min(1e-8)
    return mean, std


def zscore_transform(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (x.double() - mean) / std


def fit_l2_logistic(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    lr: float = LOGISTIC_LR,
    steps: int = LOGISTIC_STEPS,
    c: float = LOGISTIC_C,
    seed: int = TORCH_FIT_SEED,
) -> dict[str, torch.Tensor]:
    """Fit L2-logistic; returns weight [D] and bias scalar on CPU float64.

    Loss (registered): mean(BCEWithLogits(x@w+b, y)) + (1/(2*C))*||w||^2
    with C=1 ⇒ mean BCE + 0.5 * ||w||^2. Intercept unpenalized.
    """
    torch.manual_seed(seed)
    xd = x.detach().double().cpu()
    yd = y.detach().double().cpu().view(-1)
    n, d = xd.shape
    w = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    inv_2c = 1.0 / (2.0 * c)
    for _ in range(steps):
        opt.zero_grad()
        logits = xd @ w + b
        # mean log-loss + (1/(2C)) ||w||^2  (sklearn C convention; no N-average on penalty)
        loss = F.binary_cross_entropy_with_logits(logits, yd) + inv_2c * (w * w).sum()
        loss.backward()
        opt.step()
    return {"weight": w.detach().clone(), "bias": b.detach().clone()}


def decision_scores(
    x: torch.Tensor, model: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Linear decision scores (logits), not probabilities."""
    xd = x.detach().double().cpu()
    return xd @ model["weight"] + model["bias"]


def stratified_kfold_indices(
    labels: torch.Tensor, n_folds: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return list of (train_idx, test_idx) for stratified k-fold on 0/1 labels."""
    y = labels.detach().long().cpu().view(-1)
    n = len(y)
    if n_folds < 2 or n_folds > n:
        raise ValueError(f"invalid n_folds={n_folds} for n={n}")
    g = torch.Generator().manual_seed(int(seed))
    folds_for = torch.empty(n, dtype=torch.long)
    for cls in (0, 1):
        idx = (y == cls).nonzero(as_tuple=False).flatten()
        if len(idx) < n_folds:
            # still assign round-robin so every fold gets what exists
            order = idx[torch.randperm(len(idx), generator=g)] if len(idx) else idx
        else:
            order = idx[torch.randperm(len(idx), generator=g)]
        for i, row in enumerate(order.tolist()):
            folds_for[row] = i % n_folds
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for fold in range(n_folds):
        test_idx = (folds_for == fold).nonzero(as_tuple=False).flatten()
        train_idx = (folds_for != fold).nonzero(as_tuple=False).flatten()
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise RuntimeError(f"empty fold {fold} under seed={seed}")
        out.append((train_idx, test_idx))
    return out


def pooled_oof_scores(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_folds: int = CV_N_FOLDS,
    fold_seed: int = 0,
    lr: float = LOGISTIC_LR,
    steps: int = LOGISTIC_STEPS,
    fit_seed: int = TORCH_FIT_SEED,
) -> torch.Tensor:
    """One pooled OOF decision score per row via stratified k-fold + train-only z-score."""
    x = features.detach().double().cpu()
    y = labels.detach().double().cpu().view(-1)
    scores = torch.empty(len(y), dtype=torch.float64)
    for train_idx, test_idx in stratified_kfold_indices(y, n_folds, fold_seed):
        mean, std = zscore_fit(x[train_idx])
        x_tr = zscore_transform(x[train_idx], mean, std)
        x_te = zscore_transform(x[test_idx], mean, std)
        model = fit_l2_logistic(x_tr, y[train_idx], lr=lr, steps=steps, seed=fit_seed)
        scores[test_idx] = decision_scores(x_te, model)
    return scores


def pooled_oof_auc(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_folds: int = CV_N_FOLDS,
    fold_seed: int = 0,
    **fit_kw: Any,
) -> float:
    from experiments.campaign_frontier_rows import auc_mann_whitney

    scores = pooled_oof_scores(
        features, labels, n_folds=n_folds, fold_seed=fold_seed, **fit_kw
    )
    y = labels.detach().long().cpu().view(-1)
    pos = scores[y == 1].tolist()
    neg = scores[y == 0].tolist()
    return auc_mann_whitney(pos, neg)


def cv_auc_distribution(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    repeat_seeds: list[int] | None = None,
    n_folds: int = CV_N_FOLDS,
    **fit_kw: Any,
) -> dict[str, Any]:
    seeds = list(range(CV_N_REPEATS)) if repeat_seeds is None else list(repeat_seeds)
    aucs = [
        pooled_oof_auc(features, labels, n_folds=n_folds, fold_seed=s, **fit_kw)
        for s in seeds
    ]
    t = torch.tensor(aucs, dtype=torch.float64)
    return {
        "aucs": aucs,
        "median": float(t.median()),
        "min": float(t.min()),
        "max": float(t.max()),
        "repeat_seeds": seeds,
        "n_folds": n_folds,
    }


def permutation_null_aucs(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_shuffles: int = PERM_N_SHUFFLES_TARGET,
    perm_seed: int = PERM_SEED,
    fold_seed: int = PERM_FOLD_SEED,
    n_folds: int = CV_N_FOLDS,
    **fit_kw: Any,
) -> list[float]:
    """1000 (or n_shuffles) label shuffles from one Generator(perm_seed); fold_seed fixed.

    Each shuffle: one pooled-OOF 5-fold AUC (same recipe as one Clause-1 repeat).
    """
    y0 = labels.detach().long().cpu().view(-1).clone()
    g = torch.Generator().manual_seed(int(perm_seed))
    null: list[float] = []
    for _ in range(n_shuffles):
        perm = torch.randperm(len(y0), generator=g)
        y_shuf = y0[perm].double()
        null.append(
            pooled_oof_auc(
                features, y_shuf, n_folds=n_folds, fold_seed=fold_seed, **fit_kw
            )
        )
    return null


def empirical_perm_p(observed: float, null_aucs: list[float]) -> float:
    """Two-sided around 0.5: (count |null-0.5| >= |obs-0.5| + 1) / (n+1)."""
    n = len(null_aucs)
    if n == 0:
        raise ValueError("empty null")
    thr = abs(observed - 0.5)
    extreme = sum(1 for a in null_aucs if abs(a - 0.5) >= thr - 1e-15)
    return (extreme + 1) / (n + 1)


def score_deciles(scores: torch.Tensor) -> list[float]:
    """10 thresholds at the 10th..100th percentiles of scores (fixed grid)."""
    s = scores.detach().double().cpu().view(-1)
    if not len(s):
        raise ValueError("empty scores for deciles")
    q = torch.arange(1, 11, dtype=torch.float64) / 10
    return [float(x) for x in torch.quantile(s, q)]


# =============================================================================
# 50/50 half-A / half-B split (stratify by B0-correctness)
# =============================================================================


def split_half_ab(
    n: int,
    b0_correct: torch.Tensor,
    *,
    seed: int = HALF_SPLIT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """50/50 split of positions 0..n-1 stratified by B0-correctness (seed registered)."""
    y = b0_correct.detach().bool().cpu().view(-1)
    if len(y) != n:
        raise ValueError("b0_correct length must equal n")
    g = torch.Generator().manual_seed(int(seed))
    half_a_parts: list[torch.Tensor] = []
    half_b_parts: list[torch.Tensor] = []
    for cls in (False, True):
        idx = (y == cls).nonzero(as_tuple=False).flatten()
        order = idx[torch.randperm(len(idx), generator=g)]
        mid = len(order) // 2
        half_a_parts.append(order[:mid])
        half_b_parts.append(order[mid:])
    half_a = torch.sort(torch.cat(half_a_parts)).values
    half_b = torch.sort(torch.cat(half_b_parts)).values
    if len(half_a) + len(half_b) != n:
        raise AssertionError("half-A + half-B size mismatch")
    if len(torch.isin(half_a, half_b).nonzero()) != 0:
        # disjointness
        inter = set(half_a.tolist()) & set(half_b.tolist())
        if inter:
            raise AssertionError(f"half-A ∩ half-B nonempty: {sorted(inter)[:5]}...")
    return half_a, half_b


# =============================================================================
# Discriminator-filter claim semantics (Clause 2 merge)
# =============================================================================


def discriminator_filter_claim(
    b0_pred: torch.Tensor,
    b1_pred: torch.Tensor,
    b1_conf: torch.Tensor,
    disc_scores: torch.Tensor,
    *,
    tau: float,
    threshold: float,
) -> torch.Tensor:
    """Final pred: B1 iff B1 not weak under gate (pred>=0 & conf>tau) AND disc>=threshold.

    Otherwise B0. B0 is never filtered by the discriminator; weak B1 is never resurrected.
    Mirrors BlockStack._merge_upstream weak = (pred < 0) | (conf <= tau) for carry=gated.
    """
    b0 = b0_pred
    b1 = b1_pred
    conf = b1_conf
    # disc_scores are CPU float64 from the torch logistic; align to pred device/dtype for where()
    disc = disc_scores.to(device=b0.device, dtype=conf.dtype if conf is not None else b0.dtype)
    weak = (b1 < 0) | (conf <= tau)
    claim_b1 = (~weak) & (disc >= threshold)
    return torch.where(claim_b1, b1, b0)


def frame_filter_claim(
    baseline_pred: torch.Tensor,
    candidate_pred: torch.Tensor,
    disc_scores: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """#86 claim filter: use frame-augmented pred only when disc >= threshold; else baseline."""
    disc = disc_scores.to(device=baseline_pred.device)
    return torch.where(disc >= threshold, candidate_pred, baseline_pred)


# =============================================================================
# Stage-1 reconstruction (strategy a)
# =============================================================================


def _load_frontier_stage1_counts() -> dict[str, int]:
    if not FRONTIER_ARTIFACT.exists():
        raise FileNotFoundError(f"missing {FRONTIER_ARTIFACT}")
    payload = json.loads(FRONTIER_ARTIFACT.read_text())
    return dict(payload["stages"][0]["counts"])


def reconstruct_stage1_rows() -> dict[str, Any]:
    """Strategy (a): _stage1_and_2 summary + independent re-run for indices/traces."""
    from experiments import campaign_frontier_rows as fr
    from experiments import campaign_wikitext_gated_rescue as gated_rescue
    from pil.wyly_block import WylyBlock

    log("=== STAGE-1 RECONSTRUCTION (strategy a) ===")
    artifact_counts = _load_frontier_stage1_counts()
    log(f"data/frontier_rows.json stage1 counts: {artifact_counts}")
    if artifact_counts.get("gains") != 143 or artifact_counts.get("regressions") != 62:
        raise RuntimeError(
            f"STOP: frontier_rows.json stage1 counts are not 143/62: {artifact_counts}"
        )

    stage1_summary, _stage2 = fr._stage1_and_2()
    summary_counts = stage1_summary["counts"]
    log(f"_stage1_and_2 counts: {summary_counts}")

    model, rules, frozen_core, _ = gated_rescue.load_frozen_b0(gated_rescue.B0_FREEZE)
    if not gated_rescue.b0_within_registered_band(frozen_core["agree"]):
        raise RuntimeError("STOP: frozen B0 fails registered band")
    ids, y, cls, uv, tr, te = gated_rescue.v5.load_ds()
    yv = cls[y]
    val_te, test_te = gated_rescue.split_te(te)
    active, exclusions = gated_rescue.active_candidate_families(
        [name for name, _ in rules]
    )
    counts_fn = gated_rescue._counts_row_fn(model, cls)
    conf_fns = fr._b0_conf_fns(gated_rescue, rules)
    b0 = WylyBlock(0, "B0")
    b0.rules, b0.conf_fns = list(rules), conf_fns
    _, _, tr_trace = fr.core_cover_sw_runner_traced(model, rules, ids, yv, cls, tr)
    test_trace = fr.assert_runner_trace_parity(model, rules, ids, yv, cls, test_te)
    val_trace = fr.assert_runner_trace_parity(model, rules, ids, yv, cls, val_te)
    val_conf = val_trace["conf"]
    taus = gated_rescue.tau_deciles(val_conf)

    tau_runs = []
    fitted = []
    for tau in taus:
        candidates, low_train, mining_log = gated_rescue.fit_candidate_pool(
            model, ids, yv, cls, uv, tr, tr_trace["conf"], tau, val_te, test_te, active
        )
        stack, admitted, marginals, agree = gated_rescue._run_gated_tau(
            rules, conf_fns, candidates, counts_fn, ids, yv, val_te, tau
        )
        tau_runs.append({
            "tau": tau,
            "val_agree": agree,
            "admitted": admitted,
            "marginals": marginals,
            "mined_train_rows": len(low_train),
            "mining_log": mining_log,
            "candidates": candidates,
            "stack": stack,
        })
        fitted.append((candidates, stack, admitted, marginals))
    best = min(range(len(taus)), key=lambda i: (-tau_runs[i]["val_agree"], tau_runs[i]["tau"]))
    chosen = tau_runs[best]
    chosen_candidates, stack, admitted, _ = fitted[best]
    flat, flat_admitted, _ = gated_rescue._run_flat(
        model, rules, conf_fns, chosen_candidates, counts_fn, ids, yv, val_te
    )
    base_pred, _ = b0.predict_cover(ids[test_te], counts_row_fn=counts_fn)
    flat_pred, _ = flat.predict_cover(ids[test_te], counts_row_fn=counts_fn)
    stack.forward(ids[test_te], counts_row_fn=counts_fn)
    gated_pred = stack.last_carried[-1].pred
    target = yv[test_te]
    base_correct = base_pred == target
    flat_correct = flat_pred == target
    gated_correct = gated_pred == target
    row_sets = fr.extract_row_sets(base_correct, gated_correct)
    recon_counts = {name: len(rows) for name, rows in row_sets.items()}
    log(f"reconstruction counts: {recon_counts}")

    # STOP-loud: three-way agreement
    for label, counts in (
        ("reconstruction", recon_counts),
        ("_stage1_and_2", summary_counts),
        ("frontier_rows.json", artifact_counts),
    ):
        if counts.get("gains") != 143 or counts.get("regressions") != 62:
            raise RuntimeError(
                f"STOP: {label} stage1 counts disagree with registered 143/62: {counts}"
            )
    if recon_counts != summary_counts:
        raise RuntimeError(
            f"STOP: recon counts {recon_counts} != summary {summary_counts}"
        )
    if recon_counts["gains"] != artifact_counts["gains"] or recon_counts[
        "regressions"
    ] != artifact_counts["regressions"]:
        raise RuntimeError(
            f"STOP: recon {recon_counts} != artifact {artifact_counts}"
        )

    test_consensus = fr._teacher_consensus(gated_rescue, test_te)
    test_axes = fr._axis_values(test_trace, test_consensus)
    val_consensus = fr._teacher_consensus(gated_rescue, val_te)
    val_axes = fr._axis_values(val_trace, val_consensus)

    # Feature matrix on the 205 gain/regression rows (label 1=gain, 0=regression)
    gain_idx = row_sets["gains"]
    reg_idx = row_sets["regressions"]
    pos = torch.cat([gain_idx, reg_idx])
    labels = torch.cat([
        torch.ones(len(gain_idx), dtype=torch.float64),
        torch.zeros(len(reg_idx), dtype=torch.float64),
    ])
    feat = torch.stack(
        [test_axes[name][pos].double() for name in fr.AXES], dim=1
    )  # (205, 4)

    return {
        "model": model,
        "rules": rules,
        "ids": ids,
        "yv": yv,
        "cls": cls,
        "uv": uv,
        "tr": tr,
        "te": te,
        "val_te": val_te,
        "test_te": test_te,
        "active": active,
        "exclusions": exclusions,
        "counts_fn": counts_fn,
        "conf_fns": conf_fns,
        "b0": b0,
        "tr_trace": tr_trace,
        "test_trace": test_trace,
        "val_trace": val_trace,
        "taus": taus,
        "tau_runs": tau_runs,
        "fitted": fitted,
        "best_idx": best,
        "chosen_tau": chosen["tau"],
        "chosen_admitted": list(admitted),
        "chosen_candidates": chosen_candidates,
        "chosen_stack": stack,
        "flat_block": flat,
        "flat_admitted": list(flat_admitted),
        "row_sets": row_sets,
        "recon_counts": recon_counts,
        "summary_counts": summary_counts,
        "artifact_counts": artifact_counts,
        "base_correct_test": base_correct,
        "flat_correct_test": flat_correct,
        "gated_correct_test": gated_correct,
        "base_pred_test": base_pred,
        "flat_pred_test": flat_pred,
        "gated_pred_test": gated_pred,
        "test_axes": test_axes,
        "val_axes": val_axes,
        "features_205": feat,
        "labels_205": labels,
        "pos_205": pos,
        "stage1_summary": stage1_summary,
    }


def _axis_matrix(axes: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
    return torch.stack([axes[n].double() for n in names], dim=1)


def fit_scorer_on_rows(
    axes_matrix: torch.Tensor,
    row_indices: torch.Tensor,
    labels_for_rows: torch.Tensor,
    *,
    seed: int = TORCH_FIT_SEED,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Fit z-score + logistic on given rows; return model, mean, std."""
    x = axes_matrix[row_indices]
    mean, std = zscore_fit(x)
    xz = zscore_transform(x, mean, std)
    model = fit_l2_logistic(xz, labels_for_rows, seed=seed)
    return model, mean, std


def score_all_rows(
    axes_matrix: torch.Tensor,
    model: dict[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return decision_scores(zscore_transform(axes_matrix, mean, std), model)


def half_a_oof_decile_grid(
    axes_matrix: torch.Tensor,
    half_a_gain_reg_idx: torch.Tensor,
    half_a_labels: torch.Tensor,
    *,
    n_folds: int = CV_N_FOLDS,
    fold_seed: int = HALF_A_OOF_SEED,
) -> tuple[list[float], torch.Tensor]:
    """Stratified k-fold OOF scores on half-A gain/reg rows → 10 decile thresholds.

    Design: OOF (not in-sample) within half-A, matching the registration's
    'half-A OOF-score deciles' literal reading.
    """
    x = axes_matrix[half_a_gain_reg_idx]
    oof = pooled_oof_scores(x, half_a_labels, n_folds=n_folds, fold_seed=fold_seed)
    return score_deciles(oof), oof


# =============================================================================
# Clause 2 val selection + half-B eval
# =============================================================================


def filtered_agree_and_regressions(
    final_pred: torch.Tensor,
    b0_pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, int]:
    tgt = target.to(device=final_pred.device)
    b0 = b0_pred.to(device=final_pred.device)
    correct = final_pred == tgt
    b0_correct = b0 == tgt
    agree = float(correct.float().mean())
    regressions = int((b0_correct & ~correct).sum())
    return agree, regressions


def clause2_val_sweep(
    ctx: dict[str, Any],
    half_a_model: dict[str, torch.Tensor],
    half_a_mean: torch.Tensor,
    half_a_std: torch.Tensor,
    threshold_grid: list[float],
    *,
    grid_id: int | None = None,
) -> list[dict[str, Any]]:
    """Sweep all (tau, threshold) on val_te; return table of points with admissibility."""
    from experiments import campaign_wikitext_gated_rescue as gated_rescue

    # Capture grid identity for fixedness tests (object/call id, not re-derived per tau).
    _ = grid_id
    ids, yv = ctx["ids"], ctx["yv"]
    val_te = ctx["val_te"]
    counts_fn = ctx["counts_fn"]
    rules, conf_fns = ctx["rules"], ctx["conf_fns"]
    b0 = ctx["b0"]
    val_axes_mat = _axis_matrix(ctx["val_axes"], __import__(
        "experiments.campaign_frontier_rows", fromlist=["AXES"]
    ).AXES)
    disc_val = score_all_rows(val_axes_mat, half_a_model, half_a_mean, half_a_std)
    target = yv[val_te]
    b0_pred_val, _ = b0.predict_cover(ids[val_te], counts_row_fn=counts_fn)

    table: list[dict[str, Any]] = []
    # Use the same threshold_grid object for every tau (fixedness).
    fixed_grid = threshold_grid
    for run in ctx["tau_runs"]:
        tau = run["tau"]
        candidates = run["candidates"]
        stack = run["stack"]
        # Re-run flat at this tau's candidate pool (as _stage1_and_2 does per chosen;
        # we recompute flat per tau for fair (ii)/(iv) analogs).
        flat_block, _, _ = gated_rescue._run_flat(
            ctx["model"], rules, conf_fns, candidates, counts_fn, ids, yv, val_te
        )
        flat_pred_val, _ = flat_block.predict_cover(ids[val_te], counts_row_fn=counts_fn)
        flat_agree = float((flat_pred_val == target).float().mean())
        flat_regs = int(((b0_pred_val == target) & (flat_pred_val != target)).sum())

        stack.forward(ids[val_te], counts_row_fn=counts_fn)
        s0, s1 = stack.last_states[0], stack.last_states[1]
        assert s0.pred is not None and s1.pred is not None and s1.conf is not None

        for thr in fixed_grid:
            final = discriminator_filter_claim(
                s0.pred, s1.pred, s1.conf, disc_val, tau=tau, threshold=thr
            )
            filt_agree, filt_regs = filtered_agree_and_regressions(
                final, b0_pred_val, target
            )
            val_gain = filt_agree - flat_agree
            admissible = (val_gain >= VAL_GAIN_ADMIT) and (filt_regs <= flat_regs)
            table.append({
                "tau": float(tau),
                "threshold": float(thr),
                "val_filtered_agree": filt_agree,
                "val_flat_agree": flat_agree,
                "val_gain": val_gain,
                "val_filtered_regressions": filt_regs,
                "val_flat_regressions": flat_regs,
                "admissible": admissible,
            })
    return table


def select_best_admissible(table: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in table if row["admissible"]]
    if not passing:
        return None
    return max(passing, key=lambda r: (r["val_gain"], -r["tau"], -r["threshold"]))


def evaluate_half_b(
    ctx: dict[str, Any],
    half_b_idx: torch.Tensor,
    half_a_model: dict[str, torch.Tensor],
    half_a_mean: torch.Tensor,
    half_a_std: torch.Tensor,
    tau: float,
    threshold: float,
) -> dict[str, Any]:
    """#81 four-clause contrast on half-B only; flat regressions recomputed on half-B."""
    from experiments import campaign_wikitext_gated_rescue as gated_rescue
    from experiments.campaign_frontier_rows import AXES

    ids, yv = ctx["ids"], ctx["yv"]
    test_te = ctx["test_te"]
    counts_fn = ctx["counts_fn"]
    rules, conf_fns = ctx["rules"], ctx["conf_fns"]
    b0 = ctx["b0"]

    # Positions in test_te space
    hb = half_b_idx.long().cpu()
    test_ids = ids[test_te][hb]
    target = yv[test_te][hb]

    # Find the tau run matching selected tau
    run = min(ctx["tau_runs"], key=lambda r: abs(r["tau"] - tau))
    if abs(run["tau"] - tau) > 1e-12:
        raise RuntimeError(f"selected tau {tau} not found in tau_runs")
    candidates = run["candidates"]
    stack = run["stack"]
    gated_admitted = list(run["admitted"])

    flat_block, flat_admitted, _ = gated_rescue._run_flat(
        ctx["model"], rules, conf_fns, candidates, counts_fn, ids, yv, ctx["val_te"]
    )
    # Predictions on half-B subset of test_te
    b0_pred, _ = b0.predict_cover(test_ids, counts_row_fn=counts_fn)
    flat_pred, _ = flat_block.predict_cover(test_ids, counts_row_fn=counts_fn)
    stack.forward(test_ids, counts_row_fn=counts_fn)
    s0, s1 = stack.last_states[0], stack.last_states[1]

    test_axes_mat = _axis_matrix(ctx["test_axes"], AXES)
    disc_all = score_all_rows(test_axes_mat, half_a_model, half_a_mean, half_a_std)
    disc_hb = disc_all[hb]

    filt_pred = discriminator_filter_claim(
        s0.pred, s1.pred, s1.conf, disc_hb, tau=run["tau"], threshold=threshold
    )

    b0_correct = b0_pred == target
    flat_correct = flat_pred == target
    filt_correct = filt_pred == target
    flat_agree = float(flat_correct.float().mean())
    filt_agree = float(filt_correct.float().mean())
    # flat regressions vs B0 on half-B — MUST be recomputed here, not copied from val_te
    flat_regs_half_b = int((b0_correct & ~flat_correct).sum())
    filt_regs_half_b = int((b0_correct & ~filt_correct).sum())
    b = int((flat_correct & ~filt_correct).sum())
    c = int((~flat_correct & filt_correct).sum())
    p_disc = gated_rescue.exact_discordant_p(b, c)

    flat_families = {
        gated_rescue._candidate_family_for_rule(n) or n for n in flat_admitted
    }
    gated_families = {
        gated_rescue._candidate_family_for_rule(n) or n for n in gated_admitted
    }
    # families gated admits that flat declines
    flat_declined = [
        fam for fam in gated_families if fam not in flat_families and fam is not None
    ]
    # also count by rule name prefix if family map fails
    clause_i = len(flat_declined) >= 1 or (
        len(set(gated_admitted) - set(flat_admitted)) >= 1
        and any(
            g not in flat_admitted for g in gated_admitted
        )
    )
    # More precise: (i) gated admits >=1 family flat declines
    gated_only = []
    for name in gated_admitted:
        fam = gated_rescue._candidate_family_for_rule(name) or name
        flat_fams = {
            gated_rescue._candidate_family_for_rule(n) or n for n in flat_admitted
        }
        if fam not in flat_fams:
            gated_only.append(name)
    clause_i = len(gated_only) >= 1
    clause_ii = filt_agree >= flat_agree + HALF_B_AGREE_DELTA
    clause_iii = p_disc < 0.05
    clause_iv = filt_regs_half_b <= flat_regs_half_b
    passed = clause_i and clause_ii and clause_iii and clause_iv

    return {
        "n_half_b": int(len(hb)),
        "tau": float(run["tau"]),
        "threshold": float(threshold),
        "flat_admitted": list(flat_admitted),
        "gated_admitted": list(gated_admitted),
        "gated_admits_families_flat_declines": gated_only,
        "clause_i": clause_i,
        "clause_ii": clause_ii,
        "clause_iii": clause_iii,
        "clause_iv": clause_iv,
        "flat_agree": flat_agree,
        "filtered_agree": filt_agree,
        "agree_delta": filt_agree - flat_agree,
        "discordant_b": b,
        "discordant_c": c,
        "exact_binomial_p": p_disc,
        "flat_regressions_half_b": flat_regs_half_b,
        "filtered_regressions_half_b": filt_regs_half_b,
        "pass": passed,
        "flat_regs_source": "recomputed_on_half_b",
    }


# =============================================================================
# #86 grid replay (subprocess; val_te only)
# =============================================================================


def _artifact_86_clearing_points() -> list[tuple[int, float]]:
    from experiments import campaign_frames_at_scale as frames

    art = json.loads(FRAMES_ARTIFACT.read_text())
    points = [
        (int(row["support_gate"]), float(row["interaction_gate"]))
        for row in art["scales"]["2.8b"]["grid"]
        if row["marginal"] >= frames.ADMIT_THRESH
    ]
    if len(points) != 5:
        raise RuntimeError(
            f"STOP: expected 5 judge-clearing #86 points, got {len(points)}: {points}"
        )
    return points


def _run_86_child(
    destination: Path,
    half_a_model: dict[str, Any],
    half_a_mean: list[float],
    half_a_std: list[float],
    threshold_grid: list[float],
) -> int:
    """Child process: regenerate 2.8b, mine clearing points, filter on val_te only."""
    # Import order: frames binds 2.8b before gated helpers (same as frontier stage3).
    from experiments import campaign_frames_at_scale as frames_at_scale
    from experiments import campaign_frontier_rows as fr
    from experiments import campaign_wikitext_gated_rescue as gated_rescue

    if os.environ.get("WYLY_SEED") != "0":
        raise AssertionError("#86 replay requires WYLY_SEED=0")

    historical = _artifact_86_clearing_points()
    model, captured_rules, _ = frames_at_scale._regenerate()
    ids, y, cls, uv, tr, te = frames_at_scale.v5.load_ds()
    yv = cls[y]
    val_te, test_te = frames_at_scale.split_te(te)
    # #86's test_te stays untouched — never score it.
    _ = test_te
    fit = frames_at_scale._fit_region(tr)
    baseline_rules = [
        rule for rule in captured_rules if not rule[0].startswith("mined frames")
    ]

    weight = torch.tensor(half_a_model["weight"], dtype=torch.float64)
    bias = torch.tensor(half_a_model["bias"], dtype=torch.float64)
    mean = torch.tensor(half_a_mean, dtype=torch.float64)
    std = torch.tensor(half_a_std, dtype=torch.float64)
    model_dict = {"weight": weight, "bias": bias}

    base_trace = fr.assert_runner_trace_parity(
        model, baseline_rules, ids, yv, cls, val_te
    )
    # Teacher consensus via gated_rescue helpers (same as #87 stage3 path).
    axes = fr._axis_values(base_trace, fr._teacher_consensus(gated_rescue, val_te))
    axes_mat = _axis_matrix(axes, fr.AXES)
    disc = score_all_rows(axes_mat, model_dict, mean, std)

    target = yv[val_te]
    rows_out: list[dict[str, Any]] = []
    any_rescue = False
    fixed_grid = list(threshold_grid)  # fixed object for this child

    for support, interaction in historical:
        mined, _ = frames_at_scale._mine_once(
            model, baseline_rules, ids, yv, cls, uv, fit, support, interaction
        )
        scored = frames_at_scale._score_candidate(
            model, baseline_rules, mined, ids, yv, cls, val_te
        )
        # Sanity: point should clear judge bar on val (historical did).
        base_pred = scored["baseline_pred"]
        cand_pred = scored["candidate_pred"]
        for thr in fixed_grid:
            filt_pred = frame_filter_claim(
                base_pred, cand_pred, disc, threshold=thr
            )
            base_correct = base_pred == target
            filt_correct = filt_pred == target
            cand_correct = cand_pred == target
            filt_agree = float(filt_correct.float().mean())
            base_agree = float(base_correct.float().mean())
            marg = filt_agree - base_agree
            regs = int((base_correct & ~filt_correct).sum())
            # discordant for binomial: baseline vs filtered (val-level)
            # b = baseline correct & filtered wrong; c = baseline wrong & filtered correct
            b_dc = int((base_correct & ~filt_correct).sum())
            c_dc = int((~base_correct & filt_correct).sum())
            p_val = gated_rescue.exact_discordant_p(b_dc, c_dc)
            rescue = (
                marg >= RESCUE_86_MARGINAL
                and p_val < 0.05
                and regs == 0
            )
            any_rescue = any_rescue or rescue
            rows_out.append({
                "support_gate": support,
                "interaction_gate": interaction,
                "threshold": float(thr),
                "val_baseline_agree": base_agree,
                "val_filtered_agree": filt_agree,
                "val_unfiltered_candidate_agree": float(cand_correct.float().mean()),
                "val_marginal": marg,
                "val_regressions": regs,
                "val_discordant_b": b_dc,
                "val_discordant_c": c_dc,
                "val_exact_binomial_p": p_val,
                "val_level_not_test": True,
                "rescue": rescue,
            })

    report = {
        "historical_clearing_points": historical,
        "n_clearing_points": len(historical),
        "threshold_grid": fixed_grid,
        "n_sweep_rows": len(rows_out),
        "sweep": rows_out,
        "rescue": any_rescue,
        "honesty": (
            "#86's test_te stays untouched; the rescue claim for #86 is val-level, "
            "flagged as such"
        ),
    }
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


def run_86_subprocess(
    half_a_model: dict[str, torch.Tensor],
    half_a_mean: torch.Tensor,
    half_a_std: torch.Tensor,
    threshold_grid: list[float],
) -> dict[str, Any]:
    env = dict(os.environ)
    env.update({
        "WYLY_TAG": "pythia2.8b",
        "WYLY_DS": "wikitext",
        "WYLY_LIB": "mined",
        "WYLY_JUDGE": "cover",
        "WYLY_ONLINE": "1",
        "WYLY_COVER": "sw",
        "WYLY_SEED": "0",
    })
    for key in (
        "WYLY_CONCEPTS", "WYLY_POINTER", "WYLY_TPOINTER", "WYLY_DX", "WYLY_CX", "WYLY_FOLDS"
    ):
        env.pop(key, None)
    payload = {
        "weight": half_a_model["weight"].detach().cpu().tolist(),
        "bias": float(half_a_model["bias"].detach().cpu()),
        "mean": half_a_mean.detach().cpu().tolist(),
        "std": half_a_std.detach().cpu().tolist(),
        "threshold_grid": list(threshold_grid),
    }
    with tempfile.TemporaryDirectory(prefix="composite-disc-86-") as temporary:
        dest = Path(temporary) / "replay86.json"
        model_path = Path(temporary) / "half_a_model.json"
        model_path.write_text(json.dumps(payload))
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--frames86-child",
                str(dest),
                "--frames86-model",
                str(model_path),
            ],
            env=env,
            check=False,
        )
        if completed.returncode != 0 or not dest.exists():
            raise RuntimeError(
                f"#86 replay subprocess failed with exit {completed.returncode}"
            )
        return json.loads(dest.read_text())


# =============================================================================
# Main orchestration
# =============================================================================


def _plateau_wording(
    clause1_pass: bool,
    clause1_auc: float,
    clause1_p: float,
    clause2_pass: bool,
    clause2_detail: str,
) -> str:
    return PLATEAU_TEMPLATE.format(
        clause1_auc=clause1_auc,
        clause1_pass=clause1_pass,
        clause1_p=clause1_p,
        clause2_detail=clause2_detail,
    )


def run_campaign() -> dict[str, Any]:
    t0 = time.perf_counter()
    log("REGISTERED RULES (verbatim): " + REGISTERED_RULES)
    log(
        f"sklearn: ABSENT — torch L2-logistic path "
        f"(Adam lr={LOGISTIC_LR}, steps={LOGISTIC_STEPS}, C={LOGISTIC_C})"
    )
    log(
        "Loss formula: mean(BCEWithLogits) + 0.5*||w||^2 "
        "(sklearn C=1; intercept unpenalized)"
    )
    log(
        "Stage-1 reconstruction strategy: (a) _stage1_and_2 summary + independent "
        "re-run of the same call sequence; STOP if counts disagree with each other "
        "or data/frontier_rows.json (143/62)."
    )
    log(
        "Permutation null design: single Generator(seed=42) drives all shuffles; "
        "each shuffle = one pooled-OOF 5-fold AUC with fold-generation seed=0 held "
        "fixed (labels only shuffled). Not 20 repeats × N shuffles."
    )
    log(
        "Half-A threshold grid: stratified 5-fold OOF scores within half-A "
        "gain/regression rows (seed=0) → 10 decile thresholds (not in-sample)."
    )

    ctx = reconstruct_stage1_rows()
    feat = ctx["features_205"]
    labels = ctx["labels_205"]
    log(f"Clause-1 feature matrix shape: {tuple(feat.shape)} (expect (205, 4))")
    if feat.shape != (205, 4):
        raise RuntimeError(f"STOP: expected (205, 4) features, got {tuple(feat.shape)}")

    # ----- CLAUSE 1 -----
    log("")
    log(CLAUSE_1_VERBATIM)
    log("Running stratified 5-fold × 20 repeats (seeds 0..19)...")
    t_cv = time.perf_counter()
    cv_dist = cv_auc_distribution(feat, labels)
    log(
        f"CV pooled-OOF AUC: median={cv_dist['median']:.6f} "
        f"min={cv_dist['min']:.6f} max={cv_dist['max']:.6f} "
        f"({time.perf_counter() - t_cv:.1f}s)"
    )
    for i, a in enumerate(cv_dist["aucs"]):
        log(f"  repeat seed={i}: AUC={a:.6f}")

    n_shuffles = PERM_N_SHUFFLES_TARGET
    perm_reduction_reason = None
    log(f"Permutation null: targeting {n_shuffles} shuffles (seed={PERM_SEED})...")
    t_perm = time.perf_counter()
    # Time one shuffle to decide whether to reduce
    t_one = time.perf_counter()
    _ = permutation_null_aucs(feat, labels, n_shuffles=1)
    one_s = time.perf_counter() - t_one
    est_total = one_s * n_shuffles
    log(f"  one-shuffle wall ≈ {one_s:.2f}s; est. {n_shuffles} ≈ {est_total:.0f}s")
    if est_total > 1800:  # >30 min → drop to 200
        n_shuffles = 200
        perm_reduction_reason = (
            f"one shuffle took {one_s:.2f}s; 1000 would be ~{est_total:.0f}s "
            f"(>1800s threshold); reduced to 200 minimum per registration"
        )
        log(f"  REDUCING shuffles to {n_shuffles}: {perm_reduction_reason}")
    null_aucs = permutation_null_aucs(feat, labels, n_shuffles=n_shuffles)
    # first shuffle already counted in timing probe; we recompute full set cleanly above
    perm_p = empirical_perm_p(cv_dist["median"], null_aucs)
    null_t = torch.tensor(null_aucs, dtype=torch.float64)
    log(
        f"Null AUC: median={float(null_t.median()):.6f} "
        f"min={float(null_t.min()):.6f} max={float(null_t.max()):.6f} "
        f"({time.perf_counter() - t_perm:.1f}s, n={n_shuffles})"
    )
    log(f"CLAUSE 1 empirical two-sided p = {perm_p:.6g}")
    clause1_pass = (cv_dist["median"] >= 0.70) and (perm_p < 0.05)
    log(f"CLAUSE 1 PASS = {clause1_pass}")

    # ----- half split + half-A model -----
    from experiments.campaign_frontier_rows import AXES

    test_axes_mat = _axis_matrix(ctx["test_axes"], AXES)
    half_a, half_b = split_half_ab(
        len(ctx["test_te"]), ctx["base_correct_test"], seed=HALF_SPLIT_SEED
    )
    log("")
    log(
        f"test_te 50/50: |A|={len(half_a)} |B|={len(half_b)} "
        f"sum={len(half_a) + len(half_b)} (expect {len(ctx['test_te'])})"
    )
    gain_set = set(ctx["row_sets"]["gains"].tolist())
    reg_set = set(ctx["row_sets"]["regressions"].tolist())
    ha_list = half_a.tolist()
    hb_list = half_b.tolist()
    ha_gains = torch.tensor([i for i in ha_list if i in gain_set], dtype=torch.long)
    ha_regs = torch.tensor([i for i in ha_list if i in reg_set], dtype=torch.long)
    hb_gains = torch.tensor([i for i in hb_list if i in gain_set], dtype=torch.long)
    hb_regs = torch.tensor([i for i in hb_list if i in reg_set], dtype=torch.long)
    log(
        f"half-A gain/reg: {len(ha_gains)}/{len(ha_regs)}; "
        f"half-B gain/reg: {len(hb_gains)}/{len(hb_regs)}"
    )
    ha_gr = torch.cat([ha_gains, ha_regs])
    ha_labels = torch.cat([
        torch.ones(len(ha_gains), dtype=torch.float64),
        torch.zeros(len(ha_regs), dtype=torch.float64),
    ])
    if len(ha_gains) == 0 or len(ha_regs) == 0:
        raise RuntimeError("STOP: half-A missing a class for discriminator fit")

    threshold_grid, ha_oof = half_a_oof_decile_grid(
        test_axes_mat, ha_gr, ha_labels
    )
    log(f"half-A OOF-score decile grid (10): {threshold_grid}")
    half_a_model, half_a_mean, half_a_std = fit_scorer_on_rows(
        test_axes_mat, ha_gr, ha_labels, seed=TORCH_FIT_SEED
    )
    # Freeze grid as a single list object for fixedness
    threshold_grid_frozen = list(threshold_grid)

    # ----- CLAUSE 2 -----
    log("")
    log(CLAUSE_2_VERBATIM)
    log("Val_te selection sweep over 10 taus × 10 thresholds...")
    t_sw = time.perf_counter()
    val_table = clause2_val_sweep(
        ctx,
        half_a_model,
        half_a_mean,
        half_a_std,
        threshold_grid_frozen,
        grid_id=id(threshold_grid_frozen),
    )
    n_adm = sum(1 for r in val_table if r["admissible"])
    log(
        f"val sweep: {len(val_table)} points, {n_adm} admissible "
        f"({time.perf_counter() - t_sw:.1f}s)"
    )
    selected = select_best_admissible(val_table)
    half_b_eval: dict[str, Any] | None = None
    if selected is None:
        log(NO_ADMISSIBLE_MSG)
        clause2_pass = False
        clause2_detail = (
            f"found 0 admissible val-level (tau, threshold) points "
            f"(of {len(val_table)}); half-B eval skipped; Clause 2 FAILED"
        )
    else:
        log(
            f"selected operating point: tau={selected['tau']:.6f} "
            f"threshold={selected['threshold']:.6f} "
            f"val_gain={selected['val_gain']:.6f} "
            f"val_regs={selected['val_filtered_regressions']}"
        )
        half_b_eval = evaluate_half_b(
            ctx,
            half_b,
            half_a_model,
            half_a_mean,
            half_a_std,
            selected["tau"],
            selected["threshold"],
        )
        log("half-B four-clause eval: " + json.dumps({
            k: half_b_eval[k]
            for k in (
                "clause_i", "clause_ii", "clause_iii", "clause_iv",
                "flat_agree", "filtered_agree", "agree_delta",
                "discordant_b", "discordant_c", "exact_binomial_p",
                "flat_regressions_half_b", "filtered_regressions_half_b", "pass",
            )
        }, sort_keys=True))
        clause2_pass = bool(half_b_eval["pass"])
        clause2_detail = (
            f"found {n_adm} admissible val-level points; selected val_gain="
            f"{selected['val_gain']:.6f}; half-B clauses "
            f"i={half_b_eval['clause_i']} ii={half_b_eval['clause_ii']} "
            f"iii={half_b_eval['clause_iii']} iv={half_b_eval['clause_iv']} "
            f"(pass={clause2_pass}); half-B agree_delta="
            f"{half_b_eval['agree_delta']:.6f}, "
            f"filt_regs={half_b_eval['filtered_regressions_half_b']}, "
            f"flat_regs={half_b_eval['flat_regressions_half_b']}"
        )
    log(f"CLAUSE 2 PASS = {clause2_pass}")

    # ----- #86 replay (always) -----
    log("")
    log("=== #86 GRID REPLAY (val_te only; test_te untouched) ===")
    log(
        "#86's test_te stays untouched; the rescue claim for #86 is val-level, "
        "flagged as such"
    )
    t_86 = time.perf_counter()
    replay86 = run_86_subprocess(
        half_a_model, half_a_mean, half_a_std, threshold_grid_frozen
    )
    log(
        f"#86 rescue={replay86['rescue']} "
        f"({replay86['n_sweep_rows']} rows; {time.perf_counter() - t_86:.1f}s)"
    )
    n_rescue_rows = sum(1 for r in replay86["sweep"] if r["rescue"])
    log(f"#86 rescue-true rows: {n_rescue_rows} / {replay86['n_sweep_rows']}")

    success = clause1_pass and clause2_pass
    if success:
        verdict = "SUCCESS"
        log("VERDICT: SUCCESS (CLAUSE 1 and CLAUSE 2 both passed)")
    else:
        verdict = _plateau_wording(
            clause1_pass, cv_dist["median"], perm_p, clause2_pass, clause2_detail
        )
        log("VERDICT (plateau wording):")
        log(verdict)

    log("")
    log("Two-trainings note: " + TWO_TRAININGS_NOTE)

    honesty = {
        "one_shot_thread_ends": True,
        "features_frozen_from_87": list(AXES),
        "seeds": {
            "WYLY_SEED": 0,
            "cv_repeat_seeds": CV_REPEAT_SEEDS,
            "permutation_generator_seed": PERM_SEED,
            "permutation_fold_seed": PERM_FOLD_SEED,
            "half_split_seed": HALF_SPLIT_SEED,
            "half_a_oof_seed": HALF_A_OOF_SEED,
            "torch_fit_seed": TORCH_FIT_SEED,
        },
        "sklearn_absent_torch_path": True,
        "logistic_hyperparams": {
            "optimizer": LOGISTIC_OPTIM,
            "lr": LOGISTIC_LR,
            "steps": LOGISTIC_STEPS,
            "C": LOGISTIC_C,
            "loss": (
                "mean(BCEWithLogits(x@w+b, y)) + (1/(2*C))*||w||^2 with C=1; "
                "intercept unpenalized"
            ),
        },
        "permutation_null_design": (
            "single Generator(seed=42) for all label shuffles; each shuffle one "
            "pooled-OOF 5-fold AUC with fold_seed=0 fixed"
        ),
        "half_a_decile_design": (
            "stratified 5-fold OOF scores within half-A gain/regression rows "
            "(not in-sample)"
        ),
        "stage1_reconstruction_strategy": "a",
        "frames86_test_te_untouched": True,
        "frames86_rescue_is_val_level": True,
        "frames86_val_level_flag_verbatim": (
            "#86's test_te stays untouched; the rescue claim for #86 is val-level, "
            "flagged as such"
        ),
        "no_preregistered_plateau_text_found_in_repo": True,
        "plateau_wording_constructed_following_82_83_86_87_pattern": True,
        "two_trainings_note": TWO_TRAININGS_NOTE,
        "permutation_shuffles_used": n_shuffles,
        "permutation_shuffle_reduction_reason": perm_reduction_reason,
    }

    report: dict[str, Any] = {
        "registered_rules_verbatim": REGISTERED_RULES,
        "clause1": {
            "verbatim": CLAUSE_1_VERBATIM,
            "cv_auc": {
                "median": cv_dist["median"],
                "min": cv_dist["min"],
                "max": cv_dist["max"],
                "aucs": cv_dist["aucs"],
                "n_repeats": len(cv_dist["aucs"]),
                "n_folds": CV_N_FOLDS,
            },
            "permutation_shuffles_used": n_shuffles,
            "permutation_shuffle_reduction_reason": perm_reduction_reason,
            "permutation_seed": PERM_SEED,
            "null_auc_median": float(null_t.median()),
            "null_auc_min": float(null_t.min()),
            "null_auc_max": float(null_t.max()),
            "empirical_p_two_sided": perm_p,
            "pass": clause1_pass,
            "pass_rule": "median_auc >= 0.70 AND empirical_p < 0.05",
        },
        "half_split": {
            "seed": HALF_SPLIT_SEED,
            "n_test_te": len(ctx["test_te"]),
            "n_half_a": len(half_a),
            "n_half_b": len(half_b),
            "half_a_gains": len(ha_gains),
            "half_a_regressions": len(ha_regs),
            "half_b_gains": len(hb_gains),
            "half_b_regressions": len(hb_regs),
            "stratify": "B0-correctness on test_te",
        },
        "half_a_model": {
            "weight": half_a_model["weight"].tolist(),
            "bias": float(half_a_model["bias"]),
            "z_mean": half_a_mean.tolist(),
            "z_std": half_a_std.tolist(),
            "oof_decile_grid": threshold_grid_frozen,
            "oof_design": "stratified_5fold_within_half_a",
        },
        "clause2": {
            "verbatim": CLAUSE_2_VERBATIM,
            "val_selection_sweep": val_table,
            "n_admissible_val_points": n_adm,
            "selected": selected,
            "no_admissible_message_emitted": selected is None,
            "half_b_eval": half_b_eval,
            "pass": clause2_pass,
        },
        "frames86_replay": replay86,
        "verdict": verdict,
        "success": success,
        "stage1_counts": ctx["recon_counts"],
        "chosen_tau_stage1": ctx["chosen_tau"],
        "honesty": honesty,
        "wall_seconds": time.perf_counter() - t0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    log(f"JSON: {OUTPUT}")
    log(f"wall_seconds: {report['wall_seconds']:.1f}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames86-child", type=Path)
    parser.add_argument("--frames86-model", type=Path)
    args = parser.parse_args(argv)
    if args.frames86_child is not None:
        if args.frames86_model is None:
            raise SystemExit("--frames86-model required with --frames86-child")
        payload = json.loads(args.frames86_model.read_text())
        half_a_model = {
            "weight": payload["weight"],
            "bias": payload["bias"],
        }
        return _run_86_child(
            args.frames86_child,
            half_a_model,
            payload["mean"],
            payload["std"],
            payload["threshold_grid"],
        )
    run_campaign()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
