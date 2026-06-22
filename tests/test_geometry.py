"""Correctness tests -- pin the three bugs found in the starter so they can't return."""

from __future__ import annotations

import torch

from pil.geometry import (
    frame_potential,
    incidences,
    logits_from_incidences,
    margin_to_worst,
    normalize_rows,
    participation_ratio,
    welch_bound,
)
from pil.learner import PILConfig, ProjectiveIncidenceLearner, create_synthetic_problem
from pil.synthetic import create_clustered_problem, within_cluster_cosine


def test_margin_batched_matches_bruteforce():
    """Bug #1: batched-tensor target must give the true per-row margin (starter mis-indexed)."""
    torch.manual_seed(0)
    L = torch.randn(7, 11)
    target = torch.randint(0, 11, (7,))
    got = margin_to_worst(L, target)
    for i in range(7):
        t = int(target[i])
        wrong = torch.cat([L[i, :t], L[i, t + 1 :]])
        assert torch.allclose(got[i], L[i, t] - wrong.max())


def test_margin_scalar_int_still_works():
    L = torch.randn(4, 6)
    got = margin_to_worst(L, 2)
    for i in range(4):
        wrong = torch.cat([L[i, :2], L[i, 3:]])
        assert torch.allclose(got[i], L[i, 2] - wrong.max())


def test_batched_logits_preserve_batch():
    """Bug #2: 3-D (B,J,dim) sources must yield (B,V) logits, not a single pooled vector."""
    B, J, dim, V = 5, 9, 8, 13
    d = torch.randn(B, J, dim)
    U = torch.randn(V, dim)
    c = incidences(d, U)
    assert c.shape == (B, J, V)
    L = logits_from_incidences(c)
    assert L.shape == (B, V)
    # equals the per-example 2-D path
    for b in range(B):
        assert torch.allclose(L[b], logits_from_incidences(incidences(d[b], U)), atol=1e-5)


def test_frame_potential_nonneg_and_welch_floor():
    """Bug #3: the Gram objective must not penalize frame norms; off-diagonal only, >= Welch."""
    torch.manual_seed(0)
    U = normalize_rows(torch.randn(40, 8))
    fp = frame_potential(U).item()
    wb = welch_bound(40, 8)
    assert fp >= 0.0
    # a random over-complete frame sits above its Welch floor
    assert fp >= wb - 1e-6
    # scaling rows must NOT change the cosine frame potential (norm-decoupled)
    fp_scaled = frame_potential(3.0 * U).item()
    assert abs(fp - fp_scaled) < 1e-5


def test_participation_ratio_bounds():
    one_hot = torch.tensor([0.0, 5.0, 0.0, 0.0])
    assert abs(participation_ratio(one_hot).item() - 1.0) < 1e-4
    flat = torch.ones(10)
    assert abs(participation_ratio(flat).item() - 10.0) < 1e-3


def test_clustered_problem_plants_synonymy():
    """The hard generator must actually plant high within-cluster synonymy and shapes."""
    cfg = PILConfig(dim=24, n_propositions=96, n_sources_per_step=20, device="cpu", seed=0)
    src, tgt, U_gt, cl = create_clustered_problem(
        cfg, n_examples=64, n_clusters=12, cluster_spread=0.10
    )
    assert src.shape == (64, 20, 24)
    assert tgt.shape == (64,) and U_gt.shape == (96, 24) and cl.shape == (96,)
    wc = within_cluster_cosine(U_gt, cl)
    # expected ~ 1/(1 + spread^2 * dim) = 1/(1+0.01*24) ~ 0.81; well above a random pair
    assert wc > 0.6, f"within-cluster cosine too low: {wc}"


def test_training_step_reduces_loss():
    cfg = PILConfig(dim=16, n_propositions=24, n_sources_per_step=12, device="cpu", seed=0)
    model = ProjectiveIncidenceLearner(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    src, tgt, _ = create_synthetic_problem(cfg, n_examples=64, sparsity=0.2, noise=0.05)
    first = model.training_step(src[:16], tgt[:16], opt)["loss"]
    for _ in range(200):
        model.training_step(src[:16], tgt[:16], opt)
    last = model.training_step(src[:16], tgt[:16], opt)["loss"]
    assert last < first
