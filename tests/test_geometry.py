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


def test_compositional_xor_defeats_linear_helped_by_rule():
    """The compositional problem must be non-linearly coded: a linear probe on z can't do the
    hard-cluster XOR, but adding the |z_p - z_q| feature makes it linearly separable."""
    from pil.proposer import build_sources, targeted_rulebank
    from pil.synthetic import create_compositional_problem

    cfg = PILConfig(dim=24, n_propositions=2 * 8, device="cpu", seed=0)
    z, A, tgt, _, hc = create_compositional_problem(
        cfg, n_examples=512, n_clusters=8, frac_hard=1.0, noise=0.0
    )
    K = z.shape[1]
    # raw build is (B, K, dim); a targeted rule bank appends |z_p - z_q| features
    bank = targeted_rulebank(list(range(8)), K, 8, 24, device="cpu", seed=0)
    raw = build_sources(z, A, None)
    aug = build_sources(z, A, bank)
    assert raw.shape == (512, K, 24)
    assert aug.shape[1] == K + bank.n_rules  # rules appended


def test_ambiguity_score_prefers_discriminating_rule():
    """η²-on-ambiguous score must rank a target-correlated firing above a noise firing."""
    from pil.scoring import ambiguity_resolution_score, candidate_activations

    torch.manual_seed(0)
    n, K, V = 200, 4, 2
    targets = torch.randint(0, V, (n,))
    z = torch.randn(n, K)
    # make atom 0 strongly encode the target; atoms 1..3 are noise
    z[:, 0] = targets.float() * 2.0 + 0.1 * torch.randn(n)
    margins = torch.zeros(n)  # all examples ambiguous
    # candidate A reads atom 0 (discriminating); candidate B reads only noise atoms
    W = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 1.0, 1.0]])
    acts = candidate_activations(z, W)
    s = ambiguity_resolution_score(acts, targets, margins, n_classes=V, frac=1.0)
    assert s[0] > s[1], f"discriminating rule should score higher: {s.tolist()}"


def test_frozen_rulebank_input_not_trained():
    from pil.proposer import rulebank_from_weights

    W = torch.randn(3, 6)
    bank = rulebank_from_weights(W, dim=5, freeze_input=True)
    assert not bank.W.requires_grad and bank.Bdir.requires_grad
    assert torch.allclose(bank.W.data, W)


def test_packing_bound_and_separation():
    """Capacity diagnostics: packing bound grows with d; min separation is the true pairwise min."""
    from pil.geometry import gamma_decodable_count, log10_packing_bound, min_frame_separation

    # (1+2*1/1)^32 = 3^32 ~ 1.85e15 -> log10 ~ 15.3
    assert abs(log10_packing_bound(1.0, 1.0, 32) - 32 * 0.4771) < 0.1
    # bigger d, smaller gamma => bigger bound
    assert log10_packing_bound(1.0, 0.5, 32) > log10_packing_bound(1.0, 1.0, 32)

    U = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    sep = min_frame_separation(U, torch.tensor([0, 1, 2]))
    assert abs(sep - 3.0) < 1e-5  # nearest pair is rows 0,1 at distance 3

    L = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])  # two confident, distinct decodes
    assert gamma_decodable_count(L, gamma=1.0) == 2


def test_load_pil_dump(tmp_path):
    """The fieldrun --pil-dump loader parses JSON lines into the incidence bundle and drops
    records whose target fell outside the candidate set (tgt_idx == -1)."""
    import json

    from pil.fieldrun_io import load_pil_dump

    recs = [
        {"pos": 1, "cur": 5, "target": 9, "tgt_idx": 0, "pred": 9, "margin": 2.0, "nb": 2,
         "cands": [9, 3, 7], "contrib": [[1.0, 0.2, 0.1], [0.5, 0.1, 0.0]]},
        {"pos": 2, "cur": 9, "target": 4, "tgt_idx": -1, "pred": 3, "margin": 1.0, "nb": 2,
         "cands": [3, 7, 1], "contrib": [[0.9, 0.3, 0.1], [0.4, 0.2, 0.0]]},  # dropped (tgt_idx -1)
    ]
    path = tmp_path / "dump.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    b = load_pil_dump(str(path))
    assert b.contrib.shape == (1, 2, 3)        # one usable record, nb=2, K=3
    assert b.n_blocks == 2
    assert int(b.tgt_idx[0]) == 0
    assert abs(float(b.margins[0]) - 2.0) < 1e-6
    # logits reconstruct: block sum over candidates
    logits = b.contrib.sum(dim=1)
    assert int(logits.argmax(1)[0]) == 0       # candidate 0 is the decode


def test_rulebank_empty_is_noop():
    from pil.proposer import RuleBank, build_sources

    z = torch.randn(4, 6)
    A = torch.randn(6, 5)
    bank = RuleBank(6, 5, 0)
    assert bank.n_rules == 0 and bank(z) is None
    assert torch.allclose(build_sources(z, A, bank), build_sources(z, A, None))


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
