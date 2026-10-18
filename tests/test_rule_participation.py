"""Ablation reports the cluster whose within-cluster margin actually moved."""

from __future__ import annotations

import torch

from pil.learner import PILConfig, ProjectiveIncidenceLearner
from pil.proposer import RuleBank, build_sources, targeted_rulebank
from pil.rule_participation import (
    MARGIN_MOVE_TOL,
    bank_verdict,
    budget_agreement,
    cluster_atoms,
    code_pair_mass,
    logits_with_rule_zeroed,
    mass_share,
    moved_hard_clusters,
    per_cluster_within_margins,
    rule_agreement,
    rule_margin_abs_delta,
    rule_moved_clusters,
    rule_verdict,
)
from pil.synthetic import create_compositional_problem


def _wire_one_cluster(n_clusters: int, wired: int, dim: int, n_examples: int = 256):
    """A one-rule bank that fires only on ``wired``'s code atoms, and a frame that reads it."""
    cfg = PILConfig(dim=dim, n_propositions=2 * n_clusters, device="cpu", seed=0)
    z, A, tgt, _, hc = create_compositional_problem(
        cfg, n_examples=n_examples, n_clusters=n_clusters, n_hard=n_clusters, noise=0.0, signal=True
    )
    K = z.shape[1]
    bank = RuleBank(K, dim, 1, device="cpu", seed=1)
    with torch.no_grad():
        bank.W.zero_()
        bank.b.zero_()
        p = n_clusters + 2 * wired
        bank.W[0, p] = 1.0
        bank.W[0, p + 1] = -1.0
        bank.Bdir.zero_()
        bank.Bdir[0, 0] = 1.0
    model = ProjectiveIncidenceLearner(cfg)
    with torch.no_grad():
        model.U.zero_()
        model.bias.zero_()
        # Synonym pair of the wired cluster: positive pole along the rule direction, negative against it.
        model.U[2 * wired + 1, 0] = 1.0
        model.U[2 * wired, 0] = -1.0
        # The other cluster's tokens sit on an orthogonal axis, so a leaked rule write would still
        # not move their within-cluster margin. With noise 0 the leak is exactly zero anyway.
        other = 1 - wired
        model.U[2 * other, 1] = 1.0
        model.U[2 * other + 1, 1] = -1.0
    return model, z, A, tgt, hc, bank


def test_ablation_reports_only_the_cluster_whose_margin_moved():
    wired = 1  # not cluster 0, so a helper that hardcodes {0} fails
    n_clusters, dim = 2, 8
    model, z, A, tgt, hc, bank = _wire_one_cluster(n_clusters, wired, dim)

    reported = rule_moved_clusters(model, z, A, bank, tgt, hc, rule=0, tol=MARGIN_MOVE_TOL)

    before = per_cluster_within_margins(
        logits_with_rule_zeroed(model, z, A, bank, None), tgt, n_clusters
    )
    # Independent removal on the same build_sources path, not via rule_moved_clusters.
    sources = build_sources(z, A, bank).clone()
    sources[:, A.shape[0] + 0, :] = 0
    after_logits = model.forward(sources, target=None)["logits"]
    after = per_cluster_within_margins(after_logits, tgt, n_clusters)
    expected = moved_hard_clusters(before, after, hc, MARGIN_MOVE_TOL)

    assert expected == [wired]
    assert reported == expected
    assert 0 not in reported
    # Skipping the ablation (returning the un-ablated cluster set, or nothing) cannot pass.
    assert reported != []
    assert float((before - after).abs()[wired]) > MARGIN_MOVE_TOL
    assert float((before - after).abs()[0]) <= MARGIN_MOVE_TOL


def test_mass_share_cut_is_strictly_above_half():
    one = mass_share(torch.tensor([0.0, 4.0, 0.0]))
    assert one[0] == 1 and one[3] == "single" and one[1] == 1.0
    even = mass_share(torch.tensor([1.0, 1.0]))
    assert even[3] == "spread" and even[1] == 0.5
    assert mass_share(torch.zeros(3))[3] == "idle"
    # A helper that ignores the vector and always returns single fails the even case.
    assert mass_share(torch.tensor([0.2, 0.2, 0.6]))[3] == "single"


def test_targeted_rulebank_code_weights_are_single_and_an_even_split_is_spread():
    n_clusters = 2
    bank = targeted_rulebank([1], n_atoms=3 * n_clusters, n_clusters=n_clusters, dim=4, device="cpu", seed=0)
    for rule in range(bank.n_rules):
        leader, _top1, _top3, label = mass_share(code_pair_mass(bank.W[rule], n_clusters), idle_at=0.0)
        assert label == "single"
        assert leader == 1
    split = torch.zeros(3 * n_clusters)
    split[n_clusters + 0] = 1.0  # cluster 0 code
    split[n_clusters + 1] = 1.0
    split[n_clusters + 2] = 1.0  # cluster 1 code
    split[n_clusters + 3] = 1.0
    leader, top1, _top3, label = mass_share(code_pair_mass(split, n_clusters), idle_at=0.0)
    assert label == "spread" and top1 == 0.5
    assert leader in (0, 1)


def test_clean_margin_of_wired_rule_is_single_on_its_cluster():
    wired = 1
    n_clusters = 2
    model, z, A, tgt, _hc, bank = _wire_one_cluster(n_clusters, wired, dim=8)
    delta = rule_margin_abs_delta(model, z, A, bank, tgt, n_clusters, rule=0)
    leader, top1, _top3, label = mass_share(delta)
    assert label == "single"
    assert leader == wired
    assert top1 > 0.5
    # The other cluster is the one the wire did not touch.
    assert float(delta[1 - wired]) <= MARGIN_MOVE_TOL


def test_clean_counterpart_keeps_targets_when_noise_is_zero():
    cfg = PILConfig(dim=8, n_propositions=8, device="cpu", seed=0)
    _zn, _an, tgt_n, _, _ = create_compositional_problem(
        cfg, n_examples=64, n_clusters=4, n_hard=4, noise=0.08
    )
    _z0, _a0, tgt_0, _, _ = create_compositional_problem(
        cfg, n_examples=64, n_clusters=4, n_hard=4, noise=0.0
    )
    assert torch.equal(tgt_n, tgt_0)
    assert not torch.allclose(_zn, _z0)


def test_rule_agreement_requires_all_three_measures():
    assert rule_agreement("single", "single", "single") == "single"
    assert rule_agreement("spread", "spread", "spread") == "spread"
    assert rule_agreement("single", "single", "spread") == "disagree"
    assert rule_agreement("idle", "single", "single") == "idle"
    assert budget_agreement(["single", "idle"]) == "agree_single"
    assert budget_agreement(["single", "spread"]) == "disagree"
    assert budget_agreement(["spread", "idle"]) == "agree_spread"


def test_two_disjoint_clusters_are_packing_and_one_is_reuse():
    n_clusters = 4
    assert cluster_atoms(0, n_clusters).isdisjoint(cluster_atoms(1, n_clusters))
    assert rule_verdict([0], n_clusters) == "reuse"
    assert rule_verdict([0, 1], n_clusters) == "packing"
    assert rule_verdict([], n_clusters) == "idle"
    assert bank_verdict(["reuse", "packing"]) == "packing"
    assert bank_verdict(["reuse", "idle"]) == "reuse"
    assert bank_verdict(["idle", "idle"]) == "idle"
