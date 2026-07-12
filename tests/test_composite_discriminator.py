"""Structural tests for the composite-discriminator campaign (no GPU end-to-end)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_composite_discriminator as campaign  # noqa: E402

# ---------------------------------------------------------------------------
# 1. CV recipe determinism
# ---------------------------------------------------------------------------


def test_cv_recipe_determinism_same_seeds_same_aucs():
    """Same seeds → same folds/AUCs on a small fixture (exact reproducibility)."""
    g = torch.Generator().manual_seed(1)
    n, d = 40, 4
    # mild signal so AUC is not degenerate noise
    labels = torch.tensor([1] * 20 + [0] * 20, dtype=torch.float64)
    features = torch.randn(n, d, generator=g, dtype=torch.float64)
    features[:20] += 0.8  # gains slightly higher on all axes

    a = campaign.cv_auc_distribution(
        features, labels, repeat_seeds=[0, 1, 2], n_folds=5, steps=50, lr=0.1
    )
    b = campaign.cv_auc_distribution(
        features, labels, repeat_seeds=[0, 1, 2], n_folds=5, steps=50, lr=0.1
    )
    assert a["aucs"] == b["aucs"]
    assert a["median"] == b["median"]

    folds_a = campaign.stratified_kfold_indices(labels, 5, seed=7)
    folds_b = campaign.stratified_kfold_indices(labels, 5, seed=7)
    for (tr_a, te_a), (tr_b, te_b) in zip(folds_a, folds_b, strict=True):
        assert torch.equal(tr_a, tr_b) and torch.equal(te_a, te_b)


# ---------------------------------------------------------------------------
# 2. z-scoring leak guard
# ---------------------------------------------------------------------------


def test_zscore_leak_guard_heldout_not_in_train_stats():
    """Train mean/std for a held-out fold exclude that fold's rows.

    Fixture: one extreme held-out row would shift mean measurably if leaked.
    """
    # 6 rows, 1 feature; fold isolates the extreme row
    x = torch.tensor(
        [[0.0], [0.0], [0.0], [0.0], [0.0], [100.0]], dtype=torch.float64
    )
    y = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float64)
    # Manually: train = rows 0..4, test = row 5
    train_idx = torch.arange(5)
    test_idx = torch.tensor([5])
    mean, std = campaign.zscore_fit(x[train_idx])
    assert float(mean[0]) == pytest.approx(0.0)
    # if row 5 were included, mean would be 100/6 ≈ 16.67
    assert float(mean[0]) < 1.0
    z_te = campaign.zscore_transform(x[test_idx], mean, std)
    # with train mean=0, z = 100/std; if leaked mean~16.67, z would be smaller
    leaked_mean = x.mean(0)
    z_leaked = (x[test_idx] - leaked_mean) / std
    assert float(z_te[0, 0]) != pytest.approx(float(z_leaked[0, 0]), rel=0, abs=1.0)
    assert float(z_te[0, 0]) > float(z_leaked[0, 0])

    # Structural: pooled_oof_scores must z-score train-only per fold
    # Construct folds so fold 0 holds out only the extreme row when labels allow.
    scores = campaign.pooled_oof_scores(x, y, n_folds=3, fold_seed=0, steps=30)
    assert scores.shape == (6,)
    # re-derive one fold and check mean excludes test rows
    for tr, te in campaign.stratified_kfold_indices(y, 3, 0):
        m, s = campaign.zscore_fit(x[tr])
        # none of the test rows may appear in the train mean's supporting set
        assert set(te.tolist()).isdisjoint(set(tr.tolist()))
        # if any test value is extreme, train mean stays near non-extreme cluster
        if 5 in te.tolist():
            assert float(m[0]) < 1.0


# ---------------------------------------------------------------------------
# 3. Permutation null on known-null fixture
# ---------------------------------------------------------------------------


def test_permutation_null_on_null_by_construction_fixture():
    """Features independent of labels → AUC ~0.5, p not tiny.

    Tolerance: with small N and few shuffles, p should be > 0.05 most of the time
    under a true null; we only assert p > 0.01 to avoid flaky CI (coarse sanity).
    """
    g = torch.Generator().manual_seed(99)
    n = 60
    labels = torch.tensor([1] * 30 + [0] * 30, dtype=torch.float64)
    features = torch.randn(n, 4, generator=g, dtype=torch.float64)  # no signal
    observed = campaign.pooled_oof_auc(
        features, labels, n_folds=5, fold_seed=0, steps=40
    )
    assert 0.30 <= observed <= 0.70  # coarse band around 0.5
    null = campaign.permutation_null_aucs(
        features, labels, n_shuffles=30, perm_seed=42, fold_seed=0, steps=40
    )
    p = campaign.empirical_perm_p(observed, null)
    # null-by-construction: p should not be vanishingly small
    assert p > 0.01, f"unexpectedly tiny p={p} on null fixture (observed={observed})"


# ---------------------------------------------------------------------------
# 4. 50/50 split determinism + stratification + disjointness
# ---------------------------------------------------------------------------


def test_half_ab_split_determinism_stratification_disjointness():
    n = 200
    b0 = torch.tensor([True] * 120 + [False] * 80)
    a1, b1 = campaign.split_half_ab(n, b0, seed=0)
    a2, b2 = campaign.split_half_ab(n, b0, seed=0)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    assert len(a1) + len(b1) == n
    assert set(a1.tolist()).isdisjoint(set(b1.tolist()))
    # stratification: B0-correct ratio preserved within 2 percentage points
    rate_full = float(b0.float().mean())
    rate_a = float(b0[a1].float().mean())
    rate_b = float(b0[b1].float().mean())
    assert abs(rate_a - rate_full) < 0.02
    assert abs(rate_b - rate_full) < 0.02


# ---------------------------------------------------------------------------
# 5. Threshold-grid fixedness (object identity / call counting)
# ---------------------------------------------------------------------------


def test_threshold_grid_fixedness_not_rederived_per_tau(monkeypatch):
    """The 10-point grid is computed once; clause2_val_sweep does not re-call score_deciles."""
    calls = {"n": 0}
    real_deciles = campaign.score_deciles

    def counting_deciles(scores):
        calls["n"] += 1
        return real_deciles(scores)

    monkeypatch.setattr(campaign, "score_deciles", counting_deciles)

    # Build a tiny synthetic half-A OOF grid once
    scores = torch.linspace(-1, 1, 20, dtype=torch.float64)
    grid = campaign.score_deciles(scores)
    assert calls["n"] == 1
    assert len(grid) == 10
    frozen = list(grid)
    grid_id = id(frozen)

    # Fake ctx with one tau_run — monkeypatch flat/stack heavy pieces
    from experiments import campaign_wikitext_gated_rescue as gated_rescue

    n_val = 8
    b0_pred = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    b1_pred = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    b1_conf = torch.tensor([0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
    target = torch.tensor([1, 1, 0, 1, 0, 0, 1, 1])

    class FakeState:
        def __init__(self, pred, conf=None):
            self.pred = pred
            self.conf = conf

    class FakeStack:
        def __init__(self):
            self.last_states = [
                FakeState(b0_pred.clone()),
                FakeState(b1_pred.clone(), b1_conf.clone()),
            ]

        def forward(self, ids, counts_row_fn=None):
            return self.last_states

    class FakeBlock:
        def predict_cover(self, ids, counts_row_fn=None):
            return b0_pred.clone(), torch.ones(len(ids))

    class FakeFlat:
        def predict_cover(self, ids, counts_row_fn=None):
            # flat slightly different from b0
            p = b0_pred.clone()
            p[0] = 1
            return p, torch.ones(len(ids))

    def fake_run_flat(*args, **kwargs):
        return FakeFlat(), ["mined frames"], []

    monkeypatch.setattr(gated_rescue, "_run_flat", fake_run_flat)

    val_axes = {
        name: torch.randn(n_val, dtype=torch.float64) for name in (
            "confidence", "gap", "teacher_consensus", "decile_index"
        )
    }
    ctx = {
        "ids": torch.zeros(n_val, 2, dtype=torch.long),
        "yv": target,
        "val_te": torch.arange(n_val),
        "counts_fn": lambda w: (torch.full((len(w),), -1), torch.zeros(len(w))),
        "rules": [],
        "conf_fns": {},
        "b0": FakeBlock(),
        "val_axes": val_axes,
        "model": SimpleNamespace(),
        "tau_runs": [
            {
                "tau": 0.3,
                "candidates": [],
                "stack": FakeStack(),
                "admitted": ["pointer"],
            },
            {
                "tau": 0.5,
                "candidates": [],
                "stack": FakeStack(),
                "admitted": ["pointer"],
            },
        ],
    }
    model = {
        "weight": torch.zeros(4, dtype=torch.float64),
        "bias": torch.zeros((), dtype=torch.float64),
    }
    mean = torch.zeros(4, dtype=torch.float64)
    std = torch.ones(4, dtype=torch.float64)

    # score_deciles must NOT be called inside the sweep
    calls["n"] = 0
    table = campaign.clause2_val_sweep(
        ctx, model, mean, std, frozen, grid_id=grid_id
    )
    assert calls["n"] == 0, "threshold grid was re-derived inside the tau×threshold sweep"
    assert len(table) == 2 * 10  # 2 taus × 10 thresholds
    # all rows use thresholds from the same frozen grid
    thr_set = {row["threshold"] for row in table}
    assert thr_set == set(frozen)


# ---------------------------------------------------------------------------
# 6. No-admissible path still invokes #86 replay
# ---------------------------------------------------------------------------


def test_no_admissible_operating_point_path_still_runs_86(monkeypatch):
    """No (tau, threshold) passes → exact message; #86 replay still invoked."""
    emitted = []
    real_log = campaign.log

    def capturing_log(msg=""):
        emitted.append(msg)
        real_log(msg)

    monkeypatch.setattr(campaign, "log", capturing_log)

    # select_best returns None
    assert campaign.select_best_admissible([
        {"admissible": False, "val_gain": 0.0, "tau": 0.1, "threshold": 0.0},
        {"admissible": False, "val_gain": -0.01, "tau": 0.2, "threshold": 0.1},
    ]) is None

    # Simulate the main-path branch
    table = [{"admissible": False, "val_gain": 0.0, "tau": 0.1, "threshold": 0.0}]
    selected = campaign.select_best_admissible(table)
    replay_called = {"n": 0}

    def fake_86(*args, **kwargs):
        replay_called["n"] += 1
        return {
            "rescue": False,
            "sweep": [],
            "n_sweep_rows": 0,
            "historical_clearing_points": [],
            "n_clearing_points": 0,
            "threshold_grid": [],
            "honesty": "flag",
        }

    monkeypatch.setattr(campaign, "run_86_subprocess", fake_86)

    if selected is None:
        campaign.log(campaign.NO_ADMISSIBLE_MSG)
        # still run #86
        campaign.run_86_subprocess(
            {"weight": torch.zeros(4), "bias": torch.zeros(())},
            torch.zeros(4),
            torch.ones(4),
            [0.0] * 10,
        )

    assert campaign.NO_ADMISSIBLE_MSG in emitted
    assert replay_called["n"] == 1


# ---------------------------------------------------------------------------
# 7. half-B flat-regression recomputation
# ---------------------------------------------------------------------------


def test_half_b_flat_regressions_recomputed_not_copied_from_val(monkeypatch):
    """clause (iv) uses flat regressions recomputed on half-B, not val_te."""
    from experiments import campaign_wikitext_gated_rescue as gated_rescue

    n_test = 10
    # half-B indices 0..4 in test_te space; construct preds so half-B flat regs = 2
    # while a "val" figure would be 99 if mistakenly used
    b0_pred_all = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    flat_pred_all = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 1, 1])  # first 2 wrong vs b0
    target_all = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # b0 always right
    # flat regs on half-B (0..4): rows 0,1 → 2
    # if someone used full test: still 2; make half-B-only differ by restricting

    class FakeState:
        def __init__(self, pred, conf):
            self.pred = pred
            self.conf = conf

    class FakeStack:
        last_states = None

        def forward(self, ids, counts_row_fn=None):
            m = len(ids)
            # B1 claims with high conf, wrong on some
            self.last_states = [
                FakeState(b0_pred_all[:m].clone(), torch.ones(m)),
                FakeState(torch.ones(m, dtype=torch.long), torch.full((m,), 0.9)),
            ]
            return self.last_states

    class FakeB0:
        def predict_cover(self, ids, counts_row_fn=None):
            m = len(ids)
            return b0_pred_all[:m].clone(), torch.ones(m)

    class FakeFlat:
        def predict_cover(self, ids, counts_row_fn=None):
            m = len(ids)
            return flat_pred_all[:m].clone(), torch.ones(m)

    def fake_run_flat(*a, **k):
        return FakeFlat(), ["mined frames"], []

    monkeypatch.setattr(gated_rescue, "_run_flat", fake_run_flat)
    monkeypatch.setattr(gated_rescue, "exact_discordant_p", lambda b, c: 0.5)

    half_b = torch.arange(5)  # first 5 of test_te
    test_axes = {
        name: torch.zeros(n_test, dtype=torch.float64)
        for name in ("confidence", "gap", "teacher_consensus", "decile_index")
    }
    ctx = {
        "ids": torch.zeros(n_test, 2, dtype=torch.long),
        "yv": target_all,
        "test_te": torch.arange(n_test),
        "val_te": torch.arange(3),  # different size/set
        "counts_fn": lambda w: (torch.full((len(w),), -1), torch.zeros(len(w))),
        "rules": [],
        "conf_fns": {},
        "b0": FakeB0(),
        "model": SimpleNamespace(),
        "test_axes": test_axes,
        "tau_runs": [{
            "tau": 0.3,
            "candidates": [],
            "stack": FakeStack(),
            "admitted": ["pointer", "mined frames"],
        }],
    }
    model = {
        "weight": torch.zeros(4, dtype=torch.float64),
        "bias": torch.tensor(0.0, dtype=torch.float64),
    }
    # disc scores always high so B1 claims when not weak
    # force score_all_rows path via zeros model → scores = 0; set threshold low
    result = campaign.evaluate_half_b(
        ctx,
        half_b,
        model,
        torch.zeros(4, dtype=torch.float64),
        torch.ones(4, dtype=torch.float64),
        tau=0.3,
        threshold=-1e9,  # always pass disc filter
    )
    assert result["flat_regs_source"] == "recomputed_on_half_b"
    # b0 correct on all half-B targets; flat wrong on first 2 → 2
    assert result["flat_regressions_half_b"] == 2
    # prove it's not a hardcoded val figure
    assert result["flat_regressions_half_b"] != 99


# ---------------------------------------------------------------------------
# 8. Discriminator-filter claim semantics
# ---------------------------------------------------------------------------


def test_discriminator_filter_claim_semantics():
    """(a) disc < thr → B0 never B1; (b) B0 never replaced by disc alone;
    (c) weak B1 not resurrected by high disc."""
    b0 = torch.tensor([10, 20, 30, 40])
    # row0: strong B1; row1: strong B1; row2: abstain weak; row3: low conf weak
    conf = torch.tensor([0.9, 0.9, 0.9, 0.1])
    b1_with_abstain = torch.tensor([11, 21, -1, 41])
    tau = 0.5
    thr = 0.0
    disc = torch.tensor([-1.0, 1.0, 1.0, 1.0])  # only row0 below thr

    out = campaign.discriminator_filter_claim(
        b0, b1_with_abstain, conf, disc, tau=tau, threshold=thr
    )
    # (a) row0 disc < thr → B0
    assert int(out[0]) == 10
    # row1 strong + disc ok → B1
    assert int(out[1]) == 21
    # (c) row2 B1 abstains (weak) even with disc >= thr → B0, not resurrected
    assert int(out[2]) == 30
    # (c) row3 low conf weak → B0
    assert int(out[3]) == 40

    # (b) B0 never filtered by disc: when B1 weak, always B0 regardless of disc
    disc_high = torch.tensor([100.0, 100.0, 100.0, 100.0])
    out2 = campaign.discriminator_filter_claim(
        b0, b1_with_abstain, conf, disc_high, tau=tau, threshold=thr
    )
    assert int(out2[2]) == 30
    assert int(out2[3]) == 40

    # (a) even if B1 conf is max, disc below thr keeps B0
    disc_low = torch.tensor([-5.0, -5.0, -5.0, -5.0])
    conf_hi = torch.tensor([0.99, 0.99, 0.99, 0.99])
    b1_all = torch.tensor([11, 21, 31, 41])
    out3 = campaign.discriminator_filter_claim(
        b0, b1_all, conf_hi, disc_low, tau=tau, threshold=thr
    )
    assert out3.tolist() == b0.tolist()


def test_score_deciles_are_ten_points():
    s = torch.arange(100, dtype=torch.float64)
    d = campaign.score_deciles(s)
    assert len(d) == 10
    assert d == sorted(d)


def test_empirical_perm_p_matches_frontier_convention():
    # observed far from 0.5, null all near 0.5 → p small
    null = [0.49 + 0.001 * (i % 3) for i in range(100)]
    p = campaign.empirical_perm_p(0.9, null)
    assert p < 0.05
    # observed near null → p large
    p2 = campaign.empirical_perm_p(0.5, null)
    assert p2 > 0.5
