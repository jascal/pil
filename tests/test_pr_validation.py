"""PR-validation quantities and statistics behave as the prereg defines them."""

import numpy as np
import pytest

from pil.pr_validation import (
    bootstrap_ci,
    centred_target,
    erank,
    k_suf,
    merge_layers,
    partial_spearman,
    pr,
    pr_head_decides,
    rankdata,
    spearman,
    split_ratio,
)


def test_pic_counterexample_pr_one_but_two_blocks_needed():
    """pic §4.1: c_a = (1,2,0), c_b = (0,−2,½) — t wins, PR of target weights is 1, no singleton decides."""
    C = np.array([[1.0, 2.0, 0.0], [0.0, -2.0, 0.5]])
    t, v2 = 0, 2                                    # full sum (1, 0, ½): runner-up is token 2
    assert pr(C[:, t]) == pytest.approx(1.0)
    assert k_suf(C, t, v2) == 2
    assert not pr_head_decides(C, t, C[:, t], 1.0)  # the "one effective block" alone picks token 1


def test_k_suf_singleton_and_miss():
    C = np.array([[5.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert k_suf(C, 0, 1) == 1
    C2 = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert k_suf(C2, 0, 1) == 3                     # nb + 1: even the full sum misses t


def test_centred_target_removes_common_mode():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(5, 8))
    cands = np.arange(8)
    base = centred_target(C, 2, cands)
    shifted = centred_target(C + rng.normal(size=(5, 1)), 2, cands)   # per-block offset, same for all tokens
    assert np.allclose(base, shifted)


def test_merge_layers_and_split_ratio():
    C = np.arange(15, dtype=float).reshape(5, 3)    # embed + 2 layers x (attn, mlp)
    M = merge_layers(C)
    assert M.shape == (3, 3) and np.allclose(M[0], C[0]) and np.allclose(M[1], C[1] + C[2])
    assert split_ratio(np.array([4.0, 0.0, 0.0]), parts=4) == pytest.approx(4.0)   # one block -> four equal
    with pytest.raises(ValueError):
        merge_layers(np.zeros((4, 3)))


def test_erank_bounds():
    assert erank(np.eye(4)) == pytest.approx(4.0)
    assert erank(np.outer(np.ones(4), np.arange(1.0, 4.0))) == pytest.approx(1.0)


def test_rankdata_ties():
    assert rankdata(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_partial_spearman_removes_shared_driver():
    rng = np.random.default_rng(1)
    z = rng.normal(size=4000)
    x = z + 0.3 * rng.normal(size=4000)
    y = z + 0.3 * rng.normal(size=4000)
    assert spearman(x, y) > 0.8
    assert abs(partial_spearman(x, y, z)) < 0.1
    w = rng.normal(size=4000)                        # a genuine extra shared signal survives
    assert partial_spearman(x + w, y + w, z) > 0.5


def test_cluster_bootstrap_ci_brackets_statistic():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    y = x + rng.normal(size=400)
    clusters = np.repeat(np.arange(40), 10)
    lo, hi = bootstrap_ci(spearman, (x, y), clusters, reps=300)
    assert lo < spearman(x, y) < hi
