import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from ltt_certificate import GRID_Q, decide, grid_from_train, ltt_tau, p_value  # noqa: E402


def test_grid_is_descending_train_quantiles():
    s = torch.arange(1000).float()
    g = grid_from_train(s)
    assert len(g) == len(GRID_Q) == 95 and GRID_Q[0] == 0.95 and GRID_Q[-1] == 0.01
    assert bool((g[:-1] >= g[1:]).all())
    assert float(g[0]) == pytest.approx(949.05)


def test_p_value():
    assert p_value(0, 0) == 1.0
    assert p_value(22, 0) == pytest.approx(0.9**22)  # 0.098 <= 0.10: rejectable with 22 clean classes
    assert p_value(21, 0) > 0.10
    assert p_value(100, 5) == pytest.approx(binom.cdf(5, 100, 0.10))


def test_ltt_stops_at_first_non_rejection():
    grid = torch.tensor([3.0, 2.0, 1.0, 0.0])
    score = torch.tensor([3.5] * 40 + [2.5] * 40 + [1.5] * 40 + [0.5] * 40)
    cert = torch.ones(160, dtype=torch.bool)
    ok = torch.tensor([True] * 40 + [True] * 40 + [False] * 40 + [True] * 40)
    chosen, steps = ltt_tau(grid, score, cert, ok)
    assert chosen[0] == 2.0 and chosen[1] == 80 and chosen[2] == 0
    assert [s[4] for s in steps] == [True, True, False]  # stops at tau=1; tau=0 never tested
    none, steps = ltt_tau(grid, score[:10], cert[:10], ok[:10])
    assert none is None and len(steps) == 1


def test_ltt_controls_the_rate_among_issued():
    """Simulation: score ~ U(0,1); P(wrong | score s) = 0.3 * (1 - s). Over many calibration draws, the
    selected threshold's TRUE false rate exceeds alpha at most about delta of the time."""
    rng = np.random.default_rng(0)
    grid = torch.tensor(np.linspace(0.95, 0.0, 96))

    def true_fc(t):  # E[0.3 (1 - s) | s >= t] for s ~ U(0,1) = 0.15 (1 - t)
        return 0.15 * (1 - t)

    viol, trials = 0, 400
    for _ in range(trials):
        s = rng.uniform(size=1800)
        wrong = rng.uniform(size=1800) < 0.3 * (1 - s)
        chosen, _ = ltt_tau(grid, torch.tensor(s), torch.ones(1800, dtype=torch.bool), torch.tensor(~wrong))
        if chosen is not None and true_fc(chosen[0]) > 0.10:
            viol += 1
    assert viol / trials <= 0.10 + 3 * (0.1 * 0.9 / trials) ** 0.5


def _seed(fc, cov):
    return dict(FCi=fc, coverage=cov)


def test_decision_rules():
    v = decide(
        [_seed(0.05, 0.25), _seed(0.11, 0.22), _seed(None, 0.0 + 0.2), _seed(0.08, 0.28), _seed(0.07, 0.3)]
    )
    assert (v["H1"]["verdict"], v["H2"]["verdict"], v["H3"]["verdict"]) == ("pass", "pass", "pass")
    v = decide([_seed(0.12, 0.1), _seed(0.11, 0.3), _seed(0.05, 0.2), _seed(0.05, 0.2), _seed(0.05, 0.2)])
    assert v["H1"]["violations"] == 2
    assert (v["H1"]["verdict"], v["H2"]["verdict"], v["H3"]["verdict"]) == ("fail", "pass", "fail")
