import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from ltt_retest import GRID_Q, audit_draw, decide, grid_from_train, is_violation, seed_stats  # noqa: E402


def test_grid_starts_at_080():
    assert len(GRID_Q) == 80 and GRID_Q[0] == 0.80 and GRID_Q[-1] == 0.01
    g = grid_from_train(torch.arange(1000).float())
    assert bool((g[:-1] >= g[1:]).all()) and float(g[0]) == pytest.approx(799.2)


def test_audit_draw_issues_on_audit_set():
    grid = torch.tensor([3.0, 2.0, 1.0])
    s_pool = torch.tensor([3.5] * 50 + [2.5] * 50 + [1.5] * 50)
    ok_pool = torch.tensor([True] * 100 + [False] * 50)
    c_pool = torch.ones(150, dtype=torch.bool)
    s_aud = torch.tensor([3.5, 2.5, 2.5, 1.5, 0.5])
    c_aud = torch.tensor([True, True, False, True, True])
    ok_aud = torch.tensor([True, False, True, True, True])
    d = audit_draw(grid, s_pool, c_pool, ok_pool, torch.arange(150), s_aud, c_aud, ok_aud)
    assert d["tau"] == 2.0 and d["cal_n"] == 100 and d["cal_w"] == 0
    assert d["issued"].tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]  # uncertified audit class is never issued
    assert d["coverage"] == pytest.approx(0.4) and d["FCa"] == pytest.approx(0.5)
    none = audit_draw(
        grid, s_pool, c_pool, ok_pool, torch.arange(10), s_aud, c_aud, ok_aud
    )  # too few to reject
    assert none["tau"] is None and none["FCa"] is None and none["coverage"] == 0.0


def test_violation_and_stats():
    assert is_violation(0.11) and not is_violation(0.10) and not is_violation(None)
    draws = [
        dict(tau=1.0, FCa=0.12, coverage=0.4, faith_issued=0.9, held_faith_issued=0.95),
        dict(tau=1.0, FCa=0.05, coverage=0.3, faith_issued=0.95, held_faith_issued=0.97),
        dict(tau=None, FCa=None, coverage=0.0, faith_issued=None, held_faith_issued=None),
        dict(tau=1.0, FCa=0.08, coverage=0.5, faith_issued=0.92, held_faith_issued=0.96),
    ]
    st = seed_stats(draws)
    assert st["violation_rate"] == 0.25 and st["non_issuance"] == 0.25
    assert st["coverage"] == pytest.approx(0.3)
    assert st["FCa_mean"] == pytest.approx((0.12 + 0.05 + 0.08) / 3)


def test_decision_rules():
    v = decide(dict(violation_rate=0.12, coverage=0.3, non_issuance=0.05))
    assert [v[h]["verdict"] for h in ("H1", "H2", "H3")] == ["pass", "pass", "pass"]
    v = decide(dict(violation_rate=0.2, coverage=0.2, non_issuance=0.2))
    assert [v[h]["verdict"] for h in ("H1", "H2", "H3")] == ["fail", "fail", "fail"]
