import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_wikitext_gated_rescue as campaign  # noqa: E402


def test_traced_matches_original_on_identical_synthetic_inputs(monkeypatch):
    ids = torch.tensor([[0, 1], [1, 0], [2, 1], [0, 2]])
    yv = torch.tensor([2, 1, 0, 1])
    cls = torch.tensor([0, 1, 2])
    idxs = torch.arange(4)
    model = SimpleNamespace(
        counts=torch.tensor([[0, 2, 0], [0, 0, 3], [2, 0, 0]]),
        counts_calib=None,
        rules2=[],
    )

    def rule(w):
        return torch.where(w[:, -1] == 1, torch.tensor(2), torch.tensor(-1))

    rules = [("synthetic", rule)]
    monkeypatch.setattr(campaign.v5, "STRATA", False)
    monkeypatch.setitem(
        campaign.v5.CONF_FNS,
        "synthetic",
        lambda w: (rule(w), torch.full((len(w),), 0.8)),
    )
    _, original = campaign.v5.core_cover_sw(
        model, rules, ids, yv, cls, idxs, return_pred=True
    )
    _, traced, _ = campaign.core_cover_sw_traced(model, rules, ids, yv, cls, idxs)
    assert torch.equal(traced, original)


def test_te_split_is_deterministic_disjoint_and_exhaustive():
    te = torch.arange(100, 140)
    a_val, a_test = campaign.split_te(te)
    b_val, b_test = campaign.split_te(te)
    assert torch.equal(a_val, b_val)
    assert torch.equal(a_test, b_test)
    assert not torch.isin(a_val, a_test).any()
    assert set(torch.cat([a_val, a_test]).tolist()) == set(te.tolist())


def test_tau_grid_is_exact_deterministic_deciles():
    conf = torch.tensor([0.1, 0.7, 0.2, 0.9, 0.4, 0.3, 0.8, 0.5, 1.0, 0.6])
    expected = torch.quantile(conf.double(), torch.arange(1, 11).double() / 10).tolist()
    assert campaign.tau_deciles(conf) == expected
    assert campaign.tau_deciles(conf) == campaign.tau_deciles(conf.clone())


def test_exact_binomial_helper_matches_scipy_and_closed_form():
    assert campaign.exact_discordant_p(1, 3) == binomtest(1, 4, 0.5).pvalue
    assert campaign.exact_discordant_p(0, 4) == 0.125
    assert campaign.exact_discordant_p(0, 0) == 1.0


def test_mined_frames_slice_cannot_overlap_held_out():
    tr = torch.arange(10)
    conf = torch.linspace(0, 1, 10)
    val_te = torch.arange(10, 13)
    test_te = torch.arange(13, 20)
    fit = campaign.mining_fit_slice(tr, conf, 0.5, val_te, test_te)
    assert not torch.isin(fit, val_te).any()
    assert not torch.isin(fit, test_te).any()
    leaking_tr = torch.tensor([0, 10])
    try:
        campaign.mining_fit_slice(leaking_tr, torch.tensor([0.1, 0.1]), 0.5, val_te, test_te)
    except ValueError:
        pass
    else:
        raise AssertionError("expected structural leak guard to fail")


def test_candidate_pool_is_fixed_for_both_arms():
    expected = (
        "pointer",
        "dstate gate",
        "sentpair/dgate2",
        "attrib-subj gate",
        "mined frames",
    )
    assert campaign.CANDIDATE_FAMILIES == expected
    flat_labels = campaign.CANDIDATE_FAMILIES
    gated_labels = campaign.CANDIDATE_FAMILIES
    assert flat_labels == gated_labels == expected


def test_b0_score_band_is_inclusive():
    assert campaign.b0_within_registered_band(0.341)
    assert campaign.b0_within_registered_band(0.351)
    assert not campaign.b0_within_registered_band(0.340)
    assert not campaign.b0_within_registered_band(0.352)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_frozen_b0_round_trip_preserves_predictions(tmp_path, monkeypatch, device):
    monkeypatch.setattr(campaign.v5, "CONF_FNS", {})
    monkeypatch.setattr(campaign.v5, "RULE_CONF", {})
    monkeypatch.setattr(campaign.v5, "STRATA", False)
    monkeypatch.setattr(campaign.v5, "DEV", device)
    vocab = 4
    ids = torch.tensor(
        [[0, 1, 0, 1], [2, 1, 2, 0], [3, 0, 3, 1], [1, 2, 0, 2]],
        device=device,
    )
    yv = torch.tensor([2, 3, 0, 1], device=device)
    cls = torch.arange(vocab, device=device)
    idxs = torch.arange(len(ids), device=device)
    model = SimpleNamespace(
        counts=torch.tensor(
            [[2, 0, 0, 0], [0, 0, 3, 0], [0, 1, 0, 0], [0, 0, 0, 4]],
            device=device,
        ),
        counts_calib=torch.tensor([0.4, 0.5, 0.6, 0.7], device=device),
        rules2=[],
    )
    members = torch.tensor([False, True, False, True], device=device)
    base = vocab + 2
    table = campaign.v5.KeyTable(
        torch.tensor([(1 + 1) * base + 1, (3 + 1) * base + 1], device=device),
        torch.tensor([3, 2], device=device),
        torch.tensor([0.9, 0.8], device=device),
    )
    dname = "synthetic recent [2]"
    dspec = ("dfeat", ("recent-member", {"members": members}), table, base)
    dfn, dconf = campaign._reconstruct_rule(dname, dspec)
    iname = "induction L=2"
    rules = [(iname, campaign.v5.mir_induction_L(2)), (dname, dfn)]
    campaign.v5.RULE_CONF[iname] = 0.75
    campaign.v5.CONF_FNS[dname] = dconf
    before, before_pred, _ = campaign.core_cover_sw_traced(
        model, rules, ids, yv, cls, idxs
    )

    path = tmp_path / "synthetic_b0.pt"
    campaign.freeze_b0(
        path,
        model,
        rules,
        before,
        1.25,
        emit_info={dname: dspec},
        rule_conf={iname: 0.75},
    )
    campaign.v5.CONF_FNS.clear()
    campaign.v5.RULE_CONF.clear()
    loaded_model, loaded_rules, loaded_core, loaded_wall = campaign.load_frozen_b0(path)
    after, after_pred, _ = campaign.core_cover_sw_traced(
        loaded_model, loaded_rules, ids, yv, cls, idxs
    )

    assert torch.equal(after_pred, before_pred)
    assert after == before == loaded_core
    assert loaded_wall == 1.25
    assert [name for name, _ in loaded_rules] == [iname, dname]


def test_pool_exclusion_only_removes_colliding_b0_families():
    full, no_exclusions = campaign.active_candidate_families(["induction L=2"])
    assert full == campaign.CANDIDATE_FAMILIES
    assert no_exclusions == []

    reduced, exclusions = campaign.active_candidate_families(
        ["induction L=2", "mined frames (online)"]
    )
    assert reduced == tuple(
        name for name in campaign.CANDIDATE_FAMILIES if name != "mined frames"
    )
    assert exclusions == [
        {"family": "mined frames", "reason": "already admitted into B0′"}
    ]
