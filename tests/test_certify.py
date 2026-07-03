"""TrajectoryCertificate: the T-traj runtime harness obeys the theorem it measures."""

import torch

from pil.certify import TrajectoryCertificate
from pil.learner import PILConfig, ProjectiveIncidenceLearner, create_synthetic_problem


def small_setup(seed=0):
    cfg = PILConfig(dim=16, n_propositions=32, n_sources_per_step=8, seed=seed, device="cpu")
    src, tgt, _ = create_synthetic_problem(cfg, n_examples=64)
    model = ProjectiveIncidenceLearner(cfg)
    model.normalize_U()
    return cfg, model, src, tgt


def test_theorem_self_checks_hold_on_real_steps():
    """Run real AdamW+renorm steps; the theorem's two forbidden events must not occur."""
    cfg, model, src, tgt = small_setup()
    bank = src[:32]
    cert = TrajectoryCertificate(model, bank, true_targets=tgt[:32])
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    for _ in range(60):
        out = model.forward(src, target=tgt)
        opt.zero_grad()
        out["total_loss"].backward()
        opt.step()
        model.normalize_U()
        cert.step()
    s = cert.summary()
    assert s["viol_premise_flip"] == 0        # flip while premise held: theorem forbids
    assert s["viol_transfer"] == 0            # margin fell below m - 2*delta: forbidden
    assert s["n_steps"] == 60
    # telescoped bound is a valid lower bound: slack >= ~0
    assert s["telescope_slack_p10"] >= -1e-3


def test_effective_update_measured_post_normalization():
    """A step whose raw update is large but renormalizes away must show small eps."""
    cfg, model, src, tgt = small_setup()
    cert = TrajectoryCertificate(model, src[:16])
    with torch.no_grad():
        model.U.mul_(7.0)          # pure scale change...
    model.normalize_U()            # ...restored by row renorm: effective update ~ 0
    rec = cert.step()
    assert rec["eps"] < 1e-5
    assert rec["new_flips"] == 0


def test_frozen_decision_and_flip_detection():
    """Forcing the frame to swap two propositions' roles must register flips, not silence."""
    cfg, model, src, tgt = small_setup()
    cert = TrajectoryCertificate(model, src[:16])
    with torch.no_grad():
        model.U.copy_(torch.randn_like(model.U))   # unrelated frame: decisions scrambled
    model.normalize_U()
    rec = cert.step()
    assert rec["new_flips"] > 0
    # a huge step has a huge delta -> premise cannot have held where a flip happened
    assert cert.viol_premise_flip == 0


def test_rho_constancy_asserted():
    cfg, model, src, tgt = small_setup()
    cert = TrajectoryCertificate(model, src[:16])
    cert.sources[0, 0, 0] += 1.0               # mutate the bank
    try:
        cert.step()
        raise AssertionError("expected rho-constancy assertion")
    except AssertionError as e:
        assert "rho constancy" in str(e)


def test_training_step_hook_off_by_default():
    cfg, model, src, tgt = small_setup()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model.training_step(src, tgt, opt)               # certifier=None path
    assert "loss" in out
    cert = TrajectoryCertificate(model, src[:16])
    model.training_step(src, tgt, opt, certifier=cert)
    assert cert.n_steps == 1
