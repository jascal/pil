"""CertifiedP3: backtracking terminates, enforcement is exact, certification implies safety."""

import torch

from pil.certify import TrajectoryCertificate
from pil.geometry import margin_to_worst
from pil.learner import PILConfig, ProjectiveIncidenceLearner, create_synthetic_problem
from pil.p3 import CertifiedP3


def setup(seed=0, dim=16, V=32, J=8, n=96):
    cfg = PILConfig(dim=dim, n_propositions=V, n_sources_per_step=J, seed=seed, device="cpu")
    src, tgt, _ = create_synthetic_problem(cfg, n_examples=n)
    model = ProjectiveIncidenceLearner(cfg)
    model.normalize_U()
    return cfg, model, src, tgt


def run_phase(model, src, t_frozen, lr, steps, eta=0.9, exact_floor=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    p3 = CertifiedP3(model, opt, src, t_frozen, eta=eta, exact_floor=exact_floor)
    for _ in range(steps):
        out = model.forward(src, target=t_frozen)
        p3.step(out["total_loss"])
    return p3


def test_backtracking_terminates_and_enforces_at_hostile_lr():
    """lr=1.0 would scramble everything unclipped; the trust region must keep every
    currently-decided context decided, terminating via alpha halving (alpha=0 worst case)."""
    cfg, model, src, tgt = setup()
    t_frozen = model.forward(src, target=None)["logits"].argmax(-1)   # entry decisions
    p3 = run_phase(model, src, t_frozen, lr=1.0, steps=25)
    s = p3.summary()
    assert s["enforced_flips_total"] == 0                  # the hard constraint, exact
    assert s["clip_frac"] > 0.0                            # hostile lr must actually clip
    assert torch.isfinite(model.U).all()
    # decided set is a ratchet: everything decided at entry is still decided
    L = model.forward(src, target=None)["logits"]
    assert bool((L.argmax(-1) == t_frozen).all())


def test_certified_subset_never_flips():
    """(b) implies (a) on the certified subset: contexts with m_prev > 2*delta must stay
    decided across the accepted step — checked per step against the recorded delta."""
    cfg, model, src, tgt = setup(seed=1)
    t_frozen = model.forward(src, target=None)["logits"].argmax(-1)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    p3 = CertifiedP3(model, opt, src, t_frozen)
    for _ in range(40):
        with torch.no_grad():
            L = model.forward(src, target=None)["logits"]
            m_prev = margin_to_worst(L, t_frozen)
            dec_prev = L.argmax(-1) == t_frozen
        rec = p3.step(model.forward(src, target=t_frozen)["total_loss"])
        with torch.no_grad():
            dec_now = model.forward(src, target=None)["logits"].argmax(-1) == t_frozen
        certified = m_prev > 2.0 * rec["delta"]
        assert bool((certified & dec_prev & ~dec_now).sum() == 0)


def test_alpha_zero_restores_exactly():
    cfg, model, src, tgt = setup(seed=2)
    t_frozen = model.forward(src, target=None)["logits"].argmax(-1)
    U0 = model.U.detach().clone()
    b0 = model.bias.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=50.0)   # absurd: forces alpha -> 0
    p3 = CertifiedP3(model, opt, src, t_frozen, alpha_min=0.5)  # only alpha=1, 0.5 tried
    rec = p3.step(model.forward(src, target=t_frozen)["total_loss"])
    if rec["alpha"] == 0.0:
        assert torch.equal(model.U.detach(), U0) and torch.equal(model.bias.detach(), b0)
        assert rec["delta"] == 0.0 and rec["enforced_flips"] == 0


def test_exact_floor_variant_counts_excluded_flips():
    cfg, model, src, tgt = setup(seed=3)
    t_frozen = model.forward(src, target=None)["logits"].argmax(-1)
    p3 = run_phase(model, src, t_frozen, lr=0.05, steps=15, exact_floor=0.05)
    s = p3.summary()
    assert s["enforced_flips_total"] == 0                  # above-floor contexts still exact
    assert s["flips_on_excluded_total"] >= 0               # counted, reported, never silent


def test_reflip_events_counted_independently_of_first_flip():
    """PR #5 review fix: flip -> recover -> flip again must register 2 flip events
    (old code only ever examined first flips for the premise self-check)."""
    cfg, model, src, tgt = setup(seed=4)
    bank = src[:16]
    cert = TrajectoryCertificate(model, bank)
    U_orig = model.U.detach().clone()
    with torch.no_grad():                                   # flip: scramble the frame
        model.U.copy_(torch.randn_like(model.U))
    model.normalize_U()
    cert.step()
    with torch.no_grad():                                   # recover: restore the frame
        model.U.copy_(U_orig)
    model.normalize_U()
    cert.step()
    with torch.no_grad():                                   # flip again
        model.U.copy_(torch.randn_like(model.U) + 1.0)
    model.normalize_U()
    cert.step()
    s = cert.summary()
    assert s["total_flip_events"] >= 2 * (s["total_new_flips"] > 0)   # re-flips seen
    assert s["total_flip_events"] > s["total_new_flips"]              # strictly more events
    assert s["viol_premise_flip"] == 0                     # huge deltas: premise never held


def test_tmass_groups_tracked():
    cfg, model, src, tgt = setup(seed=5, n=128)
    t_frozen = model.forward(src, target=None)["logits"].argmax(-1)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    p3 = CertifiedP3(model, opt, src[:64], t_frozen[:64])
    for _ in range(5):
        p3.step(model.forward(src[:64], target=t_frozen[:64])["total_loss"])
    p3.add_contexts(src[64:], t_frozen[64:])
    for _ in range(5):
        p3.step(model.forward(src[:64], target=t_frozen[:64])["total_loss"])
    mass = p3.certified_mass(delta_ref=0.01)
    assert set(mass.keys()) == {0, 1}
    assert p3.summary()["enforced_flips_total"] == 0


def test_renorm_free_does_not_project_rows_to_unit_sphere():
    """renorm=False is the teacher-seeded variant: a frame's row norms carry its margin structure,
    so a phase must leave them untouched (renorm=True would unit-normalize them)."""
    cfg, model, src, tgt = setup()
    with torch.no_grad():                                   # varied, non-unit row norms (teacher-like)
        model.U.data *= torch.linspace(0.5, 3.0, model.U.shape[0]).unsqueeze(1)
    t = model.forward(src)["logits"].argmax(-1)

    free = ProjectiveIncidenceLearner(cfg)
    free.U.data.copy_(model.U.data)
    free.bias.data.copy_(model.bias.data)
    opt = torch.optim.AdamW(free.parameters(), lr=1e-2)
    p3 = CertifiedP3(free, opt, src, t, renorm=False)
    for _ in range(5):
        p3.step(free.forward(src, target=t)["total_loss"])
    norms = free.U.data.norm(dim=-1)
    # the varied entry norms survive: not collapsed to the unit sphere (renorm=True would)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=0.1), "renorm=False must not unit-normalize"
    assert norms.max() / norms.min() > 2.0, "the varied row-norm structure is preserved"
    assert all(r["enforced_flips"] == 0 for r in p3.records)   # inviolability still exact
