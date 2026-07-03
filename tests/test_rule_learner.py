"""PICRuleLearner end-to-end: XOR must be solved exactly, hard == soft == exported."""

import torch

from pil.rule_learner import PICRuleLearner, RuleLearnerConfig
from pil.rules import RuleProgram, RuleProgramConfig


def xor_data(n=1500, seed=0):
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n, 2), generator=g)
    return bits + 10, (bits[:, 0] ^ bits[:, 1]) + 10


def test_xor_end_to_end():
    X, y = xor_data()
    cfg = RuleProgramConfig(vocab_size=50304, window=2, frame_dim=8, candidates=[10, 11], seed=0)
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=4, init_rules=16, births_per_phase=8, epochs_per_phase=15,
        max_rules=64, seed=0, verbose=False,
    )
    rep = PICRuleLearner(prog, lc).fit(X, y)
    assert rep.final["hard_val_acc"] == 1.0
    assert rep.final["soft_hard_agreement"] == 1.0
    assert rep.final["mean_binary_gap"] < 1e-3            # structure actually hardened


def test_data_seeded_rules_fire():
    """Random-in-data-measure init: newborn rules must be alive on the data."""
    X, y = xor_data()
    cfg = RuleProgramConfig(vocab_size=50304, window=2, frame_dim=8, candidates=[10, 11], seed=0)
    prog = RuleProgram(cfg)
    learner = PICRuleLearner(prog, RuleLearnerConfig(seed=0, verbose=False))
    learner.seed_rules(X, 16)
    fire = (torch.cat(prog.all_gates(X), dim=1) > 0.5).float().mean(dim=0)
    assert (fire > 0).all()                                # every rule fires somewhere
    assert fire.mean() > 0.05                              # and not vanishingly rarely


def test_feedback_utility_is_exact_ablation():
    """Utility is the exact NLL delta of removing a rule -- positive for a load-bearing
    rule, ~0 for a decode-neutral (zero-head) one. (Redundant ensembles legitimately give
    ~0 per-rule utility, so this pins the signal on a *non-redundant* program.)"""
    import torch as t

    from pil.rules import PRESENT

    X, y = xor_data()
    cfg = RuleProgramConfig(vocab_size=50304, window=2, frame_dim=8, candidates=[10, 11], seed=0)
    prog = RuleProgram(cfg)
    # one rule: fires on (1,0); head set manually to point hard at candidate index 1
    anchors = t.tensor([[11, 10], [10, 10]])
    rel = t.full((2, 2), PRESENT)
    sgn = t.full((2, 2), PRESENT)
    prog.add_rules_stratum1(anchors, rel, sgn)
    prog.round_structure()
    with t.no_grad():
        prog.heads[0][0] = 5.0 * prog.U[1]        # load-bearing
        # rule 1 keeps its zero head: decode-neutral
    learner = PICRuleLearner(prog, RuleLearnerConfig(seed=0, verbose=False))
    stats = learner.feedback(X, prog.target_index(y))
    assert float(stats["utility"][0]) > 0.05
    assert abs(float(stats["utility"][1])) < 1e-6
    # purity: rule 0 fires only on (1,0) contexts whose target is XOR=1 -> pure
    assert float(stats["purity"][0]) > 0.99
