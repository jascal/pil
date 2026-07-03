"""RuleProgram semantics: gates, threshold rules, stable ids, hardening."""

import torch

from pil.rules import ABSENT, PRESENT, RuleProgram, RuleProgramConfig, program_summary


def make_prog(**kw) -> RuleProgram:
    cfg = RuleProgramConfig(vocab_size=100, window=4, frame_dim=8, candidates=[10, 11], **kw)
    return RuleProgram(cfg)


def add_and_rule(prog, anchors, offsets, signs=None):
    """One AND rule: literals at ``offsets`` matching ``anchors`` (signs: True=pos)."""
    W = prog.cfg.window
    a = torch.zeros(1, W, dtype=torch.long)
    rel = torch.full((1, W), ABSENT)
    sgn = torch.full((1, W), PRESENT)
    for i, o in enumerate(offsets):
        a[0, o] = anchors[i]
        rel[0, o] = PRESENT
        if signs is not None and not signs[i]:
            sgn[0, o] = -PRESENT
    return prog.add_rules_stratum1(a, rel, sgn)


def test_and_gate_boolean_semantics():
    prog = make_prog()
    add_and_rule(prog, [5, 7], [0, 2])                     # x0==5 AND x2==7
    add_and_rule(prog, [5, 7], [0, 2], signs=[True, False])  # x0==5 AND x2!=7
    prog.round_structure()
    x = torch.tensor([[5, 0, 7, 0], [5, 0, 8, 0], [4, 0, 7, 0]])
    g = prog.all_gates(x, hard=True)[0]
    assert g[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert g[:, 1].tolist() == [0.0, 1.0, 0.0]
    # soft eval at hardened parameters agrees with Boolean
    gs = prog.all_gates(x, hard=False)[0]
    assert torch.allclose(gs.round(), g)


def test_threshold_gate_count_semantics():
    prog = make_prog()
    W = prog.cfg.window
    a = torch.tensor([[5, 5, 5, 5]])
    rel = torch.full((1, W), PRESENT)
    sgn = torch.full((1, W), PRESENT)
    prog.add_rules_stratum1(a, rel, sgn, thr=torch.tensor([2.5]), is_thresh=torch.tensor([True]))
    prog.round_structure()
    x = torch.tensor([[5, 5, 5, 0], [5, 5, 0, 0], [5, 5, 5, 5]])
    g = prog.all_gates(x, hard=True)[0]
    assert g[:, 0].tolist() == [1.0, 0.0, 1.0]             # count >= ceil(2.5) = 3
    gs = prog.all_gates(x, hard=False)[0]
    assert torch.allclose(gs.round(), g)


def test_deep_stratum_and_stable_ids():
    prog = make_prog()
    ids = add_and_rule(prog, [5], [0]) + add_and_rule(prog, [7], [1])
    si = prog.add_stratum()
    # deep rule: fired(ids[0]) AND NOT fired(ids[1])
    rel = torch.tensor([[PRESENT, PRESENT]])
    sgn = torch.tensor([[PRESENT, -PRESENT]])
    prog.add_rules_deep(si, rel, sgn)
    prog.round_structure()
    x = torch.tensor([[5, 7, 0, 0], [5, 0, 0, 0], [0, 7, 0, 0]])
    g = prog.all_gates(x, hard=True)
    assert g[1][:, 0].tolist() == [0.0, 1.0, 0.0]

    # prune the first stratum-1 rule: the deep rule's literal column must be dropped
    prog.prune({ids[0]})
    assert prog.n_rules == 2
    assert prog.strata[1].in_ids.tolist() == [ids[1]]
    g2 = prog.all_gates(x, hard=True)
    # deep rule now reads only NOT fired(ids[1])
    assert g2[1][:, 0].tolist() == [0.0, 1.0, 0.0]


def test_zero_head_birth_is_decode_neutral():
    prog = make_prog()
    x = torch.tensor([[5, 0, 7, 0]])
    L0 = prog(x)["logits"].clone()
    add_and_rule(prog, [5, 7], [0, 2])
    L1 = prog(x)["logits"]
    assert torch.allclose(L0, L1)


def test_extend_atoms_preserves_semantics():
    prog = make_prog()
    add_and_rule(prog, [5], [0])
    si = prog.add_stratum()
    prog.add_rules_deep(si, torch.tensor([[PRESENT]]), torch.tensor([[PRESENT]]))
    x = torch.tensor([[5, 0, 0, 0], [4, 0, 0, 0]])
    before = prog.all_gates(x, hard=True)[1]
    new = add_and_rule(prog, [9], [3])
    prog.strata[si].extend_atoms(new)
    after = prog.all_gates(x, hard=True)[1]
    assert torch.equal(before, after)      # newly-exposed atoms arrive structurally absent


def test_eq_atoms_relational_literal():
    import torch as t

    from pil.rules import ABSENT as AB
    from pil.rules import PRESENT as PR

    cfg = RuleProgramConfig(
        vocab_size=100, window=3, frame_dim=8, candidates=[10, 11], eq_atoms=True
    )
    prog = RuleProgram(cfg)
    s1 = prog.strata[0]
    assert s1.eq_pairs.tolist() == [[0, 1], [0, 2], [1, 2]]
    # rule: x0 == x2 (eq pair index 1 -> literal column 3+1=4), no token literals
    rel = t.full((1, s1.n_lits), AB)
    rel[0, 3 + 1] = PR
    sgn = t.full((1, s1.n_lits), PR)
    prog.add_rules_stratum1(t.zeros(1, 3, dtype=t.long), rel, sgn)
    # negated eq: x0 != x2
    sgn2 = sgn.clone()
    sgn2[0, 3 + 1] = -PR
    prog.add_rules_stratum1(t.zeros(1, 3, dtype=t.long), rel.clone(), sgn2)
    prog.round_structure()
    x = t.tensor([[7, 1, 7], [7, 1, 8]])
    g = prog.all_gates(x, hard=True)[0]
    assert g[:, 0].tolist() == [1.0, 0.0]
    assert g[:, 1].tolist() == [0.0, 1.0]
    gs = prog.all_gates(x, hard=False)[0]
    assert t.allclose(gs.round(), g)


def test_ordinal_le_literal():
    import torch as t

    from pil.rules import ABSENT as AB
    from pil.rules import PRESENT as PR

    prog = make_prog()
    W = prog.cfg.window
    rel = t.full((2, W), AB)
    sgn = t.full((2, W), PR)
    rel[:, 1] = PR
    sgn[1, 1] = -PR                       # rule 1: x1 > 5
    is_le = t.zeros(2, W, dtype=t.bool)
    is_le[:, 1] = True
    anchors = t.full((2, W), 5, dtype=t.long)
    prog.add_rules_stratum1(anchors, rel, sgn, is_le=is_le)
    prog.round_structure()
    x = t.tensor([[0, 3, 0, 0], [0, 5, 0, 0], [0, 7, 0, 0]])
    g = prog.all_gates(x, hard=True)[0]
    assert g[:, 0].tolist() == [1.0, 1.0, 0.0]    # x1 <= 5
    assert g[:, 1].tolist() == [0.0, 0.0, 1.0]    # x1 > 5
    gs = prog.all_gates(x, hard=False)[0]
    assert t.allclose(gs.round(), g)


def test_target_index_and_summary():
    prog = make_prog()
    idx = prog.target_index(torch.tensor([10, 11, 10]))
    assert idx.tolist() == [0, 1, 0]
    s = program_summary(prog)
    assert s["n_rules"] == 0
