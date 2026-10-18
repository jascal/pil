"""Pole labels come from head_incidences of a real RuleProgram, not from a constant."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from pil.pole import ablated_incidence, decide_delta, incidence_logits, pole_label
from pil.rules import PRESENT, RuleProgram, RuleProgramConfig


def _program_with_head(row: torch.Tensor) -> RuleProgram:
    v = int(row.numel())
    cfg = RuleProgramConfig(
        vocab_size=v, window=1, frame_dim=v, candidates=list(range(v)), seed=0, device="cpu"
    )
    prog = RuleProgram(cfg)
    with torch.no_grad():
        prog.U.copy_(torch.eye(v))
    prog.add_rules_stratum1(
        torch.zeros(1, 1, dtype=torch.long),
        torch.full((1, 1), PRESENT),
        torch.full((1, 1), PRESENT),
    )
    with torch.no_grad():
        prog.heads[0].copy_(row.reshape(1, v))
    return prog


def _label_from_program(row: torch.Tensor):
    prog = _program_with_head(row)
    incidence = prog.head_incidences()[0]
    assert torch.allclose(incidence, row.flatten(), atol=1e-5)
    return pole_label(incidence)


def test_bipolar_thin_and_scattered_rows_disagree():
    bipolar = _label_from_program(torch.tensor([1.0, -1.0]))
    thin = _label_from_program(torch.tensor([1.0, -1e-4]))
    scattered = _label_from_program(torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0]))
    assert bipolar[3] == "bipolar" and bipolar[1] == 1 and bipolar[0] > 0.05
    assert thin[3] == "thin" and thin[0] <= 0.05
    assert scattered[3] == "scattered" and scattered[2] <= 0.5
    assert bipolar[3] != thin[3] != scattered[3]


def test_all_neg_flips_the_hard_argmax_and_scattered_neg_holds():
    prog = _program_with_head(torch.tensor([1.0, -1.0]))
    with torch.no_grad():
        prog.bias[1] = 1.5
    x = torch.zeros(1, 1, dtype=torch.long)
    incidences = prog.head_incidences()
    before = incidence_logits(prog, x, incidences)
    forward = prog.forward(x, hard=True)["logits"]
    assert torch.equal(before.argmax(dim=-1), forward.argmax(dim=-1))
    after_all = incidence_logits(prog, x, ablated_incidence(incidences, "all_neg"))
    after_scattered = incidence_logits(prog, x, ablated_incidence(incidences, "scattered_neg"))
    assert decide_delta(before, after_all)[0] == 1
    assert int(before.argmax()) == 0 and int(after_all.argmax()) == 1
    assert decide_delta(before, after_scattered)[0] == 0


def test_bipolar_neg_zeros_only_the_bipolar_row():
    row_bipolar = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0])
    row_scattered = torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0])
    prog = _program_with_head(row_bipolar)
    prog.add_rules_stratum1(
        torch.zeros(1, 1, dtype=torch.long),
        torch.full((1, 1), PRESENT),
        torch.full((1, 1), PRESENT),
    )
    with torch.no_grad():
        prog.heads[0][1].copy_(row_scattered)
    incidences = prog.head_incidences()
    assert pole_label(incidences[0])[3] == "bipolar"
    assert pole_label(incidences[1])[3] == "scattered"
    edited = ablated_incidence(incidences, "bipolar_neg")
    assert torch.allclose(edited[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(edited[1], incidences[1])


def test_modadd_builder_uses_97_residue_tokens():
    pkg = Path("/home/allans/code/rosetta/models/pythia160m")
    if not pkg.is_dir():
        pytest.skip("rosetta package is not on this machine")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from rule_learner_bench import build_modadd

    _xtr, _ytr, _xte, _yte, prog, _lc, _accepted, _scores = build_modadd()
    assert int(prog.candidate_ids.numel()) == 97
