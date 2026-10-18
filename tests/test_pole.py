"""Pole labels come from head_incidences of a real RuleProgram, not from a constant."""

from __future__ import annotations

import torch

from pil.pole import pole_label
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
