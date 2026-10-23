"""Cumulative deletion preserves the source path and measures non-additive damage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from xor_coalition_sweep import curve_summary, deleted_logits

from pil.learner import PILConfig, ProjectiveIncidenceLearner
from pil.proposer import RuleBank, build_sources


def test_subset_deletion_matches_independent_forward_and_keeps_raw_sources():
    model = ProjectiveIncidenceLearner(PILConfig(dim=4, n_propositions=4, device="cpu"))
    bank = RuleBank(3, 4, 3)
    z = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    A = torch.eye(3, 4)
    sources = build_sources(z, A, bank)
    original = sources.clone()
    expected = sources.clone()
    expected[:, [3, 5]] = 0
    assert torch.equal(deleted_logits(model, sources, 3, [0, 2]), model(expected)["logits"])
    assert torch.equal(sources, original)
    # Removing every rule must equal the raw-only forward, without deleting bias or atoms.
    assert torch.allclose(
        deleted_logits(model, sources, 3, [0, 1, 2]), model(build_sources(z, A, None))["logits"], atol=1e-6
    )
    assert torch.equal(deleted_logits(model, sources, 3, []), model(sources)["logits"])


def test_curve_summary_retains_recovery_and_never_crossed_cases():
    # A recovery does not erase an earlier threshold crossing. AUC uses the whole curve.
    result = curve_summary([0.9, 0.8, 0.95])
    assert result["first_material_deletions"] == 1
    assert result["auc"] == pytest.approx(0.8625)
    assert curve_summary([0.9, 0.89, 0.91])["first_material_deletions"] is None
