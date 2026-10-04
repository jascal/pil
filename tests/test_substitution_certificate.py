import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from substitution_certificate import (  # noqa: E402
    SCALE,
    certificate_facts,
    certify_all_python,
    certify_all_souffle,
    certify_python,
    certify_souffle,
    coverage_report,
    margins_and_deltas,
)


def _case(seed, noise, n=400, vocab=60, dim=12):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(vocab, dim))
    r = rng.normal(size=(n, dim)) * 2
    r_hat = r + rng.normal(size=(n, dim)) * noise
    return r, r_hat, U


@pytest.mark.parametrize("noise", [0.01, 0.1, 0.5, 2.0])
def test_certified_contexts_never_change_their_argmax(noise):
    r, r_hat, U = _case(0, noise)
    report = coverage_report(r, r_hat, U, use_souffle=False)  # raises if a certified context flips
    assert 0.0 <= report["coverage"] <= 1.0


def test_coverage_shrinks_as_the_fit_degrades():
    covs = [coverage_report(*_case(1, noise), use_souffle=False)["coverage"] for noise in (0.01, 0.3, 3.0)]
    assert covs[0] > covs[1] > covs[2]


def test_exact_substitution_is_certified_and_boundary_is_refused():
    r, _, U = _case(2, 0.0)
    t, m, d = margins_and_deltas(r, r, U)
    assert (d == 0).all() and certify_python(m, d).all()
    # margin exactly 2*delta: the theorem needs a STRICT inequality, so it must be refused
    assert not certify_python(np.array([2.0]), np.array([1.0]))[0]


def test_fixed_point_rounding_is_conservative():
    m = np.array([2.0 + 1e-12])  # truly > 2*delta, but within float/fixed-point slack
    d = np.array([1.0])
    assert not certify_python(m, d)[0]
    assert certify_python(np.array([2.0 + 10 / SCALE + 1e-5]), d)[0]


@pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle on PATH")
@pytest.mark.parametrize("noise", [0.05, 0.5])
def test_souffle_query_matches_python_twin(noise):
    r, r_hat, U = _case(3, noise)
    _, m, d = margins_and_deltas(r, r_hat, U)
    assert (certify_souffle(m, d) == certify_python(m, d)).all()


@pytest.mark.parametrize("noise", [0.05, 0.4, 1.5])
def test_pairwise_is_exact_and_dominates_uniform_and_hybrid(noise):
    r, r_hat, U = _case(4, noise)
    facts = certificate_facts(r, r_hat, U, k=8)
    c = certify_all_python(facts)
    agree = facts["t_hat"] == facts["t"]
    assert not (c["uniform"] & ~c["pairwise"]).any()   # uniform_implies_pairwise
    assert not (c["hybrid"] & ~c["pairwise"]).any()    # hybrid is a sufficient condition
    assert not (c["pairwise"] & ~agree).any()          # sound
    # exact up to the rounding slack: an agreeing context is refused only when its slack is within tau
    refused_agreeing = agree & ~c["pairwise"]
    assert (facts["pslack"][refused_agreeing] < 1e-5).all()


def test_report_carries_all_three_forms():
    report = coverage_report(*_case(5, 0.2), use_souffle=False, k=8)
    assert {"uniform", "pairwise", "hybrid"} <= set(report)
    assert report["uniform"]["coverage"] == report["coverage"]
    assert report["uniform"]["coverage"] <= report["pairwise"]["coverage"]
    assert report["hybrid"]["coverage"] <= report["pairwise"]["coverage"]


@pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle on PATH")
def test_souffle_matches_python_for_all_three_certificates():
    facts = certificate_facts(*_case(6, 0.3), k=8)
    py, sf = certify_all_python(facts), certify_all_souffle(facts)
    for name in ("uniform", "pairwise", "hybrid"):
        assert (py[name] == sf[name]).all(), name
