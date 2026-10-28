import numpy as np

from pil.pdc_admission import affine_basis, constraints, margin_admission, solve_affine


def test_positive_admission_checks_the_actual_tail_guard():
    source = np.array([[1.0, 0.0], [0.0, 1.0]])
    frame = np.array([[1.0, 0.0], [-1.0, 0.0]])
    verdict = margin_admission(source, frame, np.zeros(2), 0, [0], [0, 1])
    assert verdict["admitted"]
    assert verdict["margin"] > verdict["gamma"] > 2 * verdict["delta"]
    assert verdict["full_strict_target"]


def test_target_must_be_inside_tail_bound_and_equality_is_not_admission():
    # A target-only tail drift would be missed by a rival-only bound.
    frame = np.array([[1.0, 0.0], [0.0, 1.0]])
    verdict = margin_admission(np.array([[1.0, 0.0], [-2.0, 0.0]]), frame,
        np.zeros(2), 0, [0], [0, 1])
    assert not verdict["admitted"]
    assert not verdict["full_strict_target"]
    boundary = margin_admission(np.array([[2.0, 0.0], [-2.0, 0.0]]), frame,
        np.zeros(2), 0, [0], [0, 1])
    assert boundary["margin"] == 2 * boundary["delta"]
    assert not boundary["admitted"]


def test_three_context_refusal_has_exact_farkas_duals_and_affine_bias():
    writes = np.array([[[1., 0.], [0., 1.]], [[1., 0.], [0., 0.]], [[0., 1.], [0., 0.]]])
    residuals = writes.sum(axis=1)
    candidates = [[0, 1], [1, 0], [1, 0]]
    basis = affine_basis(residuals, writes[0])
    for source in [0, 1]:
        matrix, rhs, _ = constraints(residuals, candidates, writes[0], 0, [source], 1., basis)
        result = solve_affine(matrix, rhs)
        assert result["status"] == "infeasible"
        assert result["dual_exact_rational_check"]
        assert result["dual_rhs"] < 0
    # An empty coalition here just means no extra LP requirement: test full coalition.
    matrix, rhs, _ = constraints(residuals, candidates, writes[0], 0, [0, 1], 1., basis)
    assert solve_affine(matrix, rhs)["status"] == "feasible"
