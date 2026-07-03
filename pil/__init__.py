"""pil -- Projective Incidence Learning.

Learning dynamics for Projective Incidence Calculus (PIC): propose many parallel
partial-evidence vectors, gate them by margin / Gram kernel, and refine the
proposition frame toward higher retrievability and better conditioning.

Companion to fieldrun's PIC formalization (``fieldrun/PIC_PROPOSAL.md``).
"""

from __future__ import annotations

from .datalog_export import export_program, run_souffle, verify_export
from .geometry import (
    cosine_gram,
    frame_potential,
    gamma_decodable_count,
    gram_matrix,
    incidences,
    log10_packing_bound,
    logits_from_incidences,
    margin_to_worst,
    min_frame_separation,
    normalize_rows,
    participation_ratio,
    power_diagram_weights,
    welch_bound,
)
from .learner import (
    PILConfig,
    ProjectiveIncidenceLearner,
    create_synthetic_problem,
)
from .proposer import RuleBank, build_sources, rulebank_from_weights, targeted_rulebank
from .rule_learner import FitReport, PICRuleLearner, RuleLearnerConfig
from .rules import RuleProgram, RuleProgramConfig, certified_fraction, program_summary
from .schemas import (
    Schema,
    SchemaBank,
    arithmetic_library,
    copy_library,
    propose_schemas,
)
from .scoring import (
    ambiguity_resolution_score,
    candidate_activations,
    propose_and_select,
    select_top_k,
)
from .synthetic import (
    create_clustered_problem,
    create_compositional_problem,
    frame_alignment,
    within_cluster_cosine,
    within_cluster_report,
)
from .tokens import TokenSpace

__version__ = "0.0.1"

__all__ = [
    "PILConfig",
    "ProjectiveIncidenceLearner",
    "RuleProgram",
    "RuleProgramConfig",
    "program_summary",
    "certified_fraction",
    "PICRuleLearner",
    "RuleLearnerConfig",
    "FitReport",
    "TokenSpace",
    "export_program",
    "run_souffle",
    "verify_export",
    "Schema",
    "SchemaBank",
    "arithmetic_library",
    "copy_library",
    "propose_schemas",
    "create_synthetic_problem",
    "create_clustered_problem",
    "create_compositional_problem",
    "frame_alignment",
    "within_cluster_cosine",
    "within_cluster_report",
    "RuleBank",
    "build_sources",
    "targeted_rulebank",
    "rulebank_from_weights",
    "ambiguity_resolution_score",
    "candidate_activations",
    "propose_and_select",
    "select_top_k",
    "cosine_gram",
    "frame_potential",
    "gram_matrix",
    "incidences",
    "logits_from_incidences",
    "margin_to_worst",
    "normalize_rows",
    "participation_ratio",
    "power_diagram_weights",
    "welch_bound",
    "log10_packing_bound",
    "gamma_decodable_count",
    "min_frame_separation",
]
