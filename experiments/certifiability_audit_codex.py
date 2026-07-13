"""Pre-registered corpus-side certifiability audit across seven domains.

This pilot builds no expert and performs no model forward pass.  It combines
already-recorded recovery/coverage outcomes with cheap corpus-only measures,
then applies the signed Spearman/permutation/bootstrap procedure unchanged.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "data" / "certifiability_audit_codex.json"

DOMAINS = ("wikitext", "wt103", "code", "sudoku", "bAbI", "SCAN", "elements")
MEASURES = (
    "register_density",
    "effective_output_rank",
    "hard_constraint_recovery",
    "linear_concept_coverage",
    "chain_recoverability_2hop",
)
SIGNS = {
    "register_density": 1,
    "effective_output_rank": -1,
    "hard_constraint_recovery": 1,
    "linear_concept_coverage": 1,
    "chain_recoverability_2hop": 1,
}
BOOTSTRAP_SEED = 2718
BOOTSTRAP_SAMPLES = 10_000
PERMUTATION_SEED = 1729
FIRES_RHO = 0.75

GROUND_TRUTH: dict[str, dict[str, Any]] = {
    "wikitext": {
        "value": 0.346,
        "exact_figure": "wikitext to a new arc best 0.346",
        "unit": "held-out core_sw agreement",
        "source": "docs/notes/wyly_domain_structure.md",
        "definition": (
            "Certified full-stack core_sw agreement on held-out pythia-70m "
            "wikitext windows under the note's common 80k-window protocol."
        ),
        "choice_note": (
            "Selected the later deep-discourse arc best 0.346; rejected the earlier "
            "refreshed-matrix 0.342 and original 0.329 as superseded results."
        ),
    },
    "wt103": {
        "value": 0.350,
        "exact_figure": "wt103 ... core_sw 0.350",
        "unit": "held-out core_sw agreement",
        "source": "docs/notes/wyly_domain_structure.md",
        "definition": (
            "Certified full-stack core_sw agreement on held-out pythia-70m "
            "WikiText-103 windows under the common 80k-window protocol."
        ),
        "choice_note": "Selected refreshed full-stack 0.350; original 0.341 was superseded.",
    },
    "code": {
        "value": 0.611,
        "exact_figure": "code ... core_sw 0.611",
        "unit": "held-out core_sw agreement",
        "source": "docs/notes/wyly_domain_structure.md",
        "definition": (
            "Certified full-stack core_sw agreement on held-out pythia-70m code "
            "windows under the common 80k-window protocol."
        ),
        "choice_note": (
            "Selected refreshed full-stack 0.611; rejected the later bracket-only "
            "probe's 0.5742 because that run isolates a residual intervention."
        ),
    },
    "sudoku": {
        "value": 0.520,
        "exact_figure": "sudoku ... core_sw 0.520",
        "unit": "held-out core_sw agreement",
        "source": "docs/notes/wyly_domain_structure.md",
        "definition": (
            "Certified full-stack core_sw agreement on held-out pythia-70m sudoku "
            "windows under the same domain-structure protocol."
        ),
        "choice_note": (
            "Selected the common-protocol 0.520. The later legality note's +0.090 "
            "is an intervention marginal, not total held-out recovery, so it is a predictor."
        ),
    },
    "bAbI": {
        "value": 0.998,
        "exact_figure": "served babi_qa3 bench 0.998 (998/1000)",
        "unit": "served held-out benchmark agreement",
        "source": "docs/notes/wyly_estate2.md",
        "definition": (
            "Agreement of the served multi-rule bAbI qa3 package on 1,000 held-out "
            "benchmark queries after the query-batch load-order fix."
        ),
        "choice_note": (
            "Selected qa3 because it is the harder multi-step bAbI task. Rejected qa1 "
            "1.000 because its benchmark is documented as saturated/contaminated; qa2 "
            "1.000 is reported as a supporting alternative."
        ),
    },
    "SCAN": {
        "value": 1.0,
        "exact_figure": "prim_compose 1.000; parse cover 1.000 (all three splits)",
        "unit": "held-out exact-match sequence recovery",
        "source": "docs/notes/scan_standalone.md",
        "definition": (
            "Exact full-action-sequence recovery by the compositional rule program on "
            "the official length, addprim_jump, and simple held-out splits."
        ),
        "choice_note": (
            "Used the final 1.000 result across all three splits; rejected the explicitly "
            "labeled earlier ceilings (0.916/0.935/0.539)."
        ),
    },
    "elements": {
        "value": 0.70,
        "exact_figure": "embedded | topk ... cov>=0.95 70.0%",
        "unit": "ground-truth feature coverage at AUC >= 0.95",
        "source": "/home/allans/code/sm-sae/README.md",
        "definition": (
            "Fraction of the 110 Standard-Model ground-truth features recovered by "
            "the canonical embedded TopK SAE at best-feature AUC at least 0.95."
        ),
        "choice_note": (
            "Used the canonical exact-sparsity embedded TopK cell. Rejected periodic-table "
            "package accuracy 0.751 because the task design explicitly names sm-sae as the "
            "elements substrate; did not select the maximum 71.8% L1 cell post hoc."
        ),
    },
}

MEASURE_DEFINITIONS = {
    "register_density": (
        "Fraction of in-scope next-token positions governed by a recorded hard "
        "constraint. Sudoku uses union-forced/all solution-cell rows; bAbI uses "
        "answer positions/all whitespace-token positions; SCAN uses deterministic "
        "grammar-governed output positions; wikitext is exactly zero because its "
        "registered arm is explicitly ALL-SOFT with no hard term."
    ),
    "effective_output_rank": (
        "exp(H), where H = sum_c p(c) H(next_byte | previous_byte=c), estimated "
        "from every adjacent byte pair in the named local training corpus. Lower raw "
        "rank means more certifiable and is sign-flipped in the composite."
    ),
    "hard_constraint_recovery": (
        "Recorded fraction of errors/query outputs recoverable by an already-certified "
        "or already-analysed hard constraint: bracket mate for code, union-forced for "
        "sudoku, estate2 state fold for bAbI, deterministic composition for SCAN, and "
        "zero for the explicitly no-hard-term wikitext arm."
    ),
    "linear_concept_coverage": (
        "Fraction of measurable ground-truth concepts linearly recoverable at AUC >= "
        "0.90 directly from a corpus/state representation, with no fresh model pass."
    ),
    "chain_recoverability_2hop": (
        "Incremental agreement of certified 2-hop chaining over the union of the "
        "1-hop and memorizer baselines on the rule's fired held-out subset."
    ),
    "composite": (
        "For each measure, population-z-score its non-null domain values, multiply by "
        "the registered sign so higher means more certifiable, then take each domain's "
        "mean over only its available aligned z-scores. Constant measures would receive "
        "z=0 for all available cells; no missing cell is imputed."
    ),
}

MEASURE_SOURCES: dict[str, dict[str, dict[str, str]]] = {
    "wikitext": {
        "register_density": {
            "source": "docs/notes/wikitext_accuracy.md",
            "detail": "ALL-SOFT endpoint; text has no hard term, so density is exactly 0.",
        },
        "effective_output_rank": {
            "source": "data/wikitext2_train.txt",
            "detail": "Fresh full-file adjacent-byte corpus pass using the registered formula.",
        },
        "hard_constraint_recovery": {
            "source": "docs/notes/wikitext_accuracy.md",
            "detail": "ALL-SOFT endpoint has no certified hard term, so recovery is exactly 0.",
        },
        "chain_recoverability_2hop": {
            "source": "docs/notes/khop_realtext.md",
            "detail": "codex recovery -0.1092 (agree_2hop 0.031 vs baseline 0.141).",
        },
    },
    "wt103": {
        "effective_output_rank": {
            "source": "data/wt103_train.txt",
            "detail": "Fresh full-file adjacent-byte corpus pass using the registered formula.",
        },
    },
    "code": {
        "effective_output_rank": {
            "source": "data/code_train.txt",
            "detail": "Fresh full-file adjacent-byte corpus pass using the registered formula.",
        },
        "hard_constraint_recovery": {
            "source": "docs/notes/code_legality_probe.md",
            "detail": "GATE = 76 / 5109 residual errors = 0.0148757.",
        },
    },
    "sudoku": {
        "register_density": {
            "source": "docs/notes/sudoku_forced_move.md",
            "detail": "Base union-forced fraction over all 1,359 solution-cell rows = 0.980.",
        },
        "effective_output_rank": {
            "source": "data/corpus_sudoku.txt",
            "detail": "Fresh full-file adjacent-byte corpus pass using the registered formula.",
        },
        "hard_constraint_recovery": {
            "source": "docs/notes/sudoku_forced_move.md",
            "detail": "GATE union-recoverable fraction of residual errors = 0.980.",
        },
    },
    "bAbI": {
        "register_density": {
            "source": "data/corpus_babi.txt",
            "detail": "Fresh count: answer positions divided by all whitespace-token positions.",
        },
        "effective_output_rank": {
            "source": "data/corpus_babi.txt",
            "detail": "Fresh full-file adjacent-byte corpus pass using the registered formula.",
        },
        "hard_constraint_recovery": {
            "source": "docs/notes/wyly_estate2.md",
            "detail": "Raw estate2/before feature is 1.000 on judge queries.",
        },
    },
    "SCAN": {
        "register_density": {
            "source": "docs/notes/scan_standalone.md",
            "detail": (
                "Parse cover 1.000 on all three splits; every parsed command's output "
                "positions are governed by the deterministic compositional grammar."
            ),
        },
        "effective_output_rank": {
            "source": "data/scan/length_split/tasks_train_length.txt",
            "detail": "Fresh full-file adjacent-byte corpus pass using the registered formula.",
        },
        "hard_constraint_recovery": {
            "source": "docs/notes/scan_standalone.md",
            "detail": "prim_compose exact-match and parse cover are 1.000 on all three splits.",
        },
    },
    "elements": {
        "register_density": {
            "source": "/home/allans/code/sm-sae/README.md",
            "detail": (
                "168/168 Standard-Model interaction vertices close on the seven-component "
                "conservation algebra; all recorded interaction outputs are hard-constrained."
            ),
        },
        "linear_concept_coverage": {
            "source": (
                "/home/allans/code/sm-sae/openspec/changes/archive/"
                "cascade-rollout-entropy-measurement/proposal.md"
            ),
            "detail": "state_t-direct linear probe: 65% of 74 measurable GT features at AUC >= 0.9.",
        },
    },
}

ABSENT_CELL_REASONS = {
    "wikitext": {
        "linear_concept_coverage": "No corpus-only recorded coverage; a fresh model pass is forbidden.",
    },
    "wt103": {
        "register_density": "No wt103-specific hard-token density was recorded.",
        "hard_constraint_recovery": "No wt103-specific certified hard-register recovery was recorded.",
        "linear_concept_coverage": "No corpus-only recorded coverage; a fresh model pass is forbidden.",
        "chain_recoverability_2hop": "No cheap wt103 B1 result was recorded.",
    },
    "code": {
        "register_density": (
            "The note records recovery among residual errors, not the all-position governed "
            "fraction required for density; it was not repurposed."
        ),
        "linear_concept_coverage": "No corpus-only recorded coverage; a fresh model pass is forbidden.",
        "chain_recoverability_2hop": "No cheap code B1 result was recorded.",
    },
    "sudoku": {
        "linear_concept_coverage": "No corpus-only recorded coverage; a fresh model pass is forbidden.",
        "chain_recoverability_2hop": "No cheap sudoku B1 result was recorded.",
    },
    "bAbI": {
        "linear_concept_coverage": "No corpus-only recorded coverage; a fresh model pass is forbidden.",
        "chain_recoverability_2hop": "No cheap bAbI B1 result was recorded.",
    },
    "SCAN": {
        "linear_concept_coverage": "No corpus-only recorded coverage; a fresh model pass is forbidden.",
        "chain_recoverability_2hop": "No cheap SCAN B1 result was recorded.",
    },
    "elements": {
        "effective_output_rank": (
            "sm-sae provides a state/interaction tensor substrate, not a compatible ordered "
            "next-token corpus; metadata serialization was not treated as a corpus."
        ),
        "hard_constraint_recovery": (
            "Conservation validates/prunes interactions but the notes do not report a distinct "
            "fraction of outputs uniquely recovered by it."
        ),
        "chain_recoverability_2hop": "No cheap Standard-Model B1 result was recorded.",
    },
}


def parse_cited_figure(text: str, metric_label: str) -> dict[str, Any]:
    """Parse one labeled decimal/percentage figure from a notes-style excerpt."""
    label_pattern = r"\s+".join(re.escape(part) for part in metric_label.split())
    pattern = re.compile(
        rf"{label_pattern}[^\n\d+-]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%)?",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None or match.group(1) is None:
        raise ValueError(f"could not find figure for {metric_label!r}")
    exact = re.sub(r"\s+", "", match.group(1))
    is_percent = exact.endswith("%")
    value = float(exact.removesuffix("%"))
    if is_percent:
        value /= 100.0
    return {"value": value, "exact_figure": exact, "unit": "%" if is_percent else "ratio"}


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average
        start = stop
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rho with average ranks for ties; None for an undefined correlation."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) < 2:
        return None
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    centered_x = [value - mean_x for value in rx]
    centered_y = [value - mean_y for value in ry]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y, strict=True)) / denominator


def permutation_p(
    x: Sequence[float],
    y: Sequence[float],
    *,
    seed: int = PERMUTATION_SEED,
    random_samples: int = 10_000,
) -> float | None:
    """Registered two-sided permutation p, exhaustive through n=8."""
    observed = spearman_rho(x, y)
    if observed is None:
        return None
    threshold = abs(observed) - 1e-12
    if len(y) <= 8:
        permutations: Any = itertools.permutations(y)
        total = 0
        extreme = 0
        for permuted in permutations:
            rho = spearman_rho(x, permuted)
            total += 1
            if rho is not None and abs(rho) >= threshold:
                extreme += 1
        return extreme / total

    rng = random.Random(seed)
    extreme = 0
    permuted = list(y)
    for _ in range(random_samples):
        rng.shuffle(permuted)
        rho = spearman_rho(x, permuted)
        if rho is not None and abs(rho) >= threshold:
            extreme += 1
    return extreme / random_samples


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float] | None:
    """Domain-pair bootstrap percentile CI, omitting undefined resamples."""
    if spearman_rho(x, y) is None:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(x)) for _ in x]
        estimate = spearman_rho(
            [x[index] for index in indices], [y[index] for index in indices]
        )
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def compute_composite(
    rows: Mapping[str, Mapping[str, float | None]],
    signs: Mapping[str, int] = SIGNS,
) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
    """Return available-z mean scores and aligned per-measure z-score cells."""
    aligned: dict[str, dict[str, float | None]] = {
        domain: {measure: None for measure in signs} for domain in rows
    }
    for measure, sign in signs.items():
        available = [
            float(row[measure])
            for row in rows.values()
            if row.get(measure) is not None
        ]
        if not available:
            continue
        mean = sum(available) / len(available)
        variance = sum((value - mean) ** 2 for value in available) / len(available)
        std = math.sqrt(variance)
        for domain, row in rows.items():
            value = row.get(measure)
            if value is None:
                continue
            zscore = 0.0 if std == 0.0 else (float(value) - mean) / std
            aligned[domain][measure] = sign * zscore

    scores: dict[str, float | None] = {}
    for domain, cells in aligned.items():
        available = [value for value in cells.values() if value is not None]
        scores[domain] = sum(available) / len(available) if available else None
    return scores, aligned


def effective_byte_rank(path: Path) -> tuple[float, int]:
    """Frequency-weighted empirical bigram conditional effective rank."""
    data = path.read_bytes()
    if len(data) < 2:
        raise ValueError(f"corpus must have at least two bytes: {path}")
    transitions: dict[int, Counter[int]] = defaultdict(Counter)
    for current, following in zip(data[:-1], data[1:], strict=True):
        transitions[current][following] += 1
    n_pairs = len(data) - 1
    mean_entropy = 0.0
    for counts in transitions.values():
        context_total = sum(counts.values())
        entropy = -sum(
            (count / context_total) * math.log(count / context_total)
            for count in counts.values()
        )
        mean_entropy += (context_total / n_pairs) * entropy
    return math.exp(mean_entropy), n_pairs


def babi_answer_density(path: Path) -> tuple[float, int, int]:
    """Fraction of whitespace-token positions that are bAbI answer positions."""
    text = path.read_text()
    answers = text.count(" A: ")
    positions = len(text.split())
    return answers / positions, answers, positions


def corpus_measures() -> tuple[dict[str, dict[str, float | None]], dict[str, Any]]:
    """Mine recorded cells and compute the registered cheap corpus passes."""
    rows = {domain: {measure: None for measure in MEASURES} for domain in DOMAINS}
    corpus_paths = {
        "wikitext": REPO / "data" / "wikitext2_train.txt",
        "wt103": REPO / "data" / "wt103_train.txt",
        "code": REPO / "data" / "code_train.txt",
        "sudoku": REPO / "data" / "corpus_sudoku.txt",
        "bAbI": REPO / "data" / "corpus_babi.txt",
        "SCAN": REPO / "data" / "scan" / "length_split" / "tasks_train_length.txt",
    }
    rank_pairs: dict[str, int] = {}
    for domain, path in corpus_paths.items():
        rank, n_pairs = effective_byte_rank(path)
        rows[domain]["effective_output_rank"] = rank
        rank_pairs[domain] = n_pairs

    answer_density, n_answers, n_positions = babi_answer_density(corpus_paths["bAbI"])
    rows["wikitext"]["register_density"] = 0.0
    rows["sudoku"]["register_density"] = 0.9801324503311258
    rows["bAbI"]["register_density"] = answer_density
    rows["SCAN"]["register_density"] = 1.0
    rows["elements"]["register_density"] = 1.0

    rows["wikitext"]["hard_constraint_recovery"] = 0.0
    rows["code"]["hard_constraint_recovery"] = 0.014875709532198082
    rows["sudoku"]["hard_constraint_recovery"] = 0.979982593559617
    rows["bAbI"]["hard_constraint_recovery"] = 1.0
    rows["SCAN"]["hard_constraint_recovery"] = 1.0

    rows["elements"]["linear_concept_coverage"] = 0.65
    rows["wikitext"]["chain_recoverability_2hop"] = -0.1092

    detail = {
        "effective_output_rank_pairs": rank_pairs,
        "bAbI_register_density_counts": {
            "answer_positions": n_answers,
            "all_whitespace_token_positions": n_positions,
        },
    }
    return rows, detail


def _statistic(
    predictor: Mapping[str, float | None],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    domains = [
        domain
        for domain in DOMAINS
        if predictor.get(domain) is not None and domain in GROUND_TRUTH
    ]
    x = [float(predictor[domain]) for domain in domains]
    y = [float(GROUND_TRUTH[domain]["value"]) for domain in domains]
    return {
        "rho": spearman_rho(x, y),
        "permutation_p_two_sided": permutation_p(x, y),
        "bootstrap_ci_95": bootstrap_ci(x, y, seed=bootstrap_seed),
        "n_used": len(domains),
        "domains_used": domains,
    }


def _ranked(scores: Mapping[str, float | None]) -> list[dict[str, Any]]:
    ordered = sorted(
        ((domain, score) for domain, score in scores.items() if score is not None),
        key=lambda item: (-float(item[1]), DOMAINS.index(item[0])),
    )
    return [
        {"rank": rank, "domain": domain, "score": score}
        for rank, (domain, score) in enumerate(ordered, start=1)
    ]


def run_audit() -> dict[str, Any]:
    """Run the fixed audit and return the JSON-ready artifact."""
    measures, corpus_detail = corpus_measures()
    composite, aligned_zscores = compute_composite(measures)

    statistics: dict[str, dict[str, Any]] = {}
    for index, measure in enumerate(MEASURES):
        predictor = {domain: measures[domain][measure] for domain in DOMAINS}
        statistics[measure] = _statistic(
            predictor, bootstrap_seed=BOOTSTRAP_SEED + index
        )
    statistics["composite"] = _statistic(
        composite, bootstrap_seed=BOOTSTRAP_SEED + len(MEASURES)
    )

    composite_stat = statistics["composite"]
    rho = composite_stat["rho"]
    ci = composite_stat["bootstrap_ci_95"]
    if rho is not None and rho >= FIRES_RHO and ci is not None and ci[0] > 0.0:
        verdict = "fires_directional"
        verdict_reason = (
            f"Composite rho={rho:.3f} clears 0.75 and bootstrap CI lower={ci[0]:.3f} "
            "is positive; n=7 remains weak-powered and directional."
        )
    else:
        verdict = "dead"
        reasons = []
        if rho is None or rho < FIRES_RHO:
            reasons.append("composite rho does not clear 0.75")
        if ci is None or ci[0] <= 0.0:
            reasons.append("composite bootstrap CI lower bound is <= 0")
        verdict_reason = "; ".join(reasons) + ". Directional only at n=7."

    domain_table = []
    for domain in DOMAINS:
        domain_table.append(
            {
                "domain": domain,
                "corpus_measures": measures[domain],
                "aligned_zscores": aligned_zscores[domain],
                "composite_score": composite[domain],
                "ground_truth": GROUND_TRUTH[domain],
            }
        )

    predicted_ranking = _ranked(composite)
    ground_truth_ranking = _ranked(
        {domain: GROUND_TRUTH[domain]["value"] for domain in DOMAINS}
    )
    computable = {
        measure: [domain for domain in DOMAINS if measures[domain][measure] is not None]
        for measure in MEASURES
    }
    absent = {
        domain: [measure for measure in MEASURES if measures[domain][measure] is None]
        for domain in DOMAINS
    }

    return {
        "schema_version": 1,
        "audit": "certifiability_audit_codex",
        "directional": True,
        "domains": domain_table,
        "statistics": statistics,
        "ranking": {
            "predicted_certifiability": predicted_ranking,
            "ground_truth": ground_truth_ranking,
            "rank_agreement": {
                "spearman_rho": composite_stat["rho"],
                "statement": (
                    "This is exactly the reported composite-vs-ground-truth Spearman rho, "
                    "because both compare the same two induced rank orderings."
                ),
            },
        },
        "provenance": {
            "preregistration": "/home/allans/code/PIL_CERTIFIABILITY_AUDIT_PREREG.md",
            "measure_definitions": MEASURE_DEFINITIONS,
            "measure_signs": SIGNS,
            "ground_truth_sources": GROUND_TRUTH,
            "n_domains": len(DOMAINS),
            "excluded_domains": [],
            "hard_stop_triggered": False,
            "computable_by_measure": computable,
            "absent_by_domain": absent,
            "measure_sources_by_domain": MEASURE_SOURCES,
            "absent_cell_reasons": ABSENT_CELL_REASONS,
            "corpus_pass_detail": corpus_detail,
            "permutation": {
                "procedure": (
                    "Exhaustive shuffle of ground-truth labels for every reported cell "
                    "because every n_used <= 8; two-sided fraction |rho_perm| >= |rho_obs|."
                ),
                "random_seed": PERMUTATION_SEED,
                "seed_note": "Recorded but unused because all permutations are exhaustive.",
            },
            "bootstrap": {
                "procedure": (
                    "Paired domain resampling with replacement, 10,000 draws; percentile "
                    "2.5/97.5 CI; undefined constant resamples omitted."
                ),
                "base_random_seed": BOOTSTRAP_SEED,
                "samples": BOOTSTRAP_SAMPLES,
                "caveat": "At n around 7 this percentile CI is wide and approximate.",
            },
            "ground_truth_consistency_concern": (
                "The four core_sw domains share one held-out agreement protocol, while bAbI/SCAN "
                "are served task exact-match and elements is sm-sae GT-feature cov95. All are clean "
                "recovery/coverage figures, but protocol heterogeneity makes the n=7 result only "
                "directional and may dominate rank differences."
            ),
            "linear_concept_drop_note": (
                "No fresh model pass was allowed. Linear-concept coverage is therefore null for "
                "six domains and uses only sm-sae's already-recorded corpus/state-direct 65% "
                "coverage at AUC >= 0.90 for elements."
            ),
            "verdict": verdict,
            "verdict_reason": verdict_reason,
        },
        "verdict": verdict,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_scoreboard(report: Mapping[str, Any]) -> None:
    """Print the full human-readable audit scoreboard."""
    print("CERTIFIABILITY AUDIT (empirical-directional)")
    print("=" * 79)
    print("Per-domain table (null means absent; no imputation)")
    header = (
        f"{'domain':<10} {'reg':>8} {'eff-rank':>10} {'hard-rec':>10} "
        f"{'linear':>8} {'2-hop':>8} {'comp':>9} {'GT':>8}"
    )
    print(header)
    for row in report["domains"]:
        measures = row["corpus_measures"]
        print(
            f"{row['domain']:<10} "
            f"{_fmt(measures['register_density']):>8} "
            f"{_fmt(measures['effective_output_rank']):>10} "
            f"{_fmt(measures['hard_constraint_recovery']):>10} "
            f"{_fmt(measures['linear_concept_coverage']):>8} "
            f"{_fmt(measures['chain_recoverability_2hop']):>8} "
            f"{_fmt(row['composite_score']):>9} "
            f"{_fmt(row['ground_truth']['value']):>8}"
        )
        gt = row["ground_truth"]
        print(f"  GT source: {gt['source']} — {gt['exact_figure']} [{gt['unit']}]")
        print(f"  definition: {gt['definition']}")

    print("\nCorrelation scoreboard")
    print(f"{'measure':<30} {'n':>3} {'rho':>9} {'perm-p':>10} {'bootstrap 95% CI':>24}")
    for measure in (*MEASURES, "composite"):
        statistic = report["statistics"][measure]
        ci = statistic["bootstrap_ci_95"]
        ci_text = "null" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(
            f"{measure:<30} {statistic['n_used']:>3} "
            f"{_fmt(statistic['rho']):>9} "
            f"{_fmt(statistic['permutation_p_two_sided'], 6):>10} "
            f"{ci_text:>24}"
        )

    print("\nPredicted certifiability ranking")
    for item in report["ranking"]["predicted_certifiability"]:
        print(f"  {item['rank']}. {item['domain']} (composite={item['score']:.4f})")
    print("Ground-truth ranking")
    for item in report["ranking"]["ground_truth"]:
        print(f"  {item['rank']}. {item['domain']} (recovery={item['score']:.4f})")
    print(
        "Rank agreement: rho="
        f"{_fmt(report['ranking']['rank_agreement']['spearman_rho'])}; "
        "identical to composite rho by construction."
    )

    provenance = report["provenance"]
    print(f"\nn_domains: {provenance['n_domains']}")
    print("Computable measures:")
    for measure, domains in provenance["computable_by_measure"].items():
        print(f"  {measure}: {', '.join(domains) if domains else 'none'}")
    print("Provenance summary:")
    print(f"  permutation: {provenance['permutation']['procedure']}")
    print(f"  bootstrap: {provenance['bootstrap']['procedure']}")
    print(f"  caveat: {provenance['ground_truth_consistency_concern']}")
    print(f"VERDICT: {report['verdict']}")
    print(f"Reason: {provenance['verdict_reason']}")


def main() -> None:
    report = run_audit()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print_scoreboard(report)


if __name__ == "__main__":
    main()
