"""Probe E — certifiability-audit pilot (corpus-side measures → recovery rank).

Pre-registered in PIL_CERTIFIABILITY_AUDIT_PREREG.md (SIGNED 2026-07-13) BEFORE numbers.
Implements that prereg exactly: no new expert, no model forward pass, corpus-side +
already-recorded figures only. ρ bar 0.75 / composite definition / directional reading
are fixed by the prereg — not tuned here.

Ground-truth definition (ONE, applied everywhere a domain is kept):
  The certified/rule-based served tier's own accuracy on its held-out/test evaluation,
  as most recently recorded, expressed as a fraction in [0, 1] — NOT a crystal ratio
  (core_sw/student), NOT residual-error recovery fractions, NOT admission marginals.

Corpus-side measure definitions (predictors):
  1. register_density — with fixed local context order k=CONTEXT_ORDER, for every
     position t compute the empirical set of continuations observed for
     tokens[t-k:t] elsewhere in the SAME corpus (held-in-corpus n-gram determinism
     scan). A position is "hard" iff that context was seen >= 2 times AND has
     exactly one distinct continuation. register_density = hard / scored, where
     scored positions are those whose context was seen >= 2 times. Singleton
     contexts (n=1) are vacuously non-forcing and excluded from the denominator.
  2. effective_output_rank — for each context seen >= 2 times, empirical next-token
     distribution p, Shannon entropy H = -sum p log p (nats), effective rank =
     exp(H). Domain number = occurrence-weighted mean of exp(H) across scored
     contexts (token-position-weighted, not type-unweighted). Lower rank = more
     deterministic = more certifiable → sign-flipped (negated) before composite /
     correlation so higher = more certifiable.
  3. hard_constraint_recovery_pct — MINED from residual/register notes where present;
     ABSENT otherwise (never fabricated).
  4. linear_concept_coverage — DROPPED (needs model/SAE activations; prereg open
     decision 2: if fresh model pass, drop; keep audit corpus-side).
  5. chain_recoverability_2hop — MINED where recorded (wikitext only in this set);
     ABSENT otherwise.
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "experiments" / "data" / "certifiability_audit.json"

# House pattern from experiments/campaign_frontier_rows.py
PERMUTATIONS = 10_000
BOOTSTRAPS = 5_000
SEED = 20260713  # prereg date; fixed for reproducibility
CONTEXT_ORDER = 4  # local context order k for n-gram measures
MIN_DOMAINS_FOR_CORRELATION = 5
FIRES_RHO_BAR = 0.75

TOKENIZER_NOTE = (
    "Whitespace split via str.split() (Python default whitespace tokenization). "
    "Applied uniformly to every domain; no domain-specific tokenizer."
)

GROUND_TRUTH_DEFINITION = (
    "the certified/rule-based served tier's own accuracy on its held-out/test "
    "evaluation, as most recently recorded, expressed as a fraction in [0, 1]"
)

# ---------------------------------------------------------------------------
# Ground truth (mined + cited from docs/notes/; ONE definition, no force-fit)
# ---------------------------------------------------------------------------

# Each entry: value matches GROUND_TRUTH_DEFINITION, or excluded with reason.
MINED_GROUND_TRUTH: dict[str, dict[str, Any]] = {
    "wikitext": {
        "value": 0.346,
        "source_file": "docs/notes/wyly_domain_structure.md",
        "locator": (
            "section 'Deep-discourse roles': 'taking wikitext to a new arc best 0.346' "
            "(latest core_sw-class held-out certified accuracy; later sections only "
            "restate crystal 99.7% without a newer core figure)"
        ),
        "quoted": "taking wikitext to a **new arc best 0.346**",
        "definition_match": (
            "core_sw-class certified-tier held-out accuracy under the full Wyly stack; "
            "latest explicit core figure after the refreshed-matrix 0.342"
        ),
        "excluded": False,
        "not_used": (
            "crystal 99.7% is a ratio (core_sw/student), not raw held-out accuracy — "
            "rejected to keep units consistent with other domains"
        ),
    },
    "wt103": {
        "value": 0.350,
        "source_file": "docs/notes/wyly_domain_structure.md",
        "locator": (
            "section 'The refreshed matrix: the threshold dissolves' table row "
            "wt103 core_sw=0.350 (no later core_sw override for wt103 in this note)"
        ),
        "quoted": "wt103  0.400  0.316  73.1%    0.357    0.350    98.0%",
        "definition_match": (
            "core_sw = certified/rule tier held-out accuracy under the complete stack"
        ),
        "excluded": False,
        "not_used": "crystal 98.0% is a ratio, not used",
    },
    "code": {
        "value": 0.611,
        "source_file": "docs/notes/wyly_domain_structure.md",
        "locator": (
            "section 'The refreshed matrix' table row code core_sw=0.611 "
            "(latest domain-structure core for code; supersedes original 0.605)"
        ),
        "quoted": "code  0.251  0.692  91.9%    0.584    0.611   104.7%",
        "definition_match": (
            "core_sw = certified/rule tier held-out accuracy under the complete stack"
        ),
        "excluded": False,
        "not_used": (
            "code_legality_probe.md regenerated core_sw ~0.574 is a residual-probe "
            "re-run for GATE analysis, not the domain-structure recovery figure; "
            "GATE 0.0149 is hard-constraint-recovery predictor, not ground truth"
        ),
    },
    "sudoku": {
        "value": 0.520,
        "source_file": "docs/notes/wyly_domain_structure.md",
        "locator": (
            "section 'The constraint/planning wing' table row sudoku core_sw=0.520"
        ),
        "quoted": "sudoku  0.269  0.307  99.9%    0.544    0.520    95.6%",
        "definition_match": (
            "core_sw under the same Wyly domain-structure protocol as "
            "wikitext/wt103/code — held-out certified-tier accuracy"
        ),
        "excluded": False,
        "not_used": (
            "sudoku_forced_move.md union-recoverable residual fraction 0.980 is "
            "'fraction of memorizer errors a constraint register could fix' — a "
            "different quantity (error-recovery, not held-out accuracy); reserved as "
            "hard-constraint-recovery predictor. legality_certificate.md Soufflé "
            "cert 100% / judge marginal +0.090 is admission marginal on late-reveal "
            "cells only, not the cross-domain held-out accuracy definition."
        ),
    },
    "bAbI": {
        "value": 0.527,
        "source_file": "docs/notes/wyly_babi.md",
        "locator": (
            "section 'The region judge': 'Package on unseen test: **0.527** (from 0.509)' "
            "— 1,000 questions over 200 unseen test stories"
        ),
        "quoted": "Package on unseen test: **0.527** (from 0.509)",
        "definition_match": (
            "wyly package (certified/rule served tier) accuracy on the held-out "
            "bAbI qa1 benchmark — same type as core_sw held-out accuracy"
        ),
        "excluded": False,
        "not_used": (
            "0.509 is the pre-region-judge figure (superseded by 0.527). ERRATUM "
            "1.000 on current on-disk package is bench saturation/contamination "
            "(verbatim story-prefixes; cannot distinguish binding from memorization) "
            "— not genuine held-out recovery. qa1_config_holdout.md coverage/precision "
            "1.0/1.0 is a hand-authored rule on one synthetic held-out-config residual "
            "slice, not domain-level package accuracy."
        ),
    },
    "SCAN": {
        "value": 1.0,
        "source_file": "docs/notes/scan_standalone.md",
        "locator": (
            "Scoreboard after residual bare leaves (prim_compose exact-match 1.000 on "
            "length, addprim_jump, simple) + 'Residual templates' closing simple to "
            "1.000; domain figure = unweighted mean across the three official splits"
        ),
        "quoted": (
            "prim_compose exact-match 1.000 / 1.000 / 1.000 on length / addprim_jump / "
            "simple (after residual bare leaves); residual templates close simple to 1.000"
        ),
        "definition_match": (
            "certified/compositional rule stack exact-match accuracy on held-out test "
            "— SCAN analogue of core_sw; aggregated as mean of the three official splits "
            "because no single split is designated THE domain metric and the domain "
            "question is multi-split"
        ),
        "excluded": False,
        "split_values": {
            "length": 1.0,
            "addprim_jump": 1.0,
            "simple": 1.0,
        },
        "aggregation": "unweighted mean of three official splits",
        "not_used": (
            "Intermediate learned-stack tables (0.916/0.935/0.291) predate residual "
            "bare leaves / residual templates; exact-dictionary baselines (~0) are not "
            "compositional rule recovery"
        ),
    },
    "elements": {
        "value": 0.751,
        "source_file": "docs/notes/wyly_element_expert.md",
        "locator": (
            "section 'The package vs its teacher' table: k≤9 package overall **0.751** "
            "on the 708-cloze periodic-table benchmark"
        ),
        "quoted": "**k≤9** | **0.751** | **beats the teacher's 0.732**",
        "definition_match": (
            "wyly package accuracy on the held-out 708-cloze benchmark — certified/"
            "rule served tier accuracy (pil-native elements domain, not sm-sae)"
        ),
        "excluded": False,
        "not_used": (
            "Teacher 0.732 is parametric, not certified-rule recovery; sibling "
            "repo sm-sae is Standard Model of physics with no comparable recovery figure"
        ),
    },
}

# Mined predictors (not ground truth)
HARD_CONSTRAINT_RECOVERY: dict[str, dict[str, Any]] = {
    "code": {
        "value": 0.0149,
        "source_file": "docs/notes/code_legality_probe.md",
        "locator": "GATE = mate-recoverable/all-residual = 0.0149 (cross-vendor)",
        "status": "mined",
    },
    "sudoku": {
        "value": 0.980,
        "source_file": "docs/notes/sudoku_forced_move.md",
        "locator": (
            "GATE: absolute union-recoverable fraction of residual errors = 0.980 "
            "(not used as ground truth; reserved as hard-constraint-recovery predictor)"
        ),
        "status": "mined",
    },
}

CHAIN_RECOVERABILITY: dict[str, dict[str, Any]] = {
    "wikitext": {
        "value": -0.11,
        "source_file": "docs/notes/khop_realtext.md",
        "locator": "B1 #110: recovery −0.11 (cross-vendor; grok −0.1107 / codex −0.1092)",
        "status": "mined",
    },
}

CORPUS_PATHS: dict[str, list[Path]] = {
    "wikitext": [REPO / "data" / "wikitext2_train.txt"],
    "wt103": [REPO / "data" / "wt103_train.txt"],
    "code": [REPO / "data" / "code_train.txt"],
    "sudoku": [REPO / "data" / "corpus_sudoku.txt"],
    "bAbI": [REPO / "data" / "corpus_babi.txt"],
    "SCAN": [
        REPO / "data" / "scan" / "length_split" / "tasks_train_length.txt",
        REPO / "data" / "scan" / "add_prim_split" / "tasks_train_addprim_jump.txt",
        REPO / "data" / "scan" / "simple_split" / "tasks_train_simple.txt",
    ],
    "elements": [REPO / "data" / "corpus_elements.txt"],
}

DOMAIN_ORDER = ["wikitext", "wt103", "code", "sudoku", "bAbI", "SCAN", "elements"]


def tokenize(text: str) -> list[str]:
    """Uniform whitespace tokenizer for all domains."""
    return text.split()


def parse_core_sw_table_line(line: str) -> dict[str, Any] | None:
    """Parse a domain-structure notes table row into corpus + core_sw (+ optional fields).

    Expected shape (whitespace-separated after optional leading spaces)::

        code  0.251  0.692  91.9%    0.584    0.611   104.7%

    Columns: corpus gzip gold [copy%] student core_sw crystal [...]
    Returns None if the line is not a data row.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("corpus") or stripped.startswith("---"):
        return None
    # drop trailing parenthetical notes
    stripped = re.sub(r"\s{2,}\(.*\)$", "", stripped)
    parts = stripped.split()
    if len(parts) < 6:
        return None
    corpus = parts[0]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", corpus):
        return None
    # Prefer the long form with copy%: corpus gzip gold copy% student core_sw crystal
    # Short form without copy%: corpus gzip gold student core_sw crystal
    nums: list[float] = []
    for p in parts[1:]:
        if p.endswith("%"):
            try:
                nums.append(float(p[:-1]) / 100.0)
            except ValueError:
                return None
        else:
            try:
                nums.append(float(p))
            except ValueError:
                return None
    if len(nums) < 5:
        return None
    # Heuristic: if 3rd numeric looks like a percentage-as-fraction from copy%
    # (was already converted) and we have >=6 floats, long form.
    if len(nums) >= 6:
        # gzip, gold, copy, student, core_sw, crystal
        core_sw = nums[4]
        crystal = nums[5]
        student = nums[3]
    else:
        # gzip, gold, student, core_sw, crystal
        core_sw = nums[3]
        crystal = nums[4]
        student = nums[2]
    if not (0.0 <= core_sw <= 1.5):
        return None
    return {
        "corpus": corpus,
        "core_sw": core_sw,
        "crystal": crystal,
        "student": student,
        "source_line": line.rstrip("\n"),
    }


def load_tokens(paths: list[Path]) -> list[str]:
    tokens: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        tokens.extend(tokenize(text))
    return tokens


def ngram_corpus_measures(
    tokens: list[str], k: int = CONTEXT_ORDER
) -> dict[str, Any]:
    """Compute register_density and occurrence-weighted mean effective output rank.

    Formulas (verbatim):
      For each position t in [k, n):
        context = tokens[t-k:t]; continuation = tokens[t]
      Let C(ctx) be the multiset of continuations observed for ctx in the corpus.
      scoreable iff |C(ctx)| >= 2 (occurrence count).
      hard iff scoreable and |unique C(ctx)| == 1.
      register_density = (# hard positions) / (# scoreable positions)
      scoreable_fraction = (# scoreable positions) / (# positions with a full context)

      For each scoreable context with empirical p over continuations:
        H = -sum_i p_i * log(p_i)   (nats)
        effective_rank(ctx) = exp(H)
      mean_effective_rank = sum_ctx  n_ctx * exp(H_ctx)  /  sum_ctx n_ctx
        (n_ctx = occurrence count of context; token-position-weighted mean)
    """
    n = len(tokens)
    if n <= k:
        return {
            "register_density": float("nan"),
            "scoreable_fraction": 0.0,
            "n_positions": 0,
            "n_scoreable": 0,
            "n_hard": 0,
            "mean_effective_rank": float("nan"),
            "n_tokens": n,
            "k": k,
        }

    # context -> Counter of next tokens
    cont: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for t in range(k, n):
        ctx = tuple(tokens[t - k : t])
        cont[ctx][tokens[t]] += 1

    n_positions = n - k
    n_scoreable = 0
    n_hard = 0
    weighted_rank_sum = 0.0
    weight_sum = 0

    for t in range(k, n):
        ctx = tuple(tokens[t - k : t])
        counts = cont[ctx]
        total = sum(counts.values())
        if total < 2:
            continue
        n_scoreable += 1
        if len(counts) == 1:
            n_hard += 1

    # occurrence-weighted mean effective rank over scored *contexts* (by n_ctx),
    # equivalent to position-weighted mean over scoreable positions
    for counts in cont.values():
        total = sum(counts.values())
        if total < 2:
            continue
        entropy = 0.0
        for c in counts.values():
            p = c / total
            entropy -= p * math.log(p)
        rank = math.exp(entropy)
        weighted_rank_sum += total * rank
        weight_sum += total

    register_density = (n_hard / n_scoreable) if n_scoreable else float("nan")
    scoreable_fraction = n_scoreable / n_positions if n_positions else 0.0
    mean_eff = (weighted_rank_sum / weight_sum) if weight_sum else float("nan")

    return {
        "register_density": register_density,
        "scoreable_fraction": scoreable_fraction,
        "n_positions": n_positions,
        "n_scoreable": n_scoreable,
        "n_hard": n_hard,
        "mean_effective_rank": mean_eff,
        "n_tokens": n,
        "k": k,
    }


def spearman_rho(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Spearman rank correlation; raises if undefined (constant input)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        r, _ = spearmanr(x, y)
    if r is None or (isinstance(r, float) and math.isnan(r)):
        raise ValueError("spearman rho undefined (constant ranks)")
    return float(r)


def spearman_with_perm_bootstrap(
    measure: list[float],
    ground_truth: list[float],
    *,
    permutations: int = PERMUTATIONS,
    bootstraps: int = BOOTSTRAPS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Spearman ρ + prereg one-sided p + frontier-style two-sided p + percentile CI.

    Prereg calls for: fraction of shuffles with rho >= observed (one-sided, higher-is-
    better for the audit). House pattern in campaign_frontier_rows.py also reports a
    two-sided permutation p (fraction where |stat| is as extreme as |observed|, with
    the +1 Laplace correction). Both are reported and labeled.
    """
    n = len(measure)
    if n < 2:
        return {
            "rho": None,
            "n": n,
            "permutation_p_rho_ge_observed": None,
            "permutation_p_two_sided": None,
            "bootstrap_95_ci": None,
            "bootstrap_method": "percentile",
            "permutations": permutations,
            "bootstraps": bootstraps,
            "seed": seed,
            "note": "n < 2; correlation undefined",
        }

    m = np.asarray(measure, dtype=np.float64)
    g = np.asarray(ground_truth, dtype=np.float64)
    try:
        observed = spearman_rho(m, g)
    except ValueError:
        return {
            "rho": None,
            "n": n,
            "permutation_p_rho_ge_observed": None,
            "permutation_p_two_sided": None,
            "bootstrap_95_ci": None,
            "bootstrap_method": "percentile",
            "permutations": permutations,
            "bootstraps": bootstraps,
            "seed": seed,
            "note": "rho undefined on observed sample",
        }

    rng = np.random.default_rng(seed)
    count_ge = 0
    count_abs = 0
    for _ in range(permutations):
        shuffled = rng.permutation(g)
        try:
            r = spearman_rho(m, shuffled)
        except ValueError:
            continue
        if r >= observed - 1e-15:
            count_ge += 1
        if abs(r) >= abs(observed) - 1e-15:
            count_abs += 1
    # +1 correction matching campaign_frontier_rows.py
    p_ge = (count_ge + 1) / (permutations + 1)
    p_two = (count_abs + 1) / (permutations + 1)

    samples: list[float] = []
    for _ in range(bootstraps):
        idx = rng.integers(0, n, size=n)
        try:
            samples.append(spearman_rho(m[idx], g[idx]))
        except ValueError:
            continue
    if samples:
        lo, hi = np.percentile(samples, [2.5, 97.5])
        ci: list[float] | None = [float(lo), float(hi)]
    else:
        ci = None

    return {
        "rho": observed,
        "n": n,
        "permutation_p_rho_ge_observed": p_ge,
        "permutation_p_two_sided": p_two,
        "bootstrap_95_ci": ci,
        "bootstrap_method": "percentile",
        "permutations": permutations,
        "bootstraps": bootstraps,
        "seed": seed,
        "p_definitions": {
            "permutation_p_rho_ge_observed": (
                "fraction of GT-label shuffles with rho >= observed "
                "(+1/(N+1) correction); the prereg's stated p"
            ),
            "permutation_p_two_sided": (
                "fraction of shuffles with |rho| >= |observed| (+1/(N+1)); "
                "frontier_rows-style two-sided handling"
            ),
        },
    }


def zscore(values: list[float]) -> list[float]:
    """Per-measure z-score across domains that have the measure (mean 0, std 1)."""
    arr = np.asarray(values, dtype=np.float64)
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=0))
    if sigma == 0.0 or math.isnan(sigma):
        return [0.0 for _ in values]
    return [float((v - mu) / sigma) for v in values]


def build_composite(
    domain_measure_values: dict[str, dict[str, float | None]],
    measure_keys: list[str],
    lower_is_better: set[str],
) -> tuple[dict[str, float], dict[str, list[str]], dict[str, dict[str, float]]]:
    """Z-score each measure (sign-aligned so higher=more certifiable), mean per domain.

    A domain missing some measures still gets a composite from the ones it has.
    effective_output_rank (and any lower_is_better key) is negated before z-scoring.
    """
    # Collect present values per measure (sign-aligned raw)
    aligned: dict[str, dict[str, float]] = {m: {} for m in measure_keys}
    for domain, meas in domain_measure_values.items():
        for m in measure_keys:
            v = meas.get(m)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            aligned[m][domain] = -float(v) if m in lower_is_better else float(v)

    # Z per measure over domains that have it
    z_by_domain_measure: dict[str, dict[str, float]] = {
        d: {} for d in domain_measure_values
    }
    for m, by_dom in aligned.items():
        if not by_dom:
            continue
        domains = list(by_dom.keys())
        zs = zscore([by_dom[d] for d in domains])
        for d, z in zip(domains, zs, strict=True):
            z_by_domain_measure[d][m] = z

    composites: dict[str, float] = {}
    backing: dict[str, list[str]] = {}
    for domain in domain_measure_values:
        zs = z_by_domain_measure[domain]
        if not zs:
            continue
        composites[domain] = float(np.mean(list(zs.values())))
        backing[domain] = sorted(zs.keys())
    return composites, backing, z_by_domain_measure


def cross_attribution() -> dict[str, Any]:
    """2x2 cross-attribution: each lane's composite ranks x each lane's GT vector.

    Pure arithmetic on hard-coded literals — no I/O, no corpus reads.
    Localizes Probe E ranking sign-flip to composite construction (not GT mining).
    """
    # provenance: two independently raced lane reports (codex, grok) + this
    # session's reconciliation; see docs/notes/certifiability_audit.md
    domain_order_xa = [
        "wikitext",
        "wt103",
        "code",
        "sudoku",
        "bAbI",
        "SCAN",
        "elements",
    ]
    # Composite RANKS per lane, 1 = most certifiable (best) ... 7 = least.
    codex_composite_rank = {
        "wikitext": 5,
        "wt103": 7,
        "code": 6,
        "sudoku": 2,
        "bAbI": 4,
        "SCAN": 1,
        "elements": 3,
    }
    grok_composite_rank = {
        "wikitext": 3,
        "wt103": 4,
        "code": 2,
        "sudoku": 5,
        "bAbI": 1,
        "SCAN": 6,
        "elements": 7,
    }
    codex_gt = {
        "wikitext": 0.346,
        "wt103": 0.350,
        "code": 0.611,
        "sudoku": 0.520,
        "bAbI": 0.998,
        "SCAN": 1.000,
        "elements": 0.700,
    }
    grok_gt = {
        "wikitext": 0.346,
        "wt103": 0.350,
        "code": 0.611,
        "sudoku": 0.520,
        "bAbI": 0.527,
        "SCAN": 1.000,
        "elements": 0.751,
    }

    n = len(domain_order_xa)
    # Rank dicts use 1=best; spearman_rho treats larger = more certifiable.
    # Convert: score = n + 1 - rank so higher score = more certifiable.
    codex_scores = [float(n + 1 - codex_composite_rank[d]) for d in domain_order_xa]
    grok_scores = [float(n + 1 - grok_composite_rank[d]) for d in domain_order_xa]
    codex_gt_vals = [float(codex_gt[d]) for d in domain_order_xa]
    grok_gt_vals = [float(grok_gt[d]) for d in domain_order_xa]

    codex_comp_x_codex_gt = spearman_rho(codex_scores, codex_gt_vals)
    codex_comp_x_grok_gt = spearman_rho(codex_scores, grok_gt_vals)
    grok_comp_x_codex_gt = spearman_rho(grok_scores, codex_gt_vals)
    grok_comp_x_grok_gt = spearman_rho(grok_scores, grok_gt_vals)

    # Deltas from unrounded rhos; then round deltas for storage/printing.
    codex_gt_col_delta = abs(codex_comp_x_codex_gt - grok_comp_x_codex_gt)
    grok_gt_col_delta = abs(codex_comp_x_grok_gt - grok_comp_x_grok_gt)
    codex_comp_row_delta = abs(codex_comp_x_codex_gt - codex_comp_x_grok_gt)
    grok_comp_row_delta = abs(grok_comp_x_codex_gt - grok_comp_x_grok_gt)

    max_composite_swap = max(codex_gt_col_delta, grok_gt_col_delta)
    max_gt_swap = max(codex_comp_row_delta, grok_comp_row_delta)

    return {
        "matrix": {
            "codex_comp_x_codex_gt": round(codex_comp_x_codex_gt, 4),
            "codex_comp_x_grok_gt": round(codex_comp_x_grok_gt, 4),
            "grok_comp_x_codex_gt": round(grok_comp_x_codex_gt, 4),
            "grok_comp_x_grok_gt": round(grok_comp_x_grok_gt, 4),
        },
        "composite_swap_delta": {
            "codex_gt_col": round(codex_gt_col_delta, 4),
            "grok_gt_col": round(grok_gt_col_delta, 4),
        },
        "gt_swap_delta": {
            "codex_comp_row": round(codex_comp_row_delta, 4),
            "grok_comp_row": round(grok_comp_row_delta, 4),
        },
        "max_composite_swap_delta": round(max_composite_swap, 4),
        "max_gt_swap_delta": round(max_gt_swap, 4),
        "note": (
            "sign-instability is composite-dominant: swapping composite construction "
            "moves rho by 0.79-1.0 and flips sign; swapping GT vector moves rho by "
            "<=0.25 and never flips sign"
        ),
    }


def mine_ground_truth() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Return kept domain GT records and exclusion list."""
    kept: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, str]] = []
    for domain in DOMAIN_ORDER:
        rec = MINED_GROUND_TRUTH[domain]
        if rec.get("excluded"):
            excluded.append(
                {"domain": domain, "reason": str(rec.get("exclude_reason", "excluded"))}
            )
            continue
        kept[domain] = rec
    return kept, excluded


def compute_all_domain_measures() -> dict[str, dict[str, Any]]:
    """Compute / mine all predictors for every domain in DOMAIN_ORDER."""
    out: dict[str, dict[str, Any]] = {}
    for domain in DOMAIN_ORDER:
        paths = CORPUS_PATHS[domain]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{domain}: missing corpus files {missing}")
        tokens = load_tokens(paths)
        ng = ngram_corpus_measures(tokens, k=CONTEXT_ORDER)

        hcr = HARD_CONSTRAINT_RECOVERY.get(domain)
        chain = CHAIN_RECOVERABILITY.get(domain)

        out[domain] = {
            "register_density": {
                "value": ng["register_density"],
                "status": "computed",
                "source": "corpus n-gram determinism scan",
                "scoreable_fraction": ng["scoreable_fraction"],
                "n_positions": ng["n_positions"],
                "n_scoreable": ng["n_scoreable"],
                "n_hard": ng["n_hard"],
                "n_tokens": ng["n_tokens"],
                "k": ng["k"],
                "corpus_paths": [str(p.relative_to(REPO)) for p in paths],
            },
            "effective_output_rank": {
                "value": ng["mean_effective_rank"],
                "status": "computed",
                "source": "corpus n-gram entropy effective rank",
                "sign_flip_for_composite": "negated (higher rank → less certifiable)",
                "k": ng["k"],
            },
            "hard_constraint_recovery_pct": (
                {
                    "value": hcr["value"],
                    "status": "mined",
                    "source_file": hcr["source_file"],
                    "locator": hcr["locator"],
                }
                if hcr
                else {
                    "value": None,
                    "status": "ABSENT",
                    "reason": (
                        "no recorded hard-constraint-recovery figure in docs/notes/ "
                        "for this domain"
                    ),
                }
            ),
            "linear_concept_coverage": {
                "value": None,
                "status": "DROPPED",
                "reason": (
                    "requires model activations / SAE pass; prereg open-decision-2 "
                    "resolution: if fresh model pass needed, DROP (keep audit corpus-side)"
                ),
            },
            "chain_recoverability_2hop": (
                {
                    "value": chain["value"],
                    "status": "mined",
                    "source_file": chain["source_file"],
                    "locator": chain["locator"],
                }
                if chain
                else {
                    "value": None,
                    "status": "ABSENT",
                    "reason": (
                        "no recorded 2-hop / chain-recoverability figure in docs/notes/ "
                        "outside wikitext (khop_realtext.md); grepped 2-hop/khop/chain-"
                        "recover — only wikitext has a domain recovery number"
                    ),
                }
            ),
        }
    return out


def run_audit() -> dict[str, Any]:
    kept_gt, excluded = mine_ground_truth()
    n_domains = len(kept_gt)
    measures_raw = compute_all_domain_measures()

    # Per-domain table rows (kept domains only for correlation; all listed in provenance)
    per_domain: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        gt = MINED_GROUND_TRUTH[domain]
        m = measures_raw[domain]
        per_domain[domain] = {
            "ground_truth_recovery": {
                "value": gt["value"] if not gt.get("excluded") else None,
                "excluded": bool(gt.get("excluded")),
                "definition": GROUND_TRUTH_DEFINITION,
                "source_file": gt.get("source_file"),
                "locator": gt.get("locator"),
                "quoted": gt.get("quoted"),
                "definition_match": gt.get("definition_match"),
                "not_used_alternatives": gt.get("not_used"),
            },
            "measures": m,
        }

    # Build value matrix for kept domains
    kept_domains = [d for d in DOMAIN_ORDER if d in kept_gt]
    domain_vals: dict[str, dict[str, float | None]] = {}
    for d in kept_domains:
        m = measures_raw[d]
        domain_vals[d] = {
            "register_density": m["register_density"]["value"],
            "effective_output_rank": m["effective_output_rank"]["value"],
            "hard_constraint_recovery_pct": m["hard_constraint_recovery_pct"]["value"],
            "chain_recoverability_2hop": m["chain_recoverability_2hop"]["value"],
        }

    measure_keys = [
        "register_density",
        "effective_output_rank",
        "hard_constraint_recovery_pct",
        "chain_recoverability_2hop",
    ]
    lower_is_better = {"effective_output_rank"}

    # Per-measure Spearman vs GT (over domains where measure present)
    gt_by_domain = {d: float(kept_gt[d]["value"]) for d in kept_domains}
    measure_stats: dict[str, Any] = {}
    for mkey in measure_keys:
        pairs_m: list[float] = []
        pairs_g: list[float] = []
        present_domains: list[str] = []
        for d in kept_domains:
            v = domain_vals[d][mkey]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            # sign-align for correlation too: lower-is-better measures flipped
            aligned = -float(v) if mkey in lower_is_better else float(v)
            pairs_m.append(aligned)
            pairs_g.append(gt_by_domain[d])
            present_domains.append(d)
        stats = spearman_with_perm_bootstrap(pairs_m, pairs_g, seed=SEED)
        stats["domains"] = present_domains
        stats["sign_aligned"] = (
            "negated before correlation" if mkey in lower_is_better else "as-is (higher=more certifiable)"
        )
        measure_stats[mkey] = stats

    # Composite
    composites, backing, z_detail = build_composite(
        domain_vals, measure_keys, lower_is_better
    )
    comp_domains = [d for d in kept_domains if d in composites]
    comp_m = [composites[d] for d in comp_domains]
    comp_g = [gt_by_domain[d] for d in comp_domains]
    composite_stats = spearman_with_perm_bootstrap(comp_m, comp_g, seed=SEED + 1)
    composite_stats["domains"] = comp_domains
    composite_stats["backing_measures_per_domain"] = backing
    composite_stats["z_scores_per_domain"] = {
        d: z_detail[d] for d in comp_domains
    }

    # Ranking by composite (desc) vs ground-truth order
    predicted_ranking = sorted(
        comp_domains, key=lambda d: composites[d], reverse=True
    )
    ground_truth_ranking = sorted(
        kept_domains, key=lambda d: gt_by_domain[d], reverse=True
    )
    rank_agreement_note = (
        "Spearman rho between predicted ranking (by composite) and ground-truth "
        "recovery order is exactly the composite-vs-ground-truth Spearman rho "
        "already reported — no second metric invented."
    )

    # Verdict
    stop_triggered = n_domains < MIN_DOMAINS_FOR_CORRELATION
    rho_c = composite_stats.get("rho")
    ci = composite_stats.get("bootstrap_95_ci")
    ci_spans_nonpositive = (
        ci is not None and float(ci[0]) <= 0.0
    )

    if stop_triggered:
        verdict = "inconclusive"
        verdict_reason = (
            f"STOP rule: n_domains={n_domains} < {MIN_DOMAINS_FOR_CORRELATION} "
            "with a clean consistently-defined ground truth; correlation not "
            "interpreted as the full study. DEAD/FIRES branches not applied."
        )
    elif rho_c is None:
        verdict = "inconclusive"
        verdict_reason = "composite rho undefined (insufficient or degenerate data)"
    elif float(rho_c) >= FIRES_RHO_BAR and not ci_spans_nonpositive:
        verdict = "FIRES"
        verdict_reason = (
            f"composite rho={float(rho_c):.4f} >= {FIRES_RHO_BAR} and bootstrap CI "
            f"does not span <=0 ({ci}). empirical-directional only: licenses domain-"
            "targeting strategy + host ranking; does NOT certify the instrument "
            f"(n={n_domains} is weak-powered)."
        )
    else:
        verdict = "DEAD"
        parts = []
        if rho_c is not None and float(rho_c) < FIRES_RHO_BAR:
            parts.append(f"composite rho={float(rho_c):.4f} < {FIRES_RHO_BAR}")
        if ci_spans_nonpositive:
            parts.append(f"bootstrap CI spans <=0 ({ci})")
        verdict_reason = (
            "; ".join(parts)
            + ". Corpus-side measures do not predict certifiability at this "
            "resolution (directional). Domain-targeting is not licensed from the "
            "corpus alone."
        )

    absent_hcr = [
        d for d in DOMAIN_ORDER
        if measures_raw[d]["hard_constraint_recovery_pct"]["status"] == "ABSENT"
    ]
    absent_chain = [
        d for d in DOMAIN_ORDER
        if measures_raw[d]["chain_recoverability_2hop"]["status"] == "ABSENT"
    ]

    result: dict[str, Any] = {
        "tag": "empirical-directional",
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "n_domains": n_domains,
        "fires_rho_bar": FIRES_RHO_BAR,
        "per_domain": per_domain,
        "composite_values": composites,
        "composite_backing": backing,
        "measure_correlations": measure_stats,
        "composite_correlation": composite_stats,
        "predicted_certifiability_ranking": predicted_ranking,
        "ground_truth_ranking": ground_truth_ranking,
        "rank_agreement": {
            "statistic": "Spearman rho(composite, ground_truth)",
            "rho": composite_stats.get("rho"),
            "note": rank_agreement_note,
        },
        "provenance": {
            "ground_truth_definition": GROUND_TRUTH_DEFINITION,
            "ground_truth_citations": {
                d: {
                    "value": MINED_GROUND_TRUTH[d]["value"],
                    "source_file": MINED_GROUND_TRUTH[d]["source_file"],
                    "locator": MINED_GROUND_TRUTH[d]["locator"],
                    "quoted": MINED_GROUND_TRUTH[d]["quoted"],
                    "definition_match": MINED_GROUND_TRUTH[d]["definition_match"],
                    "not_used_alternatives": MINED_GROUND_TRUTH[d]["not_used"],
                }
                for d in DOMAIN_ORDER
            },
            "excluded_domains": excluded,
            "n_domains": n_domains,
            "seed": SEED,
            "permutations": PERMUTATIONS,
            "bootstraps": BOOTSTRAPS,
            "tokenizer": TOKENIZER_NOTE,
            "context_order_k": CONTEXT_ORDER,
            "measure_definitions": {
                "register_density": (
                    f"hard positions / scoreable positions with context order k="
                    f"{CONTEXT_ORDER}; hard = context seen >=2 times with exactly 1 "
                    "distinct continuation; scoreable = context seen >=2 times "
                    "(singletons excluded as vacuously non-forcing)"
                ),
                "effective_output_rank": (
                    "occurrence-weighted mean of exp(H) over contexts seen >=2 times; "
                    "H = Shannon entropy (nats) of empirical next-token distribution; "
                    "SIGN-FLIPPED (negated) for composite and correlation so higher = "
                    "more certifiable"
                ),
                "hard_constraint_recovery_pct": (
                    "mined from residual/register notes; ABSENT if no recorded figure"
                ),
                "linear_concept_coverage": (
                    "DROPPED for all domains — needs model activations / SAE pass "
                    "(prereg open decision 2)"
                ),
                "chain_recoverability_2hop": (
                    "mined B1 2-hop recovery where recorded; ABSENT otherwise "
                    "(do not re-run khop campaign)"
                ),
                "composite": (
                    "z-score each present measure across domains that have it "
                    "(sign-aligned higher=more certifiable); per domain average the "
                    "z-scores of measures present for that domain"
                ),
            },
            "linear_concept_coverage_drop": (
                "DROPPED for all 7 domains: requires model activations / SAE pass; "
                "prereg open-decision-2 resolution says drop if not corpus-side"
            ),
            "hard_constraint_recovery_absent_for": absent_hcr,
            "chain_recoverability_absent_for": absent_chain,
            "scan_aggregation": (
                "ground truth = unweighted mean of length/addprim_jump/simple "
                "certified-stack exact-match; corpus-side measures computed on "
                "concatenated train texts of the same three official splits"
            ),
            "prereg": "PIL_CERTIFIABILITY_AUDIT_PREREG.md (SIGNED 2026-07-13)",
        },
    }
    return result


def _fmt_val(v: Any) -> str:
    if v is None:
        return "ABSENT"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def print_scoreboard(result: dict[str, Any]) -> None:
    print("=" * 78)
    print("Probe E — certifiability-audit pilot (corpus-side → recovery)")
    print("tag: empirical-directional | prereg bar: composite rho >= 0.75")
    print("=" * 78)
    print()
    print(f"Ground-truth definition:\n  {GROUND_TRUTH_DEFINITION}")
    print(f"n_domains = {result['n_domains']}  (STOP if < {MIN_DOMAINS_FOR_CORRELATION})")
    excl = result["provenance"]["excluded_domains"]
    if excl:
        print("Excluded domains:")
        for e in excl:
            print(f"  - {e['domain']}: {e['reason']}")
    else:
        print("Excluded domains: (none)")
    print()
    print("--- Per-domain table ---")
    hdr = (
        f"{'domain':<10} {'GT':>7} {'reg_den':>8} {'eff_rank':>9} "
        f"{'hcr%':>8} {'chain2h':>8} {'composite':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    composites = result.get("composite_values", {})
    for domain in DOMAIN_ORDER:
        row = result["per_domain"][domain]
        gt = row["ground_truth_recovery"]
        m = row["measures"]
        print(
            f"{domain:<10} "
            f"{_fmt_val(gt['value']):>7} "
            f"{_fmt_val(m['register_density']['value']):>8} "
            f"{_fmt_val(m['effective_output_rank']['value']):>9} "
            f"{_fmt_val(m['hard_constraint_recovery_pct']['value']):>8} "
            f"{_fmt_val(m['chain_recoverability_2hop']['value']):>8} "
            f"{_fmt_val(composites.get(domain)):>10}"
        )
        print(f"           GT cite: {gt.get('source_file')} — {gt.get('locator')}")
        print(f"           GT quote: {gt.get('quoted')}")
        rd = m["register_density"]
        print(
            f"           scoreable_frac={_fmt_val(rd.get('scoreable_fraction'))} "
            f"n_tok={rd.get('n_tokens')}  "
            f"hcr={m['hard_constraint_recovery_pct']['status']}  "
            f"chain={m['chain_recoverability_2hop']['status']}  "
            f"lin_concept={m['linear_concept_coverage']['status']}"
        )
        if domain in result.get("composite_backing", {}):
            print(
                f"           composite backed by: "
                f"{result['composite_backing'][domain]}"
            )
    print()
    print("--- Correlations (Spearman) vs ground-truth recovery ---")
    print(
        "p_ge = prereg one-sided (frac shuffles with rho>=obs); "
        "p_2s = frontier-style two-sided (|rho|)"
    )
    for name, stats in result["measure_correlations"].items():
        print(f"  [{name}] n={stats.get('n')} domains={stats.get('domains')}")
        print(
            f"    rho={_fmt_val(stats.get('rho'))}  "
            f"p_ge={_fmt_val(stats.get('permutation_p_rho_ge_observed'))}  "
            f"p_2s={_fmt_val(stats.get('permutation_p_two_sided'))}  "
            f"CI95={stats.get('bootstrap_95_ci')}  "
            f"sign={stats.get('sign_aligned')}"
        )
    cs = result["composite_correlation"]
    print(f"  [COMPOSITE] n={cs.get('n')} domains={cs.get('domains')}")
    print(
        f"    rho={_fmt_val(cs.get('rho'))}  "
        f"p_ge={_fmt_val(cs.get('permutation_p_rho_ge_observed'))}  "
        f"p_2s={_fmt_val(cs.get('permutation_p_two_sided'))}  "
        f"CI95={cs.get('bootstrap_95_ci')}"
    )
    print()
    print("--- Predicted certifiability ranking (composite desc) ---")
    print(f"  predicted:    {result['predicted_certifiability_ranking']}")
    print(f"  ground-truth: {result['ground_truth_ranking']}")
    print(f"  rank agreement: {result['rank_agreement']['note']}")
    print(f"  agreement rho: {_fmt_val(result['rank_agreement'].get('rho'))}")
    print()
    xa = cross_attribution()
    m = xa["matrix"]
    print("--- Cross-attribution (composite x GT, 2x2) ---")
    print("              codex GT   grok GT")
    print(
        f"  codex comp  {m['codex_comp_x_codex_gt']:+.4f}    "
        f"{m['codex_comp_x_grok_gt']:+.4f}"
    )
    print(
        f"  grok  comp  {m['grok_comp_x_codex_gt']:+.4f}    "
        f"{m['grok_comp_x_grok_gt']:+.4f}"
    )
    print(
        f"  max composite-swap |delta| = "
        f"{xa['max_composite_swap_delta']:.4f}   (>= 0.78 threshold)"
    )
    print(
        f"  max GT-swap |delta|         = "
        f"{xa['max_gt_swap_delta']:.4f}   (<= 0.25 threshold)"
    )
    print()
    print("--- Provenance (abbrev) ---")
    prov = result["provenance"]
    print(f"  seed={prov['seed']}  k={prov['context_order_k']}  "
          f"perm={prov['permutations']}  boot={prov['bootstraps']}")
    print(f"  tokenizer: {prov['tokenizer']}")
    print(f"  linear-concept: {prov['linear_concept_coverage_drop']}")
    print(f"  HCR absent for: {prov['hard_constraint_recovery_absent_for']}")
    print(f"  chain-2hop absent for: {prov['chain_recoverability_absent_for']}")
    print(f"  SCAN: {prov['scan_aggregation']}")
    print()
    print("--- VERDICT ---")
    print(f"  {result['verdict']}  [{result['tag']}]")
    print(f"  {result['verdict_reason']}")
    print()
    print(
        "Note: n≈7 is weak-powered. A FIRES licenses the domain-targeting strategy "
        "and a host ranking; it does NOT certify the instrument."
    )
    print("=" * 78)


def main() -> int:
    result = run_audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print_scoreboard(result)
    print(f"\nWrote {OUTPUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
