# The `regression_budget` manifest field — pricing the #88 PLATEAU contract

**Status:** shipped on package emit (domain-gated). Emit-only; does not change rule
admission, cover, or `decide` behavior. Round-trips as an ignored top-level key through
the serving loader (same discipline as `admission_stability`).

## What it is

A top-level package-manifest key, domain-gated to **pythia70m/wikitext**, that records the
priced regression rate next to the certificate. Source of truth:

`data/composite_discriminator.json["stage1_counts"]`

Helper: `_regression_budget_field(tag, ds)` in `experiments/wyly_lm_v5.py`, wired into
`emit_full` after the rest of the `man` dict is built. **Absent ≠ 0:** if the domain does
not match, or the artifact file is missing, the key is **omitted entirely** (never
`0.0`, never `null`).

## Honest sourcing (trigger is #88, not #87)

The field is triggered by the **settled #88 PLATEAU contract answer**: zero-regression
trusted-tier growth is not safely extractable at any measured granularity. That is the
#81→#88 routing-safety arc closing, not PR #87's discriminability verdict.

#87's composite-discriminator verdict is **MIXED** (orthogonal / unrelated to whether this
field ships). In particular, `emit_full` does **not** call
`campaign_frontier_rows._contract_corollary` — that helper is gated on `overall ==
"CONFIRMED"` and would return `None` here. The rate is read **directly** from the
persisted `stage1_counts` in `data/composite_discriminator.json`.

## Single operating point (not a swept minimum)

The rate is measured at **one** GATED1 voting operating point: Stage 1 / #81 on
`test_te`. It is **not** a swept or minimized budget. The descriptive later stages are
excluded by registration and must not be pooled into this rate:

| stage | slice | gains / regressions | role |
|---|---|---|---|
| Stage 1 / GATED1 | #81 | **143 / 62** | **the priced operating point** |
| key-retreat | #83 | 18 / 13 | descriptive only — excluded |
| threshold relaxation | #86 | 19 / 7 | descriptive only — excluded |

## Value and provenance

| field | value |
|---|---|
| gains | 143 |
| regressions | 62 |
| rate | 62 / 205 ≈ **0.302439** |
| rate_definition | deduplicated regressions / (deduplicated gains + regressions) |
| eval_unit | held-out split row |
| measured_on | Stage 1 / #81 GATED1 on test_te |
| sweep@sha | #81@3c5a96e + #86@9681595 (`data/frames_at_scale.json`) |
| domain | pythia70m/wikitext |
| contract | PLATEAU (#88) |

Gains and regressions are **read** from the JSON (not hardcoded as rate numerics); the
provenance strings above are fixed labels.

## Tags

**empirical / measured** — priced budget at the registered operating point, **not** a
proved lower bound and **not** a swept minimum.
