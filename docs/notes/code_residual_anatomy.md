# Code residual anatomy — register #2 chosen by MASS (ESCALATE: whitespace_indent, thin)

**Status:** measured + registered, **cross-vendor confirmed** (2026-07-12). Lead-directed
(steering 18:34, item 1 continued): run the residual anatomy BEFORE building any further code
register, so register candidate #2 is chosen by residual MASS, not ease of certification.
Pre-registered in `PIL_CODE_REGISTER_PREREG.md` ("Code residual ANATOMY") — categories,
constraint-shaped set, and the 0.10 decision rule fixed before numbers.
`experiments/campaign_code_residual_anatomy.py` (+ tests; + `experiments/verify_code_anatomy.py`,
grok's independent cross-check).

## Where the code residual mass actually is (n_residual = 5109)
| category | mass | constraint-shaped? |
|---|---|---|
| **identifier** | **0.502** | no (semantic) |
| operator | 0.107 | no |
| **whitespace_indent** | **0.1039** | **yes (live candidate)** |
| literal | 0.080 | no |
| bracket | 0.057 | yes — but REFUTED (#97) |
| keyword | 0.051 | no |
| terminator_sep | 0.051 | yes (live, below bar) |
| other | 0.048 | no |
| comment | 0.000 | no |

## Verdict: ESCALATE (whitespace_indent), cross-vendor confirmed
- Decision rule: a NON-bracket constraint-shaped class (`whitespace_indent` / `terminator_sep`)
  at ≥ 0.10 → ESCALATE; else PLATEAU_N1. `whitespace_indent = 0.1039 ≥ 0.10` → **ESCALATE**.
- **Cross-vendor identical:** codex 531/5109 = 0.1039; grok independent re-implementation
  (different classification + surface-rendering path) **531/5109 = 0.1039, exact**. The boundary
  is NOT a classifier artifact.
- **Thin margin, registered honestly:** ~0.0039 above the bar (~20 tokens from flipping). The
  pre-registered rule fires ESCALATE and we honor it (no post-hoc goalpost move).

## Honest interpretation
- **The computed route on code is LARGELY capped:** ~50% of residual is identifier prediction —
  a free semantic choice no legality/structure register can force. Bracket legality is dead
  (#97). operator/keyword/literal (~24% together) are semantic too.
- **`whitespace_indent` (0.104) is the ONE live register-#2 candidate** — an indent/block-structure
  register. It clears the mass bar (barely), but whether an indent register can actually FORCE
  the recoverable whitespace (vs indentation being partly free) is UNTESTED — it could be another
  bracket-style DEAD. That is the escalated build's #89-style question.

## Tags
| Claim | Tag |
|---|---|
| residual mass distribution (identifier-dominated) | **empirical** (descriptive, cross-vendor) |
| register-#2 candidate exists (whitespace_indent ≥ 0.10) | **empirical** — ESCALATE per the fixed rule, thin margin recorded |
| an indent register recovers the whitespace residual | **NOT tested** — the escalated build/probe |
| computed-route-on-code is fully plateaued | **NOT claimed** — one live candidate survives; but the route is largely capped by the 50% semantic identifier mass |

## Escalation (per the lead's "escalate the build at the boundary")
ESCALATE triggered → the whitespace_indent / indent-consistency register build is escalated to
the lead as a commitment boundary, framed honestly: the one thin live candidate over an
otherwise-capped (identifier-dominated) residual. See `WYLY_STEERING.md` reply.

## Process (lane balancing)
Built by codex; independently confirmed by grok (exact number, different code path) —
[[lane-balancing-rule]]. grok's probe (#97) → codex verify (#97) → codex anatomy (#98) → grok
verify (#98): both lanes exercised on both slices.
