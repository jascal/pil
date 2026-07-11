# Two-sided gate contrast (slice #82) — no admissible operating point; the trade is now a measured frontier

**Status:** measured (2026-07-11), registered rules applied verbatim (workspace prereg
log, recorded before any code). Controlled follow-up to
[#81](wikitext_gated_rescue.md): same frozen B0′ (loaded, drift-checked), same seed-0
val_te/test_te split, same 5-candidate pool, FLAT re-run fresh; the ONLY change is the
gate form — GATED2 claims a row iff **B0.conf ≤ tau_low AND B1.conf ≥ tau_high**.
`experiments/campaign_wikitext_gated2.py` (+9 tests). BlockStack's built-in gated carry
cannot express the two-sided condition (it never reads B0's confidence in the claim
decision), so GATED2 uses a local `two_sided_merge` over per-block (pred, conf); GATED1's
reference row still runs through BlockStack — and **reproduced #81 exactly**
(+0.008829, 62 regressions).

## Registered outcome: "no admissible operating point" (val stage; test never touched)

Registered selection rule: among the 100 (tau_low × tau_high) validation points, choose
max val agree among those passing BOTH val analogs — agree ≥ flat_val + 0.005 (= 0.3396)
AND regressions ≤ flat's val regressions (= 1). **Zero of 100 pairs qualify.** Per the
registration, GATED2's test evaluation was never run.

The frontier's anchor points (val_te, n = 2,869):

| operating point | tau_high | val agree | Δ vs flat | val regressions | B1 coverage |
|---|---:|---:|---:|---:|---:|
| best agree (= degenerate one-sided) | −∞ | 0.3419 | +0.0073 | 57 | 732 rows |
| **best zero-regression** | 0.401 | 0.3364 | **+0.0017** | **0** | 216 rows |
| registered bar | | 0.3396 | +0.0050 | ≤ 1 | |

## Reading (registered wording, no flip)

- **The two-sided gate eliminates the leak but pays for it with most of the gain.** At
  full safety (0 regressions) conditional routing still nets +0.0017 over flat — a real,
  positive, sub-bar effect at 216-row coverage. Tightening tau_high walks the frontier
  smoothly between #81's one-sided point (+0.0073/57 on val) and the safe point
  (+0.0017/0). **The gain-vs-safety trade is now a measured curve, not a dispute.**
- Per the pre-named outcome: **the leak is not gate-form-fixable at this granularity at
  the registered bar** — selectivity applied at *claim time* discards most of what the
  rescued families know. The next lever, named with measured motivation and NOT built
  here: **regression-aware admission** — move the safety pressure from claim time to
  admit time (penalize regressions in the candidate's marginal), so the judge selects
  rules that are safe by construction rather than muzzling risky rules row-by-row.
- Run-variance note (steering item 3): no new variance was observable in this slice *by
  design* — B0′ is frozen and GATED1 reproduced bit-identically. The #81 finding
  (composition run-variance under regeneration) stands and still merits its own seed-sweep
  slice; this slice neither confirms nor retires it.

## Tags

| Claim | Tag |
|---|---|
| GATED1 (+0.0088/62) reproduced exactly from frozen B0′ | **empirical** |
| zero of 100 two-sided operating points pass both registered val analogs | **empirical** (registered outcome, verbatim) |
| zero-regression routing is net-positive but sub-bar (+0.0017 val, 216 rows) | **empirical** (descriptive frontier) |
| gain-vs-safety frontier: safety costs ~⅔ of the routing gain at this granularity | **empirical** |
| regression-aware **admission** closes the gap | **open** (measured motivation; next lever) |

## Honesty notes

Single registered selection; test_te untouched (structurally asserted in tests — a spy
verifies the no-admissible path never reads test). Candidates `template_fixed`-class;
`frac_induced` unaffected. Two minor implementation notes disclosed by the lane: the
tau-decile grid retains duplicate deciles verbatim per spec (full 100-row frontier), and
the FLAT gate check is implemented as an inclusive range rather than an abs-difference to
avoid a float edge at the boundary; the JSON's `gaps_or_ambiguities` field under-reports
these relative to the lane's prose (cosmetic).
