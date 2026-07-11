# Key-retreat admission (slice #83) — the gain and the leak are entangled below key granularity

**Status:** measured (2026-07-11), registered rules applied verbatim (workspace prereg
log, recorded before any code). Third and concluding slice of the routing thread
([#81](wikitext_gated_rescue.md) → [#82](wikitext_gated2.md) → here). Same frozen B0′,
seed-0 splits, pool, FLAT + GATED1 reference (reproduced again: +0.008829/62).
`experiments/campaign_wikitext_gated_ra.py` (+11 tests incl. the ban-criterion truth
table and a spy proving the structural-stop path never touches test).

## Registered outcome: Part A gate FAILED — Part B never ran

**Val→test ban-set coverage = 6.5% (4/62), against a 50% proceed gate.** The pre-named
negative applies verbatim: *"regressions are row-idiosyncratic at key granularity;
structural retreat cannot fix them."*

Why — and this is the finding, verified from raw per-(rule, key) counts independently of
the campaign code:

- The regressions ARE concentrated: pointer's single dominant key (cell 18) accounts for
  13 of 62 val regressions.
- But that same key produces **18 val gains** — net-positive — so the registered ban
  criterion (regressions ≥ gains ∧ ≥ 1) correctly declines to ban it. Banning it would
  re-import #82's failure (buying safety with the gain).
- The keys that ARE net-negative and banned (4 dstate table entries + 1 frame anchor)
  cover only 4 of 62 test regressions.

**The gain and the leak co-occur at the same keys, decided row by row.** No key-granular
structure separates them: concentration without separability. Claim-time confidence
(#82) and admit-time key structure (#83) both fail for the same underlying reason — the
information that distinguishes a paying row from a regressing row is not in the rule's
key or in the blocks' confidences at this granularity.

## The routing thread, concluded (this granularity)

| mechanism | slice | outcome |
|---|---|---|
| one-sided confidence gate (claim time) | #81 | rescues (+0.0088, p≈5e-08) but leaks (62 vs 3) — registered FAIL on safety |
| two-sided gate (claim time) | #82 | no admissible operating point; safety costs ⅔ of gain (frontier measured) |
| key-retreat (admit time, structural) | #83 | gated off: ban-set coverage 6.5% — regressions row-idiosyncratic |

Registered conclusion: **conditional routing's gain on wikitext is real but cannot be made
safe by any measured mechanism at rule/key/confidence granularity.** Achievability
remains **open** via two named, unmeasured paths: (a) the **relative/strata gate**
(challenger-beats-incumbent per row — the intra-block strata form; a different gate shape
the #82 sweep brackets but did not measure); (b) **sub-key context features** — whatever
separates a paying row from a regressing row at pointer cell 18 is a feature the current
literal space does not expose; finding it converges with the new-register work
(constraint/legality kinds) already queued for the non-entropy residual.

## Tags

| Claim | Tag |
|---|---|
| ban-set coverage 6.5% (4/62); Part A gate failed; Part B never ran | **empirical** (registered outcome, verbatim; test untouched, spy-asserted) |
| dominant regression key is net-positive (18 gains / 13 regressions at pointer cell 18) | **empirical** (recomputed from raw counts) |
| gain and leak entangled below key granularity on this pool | **empirical** |
| routing gain not safely extractable at rule/key/confidence granularity | **empirical** (three mechanisms measured) |
| relative/strata gate; sub-key context features | **open** — named, unmeasured |

## Disclosures

kgram appears in the registered verdict-scope wording but is a B0-only family, never in
the B1 pool — the scope sentence over-included it (cosmetic prereg wording gap; no effect,
Part B never ran). dstate's pre-named "~10-value" keyspace describes its feature-only
reading; retreat used the full composite table key (195 entries) — both reported. Val used
three ways (bans/marginals/selection) as registered; single split; test untouched.
Candidates `template_fixed`-class; `frac_induced` unaffected.
