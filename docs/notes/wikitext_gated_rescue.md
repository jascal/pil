# Wikitext residual anatomy + gated-rescue contrast (slice #81)

**Status:** measured (2026-07-11), all registered rules applied verbatim (workspace prereg
log; one pre-Part-A amendment, recorded before any Part A/B numbers existed — below).
First slice on the real-text convergence of axis 1 (coverage) and axis 2 (conditional
routing): wikitext, pythia70m, ~11.5k held-out te windows split val_te 2,869 / test_te
8,608 (seed 0, pinned). `experiments/campaign_wikitext_gated_rescue.py` + 10 tests
(traced-arbitration parity, device-parametrized freeze/load round trip, split/tau/pool
fixedness). Implemented by the codex lane (its first full slice); verified independently
on both CPU and GPU paths; scoreboard reproduced bit-for-bit-modulo-1e-8.

## Finding 0 (pre-Part-A): admission composition is run-variant

The spec's B0 gate initially required an exact match to the archived 8-rule artifact —
stricter than the registration (score-band 0.346 ± 0.005). Two independent regenerations
both hit that STOP honestly: their rule compositions differ from the archive AND from each
other (family swaps: clause gate ↔ sincedot / mined frames; consistent "cap-echo" →
"cap-echo s1" rename indicating code drift since the archive), while **all three land at
core_sw 0.345–0.347**. **Admission composition on wikitext is run-variant at fixed
aggregate performance** (empirical, 3 compositions). Implications: the certified tier's
*identity* is one sample from a family of near-equivalent rule sets; provenance describes
what a tier is, not how to remake it bit-exact. Open lever: fully seed the sleep-wake
admission loop, then re-freeze an authoritative artifact. The slice proceeded under the
registered score gate with **B0′ frozen once** (`data/wikitext_gated_rescue_b0.pt`,
composition disclosed in the scoreboard; loaded thereafter — reload drift ≤ 6e-08).

## Part A — anatomy of the residual (descriptive)

B0′ core_sw agree = 0.3451 (inside the registered band). On te: 3,961 correct / 7,516
error rows.

- **Confidence calibration is cleanly monotone** — agree 0.066 in the bottom conf decile
  rising to ~0.49 by decile 7. Errors concentrate exactly where a gate can see them; the
  Part-B tau chosen on val_te (0.301) sits at the calibration knee.
- **Teacher-entropy overlay (primary instrument: inter-teacher agreement, 8 ladder dumps
  on identical windows):** on error rows the median number of teachers agreeing with
  pythia70m's decision is **3/8** (mean 3.84; 29% of error rows have ≤ 1 other teacher
  agreeing) vs **median 7/8** on correct rows (mean 5.76). A large fraction of the
  residual is teacher-idiosyncratic — unrecoverable by any family. **Empirical**, proxy
  named: inter-teacher agreement (the dumps carry argmax decisions, not probabilities).
- Tertiary: copy incidence ≈ 0.001 on both correct and error rows — the residual is not
  copy-shaped (consistent with the pointer-cell finding of the domain-structure note).

## Part B — the registered contrast, verbatim

> "Gated rescues" iff (i) gated admits ≥ 1 family flat declines in the SAME run, AND
> (ii) gated test_te agree ≥ flat + 0.005, AND (iii) two-sided exact binomial p < 0.05 on
> discordant flips, AND (iv) gated collateral regressions ≤ flat's.

| | flat | gated |
|---|---|---|
| admitted | mined frames (+0.0014) | **dstate (+0.0014), pointer (+0.0077), mined frames** |
| test_te agree | 0.3497 | **0.3585** (+0.0088) |
| collateral regressions | 3 | 62 |
| discordant | b = 59 (flat-right/gated-wrong) | c = 135 (gated-right/flat-wrong) |
| exact binomial p | | 4.9 × 10⁻⁸ |

**Verdict: "gated does not rescue" — clauses (i)(ii)(iii) TRUE, (iv) FALSE.** No flip.

The texture matters as much as the verdict:

- **Conditional routing finds real signal flat admission wastes.** Gated rescued *two*
  flat-declined families — including **pointer, pre-named the weakest candidate**, whose
  behind-the-gate marginal (+0.0077) is the largest single marginal in the run and larger
  than the entire mined-frames contribution that headlined follow-up 5. The pre-naming
  makes this maximally credible: the family priors would exclude is the one the gate
  extracts the most from. Net effect is strongly positive (+76 net rows, p ≈ 5e-08).
- **The current gate leaks.** Firing B1 wherever B0 is unconfident flipped 62
  previously-correct windows (0.7% of test_te) vs flat's 3. The registered safety clause
  ("regressions ≤ flat's") is conservative by design and it binds. The measured next
  lever — not built in this slice — is **regression-aware gating**: fire B1 only where B0
  conf is low AND B1 conf is high, or penalize regressions at admission time.

## Tags

| Claim | Tag |
|---|---|
| admission composition run-variant at fixed aggregate score (3 compositions, 0.345–0.347) | **empirical** |
| residual calibration monotone; errors concentrate at low conf | **empirical** |
| large teacher-idiosyncratic fraction in the residual (median inter-teacher agreement 3/8 on errors vs 7/8 on correct) | **empirical** (proxy: inter-teacher agreement) |
| gated routing admits families flat declines, with net-positive significant agreement gain | **empirical** |
| registered verdict "gated does not rescue" (collateral clause) | **empirical** (verbatim) |
| regression-aware gating turns the net gain into a clean win | **open** (next lever, measured motivation) |
| seeding the admission loop restores composition reproducibility | **open** |

## Honesty notes

All candidates are `template_fixed`-class; `frac_induced` unaffected. Declines were
re-established fresh in-run against the pinned B0′ (never inherited from history).
Admission-time ensemble = B0′ both arms; eval-time ensembles pinned separately; tau swept
on val_te only over the fixed decile grid; frames re-mined on the train low-conf slice
only. The pre-Part-A gate amendment is recorded in the prereg log with explicit timing
(no Part A/B numbers existed). Device robustness: freeze/load verified on CPU and GPU;
CPU↔GPU drift ≤ 6e-08, tolerance 1e-6.
