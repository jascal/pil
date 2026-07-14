# Probe A — the semantic-forcing test (constraint vs imitation) — Arm-2 fires (cross-vendor); Arm-1 control caveat

Pre-reg: `PIL_PROBE_A_SEMANTIC_FORCING_PREREG.md` (signed + amendment). Raced grok + codex; a first-race
divergence was resolved to two grok bugs + an unpinned teacher (source-audited + fable-verified), fixed, and
re-run to cross-vendor agreement on a pinned teacher (GPT-2). No expert build — probe instrumentation only.

## Question
Does a perturbation-propagation "forcing test" separate a genuine semantic constraint from imitation (the
constraint-vs-imitation discriminator the #99 indentation trap lacked)? Perturb one entity → does the forced
DISTANT token propagate? A genuine constraint follows the perturbed feature; imitation tracks the surface cue.

## Verdict
**Arm 2 (English number-agreement + attractor): FIRES — cross-vendor, empirical, domain-restricted to
canonical templated agreement.** On GPT-2 with wikitext-mined stimuli, both lanes agree:

| lane | bar1 (forces) | bar2 (derivable) | constraint-prop | imitators (unpert→prop) |
|------|------|------|------|------|
| grok (corrected) | 0.984 | 0.984 | 1.000 | nearest-noun 1.0→0.0 · bigram 1.0→0.0 |
| codex | 0.969 | 1.000 | 1.000 | nearest-noun 1.0→0.0 · bigram 1.0→0.0 |

Number-agreement genuinely FORCES (the teacher treats a violation as wrong ~97–98%), DERIVES from the
subject-token grounded number-concept, and SEPARATES cleanly from two strong imitators (both perfect
unperturbed, both fail entirely to propagate under a subject-number flip while the constraint follows it).

**Arm 1 (bAbI moveloc positive control): halted on bar1 — cross-vendor, but the plumbing validated.** Both
lanes (GPT-2): bar1 ~0.70 (< 0.95), bar2 1.0, constraint-prop 1.0, imitator 1.0→0.0. The perturbation /
propagation / imitator PLUMBING is textbook in both lanes (this is the machinery Arm-2 relies on); only bar1
(teacher violation-wrongness) misses — because GPT-2 zero-shot with the "{entity} is now in the ___" frame has
no story context and cannot track bAbI locations. bar1 is **teacher-competence-confounded**, not a broken harness.

## Honest framing (fable-reviewed — what carries the fire, what does not)
- **Load-bearing:** bar1 (agreement forces, real-teacher-verified) + the discriminator (both strong imitators
  fail to propagate while the constraint follows the subject).
- **Near-oracle (certifies the pipeline, not the hard part):** constraint-propagation = 1.0 — the subject's
  number is surface-readable, so the subject-token probe recovers it ~perfectly. This certifies the derivation
  pipeline; it is NOT the substantive evidence.
- **Domain restriction:** templated syntax with mined/curated noun lexis → a fire certifies "agreement forces
  on canonical templated syntax," NOT "in the wild."

## The bug audit (why the race was decisive)
The first race diverged on BOTH arms (grok pythia-70m: Arm-1 fires / Arm-2 halted; codex GPT-2: Arm-1 halted /
Arm-2 fires). A source audit + fable verification found the divergence was two grok implementation bugs + an
unpinned teacher, not a real disagreement:
1. grok Arm-1 bar1 was a `live_location == gold` tautology (never called the teacher) → its Arm-1 "fire" was
   vacuous. Fixed to a real-teacher score.
2. grok Arm-2 read the number-concept at the ATTRACTOR (last) token, not the SUBJECT token → its "constraint"
   partly tracked the surface cue → propagation spuriously collapsed to 0.875 (its own oracle diagnostic stayed
   1.0). Fixed to read the subject token.
3. Teacher pinned to GPT-2 (the first race used two teachers). Corrected grok reproduces codex on both arms.
A single lane would have shipped a teacher-dependent artifact as a verdict — the race + the audit caught it.

## Design lesson for future forcing-tests (bar1)
bar1 (violation-wrongness on teacher outputs) conflates "does the constraint force?" with "is this teacher
competent at the task?" It is trustworthy ONLY where the teacher is task-competent: GPT-2 is competent at
agreement → Arm-2 bar1 is meaningful; GPT-2 is not competent at bAbI-QA zero-shot → Arm-1 bar1 is not. Future
forcing-tests: pick a teacher competent at the task, or score bar1 against a task-competent oracle.

## Program significance (measure-first green light, NOT a build)
Number-agreement is a genuine semantic-forcing constraint — the first natural-text (templated) positive of the
constraint-vs-imitation test, and a candidate for register #2 (after sudoku legality #92, the sole certified
compute-register). It does NOT build a register or prove residual recovery — that is a separate slice. Directly
relevant to the lead's germandata thread: German agreement (declension / government) is the closed-class,
richer version of exactly what Probe A validated on English — the strongest available host for a semantic
legality register. Tag: empirical, domain-restricted.
