# bAbI qa1: the generalization benchmark — and what the judge optimizes

The first third-party benchmark with a real train/test split: the corpus is the 2,000 TRAIN
stories (rendered with inline `Q: … A: …`), the benchmark is **1,000 questions over 200 UNSEEN
test stories** — tables cannot memorize the answers; only template + binding rules generalize.

| arm | accuracy on unseen test |
|---|---|
| Qwen2.5-3B-Instruct (direct) | **0.782** |
| wyly package (from Qwen decisions on train) | **0.509** |

The package's 0.509 is genuine generalization — its admitted rules (mined frames, kgram k=10,
skip) fire on story shapes, not memorized stories. But the 27-point gap has a sharp cause: **the
generalizing rules were declined**. The pointer and entity-echo gates — exactly the machinery
that computes "the most recent location of X" in-context — showed ~zero admission marginal,
because on the ×3-repeated training corpus the k=10 tables memorize story fragments outright;
the judge's val split lives inside the same repeated-text distribution, so memorization
dominates every admission race. **The admission objective rewards memorization wherever the
val distribution permits it; only truly held-out evaluation exposes the difference.** Named
fixes: story-level train/val splits (val stories never in fit text), single-pass corpora, or a
judge term that discounts rules whose keys are corpus-instance-specific.

Also banked from this run: the **int64 key-overflow gotcha** — bAbI's 38-token vocabulary made
the k=11 tier's pair keys exceed 2^63 ((vocab+1)^(k+1)), silently corrupting tables; the
SAFE_KGRAMS guard now drops unsafe tiers per-vocabulary.

Teacher gold on train windows 0.749; student core_sw 0.776 @ 100% (imitation is fine — the gap
is generalization, not fidelity). Next: MMLU-college-chemistry through claymore tools mode (the
LLM+expert composed unit), and the feedback-sharpening experiment.

## The region judge: the instrument works, and it corrects the diagnosis

`WYLY_VAL_REGION=1`: the judge's val becomes a contiguous held-out corpus region (fit excludes
every window overlapping its text by a full window-length margin) — story-level judging, and a
general fix for stride-overlap val contamination. Single-pass corpus. Results:

- Package on unseen test: **0.527** (from 0.509) — cleaner admissions (mined cframes joined).
- **The pointer verdict flips from ambiguous to decisively negative**: on held-out stories its
  marginal is **−0.034 (0/3 folds)** — not under-credited, but *actively harmful*: its verbatim
  suffix-copy binds to the entity's previous **question line** and copies the stale location
  (people move). The earlier "memorization crowded it out" story is corrected: memorization did
  dominate the old val, but the pointer also genuinely doesn't fit qa1.
- What the 0.527 is: the k-gram/frame rules implement "answer = the most recent mover's
  location" — right whenever the queried person moved last. The residual needs a rule kind the
  library lacks: **entity-conditioned movement binding** — the token at +k after the queried
  entity's most recent occurrence *in a movement sentence* (a member-set-filtered prev-occ:
  skip occurrences inside question contexts). Named as the next composable extractor.

The methodological deliverable stands regardless: with held-out-region judging, admission
verdicts are trustworthy — a declined rule is now evidence about the rule, not about val
contamination.

## The movement-binding arc completes: three forms, three verdicts, one root cause

1. Verbatim pointer: **negative** (−0.034) — copies stale locations from question lines.
2. Fixed-shift move-echo (filtered prev-occ + succ 3/4/5): declined — bAbI's movement templates
   put the location at varying offsets; any fixed shift is right ~1/3 of the time.
3. `next-member-after` (moveloc: the first location-class token after the filtered occurrence,
   template-length-invariant, location class from learned concept groups): **declined at ~0.000**
   — form-correct, and still no window marginal.

The root cause is now clean: **the judge's val is window-shaped; the benchmark is query-shaped.**
Val windows are random 256-token slices where questions sit mid-window with text continuing —
there the k-gram tiers already answer well. Benchmark prompts END at "A:" on fresh stories —
exactly where kgrams fail and moveloc would win, and exactly the positions the judge never
scores. Admission optimizes window-imitation, not deployment queries. Named next upgrade:
**query-shaped judging** (val = cloze-formatted queries from held-out stories, scored on answer
tokens). bAbI stands at package 0.527 vs Qwen 0.782; the machinery (filtered prev-occ with
avoid/look, next-member-after) is certified-family and shipped in the schema (rosetta 872c3f3,
sgiandubh PR #23) for when the judge can see what it's worth.
## ERRATUM (2026-07-11, slice #79)

The qa1 residual this note reports (served 0.527 vs teacher-bench 0.782, and the miss-set
analysis built on it) is **stale for the current on-disk artifact**: the present
`wyly_expert_package_v5_babi` scores **1.000** on `babi_bench.json` — including on the
deduplicated non-verbatim story-prefix subset (41/41). This is bench saturation, not a
solved generalization gap: 65/110 unique bench story-prefixes appear verbatim in the
training corpus and 801/1000 bench rows are internal duplicates, so the bench's
unseen-configuration mass is ≈0 and it cannot distinguish binding from memorization.
See docs/notes/qa1_cond_headroom.md. The window-vs-query judge lesson in this note stands
(re-measured at ≈24×, empirical-unregistered).
