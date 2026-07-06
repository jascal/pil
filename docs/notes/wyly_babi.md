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