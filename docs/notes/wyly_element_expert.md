# The benchmarked domain expert: periodic table, end to end

The full loop, benchmark-first: **the benchmark defines the expert's required scope** (a closed
domain: 118 elements × 6 properties = 708 exact-match clozes), the teacher is chosen by a
**bake-off on that benchmark**, the corpus is **engineered for coverage** (every benchmark fact
in 4–5 surface templates × 25 passes = 17,700 statements), and the package built from the
teacher's decisions is evaluated on the **same benchmark** — the first external-task evaluation
of a wyly expert, teacher vs package on equal terms.

## Teacher bake-off (what this machine can run)

| model | overall | number | period | symbol | group | mass | category |
|---|---|---|---|---|---|---|---|
| **Qwen2.5-3B-Instruct** (fp16) | **0.732** | .983 | .805 | .814 | .720 | .576 | .492 |
| pythia-6.9b (int8) | 0.326 | .407 | .017 | .890 | .212 | .093 | .339 |
| pythia-2.8b (fp16) | 0.225 | .229 | .051 | .847 | .051 | .000 | .169 |

Qwen wins decisively → the pipeline moved to **Qwen token space** end to end (windows tokenized
with Qwen's tokenizer; the student grounded in Qwen's own embedding via `WYLY_EMBED`; the package
ships Qwen's tokenizer).

## The package vs its teacher

| run | package | notes |
|---|---|---|
| kgram k≤3 | 0.150 | the trigram "of Gold is" COLLIDES across all six properties — the cloze discriminator ("atomic number" vs "chemical symbol") sits at offsets 4–5, beyond key reach |
| k≤5 | 0.489 | number .856, category .610 (already beats the teacher's .492) |
| **k≤9** | **0.751** | **beats the teacher's 0.732** |

Final per-property (k≤9, conf floor 0.5 — abstained on only 5 in-domain, **4/4 out-of-domain**):

| | package | teacher | Δ |
|---|---|---|---|
| category | **0.966** | 0.492 | +0.474 |
| number | 0.949 | 0.983 | −0.034 |
| symbol | **0.907** | 0.814 | +0.093 |
| group | **0.856** | 0.720 | +0.136 |
| period | **0.814** | 0.805 | +0.009 |
| mass | 0.017 | 0.576 | −0.559 |

## The three lessons

1. **Coverage is necessary but not sufficient — KEY REACH decides.** The corpus contained every
   fact from run one; the score tripled purely by extending kgram tiers so keys could bind the
   element AND the property word (0.150 → 0.489 → 0.751). What the expert must *distinguish*
   dictates how far its keys must reach.
2. **The engineered expert beats parametric recall where recall is patchy** (categories +0.474,
   groups +0.136) and loses only where its own mechanics fail: **mass** — Qwen tokenizes numbers
   digit-by-digit, so each emitted digit pushes the element binding further beyond every fixed
   key window (the chain starts "19…" and derails). Named fix: long-range mined-frame offsets
   (currently capped at 8). Numbers-as-digits is the package's structural enemy, not knowledge.
3. **The abstention knob works**: a serve-time confidence floor (0.5) buys the full bounded
   contract — 4/4 OOD refusals — at almost zero in-domain cost (5/708). This is the deployment
   setting the claymore hub composes with.

Provenance: teacher `wyly_teacher_qwen3b_elements_L256.pt` (gold 0.688 on corpus); package
51,826 rules (ngrams + 252 gates + 4 dgates); logs `v5_elements*.log`, `bench_*.log` in the
review artifacts dir. Next (queued): the three-arm third-party evaluation — bAbI +
MMLU-college-chemistry through claymore tools mode (hub-LLM + this expert as the evaluated unit).
