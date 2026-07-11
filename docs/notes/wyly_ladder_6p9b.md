# The 6.9b rung — §8.5 certifiability ladder, v4-canonical (slice #78)

**Status:** measured (2026-07-11), pre-registered band met. Extends the §8.5 scaling law
(WYLY_LM_ENDGAME_REVIEW_FABLE.md) by one rung under the **unchanged protocol**: same 80k
wikitext L=256 windows, deterministic 85/15 index split, top-V=4096 classes, the fixed v4
self-compiling student (8×1200 wake SGD, sleep judge, induction library), teacher scale the
only free variable. Route deliberately chosen over adopting the existing v5-instrument 6.9b
numbers (wyly_domain_structure.md) — §8.6 frames v5 as an instrument change; this rung is a
pure ladder extension. Artifacts: `data/wyly_v4_state_pythia6.9b.pt` +
`wyly_expert_package_v4_pythia6.9b/` (new), `wyly_ladder.py` MODELS + `wyly_ladder2.py`
BASE85/TEACHERS extended.

## Pre-registered band (recorded before the v4 run started)

Plateau-consistent iff v4 core ∈ **[0.21, 0.235]**; above 0.245 or below 0.19 = anomaly,
investigate dtype first. Independent-instrument prior: v5 base core 0.224.

## The rung

| teacher | gold | copy% | big→T | d-big | student | **core** | **core/stu** | rules |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pythia6.9b (int8) | 0.497 | 23.0% | 0.202 | 0.182 | 0.259 | **0.224** | 86.6% | [1, 2, 3] |

- **Core 0.224 — inside the band, and equal to the independent v5 instrument's base core
  to three decimals.** The plateau holds at 500× teacher scale: 0.285 → 0.224 over
  14M→6.9B, near-flat above 410M. **Empirical** (this ladder, these windows; the fixed
  student + fixed library confound carries — a falling curve bounds *this student*, not
  certifiability).
- **Crystallization 86.6%** — back near the 410M peak (86.8%): the fraction of captured
  behavior that certifies keeps *rising* while absolute capture falls.
- **The rule library is scale-stable**: the sleep judge admits exactly depths {1, 2, 3}
  at 6.9B, as at every rung. Student rules-marginal +0.017 overall, +0.097 on the
  teacher-copy subset (v4 run log).
- Inter-scale matrix stays monotone; 6.9b's nearest neighbor is 2.8b (0.763).

## dtype comparability — measured, not hand-waved

The 6.9b dump is int8 (8 GB card); 2.8b is fp16; smaller rungs fp32. At 2.8b, where both
dumps now exist (`wyly_teacher_pythia2.8b_int8_L256.pt`, new):

| column | fp16 | int8 | Δ |
|---|---:|---:|---:|
| gold | 0.4846 | 0.4844 | 0.0002 |
| copy% | 0.2354 | 0.2353 | 0.0001 |

Per-decision agreement int8 vs fp16 = **0.966** — ~3.4% of individual decisions flip, but
the ladder's aggregate columns are dtype-insensitive at the third decimal. **Empirical**
(one rung, wikitext windows). The 6.9b row therefore carries no material dtype caveat at
the aggregate level; per-decision analyses at 6.9b would.

## Scope notes

- 6.9b is this machine's hardware ceiling (12B int8 exceeds 8 GB VRAM); rungs beyond
  require rented compute — a deliberate decision, not a default (**open**).
- Paper table/figure rows (selfcompiler.tex stops at 2.8b) deferred to the pending
  Pythia-rows batch — no local LaTeX toolchain to verify a build.
- Known cosmetic nit: `wyly_lm_v4.py:70` prints a hardcoded "imitate pythia-70m" banner
  regardless of `WYLY_V4_TAG`; artifacts and manifest carry the correct tag.

## Tags

| Claim | Tag |
|---|---|
| core 0.224 at 6.9B under the unchanged §8.5 protocol; band [0.21, 0.235] met | **empirical** |
| v4 core == independent v5 base core (0.224) — cross-instrument agreement | **empirical** |
| aggregate ladder columns dtype-insensitive (fp16↔int8 at 2.8b: Δgold 0.0002) | **empirical** |
| admitted library {1,2,3} scale-stable through 6.9B | **empirical** |
| plateau beyond 6.9B | **open** (compute-gated) |
