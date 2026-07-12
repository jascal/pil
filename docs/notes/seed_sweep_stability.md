# Admission-loop seed sweep (slice #84) — composition = stable core + seeded fringe; the knob already exists

**Status:** measured (2026-07-11), registered rules + two pre-data amendments applied
verbatim (workspace prereg log; amendment 1 recorded at 0/16 Part-B runs after the lane's
correct runtime halt; amendment 2 = the device-hypothesis reframe, also pre-data).
Steering item 3, delivered as scoped. `experiments/campaign_seed_sweep.py` (resumable,
phased, subprocess-isolated; internal-seed patch applied at the shim — `wyly_lm_v5.py`
untouched) + 10 tests. 17 runs total: 3 CPU same-seed, 2 GPU same-seed, 4 external-seed
(GPU), 8 internal-seed (GPU); ~100s/run on the GPU host.

## Registered outcomes

| check | outcome |
|---|---|
| Part A — CPU same-seed ×3 | **PASS**: bit-identical compositions and core_sw (0.3451250195503235) |
| B1 — GPU same-seed ×2 | **PASS**: bit-identical |
| Cross-device (CPU vs GPU, internal seed 0) | **compositions bit-identical**; core_sw delta 8.7e-05 (float32 non-associativity) |
| B2 — external seeds 1–4 | **exact invariance** (delta 0.0) — external seeds never reach the admission loop (v5 pins internal seed 0) |
| Registered band [0.341, 0.351] | **16/17 in band; internal seed 7 = 0.35140, +0.0004 above the top** — reported verbatim (a good-direction miss is still a miss) |
| Pre-registered CPU-bisect conditional | correctly never fired (GPU binary passed) |

## B3 — the real variance source: the internal seed

Eight internal-seed runs (the loop's own generator redirected at the shim) produce genuine
composition diversity: **17 distinct rule base-names** across seeds, pairwise Jaccard
min 0.231 / median 0.427 / max 0.667. **Both pre-named readings apply simultaneously** —
they are not mutually exclusive:

- **STABLE-CORE**: {induction L=2 (8/8), prevsent-head gate (8/8), cmember (7/8),
  kgram k=2 online (7/8)} — a four-rule core survives any seed.
- **SEED-NOISE**: the remaining ~13 names (sincedot, cap-echo, clause, mined frames,
  induction L=3, …) each appear in < 5/8 seeds — an idiosyncratic fringe.

Consensus re-score (membership = the 4-rule core, tables from the seed-0 fit):
agree = 0.3406 vs ~0.347 median full composition — the fringe carries ~0.006. Per the
registered caveat this is reported descriptively; the atypical-seed-0-fit check was
printed, not chased.

## Attribution — what #81's "run-variance" actually was (correction)

- **Not device variance (in the controlled measurement):** CPU and GPU agree bit-for-bit
  at internal seed 0.
- **Not demonstrated code drift:** the SYSTEMATIC drift list vs the archived artifact is
  **empty** — no difference appears across all 8 internal seeds; even the
  cap-echo→"cap-echo s1" rename is absent in some seeds (the rule itself isn't always
  admitted). The archived composition — 4-rule stable core plus four fringe rules — is
  consistent with being **one draw from the current code's internal-seed distribution**.
- **Honest residue (open):** one historical GPU regeneration (pre-freeze, #81 era)
  produced a composition (mined-frames-in / sincedot-out) that the controlled pairs do not
  reproduce. Likeliest reading: **intermittent GPU-atomics nondeterminism occasionally
  flips a near-threshold admission** — functionally an internal-seed change, low
  probability, not excludable by 2 GPU reps. Bounding that flip rate is an explicitly
  optional follow-up (more GPU same-seed reps); nothing downstream depends on it.

## The certified-package contract (steering item 3 — required section)

1. **A package is a specific rule set, and it IS reproducible today**: pin
   **(code version, internal seed)** and composition is bit-stable across devices in
   controlled measurement (with the intermittency caveat above). The "determinism knob"
   steering asked about **already exists** — v5's hard-coded internal seed 0 — it is just
   not *exposed*. Recommendation (not built): surface it (e.g. `WYLY_SEED`) so intentional
   variation is possible and provenance can record it.
2. **Stability belongs next to the certificate**: the sweep shows admitted rules are not
   equal citizens — a package manifest should carry each rule's **admission stability**
   (n/N seeds), so consumers can distinguish the 8/8 core from 1/8 fringe. The fringe is
   interchangeable (~0.006 aggregate among near-equivalent draws); the core is the
   identity of the tier.
3. **Consensus admission** (admit-by-stability across a seed ensemble) is the natural
   freeze policy if fringe churn ever matters for serving; priced, not built.

## Tags

| Claim | Tag |
|---|---|
| admission loop deterministic given (code, device, internal seed); CPU≡GPU at seed 0 | **empirical** (17 runs; 2-rep GPU limit noted) |
| external seeds are no-ops (v5 pins internal seed 0) | **empirical** (exact invariance ×4) |
| composition = 4-rule stable core + seed-varying fringe (~0.006 aggregate) | **empirical** (8 internal seeds) |
| no demonstrated code drift; archive consistent with a same-code draw | **empirical** (systematic list empty) |
| intermittent device-level admission flips | **open** (one historical instance; unbounded rate) |
| registered band: 16/17; seed 7 over by +0.0004 | **empirical** (reported verbatim) |

## Walls-taxonomy movement

"Tier composition reproducibility" moves from **Open thread → mostly Settled** (variance
attributed: internal-seed fringe; reproducibility recipe stated) **+ one Priced item**
(expose the seed knob + stability annotations in the manifest; trigger: the next package
emit touched for any reason) **+ one small Open residue** (intermittent GPU flip rate).
