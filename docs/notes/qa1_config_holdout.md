# Config-holdout qa1 battery (slice #80) — binding is recoverable; memorization provably is not

**Status:** measured (2026-07-11), pre-registered rules applied verbatim (workspace prereg
log; design pinned by pre-measurement before any code — all 24 entity×location pairs are
covered in corpus_babi.txt, so the instrument is fully generated). `pil/qa1_battery.py`
(generator; stdlib-only, seeded, round-trip parse parity with #79's loaders unit-tested)
+ `experiments/campaign_qa1_config_holdout.py` + 14 tests. Two-lane race (grok / codex);
the codex worktree was reaped by an infrastructure race before it wrote anything (honest
null report); the grok implementation won by default and passed independent verification
(scoreboard reproduced exactly).

## The instrument

Generated qa1 world, original vocabulary (4 entities × 6 locations × 5 mined verbs),
**4 held-out entity×location pairs** excluded from every train movement (atoms all seen —
each entity still visits 5 locations, each location still visited by 3 entities; joint
configs unseen; generator-asserted, deliberate-violation path tested). B0′ = a memorizing
cover of the served-package family fit on generated train — counts + kgram k=2,3 **plus a
joint (entity, location) pair-memory tier** (stronger than spec'd; makes the negative
stronger). Test blocks: IID (sanity), block-1 (holdout-pair queries), names (diagnostic
only). Teacher-forced query tails; story-level splits; pinned baseline ensemble = B0′;
pinned val = held-out-story IID query tails (the #79 lesson, honored).

## Results

| block | B0′ agree | abstain | wrong | |R| |
|---|---:|---:|---:|---:|
| train (sanity) | 1.000 | 0 | 0 | 0 |
| IID | 1.000 | 0 | 0 | 0 |
| **block-1 (holdout)** | **0.000** | **610** | **0** | **610** |
| names (diagnostic) | 0.000 | 1000 | 0 | — |

**The residual reappears exactly as registered — and it is 100% abstention, 0 wrong**: the
memorizing family cannot cover unseen joint configurations and (in this world) knows it.

- **H1 (registered: ≥ 0.5 of R at ≥ 0.8 precision): PASS at coverage 1.000 / precision
  1.000.** The moveloc family — keying on entity-conditioned recency, not the pair —
  recovers the *entire* holdout residual. Mined verb set identical to #79
  ({journeyed, moved, travelled, went, went back}, det 1.0). This is the
  binding-vs-memorization contrast measured cleanly: **compositional binding over seen
  vocabulary is fully recoverable by a hand-authored family admitted through the existing
  judge, on a residual that memorization provably cannot touch.**
- **H2 (registered: gated ≥ flat + 0.02 ∧ regressions ≤ ∧ p < 0.05): FAIL — degenerately.**
  Both arms admit moveloc and reach 1.000 on block-1 (b = c = 0, p = 1). With a *complete*
  family, the flat-vs-gated contrast is structurally unsatisfiable at ceiling: no
  information about conditional carry survives. The wyly_multilayer routing negative is
  *technically* extended, but the honest statement is narrower: **conditional carry remains
  untested where it could matter** — it needs a testbed whose new family is only *partially*
  sufficient, so arbitration has residual to fight over. **Open.**
- Names block: full abstention (token-keyed tiers fail trivially on novel names, as
  predicted; diagnostic only, gates nothing).

## Tags

| Claim | Tag |
|---|---|
| generator holdout/coverage/round-trip invariants | **empirical** (14 tests incl. violation path) |
| memorizing family (incl. joint pair-memory) scores 0.000 on held-out configs (610/610 abstain) | **empirical** |
| moveloc recovers 100% of the holdout residual at precision 1.0 through the judge | **empirical** |
| H2 registered rule FAIL (verbatim) — ceiling-degenerate, uninformative about gating | **empirical** (degeneracy noted) |
| conditional carry helps when the family is partial | **open** — needs a partial-family testbed |
| family-level composition on *real* (non-generated) text | **open** |

## Lessons (binding on future preregs)

1. (from #79) Pin the baseline ensemble and val distribution — honored here.
2. (new) **Condition registered contrasts on non-degeneracy**: a "B ≥ A + δ" rule is
   vacuous when A can reach ceiling; either cap the family's coverage in the instrument or
   register the contrast only on the sub-ceiling regime.

## What this means for the program (axis 2)

The compositionality thrust has its first on-text positive: the judge+family machinery
composes at the **family level** — a rule form that binds over seen atoms generalizes to
unseen joint configurations, while the entire memorizing tier abstains. What is *not* yet
measured: (a) whether **conditional routing** (gated carry) ever earns its keep — requires
a partial-family instrument; (b) whether any of this transfers to **non-generated text**,
where families are imperfect by nature — which is exactly where path A (real-text
trusted-tier families) and this thrust converge. Provenance: moveloc is hand-authored
(`template_fixed`-class); `frac_induced` unaffected.

## Race note (process)

The codex lane's isolated worktree was auto-reaped when its wrapper agent idled before the
CLI finished — an infrastructure interaction, not a model failure; codex verified it had
mutated nothing and reported the null honestly. Lane recipe updated: raced CLI lanes must
keep their wrapper alive until the CLI exits.
