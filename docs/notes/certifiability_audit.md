# Certifiability audit (Probe E) — corpus-side → recovery — DEAD (directional), ranking analyst-fragile

Pre-reg: `PIL_CERTIFIABILITY_AUDIT_PREREG.md` (SIGNED). Raced grok + codex — independent mines, because the
load-bearing confound was mis-mining the ground truth. n=7: wikitext, wt103, code, sudoku, bAbI, SCAN, elements.
No new expert builds; corpus-side measures + already-recorded recovery only.

## Verdict: DEAD (empirical-directional)
Corpus-side measures do NOT predict certifiable-recovery at the registered bar (composite ρ ≥ 0.75). Both
lanes independently fail:
- grok (prereg-compliant consistent GT): composite ρ = −0.429, 95% CI [−1.0, 0.615], perm p = 0.85 → DEAD.
- codex (protocol-mixed GT): composite ρ = +0.607, 95% CI [−0.22, 1.0], perm p = 0.17 → DEAD (below bar; CI spans 0).
Bar was directional by design (n=7, weak-powered). DEAD stated flat: domain-targeting is NOT licensed from
the corpus alone at this resolution.

## Ground-truth reconciliation (the raced confound)
Lanes agreed on 5/7 figures (wikitext 0.346, wt103 0.350, code 0.611, sudoku 0.520, SCAN 1.000); diverged on 2:
- **bAbI**: grok 0.527 (region-judge held-out) vs codex 0.998 (served babi_qa3 bench). ADJUDICATED to 0.527 —
  the served-bench figure is the known-saturated one (#79 qa1 bench saturated; #80 held-out is the sanctioned
  protocol). codex's 0.998 is superseded.
- **elements**: grok 0.751 (pil element-expert cloze, served-tier accuracy) vs codex 0.700 (sm-sae cov95 —
  a different instrument, SAE feature-coverage not the served expert).
grok applied ONE consistent definition ("the certified served tier's held-out accuracy") across all 7, as the
prereg's consistency requirement demands; codex mixed protocols on these two. → grok's GT vector is canonical.

## The deliverable (host ranking) is analyst-fragile — and it's the COMPOSITE, not the GT
The two composites have opposite-signed ρ (+0.607 vs −0.429) and nearly-reversed rankings (SCAN is codex #1,
grok #6). A 2×2 cross-attribution (each lane's composite × each lane's GT vector) localizes the fragility:

|            | codex GT | grok GT |
|------------|----------|---------|
| codex comp | +0.607   | +0.571  |
| grok comp  | −0.179   | −0.429  |

Swapping the COMPOSITE construction (which corpus-side measures are available/included per domain) moves ρ by
0.79–1.0 and FLIPS its sign; swapping the GT vector moves ρ by ≤0.25 and NEVER flips it. So the sign-instability
is driven by composite construction, NOT by ground-truth mining. → No host ranking is quotable from this probe at n=7.

## Consequence (recommendation, NOT a probe finding)
Probe E does not license domain-targeting. The gated-beam deployment should pick its host by the mechanism-level
signal (hard-token / register density directly — the quantity the beam's certified hard term acts on), not by
this corpus→recovery correlation. [Tag: recommendation — this probe established the DEAD + the fragility locus,
not the alternative host-selection rule.]

## Honest scope
n=7 → directional, not powered. DEAD is robust (both lanes, both CIs span 0). One suggestive single-measure
signal — codex's hard-constraint-recovery ρ = 0.872 (n=5) — is underpowered and is not the registered composite.
Tag: empirical-directional. The 2×2 is reproducible in `experiments/campaign_certifiability_audit.py::cross_attribution`.
