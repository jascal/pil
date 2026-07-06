# The composed unit: hub LLM + certified expert, evaluated

Claymore tools mode: a 3B hub LLM (Qwen2.5-3B-Instruct via llama.cpp) calls the element expert as
a tool; claymore executes the spoke queries; the LLM synthesizes with provenance. The mechanism
works end to end — the flagship smoke:

> Q: "What is the chemical symbol of gold?" → hub decomposes → spoke: "Au." (cited: online
> 6-gram tier) → "The chemical symbol of gold is Au. … Sources: [elements] wyly-v5…6-gram tier"

## Three arms, two benchmarks

| arm | MMLU college-chem (100) | element MC (60, in-domain) |
|---|---|---|
| A: LLM direct | **0.440** | **0.683** |
| B: LLM + expert (hub) | 0.080 | 0.300 |
| C: expert option-scoring | 0.210 | 0.217 |

## The findings, attributed from the hub's own sub-query log

1. **Out-of-domain (MMLU), the refusal contract dominates**: 172 sub-queries (NMR, EPR, Lewis
   acidity) — the spoke **correctly abstained on every one**, and claymore's hard promise
   (all-abstain → refuse, in code, BY DESIGN) turned honesty into benchmark zeros. Bounded
   deployment and open benchmarks want different hubs; a labeled-ungrounded fallback mode is the
   named claymore feature if benchmark behavior is wanted.
2. **The relevance-gate lesson repeated**: the first in-domain run silently dropped correct
   cited answers (word-level Jaccard vs subword citations — the exact federation-demo lesson);
   `--min-relevance 0` is mandatory for next-token spokes. Fixed: B 0.267 → 0.300.
3. **What keeps B below A in-domain**: (i) spoke phrasing sensitivity — sub-query surface forms
   that differ from corpus templates misfire the cover (Osmium's group answered 6 instead of 8
   by a mined frame on an off-template phrasing); (ii) 3B tool-loop reliability. Named
   improvements: query canonicalization at the spoke boundary, wider template coverage, a larger
   hub LLM.
4. **Arm C (option-scoring through the cover) ≈ chance**: sw-cover confidences are per-key
   accuracies, not sequence likelihoods — the MC ranker needs a real scoring rule.
5. Infrastructure: spoke CHAIN mode (sgiandubh PR #24, merged); the composed-unit eval used a
   python spoke shim (`wyly_pyspoke.py`) after a C++/Qwen-tokenizer divergence was observed —
   parity on non-pythia tokenizers is an open question for the Rust FFI.

The composed-unit thesis survives in precise form: **provenance flows and honesty composes — but
the hub LLM is the weakest link at 3B, and the spoke's coverage contract extends to QUERY
PHRASINGS, not just facts.**
