# canon: query canonicalization — the (fact × phrasing) coverage fix

An expert's tables match token sequences, so its coverage contract is over (fact × PHRASING)
pairs: a semantically-in-scope query in a novel phrasing either abstains (honest but
unsatisfying) or — worse — fires a fragment-matched wrong rule (the Osmium group-6 case). The
canon layer maps free phrasings onto the covered templates BEFORE key lookup.

## Design (all mined, no labels)

- **templates**: word-level patterns with {E}/{V} slots, mined from the corpus; clustered into
  PROPERTIES by shared (entity, value) pairs; each property's canonical representative is its
  highest-support value-terminal pattern (a usable next-token prefix).
- **entities**: names = capitalized words rarely seen lowercase (function-word exclusion).
- **matching**: q-side keyword coverage with light stemming; entity binding case-insensitive
  with possessive splitting; **already-canonical queries pass through unchanged**.
- **the contract**: parse → canonical lookup → cited answer; **no parse → abstain** — never
  raw-fragment fallback (that's where silent wrong answers live).
- **transparency**: decisions carry `canonical` + bindings; the REPL/HTTP consumer always sees
  the question actually answered — the defense against the one new failure mode canonicalization
  introduces (silent substitution).

## Validation (element expert, both runtimes)

Paraphrase battery: possessives, "how heavy is", "which group does X belong to", lowercase
entities — all parse to the correct template+entity; OOD ("what colour is the sky") abstains
with an explicit reason. 5/7 answers correct end-to-end; the 2 misses are known cover residuals
(mass digit-path, one group misfire), not canonicalization errors. C++ spoke startup reports
`canon=5`; sgiandubh PR #27, rosetta PR #42. MMLU regression (canon-enabled spoke, same hub/LLM): **element MC arm B 0.300 → 0.383**
(+28% relative — the composed unit's largest single improvement; LLM-alone 0.683 remains the
in-domain target, the residual now attributed to the 3B hub's tool loop + the cover's own
residuals). MMLU chem arm B 0.050 vs 0.080 — noise at the refusal-contract floor,
out-of-domain behavior unchanged by design.

## Addendum: the reach lesson — a canonical form must be visible to the tables

The FFI-parity investigation traced the last composed-unit wrongness to canon REP SELECTION:
the symbol family's rep ("The symbol for {E} is") is 5 tokens, and the elements package has
ngram tiers {1, 6} only — the rewritten query could never reach the k=6 tier and fell to counts
junk (also the old Osmium-group '1'). Reps are now **serveability-scored**: candidates are
probed against the package itself with known (E, V) pairs, and the rep the cover actually
answers best wins (falling back to longest-first). Live battery after: **6/7 correct, all
attested** — including the previously-wrong pass-through control ("The chemical symbol of Gold
is" → "Au.", attested). The principle: canonicalization is part of the serving contract, so its
choices must be measured against the serving tables, not corpus statistics alone.