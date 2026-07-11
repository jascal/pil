# CFQ residual atoms — API portability + symbolic ceiling

Real CFQ (MCD) via `build_cfq.py`. **Not** an isomorphic toy.

## What this actually tests

| Claim | Fair? |
|---|---|
| Same `ResidualFamily.propose/admit` (naive) as SCAN/listops | **Yes** — API reuse |
| Induced compositional **join** structure | **No** — `RelationAtomTemplate` is a vote pass-through |
| Bag set-F1 is a join metric | **Weak** — set-union of word→path associations |
| Exact SPARQL from residuals | **No** — always 0 without a generator |

## Method

```text
1-ns co-occurrence → base atoms (word, path)
multi-ns votes → residual candidates (pass-through template)
predict = ∪ atoms for content words in Q
score = set-F1(pred paths, gold paths)
```

## Required baselines

| baseline | role |
|---|---|
| **freq prior** | always top-K train paths (question-blind) |
| base | 1-ns word→path only |
| hardcode residual | base + all residual candidates |
| certified admit | val-marginal greedy (may **lose** to hardcode) |

## Holdouts

| name | meaning |
|---|---|
| **weakened_base** | drop path from base **and** residual votes — other atoms may still help |
| **recovery** | drop path from base only; residual may re-emit that path |
| **deep** | test queries with ≥6 ns: predicates (ceiling / stress) |

## Scoreboard (mcd1, empirical)

| metric | value | reading |
|---|---:|---|
| freq prior @20 | ~0.20 | anchors the board |
| base | ~0.17 | can **lose** to freq prior |
| hardcode residual | ~0.25–0.26 | best symbolic bag-F1 here |
| certified admit | ~0.24 | **often &lt; hardcode** (val misalign) |
| exact SPARQL | **0.000** | no generator |
| deep hard/admit | ~0.27–0.28 | plateau ≪ 1 |

### Negative result (primary finding)

**Symbolic word→path bag set-F1 is at a frequency-prior ceiling on CFQ.**  
Adding atoms helps a little; certified selection can hurt; nothing here composes
joins or emits SPARQL. **Pivot effort** to `residual_as_schema.md` (learner bridge
+ compositional decode), not more atom packs. If CFQ stays symbolic, use a metric
with headroom (e.g. triple/edge-F1 over the query graph).

## Run

```bash
.venv/bin/python -u experiments/campaign_cfq_residual.py
# CFQ_SPLITS=mcd1  CFQ_FREQ_K=20  CFQ_RES_TOPK=60
```

## Tags

| claim | tag |
|---|---|
| API reuse on CFQ | **empirical** |
| Join induction | **not claimed** |
| set-F1 ceiling ~0.26 | **empirical** |
| Schema / soft-semiring path | **open** (design note) |
