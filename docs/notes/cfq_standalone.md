# CFQ standalone (nested relational composition)

**Why CFQ (after SCAN).** SCAN closed hard **command composition** (length / addprim) to
the train-mined grammar ceiling. CFQ (Keysers et al., ICLR 2020) is the next yardstick:
**English questions → Freebase SPARQL** with official **MCD** splits that maximize
*compound divergence* — new combinations of known atoms (nested relational structure).

## Setup (standalone constraints)

| knob | value |
|---|---|
| Alphabet | WordCodec on **train only** (question + SPARQL tokens) |
| Labels | gold SPARQL strings (no teacher LLM) |
| Soft SGD | none (lookup / soft-structure baselines) |
| Origin | standalone |

Source: HuggingFace `google-research-datasets/cfq` (GCS tarball requires Google auth).

```bash
.venv/bin/pip install datasets   # build-time only
.venv/bin/python experiments/build_cfq.py
.venv/bin/python -u experiments/campaign_cfq_standalone.py
```

Optional: `CFQ_CONFIGS=mcd1,random_split` `CFQ_MAX_TRAIN=20000` for smoke builds.

## Splits

| tag | Official config | Role |
|---|---|---|
| **mcd1 / mcd2 / mcd3** | MCD1–3 | Hard compositional generalization (leaderboard = mean of three) |
| **random** | random_split | i.i.d. control (still unique questions → exact ~0) |

Neural seq2seq on MCD is historically hard (~5–40% exact SPARQL for large Transformers;
specialized architecture leaderboard higher). Flat lookup is not a path.

## Baselines

| Method | Meaning |
|---|---|
| **exact** | Full question string → SPARQL |
| **bag→SPARQL** | `frozenset(question tokens)` → majority train SPARQL |
| **bag token-F1** | Token multiset F1 of bag prediction vs gold |
| **word→path F1** | Soft structure: question words co-occur with SPARQL `ns:` paths / `M*` |

## Scoreboard

| split | exact | bag exact | token-F1 | path-F1 (ns: set) | n_test |
|---|---:|---:|---:|---:|---:|
| mcd1 | 0.000 | 0.001 | 0.038 | **0.496** | 11968 |
| mcd2 | 0.000 | 0.001 | 0.048 | **0.486** | 11968 |
| mcd3 | 0.000 | 0.001 | 0.045 | **0.460** | 11968 |
| random | 0.000 | **0.136** | 0.222 | **0.500** | 11967 |
| **MCD-mean** | 0.000 | ~0.001 | ~0.044 | **~0.481** | — |

**Reading:** Exact and bag→full-SPARQL are ~0 on MCD (composition tax). Random bag exact
0.136 shows partial bag collisions when compounds are not adversarially split. Soft
**path-F1 ~0.48** means question words already co-activate roughly the right Freebase
predicates — the multi-layer gap is **joining / nesting** those relations into exact
SPARQL graph patterns (same story as SCAN tables vs prim_compose).

## Why multi-layer

```text
Block 0  — entity / type lexicon (person, film, M0…)
Block 1  — binary relations (spouse, directed_by, influenced…)
Block 2  — nested joins / conjunctions / filters in SPARQL graph patterns
```

SCAN needed joint unary∪binary admit; CFQ needs **relational residual** depth that
differs by block (not only partitioned provenance).

## Next

1. Canonical SPARQL graph-pattern admit (triple templates + join rules).
2. Residual B2 rules that fire only when B0/B1 paths are present.
3. Optional: CFQ MCD mean as the multi-layer campaign headline metric.

Cross-links: `docs/notes/scan_standalone.md`, `pil/wyly_block.py`,
`experiments/campaign_scan_prims.py`.
