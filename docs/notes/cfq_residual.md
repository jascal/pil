# CFQ join residuals (generality test)

Not an isomorphic toy: real CFQ (Keysers et al.) via `build_cfq.py`.

## Method

Same code path as SCAN/listops:

```text
mine base atoms (1-ns) → RelationAtomTemplate propose (multi-ns votes)
  → ResidualFamily.admit (naive, celf=False) → set-F1 join score
```

| Piece | Role |
|---|---|
| Base maps | `(word, path) → [path]` from single-predicate train queries |
| Residual | more word→path atoms from multi-ns co-occurrence |
| Join | set-union of atoms over question content words |
| Metric | predicate **set-F1** (structure); **exact SPARQL = 0** (no generator) |

## Honest holdouts

| Holdout | Intent |
|---|---|
| Relation path holdout | Drop frequent path from base; residual may help recover |
| Deep queries (≥6 ns) | Join stress — F1 stays low without full SPARQL gen |
| Exact SPARQL | Always ~0 until a real generator exists |

## Run

```bash
.venv/bin/python experiments/build_cfq.py   # once
.venv/bin/python -u experiments/campaign_cfq_residual.py
# CFQ_SPLITS=mcd1,mcd2  CFQ_MAX_VAL=2000
```

## Scoreboard (mcd1, structure set-F1)

| split | set-F1 base | hardcode | admit | exact SPARQL | holdout base→admit | deep admit |
|---|---:|---:|---:|---:|---:|---:|
| mcd1 | 0.166 | 0.253 | **0.241** | **0.000** | 0.159→**0.236** (helps) | 0.279 |

Accept: residual admit helps over base (0.166→0.241) **and** exact SPARQL honestly fails (0).

## Tags

- Structure set-F1 gains: **empirical**
- Exact SPARQL: **open** (honest zero)
- Generality vs listops: CFQ is non-isomorphic — **empirical transfer of ResidualFamily API**, not of nfold markers
