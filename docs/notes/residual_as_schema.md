# Residual candidates as PIC schemas (design note)

**Status:** design only (this PR). First implementation slice in a follow-up.

**Goal:** Stop maintaining a parallel symbolic admit loop forever. Residual
leaves/joins should become **schemas** selected by the existing
`pil/rule_learner.py` machinery (birth / death / boost-on-residual), the same way
modular arithmetic schemas already are (`pil/schemas.py`).

## Mapping

| Residual stack (today) | PIC learner (target) |
|---|---|
| `ResidualCandidate` (src→tgt, `template_id`) | `Schema(name, predict, datalog)` |
| `ResidualFamily.propose` | schema library / birth proposer over templates |
| `ResidualFamily.admit` (val marginal greedy) | `propose_schemas` + death-by-exact-ablation |
| MapDict expand / set-F1 score | soft semiring decode + NLL / margin on structured labels |
| `template_id` provenance | schema name + Datalog export clause |

### ResidualCandidate → Schema (sketch)

```text
name     = f"{template_id}/{src_key}"          # e.g. relation_atom/marry|ns:people...
predict  = fires when context matches src pattern; emits tgt symbol ids (-1 abstain)
datalog  = template-specific clause body binding C to the decided token/path id
w_s      = learnable weight (zero at birth → decode-neutral until SGD earns it)
```

**n-fold as schema** (parity with `add_mod`):

```text
nfold[k](head=x, marker=m):
  if context ends with m and bare x is unknown:
    propose unit = tgt / k   (unit already induced offline or via numeric length)
  predict: map unit tokens through v2t when values available; else abstain
```

Offline induction (`induce_nfold_markers`) remains the **birth proposer** that
*seeds* candidate schemas; the learner **selects** which survive.

**relation_atom / join** (CFQ):

```text
relation_atom[word, path]:
  if word ∈ content(question): fire path id
  join = soft-OR / set-union over firing atoms  (complementary — matches non-submodular admit)
```

## Admit → birth / death

| Residual admit step | rule_learner analogue |
|---|---|
| Propose candidates | birth from residual boost / schema library eval |
| Val marginal > thresh | accept schema into `SchemaBank` (or RuleProgram source) |
| Never helps / hurts under ablation | death-by-exact-ablation on probe batch |
| Complementary joins need multiple atoms | multi-source soft-OR (already in cover / semiring) |

**Important:** residual val marginal is **not submodular**. The learner’s exact
ablation death is O(K) per probe and does not assume submodularity — preferred
over CELF for CFQ joins.

## First concrete integration slice (next implement PR)

Smallest end-to-end path:

1. **Offline:** keep `ResidualFamily.propose` to emit candidates (nfold / relation_atom).
2. **Bridge:** `residual_candidates_to_schemas(cands, stoi) -> list[Schema]` with
   predict firing on bag-of-word / marker presence in a fixed window.
3. **Select:** run `propose_schemas`-style train exact-match (or set-F1 for CFQ
   structure ids) threshold accept into a tiny `SchemaBank`.
4. **Metric:** same campaign scoreboards (SCAN simple / CFQ set-F1) with and without
   SchemaBank — must match residual admit within ε on structure metrics.
5. **Export:** one Datalog clause per admitted residual schema (certification).

Out of scope for the first slice: full soft NLL training of residual weights,
KeyTable path, rewriting all of expand into a semiring interpreter.

## Tag discipline

| Claim | Tag |
|---|---|
| ResidualFamily on SCAN/listops/CFQ structure | **empirical** |
| Schema bridge design | **open** until implement PR |
| CELF for residual admit | **proved unsound** as default; opt-in only |
| Full SPARQL generation from atoms | **open** (exact SPARQL stays ~0) |

## Relation to CFQ residual campaign

`campaign_cfq_residual.py` is the **empirical** join-atom yardstick using the
symbolic ResidualFamily loop. When the schema bridge lands, re-run that campaign
with SchemaBank selection and show parity + learner diagnostics (birth/death counts).
