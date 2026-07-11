# Residual candidates as PIC schemas (design note)

**Status:** first slice landed — the **bridge + Datalog export** (steps 2 & 5)
are implemented in `pil/residual_schema.py` (token-presence residuals; souffle
round-trip verified). Selection reuses single-token `propose_schemas` (step 3,
partial); CFQ set-F1 parity (step 4) stays **open**. Original design below.

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

## First concrete integration slice

Smallest end-to-end path (status per step):

1. **Offline:** keep `ResidualFamily.propose` to emit candidates (nfold / relation_atom).
   *(unchanged — already lands residual candidates)*
2. ✅ **Bridge — done:** `residual_candidates_to_schemas(cands, stoi) -> (list[Schema], skipped)`
   in `pil/residual_schema.py`. Predict fires the single target token when the
   source word is present anywhere in the window (`(x == word_id).any`); multi-token
   `tgt` (n-fold) and unknown symbols are skipped with a reason, not swallowed.
3. ◐ **Select — partial:** reuses `propose_schemas` single-token exact-match
   (`test_propose_schemas_selects_relation_atoms`). Set-F1 selection for CFQ bag
   prediction is **not** wired yet — that is the open half of step 4.
4. ○ **Metric — open:** CFQ needs a shared `stoi` (question words + `ns:` paths
   in one vocab) and a set-F1 selector before SchemaBank-vs-residual parity is
   measurable. No CFQ parity number is claimed yet.
5. ✅ **Export — done:** one presence clause per schema (`tok(I,_,word), C=path`);
   `test_export_datalog_roundtrips_via_souffle` confirms the exported program
   decodes identically to the tensor forward (agreement 1.0).

Out of scope for this slice: CFQ `stoi` construction + set-F1 selection (step 4),
full soft NLL training of residual weights, n-fold unit schemas (multi-token tgt),
KeyTable path, rewriting all of expand into a semiring interpreter.

## Tag discipline

| Claim | Tag |
|---|---|
| ResidualFamily on SCAN/listops/CFQ structure | **empirical** |
| Schema bridge (candidate → presence Schema) | **empirical** (unit-tested) |
| Bridge Datalog clause ≡ tensor predict | **proved** (souffle round-trip, agreement 1.0) |
| SchemaBank matches residual admit on CFQ set-F1 | **open** (step 4 unwired) |
| CELF for residual admit | **proved unsound** as default; opt-in only |
| Full SPARQL generation from atoms | **open** (exact SPARQL stays ~0) |

## Relation to CFQ residual campaign

`campaign_cfq_residual.py` shows the **symbolic ceiling**: bag set-F1 plateaus near a
frequency prior (~0.25), certified admit can lose to hardcode, exact SPARQL = 0.
That is not a failure of the ResidualFamily *API* — it is evidence that **more
word→path atoms will not learn CFQ joins**. The schema bridge + soft-semiring
decode over structured labels is the intended next implement PR.

When the bridge lands: re-run CFQ with SchemaBank selection; report parity on any
structure metric that still applies, plus birth/death counts — and prefer a
metric with headroom (triple/edge-F1) over bag-of-predicate set-F1 alone.