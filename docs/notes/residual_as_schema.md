# Residual candidates as PIC schemas (design note)

**Status:** steps 2–5 landed. The **bridge + Datalog export** (steps 2 & 5) are in
`pil/residual_schema.py` (token-presence residuals; souffle round-trip verified). The
**set-F1 selector + shared vocab** (steps 3 & 4) are now wired: `propose_schemas_setf1`
mirrors `ResidualFamily._admit_naive` and, on real CFQ **mcd1**, admits the identical
16 `(word,path)` pairs the residual admit loop does, with val/test set-F1 equal to
<1e-9 (`experiments/campaign_cfq_schema_parity.py`; test set-F1 0.239). Original design
below.

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
3. ✅ **Select — done:** `propose_schemas_setf1` (`pil/residual_schema.py`) is the
   set-valued analogue of `propose_schemas` — greedy val-marginal over a set-union
   schema decode, mirroring `_admit_naive` line-for-line (strict `>` thresh,
   first-wins ties, base recomputed each round, candidate order preserved). The
   single-token `propose_schemas` path is untouched.
4. ✅ **Metric — measured:** `cfq_stoi_from` builds the shared vocab (question words +
   `ns:` paths in one id space); `mean_set_f1_schemas` scores in Python floats, matching
   `campaign_cfq_residual.mean_set_f1` exactly. Parity is measured on mcd1: selector and
   residual admit select the **same 16 atoms**; val/test set-F1 equal to <1e-9.
5. ✅ **Export — done:** one presence clause per schema (`tok(I,_,word), C=path`);
   `test_export_datalog_roundtrips_via_souffle` confirms the exported program
   decodes identically to the tensor forward (agreement 1.0).

Still out of scope: full soft NLL training of residual weights, n-fold unit schemas
(multi-token tgt), KeyTable path, rewriting all of expand into a semiring interpreter,
and exact SPARQL generation (stays ~0).

## Tag discipline

| Claim | Tag |
|---|---|
| ResidualFamily on SCAN/listops/CFQ structure | **empirical** |
| Schema bridge (candidate → presence Schema) | **empirical** (unit-tested) |
| Bridge Datalog clause ≡ tensor predict | **proved** (souffle round-trip, agreement 1.0) |
| Set-F1 selector reproduces residual admit on CFQ mcd1 | **empirical** (mcd1: same 16 atoms; val/test set-F1 equal <1e-9) |
| CELF for residual admit | **proved unsound** as default; opt-in only |
| Full SPARQL generation from atoms | **open** (exact SPARQL stays ~0) |

## Relation to CFQ residual campaign

`campaign_cfq_residual.py` shows the **symbolic ceiling**: bag set-F1 plateaus near a
frequency prior (~0.25), certified admit can lose to hardcode, exact SPARQL = 0.
That is not a failure of the ResidualFamily *API* — it is evidence that **more
word→path atoms will not learn CFQ joins**. The schema bridge + soft-semiring
decode over structured labels is the intended next implement PR.

**Landed:** `campaign_cfq_schema_parity.py` re-runs mcd1 with the set-F1 selector on the
same base atoms + candidate pool + val split as the residual admit loop, and confirms
identical selection (16/16 atoms) and equal bag set-F1 (test 0.239). This closes the
*parity* question — the selector is a faithful drop-in for the symbolic admit loop, so
the parallel admit path can be retired for token-presence residuals. It does **not**
lift the ceiling: bag set-F1 still plateaus near the frequency prior. The next headroom
lever is a metric/decoder that rewards join structure (triple/edge-F1, soft-semiring
decode), not more atoms.