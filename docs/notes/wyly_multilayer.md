# Multi-layer Wyly (block stack)

Post SCAN/CFQ foundation notes: how depth is structured, what is measured, and
where gains still come from.

## Architecture (`pil/wyly_block.py`)

| Piece | Role |
|---|---|
| **WylyBlock** | Features + candidates + greedy admit + local SW cover |
| **BlockState** | Symbolic residual: features, pred/conf, free-form `residual` dict, meta |
| **BlockStack** | Ordered blocks; carry modes; `admit_layer`; per-block marginals |
| **StackSpec / BlockSpec** | Configurable depth, families, local vs stack admit |
| **scan_stack_spec()** | Canonical SCAN 3-block: prims → unary → binary |

### Carry modes

| mode | Behavior |
|---|---|
| `replace` | Next block sees only latest BlockState |
| `merge` | Features + residual accumulate (default for SCAN symbolic residual) |
| `gated` | Where current conf is weak/abstain, keep upstream pred/features |

Gated carry is the first **targeted residual** improvement for class-space covers
(estate-style). SCAN action-sequence expand still uses symbolic enable sets carried
in `residual` (not tensor pred).

### Admission

- **local** — score only the block’s rules.
- **stack** (default) — score `rules_through(block_id-1) ∪ trial` (freeze upstream).

**SCAN lesson:** unary and binary are synergistic on val (almost every `around`
co-occurs with `and`/`after`). Staged unary→binary stalls; **joint** combinator
admit then **partition** into B1/B2 for provenance. Depth is structured
attribution until residual *rules* differ by block.

## SCAN multi-block campaign

```bash
.venv/bin/python -u experiments/campaign_scan_multiblock.py
```

Reports: exact, prim_compose, B0 / B0+unary / full stack, staged diagnostic,
per-block val marginals, failure samples.

### Grammar residual closed (this PR)

`turn around left/right` = four turns. Previously `expand` tried `expand(["turn"])`
as body and failed → **all** length residual errors. After the fix:

| split | prim_compose (before) | prim_compose (after) |
|---|---:|---:|
| length | 0.916 | **1.000** |
| addprim_jump | 0.935 | **1.000** |
| simple | 0.539 | **0.629** |

### Multi-block learned scoreboard (`campaign_scan_multiblock.py`)

| split | exact | prim_compose | B0 prims | B0+unary | **stack** | staged B1→B2 |
|---|---:|---:|---:|---:|---:|---:|
| length | 0.000 | 1.000 | 0.000 | 0.002 | **1.000** | 1.000 |
| addprim_jump | 0.000 | 1.000 | 0.000 | 0.003 | **1.000** | 0.224 |
| simple | 0.000 | 0.629 | 0.000 | 0.002 | 0.357 | 0.357 |

Hard splits: **learned stack = prim_compose = 1.0**. Staged unary→binary still
fails on addprim (0.224) — joint admit required. Simple stack lag is fit-leaf
coverage (90/10 only admits look/walk), not missing combinators.

### Identified gaps

1. **Simple residual (~0.37 miss)** — remaining unparsed surface forms (binding
   order, multi-operator nests, missing fit leaves under 90/10). Needs richer
   residual templates or fit-all-leaf mining, not more depth theatre.
2. **Depth without residual rules** — partitioning unary/binary into B1/B2 alone
   does not beat joint flat admit. Next gains need **block-private residual
   candidates** (templates that only fire after upstream paths exist).
3. **Class-space vs sequence** — WylyBlock cover is class-token oriented; SCAN
   exact-match is full action sequences. Bridge: residual dict + expand scoring
   (current) or next-action KeyTables inside a block.
4. **CFQ** — path set-F1 ~0.48, exact SPARQL ~0. Multi-layer target is join/nest
   rules over Freebase predicates (`docs/notes/cfq_standalone.md`).
5. **Certification cost** — more blocks ⇒ more admit logs and rule names; keep
   every rule inspectable (`summary()`, admit logs). Prefer explicit residual
   over opaque stack depth.

## Tradeoffs

| Choice | Upside | Cost |
|---|---|---|
| Joint combinator admit | Recovers systematicity | Weaker “pure layer” story |
| Gated carry | True residual for unsolved slots | Extra conf calibration |
| Freeze-upstream admit | Stable lower layers | Can miss co-adapted rules |
| Grammar residual fix | Hard splits → 1.0 | Must stay explicit/certified |

## Next

1. Block-private residual templates on simple residual failures.
2. Estate2-style shared world state in residual for plan/position tasks.
3. CFQ triple-template admit on MCD.
4. Strata / rule_learner cross-block references for hierarchical edits.

Cross-links: `scan_standalone.md`, `cfq_standalone.md`,
`campaign_scan_multiblock.py`, `campaign_scan_prims.py`.
