# Domain-agnostic residual templates

Goal: residual **patterns** transfer across domains; only atoms/markers are domain packs.

## API (`pil/residual_template.py`)

| Piece | Role |
|---|---|
| **ResidualTemplate** | Abstract rewrite: `propose(short_maps, domain) → candidates` |
| **NFoldTemplate** | `(x, marker_n) → unit*n` ⇒ bare `(x,) → unit` |
| **PrefixBodyTemplate** | `(x, d) → prefix+body` with known prefix tokens ⇒ `(x,) → body` |
| **StructuralSeedTemplate** | Domain seeds (often domain-specific) |
| **DomainAtoms** | Markers, prefix set, structural seeds, enabled template ids |
| **ResidualFamily** | `propose` / `admit` / `admit_templates` / `diagnostics` |

Core admit stack (val marginal, provenance) is unchanged and domain-blind.

### Domain packs

| Pack | nfold markers | prefix tokens | structural |
|---|---|---|---|
| **scan** | twice=2, thrice=3 | I_TURN_LEFT/RIGHT | turn left/right |
| **listops** | x2=2, x3=3 | — | — |

SCAN `induce_residual_leaves` is a thin wrapper over `ResidualFamily(scan_domain_atoms())`.

## Transfer campaign

```bash
.venv/bin/python -u experiments/campaign_residual_transfer.py
```

Same propose/admit code on **listops** (synthetic, bare-leaf holdout) and **SCAN/simple**.

### Ablations

| Mode | Behavior |
|---|---|
| **base** | Short maps only |
| **hardcode** | All proposed residuals applied |
| **leaf_admit** | Greedy val-marginal per residual candidate |
| **template_admit** | Meta-admit whole `template_id`s then apply their candidates |

### Diagnostics

- `frac_proposed_agnostic` — share of candidates from domain-agnostic patterns (nfold/prefix_body)
- `proposed_by_template` / `admitted_by_template`
- structural seeds count as **not** pattern_agnostic

## Design rules

1. **Patterns** live in `pil/`; **markers** live in domain packs.
2. New domain = new `DomainAtoms` (+ optional expand/score for the task).
3. Prefer leaf/template admit over hardcoding all residuals (certifiable, val-driven).
4. Standalone: train maps only, SOFT=0, every candidate has `template_id` provenance.

## Scoreboard

| domain | base | hardcode | leaf admit | template admit | % agnostic | templates admitted |
|---|---:|---:|---:|---:|---:|---|
| listops (x2/x3 holdout) | 0.000 | **1.000** | **1.000** | **1.000** | 1.00 | nfold |
| SCAN/simple | 0.519 | **1.000** | **1.000** | **1.000** | 0.50 | nfold |

Transfer: **same** `NFoldTemplate` + `ResidualFamily.admit` code; only `DomainAtoms` markers change (`twice/thrice` vs `x2/x3`). SCAN %agnostic 0.50 reflects structural turn seeds (domain-specific).

## Marker induction (main line)

```python
from pil.residual_template import induce_nfold_markers, listops_domain_atoms, ResidualFamily

# No hand markers — discover x2→2, x3→3 from short maps
fam = ResidualFamily(listops_domain_atoms(induce_only=True))
```

`RewriteSynthesizer` (`template_id=rewrite_synth`) enumerates repeat_k / strip_prefix
as a tiny rewrite DSL (opt-in via `enabled_templates`).

`admit(..., celf=False)` is the **default** (naive greedy). Residual val scores are
generally **not submodular** (complementary leaves), so CELF lazy bounds are not
Leskovec-optimal — use `celf=True` only as an opt-in speed path when marginals are
known non-increasing. Admit log rows are admissions only.

## Honest measurement

Transfer on isomorphic listops shows **pattern reusability**, not full method
generality. The campaign's `honest_suite` holds out operators and poisons residuals:

| check | intent |
|---|---|
| operator holdout | train without x3 → test x3 acc < 1.0 |
| irregular maps | non-fold composites should not invent clean markers blindly |
| negative control | val-marginal rejects poison leaf, admits good leaf |

Real generality needs an alien domain (CFQ, COGS, …) not built around the templates.

## Roadmap (Fable alignment)

1. **Done (this line):** marker induction, CELF admit, rewrite synth sketch, honest suite.
2. **Next:** residual leaves as `pil/schemas.py` / rule_learner schemas (unify with PIC learner).
3. Shared compositional interpreter (operator table + semiring) for SCAN/listops expand.
4. KeyTable / token-space residual path for LM slices.
5. CFQ join template on a natural second dataset.
6. Wire ResidualFamily into WylyBlock B0 automatically.

## How to add a new domain

1. **Define atoms** — a `DomainAtoms` pack (or factory) with:
   - `nfold_markers` if n-fold residual applies (e.g. `{"twice": 2}` / `{"x2": 2}`)
   - `prefix_tokens` if prefix+body applies
   - `structural_seeds` only when the domain has fixed structural leaves
   - `enabled_templates` to restrict which patterns run
2. **Mine short maps** from train only: `MapDict` of `tuple[str,…] → list[str]` (same shape as SCAN len≤2 maps).
3. **Propose / admit** with the shared stack:
   ```python
   from pil.residual_template import ResidualFamily
   fam = ResidualFamily(my_domain_atoms())
   maps, log = fam.admit(short_maps, val_score_fn, thresh=1e-4)
   # or fam.propose_map(short_maps) for hardcode-all residuals
   ```
4. **Task expand/score** — domain-specific (SCAN expand, listops expand, SPARQL join). ResidualFamily does not own expand; it only returns admitted maps.
5. **Diagnostics** — log `fam.diagnostics(short_maps, admitted_src=…)` and report `% agnostic` + transfer vs a second domain when possible.
6. **Optional** — new abstract `ResidualTemplate` subclass if the domain needs a *pattern* not covered by nfold/prefix_body/structural (e.g. CFQ join). Prefer a new template class over hard-coding domain logic in the campaign.

Standalone: train maps only, SOFT=0, keep `template_id` on every candidate.
