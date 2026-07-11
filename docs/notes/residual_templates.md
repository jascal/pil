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

## Next

1. CFQ pack: relation atoms + join residual template (new pattern, still abstract).
2. Template *learning* beyond gate: induce new markers from data when n-fold structure is detected without a fixed marker lexicon.
3. Wire ResidualFamily into `WylyBlock` B0 residual registration automatically.
