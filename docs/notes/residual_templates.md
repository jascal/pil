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

- `frac_proposed_agnostic` — share of candidates from domain-agnostic pattern *classes*
  (nfold/prefix_body). **Pattern-class only — NOT provenance** (see "Provenance" below).
- `frac_admitted_induced` / `frac_proposed_induced` — the **honest** generality number:
  share of structure that is data-**induced** (via `candidate_provenance`)
- `provenance_admitted` / `provenance_proposed` — counts over {induced, supplied, template_fixed}
- `proposed_by_template` / `admitted_by_template`; structural seeds are `supplied`, not induced

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

> **`% agnostic` is pattern-*class*, not provenance** — it can overstate generality (e.g. CFQ
> reads 1.00 agnostic but 0.00 induced). See [Provenance: agnostic ≠ induced](#provenance-agnostic--induced-audit)
> for the honest `frac_induced` numbers.

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

### Provenance: agnostic ≠ induced (audit)

`frac_*_agnostic` measures agnostic pattern-*class*, **not** whether structure was
discovered from data. `candidate_provenance` splits admitted structure into **induced**
(nfold marker induced / prefix induction-recoverable / rewrite_synth), **supplied**
(hand pack: structural seeds, supplied marker/prefix), and **template_fixed** (a fixed
template over mined content — `relation_atom`, which detects no pattern). `frac_*_induced`
is the honest generality number. `experiments/campaign_generality_audit.py` (CFQ = mcd1):

| domain | agnostic (old) | **induced (honest)** | provenance admitted |
|---|---:|---:|---|
| listops | 1.00 | **1.00** | induced ×2 |
| SCAN (hand pack) | 1.00 | **0.00** | supplied ×2 |
| SCAN (induce_only) | 1.00 | **1.00** | induced ×2 |
| CFQ mcd1 | 1.00 | **0.00** | template_fixed ×16 |

**The correction:** CFQ's `frac_agnostic = 1.00` overstated generality — all 16 admitted
atoms are `relation_atom` content pass-throughs that induce **no** structure, so the
honest `frac_induced = 0.00`. A pattern-agnostic *class* is not a generality claim.

**Prefix induction (pack removal):** `scan_domain_atoms(induce_only=True)` empties the
hand `prefix_tokens` and turns on the previously-dormant prefix inducer. The residual
leaves SCAN admits then read as **induced** (0.00 → 1.00), with test exact-match held at
**1.0000** (no regression). Honest caveats: (a) `structural_seeds` (turn left/right)
remain in the pack but are **not admitted** in this config — so the claim is *"admitted
structure is fully induced,"* not *"zero packs"*; (b) prefix induction over-generates
(`I_JUMP`/`I_LOOK` alongside the turn tokens) — harmless here (extra unused candidates,
accuracy holds), but a tighter validation gate would sharpen it.

Tags: honest `frac_*_induced` metric + audit — **empirical** (CFQ mcd1, SCAN/simple).
CFQ induces zero structure — **empirical**. SCAN prefix pack removed, no regression —
**empirical**. "Method-general on an alien domain" — **open** (CFQ induced = 0; needs a
domain whose structure the templates actually induce).

## Roadmap (Fable alignment)

1. **Done:** marker induction, naive-default admit, rewrite synth sketch, honest suite,
   honest `frac_induced` provenance metric + SCAN prefix induction (this slice).
2. **Done (CFQ):** `relation_atom` join residuals on real MCD — see `cfq_residual.md`
   (set-F1 help + exact SPARQL=0).
3. **Done (bridge slice):** residual → schema bridge + Datalog export in
   `pil/residual_schema.py` (`residual_as_schema.md` steps 2 & 5; souffle round-trip).
4. **Next:** CFQ set-F1 selection over SchemaBank (residual_as_schema step 4) —
   needs a shared question-word + `ns:`-path `stoi`.
5. Shared compositional interpreter (operator table + semiring).
6. KeyTable / token-space residual path for LM slices.
7. Wire ResidualFamily into WylyBlock B0 automatically.

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
