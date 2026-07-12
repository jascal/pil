# Structural constructor bridges the regime-L lexical star wall (slice #94)

**Status:** measured + registered (2026-07-12). Pre-registered in `PIL_CFQ_JOIN_PREREG.md`
("the STRUCTURAL constructor") BEFORE numbers; lead-authorized one-shot (WYLY_STEERING item 2,
condition met). All pre-registered conditions met, independently reproduced by the architect
(S,L,N run + full leak audit). Closes the #77 hub/star wall located by #93. One arm `LSTRUCT`
in `experiments/campaign_join_battery.py` + 3 tests.

## Result

The regime-L held-out **STAR** wall (SIG/SIGW = 0.0; #93's *typed* CONSTRUCT inert, `n_adm=0`)
is **BRIDGED** by `LSTRUCT` — a structural constructor that reads the topology signal
(`topoA`/`topoB`) from the input and transfers a train-mined, **position-based
(predicate-independent)** topology→signature-shape template to held-out predicate combinations.

| regime | star | chain | iid | comp | n_adm |
|---|---|---|---|---|---|
| **L** (cue present) | **1.0** | 1.0 | 1.0 | 1.0 | 1 |
| **N** (cue withheld) | — | — | 0.0 | — | **0** |
| S (no L-cue) | — | 0.0 | 0.0 | — | 0 |

Bar MET: star comp **1.0 ≥ 0.7** with ONE admitted rule. N-control CLEAN
(`n_admitted_N=0, n_clean=True, passed=True`, `can_induce=True`). SIG/SIGW star unchanged 0.0.

## Leak-clean — two independent confirmations

1. **Signature (structural):** the fire-time path (`_lstruct_tgt`, `assemble_by_topo`,
   `_topo_of`) has **no gold/sparql parameter** — it reads only `question_tokens` + `key` +
   train-mined templates. `gold_sigs` appears only inside `_learn_topo_templates(train)`
   (mine-time, train-only, allowed). All three scoring paths (`atom_fires` L174, `expand`
   L324, `make_val_score` L561) route through `_lstruct_tgt` on input-side args.
2. **Behavioral (the clincher):** regime N has the same underlying structure but
   `with_disambig=False`, so **no `topoA`/`topoB` token**. LSTRUCT admits **nothing** there
   (`n_adm=0`, inert). A leaking arm would score in N; it doesn't. The arm works **only** when
   the topology cue is present.

## What it shows / doesn't (honest scope, committed in the prereg)

- **IS:** genuine **vocabulary-axis structural generalization**. The template is
  position-based, so it transfers the "star = all predicates co-subject" shape to held-out
  predicate **combinations** whose exact signature is never in train (that is why SIG=0). The
  mined shape rule generalizes across the predicate vocabulary.
- **IS NOT:** inference of hub structure from raw incidence. LSTRUCT bridges via the input's
  explicit topology **label**. Regime N — where that label is withheld — is the true
  underdetermined negative, and LSTRUCT is correctly inert there. The claim is "structural
  generalization **given a structural input signal**," not "structure from nothing." comp=1.0
  is clean because the shape is deterministic given `(topo, key)`; the *difficulty* is low (the
  cue is explicit), the *generalization* (across predicates) is real.

## The wall's closure

#93 located the wall; #94 closes it. The #77 hub/star wall is an **output-vocabulary** wall,
bridged by **constructive assembly** — *typed* construction (`assemble_joins_by_type`) for the
typed regime (#93), *structural/position-based* construction (`assemble_by_topo`) for the
lexical regime (#94). Thread ends (lead's one-shot). Raw-structure inference (regime N) remains
underdetermined **by construction** — a designed property of the battery, not a method plateau.

## Tags

| Claim | Tag |
|---|---|
| LSTRUCT bridges the regime-L held-out star wall (star comp 1.0 ≥ 0.7, N-clean) | **empirical** |
| leak-free (fire-time path has no gold access; behaviorally inert in regime N) | **proved-by-construction** (signature) + **empirical** (N-inertness) |
| genuine predicate-independent structural transfer (position-based template) | **empirical** |
| bridges via raw-structure inference (no input cue) | **NOT shown** — regime N underdetermined by construction |
