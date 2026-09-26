# Certified Early Exit on Restricted Decisions — Pre-Registration

**Status: SIGNED (approved as-is) — 2026-09-25. Decision rules fixed BEFORE numbers. No dump for this probe had been
generated.** A follow-up to #126 / #127 (both HALTED on full-vocabulary next-token decoding). Prompted by an
external suggestion to study **parallel constrained decoding** (`rorshopping/parallel-decisions`), where a local
LLM answers schema fields by a softmax over each field's allowed options, in one batched pass.

## 1. The move
#126 failed on two counts: even the exact suffix norm never beat the certified radius, and a full-vocabulary
check costs about 9 layers on Qwen2.5-0.5B. A **restricted decision** changes both:
- **Cost.** The check runs over the `K` allowed options, not 151,936 tokens. `K·d` vs `V·d`: for `K = 6` it
  costs about 0.0004 layer-equivalents, so the economic ceiling that capped #126 is gone.
- **Radius.** The certified radius only has to beat `K − 1` rivals (distinct option words), not the nearest of
  151,935 tokens (often near-synonyms).

Same certificate as #126 (pairwise, Cauchy–Schwarz), restricted to the options `O`: at exit `k`, with
`t_k = argmax_{v∈O} ⟨x_k, w_v⟩`, the decision is final if `∀ v ∈ O∖{t_k}: ⟨x_k, w_t − w_v⟩ > B·‖w_t − w_v‖`,
with `B ≥ ‖x_final − x_k‖`. The restricted softmax is exactly the library's decision rule, so this certifies
the **library's** answer, not the model's unrestricted next token.

## 2. PINS
- **PIN A — model.** Qwen2.5-0.5B-Instruct fieldrun bundle (as #126). Decisions are read from fieldrun source
  dumps, so the #126 coordinates, `s`-recovery and radius code apply unchanged.
- **PIN B — decision construction.** `rorshopping/parallel-decisions` at commit `45820320` (`build_prompt` +
  `CompiledField.compile` with the Qwen2.5-0.5B-Instruct tokenizer). Token sequence per decision = the prompt ids
  plus the field's suffix ids (`  "location": "`). Options `O` = the compiled first tokens of each allowed answer.
  **Fields the library flags as `collision`** (two answers share a first token) **are excluded**. If the pinned
  schema collides, the run aborts, because the restricted decode would no longer be a single position.
- **PIN C — task and data.** One enum field `location` with 6 choices (bathroom, bedroom, garden, hallway,
  kitchen, office), listed alphabetically. Its description is `"Where the person or object asked about is,
  according to the story"`. Contexts come from pil `data/babi_{bench,qa2_bench,qa3_bench}.json` (qa1/qa2/qa3; the
  bAbI story plus question, with the trailing `A:` removed). Per task, shuffle with seed 0: the first 50 contexts
  go to **CAL** and the next 100 to **EVAL**, giving CAL 150 and EVAL 300 decisions. Gold answers are carried for
  accuracy only; the certificate uses the model's own restricted decision.
- **PIN D — fieldrun capture (engineering prerequisite).** fieldrun `--source-dump` currently dumps positions from
  the *start* of a text. This probe needs (i) `--texts` JSONL rows that carry exact token `ids`, and (ii)
  `--tail N`, which dumps only the last `N` scored positions. One placeholder token is appended after the
  suffix so the decision position has a successor. These go in a fieldrun branch + PR, and the commit used is
  recorded. The change must reproduce #126's dumps exactly when `--tail` is absent.
- **PIN E — bounds:** as #126. **Oracle** `‖y_final − y_k‖`. **Weight-derived**
  `results/certified_early_exit_weight_bound.json` (unchanged). **Calibrated** conformal `(1−α)`,
  `α = 0.05`, per exit, fit on CAL.
- **PIN F — cost.** Restricted check = `K·d` multiply-adds (the `K` option rows only), in layer-equivalents
  (≈ 0.00036 for `K = 6`). **P-every** (check after every layer) is now the natural policy. Net saving =
  layers skipped − checks × cost.

## 3. Metrics (EVAL)
Coverage (certified at some `k ≤ 22`; also with ≥ 2 layers skipped), `k_c` distribution, mean layers skipped,
violations (certified answer ≠ the library-rule answer from the full model), net saving under P-every.
Uncertified `k★` = the earliest exit after which the restricted argmax stays final. Accuracy vs gold (descriptive).

## 4. Decision rules (FIXED BEFORE NUMBERS)
- **FIRES (certified):** the weight-derived bound certifies ≥ 10% of EVAL with ≥ 2 layers skipped.
- **FIRES-EMPIRICAL:** the calibrated bound certifies ≥ 25% of EVAL, the violation rate is ≤ 2α of certified,
  **and** the mean net saving under P-every is ≥ 1 layer.
- **IN-BETWEEN:** oracle coverage ≥ 25%, but neither of the above holds. → The restricted certificate works
  given the right bound; the deployable bounds are the bottleneck.
- **HALTED:** oracle coverage < 25%. → Even restricted to 6 options, norm-bounded early exit fails on this model.
- **SOUNDNESS GATE:** 0 violations for the oracle and weight-derived bounds, else the run is void.

## 5. Secondary, descriptive: option-order bias
The library's README reports 96.7% accuracy on Choice questions when the correct option is listed first vs 24.5%
otherwise. On EVAL, re-run each decision with the options reordered so the **gold answer is listed first**, and
again with it **listed last** (the other options keep alphabetical order). Report:
- accuracy in each order;
- the per-block **order effect**: the change in each block's centred incidence on the gold token between the two
  orders;
- the share of the total effect carried by the top-5 blocks, and their layers.

Not a decision variable. It is the first PIC-side look at which blocks carry the bias.

## 6. Architect's predictions (recorded, not decision variables)
- **The weight-derived bound stays vacuous** (it is ~10⁷ in raw units whatever the option set).
- The oracle does much better than #126 but still **HALTS or lands IN-BETWEEN**. The last layer writes a vector
  about as large as the residual, and restricting the rivals raises the radius but does not shrink that write.
  **Honest uncertainty:** I cannot bound the restricted radius a priori, so this is the least confident
  prediction in the series.
- **Uncertified headroom is larger than #126** (a 6-way choice resolves earlier than a 152k-way one), perhaps 3–6
  layers.
- bAbI accuracy: qa1 high (> 80%), qa2/qa3 much lower.
- The order bias is visible but smaller than the README's 96.7 / 24.5 (their Choice questions are harder). The
  effect sits in late blocks.

## 7. Controls
- fieldrun `--tail` reproduces the #126 dumps bit-for-bit when absent, and matches a full dump at the tail
  position.
- `s`-recovery and full reconstruction (the #126 controls).
- A collision check on the compiled field.
- CAL and EVAL disjoint; `q̂` fit on CAL; EVAL read once.
- pil `ruff` + `pytest`.
- **Engine agreement (reported, not a gate):** the library's own Torch fp16 decision (`Decider("Qwen/Qwen2.5-0.5B-
  Instruct")`, pinned commit) vs fieldrun int8's restricted argmax on EVAL. If they disagree often, the certificate
  is about fieldrun's int8 decision, and that is stated.

## 8. Risks / open
- **Multi-token answers** are out of scope (collision exclusion). Real schemas with shared prefixes need a
  sequence-level certificate, which is open.
- **One task family** (bAbI locations, 6 options). The README's hard Choice questions (up to 20 options) are not
  public in this repo.
- **Int8 vs fp16:** see the engine-agreement control.
- **Multiple looks:** P-every applies the per-`k` conformal guarantee up to 23 times per decision, as in #126.

## 9. Scope fences
Measurement only. One small fieldrun capture change (PIN D, its own PR). No retraining, no change to
parallel-decisions.
