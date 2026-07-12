# The khop (2-hop) rule kind — capability, and the scope limit of its query gate

**Status:** shipped end-to-end (pil #95 emit + parity; rosetta #49 serve kind). The certified
2-hop chained-search program is now a first-class package rule: emitted from the self-compiler's
certified `(lo, hi)` program and served to **exact parity** by the real loader. This note records
what the capability is and — importantly — an honest limit on making it *general*.

## What khop is

A `khop` rule (`{kind:"khop", lo, hi, id, tier, basis, confidence, citation}`) fires a two-hop
chained lookup over the context window `ctx`:

1. **Hop 1:** rightmost earlier position where `ctx[i] == query` (query = `ctx[-1]`); its
   successor is the **bridge**.
2. **Hop 2:** rightmost earlier position where `ctx[i] == bridge`, **excluding the bridge's own
   site** (`i != p1+1` — load-bearing; without it the bridge re-matches itself and the second hop
   is wrong); its successor is the **prediction**.

Abstains (returns `None`) when the query is out of range, or either hop has no match. It captures
composition through an intermediary ("X→Y, Y→Z ⇒ from X reach Z") that a single-hop rule cannot
express — a step from retrieval toward derivation (the ergo/claymore "derive, don't retrieve"
family). Reference: `mir_khop2` / `_DL_KHOP` in `experiments/wyly_selfcompile.py`.

## The scope limit: the chaining is order-agnostic; the `[lo,hi]` gate is not

Two parts behave differently under the token-ID ordering:

- **The 2-hop chaining is order-agnostic and general.** It is pure **equality** matching
  (`ctx[i] == query`, `ctx[i] == bridge`) — it does not care where token IDs sit relative to one
  another. The bridge and the prediction can be **any** tokens; no range constrains them. This is
  the real, transferable mechanism.
- **The `[lo,hi]` gate is a synthetic-vocabulary convenience.** `lo <= query <= hi` is the *only*
  numeric range in the program, and it does one thing: select *which queries the rule claims*. It
  is really a **set-membership test** — "is the query one of the tokens this rule was certified
  over?" — and a contiguous interval is an exact, two-integer encoding of that set **only when the
  set is contiguous in ID space**. In the self-compiler it is, *by construction*: `gen_khop2` lays
  out the designated query tokens as one block, so `lo, hi = qt.min(), qt.max()` captures them
  exactly. The apparent power of "just a token range" is borrowed from the task's vocabulary
  layout, not from the gate.

**Consequence for real text.** BPE/byte token IDs are assigned by merge order — arbitrary with
respect to any syntactic or semantic grouping. The tokens a real khop rule should fire on would be
**scattered** across ID space, so a single contiguous `[lo,hi]` would fire on a semantically random
slice. Porting khop to real data requires replacing the range gate with a proper **set-membership
gate** — an explicit token set or a learned membership feature (the "mined-frame store as
first-class gate rules" the schema-gap note lists alongside khop). The certified *chaining*
transfers unchanged; only the *query-selection* mechanism must change from range to set. That port
is mechanical but **not done**.

## Tags

| Claim | Tag |
|---|---|
| the 2-hop chaining program is certified (Datalog ≡ mirror) and served to parity | **proved** (Soufflé cert) + **empirical** (parity test) |
| the chaining mechanism is token-ID-order-agnostic (pure equality) | **proved-by-construction** (no range on the hops) |
| a contiguous `[lo,hi]` query gate is exact | **empirical, SYNTHETIC-VOCAB ONLY** — true because `gen_khop2` lays out queries contiguously; not a general query-selection mechanism |
| khop generalizes as-is to a real tokenizer | **NOT shown** — the range gate must become a set-membership/gate rule (mechanical, open) |

## Related

Same pattern as the structural constructor ([[hub-arm-output-vocabulary-correction]] /
`docs/notes/lstruct.md`): the certified **machinery** is real and general, but the demo leans on a
**clean synthetic signal** (here a contiguous query vocabulary; there the explicit `topoA/topoB`
cue). Honest *given the signal*; needs the signal supplied differently on real data.
