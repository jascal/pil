# SCAN standalone (compositional generalization)

**Why SCAN (not more bAbI).** qa1–3 are saturated for multi-layer measurement.
SCAN tests **systematic composition** of known operators (twice, around, and/after)
under official hard splits.

## Setup (standalone constraints)

| knob | value |
|---|---|
| Alphabet | WordCodec on **train only** |
| Labels | gold command→action pairs (no teacher LLM) |
| Soft SGD | none (symbolic baselines) |
| Origin | standalone |

Data: Lake & Baroni SCAN files under `data/scan/{length,addprim_split,simple}_split/`.  
Build: `experiments/build_scan.py`  
Campaign: `experiments/campaign_scan_standalone.py`

## Baselines

| Method | Meaning |
|---|---|
| **exact** | Full-command dictionary from train |
| **lcp** | Longest train command that is a contiguous sub-command of the test command |
| **prim_compose** | Prims mined from short train commands + fixed SCAN combinators (twice/thrice/around/opposite/and/after) |

Exact is ~0 even on `simple` because each full command string appears once in the generative
grammar (train/test partition unique commands). Composition is the only path.

## Scoreboard (exact-match full action sequences)

| split | exact | lcp | prim_compose | parse cover | n_test |
|---|---:|---:|---:|---:|---:|
| length | 0.000 | 0.000 | **1.000** | 1.000 | 3920 |
| addprim_jump | 0.000 | 0.000 | **1.000** | 1.000 | 7706 |
| simple | 0.000 | 0.000 | **1.000** | 1.000 | 4182 |

*(After `turn around L/R` + residual bare leaves from short composites; earlier
ceilings were ~0.916 / 0.935 / 0.539 then 1.0 / 1.0 / 0.629.)*

Exact is ~0 even on `simple` because full command strings are unique in the generative
grammar (no train/test command collision). Composition is mandatory.

`prim_compose` is the **fixed-combinator ceiling** (train-mined prims + SCAN combinators).
Learned multi-block admit matches it on hard splits; remaining simple gap is unparsed
surface forms / fit-leaf coverage — see `docs/notes/wyly_multilayer.md`.

## Data setup

```bash
git clone --depth 1 https://github.com/brendenlake/SCAN.git /tmp/SCAN
mkdir -p data/scan
cp -a /tmp/SCAN/length_split /tmp/SCAN/add_prim_split /tmp/SCAN/simple_split data/scan/
.venv/bin/python experiments/build_scan.py
.venv/bin/python -u experiments/campaign_scan_standalone.py
```

(`data/` is gitignored; vendoring is local.)

## Why this enables multi-layer work

```text
Block 0  — primitive lexicon (walk/jump/look/run/turn)
Block 1  — unary combinators (twice, around, opposite)
Block 2  — binary composition (and, after)
```

Hard splits (length, addprim_jump) should show a **flat exact gap** and reward
admitted compositional rules. bAbI could not show that (B1 empty at ceiling).

## Learned admit + 2-block

Campaign: `experiments/campaign_scan_learned.py`

| Method | What |
|---|---|
| **prims (fit)** | Short commands on 90% of train (bulk) |
| **learned_admit** | Greedy val-marginal combinators; **`dir` always-on** (leaf syntax) |
| **block0** | Prims only |
| **block_stack** | B0 prims + B1 admitted combinators (`pil/wyly_block.py`) |

### Scoreboard (with leaf `dir` always-on)

| split | exact | prim_compose | learned_admit | B0 prims | 2-block stack |
|---|---:|---:|---:|---:|---:|
| length | 0.000 | 0.916 | **0.916** | 0.000 | **0.916** |
| addprim_jump | 0.000 | 0.935 | **0.935** | 0.000 | **0.935** |
| simple | 0.000 | 0.539 | 0.291† | 0.000 | 0.291† |

† True-leaf + combinator path on simple is lower than bulk short-map prims (0.399) because
fit misses some unigram leaves (`jump`/`run` land in val); bulk `len≤2` maps shortcut
composition. Hard splits match prim_compose once `dir` is leaf syntax.

**Lesson (dir):** Gating `P left/right` as an admit-able combinator collapses addprim
systematicity (0.644): val never contains `jump left`, only bare `jump`, so `dir` has
zero val marginal but is required at test. Phrase formation is B0 lexicon syntax, not B1.

## Prim admit + 3-block

Campaign: `experiments/campaign_scan_prims.py`

| Layer | What |
|---|---|
| **B0** | Greedy admit true leaves (unigram + `turn L/R`) scored under full combinator grammar |
| **B1+B2** | Joint greedy admit over unary∪binary; partition into B1 unary / B2 binary for provenance |
| **staged diag** | Unary-then-binary without joint pool (credit-assignment stress test) |

### Scoreboard

| split | exact | prim_compose | flat+dir | B0+full | **stack** | staged B1→B2 | fit leaves |
|---|---:|---:|---:|---:|---:|---:|---:|
| length | 0.000 | 0.916 | 0.916 | 0.916 | **0.916** | 0.916 | 0.916 |
| addprim_jump | 0.000 | 0.935 | 0.935 | 0.935 | **0.935** | **0.079** | 0.935 |
| simple | 0.000 | 0.539 | 0.291 | 0.291 | **0.291** | 0.177 | 0.291 |

Admitted prims (hard splits): `jump`, `look`, `run`, `walk`. Unary: `twice/thrice/around/opposite`.
Binary: `and`, `after`.

**Lessons:**

1. **Prims need the grammar to show marginal.** Isolated B0 (no combinators) admits almost
   nothing on composed val; score leaves under full unary∪binary.
2. **Unary/binary synergy.** On addprim, nearly every `around` val command also needs
   `and`/`after`, so staged unary-then-binary stalls at 0.079; joint combinator admit
   recovers 0.935. Blocks are structured provenance; admission must be joint when
   operators co-occur.
3. **Hard splits closed** to the train-mined prim_compose ceiling. Remaining simple gap
   is fit-leaf coverage (90/10), not composition operators.

```bash
.venv/bin/python -u experiments/campaign_scan_learned.py
.venv/bin/python -u experiments/campaign_scan_prims.py
```

## Next-action tables (flat seq2seq baseline)

Campaign: `experiments/campaign_scan_seq.py`

Majority next-action tables (Python dict; int64 KeyTable packing overflows for full cmds).
Autoregressive exact-match vs teacher-forced token accuracy.

| Method | Key → value |
|---|---|
| **step** | `(full cmd, t)` → `action[t]` |
| **hist** | `(full cmd, last-W acts)` → next / EOS |
| **bag_hist** | `(frozenset(cmd), last-W)` → next |
| **suf_hist** | `(cmd[-K:], last-W)` → next |
| **hist_only** | `last-W acts` → next (action LM) |

### Scoreboard (AR exact-match full sequences | selected TF)

| split | exact | prim_compose | AR step/hist | AR bag | AR suf | TF bag | TF suf |
|---|---:|---:|---:|---:|---:|---:|---:|
| length | 0.000 | **0.916** | 0.000 | 0.000 | 0.000 | 0.809 | 0.709 |
| addprim_jump | 0.000 | **0.935** | 0.000 | 0.000 | 0.000 | 0.000 | 0.252 |
| simple | 0.000 | **0.539** | 0.000 | 0.001 | 0.000 | 0.809 | 0.800 |

**Reading:** Full-command next-action tables never transfer (unique cmds) → AR exact ~0.
Shared features (bag/suffix) can get **high teacher-forced** next-token acc on length/simple
but still collapse on full-sequence AR (one miss derails; EOS/length wrong). addprim bags
with `jump` do not appear in train compositions → TF bag ~0. Compositional admit remains
the only path that closes hard splits.

```bash
.venv/bin/python -u experiments/campaign_scan_seq.py
```

## Multi-block learned stack

Campaign: `experiments/campaign_scan_multiblock.py`  
Foundation: `pil/wyly_block.py` (`scan_stack_spec`, carry modes, `admit_layer`)  
Notes: `docs/notes/wyly_multilayer.md`

## Residual templates (simple failure modes)

Campaign: `experiments/campaign_scan_residual.py`  
`induce_residual_leaves`: bare `run` etc. from `run twice` / `run left` short maps.
Closes simple to **1.000** without extra empty blocks.

## Next

1. CFQ relational / join residual templates (`docs/notes/cfq_standalone.md`).
2. Estate2-style shared world state in block residual.

Cross-links: `pil/wyly_block.py`, `experiments/campaign_wyly_blocks.py`,
`experiments/campaign_scan_learned.py`, `experiments/campaign_scan_prims.py`,
`experiments/campaign_scan_multiblock.py`, `experiments/campaign_scan_seq.py`,
`docs/notes/cfq_standalone.md`, `docs/notes/wyly_multilayer.md`.
