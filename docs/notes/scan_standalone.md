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
| length | 0.000 | 0.000 | **0.916** | 0.916 | 3920 |
| addprim_jump | 0.000 | 0.000 | **0.935** | 0.935 | 7706 |
| simple | 0.000 | 0.000 | 0.539 | 0.539 | 4182 |

Exact is ~0 even on `simple` because full command strings are unique in the generative
grammar (no train/test command collision). Composition is mandatory.

`prim_compose` is a **semi-symbolic upper bound** (train-mined prims + fixed combinators), not
yet a learned Wyly admit path. Residual gaps are the multi-layer learning target.

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

## Next

1. Admit combinators from data (not fixed expand) under cover-marginal on hard splits.
2. Wire `pil/wyly_block.py` stack B0→B1→B2 on SCAN.
3. CFQ later for nested relational composition.
