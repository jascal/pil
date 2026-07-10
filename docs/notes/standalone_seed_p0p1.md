# Standalone seed P0+P1: SOFT=0 + WordCodec

**Goal.** Kill remaining host-LLM scaffold on bAbI: no soft SGD, no host BPE
tokenizer, no teacher labels/embeds — still host-free package at serve.

## Configuration

| knob | value |
|---|---|
| `WYLY_SOFT` | **0** (counts + sleep only) |
| `WYLY_ALPHABET` | **word** (`pil.alphabet.WordCodec`) |
| `WYLY_LABELS` | corpus |
| `WYLY_CONCEPT_INIT` | random (forced under word alphabet) |
| `WYLY_ORIGIN` | standalone |
| Teacher file / host embed / host tokenizer | **unused** |

Build: `experiments/build_word_babi.py`  
Run: `experiments/campaign_standalone_seed.py`

## Served scoreboard

| qa | served | origin | alphabet | soft | parity |
|---|---|---|---|---|---|
| qa2 | **1.000** (1000/1000) | standalone | word | false | 1000/1000 |
| qa3 | **0.998** (998/1000) | standalone | word | false | 1000/1000 |

Matches E0/E1 and Band B ceilings **without** host BPE and without SGD.

## What this closes

| Scaffold | Status |
|---|---|
| Teacher next-token labels | gone (corpus stream) |
| Host embed PCA | gone |
| Soft student SGD | gone (`SOFT=0`) |
| Host tokenizer | gone (corpus WordCodec, vocab ~40) |
| Package serve | pure symbolic (estate2 dgate + counts) |

**Residual non-LLM priors still present:** hand estate2 form (`WYLY_ESTATE2` json + fold
code), external gold query JSON, Rosetta thin runtime. Next: induced schema / corpus-mined
queries (roadmap Phase 2–3).

## Packages

- `data/wyly_expert_package_v5_babi2_word_standalone/`
- `data/wyly_expert_package_v5_babi3x_word_standalone/`
- each ships `alphabet.json` (+ hash in manifest) and `origin=standalone`
