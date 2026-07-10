# P2: corpus-mined queries (no hand query JSON)

**Goal.** Remove the last external gold file used for *admission* judging. Judge
queries are mined from the corpus stream under temporal holdout; windows/tables
are fit only on the earlier fit region.

## Protocol

| piece | source |
|---|---|
| Inline Q/A | corpus `Q: … A: ans.` (bAbI render) |
| Holdout | last 10% of Q/A events by corpus order |
| Judge set | held-out events → prompts ending at `A:`, max 1000 |
| Fit region | text through last fit-region Q/A; word windows from fit only |
| Bench | still classic test split (`babi_qa*_bench.json`) — true OOD |
| Stack | SOFT=0 + WordCodec + corpus labels + origin=standalone |

## API

- `pil/query_mine.py` — `mine_inline_qa`, `mine_and_save`
- `WYLY_QUERY_SOURCE=corpus_mined` → auto `data/wyly_queries_mined_{DS}.json`
- `build_word_babi.py` rebuilds alphabet + windows + mined queries together
- Campaign: `experiments/campaign_standalone_p2.py`

## Results (served on test benches)

| qa | served | query_source | parity (mined judge) | soft | alphabet |
|---|---|---|---|---|---|
| qa2 | **1.000** (1000/1000) | corpus_mined | 900/900 | false | word |
| qa3 | **0.998** (998/1000) | corpus_mined | 895/900 | false | word |

Same ceilings as P0+P1 with **hand** query files. Estate2 still admits at ~+0.50
cover-marginal on mined queries (slot `A`+`:`).

## Scaffold status

| Host / hand prior | Status |
|---|---|
| Teacher logits / embeds | gone |
| Soft SGD | gone |
| Host BPE | gone (WordCodec) |
| Hand `wyly_queries_*.json` for admit | **gone** (mined) |
| Hand estate2 form / member json | still present |
| External **test** bench file | still present (evaluation only) |

## Note on leakage

Fit windows are restricted to `text[:fit_end_char]` so count tables do not train on
held-out judge stories. The test bench remains a separate classic split (not the
mine holdout).
