# Standalone learner ablations E0 / E1 (no teacher labels or embeds)

**Question.** How much of the bAbI estate ceiling depends on a host LLM scaffold
(teacher next-token labels + embed PCA geometry)?

**Answer.** On qa2/qa3 **none of the served package ceiling** — structure + corpus
stream labels + query gold suffice. Soft next-token agreement drops without the
teacher; the **package does not**.

## Protocol

| | E0 | E1 |
|---|---|---|
| `WYLY_LABELS` | **corpus** (window `target`, no teacher file) | same |
| `WYLY_CONCEPT_INIT` | **random** (no host embed) | same |
| `WYLY_CONCEPTS` | 0 | **1** (ConceptSpace from counts) |
| Query judge / bench | dataset gold | same |
| Tokenizer | still Qwen (package key format) | same |
| Campaign | `experiments/campaign_e0e1_standalone.py` | same |

Teacher decisions (`wyly_teacher_*`) and `qwen3b_embed.npy` are **not read**.

## Served scoreboard

| arm | qa | served | estate2 | concept groups in package | origin |
|---|---|---|---|---|---|
| E0 | qa2 | **1.000** (1000/1000) | is | 0 | document |
| E0 | qa3 | **0.998** (998/1000) | before | 0 | document |
| E1 | qa2 | **1.000** | is | 0 | document |
| E1 | qa3 | **0.998** | before | 0 | document |
| Band B (teacher) | qa2 | 1.000 | is | — | teacher |
| Band B (teacher) | qa3 | 0.998 | before | — | teacher |

Packages: `data/wyly_expert_package_v5_babi{2,3x}_{e0,e1}`.  
Numbers: `data/standalone_e0e1_scoreboard.json`.

## What moved / what didn’t

1. **Serve ceiling is scaffold-independent on this domain.** Estate2 still admits at
   ~+0.51 cover-marginal with corpus labels; parity 1000/1000; bench matches Band B.
2. **Soft student still likes the teacher.** Corpus-label next-token agreement sits
   ~0.60–0.69 vs ~0.70–0.75 under teacher labels — distillation helps the *soft*
   path, not the *certified* one here.
3. **E1 concepts run but do not pay on package.** Sleep logs: ~27 concepts (24
   merges) on babi2, ~52–54 on babi3x; cluster-mined frames and `cmember` /
   `moveloc` candidates appear. None survive query-calibrated selection once
   estate2 is in; final packages are estate2 dgate + counts only. On bAbI,
   estate’s mined entity/loc/verb sets already supply the load-bearing “concepts.”
4. **Residual host dependency** is only the **tokenizer** (token ids as the
   package alphabet) — not logits, not embeds, not a forward pass at serve.

## Reading for the standalone-learner thesis

- **S1 (host-free expert at serve):** strengthened — works without teacher scaffold.  
- **S2 (learner with no LLM in training):** strengthened **for this structured
  domain** — corpus next-token + random geometry + estate form is enough.  
- **Concept lattice as general abstraction:** not stress-tested here; bAbI is
  already solved by a fixed world-state form. Open text (wiki) is the right
  place to ask whether ConceptSpace beats estate-like templates without a teacher.
