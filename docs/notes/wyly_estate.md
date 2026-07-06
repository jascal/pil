# estate: the entity-state register — bAbI qa1 solved at 1.000

The rule form designed for entity-state binding, after every prior form (verbatim pointer,
fixed-shift echo, next-member-after) failed admission or deployment.

## The form: (entity, attribute, value), one register per attribute

A last-writer-wins fold: at each clean occurrence of an entity-class token with a value-class
token within reach, `register[entity] := value`; the feature is `register[query entity]`. The
general shape is EAV — realized as a **family of registers, one per attribute** (each attribute
identified by its value-class), with **arbitration selecting the attribute at query time**:
different question forms key the dgate tables differently, so the right attribute's register
wins the confidence race without hand-routing. qa1 is the degenerate one-attribute (location)
case; qa2+ (possession, object location) are additional family members.

## Self-grounding — both member sets mined from the data

- **entities** = capitalized-class tokens NOT predominantly in question contexts (the avoid-set
  statistics exclude 'Where' automatically): mined {John, Mary, Daniel, Sandra} exactly.
- **values** = the transition-slot histogram (successors of 'the' within reach of an entity,
  95% mass): mined the 6 locations exactly.
- **slot signature** = the (t−2, t−1) pair where the register correctly predicts the next token
  (mined: " A", ":") — gates firing to answer positions.

## The result

| | bAbI qa1, 1000 UNSEEN test questions |
|---|---|
| Qwen2.5-3B-Instruct (the teacher) | 0.782 |
| package, best pre-estate | 0.527 |
| **package + estate** | **1.000** |

Admission: marginal **+0.518, 3/3 query folds**, admitted at sleep 0. The binding is carried by
a **6-key dgate** over a PROVED extractor (Soufflé certificate `estate`, 192/192 —
wyly_derived_certify). **The certified student surpasses its teacher by 22 points** on unseen
stories: the teacher taught imitation targets; the compiled rule computes the task exactly.

## Four bugs excavated on the way (each a general lesson)

1. **Concept groups were the wrong value class** (moveloc's failure): distributional clusters
   ≠ role classes; the transition-slot mining is the right grounding.
2. **The min_keys floor biases against surgical rules**: estate's table is ~6 keys by
   construction; support floors tuned for broad rules reject exactly the cleanest ones.
3. **Dirty-ratio double counting**: summing per-shift bincounts counts one occurrence twice;
   the OR-over-shifts per-position mask is correct (this alone had emptied the entity set).
4. **Q-colon/A-colon key conflation**: the dgate key (feature, ':') mixed question and answer
   positions, making a 1000/1000-perfect feature score 0.000 through its table — fixed by the
   mined slot signature. *A perfect feature can be destroyed by an ambiguous key.*

Probes: `experiments/probe_moveloc.py` (the form was 100% all along at token level),
`experiments/probe_estate.py` (self-grounded 1000/1000). Runtimes: rosetta PR #41,
sgiandubh PR #26 (both merged).
