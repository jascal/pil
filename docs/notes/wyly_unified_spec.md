# The unified expert package: one package, three origins, one contract

The user's direction, now shipped as PACKAGE.md v2 (rosetta PR #44): an expert package is the
SUPERSET of what any pipeline produces — document-derived, LLM-extracted, and feedback-trained
knowledge in one manifest, one arbitration, one delivery story:

> *An expert package is a set of arbitrating rules over a bounded domain; every answer is cited
> to its rule, its origin, and its evidence; anything else is an abstention.*

## The two-axis trust ladder

- **origin** (where it came from): `document` | `teacher` (+ `origin_model`) | `feedback` —
  per rule, defaulting from the manifest. wyly emissions stamp `teacher`.
- **stratum** (how it's held): 0 **attested** / 1 **certified** / 2 **supported**.

Stratum 0 is assigned AT SERVE TIME: packages may carry a **grounding sidecar** (manifest
`"grounding"` → the source text; wyly emit copies the training corpus). When the served
(canonical query + answer) statement is verbatim in the sidecar, the decision upgrades to
attested and quotes the span. Live demo (element expert, C++ spoke):

> "What is the atomic number of gold?" → canonical "The atomic number of Gold is" → **79** —
> `attested: true, stratum: 0`, quote: "…Barium belongs to group 2. **The atomic number of
> Gold is 79.** Mercury is in period 6…"

Teacher-origin knowledge held at document-grade evidence — the strongest cell of the matrix.
An estate answer, by contrast, is teacher-origin + certified (its output exists in no text) —
and the payload says so. Consumers never need to know which pipeline built the package.

## Status

Spec + both runtimes + wyly emission stamps + pyspoke/sgiandubh attestation shipped (rosetta
#44, sgiandubh #29). Remaining migration (documented in the spec, not yet coded): the classic
normrules/gram converter to manifest kinds with `origin: document` + span refs, and feedback
patches stamping `origin: feedback` when the tuple-reach patching lands. One more excavated
lesson en route: an edit script that asserts mid-way rolls back ALL its edits — grep-verify
after every multi-part patch (this bit twice today; it is now a reflex).

## Addendum: all three origins now produce; FFI parity closed

- **feedback**: `wyly_feedback.py --emit` writes patched packages with origin:feedback rules
  (677 on the element expert), each citing the prompt that produced it.
- **document**: the classic converter (rosetta #45).
- **teacher**: wyly emissions (with grounding sidecars).

The Rust-FFI tokenizer parity question is **closed**: encode is bit-exact between the FFI and
pil's TokenSpace on all probes; the historic C++-vs-python divergence decomposed entirely into
since-fixed runtime bugs (stratum plumbing, conf parsing) and the canon reach defect (above).