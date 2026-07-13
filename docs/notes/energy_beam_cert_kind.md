# Text energy-beam certificate metadata (Slice 4)

**Status:** functional and metadata-correct. The real rosetta expert package can now serve an
`M>1` / wider-beam request without crashing. The beam commits the first token of its selected
bounded trajectory; this slice makes no claim that text accuracy improves.

`cert_kind="per-token"` marks a single-step commit with no lookahead, including the unchanged
`M=1`, `beam_width=1` corner and any multi-step request that falls back to that corner.
`cert_kind="M-step-lookahead"` marks a genuine beam commit whose first token was selected over
hypothetical step-2 through step-M continuations.

The lookahead certificate is bounded by the configured beam. It is not a global-optimality
proof: increasing M or beam width can expose another trajectory and change the commit. Whether
this mechanism improves text accuracy is an explicitly deferred, separate measurement.
