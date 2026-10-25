# PDC R4 outcome, 2026-10-24

**Empirical, experiment concluded with a corrected negative-domain premise.**
Preregistration `pdc_r4_prereg.md`, SHA256
`eee3849e5ca687f869b3890b1fb9fc8e73e05a3976ff0dc524a28acef9eb838b`.
Report `results/pdc_r4/results.json`; solver-free artifact audit
`results/pdc_r4/verification.json`; per-position frame witnesses and `run.log`
are retained next to the report.

| Domain | N | Free-frame reducible | Margin admitted | Irreducible |
|---|---:|---:|---:|---:|
| Positive synthetic corpus | 64 | 64 | 64, both native and saved LP frames | 0 |
| Explicit three-context refusal control, target context 0 | 1 tested of 3 | 0 | 0 | 1, exact dual for each singleton |
| Held-out Qwen regime rerun | 80 | 80 | 11 of the first stored feasible frames | 0 |

No solver outcome was unknown. The held-out free-frame certificate set is
**still empty at N=80**. This preserves the original meaning of 0/40: absence
of certified irreducibility, not absence of source admission. The goal's
expectation that this regime is a negative admission domain is retired.
The refusal control demonstrates a genuine oracle refusal on the stated
three-context domain; it does not make the Qwen sample irreducible.

The 69 non-admitted Qwen witnesses fail the strict centered-tail margin guard
in their saved frames. They are **not** certified impossible to admit in every
faithful frame. The sweep stopped at its first feasible singleton frame as
preregistered; no search optimized the stronger admission predicate. Therefore
11/80 is the admission rate of the recorded witness procedure, not the optimum
over frames, sources or gamma. Free-frame reducibility alone is insufficient.

Every feasible LP includes one shared affine (U,b) faithful across all 80
positions, at gamma=1. The solver-free verifier reconstructs all 80 physical
frames from the saved basis and coefficients and rechecks the raw-vector
inequalities. Maximum faithfulness violation is 1.844e-10, within the fixed
1e-7 tolerance. The minimum strict guard slack among the 11 admissions is
0.102638. These are numerical data checks, not exact real-arithmetic proofs.

Positive admission uses the stored native frame U0=(1,0), U1=(-1,0), b=0.
Its omitted-source incidence is exactly zero. The refusal control has normalized
Farkas multipliers `[0,1/2,0,1/2]` and `[0,0,1/2,1/2]` on the three faithfulness
rows plus the tested singleton row. The independent verifier checks nonnegativity,
zero coefficient sums and negative RHS using rational arithmetic in the original
affine coordinates, without invoking the solver or its oracle module.

Kernel packaging is **proved** in i-orca `examples/pdc/PDC_Admission.thy`, strict
session `PDC`, Isabelle2025-2, zero holes. `admitted_source_preserves_decision`,
`pic.crystallization_admitted_write`, and `pic.crystallization_complement_certified`
assume the same strict margin > gamma > 2delta and the same candidate set, with
the target inside the tail bound. The runtime report stores those quantities.

Capture: `fieldrun/experiments/pdc/r4_qwen_sources.jsonl`, SHA256
`e8d40075d713786a95ed652f48796908b51294e91455d3330cd41be8186bcb00`.
It contains 80 records, 49 sources, dimension 896, 24 candidates per position.
The native pre-serialization reconstruction reports 1.00; the artifact's
six-decimal source serialization is the fixed input to the LP. This is a new
sample on the preregistered bundle/corpus, not a replication of the missing
historical `/tmp/tau` dump. No model gates or measured writes were trained.

Reproduce capture from fieldrun root (requires default `api` feature):

```sh
cargo build --release
target/release/fieldrun --bundle bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct --ids sweeps/corpora/english.json --recursion-explain --source-dump experiments/pdc/r4_qwen_sources.jsonl --n 80 --kcand 24
```

Then from pil root:

```sh
.venv/bin/python experiments/pdc_r4.py --dump ../fieldrun/experiments/pdc/r4_qwen_sources.jsonl --out results/pdc_r4
.venv/bin/python experiments/verify_pdc_r4.py --out results/pdc_r4 --dump ../fieldrun/experiments/pdc/r4_qwen_sources.jsonl
.venv/bin/pytest -q tests/test_pdc_admission.py
```

The three focused tests pass; relevant ruff checks pass. No original PIC file
was edited. The original success criterion demanding certified-impossible
admission on this Qwen domain is not achieved; the preregistered empty-certificate
alternative and separate refusal control are the honest outcomes. R5–R9 must
use measured admission verdicts rather than assume a positive/negative split.
