# Seed knob and admission-stability contract

`WYLY_SEED` is an integer environment variable read once when `experiments/wyly_lm_v5.py` is
imported. It defaults to `0`, preserving the historical behavior exactly. It controls the
generator used to shuffle the train/validation split in `main()` and the global Torch seed set
immediately before model construction. It intentionally does not control the local seed in the
`concept_init(random)` branch.

For the measured `pythia70m/wikitext` domain, package emission adds this object to every
admitted-tier rule except `kind: "ngram"` entries:

```json
{
  "admission_stability": {
    "n": 5,
    "N": 8,
    "sweep": "seed_sweep_summary.json@f8ac0574c436976d0ce414400a15ac5300dacdba",
    "domain": "pythia70m/wikitext"
  }
}
```

`n` is loaded from `data/seed_sweep_summary.json` after removing a trailing fitted support suffix
such as ` [278]` from the admitted rule name. A rule absent from the frequency table receives
JSON `null`, meaning “no measurement,” never `0`. The always-present counts tier and admitted
k-gram expansions are both `kind: "ngram"`, so neither carries the field. For any other
`(WYLY_TAG, WYLY_DS)` domain, or when the generated summary file is absent, the field is omitted
entirely. The commit SHA in `sweep` is deliberately fixed: it identifies the immutable historical
#84 sweep artifact and is not a lookup of the current repository revision.

`experiments/campaign_seed_sweep.py` formerly replaced `torch.Generator` and
`torch.manual_seed` temporarily to redirect literal seed-zero calls. That shim is retired; workers
now put `WYLY_SEED` in the subprocess environment before importing v5. The separate caller-side
Python, NumPy, and Torch bootstrap seeds remain unchanged for external-seed invariance checks.

Run the standing reproduction directly on the GPU host:

```console
python experiments/repro_frozen_composition.py
python experiments/repro_frozen_composition.py --seed 3
python experiments/repro_frozen_composition.py --emit-manifest /tmp/wyly-manifest.json
```

The default compares both unset and explicit `WYLY_SEED=0` regenerations with the frozen seed-zero
record. `--seed N` reports one regeneration and also checks it when a matching B3 record exists.
`--emit-manifest PATH` is available in default seed-zero mode and emits from its already captured
explicit-seed model, without a third regeneration.

**STANDING INVARIANT: any future edit to `experiments/wyly_lm_v5.py` must re-run
`experiments/repro_frozen_composition.py` in both default mode and `--seed 3` mode before merge.**

**Convention (bug class hit twice — #81 corrective and this slice's guard 1):** any script
comparing freshly computed `core_sw` numbers against a historical reference file must use a
tolerance comparison (≤1e-4 absolute; composition compared exactly), never `==` — this repo
runs across CPU (lane sandboxes) and GPU (the host) as a matter of course, and float32
reduction order differs by ~1e-4 at worst.
