# Directional calibration — development result (seen data only)

**Selection locked before EVAL2 evaluation. No confirmation verdict is assigned here.** The rules are in
[`directional_calibration_prereg.md`](./directional_calibration_prereg.md) (signed 2026-09-26 14:29 UTC). Full metrics:
`results/directional_calibration_dev.json`. CAL has 150 decisions and EVAL has 300 per model, from #129/#130.

| Model | Candidate | EVAL certified | Violations | Mean net saving (layers) | Median certified exit |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | C0 | 10/300 (3.3%) | 0 | +0.032 | 22/24 |
| Qwen2.5-0.5B | C1 | 9/300 (3.0%) | 0 | +0.028 | 22/24 |
| Qwen2.5-0.5B | C2 | 9/300 (3.0%) | 0 | +0.032 | 22/24 |
| Qwen2.5-3B | **C0** | **96/300 (32.0%)** | **0** | **+0.391** | **34/36** |
| Qwen2.5-3B | C1 | 82/300 (27.3%) | 0 | +0.328 | 34/36 |
| Qwen2.5-3B | C2 | 69/300 (23.0%) | 0 | +0.271 | 34/36 |

**Selected: C0**, the baseline. All three meet the development violation gate on 3B, and C0 has the highest
3B mean net saving. The preregistered prediction that margin stratification would be selected was wrong on this
seen set. C0 reproduces the #130 3B baseline (32% coverage, +0.39 layers); no new empirical claim follows from
the development data. The selected rule is refitted on all 450 seen decisions per model for EVAL2, as signed.

The maximum `s`-recovery residual was `6.25e-6` at 0.5B and `2.20e-5` at 3B, below the `1e-3` gate. Full pil
checks before EVAL2: 862 tests passed; ruff passed.
