# Compressed binding (T4(c)): clean-up certificates without a left inverse — outcome

**Pre-registration:** [`compressed_cleanup_prereg.md`](compressed_cleanup_prereg.md), with addendum A.
- Provenance, in order:
  - pre-registration `6a88615`, pushed before any code;
  - addendum A `c0db4fd` (smoke disclosure, no design change);
  - script and tests frozen at `e22ef2c` (`experiments/compressed_cleanup.py`, sha256 `e34deb7c…`). It reuses
    `halfspace_verdict.py`, `cleanup_radius.py` and `cleanup_certificate.py` unchanged, and the run used all of them
    unchanged.
- The theorem is jascal/i-orca#36 (`compressed_cleanup_certified`, kernel-checked), **open, awaiting the maintainer's
  merge** at the time of this run.
- Seeds: stimulus 111, split 112, fit {0, 1, 2}. 7,200 train and 3,600 test per task; 0 dropped. Ridge decoder
  `λ = 10⁻³·σ_max(W)²`.
- The dump and the run were held under one `systemd-inhibit` lock and were not interrupted (2,183 s).
- Summary: [`compressed_cleanup_summary.json`](compressed_cleanup_summary.json). Every count below is `empirical`.

## Verdict

**H1 (`F_cov ≥ 0.05` in at least one LIST `d_F=32` cell): FAIL.** Per-context coverage is **0** in all 9 LIST fits.

- **Soundness:** none of the aborts fired: the iff (`clean` = actual clean-up through the ridge decoder, `agree` =
  host agreement), `compression_offset_bound` on every (context, role, rival), and `ρ_ε ≤ ρ_dir`. 0 ties; no fit
  flagged; Soufflé and Python agreed.

## Per cell (seed mean [min, max]; 3,600 test contexts)

| task | objective | `F_cov` | E (exact clean-up) | A (host agrees) | `ε_train` | median `ε` | `ρ_ε ≤ 0` share | median `ρ_dir` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LIST | mse | 0.0000 | 0.000 | 0.713 | 0.890 | 0.772 | 1.000 | 0.2303 |
| LIST | cert | 0.0000 | 0.000 | 0.960 | 0.709 | 0.611 | 1.000 | 0.2272 |
| LIST | t6 | 0.0000 | 0.000 | 0.981 | 0.887 | 0.775 | 1.000 | 0.1819 |
| SVO | mse | 0.5556 | 0.704 | 0.785 | 0.157 | 0.077 | 1.000 | 0.5318 |
| SVO | cert | 0.2494 | 0.280 | 0.868 | 0.217 | 0.120 | 1.000 | 0.6613 |
| SVO | t6 | 0.6230 | 0.738 | 0.845 | 0.154 | 0.075 | 1.000 | 0.6888 |

- `ε(σ) = ‖(P·W − I)T(σ)‖/‖T(σ)‖`. At most 0.1% of test contexts exceeded `ε_train` in any fit.
- The radius shrink `median(max(ρ_ε, 0)/ρ_dir)` is 0 in every cell, and so is the code-centred ball coverage.

## Reading (interpretation)

1. **LIST `d_F=32` is not certifiable through a ridge decoder: compression destroys clean-up.** The compression error
   is large (median `ε` 0.61–0.78), so most of each structure tensor's norm lies in `W`'s null space and is lost in
   decoding. Clean-up is never exact (E = 0), although GPT-2 agrees with the code point on 71–98% of contexts. Decoding,
   not the decision, is the bottleneck. Nothing in the training objectives keeps `T(σ)` in `W`'s row space.
2. **Where `W` is injective, the exact per-context check still works through an approximate decoder.** The SVO control
   certifies 0.56 (`mse`) and 0.62 (`t6`) with the ridge decoder, a little below #141's 0.67 / 0.69 with the exact
   `W⁺`.
3. **The degraded radius `ρ_ε` is vacuous on these substitutes,** even at small `ε` (SVO, about 0.08). `ρ_ε ≤ 0` for
   every context in every cell. The theorem is correct, but its worst-case term `ε‖T(σ)‖‖w_s‖‖d_a‖` (Cauchy–Schwarz on
   the compression offset) exceeds the slack by a wide margin (median `ρ_ε` ≈ −2 to −22). This repeats the lesson of
   #138/#140: uniform norm bounds discard the direction of the error. The exact per-context check keeps it, and needs
   no `ε`.
4. **Bringing LIST `d_F=32` into scope** would need a substitute whose structure tensors are decodable: for example,
   training with a penalty on `‖(P·W − I)T(σ)‖`, or `W` constrained to be injective on the span of the structures. That
   is a new study, not a change to this one.

## Not claimed

- One decoder (ridge with a fixed `λ`). A decoder tuned to the structures that occur could reduce `ε`.
- The certificate is per context; coverage is `empirical`. GPT-2 small, two templated tasks, the TPR family only.
