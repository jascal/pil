# XOR coalition follow-up — validation

- Focused tests: `pytest -q tests/test_xor_coalition_sweep.py tests/test_rule_participation.py`:
  **10 passed**.
- Ruff passes on both new experiment scripts and the new tests; `git diff --check` passes.
- All 45 banks completed; frozen-source SHA-256 hashes still match the raw JSON.
  The prior seed-0 M=8/16 accuracies reproduce exactly. Every cumulative deletion
  order has identical intact and raw-only endpoints. The generated figure was inspected.
- Broad regression run (one CPU thread), excluding
  `test_fixed_recipe_anchor_reproduces_real_gate_dev_numbers` after a slow full-run attempt:
  **985 passed, 1 failed, 1 skipped, 1 deselected** in 111.66 s.
- The failure is the unchanged
  `tests/test_whitening_resweep_grok.py::test_fixed_recipe_anchor_reproduces_gate_dev`:
  attachment accuracy 0.48685224839400426 versus persisted 0.4760599571734475
  (tolerance 1e-6). Running this test alone, without collecting the new XOR tests,
  reproduces exactly the same failure (86.59 s). The implementation it tests is unchanged.
  No whitening code or anchors were modified in this project.
- The excluded Codex whitening anchor was also attempted in isolation and stopped
  while still running without a result; it remains unverified in this session.
