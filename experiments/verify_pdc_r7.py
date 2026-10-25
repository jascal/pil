"""Independent NumPy endpoint admission and fixed-outside decode audit."""

import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--results", required=True)
a = p.parse_args()
d = np.load(a.data)
root = Path(a.results)
results = json.loads((root / "results.json").read_text())
report = {}
for arm in ["decode", "admission"]:
    frame = np.load(root / f"{arm}_final.npz")
    U, b = frame["U"], frame["bias"]
    r = d["protected_residuals"]
    scores = r @ U.T + b
    full = np.concatenate((scores, d["outside_max"][:, None]), axis=1)
    assert np.array_equal(full.argmax(1), d["teacher_indices"])
    incidence = d["native_sources"] @ U.T
    partial = incidence + b
    target = d["teacher_indices"][:75]
    chosen = np.take_along_axis(partial, target[:, None, None], axis=2).squeeze(2)
    for i, t in enumerate(target):
        partial[i, :, t] = -np.inf
    margin = chosen - partial.max(2)
    tail = (r[:75] @ U.T)[:, None, :] - incidence
    delta2 = np.ptp(tail, axis=2)
    violation = np.maximum(0.5, delta2) - margin
    mask = (violation < 0).any(1)
    saved = results[arm]["final"]
    assert np.array_equal(mask, [x["admitted"] for x in saved])
    errors = []
    for i, x in enumerate(saved):
        j = x["source"]
        errors.extend([abs(margin[i, j] - x["margin"]), abs(delta2[i, j] - x["two_delta"])])
        if x["admitted"]:
            assert margin[i, j] > x["gamma"] > delta2[i, j]
    assert max(errors) < 1e-8
    norm = np.linalg.norm(U, axis=1)
    G = (U / norm[:, None]) @ (U / norm[:, None]).T
    M, k = U.shape
    fp = (np.square(G).sum() - np.square(G.diagonal()).sum()) / (M * (M - 1))
    W = max(0, (M - k) / (k * (M - 1)))
    assert fp + 1e-10 >= W
    steps = json.loads((root / f"{arm}_steps.json").read_text())
    assert len(steps) == 200
    assert all(s["enforced_flips"] == 0 and 2 * s["delta"] <= 0.9 * s["m_protect"] for s in steps)
    counts = [results[arm]["initial_admitted"]] + [s["admitted"] for s in steps]
    assert all(x <= y for x, y in zip(counts, counts[1:], strict=False))
    report[arm] = dict(
        admitted=int(mask.sum()),
        full_host_matches=len(r),
        endpoint_error=max(errors),
        fp=float(fp),
        welch=W,
        nonzero_steps=sum(s["alpha"] > 0 for s in steps),
        per_step_premises="checked logged budgets; endpoint independently reconstructed",
    )
(root / "verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
