"""Check saved R4 witnesses without invoking the LP solver or its oracle code."""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def verify(out, dump):
    report = json.loads((out / "results.json").read_text())
    assert hashlib.sha256(dump.read_bytes()).hexdigest() == report["dump_sha256"]
    raw = [json.loads(line) for line in dump.read_text().splitlines()]
    writes = np.array([r["d"] for r in raw])
    residuals = writes.sum(axis=1)
    assert len(raw) == len(report["heldout"]["records"]) == 80
    checked, admitted, max_violation = 0, 0, 0.
    guard_slacks = []
    for row in report["heldout"]["records"]:
        x0, witness = row["position"], row["witness"]
        assert row["status"] == "reducible"
        path = Path(witness["frame_path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == witness["frame_sha256"]
        data = np.load(path)
        frame = data["coefficients"] @ data["basis"].T
        token_index = {int(t): i for i, t in enumerate(data["tokens"])}
        scores = residuals @ frame[:, :-1].T + frame[:, -1]
        gaps = [scores[n, token_index[r["pred"]]] - scores[n, token_index[int(v)]]
            for n, r in enumerate(raw) for v in r["cands"][1:]]
        assert min(gaps) >= 1 - 1e-7
        max_violation = max(max_violation, 1 - min(gaps))
        source = writes[x0, witness["source"]]
        candidates = [token_index[int(v)] for v in raw[x0]["cands"]]
        partial = frame[candidates, :-1] @ source + frame[candidates, -1]
        margin = float(partial[0] - max(partial[1:]))
        assert margin >= 1 - 1e-7
        tail = frame[candidates, :-1] @ (residuals[x0] - source)
        delta = float((max(tail) - min(tail)) / 2)
        kappa = float((max(tail) + min(tail)) / 2)
        certificate = witness["admission"]
        assert abs(margin - certificate["margin"]) < 1e-7
        assert abs(delta - certificate["delta"]) < 1e-7
        assert abs(kappa - certificate["kappa"]) < 1e-7
        assert np.max(np.abs(tail - kappa)) <= delta + 1e-7
        guard = margin > certificate["gamma"] > 2 * delta
        assert guard == certificate["admitted"]
        if guard:
            admitted += 1
            guard_slacks.append(min(margin - certificate["gamma"], certificate["gamma"] - 2 * delta))
            assert np.all(partial[0] + tail[0] > partial[1:] + tail[1:])
        checked += 1

    # Exact control dual check in the original rational affine coordinates.
    # Rows are -(score_target - score_rival) <= -1 for three full contexts,
    # followed by the tested singleton at context 0.
    def margin_row(vector, target):
        augmented = [Fraction(v) for v in [*vector, 1]]
        return [-v for v in augmented] + augmented if target == 0 else augmented + [-v for v in augmented]

    base = [margin_row([1, 1], 0), margin_row([1, 0], 1), margin_row([0, 1], 1)]
    exact_duals = 0
    for attempt in report["refusal"]["records"][0]["attempts"]:
        source = [1, 0] if attempt["source"] == 0 else [0, 1]
        matrix = base + [margin_row(source, 0)]
        multipliers = [Fraction(v) for v in attempt["dual_rational"]]
        assert all(v >= 0 for v in multipliers)
        assert all(sum(multipliers[i] * matrix[i][j] for i in range(4)) == 0 for j in range(6))
        assert -sum(multipliers) < 0
        exact_duals += 1
    positive = np.load(out / "positive_native.npz")
    pos_scores = positive["writes"][:, 0] @ positive["frame"].T + positive["bias"]
    assert all(pos_scores[n, n % 2] - pos_scores[n, 1-n % 2] >= 2 for n in range(64))
    pos_tail = positive["writes"][:, 1] @ positive["frame"].T
    assert np.all(pos_tail == 0)
    return {"saved_primal_frames_checked": checked, "margin_admissions_checked": admitted,
        "numeric_max_faithfulness_violation": max_violation,
        "minimum_admitted_guard_slack": min(guard_slacks),
        "positive_admissions_checked": 64, "exact_rational_refusal_duals_checked": exact_duals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.out, args.dump)
    (args.out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
