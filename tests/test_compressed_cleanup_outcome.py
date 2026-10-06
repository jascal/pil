"""The outcome note's per-cell table must match the committed summary JSON."""

import json
from pathlib import Path

NOTES = Path(__file__).resolve().parents[1] / "docs" / "notes"


def test_outcome_table_matches_summary():
    s = json.loads((NOTES / "compressed_cleanup_summary.json").read_text())
    md = (NOTES / "compressed_cleanup_outcome.md").read_text().splitlines()
    spec = [
        (2, "F_cov", 4),
        (3, "E", 3),
        (4, "A", 3),
        (5, "eps_train", 3),
        (6, "eps_median", 3),
        (7, "rho_eps_nonpositive", 3),
        (8, "median_rho_dir", 4),
    ]
    for key, c in s["cells"].items():
        task, _, obj = key.split("/")
        rows = [ln for ln in md if ln.startswith(f"| {task} | {obj} |")]
        assert len(rows) == 1, key
        col = [x.strip() for x in rows[0].split("|")[1:-1]]
        for i, k, dp in spec:
            assert col[i] == f"{c[k]['mean']:.{dp}f}", (key, k, col[i])
