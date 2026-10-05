"""The outcome note's per-cell table must match the committed summary JSON (replaces a one-off check)."""

import json
from pathlib import Path

NOTES = Path(__file__).resolve().parents[1] / "docs" / "notes"
COLS = [
    (2, "F_cov_D", 4),
    (3, "F_cov_train", 4),
    (4, "F_cov_test", 4),
    (5, "F_cov_never", 4),
    (6, "E_D", 3),
    (7, "A_D", 3),
    (8, "median_rho_loc_over_n_D", 4),
]


def test_outcome_table_matches_summary():
    summary = json.loads((NOTES / "domain_exhaustion_summary.json").read_text())
    md = (NOTES / "domain_exhaustion_outcome.md").read_text().splitlines()
    for key, c in summary["cells"].items():
        task, d, obj = key.split("/")
        rows = [ln for ln in md if ln.startswith(f"| {task} `d_F={d[2:]}` | {obj} |")]
        assert len(rows) == 1, key
        col = [x.strip() for x in rows[0].split("|")[1:-1]]
        for i, k, dp in COLS:
            assert col[i].replace("**", "").split(" ")[0] == f"{c[k]['mean']:.{dp}f}", (key, k)
        assert col[9].replace(",", "") == str(c["min_uncertified_D"]), key
