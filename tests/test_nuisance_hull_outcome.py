"""The outcome note's per-cell table must match the committed summary JSON."""

import json
from pathlib import Path

NOTES = Path(__file__).resolve().parents[1] / "docs" / "notes"


def test_outcome_table_matches_summary():
    summary = json.loads((NOTES / "nuisance_hull_summary.json").read_text())
    md = (NOTES / "nuisance_hull_outcome.md").read_text().splitlines()
    for key, c in summary["cells"].items():
        obj = key.split("/")[1]
        rows = [ln for ln in md if ln.startswith(f"| {obj} |")]
        assert len(rows) == 1, key
        col = [x.strip().replace("**", "") for x in rows[0].split("|")[1:-1]]
        assert col[1].split(" ")[0] == f"{c['H_all']['mean']:.3f}", key
        assert col[2].split(" ")[0] == f"{c['H_const']['mean']:.3f}", key
        assert col[3] == ", ".join(str(p["hull_classes"]) for p in c["per_seed"]), key
        assert col[4].split(" ")[0] == f"{c['F_cov']['mean']:.3f}", key
        assert col[5] == f"{c['median_rho_loc']['mean']:.3f}", key
    combos = sum(p["hull_combos_checked"] for c in summary["cells"].values() for p in c["per_seed"])
    assert f"{combos:,}" in "\n".join(md)
