"""The exploratory note's result table must match the committed summary JSON."""

import json
from pathlib import Path

NOTES = Path(__file__).resolve().parents[1] / "docs" / "notes"


def test_note_table_matches_summary():
    s = json.loads((NOTES / "ibp_feasibility_summary.json").read_text())
    md = (NOTES / "ibp_feasibility_exploratory.md").read_text().splitlines()
    for k, v in s["k"].items():
        row = [ln for ln in md if ln.startswith(f"| {k} |")]
        assert len(row) == 1, k
        col = [x.strip().replace("**", "") for x in row[0].split("|")[1:-1]]
        assert col[1] == str(v["decision_constant"])
        assert col[2] == str(v["certified_ibp"])
        assert col[3] == str(v["certified_tightest_box"])
        assert col[4] == str(v["certified_hull"])
        assert v["contained"] == 100
