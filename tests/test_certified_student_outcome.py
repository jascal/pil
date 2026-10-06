"""The outcome note's per-seed table must match the committed summary JSON."""

import json
from pathlib import Path

NOTES = Path(__file__).resolve().parents[1] / "docs" / "notes"


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def test_outcome_table_matches_summary():
    s = json.loads((NOTES / "certified_student_summary.json").read_text())
    md = (NOTES / "certified_student_outcome.md").read_text().splitlines()
    for arm, c in s["cells"].items():
        for p in c["per_seed"]:
            rows = [ln for ln in md if ln.startswith(f"| {arm} | {p['seed']} |")]
            assert len(rows) == 1, (arm, p["seed"])
            col = [x.strip() for x in rows[0].split("|")[1:-1]]
            assert col[2] == f"{p['faith_seen']:.3f} / {p['faith_held']:.3f}"
            assert col[3] == _fmt(p["cert16"]) and col[4] == _fmt(p["cert23"])
            assert col[5] == _fmt(p["CF23"]) and col[6] == _fmt(p["FC23"])
            assert col[7] == _fmt(p["held_faith_among_cert"])
    v = s["verdict"]
    assert v["H1"]["verdict"] == v["H2"]["verdict"] == v["H3"]["verdict"] == "fail"
