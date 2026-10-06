"""The outcome note's per-seed table must match the committed summary JSON."""

import json
from pathlib import Path

NOTES = Path(__file__).resolve().parents[1] / "docs" / "notes"


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def test_outcome_table_matches_summary():
    s = json.loads((NOTES / "ltt_certificate_summary.json").read_text())
    md = (NOTES / "ltt_certificate_outcome.md").read_text().splitlines()
    for p in s["cell"]["per_seed"]:
        rows = [ln for ln in md if ln.startswith(f"| {p['seed']} |")]
        assert len(rows) == 1, p["seed"]
        col = [x.strip() for x in rows[0].split("|")[1:-1]]
        assert col[1:4] == [_fmt(p["faith_seen"]), _fmt(p["gate_auc"]), str(p["rejected"])]
        assert col[4] == (f"{p['cal_w']}/{p['cal_n']}" if p["cal_n"] else "—")
        keys = ("coverage", "FCi", "faith_issued", "held_faith_issued", "coverage_emp", "FCi_emp")
        assert col[5:11] == [_fmt(p[k]) for k in keys]
    v = s["verdict"]
    assert v["H1"]["violations"] == 4
    assert [v[h]["verdict"] for h in ("H1", "H2", "H3")] == ["fail", "pass", "fail"]
