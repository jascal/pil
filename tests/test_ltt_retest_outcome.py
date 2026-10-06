"""The outcome note's per-seed table must match the committed summary JSON."""

import json
from pathlib import Path

NOTES = Path(__file__).resolve().parents[1] / "docs" / "notes"


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def test_outcome_table_matches_summary():
    s = json.loads((NOTES / "ltt_retest_summary.json").read_text())
    md = (NOTES / "ltt_retest_outcome.md").read_text().splitlines()
    for p in s["per_seed"]:
        rows = [ln for ln in md if ln.startswith(f"| {p['seed']} |")]
        assert len(rows) == 1, p["seed"]
        col = [x.strip() for x in rows[0].split("|")[1:-1]]
        keys = ("faith_seen", "gate_auc", "violation_rate", "coverage", "non_issuance", "FCa_mean")
        assert col[1:7] == [_fmt(p[k]) for k in keys]
        assert col[7:9] == [_fmt(p["faith_issued_mean"]), _fmt(p["held_faith_issued_mean"])]
        fd = p["first_draw"]
        assert col[9] == f"{_fmt(fd['coverage'])} / {_fmt(fd['FCa'])}"
    v = s["verdict"]
    assert [v[h]["verdict"] for h in ("H1", "H2", "H3")] == ["pass", "fail", "fail"]
    assert s["pooled"]["draws"] == 1000
