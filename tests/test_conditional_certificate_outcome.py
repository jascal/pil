"""The outcome note's per-seed table must match the committed summary JSON."""

import json
from pathlib import Path

NOTES = Path(__file__).resolve().parents[1] / "docs" / "notes"


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def test_outcome_table_matches_summary():
    s = json.loads((NOTES / "conditional_certificate_summary.json").read_text())
    md = (NOTES / "conditional_certificate_outcome.md").read_text().splitlines()
    for p in s["cell"]["per_seed"]:
        rows = [ln for ln in md if ln.startswith(f"| {p['seed']} |")]
        assert len(rows) == 1, p["seed"]
        col = [x.strip() for x in rows[0].split("|")[1:-1]]
        keys = ("faith_seen", "cert16", "FC16_ungated", "gate_auc", "coverage")
        assert col[1:6] == [_fmt(p[k]) for k in keys]
        assert col[6] == str(p["n_issued"])
        keys = ("FCi", "faith_issued", "held_const_issued", "held_faith_issued")
        assert col[7:11] == [_fmt(p[k]) for k in keys]
    v = s["verdict"]
    assert [v[h]["verdict"] for h in ("H1", "H2", "H3")] == ["pass", "pass", "pass"]
