import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

import wyly_lm_v5 as v5  # noqa: E402
from serve_package import decide, load_package  # noqa: E402


def test_raw_counts_align_in_keytable_and_builders():
    tab = v5.KeyTable(
        torch.tensor([9, 2]), torch.tensor([90, 20]), torch.tensor([0.7, 0.6]),
        cnt=torch.tensor([7, 3]), tot=torch.tensor([8, 4]),
    )
    assert tab.k.tolist() == [2, 9]
    assert tab.v.tolist() == [20, 90]
    assert tab.c.tolist() == pytest.approx([0.6, 0.7])
    assert tab.cnt.tolist() == [3, 7]
    assert tab.tot.tolist() == [4, 8]

    key = torch.tensor([2, 1, 2, 1, 2, 1, 1])
    val = torch.tensor([7, 5, 7, 5, 8, 5, 6])
    best, _ = v5.best_per_key(key, val, minsupp=1, mindet=0.5)
    assert best.k.tolist() == [1, 2]
    assert best.v.tolist() == [5, 7]
    assert best.cnt.tolist() == [3, 2]
    assert best.tot.tolist() == [4, 3]
    assert best.c.tolist() == pytest.approx([3 / 6, 2 / 5])

    w = torch.tensor([[1, 2, 3], [1, 2, 3]])
    ytail = torch.tensor([4, 4])
    online = v5.OnlineFrame((1,), vocab=20, dev="cpu", minsupp=1, mindet=0.5)
    online.update(w, ytail)
    online.refresh()
    assert online.table.k.tolist() == [1, 2, 3]
    assert online.table.v.tolist() == [2, 3, 4]
    assert online.table.cnt.tolist() == [2, 2, 2]
    assert online.table.tot.tolist() == [2, 2, 2]

    tuples = v5.TupleFrame((2, 1), vocab=20, dev="cpu", minsupp=1, mindet=0.5)
    tuples.update(w, ytail)
    tuples.refresh()
    rows = {tuple(k): (v, cnt, int(tot)) for k, v, cnt, tot in zip(
        tuples.keys2d.tolist(), tuples.vals2d.tolist(), tuples.cnts2d.tolist(),
        tuples.tots2d.tolist(), strict=True)}
    assert rows == {(1, 2): (3, 2, 2), (2, 3): (4, 2, 2)}
    assert sorted(tuples.leaf_cnt.tolist()) == [2, 2]
    assert sorted(int(x) for x in tuples.leaf_tot.tolist()) == [2, 2]
def _table(keys, vals, cnts, tots):
    cnt = torch.tensor(cnts)
    tot = torch.tensor(tots, dtype=torch.float)
    return v5.KeyTable(torch.tensor(keys), torch.tensor(vals), cnt.float() / (tot + v5.ALPHA),
                       cnt=cnt, tot=tot)


def test_emit_full_energy_counts_and_classic_gate(monkeypatch):
    uv = torch.arange(30)
    cls = torch.tensor([10, 11])
    model = SimpleNamespace(
        rules=[(name, None) for name in ("kg", "skip", "mate", "dfeat", "dgate2", "mined")],
        rules2=[],
        counts=torch.zeros((30, 2), dtype=torch.long),
    )
    model.counts[5] = torch.tensor([3, 1])
    codec = SimpleNamespace(token_str=lambda token: f"tok{token}")

    online = v5.OnlineFrame((1,), vocab=29, dev="cpu", minsupp=1)
    online.table = _table([1], [2], [4], [5])
    skip = _table([3], [4], [6], [7])
    b2 = 31
    mate = _table([(2 + 1) * b2 + 3], [4], [8], [9])
    members = torch.zeros(30, dtype=torch.bool)
    members[2] = True
    dfeat = _table([(2 + 1) * b2 + 3], [5], [10], [12])
    dgate2 = _table([(2 + 1) * b2 + 3], [6], [11], [13])
    mined = SimpleNamespace(
        t1=_table([(2 * 30 + 7) * 30 + 3], [8], [12], [14]),
        t2=_table([], [], [], []),
    )
    info = {
        "kg": ("kgram", online),
        "skip": ("skip", 2, skip),
        "mate": ("mate", mate, [2], [3], b2),
        "dfeat": ("dfeat", ("recent-member", {"members": members}), dfeat, b2),
        "dgate2": ("dgate2", ("sent-pair", {"members": members}), dgate2, b2),
        "mined": ("mined", mined),
    }
    monkeypatch.setattr(v5, "EMIT_INFO", info)
    monkeypatch.setattr(v5, "RULE_CONF", {})

    classic, _ = v5.emit_full(model, cls, uv, codec, 29, energy_mode=False)
    energy, _ = v5.emit_full(model, cls, uv, codec, 29, energy_mode=True)

    assert "alpha" not in classic and "schema_version" not in classic
    assert all("counts" not in rule and "total" not in rule for rule in classic["rules"])
    assert energy["alpha"] == v5.ALPHA
    assert energy["schema_version"] == 3

    assert energy["rules"][0]["counts"] == [4, 5]
    assert energy["rules"][1]["counts"] == {3: [6, 7]}
    assert energy["rules"][2]["counts"] == {"2:3": [8, 9]}
    assert energy["rules"][3]["counts"] == {"2:3": [10, 12]}
    assert energy["rules"][4]["counts"] == {"2:3": [11, 13]}
    assert energy["rules"][5]["counts"] == {3: [12, 14]}
    counts_tier = energy["rules"][6]
    assert counts_tier["support"] == 3
    assert counts_tier["total"] == 4
    assert counts_tier["counts"] == [3, 4]


def test_python_counts_round_trip_without_decision_change(tmp_path):
    manifest = {
        "model": "synthetic", "cover": "energy-beam", "M": 1, "beam_width": 1,
        "schema_version": 3, "alpha": 2.0, "W": 2, "n_rules": 3,
        "derived": [{"id": "feat0", "kind": "recent-member", "members": [5]}],
        "rules": [
            {"id": 0, "kind": "gate", "frame": {}, "slot": 1,
             "table": {"5": 9}, "confs": {"5": 0.9}, "counts": {"5": [9, 10]}},
            {"id": 1, "kind": "dgate", "feature": "feat0",
             "table": {"5:7": 11}, "confs": {"5:7": 0.8},
             "counts": {"5:7": [8, 9]}},
            {"id": 2, "kind": "ngram", "ctx": [7], "out": 12,
             "confidence": 0.7, "counts": [7, 8]},
        ],
    }
    without = copy.deepcopy(manifest)
    without.pop("schema_version")
    without.pop("alpha")
    for rule in without["rules"]:
        rule.pop("counts", None)

    loaded = []
    for name, payload in (("with", manifest), ("without", without)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        loaded.append(load_package(path))
    with_counts, without_counts = loaded

    assert with_counts[0][0]["counts"] == {5: (9, 10)}
    assert with_counts[0][1]["counts"] == {(5, 7): (8, 9)}
    assert with_counts[1][1][(7,)][-1] == (7, 8)
    assert without_counts[0][0]["counts"] == {}
    assert without_counts[0][1]["counts"] == {}
    assert without_counts[1][1][(7,)][-1] is None
    for context in ([5], [5, 7], [7], [1]):
        assert decide(context, *with_counts) == decide(context, *without_counts)
