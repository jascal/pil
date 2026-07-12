import ast
import copy
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

import wyly_lm_v5 as v5  # noqa: E402
from serve_package import decide, load_package  # noqa: E402

from experiments import repro_frozen_composition as repro  # noqa: E402


def _repro_record(names=("gate [1]",), *, agree=0.346, cover=0.5, agree_fired=0.692):
    return {
        "admitted_names": list(names),
        "core_sw": {"agree": agree, "cover": cover, "agree_fired": agree_fired},
    }


def test_frozen_composition_comparison_allows_small_core_sw_drift(tmp_path, capsys):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_repro_record()))
    actual = _repro_record(agree=0.346 + repro.CORE_SW_TOLERANCE / 2)

    repro._assert_matches(actual, reference_path)

    output = capsys.readouterr().out
    assert "agree=+5.00e-05" in output
    assert "cover=+0.00e+00" in output
    assert "agree_fired=+0.00e+00" in output
    assert "(tolerance 0.0001)" in output


def test_frozen_composition_comparison_rejects_large_core_sw_drift(tmp_path, capsys):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_repro_record()))
    actual = _repro_record(agree=0.346 + repro.CORE_SW_TOLERANCE * 2)

    with pytest.raises(AssertionError, match="core_sw drift exceeds tolerance"):
        repro._assert_matches(actual, reference_path)

    assert "agree=+2.00e-04" in capsys.readouterr().out


def test_frozen_composition_comparison_rejects_composition_drift(tmp_path, capsys):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_repro_record()))

    with pytest.raises(AssertionError):
        repro._assert_matches(_repro_record(names=("different gate",)), reference_path)

    assert "core_sw drift vs" in capsys.readouterr().out


def test_wyly_seed_is_read_once_and_threaded_to_both_internal_sites():
    source = Path(v5.__file__).read_text()
    tree = ast.parse(source)
    seeded_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "manual_seed"
    ]
    wyly_seed_calls = [
        node for node in seeded_calls
        if len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "WYLY_SEED"
    ]
    literal_zero_calls = [
        node for node in seeded_calls
        if len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 0
    ]
    assert len(wyly_seed_calls) == 2
    assert len(literal_zero_calls) == 1  # concept_init(random) deliberately remains local seed 0.
    assert source.count('WYLY_SEED = int(os.environ.get("WYLY_SEED", "0"))') == 1
    assert v5.WYLY_SEED == int(os.environ.get("WYLY_SEED", "0"))


def test_manifest_stability_annotation_decisions():
    table = [{"name": "known gate", "count": 7, "seeds": list(range(1, 8))}]
    expected = {
        "n": 7,
        "N": 8,
        "sweep": "seed_sweep_summary.json@f8ac0574c436976d0ce414400a15ac5300dacdba",
        "domain": "pythia70m/wikitext",
    }
    assert v5._stability_annotation("known gate [42]", "gate", table, True) == expected
    absent = v5._stability_annotation("unmeasured gate [9]", "dgate", table, True)
    assert absent == {**expected, "n": None}
    assert v5._stability_annotation("known gate [42]", "ngram", table, True) is None
    assert v5._stability_annotation("known gate [42]", "gate", table, False) is None


def test_rosetta_round_trip_ignores_admission_stability(tmp_path):
    manifest = {
        "model": "synthetic",
        "cover": "fixed",
        "W": 1,
        "n_rules": 3,
        "derived": [{"id": "feat0", "kind": "recent-member", "members": [5]}],
        "rules": [
            {
                "id": 0, "kind": "gate", "frame": {}, "slot": 1,
                "table": {"5": 9}, "confs": {"5": 0.9}, "citation": ["fixture"],
            },
            {
                "id": 1, "kind": "dgate", "feature": "feat0",
                "table": {"5:7": 11}, "confs": {"5:7": 0.8}, "citation": ["fixture"],
            },
            {
                "id": 2, "kind": "ngram", "ctx": [7], "out": 12,
                "confidence": 0.7, "citation": ["fixture"],
            },
        ],
    }
    annotated = copy.deepcopy(manifest)
    annotation = {"n": 5, "N": 8, "sweep": "fixture@sha", "domain": "fixture"}
    annotated["rules"][0]["admission_stability"] = annotation
    annotated["rules"][1]["admission_stability"] = annotation

    loaded = []
    for label, payload in (("plain", manifest), ("annotated", annotated)):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload))
        loaded.append(load_package(path))
    plain, with_annotation = loaded
    assert plain[0] == with_annotation[0]
    assert plain[1] == with_annotation[1]
    for context in ([5], [7], [1]):
        assert decide(context, *plain) == decide(context, *with_annotation)


def test_regression_budget_field_decisions(tmp_path, monkeypatch):
    # Real artifact on disk at data/composite_discriminator.json (stage1: 143/62).
    result = v5._regression_budget_field("pythia70m", "wikitext")
    assert result is not None
    assert result["rate"] == 62 / 205
    assert result["rate_definition"] == (
        "deduplicated regressions / (deduplicated gains + regressions)"
    )
    assert result["eval_unit"] == "held-out split row"
    assert result["measured_on"] == ["Stage 1 / #81 GATED1 on test_te"]
    assert result["sweep@sha"] == "#81@3c5a96e + #86@9681595 (data/frames_at_scale.json)"
    assert result["domain"] == "pythia70m/wikitext"
    assert result["operating_point"] == (
        "single GATED1 voting stage; not a swept/minimized budget "
        "(descriptive #83/#86 stages excluded by registration)"
    )
    assert result["contract"] == (
        "PLATEAU (#88): zero-regression trusted-tier growth not safely "
        "extractable at any measured granularity"
    )
    # Off-domain pairs omit the field entirely (absent != 0).
    for tag, ds in (
        ("pythia2.8b", "wikitext"),
        ("pythia70m", "babi"),
        ("pythia70m", "sudoku"),
    ):
        assert v5._regression_budget_field(tag, ds) is None
    # Missing artifact: never fabricate a 0.0 rate — helper returns None.
    monkeypatch.setattr(v5, "REPO", tmp_path)
    assert v5._regression_budget_field("pythia70m", "wikitext") is None


def test_rosetta_round_trip_ignores_regression_budget(tmp_path):
    manifest = {
        "model": "synthetic",
        "cover": "fixed",
        "W": 1,
        "n_rules": 3,
        "derived": [{"id": "feat0", "kind": "recent-member", "members": [5]}],
        "rules": [
            {
                "id": 0, "kind": "gate", "frame": {}, "slot": 1,
                "table": {"5": 9}, "confs": {"5": 0.9}, "citation": ["fixture"],
            },
            {
                "id": 1, "kind": "dgate", "feature": "feat0",
                "table": {"5:7": 11}, "confs": {"5:7": 0.8}, "citation": ["fixture"],
            },
            {
                "id": 2, "kind": "ngram", "ctx": [7], "out": 12,
                "confidence": 0.7, "citation": ["fixture"],
            },
        ],
    }
    with_budget = copy.deepcopy(manifest)
    with_budget["regression_budget"] = {
        "rate": 62 / 205,
        "rate_definition": "deduplicated regressions / (deduplicated gains + regressions)",
        "eval_unit": "held-out split row",
        "measured_on": ["Stage 1 / #81 GATED1 on test_te"],
        "sweep@sha": "#81@3c5a96e + #86@9681595 (data/frames_at_scale.json)",
        "domain": "pythia70m/wikitext",
        "operating_point": (
            "single GATED1 voting stage; not a swept/minimized budget "
            "(descriptive #83/#86 stages excluded by registration)"
        ),
        "contract": (
            "PLATEAU (#88): zero-regression trusted-tier growth not safely "
            "extractable at any measured granularity"
        ),
    }

    loaded = []
    for label, payload in (("plain", manifest), ("with_regression_budget", with_budget)):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload))
        loaded.append(load_package(path))
    plain, with_regression_budget = loaded
    assert plain[0] == with_regression_budget[0]
    assert plain[1] == with_regression_budget[1]
    for context in ([5], [7], [1]):
        assert decide(context, *plain) == decide(context, *with_regression_budget)
