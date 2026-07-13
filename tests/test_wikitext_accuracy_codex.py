"""Fast corpus-free checks for the Codex wikitext agreement pilot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

from beam_engine import beam_decode, det_rank_compare  # noqa: E402
from serve_package import decide, load_package  # noqa: E402
from text_oracle import TextOracle  # noqa: E402
from wikitext_accuracy_codex import agreement, delta_agreement  # noqa: E402


def _write_manifest(tmp_path: Path, name: str, cover: str, rules: list[dict], **extra):
    manifest = {
        "model": "toy",
        "cover": cover,
        "W": 1,
        "n_rules": len(rules),
        "origin": "teacher",
        "rules": rules,
        **extra,
    }
    path = tmp_path / name
    path.write_text(json.dumps(manifest))
    return load_package(path)


RANKING_RULES = [
    {
        "kind": "gate",
        "id": "higher",
        "frame": {},
        "slot": 1,
        "table": {"1": 10},
        "confs": {"1": 0.6},
        "counts": {"1": [9, 10]},
        "citation": "higher-rate branch",
    },
    {
        "kind": "ngram",
        "id": "lower",
        "ctx": [1],
        "out": 20,
        "confidence": 0.9,
        "counts": [4, 10],
        "citation": "lower-rate branch",
    },
    {
        "kind": "ngram",
        "id": "higher-next",
        "ctx": [10],
        "out": 11,
        "confidence": 0.6,
        "counts": [9, 10],
        "citation": "higher continuation",
    },
    {
        "kind": "ngram",
        "id": "lower-next",
        "ctx": [20],
        "out": 21,
        "confidence": 0.9,
        "counts": [4, 10],
        "citation": "lower continuation",
    },
]


def test_real_count_ranking_is_suppressed_without_hard_margin(tmp_path):
    assert det_rank_compare((9, 10), (4, 10)) == -1
    idioms, ngrams, manifest = _write_manifest(
        tmp_path,
        "energy.json",
        "energy-beam",
        RANKING_RULES,
        M=2,
        beam_width=8,
        schema_version=3,
        alpha=2.0,
    )

    result = decide([1], idioms, ngrams, manifest)
    corner_manifest = {**manifest, "M": 1, "beam_width": 1}
    corner = decide([1], idioms, ngrams, corner_manifest)

    assert result is not None
    assert result == corner
    assert result["answer"] == 20
    assert result["cert_kind"] == "per-token"


def _strip_cert(result):
    if result is None:
        return None
    stripped = dict(result)
    stripped.pop("cert_kind", None)
    return stripped


def test_m1_corner_is_bit_exact_to_serve_sw(tmp_path):
    idioms_sw, ngrams_sw, manifest_sw = _write_manifest(
        tmp_path, "sw.json", "support-weighted", RANKING_RULES
    )
    idioms_e, ngrams_e, manifest_e = _write_manifest(
        tmp_path,
        "corner.json",
        "energy-beam",
        RANKING_RULES,
        M=1,
        beam_width=1,
        schema_version=3,
        alpha=2.0,
    )

    for context in ([1], [10], [20], [99]):
        classic = decide(context, idioms_sw, ngrams_sw, manifest_sw)
        corner = decide(context, idioms_e, ngrams_e, manifest_e)
        assert _strip_cert(corner) == classic
        if corner is not None:
            assert corner["cert_kind"] == "per-token"


def test_text_beam_has_zero_prune_bite_when_every_path_fires(tmp_path):
    looping_rules = [
        {
            "kind": "ngram",
            "id": "loop",
            "ctx": [1],
            "out": 1,
            "confidence": 0.8,
            "counts": [8, 10],
            "citation": "loop",
        }
    ]
    idioms, ngrams, manifest = _write_manifest(
        tmp_path,
        "loop.json",
        "energy-beam",
        looping_rules,
        M=4,
        beam_width=8,
        schema_version=3,
        alpha=2.0,
    )
    oracle = TextOracle(
        [1],
        idioms,
        ngrams,
        manifest["W"],
        m_derived=manifest["_derived"],
        cmap=manifest["_cmap"],
        m_tau=manifest["_tau"],
    )

    result = beam_decode(oracle, M=4, beam_width=8, rule_id="toy", wyly_seed=0)

    assert result["prune_events"] == 0
    assert result["n_steps"] == 4
    assert result["surviving_counts"] == [1, 1, 1, 1]


def test_agreement_and_delta_helpers_count_abstention_as_wrong():
    gold = [1, 2, 3, 4]
    corner = [1, None, 9, 4]
    beam = [1, 2, None, 0]

    assert agreement(corner, gold) == 0.5
    assert agreement(beam, gold) == 0.5
    assert delta_agreement(beam, corner, gold) == 0.0
