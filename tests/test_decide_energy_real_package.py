"""Structural smoke test for gated M-step text serving on the production package."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

os.environ.setdefault("WYLY_TAG", "pythia70m")
os.environ.setdefault("WYLY_LIB", "mined")
os.environ.setdefault("WYLY_JUDGE", "cover")
os.environ.setdefault("WYLY_ONLINE", "1")
os.environ.setdefault("WYLY_COVER", "sw")

import wyly_lm_v5 as v5  # noqa: E402
from serve_package import decide, load_package  # noqa: E402

PACKAGE = REPO / "data" / "wyly_expert_package_v5" / "manifest.json"


def test_m_gt1_real_package_is_gated_to_corner():
    ids, _y, _cls, uv, _tr, te = v5.load_ds()
    idioms, ngrams, manifest = load_package(PACKAGE)
    assert manifest.get("cover") == "support-weighted"

    energy_manifest = dict(manifest)
    energy_manifest.update(cover="energy-beam", M=3, beam_width=8)
    corner_manifest = dict(manifest)
    corner_manifest.update(cover="energy-beam", M=1, beam_width=1)
    vocabulary = uv.tolist()
    results = []
    for wi in te.tolist()[:200]:
        ctx = [vocabulary[c] for c in ids[wi].tolist()]
        result = decide(ctx, idioms, ngrams, energy_manifest)
        corner = decide(ctx, idioms, ngrams, corner_manifest)
        assert result == corner
        assert result is None or (
            "answer" in result
            and result.get("cert_kind") == "per-token"
        )
        results.append(result)

    assert any(result is not None for result in results)
