import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from domain_certified import GRID, N_OCC, N_VERB, decode_certified, grid_index  # noqa: E402
from domain_exhaustion import canonical_index  # noqa: E402

NOTES = Path(__file__).resolve().parents[1] / "docs" / "notes"


def test_axis_contract_round_trips_and_detects_a_swap():
    # a few triples whose (S, V, O) and (S, O, V) readings differ
    svo = np.array([[3, 7, 11], [0, 15, 39], [39, 0, 1], [12, 5, 5 + 20]])
    f = np.stack([svo[:, 0], N_OCC + svo[:, 1], svo[:, 2]], axis=1)  # the dump's filler numbering
    idx = canonical_index(f)
    assert (idx == grid_index(svo[:, 0], svo[:, 1], svo[:, 2])).all()
    g = np.zeros(GRID, dtype=bool)
    g[idx] = True
    got = decode_certified(np.packbits(g))
    assert sorted(map(tuple, got)) == sorted(map(tuple, svo))
    # reading the grid as (S, V, O) would decode different triples
    swapped = {(s, o % N_VERB, v) for s, v, o in map(tuple, got)}
    assert swapped != set(map(tuple, svo))


def test_committed_artifact_matches_the_summary():
    m = np.load(NOTES / "domain_exhaustion_certified.npz")
    summary = json.loads((NOTES / "domain_exhaustion_summary.json").read_text())
    dom = decode_certified(m["domain"])
    assert len(dom) == N_OCC * (N_OCC - 1) * N_VERB
    assert (dom[:, 0] != dom[:, 2]).all()
    domain = set(map(tuple, dom))
    for cell, c in summary["cells"].items():
        _, d, o = cell.split("/")
        for p in c["per_seed"]:
            cert = decode_certified(m[f"{d}_{o}_seed{p['seed']}"])
            assert len(cert) == len(dom) - p["uncertified_D"]
            assert set(map(tuple, cert)) <= domain
