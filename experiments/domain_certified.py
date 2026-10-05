"""Decode the D_cert bitmasks committed by domain_exhaustion.py (docs/notes/domain_exhaustion_certified.npz).

Contract (fixed by domain_exhaustion.canonical_index at the frozen commit a0c525c):
- each mask is np.packbits (default bitorder="big") of a boolean grid of length
  N_OCC * N_OCC * N_VERB = 25,600;
- grid index = S * N_OCC * N_VERB + O * N_VERB + V: an (S, O, V) grid with V fastest. It is NOT (S, V, O);
- S, O are occupation indices 0..39 (roles subject and object); V is a verb index 0..15. In the dump's filler
  numbering the verb filler is N_OCC + V;
- the "domain" mask marks the 24,960 valid cells (S != O); certified masks are subsets of it.
"""

from __future__ import annotations

import numpy as np

N_OCC, N_VERB = 40, 16
GRID = N_OCC * N_OCC * N_VERB


def grid_index(s, v, o):
    """(subject, verb, object) indices -> grid index (the (S, O, V) layout)."""
    return np.asarray(s) * N_OCC * N_VERB + np.asarray(o) * N_VERB + np.asarray(v)


def decode_certified(packed) -> np.ndarray:
    """A packed mask -> array of certified (S, V, O) triples, one row each."""
    g = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="big")[:GRID].astype(bool)
    idx = np.flatnonzero(g)
    s, rem = np.divmod(idx, N_OCC * N_VERB)
    o, v = np.divmod(rem, N_VERB)
    return np.stack([s, v, o], axis=1)
