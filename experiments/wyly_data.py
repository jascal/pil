"""Canonical loader for the Wyly-LM raw-token next-token dataset (shared by the wyly_lm* experiments).

Dataset file: ``data/wyly_nexttoken_pythia70m.pt`` (repo-local, gitignored; ~4 MB), schema
``{"kept_ids": (N, L) int64 raw pythia token windows, "target": (N,) int64 next token, "L": int}``.
Produced by ``experiments/extract_nexttoken.py`` (see its docstring for provenance). Override the
location with the ``WYLY_DATA`` env var.

``load_windows`` also owns the prep that was copy-pasted across the five wyly_lm* scripts: restrict
targets to the top-V classes (8 GB GPU), compact the input vocab, and return the TEMPORAL split —
the one protocol every comparison must share (the sorted-by-current-token split used by some earlier
scripts makes 100% of test current-tokens unseen-as-current; bigram floor 0.000 vs 0.176 temporal —
see WYLY_LM_ENDGAME_REVIEW_FABLE.md §1.1).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
CANON = REPO / "data" / "wyly_nexttoken_pythia70m.pt"


def resolve_path() -> Path:
    p = Path(os.environ.get("WYLY_DATA", CANON))
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- run experiments/extract_nexttoken.py (or set WYLY_DATA)"
        )
    return p


def load_windows(v: int = 2048, device: str | None = None):
    """-> (ids (N,L) compact-vocab windows, y (N,) class ids, vocab, tr, te) -- temporal split."""
    d = torch.load(resolve_path(), map_location="cpu")
    ids, target = d["kept_ids"], d["target"]
    keep = torch.isin(target, torch.bincount(target).argsort(descending=True)[:v])
    ids, target = ids[keep], target[keep]
    uv, ids = ids.reshape(-1).unique(return_inverse=True)
    ids = ids.reshape(target.shape[0], -1)
    y = target.unique(return_inverse=True)[1]
    if device:
        ids, y = ids.to(device), y.to(device)
    n = ids.shape[0]
    tr = torch.arange(int(0.85 * n), device=ids.device)
    te = torch.arange(int(0.85 * n), n, device=ids.device)
    return ids, y, len(uv), tr, te
