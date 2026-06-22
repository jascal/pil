"""Bridge: load real fieldrun probe dumps as PIL sources/frames.

PIL's generative step can be seeded from the *real* partial-evidence vectors a
transformer produces -- the per-position direct-logit-attribution (DLA) vectors
``d_j`` that fieldrun's ``--probe-ablate`` program already computes -- instead of
synthetic Gaussians. This module defines the **integration contract** between the
two repos.

Status (2026-06): this is the *expected schema*, not a claim that fieldrun emits it
verbatim today. fieldrun has the quantities (residual DLAs, unembedding rows ``U_v``,
per-position margins/PR; see ``PIC_PROPOSAL.md`` §6.3's proposed ``--probe-reconstruct``
mode); wiring a ``--pil-dump`` emitter that writes this schema is the fieldrun-side task
(PIC_PROPOSAL O4). Keeping the contract here, in one place, makes that wiring a
mechanical match rather than a guess.

Expected dump: a single ``.npz`` (or a directory of them) with arrays

    sources : float32 (N, J, dim)   per-position partial-evidence vectors d_j
    targets : int64   (N,)          the model's emitted (or gold) token id per position
    U       : float32 (V, dim)      unembedding rows / proposition frame (optional)
    bias    : float32 (V,)          final logit bias (optional)
    meta    : json string           {model, layer, store, n_positions, dla_circuits, ...}

Per-position ``J`` may vary; if a ragged dump is provided as object arrays, pad to
``max J`` with zero vectors (zeros are inert under ``c_j^v = <d_j, U_v>``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass
class IncidenceBundle:
    """A loaded `fieldrun --pil-dump` (real-DLA seam): per natural-text position, the per-block DLA
    incidence matrix ``contrib[block][cand] = <d_block, U_cand>`` to the top-K candidates, with the
    ground-truth target and margin. The real-model analogue of the synthetic compositional sources —
    each DLA block is a "rule", its per-candidate contribution the incidence."""

    contrib: Tensor   # (N, nb, K)  per-block incidence to each candidate
    tgt_idx: Tensor   # (N,)        index of the target token within cands
    cands: Tensor     # (N, K)      candidate token ids
    margins: Tensor   # (N,)        model decode margin (top1 - top2)
    ent: Tensor       # (N,)        full-vocab next-token entropy (nats); exp(ent) = effective output support
    meta: dict | None = None

    @property
    def n_blocks(self) -> int:
        return int(self.contrib.shape[1])


def load_pil_dump(
    path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> IncidenceBundle:
    """Load a `fieldrun --recursion-explain --pil-dump` JSON-lines file into an :class:`IncidenceBundle`.

    Each line is ``{"pos","cur","target","tgt_idx","pred","margin","ent","nb","cands","contrib"}`` with
    ``contrib`` an ``nb x K`` matrix. Rows with a missing target (``tgt_idx == -1``) are dropped.
    ``ent`` is optional (older dumps lack it) and defaults to NaN when absent.
    """
    path = Path(path)
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("tgt_idx", -1) >= 0:
                recs.append(r)
    if not recs:
        raise ValueError(f"{path}: no usable records (all targets out of the candidate set?)")

    contrib = torch.tensor([r["contrib"] for r in recs], dtype=dtype, device=device)  # (N, nb, K)
    tgt_idx = torch.tensor([r["tgt_idx"] for r in recs], dtype=torch.long, device=device)
    cands = torch.tensor([r["cands"] for r in recs], dtype=torch.long, device=device)
    margins = torch.tensor([r["margin"] for r in recs], dtype=dtype, device=device)
    ent = torch.tensor([r.get("ent", float("nan")) for r in recs], dtype=dtype, device=device)
    meta = {"path": str(path), "n": len(recs), "nb": contrib.shape[1], "k": contrib.shape[2]}
    return IncidenceBundle(contrib=contrib, tgt_idx=tgt_idx, cands=cands, margins=margins, ent=ent, meta=meta)


@dataclass
class ProbeBundle:
    """A loaded fieldrun probe dump, ready to feed :class:`ProjectiveIncidenceLearner`."""

    sources: Tensor                 # (N, J, dim)
    targets: Tensor                 # (N,)
    U: Tensor | None = None         # (V, dim) observed frame, if dumped
    bias: Tensor | None = None      # (V,)
    meta: dict | None = None

    @property
    def dim(self) -> int:
        return int(self.sources.shape[-1])

    @property
    def n_positions(self) -> int:
        return int(self.sources.shape[0])


def _pad_ragged(seqs: list[np.ndarray], dim: int) -> np.ndarray:
    """Pad a list of (J_i, dim) arrays to (N, maxJ, dim) with inert zero vectors."""
    maxj = max(s.shape[0] for s in seqs)
    out = np.zeros((len(seqs), maxj, dim), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out


def load_probe_sources(
    path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ProbeBundle:
    """Load a fieldrun ``--pil-dump`` ``.npz`` into a :class:`ProbeBundle`.

    Raises a clear error (not a silent shape bug) if the required ``sources``/``targets``
    arrays are missing, so a contract mismatch with the fieldrun emitter surfaces loudly.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    if "sources" not in data or "targets" not in data:
        raise KeyError(
            f"{path} missing required arrays; got {list(data.keys())}. "
            "Expected at least 'sources' (N,J,dim) and 'targets' (N,). "
            "See pil.fieldrun_io module docstring for the contract."
        )

    raw = data["sources"]
    if raw.dtype == object:  # ragged per-position J
        dim = int(raw[0].shape[-1])
        sources_np = _pad_ragged(list(raw), dim)
    else:
        sources_np = np.asarray(raw, dtype=np.float32)
    if sources_np.ndim != 3:
        raise ValueError(f"sources must be (N,J,dim); got {sources_np.shape}")

    sources = torch.as_tensor(sources_np, dtype=dtype, device=device)
    targets = torch.as_tensor(np.asarray(data["targets"]), dtype=torch.long, device=device)

    U = (
        torch.as_tensor(np.asarray(data["U"], dtype=np.float32), dtype=dtype, device=device)
        if "U" in data
        else None
    )
    bias = (
        torch.as_tensor(np.asarray(data["bias"], dtype=np.float32), dtype=dtype, device=device)
        if "bias" in data
        else None
    )
    meta = None
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"]))
        except (json.JSONDecodeError, TypeError):
            meta = {"raw": str(data["meta"])}

    return ProbeBundle(sources=sources, targets=targets, U=U, bias=bias, meta=meta)
