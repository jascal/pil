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
import re
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
    pred: Tensor | None = None      # (N,) model decision token id (the faithfulness anchor)
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
    pred = (
        torch.tensor([r["pred"] for r in recs], dtype=torch.long, device=device)
        if all("pred" in r for r in recs) else None
    )
    meta = {"path": str(path), "n": len(recs), "nb": contrib.shape[1], "k": contrib.shape[2]}
    return IncidenceBundle(
        contrib=contrib, tgt_idx=tgt_idx, cands=cands, margins=margins, ent=ent,
        pred=pred, meta=meta,
    )


def recon_fraction(bundle: IncidenceBundle) -> float:
    """Faithfulness of a pil-dump: fraction of positions where the summed incidences
    ``Σ_b contrib[b][k] = ⟨r, U_cand_k⟩`` (the bias-free logit) argmax to the model
    decision ``pred``. Must be 1.0 for a bias-free unembed — the pil-side twin of the
    fieldrun ``--pil-dump`` recon self-check (fieldrun#123): a dump that fails this is
    unfit to seed the PIL/picard loop."""
    if bundle.pred is None:
        raise ValueError("dump lacks `pred`: cannot check faithfulness")
    summed = bundle.contrib.sum(dim=1)                      # (N, K)
    # a non-finite block-sum must count as a failure (parity with fieldrun#123's NaN guard),
    # not slip through an argmax whose behaviour on NaN is undefined
    finite = torch.isfinite(summed).all(dim=1)
    argmax = summed.nan_to_num(nan=float("-inf")).argmax(dim=1, keepdim=True)
    picked = bundle.cands.gather(1, argmax).squeeze(1)
    return float(((picked == bundle.pred) & finite).float().mean())


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


@dataclass
class SourceBundle:
    """A loaded ``fieldrun --source-dump``: per decode position the RAW per-block contribution vectors
    ``d̃_b ∈ ℝ^d`` (final-norm folded, in unembed space), so ``r_x = Σ_b D[x,b]`` and ``⟨d̃_b, U_v⟩`` is
    block ``b``'s exact logit contribution to token ``v``. This is what the forge-tax certificate needs:
    it holds the ``d̃_b`` fixed and frees the decoder frame ``U'`` (the raw vectors, not their projection
    onto the model's own ``U`` that ``--pil-dump`` gives). numpy-based (the LP harness, not torch)."""

    D: np.ndarray        # (N, nb, dim)  per-block contribution vectors
    r: np.ndarray        # (N, dim)      residual r_x = Σ_b D[x,b]
    cands: np.ndarray    # (N, K)        candidate token ids; cands[:,0] = the DECODED token (argmax)
    target: np.ndarray   # (N,)          ground-truth next token id
    margin: np.ndarray   # (N,)          model decode margin (top1 - top2)
    blocks: list[str]    # (nb,)         block labels (embed, L0.attn, L0.mlp, ...)

    @property
    def N(self) -> int:
        return self.D.shape[0]

    @property
    def nb(self) -> int:
        return self.D.shape[1]

    @property
    def dim(self) -> int:
        return self.D.shape[2]


def load_source_dump(path: str | Path) -> SourceBundle:
    """Load a ``fieldrun --source-dump`` JSON-lines file into a :class:`SourceBundle`.

    Each line is ``{"pos","cur","target","tgt_idx","pred","margin","dim","cands","blocks","d"}`` with
    ``d`` an ``nb x dim`` matrix of the per-block contribution vectors. Positions with a differing block
    count or candidate count are dropped (kept rectangular)."""
    path = Path(path)
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        raise ValueError(f"{path}: no records")
    nb = len(recs[0]["d"])
    k = len(recs[0]["cands"])
    recs = [r for r in recs if len(r["d"]) == nb and len(r["cands"]) == k]
    D = np.array([r["d"] for r in recs], dtype=np.float32)            # (N, nb, dim)
    cands = np.array([r["cands"] for r in recs], dtype=np.int64)       # (N, K)
    target = np.array([r["target"] for r in recs], dtype=np.int64)
    margin = np.array([r["margin"] for r in recs], dtype=np.float32)
    blocks = list(recs[0]["blocks"])
    return SourceBundle(D=D, r=D.sum(axis=1), cands=cands, target=target, margin=margin, blocks=blocks)


# ── J-lens correction: read a per-block DLA vector through the layer's averaged causal Jacobian ──────
# Where the plain DLA/logit-lens incidence ``c_b^v = ⟨d̃_b, U_v⟩`` scores block b as if it reached the output
# UNCHANGED (the identity-downstream assumption), it routes b through ``J[l] = E[∂h_final/∂h_l]`` first,
# so it scores b's TOTAL (direct + downstream) linearised effect. J is the averaged Jacobian
# fieldrun fits and exports (PR #124, ``--jlens-export``; Anthropic "Verbalizable Representations as Global
# Workspace"). This is the pil-side consumer of that export — the "J-corrected incidences" fieldrun→pil seam.
#
# Tag: ``empirical`` throughout. J is a first-order approximation, never a certificate; whether the correction
# sharpens pil's mid-stack read is decided by a λ-sweep against the λ=0 (plain logit-lens) baseline, and stays
# ``open`` until that sweep runs on real dumps.

_JLENS_BLOCK = re.compile(r"^L(\d+)\.(?:attn|mlp)$")


def load_jlens(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a ``fieldrun --jlens-export`` output (PR #124) — the per-layer averaged causal Jacobian the J-lens
    routes a mid-stack residual through before unembed, instead of the logit-lens's ``J[l] = I``.

    Returns ``(J, fitted, meta)``:
        J      : float32 (n_layer, dim, dim)  averaged Jacobian per layer (row-major); ``J[n_layer-1] = I``
        fitted : bool    (n_layer,)           True where a Jacobian was fit; False ⇒ identity (logit-lens)
        meta   : dict                         the ``.meta.json`` sidecar (``capture_point``, ``apply``, …)

    ``meta['capture_point']`` pins which tensor ``J[l]`` maps FROM: ``h_l`` = POST-block residual of layer l
    (after the attn+MLP add, PRE final-norm). That fixes the block→layer map in :func:`jcorrect_sources`."""
    path = Path(path)
    data = np.load(path)
    if "J" not in data or "fitted" not in data:
        raise KeyError(f"{path} missing 'J'/'fitted'; got {list(data.keys())} — not a --jlens-export .npz?")
    J = np.asarray(data["J"], dtype=np.float32)
    fitted = np.asarray(data["fitted"]).astype(bool)
    s = str(path)                                            # sidecar: model.npz → model.meta.json (fieldrun)
    meta_path = Path(s[:-4] + ".meta.json") if s.endswith(".npz") else Path(s + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}   # {} if no sidecar
    return J, fitted, meta


def _block_layer(label: str, n_layer: int) -> int | None:
    """A ``--source-dump`` block label → the ``J[l]`` whose capture point (``h_l`` = post-block residual of
    layer l) the block's contribution lands at, or ``None`` to leave it as the identity (logit-lens) read.

    Per ``capture_point`` (``h_l`` = POST-block residual of layer l, after the attn+MLP add):
      ``L{l}.mlp``  → ``l``   (EXACT: the MLP output is the last add that forms ``h_l``)
      ``L{l}.attn`` → ``l``   (nearest downstream captured point; drops only the single attn→mlp_l hop)
      ``embed`` / anything else → ``None`` (pre-layer-0 has no captured J ⇒ identity)."""
    m = _JLENS_BLOCK.match(label)
    if not m:
        return None
    layer = int(m.group(1))                          # int() tolerates any decimal form (e.g. "L00")
    return layer if 0 <= layer < n_layer else None


def jcorrect_sources(
    D: np.ndarray,
    blocks: list[str],
    J: np.ndarray,
    fitted: np.ndarray,
    lam: float = 1.0,
    gamma: np.ndarray | None = None,
    layernorm: bool = False,
) -> np.ndarray:
    """J-lens correction of a :class:`SourceBundle`'s per-block vectors: route each contribution through
    its layer's averaged causal Jacobian so ``c_b^v = ⟨J[l(b)] d̃_b, U_v⟩`` measures block b's TOTAL (direct +
    downstream) linearised effect on token v, not just its direct-logit-attribution.

    ``D``      : (N, nb, dim) — ``SourceBundle.D`` (per-block contribution vectors ``d̃_b``)
    ``blocks`` : the (nb,) labels — ``SourceBundle.blocks`` (``embed, L0.attn, L0.mlp, …``)
    ``J``, ``fitted`` : from :func:`load_jlens`
    ``lam``    : shrinkage toward identity, ``J' = (1-λ)I + λJ`` (fieldrun ``--jlens-shrink``). The raw fit
                 (λ=1) is noise-dominated — fieldrun's eval found **λ≈0.25–0.5** best; **λ=0 reproduces the
                 plain logit-lens incidences exactly** and is the baseline any correction must beat. SWEEP it.
    ``gamma``  : optional (dim,) final-norm gain ``γ``. ``SourceBundle.D`` is *final-norm folded* (post-norm,
                 unembed space) but ``J`` is fit in the *pre-norm* basis, so the folded-basis operator is
                 ``diag(γ) J diag(1/γ)`` (the per-position rms scale cancels; only ``γ`` remains). ``None``
                 applies ``J`` directly (exact up to that conjugation).
    ``layernorm``: LayerNorm final norm ⇒ insert the mean-centering ``P = I - 11ᵀ/d``; the exact operator is
                 ``diag(γ) P J diag(1/γ)`` (ln_f bias + inv_std are logit-inert). RMSNorm ⇒ ``False``.

    Returns a corrected copy of ``D`` (same shape); feed to :func:`pil.geometry.incidences`. Blocks with no
    mapped or fitted layer are left UNCHANGED (identity ⇒ plain logit-lens), never scrambled.

    Usage::

        sb = load_source_dump(path)                 # sb.D (N,nb,dim), sb.blocks
        J, fitted, meta = load_jlens(npz_path)      # meta['capture_point'] pins the block→layer map
        Dc = jcorrect_sources(sb.D, sb.blocks, J, fitted, lam=0.3)   # sweep lam vs the lam=0 baseline
        inc = geometry.incidences(torch.from_numpy(Dc), U)          # (N, nb, V) J-corrected incidences
    """
    if J.ndim != 3 or J.shape[1] != J.shape[2]:
        raise ValueError(f"J must be (n_layer, dim, dim); got {J.shape}")
    n_layer, dim, _ = J.shape
    if D.shape[-1] != dim:
        raise ValueError(f"D dim {D.shape[-1]} != J dim {dim} (different bundle/model?)")
    if len(blocks) != D.shape[1]:
        raise ValueError(f"blocks ({len(blocks)}) != D nb ({D.shape[1]})")
    eye = np.eye(dim, dtype=np.float32)
    g = None
    if gamma is not None:
        g = np.asarray(gamma, dtype=np.float32).reshape(-1)
        if g.shape[0] != dim:
            raise ValueError(f"gamma dim {g.shape[0]} != {dim}")
    out = D.astype(np.float32, copy=True)
    for b, label in enumerate(blocks):
        layer = _block_layer(label, n_layer)
        if layer is None or not bool(fitted[layer]):
            continue                                         # unmapped / unfit → identity (logit-lens)
        jl = J[layer]
        if g is not None:                                    # folded-basis operator: diag(γ) [P] J diag(1/γ)
            jl = jl * (1.0 / g)[None, :]                      # J diag(1/γ)
            if layernorm:                                    # P: mean-center rows (LayerNorm fold)
                jl = jl - jl.mean(axis=0, keepdims=True)
            jl = g[:, None] * jl                             # diag(γ) [P] J diag(1/γ)
        jl = (1.0 - lam) * eye + lam * jl                    # shrink toward identity
        out[:, b, :] = D[:, b, :] @ jl.T                     # fieldrun convention: J @ d  ==  d @ J.T
    return out
