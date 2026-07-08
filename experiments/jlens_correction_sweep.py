"""lambda-sweep for the J-lens DLA-incidence correction (the fieldrun->pil J-lens seam, Step 1).

Consumes a ``fieldrun --source-dump`` (raw per-block d_b + block labels) and a ``fieldrun --jlens-export``
{J_l} (PR #124), and sweeps the shrinkage ``lam`` in ``jcorrect_sources``. For each lam it summarises the
summed J-corrected read ``sum_b J[l(b)] d_b`` (candidate-restricted: it indexes the supplied model
unembedding ``U`` by the dump's ``cands``) against the lam=0 plain-logit-lens baseline:

  * recon    fraction where the summed read argmaxes to the model decode ``cands[:,0]``. lam=0 => ~1.0
             (the sanity invariant; a drop at lam>0 is EXPECTED -- J-correction need not preserve recon).
  * margin   mean margin of the decoded token vs its strongest competitor (higher => cleaner read).
  * resolve  mean fraction-of-depth at which the cumulative-by-layer read first locks to ``cands[:,0]``
             (LOWER => the paper's "resolves earlier" claim -- the metric the J-lens should improve).

The J-lens wins iff some lam>0 (fieldrun's eval found ~0.25-0.5) lowers ``resolve`` or raises ``margin``.
Tag: ``empirical`` -- J is a probe, not a certificate. Needs a REAL pair (a ``--source-dump`` + a
``--jlens`` for the same model) and the model unembedding ``U`` ((V,dim) .npy/.npz); absent those the seam
is exercised only by ``tests/test_jlens_sweep.py`` on synthetic data. Usage::

    python experiments/jlens_correction_sweep.py run.source.jsonl --jlens m.npz --U U.npy --lams 0,0.25,0.5,1
"""
from __future__ import annotations

import argparse

import numpy as np

from pil.fieldrun_io import _block_layer, jcorrect_sources, load_jlens, load_source_dump


def _load_matrix(path: str, key: str = "U") -> np.ndarray:
    """Load a dense array from ``.npy`` or ``.npz`` (``key`` else the first array), as float32."""
    a = np.load(path)
    if hasattr(a, "files"):                          # an .npz archive
        a = a[key if key in a.files else a.files[0]]
    return np.asarray(a, dtype=np.float32)


def _block_depths(blocks: list[str], n_layer: int) -> np.ndarray:
    """Per-block inclusion depth for the cumulative read: ``L{l}.*`` -> l; ``embed``/unmapped -> -1 (always
    kept, so the full read reconstructs the decode). In the layer-``l`` partial iff ``depth <= l``."""
    d = np.empty(len(blocks), dtype=np.int64)
    for i, lbl in enumerate(blocks):
        layer = _block_layer(lbl, n_layer)
        d[i] = layer if layer is not None else -1
    return d


def _metrics(dc, cands, U, depths, n_layer):
    """(recon, mean-margin, mean-resolve) for corrected D ``dc`` (N,nb,dim), candidate-restricted."""
    n = dc.shape[0]
    hit = np.zeros(n, dtype=bool)
    margin = np.zeros(n, dtype=np.float32)
    denom = float(max(n_layer - 1, 1))
    resolve = np.full(n, denom, dtype=np.float32)
    for i in range(n):
        uc = U[cands[i]]                             # (K, dim) candidate rows of the model unembedding
        full = dc[i].sum(0) @ uc.T                   # (K,) summed corrected read over the candidate set
        hit[i] = int(np.argmax(full)) == 0           # cands[:,0] is the model decode -> index 0
        margin[i] = full[0] - np.max(np.delete(full, 0))  # decoded vs strongest competitor
        for layer in range(n_layer):                 # first depth whose cumulative read locks to the decode
            if int(np.argmax(dc[i, depths <= layer].sum(0) @ uc.T)) == 0:
                resolve[i] = layer
                break
    return float(hit.mean()), float(margin.mean()), float((resolve / denom).mean())


def sweep(sb, J, fitted, U, lams, gamma=None):
    """Run the sweep. Returns ``{lam, recon, margin, resolve}`` dicts (pass lam=0 first = the baseline)."""
    n_layer = J.shape[0]
    depths = _block_depths(sb.blocks, n_layer)
    rows = []
    for lam in lams:
        dc = jcorrect_sources(sb.D, sb.blocks, J, fitted, lam=lam, gamma=gamma)
        recon, margin, resolve = _metrics(dc, sb.cands, U, depths, n_layer)
        rows.append({"lam": float(lam), "recon": recon, "margin": margin, "resolve": resolve})
    return rows


def _print_table(rows):
    base = rows[0]                                   # first lam (pass 0.0 first) is the baseline
    print(f"{'lam':>6} {'recon':>8} {'margin':>10} {'resolve':>9}  dmargin dresolve")
    for r in rows:
        dm = r["margin"] - base["margin"]
        dr = r["resolve"] - base["resolve"]          # negative dresolve = resolves EARLIER (the win)
        print(f"{r['lam']:>6.2f} {r['recon']:>8.3f} {r['margin']:>10.3f} {r['resolve']:>9.3f}"
              f"  {dm:>+7.3f} {dr:>+8.3f}")
    print("\n  recon(lam=0)~1.0 sanity check; a win = lam>0 lowers resolve or raises margin (vs row 0).")


def main():
    ap = argparse.ArgumentParser(description="lambda-sweep of the J-lens DLA correction (vs lam=0)")
    ap.add_argument("source_dump", help="a fieldrun --source-dump JSON-lines file (SourceBundle)")
    ap.add_argument("--jlens", required=True, help="a fieldrun --jlens-export .npz (J, fitted)")
    ap.add_argument("--U", required=True, help="the model unembedding (V, dim) as .npy or .npz (key 'U')")
    ap.add_argument("--gamma", help="optional final-norm gain (dim,) .npy/.npz for the exact folded operator")
    ap.add_argument("--lams", default="0,0.25,0.5,1.0", help="comma-separated lambdas (0 first)")
    a = ap.parse_args()

    sb = load_source_dump(a.source_dump)
    J, fitted, meta = load_jlens(a.jlens)
    U = _load_matrix(a.U, key="U")                       # fieldrun --tensors-export keys: U, gamma
    gamma = _load_matrix(a.gamma, key="gamma").reshape(-1) if a.gamma else None
    lams = [float(x) for x in a.lams.split(",")]
    if U.shape[1] != sb.dim or J.shape[1] != sb.dim:
        raise SystemExit(f"dim mismatch: source-dump {sb.dim}, U {U.shape[1]}, J {J.shape[1]}")

    nfit = int(fitted.sum())
    print(f"[jlens-sweep] {sb.N} positions, nb={sb.nb}, dim={sb.dim}, {nfit}/{J.shape[0]} layers fitted"
          f"{' (gamma-conj)' if gamma is not None else ''}")
    if meta.get("capture_point"):
        print(f"[jlens-sweep] capture_point: {meta['capture_point']}")
    _print_table(sweep(sb, J, fitted, U, lams, gamma=gamma))


if __name__ == "__main__":
    main()
