"""Hard synthetic sweep: does the frame-side term help when the Gram is dense?

Over-complete (V >> dim) + planted synonym clusters (high within-cluster rho). Sweeps
`frame_reg` x seeds and reports the decode/frame trade:

  retr%   top1   margin   nll    fp/welch   wc_cos(learned)   align(U,U_gt)   syn_comp%

Hypothesis: unlike the easy regime (flat in frame_reg), retrievability here is NON-monotone
in frame_reg -- decorrelating separates synonyms (margin up) until it pulls the frame too far
off the planted source directions (alignment down). Report the interior optimum if present.

Run:  python experiments/hard_synthetic.py --steps 1500 --seeds 3
"""

from __future__ import annotations

import argparse
import statistics as st

import torch

from pil.geometry import frame_potential, margin_to_worst, welch_bound
from pil.learner import PILConfig, ProjectiveIncidenceLearner
from pil.synthetic import create_clustered_problem, frame_alignment, within_cluster_cosine


def _progress(it, desc):
    try:
        from tqdm import tqdm

        return tqdm(it, desc=desc, leave=False)
    except ImportError:
        return it


@torch.no_grad()
def evaluate(model, sources, targets, U_gt, cluster_of, margin_threshold):
    model.eval()
    out = model.forward(sources, target=targets)
    L = out["logits"]
    margins = margin_to_worst(L, targets)
    retr = (margins > margin_threshold).float().mean().item()
    top1 = (L.argmax(-1) == targets).float().mean().item()

    # is the strongest competitor a synonym (same cluster)?
    masked = L.scatter(-1, targets.view(-1, 1), float("-inf"))
    comp = masked.argmax(-1)
    syn_comp = (cluster_of[comp] == cluster_of[targets]).float().mean().item()

    fp = frame_potential(model.U).item()
    wb = welch_bound(model.n_prop, model.dim)
    return {
        "retr": retr,
        "top1": top1,
        "margin": margins.mean().item(),
        "nll": out["nll"].item(),
        "fp_welch": fp / wb if wb > 0 else float("nan"),
        "wc_cos": within_cluster_cosine(model.U, cluster_of),
        "align": frame_alignment(model.U, U_gt),
        "syn_comp": syn_comp,
    }


def train_once(frame_reg, seed, dim, V, J, n_clusters, spread, steps, batch, n_examples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PILConfig(
        dim=dim, n_propositions=V, n_sources_per_step=J,
        lr=5e-3, frame_reg=frame_reg, margin_weight=0.5, margin_target=1.8,
        seed=seed, device=device,
    )
    model = ProjectiveIncidenceLearner(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    src, tgt, U_gt, cl = create_clustered_problem(
        cfg, n_examples=n_examples, n_clusters=n_clusters, cluster_spread=spread,
        sparsity=0.12, noise=0.08,
    )
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    for _ in _progress(range(steps), f"fr={frame_reg} s={seed}"):
        idx = torch.randint(0, len(tgt), (batch,), generator=g).to(device)
        model.training_step(src[idx], tgt[idx], opt)
    return evaluate(model, src, tgt, U_gt, cl, margin_threshold=cfg.margin_target * 0.8), (U_gt, cl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=24)
    p.add_argument("--V", type=int, default=192)          # 8x over-complete
    p.add_argument("--J", type=int, default=24)
    p.add_argument("--clusters", type=int, default=24)    # ~8 synonyms per cluster
    p.add_argument("--spread", type=float, default=0.10)  # within-cluster cos ~ 1/(1+spread^2*dim)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n-examples", type=int, default=512)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--grid", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.5])
    a = p.parse_args()

    # planted hardness, reported once
    cfg0 = PILConfig(dim=a.dim, n_propositions=a.V, n_sources_per_step=a.J, seed=0,
                     device="cuda" if torch.cuda.is_available() else "cpu")
    _, _, U_gt0, cl0 = create_clustered_problem(cfg0, n_examples=8, n_clusters=a.clusters,
                                                cluster_spread=a.spread)
    print(f"[hard] dim={a.dim} V={a.V} ({a.V/a.dim:.0f}x over-complete) clusters={a.clusters} "
          f"spread={a.spread}")
    print(f"[hard] planted within-cluster |cos|={within_cluster_cosine(U_gt0, cl0):.3f}  "
          f"welch_floor={welch_bound(a.V, a.dim):.4f}  seeds={a.seeds}")

    cols = ["retr", "top1", "margin", "nll", "fp_welch", "wc_cos", "align", "syn_comp"]
    print("\n" + "frame_reg".rjust(9) + "".join(c.rjust(10) for c in cols))
    rows = {}
    for fr in a.grid:
        runs = [train_once(fr, s, a.dim, a.V, a.J, a.clusters, a.spread,
                           a.steps, a.batch, a.n_examples)[0] for s in range(a.seeds)]
        agg = {c: st.mean(r[c] for r in runs) for c in cols}
        rows[fr] = agg
        line = f"{fr:>9.2f}" + "".join(f"{agg[c]:>10.3f}" for c in cols)
        print(line)

    best = max(rows, key=lambda fr: rows[fr]["retr"])
    base = rows[a.grid[0]]["retr"]
    delta = rows[best]["retr"] - base
    # an "interior optimum" only counts if it is both non-endpoint AND beats the baseline by
    # more than the seed-to-seed noise floor (~0.01); otherwise the decode is flat in frame_reg.
    NOISE = 0.02
    interior = best not in (a.grid[0], a.grid[-1]) and delta > NOISE
    print(f"\n[hard] best retr at frame_reg={best} "
          f"(retr {rows[best]['retr']:.3f} vs frame_reg=0 {base:.3f}; Δ={delta:+.3f})  "
          f"interior_optimum={'YES' if interior else f'no (Δ<{NOISE} = decode flat in frame_reg)'}")


if __name__ == "__main__":
    main()
