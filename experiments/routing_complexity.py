"""Operationalize Tropical Realization Complexity: rules needed vs routing decisions (cell capacity fixed).

Cell capacity (d, V, γ) is held fixed (and slack by ~1e59, per capacity_diagnostic). We vary the
NUMBER of independent non-linear routing decisions -- n_hard XOR-coded clusters -- and measure how
many trainable rules M are needed to solve them. If realization complexity is the binding axis, hard-
cluster accuracy should be governed by M / n_hard (rules per routing decision), NOT by capacity, and
the acc-vs-M curves for different n_hard should COLLAPSE when plotted against M / n_hard.

This grounds a definition of TRC empirically before formalizing it: TRC ≈ (rules per routing decision)
needed to reach a target margin, capacity-independent.

Run:  python experiments/routing_complexity.py --seeds 2
"""

from __future__ import annotations

import argparse
import statistics as st

import torch

from pil.learner import PILConfig, ProjectiveIncidenceLearner
from pil.proposer import RuleBank, build_sources
from pil.synthetic import create_compositional_problem, within_cluster_report


def split(z, tgt, seed):
    g = torch.Generator(device="cpu").manual_seed(seed + 99)
    perm = torch.randperm(len(tgt), generator=g).to(z.device)
    n_tr = int(0.75 * len(tgt))
    return perm[:n_tr], perm[n_tr:]


def run(n_hard, M, dim, V, K, n_clusters, steps, batch, seed, device):
    cfg = PILConfig(dim=dim, n_propositions=V, n_sources_per_step=1, lr=5e-3, frame_reg=0.0,
                    margin_weight=0.5, margin_target=1.8, seed=seed, device=device)
    z, A, tgt, _, hc = create_compositional_problem(
        cfg, n_examples=2048, n_clusters=n_clusters, n_hard=n_hard
    )
    tr, ev = split(z, tgt, seed)
    model = ProjectiveIncidenceLearner(cfg)
    bank = RuleBank(K, dim, M, device=device, seed=seed) if M > 0 else None
    params = list(model.parameters()) + (list(bank.parameters()) if bank else [])
    opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed + 11)
    for _ in range(steps):
        idx = tr[torch.randint(0, len(tr), (batch,), generator=g)]
        out = model.forward(build_sources(z[idx], A, bank), target=tgt[idx])
        opt.zero_grad()
        out["total_loss"].backward()
        opt.step()
        model.normalize_U()
    L = model.forward(build_sources(z[ev], A, bank), target=None)["logits"]
    return within_cluster_report(L, tgt[ev], hc)["hard_within_acc"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--clusters", type=int, default=24)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--n-hard", type=int, nargs="+", default=[4, 8, 16, 24])
    p.add_argument("--budget", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V, K = 2 * a.clusters, 3 * a.clusters

    print(f"[routing] dim={a.dim} V={V} (cell capacity FIXED, slack ~1e59); "
          f"vary n_hard routing decisions x rule budget M; held-out, seeds={a.seeds}")

    def cell(nh, M):
        return st.mean(run(nh, M, a.dim, V, K, a.clusters, a.steps, a.batch, s, device)
                       for s in range(a.seeds))

    grid = {(nh, M): cell(nh, M) for nh in a.n_hard for M in a.budget}

    print("\nhard-cluster within-acc   (rows = n_hard routing decisions, cols = rule budget M)")
    print("n_hard\\M".rjust(9) + "".join(f"{M:>8}" for M in a.budget))
    for nh in a.n_hard:
        print(f"{nh:>9}" + "".join(f"{grid[(nh, M)]:>8.3f}" for M in a.budget))

    # threshold: smallest M reaching acc >= 0.85, per n_hard -> the realization cost
    print("\nminimal M to reach acc>=0.85, and M/n_hard (rules per routing decision):")
    for nh in a.n_hard:
        hit = [M for M in a.budget if grid[(nh, M)] >= 0.85]
        if hit:
            print(f"  n_hard={nh:>2}:  M*={hit[0]:>3}   M*/n_hard={hit[0]/nh:.2f}")
        else:
            print(f"  n_hard={nh:>2}:  M*=>{a.budget[-1]} (not reached)")

    # collapse check: acc as a function of M/n_hard should be roughly n_hard-invariant
    print("\ncollapse check -- acc at matched M/n_hard ratio ~1, ~2, ~4:")
    for ratio in (1.0, 2.0, 4.0):
        vals = []
        for nh in a.n_hard:
            M = min(a.budget, key=lambda M, nh=nh, ratio=ratio: abs(M / nh - ratio))
            vals.append(grid[(nh, M)])
        spread = max(vals) - min(vals)
        print(f"  M/n_hard~{ratio:>3.0f}:  accs={[round(v, 2) for v in vals]}  spread={spread:.2f}")


if __name__ == "__main__":
    main()
