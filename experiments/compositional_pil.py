"""Compositional PIL: does GENERATION beat frame-tuning, and does TARGETING beat spraying?

Tests the prediction the hard-synthetic negative result set up. Synonyms collide linearly but
hard clusters are XOR-coded (non-linearly separable). Arms, matched on total optimizer steps:

  frame       no generated rules (linear readout)            -> must fail the XOR (baseline)
  untargeted  M random ReLU rules, trained jointly           -> generation, sprayed
  targeted    M rules seeded at the min-margin clusters'      -> generation, aimed at the
              code-atoms (discovered from a frame-only warmup,    tropically at-risk facets
              NOT the planted flags)

Headline metric: hard-cluster within-synonym accuracy (the XOR a linear readout cannot do).
Control: signal=False (hard clusters carry no recoverable parity) -> every arm must fail.

Run:  python experiments/compositional_pil.py --steps 1500 --seeds 2
"""

from __future__ import annotations

import argparse
import statistics as st

import torch

from pil.learner import PILConfig, ProjectiveIncidenceLearner
from pil.proposer import RuleBank, build_sources, targeted_rulebank
from pil.synthetic import create_compositional_problem, within_cluster_report


def per_cluster_margin(L, targets, n_clusters):
    sib = targets ^ 1
    m = L.gather(1, targets[:, None]).squeeze(1) - L.gather(1, sib[:, None]).squeeze(1)
    c = targets // 2
    sums = torch.zeros(n_clusters, device=L.device).scatter_add(0, c, m)
    cnts = torch.zeros(n_clusters, device=L.device).scatter_add(0, c, torch.ones_like(m))
    return sums / cnts.clamp_min(1)


def make_model(dim, V, seed, device):
    cfg = PILConfig(dim=dim, n_propositions=V, n_sources_per_step=1, lr=5e-3,
                    frame_reg=0.0, margin_weight=0.5, margin_target=1.8, seed=seed, device=device)
    return ProjectiveIncidenceLearner(cfg), cfg


def train(model, bank, z, A, targets, steps, batch, seed, device):
    params = list(model.parameters()) + (list(bank.parameters()) if bank is not None else [])
    opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed + 11)
    for _ in range(steps):
        idx = torch.randint(0, len(targets), (batch,), generator=g).to(device)
        out = model.forward(build_sources(z[idx], A, bank), target=targets[idx])
        opt.zero_grad()
        out["total_loss"].backward()
        opt.step()
        model.normalize_U()


@torch.no_grad()
def evaluate(model, bank, z, A, targets, hard_clusters):
    L = model.forward(build_sources(z, A, bank), target=None)["logits"]
    rep = within_cluster_report(L, targets, hard_clusters)
    rep["top1"] = (L.argmax(-1) == targets).float().mean().item()
    rep["cluster_acc"] = ((L.argmax(-1) // 2) == (targets // 2)).float().mean().item()
    return rep


def run_arm(arm, M, z_tr, t_tr, z_ev, t_ev, A, hard_clusters,
            dim, V, n_clusters, K, steps, batch, seed, device):
    model, _ = make_model(dim, V, seed, device)
    warm = 0
    if arm == "frame":
        bank = None
    elif arm == "untargeted":
        bank = RuleBank(K, dim, M, device=device, seed=seed)
    elif arm == "targeted":
        warm = min(300, steps // 3)
        train(model, None, z_tr, A, t_tr, warm, batch, seed, device)          # frame-only warmup
        L = model.forward(build_sources(z_tr, A, None), target=None)["logits"]  # diagnose on TRAIN
        k = max(M // 2, 1)
        worst = torch.topk(-per_cluster_margin(L, t_tr, n_clusters), k).indices.tolist()
        bank = targeted_rulebank(worst, K, n_clusters, dim, rules_per_cluster=2,
                                 device=device, seed=seed)
    else:
        raise ValueError(arm)
    train(model, bank, z_tr, A, t_tr, steps - warm, batch, seed, device)
    return evaluate(model, bank, z_ev, A, t_ev, hard_clusters)              # eval on HELD-OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--clusters", type=int, default=24)
    p.add_argument("--frac-hard", type=float, default=0.33)
    p.add_argument("--n-examples", type=int, default=2048)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--budget", type=int, nargs="+", default=[4, 8, 16, 32])
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V, K = 2 * a.clusters, 3 * a.clusters

    def build(signal, seed):
        cfg = PILConfig(dim=a.dim, n_propositions=V, seed=seed, device=device)
        return create_compositional_problem(cfg, n_examples=a.n_examples, n_clusters=a.clusters,
                                            frac_hard=a.frac_hard, signal=signal)

    z0, *_rest, hc0 = build(True, 0)
    print(f"[comp] dim={a.dim} clusters={a.clusters} V={V} K={K} "
          f"hard={int(hc0.sum())}/{a.clusters} (XOR-coded)  seeds={a.seeds}  "
          f"eval=held-out 25%")

    def split(z, tgt, seed):
        g = torch.Generator(device="cpu").manual_seed(seed + 99)
        perm = torch.randperm(len(tgt), generator=g).to(z.device)
        n_tr = int(0.75 * len(tgt))
        tr, ev = perm[:n_tr], perm[n_tr:]
        return z[tr], tgt[tr], z[ev], tgt[ev]

    def arm_mean(arm, M, signal):
        runs = []
        for s in range(a.seeds):
            z, A, tgt, _, hc = build(signal, s)
            z_tr, t_tr, z_ev, t_ev = split(z, tgt, s)
            runs.append(run_arm(arm, M, z_tr, t_tr, z_ev, t_ev, A, hc, a.dim, V, a.clusters, K,
                                a.steps, a.batch, s, device))
        keys = runs[0].keys()
        return {k: st.mean(r[k] for r in runs) for k in keys}

    cols = ["hard_within_acc", "easy_within_acc", "cluster_acc", "top1", "hard_within_margin"]
    hdr = "arm".ljust(22) + "".join(c.rjust(20) for c in cols)

    print("\n=== signal=True (XOR recoverable) ===")
    print(hdr)
    fr = arm_mean("frame", 0, True)
    print("frame (M=0)".ljust(22) + "".join(f"{fr[c]:>20.3f}" for c in cols))
    for M in a.budget:
        for arm in ("untargeted", "targeted"):
            r = arm_mean(arm, M, True)
            print(f"{arm} (M={M})".ljust(22) + "".join(f"{r[c]:>20.3f}" for c in cols))

    print("\n=== signal=False control (parity is noise -> all must fail) ===")
    print(hdr)
    for arm, M in (("frame", 0), ("targeted", max(a.budget)), ("untargeted", max(a.budget))):
        r = arm_mean(arm, M, False)
        print(f"{arm} (M={M})".ljust(22) + "".join(f"{r[c]:>20.3f}" for c in cols))


if __name__ == "__main__":
    main()
