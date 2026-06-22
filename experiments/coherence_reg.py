"""Does a generator-side soft-Welch regularizer help in the overpacked (n>M) regime? (Grok Idea 3)

§5f located the routing interference in rule-activation space R^M, appearing when n_hard > M with
coherence at ~the Welch floor. Grok's Idea 3: add an auxiliary loss penalizing rule-firing coherence
to reduce cross-talk. We test it in the OVERPACKED regime (n_hard > M, where there IS forced coherence),
sweeping the regularizer weight, reporting hard-within margin/acc and the realized routing-feature
coherence (measured with labels, for diagnosis only -- the penalty itself is label-free).

Prediction (from §5b's frame-side null + §5f's near-optimal packing): the regularizer drives the
firing coherence down but does NOT meaningfully improve margin/acc -- SGD already packs near the Welch
floor. A clear improvement would be a surprise worth having.

Run:  python experiments/coherence_reg.py --seeds 3
"""

from __future__ import annotations

import argparse
import statistics as st

import torch

from pil.learner import PILConfig, ProjectiveIncidenceLearner
from pil.proposer import RuleBank, build_sources
from pil.scoring import activation_decorrelation_penalty
from pil.synthetic import create_compositional_problem, within_cluster_report


def split(z, tgt, seed):
    g = torch.Generator(device="cpu").manual_seed(seed + 99)
    perm = torch.randperm(len(tgt), generator=g).to(z.device)
    n_tr = int(0.75 * len(tgt))
    return perm[:n_tr], perm[n_tr:]


def routing_coherence(h, targets, n_clusters):
    feats = []
    for c in range(n_clusters):
        a = h[targets == 2 * c]
        b = h[targets == 2 * c + 1]
        if len(a) and len(b):
            feats.append(a.mean(0) - b.mean(0))
    if len(feats) < 2:
        return float("nan")
    F = torch.stack(feats)
    F = F / (F.norm(dim=1, keepdim=True) + 1e-8)
    G = (F @ F.T).abs()
    n = G.shape[0]
    return ((G.sum() - n) / (n * (n - 1))).item()


def run(coh_reg, n_hard, M, dim, V, K, n_clusters, steps, batch, seed, device):
    cfg = PILConfig(dim=dim, n_propositions=V, n_sources_per_step=1, lr=5e-3, frame_reg=0.0,
                    margin_weight=0.5, margin_target=1.8, seed=seed, device=device)
    z, A, tgt, _, hc = create_compositional_problem(
        cfg, n_examples=2048, n_clusters=n_clusters, n_hard=n_hard
    )
    tr, ev = split(z, tgt, seed)
    model = ProjectiveIncidenceLearner(cfg)
    bank = RuleBank(K, dim, M, device=device, seed=seed)
    opt = torch.optim.AdamW(list(model.parameters()) + list(bank.parameters()), lr=5e-3,
                            weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed + 11)
    for _ in range(steps):
        idx = tr[torch.randint(0, len(tr), (batch,), generator=g)]
        out = model.forward(build_sources(z[idx], A, bank), target=tgt[idx])
        loss = out["total_loss"]
        if coh_reg > 0:
            h = torch.relu(z[idx] @ bank.W.T + bank.b)        # (batch, M) rule activations
            loss = loss + coh_reg * activation_decorrelation_penalty(h)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.normalize_U()
    with torch.no_grad():
        ze, te = z[ev], tgt[ev]
        L = model.forward(build_sources(ze, A, bank), target=None)["logits"]
        rep = within_cluster_report(L, te, hc)
        h = torch.relu(ze @ bank.W.T + bank.b)
        mu = routing_coherence(h, te, n_clusters)
    return rep["hard_within_margin"], rep["hard_within_acc"], mu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--clusters", type=int, default=24)
    p.add_argument("--n-hard", type=int, default=24)      # all hard -> overpacked for small M
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--M", type=int, nargs="+", default=[8, 12])
    p.add_argument("--reg", type=float, nargs="+", default=[0.0, 0.1, 0.5, 2.0])
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V, K = 2 * a.clusters, 3 * a.clusters

    print(f"[coh-reg] dim={a.dim} V={V} n_hard={a.n_hard} (overpacked for M<{a.n_hard}); "
          f"sweep soft-Welch reg weight; held-out, seeds={a.seeds}")
    for M in a.M:
        print(f"\n=== M={M} rules  (n_hard/M = {a.n_hard/M:.1f}x overpacked) ===")
        print("  reg     margin    acc    coh_mu")
        base = None
        for r in a.reg:
            runs = [run(r, a.n_hard, M, a.dim, V, K, a.clusters, a.steps, a.batch, s, device)
                    for s in range(a.seeds)]
            mg = st.mean(x[0] for x in runs)
            ac = st.mean(x[1] for x in runs)
            mu = st.mean(x[2] for x in runs)
            if base is None:
                base = (mg, ac)
            tag = "" if r == 0 else f"  (Δmargin {mg-base[0]:+.2f}, Δacc {ac-base[1]:+.2f})"
            print(f"  {r:>4}  {mg:>8.2f}  {ac:>5.2f}  {mu:>7.3f}{tag}")


if __name__ == "__main__":
    main()
