"""Does the routing margin degrade with the number of features at fixed budget? (interference probe)

Tests the OPEN routing-side Welch conjecture (the INTERFERENCE half of the two-sided packing
story; the RANK half is proved in i-orca RoutingRank.thy). At fixed rule budget M and fixed
(slack) dimension d, sweep n_hard = number of routing features packed into the M-rule subspace,
and watch the mean hard-cluster within-synonym MARGIN. §5e showed accuracy saturates; margin
should reveal whether features interfere: if the conjecture holds, margin degrades monotonically
with n_hard at fixed M (more features sharing the rank-≤min(M,d) subspace = more cross-talk),
even where accuracy stays high. Flat margin would mean no measurable interference in this regime.

Run:  python experiments/interference_probe.py --seeds 3
"""

from __future__ import annotations

import argparse
import statistics as st

import torch

from pil.geometry import welch_bound
from pil.learner import PILConfig, ProjectiveIncidenceLearner
from pil.proposer import RuleBank, build_sources
from pil.synthetic import create_compositional_problem


def split(z, tgt, seed):
    g = torch.Generator(device="cpu").manual_seed(seed + 99)
    perm = torch.randperm(len(tgt), generator=g).to(z.device)
    n_tr = int(0.75 * len(tgt))
    return perm[:n_tr], perm[n_tr:]


def routing_feature_coherence(h, targets, hard_ids):
    """Mean off-diagonal |cosine| of the realized routing features in rule-activation space.

    For each hard cluster c, its routing feature is the mean rule-activation DIFFERENCE between
    its two synonyms, f_c = E[h | target=2c] - E[h | target=2c+1] in R^M. As n_hard features are
    packed into M rules, their mutual coherence μ should rise toward the Welch floor if the
    conjecture's interference mechanism is active; it staying low means training packs them with
    little cross-talk (efficient superposition). h: (N, M); returns (μ, n_features)."""
    feats = []
    for c in hard_ids:
        a = h[targets == 2 * c]
        b = h[targets == 2 * c + 1]
        if len(a) and len(b):
            feats.append(a.mean(0) - b.mean(0))
    if len(feats) < 2:
        return float("nan"), len(feats)
    F = torch.stack(feats)
    F = F / (F.norm(dim=1, keepdim=True) + 1e-8)
    G = (F @ F.T).abs()
    n = G.shape[0]
    mu = ((G.sum() - n) / (n * (n - 1))).item()
    return mu, n


def run(n_hard, M, dim, V, K, n_clusters, steps, batch, seed, device):
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
        opt.zero_grad()
        out["total_loss"].backward()
        opt.step()
        model.normalize_U()
    with torch.no_grad():
        ze, te = z[ev], tgt[ev]
        L = model.forward(build_sources(ze, A, bank), target=None)["logits"]
        sib = te ^ 1
        m = L.gather(1, te[:, None]).squeeze(1) - L.gather(1, sib[:, None]).squeeze(1)
        ishard = hc[te // 2]
        mh = m[ishard]
        mean_m = mh.mean().item()
        p10 = torch.quantile(mh, 0.10).item()
        acc = (mh > 0).float().mean().item()
        h = torch.relu(ze @ bank.W.T + bank.b)            # rule activations (Nev, M)
        hard_ids = [c for c in range(n_clusters) if bool(hc[c])]
        mu, _ = routing_feature_coherence(h, te, hard_ids)
    welch_floor = welch_bound(len(hard_ids), M) ** 0.5   # Welch coherence floor for n_hard in M dims
    return mean_m, p10, acc, mu, welch_floor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--clusters", type=int, default=24)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--M", type=int, nargs="+", default=[8, 16])
    p.add_argument("--n-hard", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 20, 24])
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V, K = 2 * a.clusters, 3 * a.clusters

    print(f"[interference] dim={a.dim} (rank slack); sweep n_hard features into fixed budgets M={a.M}; "
          f"held-out, seeds={a.seeds}")
    margins = {}
    cohs = {}
    for M in a.M:
        print(f"\n=== M={M} rules ===")
        print("  n_hard    margin     p10    acc    coh_mu   welch_floor")
        for nh in a.n_hard:
            runs = [run(nh, M, a.dim, V, K, a.clusters, a.steps, a.batch, s, device)
                    for s in range(a.seeds)]
            mg = st.mean(r[0] for r in runs)
            p10 = st.mean(r[1] for r in runs)
            ac = st.mean(r[2] for r in runs)
            mu = st.mean(r[3] for r in runs)
            wf = runs[0][4]
            margins[(nh, M)] = mg
            cohs[(nh, M)] = mu
            print(f"  {nh:>6}  {mg:>8.2f}  {p10:>6.2f}  {ac:>5.2f}  {mu:>7.3f}  {wf:>10.3f}")

    print("\ninterference verdict (margin n=1 -> n=max; coherence growth; vs Welch):")
    for M in a.M:
        lo, hi = margins[(a.n_hard[0], M)], margins[(a.n_hard[-1], M)]
        mu_lo, mu_hi = cohs[(a.n_hard[0], M)], cohs[(a.n_hard[-1], M)]
        trend = "DEGRADES" if hi < lo - 0.10 else ("flat" if abs(hi - lo) <= 0.10 else "rises")
        print(f"  M={M:>2}: margin {lo:.2f}->{hi:.2f} ({trend});  "
              f"coh_mu {mu_lo:.3f}->{mu_hi:.3f}  (rising coherence => interference mechanism)")


if __name__ == "__main__":
    main()
