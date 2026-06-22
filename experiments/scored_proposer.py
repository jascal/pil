"""Is the generative lever proposal SELECTION or gradient-TRAINING the rule feature?

Disentangles the two by crossing selection-method x frozen/trainable input-weights on the
compositional XOR regime (synonyms separable only by a non-linear rule). Arms, matched on
budget M and steps:

  sgd          random-init, TRAINABLE input-weights (reference)
  ambig-frozen ambiguity-score-selected input-weights, FROZEN
  ambig-train  ambiguity-score-selected input-weights, TRAINABLE
  rand-frozen  random-selected input-weights, FROZEN
  rand-train   random-selected input-weights, TRAINABLE

If selection is the lever, ambig-* > rand-*. If training is the lever, *-train > *-frozen
regardless of selection. Also reports a structural metric (clusters with both synonyms
confidently decoded -- the realized gamma-separated code, DecodeCapacity.thy).

Run:  python experiments/scored_proposer.py --budget 8 16 32 --seeds 3
"""

from __future__ import annotations

import argparse
import statistics as st

import torch

from pil.learner import PILConfig, ProjectiveIncidenceLearner
from pil.proposer import RuleBank, build_sources, rulebank_from_weights
from pil.scoring import propose_and_select
from pil.synthetic import create_compositional_problem, within_cluster_report


def split(z, tgt, seed):
    g = torch.Generator(device="cpu").manual_seed(seed + 99)
    perm = torch.randperm(len(tgt), generator=g).to(z.device)
    n_tr = int(0.75 * len(tgt))
    return perm[:n_tr], perm[n_tr:]


def margins_of(model, z, A, bank, targets):
    L = model.forward(build_sources(z, A, bank), target=None)["logits"]
    sib = targets ^ 1
    return L.gather(1, targets[:, None]).squeeze(1) - L.gather(1, sib[:, None]).squeeze(1)


def resolved_clusters(L, gamma):
    """Number of clusters whose BOTH synonyms (2c, 2c+1) are each confidently decoded
    (margin >= gamma) somewhere on eval -- the structural signal from the separation lemma:
    the code locally distinguishes the synonym pair. Discriminates better than a raw distinct-
    token count, which saturates. Max = n_clusters (= V // 2)."""
    top2 = L.topk(2, dim=-1).values
    conf = (top2[:, 0] - top2[:, 1]) >= gamma
    decoded = set(L.argmax(-1)[conf].tolist())
    return sum(1 for c in range(L.shape[1] // 2) if 2 * c in decoded and 2 * c + 1 in decoded)


def train(model, bank, z, A, targets, steps, batch, seed, device):
    params = [p for p in model.parameters()] + (
        [p for p in bank.parameters() if p.requires_grad] if bank is not None else []
    )
    opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed + 11)
    for _ in range(steps):
        idx = torch.randint(0, len(targets), (batch,), generator=g).to(device)
        out = model.forward(build_sources(z[idx], A, bank), target=targets[idx])
        opt.zero_grad()
        out["total_loss"].backward()
        opt.step()
        model.normalize_U()


def run_arm(method, freeze, M, z, A, tgt, hc, V, K, dim, n_clusters, steps, batch, warm,
            seed, device, gamma=1.0):
    tr, ev = split(z, tgt, seed)
    z_tr, t_tr, z_ev, t_ev = z[tr], tgt[tr], z[ev], tgt[ev]
    cfg = PILConfig(dim=dim, n_propositions=V, n_sources_per_step=1, lr=5e-3,
                    frame_reg=0.0, margin_weight=0.5, margin_target=1.8, seed=seed, device=device)
    model = ProjectiveIncidenceLearner(cfg)

    # frame-only warmup so scoring sees real margins (also the diagnosis SGD-allocation would use)
    train(model, None, z_tr, A, t_tr, warm, batch, seed, device)

    if method == "sgd":
        bank = RuleBank(K, dim, M, device=device, seed=seed)          # random-init, trainable inputs
    else:
        m = margins_of(model, z_tr, A, None, t_tr)
        W = propose_and_select(z_tr, t_tr, m, V, n_candidates=256, k=M, method=method, seed=seed)
        bank = rulebank_from_weights(W, dim, device=device, seed=seed, freeze_input=freeze)

    train(model, bank, z_tr, A, t_tr, steps - warm, batch, seed, device)
    L = model.forward(build_sources(z_ev, A, bank), target=None)["logits"]
    rep = within_cluster_report(L, t_ev, hc)
    rep["top1"] = (L.argmax(-1) == t_ev).float().mean().item()
    rep["code"] = float(resolved_clusters(L, gamma))
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--clusters", type=int, default=24)
    p.add_argument("--frac-hard", type=float, default=0.33)
    p.add_argument("--n-examples", type=int, default=2048)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--warm", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=int, nargs="+", default=[2, 4, 8, 16])
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V, K = 2 * a.clusters, 3 * a.clusters

    cfg0 = PILConfig(dim=a.dim, n_propositions=V, seed=0, device=device)
    _, _, _, _, hc0 = create_compositional_problem(cfg0, n_examples=8, n_clusters=a.clusters,
                                                   frac_hard=a.frac_hard)
    print(f"[scored] dim={a.dim} V={V} K={K} hard={int(hc0.sum())}/{a.clusters} "
          f"frozen-input rules, eval=held-out, seeds={a.seeds}")

    def arm_mean(method, freeze, M):
        runs = []
        for s in range(a.seeds):
            cfg = PILConfig(dim=a.dim, n_propositions=V, seed=s, device=device)
            z, A, tgt, _, hc = create_compositional_problem(
                cfg, n_examples=a.n_examples, n_clusters=a.clusters, frac_hard=a.frac_hard
            )
            runs.append(run_arm(method, freeze, M, z, A, tgt, hc, V, K, a.dim, a.clusters,
                                a.steps, a.batch, a.warm, s, device))
        return (st.mean(r["hard_within_acc"] for r in runs), st.mean(r["code"] for r in runs))

    # cross selection method x frozen/trainable, to disentangle "selection quality" from
    # "trained-vs-frozen feature". sgd = random-init trainable (the reference).
    arms = [("sgd", "sgd", None), ("ambig-frozen", "ambiguity", True),
            ("ambig-train", "ambiguity", False), ("rand-frozen", "random", True),
            ("rand-train", "random", False)]
    print(f"\n(acc = hard-cluster within-synonym accuracy held-out, chance 0.5; "
          f"code = # clusters with BOTH synonyms confidently decoded, max {a.clusters})")
    print("\n" + "budget".rjust(8) + "".join(label.rjust(15) for label, _, _ in arms))
    for M in a.budget:
        cells = {label: arm_mean(method, freeze, M) for label, method, freeze in arms}
        print(f"{M:>8}" + "".join(f"{cells[label][0]:>9.3f}/{cells[label][1]:>4.0f}"
                                  for label, _, _ in arms))


if __name__ == "__main__":
    main()
