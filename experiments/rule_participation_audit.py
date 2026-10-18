"""Ablate each rule of two trained XOR banks and record which hard clusters move.

Trains with ``routing_complexity.train_bank`` at the §5e points treated as superposition
(``M=16`` and ``M=8``, ``n_hard=24``, ``dim=32``) and keeps the bank. Each rule is removed
by zeroing its emitted source on the ``build_sources`` / learner forward path. A hard
cluster counts as moved when its held-out within-cluster margin changes by more than
``MARGIN_MOVE_TOL``. The verdict uses generator atom ownership: clusters do not share
atoms, so a rule that moves two or more hard clusters is unrelated packing, and a rule
that moves one is reuse.

Run:  python experiments/rule_participation_audit.py --seed 0
"""

from __future__ import annotations

import argparse
import os

# Fixed before torch loads so two CPU invocations share one BLAS reduction order.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from routing_complexity import train_bank

from pil.learner import PILConfig
from pil.proposer import build_sources
from pil.rule_participation import (
    MARGIN_MOVE_TOL,
    bank_verdict,
    budget_agreement,
    code_pair_mass,
    mass_share,
    rule_agreement,
    rule_margin_abs_delta,
    rule_moved_clusters,
    rule_verdict,
    topic_mass,
)
from pil.synthetic import create_compositional_problem, within_cluster_report


def _fmt_moved(moved: list[int]) -> str:
    return ",".join(str(c) for c in moved) if moved else "-"


def _fmt_share(leader: int, top1: float, top3: float | None) -> str:
    lead = "-" if leader < 0 else str(leader)
    if top3 is None:
        return f"{lead}:{top1:.6f}"
    return f"{lead}:{top1:.6f}/{top3:.6f}"


def _clean_held(args, device: str, tgt, ev):
    """Noise-free counterpart. Same seed, ``noise=0``; targets must match or the run stops."""
    n_clusters = args.clusters
    cfg = PILConfig(
        dim=args.dim, n_propositions=2 * n_clusters, n_sources_per_step=1, lr=5e-3,
        frame_reg=0.0, margin_weight=0.5, margin_target=1.8, seed=args.seed, device=device,
    )
    z_clean, _, tgt_clean, _, _ = create_compositional_problem(
        cfg, n_examples=int(tgt.shape[0]), n_clusters=n_clusters, n_hard=args.n_hard, noise=0.0,
    )
    if not torch.equal(tgt_clean, tgt):
        raise RuntimeError(
            "clean counterpart targets diverged from the training batch; "
            "the generator RNG order changed"
        )
    return z_clean[ev], tgt_clean[ev]


def audit_budget(M: int, args, device: str) -> tuple[str, str, str, float]:
    n_clusters = args.clusters
    n_hard = args.n_hard
    V, K = 2 * n_clusters, 3 * n_clusters
    model, bank, z, A, tgt, hc, ev = train_bank(
        n_hard, M, args.dim, V, K, n_clusters, args.steps, args.batch, args.seed, device
    )
    assert bank is not None and bank.n_rules == M
    held_z, held_tgt = z[ev], tgt[ev]
    base = model.forward(build_sources(held_z, A, bank), target=None)["logits"]
    acc = within_cluster_report(base, held_tgt, hc)["hard_within_acc"]
    lines = [f"BUDGET M={M} n_hard={n_hard} dim={args.dim} seed={args.seed} hard_within_acc={acc:.6f}"]
    clean_z, clean_tgt = _clean_held(args, device, tgt, ev)
    verdicts: list[str] = []
    agreements: list[str] = []
    for rule in range(bank.n_rules):
        moved = rule_moved_clusters(
            model, held_z, A, bank, held_tgt, hc, rule, tol=args.tol
        )
        kind = rule_verdict(moved, n_clusters)
        verdicts.append(kind)
        lines.append(f"M={M} rule={rule} moved={_fmt_moved(moved)} verdict={kind}")
        noisy = mass_share(rule_margin_abs_delta(
            model, held_z, A, bank, held_tgt, n_clusters, rule
        ))
        clean = mass_share(rule_margin_abs_delta(
            model, clean_z, A, bank, clean_tgt, n_clusters, rule
        ))
        weight = mass_share(code_pair_mass(bank.W[rule], n_clusters), idle_at=0.0)
        topic = mass_share(topic_mass(bank.W[rule], n_clusters), idle_at=0.0)
        label = rule_agreement(noisy[3], clean[3], weight[3])
        agreements.append(label)
        lines.append(
            f"M={M} rule={rule} "
            f"noisy={_fmt_share(noisy[0], noisy[1], noisy[2])} "
            f"clean={_fmt_share(clean[0], clean[1], clean[2])} "
            f"weight={_fmt_share(weight[0], weight[1], None)} "
            f"topic={_fmt_share(topic[0], topic[1], None)} "
            f"label={label}"
        )
    overall = bank_verdict(verdicts)
    n_pack = sum(v == "packing" for v in verdicts)
    n_reuse = sum(v == "reuse" for v in verdicts)
    n_idle = sum(v == "idle" for v in verdicts)
    lines.append(
        f"BUDGET_VERDICT M={M} verdict={overall} packing={n_pack} reuse={n_reuse} idle={n_idle}"
    )
    agreement = budget_agreement(agreements)
    lines.append(f"BUDGET_AGREEMENT M={M} agreement={agreement}")
    return "\n".join(lines), overall, agreement, acc


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--clusters", type=int, default=24)
    p.add_argument("--n-hard", type=int, default=24)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budgets", type=int, nargs="+", default=[16, 8])
    p.add_argument("--tol", type=float, default=MARGIN_MOVE_TOL)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    device = args.device
    print(
        f"[participation] device={device} seed={args.seed} dim={args.dim} "
        f"clusters={args.clusters} n_hard={args.n_hard} steps={args.steps} "
        f"batch={args.batch} tol={args.tol} budgets={args.budgets}"
    )
    overalls = []
    agreements = []
    for M in args.budgets:
        text, overall, agreement, acc = audit_budget(M, args, device)
        print(text, flush=True)
        print(f"ACC_OK M={M} acc={acc:.6f} above_0.65={acc > 0.65}", flush=True)
        overalls.append(overall)
        agreements.append(agreement)
    if any(v == "packing" for v in overalls):
        final = "packing"
    elif any(v == "reuse" for v in overalls):
        final = "reuse"
    else:
        final = "idle"
    prose = "narrow" if agreements and all(a == "agree_single" for a in agreements) else "unchanged"
    print(f"OVERALL verdict={final}")
    print(f"PROSE action={prose}")


if __name__ == "__main__":
    main()
