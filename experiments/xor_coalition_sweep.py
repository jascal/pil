"""Multi-seed single and cumulative XOR rule deletions; see xor_coalition_protocol.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from routing_complexity import split, train_bank

from pil.proposer import build_sources
from pil.rule_participation import ACC_DROP, deletion_effect, mass_share, per_cluster_within_margins
from pil.synthetic import within_cluster_report


def deleted_logits(model, sources, atom_count, removed):
    """Delete a set on the actual source/forward path, preserving raw atom sources."""
    edited = sources.clone()
    if removed:
        edited[:, [atom_count + r for r in removed], :] = 0
    return model(edited, target=None)["logits"]


def curve(model, sources, atom_count, targets, hc, order):
    return [within_cluster_report(
        deleted_logits(model, sources, atom_count, order[:k]), targets, hc
    )["hard_within_acc"] for k in range(len(order) + 1)]


def curve_summary(values):
    m = len(values) - 1
    return {
        "auc": sum((a + b) / 2 for a, b in zip(values[:-1], values[1:], strict=True)) / m,
        "first_material_deletions": next(
            (k for k, v in enumerate(values) if values[0] - v > ACC_DROP), None
        ),
    }


def run_cell(nh, budget, seed, args):
    model, bank, z, A, tgt, hc, ev = train_bank(
        nh, budget, 32, 48, 72, 24, args.steps, 64, seed, "cpu"
    )
    tr, check_ev = split(z, tgt, seed)
    assert torch.equal(ev, check_ev)
    with torch.no_grad():
        held = build_sources(z[ev], A, bank)
        training = build_sources(z[tr], A, bank)
        base = model(held, target=None)["logits"]
        train_base = model(training, target=None)["logits"]
        means = per_cluster_within_margins(base, tgt[ev], 24)
        rules, rank_drops = [], []
        for r in range(budget):
            after = deleted_logits(model, held, len(A), [r])
            acc, drop, flips, load = deletion_effect(base, after, tgt[ev], hc)
            delta = (means - per_cluster_within_margins(after, tgt[ev], 24)).abs()
            hard_delta = delta[hc]
            _, top1, top3, concentration = mass_share(hard_delta)
            moved = int((hard_delta > 1e-3).sum())
            rules.append(dict(rule=r, acc_after=acc, drop=drop, flipped=flips, load=load,
                              moved_hard_clusters=moved, top1=top1, top3=top3,
                              concentration=concentration))
            rank_drops.append(deletion_effect(
                train_base, deleted_logits(model, training, len(A), [r]), tgt[tr], hc
            )[1])
        order = sorted(range(budget), key=lambda r: (-rank_drops[r], r))
        ranked = curve(model, held, len(A), tgt[ev], hc, order)
        generator = torch.Generator().manual_seed(10000 + seed)
        random_orders = [torch.randperm(budget, generator=generator).tolist()
                         for _ in range(args.random_orders)]
        random_curves = [curve(model, held, len(A), tgt[ev], hc, o) for o in random_orders]
        # All orders must have the same intact and raw-only endpoint.
        assert all(c[0] == ranked[0] and c[-1] == ranked[-1] for c in random_curves)
    return dict(n_hard=nh, budget=budget, seed=seed, intact_acc=ranked[0],
                coalition=any(r["load"] == "material" for r in rules), rules=rules,
                training_drops=rank_drops, ranked_order=order, ranked_curve=ranked,
                ranked_summary=curve_summary(ranked), random_orders=random_orders,
                random_curves=random_curves,
                random_summaries=[curve_summary(c) for c in random_curves])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--n-hard", type=int, nargs="+", default=[8, 16, 24])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--random-orders", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.random_orders < 1 or min(args.budgets) < 1 or not all(0 < n <= 24 for n in args.n_hard):
        p.error("need positive permutations/budgets and n-hard in 1..24")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
        Path(__file__), Path("experiments/routing_complexity.py"), Path("pil/rule_participation.py"),
        Path("pil/synthetic.py"), Path("docs/notes/xor_coalition_protocol.md"),
    ]}
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  torch_version=torch.__version__, sha256=provenance, cells=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for nh in args.n_hard:
        for budget in args.budgets:
            for seed in args.seeds:
                cell = run_cell(nh, budget, seed, args)
                report["cells"].append(cell)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(f"n_hard={nh} M={budget} seed={seed} acc={cell['intact_acc']:.6f} "
                      f"coalition={cell['coalition']}", flush=True)


if __name__ == "__main__":
    main()
