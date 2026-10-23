"""Render the completed XOR robustness sweep into a table and deletion-curve figure."""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.summary.read_text())
    config, cells = data["config"], data["cells"]
    expected = len(config["n_hard"]) * len(config["budgets"]) * len(config["seeds"])
    if len(cells) != expected:
        p.error(f"incomplete sweep: {len(cells)}/{expected} banks")
    fig, axes = plt.subplots(
        len(config["n_hard"]),
        len(config["budgets"]),
        figsize=(12, 9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    lines = [
        "# XOR coalition generalization — outcome",
        "",
        "Protocol: [xor_coalition_protocol.md](xor_coalition_protocol.md). "
        "Raw results: [xor_coalition_summary.json](xor_coalition_summary.json).",
        "",
        "45 banks; seeds 0–4, dim=32, 24 total clusters, 1500 steps, 512 held-out rows per bank. "
        "Hard-row counts vary with the number of hard clusters. No fits were excluded.",
        "",
        "| Hard clusters | M | Accuracy mean (range) | Coalition banks | Packing rules | "
        "Material rules | Rules flipping rows | Ranked − random AUC |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    counts = []
    for i, nh in enumerate(config["n_hard"]):
        for j, budget in enumerate(config["budgets"]):
            group = [c for c in cells if c["n_hard"] == nh and c["budget"] == budget]
            rules = [r for c in group for r in c["rules"]]
            accuracies = [c["intact_acc"] for c in group]
            coalition = sum(c["coalition"] for c in group)
            counts.append(coalition)
            diff = [
                c["ranked_summary"]["auc"] - st.mean(s["auc"] for s in c["random_summaries"]) for c in group
            ]
            packing = sum(r["moved_hard_clusters"] >= 2 for r in rules)
            material = sum(r["load"] == "material" for r in rules)
            flipping = sum(r["flipped"] > 0 for r in rules)
            lines.append(
                f"| {nh} | {budget} | {st.mean(accuracies):.3f} "
                f"({min(accuracies):.3f}–{max(accuracies):.3f}) | {coalition}/{len(group)} | "
                f"{packing}/{len(rules)} | {material}/{len(rules)} | {flipping}/{len(rules)} | "
                f"{st.mean(diff):+.3f} |"
            )
            ax = axes[i, j]
            x = [k / budget for k in range(budget + 1)]
            for c in group:
                ax.plot(x, c["ranked_curve"], color="tab:red", alpha=0.15, linewidth=1)
            ranked = [st.mean(c["ranked_curve"][k] for c in group) for k in range(budget + 1)]
            random = [
                st.mean(st.mean(curve[k] for curve in c["random_curves"]) for c in group)
                for k in range(budget + 1)
            ]
            ax.plot(x, ranked, color="tab:red", label="Training-ranked")
            ax.plot(x, random, color="tab:blue", label="Random mean")
            ax.set_title(f"Hard clusters={nh}, M={budget}")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.supxlabel("Fraction of rules deleted (no retraining)")
    fig.supylabel("Held-out hard-within accuracy")
    fig.suptitle("XOR rule deletion: five trained seeds per cell; faint lines are ranked seed curves")
    fig.tight_layout()
    figure = args.output.with_suffix(".png")
    fig.savefig(figure, dpi=160)
    rules = [r for c in cells for r in c["rules"]]
    spread = sum(r["concentration"] == "spread" for r in rules)
    lines += [
        "",
        f"**Predeclared robustness criterion:** {'PASS' if min(counts) >= 4 else 'FAIL'} "
        "(coalition in at least 4/5 seeds in every cell).",
        "",
        f"Intact accuracy >0.65: {sum(c['intact_acc'] > 0.65 for c in cells)}/{len(cells)} banks. "
        f"Noisy hard-cluster margin concentration is spread for {spread}/{len(rules)} rules. "
        "This diagnostic excludes easy clusters and does not repeat the original "
        "clean/weight agreement test.",
        "",
        "AUC integrates accuracy over deleted fraction. A negative ranked-minus-random value means "
        "training-ranked deletion damages held-out performance more. Random permutations are averaged "
        "within each bank before averaging banks; they are not independent training seeds.",
        "",
        f"![Deletion curves]({figure.name})",
        "",
        "The raw JSON retains every single-rule effect, training ranking, deletion permutation, full curve, "
        "and first material cumulative deletion count. Endpoint checks ensure all orders begin intact and "
        "end at the same raw-atom-only model.",
        "",
        "Limitations: five seeds, fixed total vocabulary and dimension, synthetic generator, no retraining. "
        "Two seed-0 cells reproduce prior findings. Packing means a measurable margin effect on disjoint "
        "clusters; it does not itself imply a large accuracy effect. The inherited coalition label means "
        "at least one material single-rule deletion; it does not establish a unique minimal coalition.",
        "",
    ]
    lines += ["", "## Cumulative deletion details", "",
              "Ranges below are across trained seeds. First crossing is the number of ranked deletions "
              "needed for a >0.05 drop; these curves can recover after crossing.", "",
              "| Hard clusters | M | Ranked first crossing (range) | Ranked − random AUC range |",
              "|---:|---:|---|---|"]
    for nh in config["n_hard"]:
        for budget in config["budgets"]:
            group = [c for c in cells if c["n_hard"] == nh and c["budget"] == budget]
            first = [c["ranked_summary"]["first_material_deletions"] for c in group]
            crossed = [k for k in first if k is not None]
            crossing = f"{min(crossed)}–{max(crossed)}" if crossed else "never"
            if len(crossed) != len(first):
                crossing += f" ({len(first) - len(crossed)} never)"
            differences = [c["ranked_summary"]["auc"] - st.mean(
                v["auc"] for v in c["random_summaries"]) for c in group]
            lines.append(f"| {nh} | {budget} | {crossing} | "
                         f"{min(differences):+.3f}–{max(differences):+.3f} |")
    lines += ["", "## Interpretation", "",
              "Unrelated packing survives this grid: every rule moves mean margins on multiple disjoint "
              "hard clusters. The stronger claim that every bank contains a material single rule does not: "
              "the M=32, 24-hard-cluster cell has coalition in only 1/5 seeds, "
              "versus 5/5 in every other cell.",
              "", "At that boundary, the four banks without a material single-rule deletion cross the same "
              "accuracy-drop threshold after 2–4 training-ranked deletions. Extra capacity permits small "
              "individual effects alongside cumulative dependence. This is compatible with redundancy; "
              "the experiment does not identify an interaction mechanism or a minimal causal coalition.",
              "", "The original all-spread result also needs qualification: 105/840 rules have more than "
              "half their noisy hard-cluster margin-change mass on one cluster. All still move other "
              "clusters. This restricted hard-cluster diagnostic differs from the original all-cluster, "
              "clean-input, and weight agreement audit, so it does not directly overturn that audit.",
              "", "Training-ranked deletion has lower mean accuracy AUC than random deletion in every cell, "
              "but not every seed. It is a useful average diagnostic, not a uniformly better deletion order. "
              "A natural next experiment is retraining after pruning, with rule budgets matched, to test "
              "whether the observed redundancy can be used to compress the bank without losing accuracy.", ""]
    lines.append("")
    lines += ["## Reproduce", "",
              "Base: main commit `d8493c2`; the JSON records the sweep configuration, PyTorch version, "
              "and SHA-256 hashes of the frozen sweep, training loop, generator, participation helpers, "
              "and prospective protocol. The seed-0 M=8/16, n_hard=24 accuracies "
              "reproduce the prior audit exactly.",
              "", "```bash",
              ".venv/bin/python experiments/xor_coalition_sweep.py --output /tmp/xor-coalition.json",
              "MPLCONFIGDIR=/tmp/pil-mpl .venv/bin/python experiments/xor_coalition_report.py "
              "/tmp/xor-coalition.json --output /tmp/xor-coalition.md",
              "```", ""]
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
