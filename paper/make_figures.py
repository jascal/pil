"""Regenerate the paper's data figures directly from the numbers in ../results/.

Every value here is transcribed from a checked-in result file (cited in each
function); no new computation. Produces PDF + PNG into paper/figures/.

    python paper/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


def fig_generation_is_the_lever():
    """Compositional XOR regime (results/comp_sweep.txt, signal=True, held-out)."""
    M = [0, 4, 8, 16, 32]
    untarg = [0.454, 0.577, 0.790, 0.899, 0.911]   # frame=M0 shared baseline
    targ = [0.454, 0.560, 0.765, 0.910, 0.934]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.axhline(0.5, ls=":", c="k", lw=0.8, label="chance (linear ceiling)")
    ax.plot(M, untarg, "o-", c="#1f77b4", label="untargeted (random) rules")
    ax.plot(M, targ, "s--", c="#d62728", label="min-margin targeted rules")
    ax.scatter([0], [0.454], c="k", zorder=5)
    ax.annotate("frame-only\n(linear, $M{=}0$)", (0, 0.454), (3.5, 0.43),
                fontsize=7, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("rule budget $M$")
    ax.set_ylabel("hard-cluster within-synonym acc.")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, "generation_lever")


def fig_rank_concentration():
    """Real-model scale sweep (results/real_dla_sweep.txt + decode_circuit.txt)."""
    models = ["0.5B", "1.5B", "3B"]
    nb = [49, 57, 73]
    eff = [11, 14, 12]            # per-position PR over blocks
    x = range(len(models))
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.bar(x, nb, width=0.6, color="#dddddd", label="total blocks $n_b = 2L{+}1$")
    ax.bar(x, eff, width=0.6, color="#1f77b4", label="effective blocks (PR)")
    for xi, e, t in zip(x, eff, nb):
        ax.text(xi, e + 1.2, str(e), ha="center", fontsize=8, color="#1f77b4")
        ax.text(xi, t + 1.2, str(t), ha="center", fontsize=8, color="#777777")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_xlabel("Qwen2.5 base model")
    ax.set_ylabel("number of DLA blocks")
    ax.set_ylim(0, 82)
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, "rank_concentration")


def fig_coherence_slack():
    """Coherence relative to the Welch floor: Qwen (loose) vs Pythia (near-optimal).

    Qwen mu/welch from results/real_dla_sweep.txt; Pythia from
    results/real_dla_pythia.txt. The synthetic task-specialized learner packs to
    ~1x (dashed reference). Qwen sits 2-3.6x above it; Pythia hugs it (1.06-1.5x),
    so near-Welch packing is family-dependent, not a universal real-model property.
    """
    labels = ["Q-0.5B", "Q-1.5B", "Q-3B", "code", "jargon", "shuffle",
              "P-70m", "P-160m", "P-410m", "P-1b"]
    ratio = [2.01, 2.18, 3.62, 2.27, 2.41, 3.07, 1.06, 1.39, 1.38, 1.50]
    groups = ["qscale", "qscale", "qscale", "qdiff", "qdiff", "qdiff",
              "pythia", "pythia", "pythia", "pythia"]
    cmap = {"qscale": "#1f77b4", "qdiff": "#d62728", "pythia": "#7b3fa0"}
    colors = [cmap[g] for g in groups]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar(range(len(labels)), ratio, color=colors, width=0.72)
    ax.axhline(1.0, ls="--", c="k", lw=0.9)
    ax.text(len(labels) - 0.5, 1.05, "synthetic SGD (Welch-optimal)",
            ha="right", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"coherence $\mu\,/\,$Welch floor")
    ax.set_ylim(0, 4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in cmap]
    ax.legend(handles, ["Qwen scale", "Qwen difficulty", "Pythia scale"],
              loc="upper left", framealpha=0.9, ncol=1)
    _save(fig, "coherence_slack")


def fig_m_dominance():
    """Realization cost is M-dominated, not per-decision (results/routing_complexity_sweep.txt)."""
    M = [2, 4, 8, 16, 32, 64]
    rows = {
        4:  [0.465, 0.518, 0.808, 0.915, 0.935, 0.950],
        8:  [0.496, 0.529, 0.757, 0.840, 0.900, 0.917],
        16: [0.498, 0.521, 0.745, 0.889, 0.908, 0.947],
        24: [0.535, 0.595, 0.759, 0.858, 0.915, 0.948],
    }
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    shades = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    for (n_hard, acc), c in zip(rows.items(), shades):
        ax.plot(M, acc, "o-", ms=3, c=c, label=f"$n_{{hard}}={n_hard}$")
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(M)
    ax.set_xticklabels(M)
    ax.set_xlabel("rule budget $M$")
    ax.set_ylabel("hard-cluster acc.")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right", framealpha=0.9, title="routing decisions")
    _save(fig, "m_dominance")


if __name__ == "__main__":
    fig_generation_is_the_lever()
    fig_rank_concentration()
    fig_coherence_slack()
    fig_m_dominance()
