"""Lightweight visualization helpers (matplotlib optional).

Imports of matplotlib are deferred so the core package and headless/CI runs never
require it; call these only when you actually want a figure.
"""

from __future__ import annotations

from typing import Any


def _require_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe; overridden if a backend is already set
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("matplotlib is required for pil.viz; `pip install matplotlib`") from e


def plot_training(history: dict[str, list[float]], margin_target: float, save_path: str | None = None) -> Any:
    """Three-panel training curve: loss, mean margin (with target line), retrievable %."""
    plt = _require_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history.get("loss", []))
    axes[0].set_title("training loss")
    axes[0].set_xlabel("eval tick")

    axes[1].plot(history.get("margin", []))
    axes[1].axhline(margin_target, color="r", linestyle="--", label="target")
    axes[1].set_title("mean worst-competitor margin")
    axes[1].legend()

    axes[2].plot([r * 100 for r in history.get("retrievable", [])])
    axes[2].set_title("retrievable fraction (%)")
    axes[2].set_ylim(0, 100)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_gram(U, save_path: str | None = None) -> Any:
    """Heatmap of the cosine Gram |rho_vw| -- the non-truth-functionality kernel (T2)."""
    plt = _require_mpl()
    from .geometry import cosine_gram

    G = cosine_gram(U).detach().cpu().abs().numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(G, cmap="magma", vmin=0, vmax=1)
    ax.set_title("|cosine Gram|  (0 = orthogonal, 1 = synonym)")
    fig.colorbar(im, ax=ax)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
