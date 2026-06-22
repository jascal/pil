"""Synthetic Projective Incidence Learning experiment.

Plants a hidden frame ``U_gt`` whose active sources point at a target proposition,
then trains PIL to recover a high-margin, well-conditioned frame. Demonstrates the
generate -> gate -> refine loop and reports decode-side retrievability alongside the
frame-side conditioning (frame potential vs the Welch floor).

Run:  python experiments/synthetic_pil.py --steps 2000
      python experiments/synthetic_pil.py --steps 2000 --visualize --save-dir runs/
"""

from __future__ import annotations

import argparse
import json

import torch

from pil.learner import PILConfig, ProjectiveIncidenceLearner, create_synthetic_problem


def _progress(iterable, desc):
    """tqdm if available, else a plain range (no hard dependency)."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except ImportError:
        return iterable


def run_synthetic_experiment(
    dim: int = 32,
    n_propositions: int = 64,
    n_sources: int = 24,
    steps: int = 3000,
    batch: int = 32,
    n_examples: int = 256,
    eval_every: int = 50,
    seed: int = 0,
    visualize: bool = False,
    save_dir: str | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PILConfig(
        dim=dim,
        n_propositions=n_propositions,
        n_sources_per_step=n_sources,
        lr=5e-3,
        frame_reg=0.05,
        margin_weight=0.5,
        margin_target=1.8,
        seed=seed,
        device=device,
    )
    print(f"[pil] device={device}  dim={dim}  V={n_propositions}  J={n_sources}")

    model = ProjectiveIncidenceLearner(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    sources, targets, U_gt = create_synthetic_problem(
        config, n_examples=n_examples, sparsity=0.12, noise=0.08
    )
    print(f"[pil] {len(targets)} synthetic examples, sources={tuple(sources.shape)}")

    gen = torch.Generator(device="cpu").manual_seed(seed + 7)
    history: dict[str, list[float]] = {"loss": [], "margin": [], "retrievable": [], "pr": []}

    pbar = _progress(range(steps), "PIL")
    last = {}
    for step in pbar:
        idx = torch.randint(0, len(targets), (batch,), generator=gen).to(device)
        # NB: pass the 3-D (batch, J, dim) slice DIRECTLY -- do not flatten (B,J) together,
        # which would pool every example's sources into one logit vector (the starter bug).
        metrics = model.training_step(
            sources=sources[idx],
            target=targets[idx],
            optimizer=optimizer,
        )

        if step % eval_every == 0 or step == steps - 1:
            ev = model.evaluate_retrievability(
                sources, targets, margin_threshold=config.margin_target * 0.8
            )
            history["loss"].append(metrics["loss"])
            history["margin"].append(ev["avg_margin"])
            history["retrievable"].append(ev["retrievable_fraction"])
            history["pr"].append(ev["mean_support_pr"])
            last = ev
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.3f}",
                    retr=f"{ev['retrievable_fraction']*100:.0f}%",
                    margin=f"{ev['avg_margin']:.2f}",
                    fp_welch=f"{ev['frame_pot_over_welch']:.2f}",
                )

    print("\n[pil] final:", json.dumps({k: round(v, 4) for k, v in last.items()}, indent=2))

    if visualize:
        from pil.viz import plot_gram, plot_training

        save = (save_dir.rstrip("/") + "/") if save_dir else None
        if save:
            import os

            os.makedirs(save_dir, exist_ok=True)
        plot_training(
            {"loss": history["loss"], "margin": history["margin"], "retrievable": history["retrievable"]},
            margin_target=config.margin_target,
            save_path=(save + "pil_synthetic_training.png") if save else None,
        )
        plot_gram(model.U, save_path=(save + "pil_gram.png") if save else None)

    return model, history, last


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--n-propositions", type=int, default=64)
    p.add_argument("--n-sources", type=int, default=24)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--save-dir", type=str, default=None)
    a = p.parse_args()
    run_synthetic_experiment(
        dim=a.dim,
        n_propositions=a.n_propositions,
        n_sources=a.n_sources,
        steps=a.steps,
        seed=a.seed,
        visualize=a.visualize,
        save_dir=a.save_dir,
    )
