"""T-traj discharge experiment 1: do real P3 steps satisfy the theorem's premises?

T-traj (kernel-proved, i-orca ``examples/pic_learn/PIC_Learn.thy``, merged main 735973f)
is unconditional math; what is NOT known is whether real optimizer steps — AdamW +
row renormalization, pil's actual P3 — live inside its premises
(``m_t > 2(ρ·ε_t + β_t)`` per step), and with how much slack. This experiment measures
exactly that, per the discharge plan (workspace ``PICARD_PROOF_PLAN.md``, experiment 1).

Regimes (sweep, don't spot-check; per-cell reporting, no pooling):

  easy      planted-frame synthetic (``create_synthetic_problem``), the learner's own loss
  hard      over-complete clustered synthetic (``create_clustered_problem``), same loss
  mwl_raw   the margin-widening loop's shipped ``raw`` objective (train_once semantics)
  real      a fieldrun ``--source-dump`` (pythia-70m residuals), realdata train_eval
            semantics — included when a dump path is supplied, else skipped with a note

Sweep: seeds x learning rates (shipped 5e-3 and the 3e-4-class) x dim/V sizes.
Tracked bank: train + held-out contexts (the corpus-mirage rule); decisions frozen at two
entry points — step 0 and 25% through (the picard-relevant "visit mid-loop" case).

Verdicts are DESCRIPTIVE (hold-rates and slack tables, tagged empirical); this experiment
does not conclude "P3 is certified" or "P3 can't be" — failures locate what a trust-region
P3 would need to clip (candidate fix, untested).

Run:  python experiments/t_traj_discharge.py --steps 600 --out results/t_traj_discharge
      [--dump <source_dump.jsonl>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from margin_widening_loop import margin_term  # noqa: E402  (the shipped raw objective)

from pil.certify import TrajectoryCertificate
from pil.learner import PILConfig, ProjectiveIncidenceLearner, create_synthetic_problem
from pil.synthetic import create_clustered_problem


def make_regime(regime: str, size: tuple[int, int], seed: int, dump: str | None):
    """Returns (cfg, sources, targets, loss_fn) for one cell. loss_fn(model, s, t) -> loss."""
    dim, V = size
    if regime == "easy":
        cfg = PILConfig(dim=dim, n_propositions=V, n_sources_per_step=24, seed=seed, device="cpu")
        src, tgt, _ = create_synthetic_problem(cfg, n_examples=640)

        def loss_fn(model, s, t):
            return model.forward(s, target=t)["total_loss"]

        return cfg, src, tgt, loss_fn
    if regime in ("hard", "mwl_raw"):
        cfg = PILConfig(dim=dim, n_propositions=V, n_sources_per_step=24,
                        frame_reg=0.05, margin_weight=0.5, seed=seed, device="cpu")
        src, tgt, _, _ = create_clustered_problem(
            cfg, n_examples=640, n_clusters=max(2, V // 8), cluster_spread=0.10,
            sparsity=0.12, noise=0.08,
        )
        if regime == "hard":
            def loss_fn(model, s, t):
                return model.forward(s, target=t)["total_loss"]
        else:
            def loss_fn(model, s, t):
                out = model.forward(s, target=t)
                return (
                    out["nll"]
                    + 0.5 * margin_term("raw", 2.0, out["logits"], s, t)
                    + cfg.frame_reg * out["frame_potential"]
                )
        return cfg, src, tgt, loss_fn
    if regime == "real":
        from pil.fieldrun_io import load_source_dump

        sb = load_source_dump(dump)
        vocab = sorted(set(sb.cands.reshape(-1).tolist()) | set(sb.target.tolist()))
        idx = {t: i for i, t in enumerate(vocab)}
        r = torch.tensor(sb.r, dtype=torch.float32)
        y = torch.tensor([idx[int(t)] for t in sb.target], dtype=torch.long)
        cfg = PILConfig(dim=sb.dim, n_propositions=len(vocab), n_sources_per_step=1,
                        frame_reg=0.05, margin_weight=0.5, seed=seed, device="cpu")

        def loss_fn(model, s, t):
            out = model.forward(s, target=t)
            return (
                out["nll"]
                + 0.5 * margin_term("raw", 2.0, out["logits"], s, t)
                + cfg.frame_reg * out["frame_potential"]
            )

        return cfg, r.unsqueeze(1), y, loss_fn
    raise ValueError(regime)


def run_cell(
    regime: str, size: tuple[int, int], lr: float, seed: int,
    steps: int, batch: int, dump: str | None, bank_n: int = 128,
) -> dict:
    cfg, src, tgt, loss_fn = make_regime(regime, size, seed, dump)
    size = (cfg.dim, cfg.n_propositions)      # real arm reads its size from the dump
    n = src.shape[0]
    g = torch.Generator().manual_seed(seed + 13)
    perm = torch.randperm(n, generator=g)
    n_ho = max(8, n // 5)
    ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]

    # tracked bank = train contexts + held-out contexts (corpus-mirage rule)
    kb = min(bank_n, len(tr_idx), len(ho_idx))
    bank_idx = torch.cat([tr_idx[:kb], ho_idx[:kb]])
    bank_src, bank_tgt = src[bank_idx], tgt[bank_idx]
    ho_mask = torch.zeros(2 * kb, dtype=torch.bool)
    ho_mask[kb:] = True

    model = ProjectiveIncidenceLearner(cfg)
    model.normalize_U()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    cert0 = TrajectoryCertificate(model, bank_src, ho_mask, bank_tgt)   # entry @ step 0
    cert_mid: TrajectoryCertificate | None = None
    mid = steps // 4

    Xtr, ytr = src[tr_idx], tgt[tr_idx]
    for step in range(steps):
        bidx = torch.randint(0, len(ytr), (min(batch, len(ytr)),), generator=g)
        loss = loss_fn(model, Xtr[bidx], ytr[bidx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.normalize_U()
        if step == mid:
            cert_mid = TrajectoryCertificate(model, bank_src, ho_mask, bank_tgt)
        cert0.step()
        if cert_mid is not None:
            cert_mid.step()

    return {
        "regime": regime, "dim": size[0], "V": size[1], "lr": lr, "seed": seed,
        "steps": steps,
        "entry0": cert0.summary(),
        "entry_mid": cert_mid.summary() if cert_mid is not None else {},
    }


def fmt_cell(row: dict) -> str:
    e0, em = row["entry0"], row["entry_mid"]

    def line(tag, s):
        if not s:
            return f"    {tag}: (none)"
        return (
            f"    {tag}: hold {s['premise_hold_rate_mean']:.3f}"
            f" (last25% {s['premise_hold_rate_last25%']:.3f};"
            f" per-ctx {s['premise_hold_rate_perctx_mean']:.3f}/"
            f"{s['premise_hold_rate_perctx_last25%']:.3f})"
            f" | flips {s['total_new_flips']} (never {s['never_flipped_frac']:.2f};"
            f" ho-never {s['heldout_split'].get('never_flipped_frac', float('nan')):.2f})"
            f" | VIOL pf={s['viol_premise_flip']} tr={s['viol_transfer']}"
            f" | beta-share {s['beta_share_mean']:.3f}"
            f" | 2*budget {s['budget_2sum']:.2f} vs m0_p50 {s['m0_p50']:.3f}"
            f" (vacuous {s['budget_vacuous_frac']:.2f} @T*~{s['budget_T*_p50']})"
            f" | slack p50 {s['telescope_slack_p50']:.2f}"
        )

    hdr = (f"  {row['regime']:<8} dim={row['dim']:<3} V={row['V']:<4} "
           f"lr={row['lr']:<7} seed={row['seed']}")
    return "\n".join([hdr, line("entry@0  ", e0), line("entry@25%", em)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-3, 3e-4])
    ap.add_argument("--dump", default=None, help="fieldrun --source-dump jsonl (real arm)")
    ap.add_argument("--out", default="results/t_traj_discharge")
    a = ap.parse_args()

    cells: list[tuple[str, tuple[int, int]]] = [
        ("easy", (32, 64)),
        ("hard", (24, 192)),
        ("hard", (48, 384)),
        ("mwl_raw", (24, 192)),
    ]
    if a.dump:
        cells.append(("real", (0, 0)))          # size read from the dump
    else:
        print("[t-traj] no --dump supplied: real-data arm skipped "
              "(generate with: fieldrun --bundle bundles/pythia-70m --recursion-explain "
              "--source-dump <out.jsonl> --n 220)")

    rows = []
    for regime, size in cells:
        for lr in a.lrs:
            for seed in a.seeds:
                row = run_cell(regime, size, lr, seed, a.steps, a.batch, a.dump)
                rows.append(row)
                print(fmt_cell(row), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(out.with_suffix(".txt"), "w") as f:
        f.write("T-traj discharge experiment 1 — per-cell premise hold-rates and slack\n")
        f.write("(theorem: i-orca examples/pic_learn/PIC_Learn.thy; all rows empirical)\n\n")
        for row in rows:
            f.write(fmt_cell(row) + "\n")
    print(f"wrote {out.with_suffix('.txt')} and .json")


if __name__ == "__main__":
    main()
