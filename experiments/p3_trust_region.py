"""Certified P3 sweep: trust-region on/off across regimes (discharge experiment 5 + S1/S2).

Per phase: the shipped raw objective (NLL + 0.5·ReLU(2 − m) + 0.05·FP — unchanged), AdamW +
row renorm, with the trust region (pil/p3.py: exact inviolability + certificate headroom)
either off (the PR #5 baseline, re-run under the fixed self-checks) or on. Measures:

  1. protection: enforced flips (must be 0 with TR on — asserted), certified fraction,
     alpha acceptance stats;
  2. does the T-traj BUDGET form come alive? (2Σδ vs m0 percentiles — vacuous in PR #5);
  3. the S1/S2 tension: margin-widening progress with TR on vs off (margin percentiles,
     normalized-margin certified-bits proxy, hinge, decided rate);
  4. T-mass data: certified mass at fixed δ_ref on the initial bank vs contexts added
     mid-phase, tracked per step (S2's empirical premise — not a theorem claim);
  5. teacher-seeded vs random-frame entry (real arm): does teacher geometry move hold-rates?

Real arm: pythia-70m --source-dump residuals; frame seed = the teacher's own unembedding
restricted to the dump vocab (HF embed_out; the dump does not carry U — noted for the
fieldrun-side backlog), row-normalized as P3 requires; t_x = the teacher decision at dump
time (cands[:,0]). No raw-residual loss terms anywhere.

Run:  python experiments/p3_trust_region.py --steps 400 \
        --dump <source_dump.jsonl> --teacher-frame <p3_teacher_frame.npz> \
        --out results/p3_trust_region
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from pil.certify import TrajectoryCertificate
from pil.learner import PILConfig, ProjectiveIncidenceLearner, create_synthetic_problem
from pil.p3 import CertifiedP3
from pil.synthetic import create_clustered_problem


def build_arm(arm: str, seed: int, dump: str | None, teacher_frame: str | None):
    """Returns (model, sources (B,J,dim), t_frozen, extra_sources, ho_sources, label)."""
    if arm in ("easy", "hard"):
        dim, V = (32, 64) if arm == "easy" else (24, 192)
        cfg = PILConfig(dim=dim, n_propositions=V, n_sources_per_step=24,
                        frame_reg=0.05, margin_weight=0.5, seed=seed, device="cpu")
        if arm == "easy":
            src, tgt, _ = create_synthetic_problem(cfg, n_examples=512)
        else:
            src, tgt, _, _ = create_clustered_problem(
                cfg, n_examples=512, n_clusters=V // 8, cluster_spread=0.10,
                sparsity=0.12, noise=0.08)
        model = ProjectiveIncidenceLearner(cfg)
        model.normalize_U()
        # visited bank 256 (t = planted target: the decision P3 must win then never lose),
        # +128 added mid-phase (T-mass), 128 held-out (measurement only)
        return model, src[:256], tgt[:256], (src[256:384], tgt[256:384]), \
            (src[384:], tgt[384:]), arm

    # real: teacher residuals; frame seed teacher or random
    from pil.fieldrun_io import load_source_dump

    sb = load_source_dump(dump)
    vocab = sorted(set(sb.cands.reshape(-1).tolist()) | set(sb.target.tolist()))
    idx = {t: i for i, t in enumerate(vocab)}
    r = torch.tensor(sb.r, dtype=torch.float32).unsqueeze(1)          # (N,1,dim)
    t_teacher = torch.tensor([idx[int(t)] for t in sb.cands[:, 0]])   # frozen teacher decisions
    cfg = PILConfig(dim=sb.dim, n_propositions=len(vocab), n_sources_per_step=1,
                    frame_reg=0.05, margin_weight=0.5, seed=seed, device="cpu")
    model = ProjectiveIncidenceLearner(cfg)
    if arm == "real_teacher":
        tf = np.load(teacher_frame)
        assert list(tf["vocab"]) == vocab, "teacher frame vocab mismatch"
        with torch.no_grad():
            model.U.copy_(torch.tensor(tf["U"], dtype=torch.float32))
    model.normalize_U()
    n = r.shape[0]
    g = torch.Generator().manual_seed(seed + 3)
    perm = torch.randperm(n, generator=g)
    a, b = int(n * 0.5), int(n * 0.75)                     # 50% bank, 25% added, 25% held-out
    return model, r[perm[:a]], t_teacher[perm[:a]], \
        (r[perm[a:b]], t_teacher[perm[a:b]]), (r[perm[b:]], t_teacher[perm[b:]]), arm


TR_VARIANTS = {
    "off": None,
    # strict: the spec's default — protect floor 0, exact check on everything
    "strict": dict(m_floor=0.0, exact_floor=None),
    # floored: the separately-flagged variant (spec: measure both, never silently pick) —
    # contexts with m <= 0.1 are excluded from BOTH checks; their flips counted+reported
    "floored": dict(m_floor=0.1, exact_floor=0.1),
}


def run_cell(arm, lr, seed, variant, steps, dump, teacher_frame, eta=0.9):
    model, src, t_frozen, (src_add, t_add), (src_ho, t_ho), label = build_arm(
        arm, seed, dump, teacher_frame)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    tr_on = variant != "off"
    p3 = (
        CertifiedP3(model, opt, src, t_frozen, eta=eta, **TR_VARIANTS[variant])
        if tr_on else None
    )
    # measurement certificate over bank + held-out (entry decisions = t_frozen / entry argmax)
    meas_bank = torch.cat([src, src_ho], dim=0)
    ho_mask = torch.zeros(meas_bank.shape[0], dtype=torch.bool)
    ho_mask[src.shape[0]:] = True
    cert = TrajectoryCertificate(model, meas_bank, ho_mask, torch.cat([t_frozen, t_ho]))

    mid = steps // 2
    delta_ref = None
    tmass = []                                             # (step, mass_g0, mass_g1)
    for step in range(steps):
        out = model.forward(src, target=t_frozen)          # objective on the visited bank
        loss = out["total_loss"]
        if tr_on:
            rec = p3.step(loss)
            assert rec["enforced_flips"] == 0
        else:
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.normalize_U()
        cert.step()
        if step == mid:
            if tr_on:
                p3.add_contexts(src_add, t_add)
            if delta_ref is None and tr_on:
                deltas = sorted(r["delta"] for r in p3.records)
                delta_ref = deltas[len(deltas) // 2]       # phase median accepted delta
        if tr_on and step >= mid and (step - mid) % 10 == 0:
            m = p3.certified_mass(delta_ref)
            tmass.append({"step": step, **{f"g{k}": v for k, v in m.items()}})

    with torch.no_grad():
        out = model.forward(src, target=t_frozen)
        m = out["margin_worst"]
        rn = src.sum(dim=-2).norm(dim=-1).clamp_min(1e-6)
        nm = (m / rn).sort().values
        fp = float(out["frame_potential"])
    row = {
        "arm": label, "lr": lr, "seed": seed, "trust_region": variant, "steps": steps,
        "hinge_final": float(torch.relu(2.0 - m).mean()),
        "margin_p10": float(m.sort().values[int(0.10 * (len(m) - 1))]),
        "margin_p50": float(m.median()),
        "nm_p10": float(nm[int(0.10 * (len(nm) - 1))]),
        "nm_p50": float(nm[int(0.50 * (len(nm) - 1))]),
        "decided_rate_bank": float((out["logits"].argmax(-1) == t_frozen).float().mean()),
        "frame_potential": fp,
        "measure": cert.summary(),
    }
    if tr_on:
        row["p3"] = p3.summary()
        row["tmass"] = tmass
        # T-mass verdict data: did old-support (g0) certified mass ever decrease?
        g0 = [t["g0"] for t in tmass]
        row["g0_decreases"] = sum(1 for i in range(1, len(g0)) if g0[i] < g0[i - 1])
    return row


def fmt(row: dict) -> str:
    m = row["measure"]
    base = (f"  {row['arm']:<13} lr={row['lr']:<7} seed={row['seed']} TR={row['trust_region']:<7}"
            f" | m_p10/p50 {row['margin_p10']:+.3f}/{row['margin_p50']:+.3f}"
            f" | nm_p10 {row['nm_p10']:+.4f}"
            f" | hinge {row['hinge_final']:.3f}"
            f" | decided {row['decided_rate_bank']:.3f}"
            f" | hold {m['premise_hold_rate_mean']:.3f}"
            f" | flip-events {m['total_flip_events']}")
    if row["trust_region"] != "off":
        p = row["p3"]
        alive = (f"pos-alive p10={p['budget_alive_pos_p10']} p50={p['budget_alive_pos_p50']} "
                 f"(n={p['m0_pos_n']})")
        base += (f"\n{'':16}TR: flips {p['enforced_flips_total']}"
                 f" excl-flips {p['flips_on_excluded_total']}"
                 f" | clip {p['clip_frac']:.2f}"
                 f" a_mean {p['alpha_mean']:.3f} a_min {p['alpha_min_seen']:.4f}"
                 f" | cert {p['certified_frac_mean']:.3f}->{p['certified_frac_final']:.3f}"
                 f" | 2Σδ {p['budget_2sum']:.3f} vs m0_pos p10/p50 "
                 f"{p.get('m0_pos_p10', float('nan')):.3f}/{p.get('m0_pos_p50', float('nan')):.3f}"
                 f" ({alive})"
                 f" | g0-decreases {row['g0_decreases']}")
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-3, 3e-4])
    ap.add_argument("--dump", default=None)
    ap.add_argument("--teacher-frame", default=None)
    ap.add_argument("--out", default="results/p3_trust_region")
    a = ap.parse_args()

    arms = ["easy", "hard"]
    if a.dump and a.teacher_frame:
        arms += ["real_teacher", "real_random"]
    else:
        print("[p3] no --dump/--teacher-frame: real arms skipped")

    rows = []
    for arm in arms:
        for lr in a.lrs:
            for seed in a.seeds:
                for variant in TR_VARIANTS:
                    row = run_cell(arm, lr, seed, variant, a.steps, a.dump, a.teacher_frame)
                    rows.append(row)
                    print(fmt(row), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(out.with_suffix(".txt"), "w") as f:
        f.write("Certified P3 trust-region sweep (theorem: i-orca examples/pic_learn, "
                "main 735973f; follow-up to pil#5)\nAll rows empirical.\n\n")
        for row in rows:
            f.write(fmt(row) + "\n")
    print(f"wrote {out.with_suffix('.txt')} and .json")

    # certified-bits payoff (S1/S2 tension), each TR variant paired with off at matched cell
    print("\n[p3] certified-bits proxy delta (TR variant vs off), per cell:")
    by_key = {(r["arm"], r["lr"], r["seed"], r["trust_region"]): r for r in rows}
    for (arm, lr, seed, tr), r in sorted(by_key.items(), key=str):
        if tr != "off":
            off = by_key.get((arm, lr, seed, "off"))
            if off and off["nm_p10"] > 0 and r["nm_p10"] > 0:
                dbits = math.log2(r["nm_p10"] / off["nm_p10"])
                print(f"  {arm:<13} lr={lr:<7} seed={seed} {tr:<7}: dbits(nm_p10) = {dbits:+.2f}")
            else:
                print(f"  {arm:<13} lr={lr:<7} seed={seed} {tr:<7}: nm_p10 <= 0 in a leg (silent)")


if __name__ == "__main__":
    main()
