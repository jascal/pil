"""Teacher-seeded P3 widening on a real frame (pythia-70m), GPU: renorm vs renorm-free.

The real-arm diagnosis (CertifiedP3 review) found that on a teacher-seeded frame ``F₀ = model U``,
pil's row-renorm convention destroys the teacher's entry margins — the frame's row *norms* carry its
margin structure, so projecting them to the unit sphere flips decisions, drives ``m0`` negative, and
empties the protect set, leaving widening nothing to hold. The untested first-order fix: skip the
renorm (``CertifiedP3(renorm=False)``). This experiment tests it head-to-head.

Frame = pythia-70m's real unembedding U (50304×512); contexts = the model's own residuals r at 1500
natural-text positions; decisions frozen at the model's argmax. Reports, for each mode:
  - entry margins m0 (median, frac>0, protect_n) — the diagnosis' failure signature;
  - a short widening phase — does m_p50 rise with zero enforced flips (inviolable + effective)?

Data: scratchpad/p3_pythia_frame.pt (built in tropic venv). Usage: python experiments/p3_teacher_seeded.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from pil.geometry import margin_to_worst, normalize_rows
from pil.p3 import CertifiedP3

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")


class FrameModel(nn.Module):
    """Minimal decode frame: logits = (Σ_J sources) @ Uᵀ + bias — the CertifiedP3 model interface."""

    def __init__(self, U: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.U = nn.Parameter(U.clone())
        b0 = bias.clone() if bias is not None else torch.zeros(U.shape[0], device=U.device)
        self.bias = nn.Parameter(b0)

    def normalize_U(self) -> None:
        with torch.no_grad():
            self.U.data = normalize_rows(self.U.data)

    def forward(self, sources, target=None):
        return {"logits": sources.sum(dim=-2) @ self.U.T + self.bias}


def run(U, r, dec, renorm, steps, device, m_floor=0.0, exact_floor=None, m_target=3.0, lr=1e-3):
    model = FrameModel(U.to(device), None).to(device)
    if renorm:
        model.normalize_U()                        # pil convention applied at entry (the failure mode)
    sources = r.to(device).unsqueeze(1)            # (N, 1, d)
    t = dec.to(device)
    opt = torch.optim.AdamW([model.U, model.bias], lr=lr)
    p3 = CertifiedP3(model, opt, sources, t, eta=0.9, m_floor=m_floor, exact_floor=exact_floor,
                     renorm=renorm)
    m0 = p3.m0.clone()
    traj = []
    flips = 0
    for step in range(steps):
        L = model.forward(sources)["logits"]
        m = margin_to_worst(L, t)
        loss = F.softplus(m_target - m).mean()     # push margins up toward m_target
        rec = p3.step(loss)
        flips += rec["enforced_flips"]
        if step in (0, steps // 2, steps - 1):
            traj.append((step, rec["margin_p50"], rec["certified_frac"], rec["protect_n"], rec["alpha"]))
    return m0, traj, flips, p3.records[-1].get("flips_on_excluded_total", 0)


def summarize_m0(m0):
    return (float(m0.median()), float((m0 > 0).float().mean()), int((m0 > 0).sum()), float(m0.min()))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(SP / "p3_pythia_frame.pt")
    U, r, dec = d["U"], d["r"], d["dec"]
    print(f"device={device}  frame U {tuple(U.shape)}  contexts N={r.shape[0]}")

    configs = [
        ("renorm (pil convention)", True, 0.0, None),
        ("renorm-FREE, m_floor=0 (entry fix only)", False, 0.0, None),
        ("renorm-FREE + m_floor=0.5 (certify robust)", False, 0.5, None),
        ("renorm-FREE + m_floor=0.5, exact_floor=0.2 (widen robust, drop fragile)", False, 0.5, 0.2),
    ]
    for tag, renorm, m_floor, exact_floor in configs:
        m0, traj, flips, excl = run(U, r, dec, renorm, steps=200, device=device,
                                    m_floor=m_floor, exact_floor=exact_floor)
        med, frac_pos, protect_n, mn = summarize_m0(m0)
        print(f"\n=== {tag} ===")
        print(f"entry m0:  median {med:+.3f}   frac>0 {frac_pos:.2f}   protect_n {protect_n}/{r.shape[0]}   "
              f"min {mn:+.2f}")
        print(f"{'step':>6}{'m_p50':>9}{'cert_frac':>11}{'protect_n':>11}{'alpha':>8}")
        for step, mp50, cf, pn, al in traj:
            print(f"{step:>6}{mp50:>9.3f}{cf:>11.3f}{pn:>11}{al:>8.4f}")
        print(f"enforced flips: {flips} (must be 0)   fragile flips allowed+counted (exact_floor): {excl}")


if __name__ == "__main__":
    main()
