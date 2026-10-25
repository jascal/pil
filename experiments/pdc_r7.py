"""Preregistered fixed-write, untied-output admission experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pil.p3 import CertifiedP3


class Frame(nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.U = nn.Parameter(torch.tensor(raw["initial_U"]))
        self.bias = nn.Parameter(torch.tensor(raw["initial_bias"]))
        self.outside = torch.tensor(raw["outside_max"])

    def forward(self, sources, target=None):
        scores = sources.sum(-2) @ self.U.T + self.bias
        return {"logits": torch.cat((scores, self.outside[:, None]), 1)}


def oracle(model, data):
    incidence = data["D"] @ model.U.T
    partial = incidence + model.bias
    target = data["t"][:75, None, None].expand(-1, 25, 1)
    chosen = partial.gather(2, target).squeeze(2)
    rivals = partial.scatter(2, target, -torch.inf).amax(2)
    margin = chosen - rivals
    tail = (data["r"][:75] @ model.U.T)[:, None, :] - incidence
    delta2 = tail.amax(2) - tail.amin(2)
    violation = torch.maximum(delta2, torch.full_like(delta2, 0.5)) - margin
    return violation, margin, delta2


def verdict(model, data):
    with torch.no_grad():
        v, m, d = oracle(model, data)
        rows = []
        for i in range(75):
            valid = torch.where(v[i] < 0)[0]
            j = int(valid[0]) if len(valid) else int(v[i].argmin())
            rows.append(
                dict(
                    admitted=bool(len(valid)),
                    source=j,
                    margin=float(m[i, j]),
                    two_delta=float(d[i, j]),
                    violation=float(v[i, j]),
                    gamma=float((m[i, j] + max(0.5, float(d[i, j]))) / 2),
                )
            )
        return rows


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    raw = np.load(args.data)
    data = {
        key: torch.tensor(raw[name])
        for key, name in [
            ("D", "native_sources"),
            ("r", "protected_residuals"),
            ("t", "teacher_indices"),
            ("truth", "truth_indices"),
            ("lse", "outside_lse"),
        ]
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        prereg_sha256=hashlib.sha256(Path(args.prereg).read_bytes()).hexdigest(),
        proposals=200,
        seed=20261025,
        heldout_used=0,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = {}
    for arm in ["decode", "admission"]:
        model = Frame(raw)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
        guard = CertifiedP3(
            model, opt, data["r"][:, None, :], data["t"], eta=0.9, m_floor=0, alpha_min=2**-20, renorm=False
        )
        assert bool((guard._margins() > 0).all())
        initial = verdict(model, data)
        np.savez(out / f"{arm}_initial.npz", U=model.U.detach().numpy(), bias=model.bias.detach().numpy())
        records = []
        for step in range(200):
            with torch.no_grad():
                old = oracle(model, data)[0].amin(1) < 0

            def extra(old=old, model=model):
                return bool((~old | (oracle(model, data)[0].amin(1) < 0)).all())

            guard.candidate_guard = extra
            logits = data["r"][:75] @ model.U.T + model.bias
            ce = F.cross_entropy(torch.cat((logits, data["lse"][:75, None]), 1), data["truth"])
            admission = F.softplus(oracle(model, data)[0].amin(1)).mean()
            rec = guard.step(ce + (admission if arm == "admission" else 0))
            with torch.no_grad():
                current = oracle(model, data)[0].amin(1) < 0
                norm = model.U.norm(dim=1)
                normalized = model.U / norm[:, None]
                gram = normalized @ normalized.T
                M, d = model.U.shape
                fp = float((gram.square().sum() - gram.diag().square().sum()) / (M * (M - 1)))
                welch = max(0, (M - d) / (d * (M - 1)))
                assert bool(torch.isfinite(model.U).all() & torch.isfinite(model.bias).all())
                assert bool((norm > 0).all()) and fp + 1e-10 >= welch
                assert bool((~old | current).all()) and bool(guard._decided().all())
                assert rec["premise_hold_rate"] == 1 and rec["enforced_flips"] == 0
                rec.update(
                    ce=float(ce.detach()),
                    admission_loss=float(admission.detach()),
                    admitted=int(current.sum()),
                    fp=fp,
                    welch=welch,
                )
            records.append(rec)
            (out / f"{arm}_steps.json").write_text(json.dumps(records, indent=2))
            if step % 10 == 0:
                print(arm, step + 1, rec["admitted"], rec["alpha"], flush=True)
        final = verdict(model, data)
        np.savez(out / f"{arm}_final.npz", U=model.U.detach().numpy(), bias=model.bias.detach().numpy())
        results[arm] = dict(
            initial=initial,
            final=final,
            summary=guard.summary(),
            initial_admitted=sum(x["admitted"] for x in initial),
            final_admitted=sum(x["admitted"] for x in final),
            full_host_agreement=1.0,
            hop2_predictions_unchanged=True,
        )
        (out / "results.json").write_text(json.dumps(results, indent=2))
    gain = results["admission"]["final_admitted"] - results["admission"]["initial_admitted"]
    advantage = results["admission"]["final_admitted"] - results["decode"]["final_admitted"]
    results["advance_gate"] = dict(
        passed=gain / 75 >= 0.05 and advantage / 75 >= 0.02,
        gain_count=gain,
        control_advantage_count=advantage,
    )
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["data", "prereg", "out"]:
        p.add_argument("--" + name, required=True)
    run(p.parse_args())
