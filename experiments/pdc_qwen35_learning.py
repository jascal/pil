"""Fixed-budget guarded Qwen output-frame learning; consumes data artifacts only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pil.p3 import CertifiedP3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class Frame(nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.U = nn.Parameter(torch.tensor(raw["initial_U"]))
        self.bias = nn.Parameter(torch.tensor(raw["initial_bias"]))
        self.register_buffer("outside", torch.tensor(raw["outside_max"]))

    def forward(self, sources, target=None):
        L = sources.sum(-2) @ self.U.T + self.bias
        return {"logits": torch.cat((L, self.outside[:, None]), dim=1)}


def oracle(model, D, r, t):
    incidence = D @ model.U.T
    partial = incidence + model.bias
    index = t[:, None, None].expand(-1, D.shape[1], 1)
    chosen = partial.gather(2, index).squeeze(2)
    rivals = partial.scatter(2, index, -torch.inf).amax(2)
    margin = chosen - rivals
    tail = (r @ model.U.T)[:, None, :] - incidence
    delta2 = tail.amax(2) - tail.amin(2)
    violation = torch.maximum(delta2, torch.full_like(delta2, 0.5)) - margin
    return violation, margin, delta2


@torch.no_grad()
def geometry(model, outside_gram, outside_active, vocab):
    M, d = model.U.shape
    norm = model.U.norm(dim=1)
    assert bool(torch.isfinite(model.U).all() & torch.isfinite(model.bias).all())
    nonzero = norm > 0
    normalized = model.U / torch.where(nonzero, norm, torch.ones_like(norm))[:, None]
    pool_gram = normalized.T @ normalized
    full_gram = outside_gram + pool_gram
    active = outside_active + int(nonzero.sum())
    off = float(full_gram.square().sum()) - active
    fp_token = off / (vocab * (vocab - 1))
    fp_active = off / (active * (active - 1))
    W_token = max(0, (vocab - d) / (d * (vocab - 1)))
    W_active = max(0, (active - d) / (d * (active - 1)))
    pool_fp = (float(pool_gram.square().sum()) - int(nonzero.sum())) / (M * (M - 1))
    W_pool = max(0, (M - d) / (d * (M - 1)))
    assert fp_token + 1e-10 >= W_token and fp_active + 1e-10 >= W_active
    return dict(
        fp_token=fp_token,
        welch_token=W_token,
        fp_active=fp_active,
        welch_active=W_active,
        pool_fp=pool_fp,
        pool_welch=W_pool,
        active_directions=active,
        updated_zero_rows=int((~nonzero).sum()),
        normalization="exact cosine for nonzero rows; zero rows remain zero",
    )


@torch.no_grad()
def verdicts(v, m, d):
    rows = []
    for i in range(len(v)):
        good = torch.where(v[i] < 0)[0]
        j = int(good[0]) if len(good) else int(v[i].argmin())
        admitted = bool(len(good))
        rows.append(
            dict(
                admitted=admitted,
                source=j,
                margin=float(m[i, j]),
                two_delta=float(d[i, j]),
                violation=float(v[i, j]),
                gamma=float((m[i, j] + max(0.5, float(d[i, j]))) / 2) if admitted else None,
            )
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "data-report", "data-audit", "prereg", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    out = a.out
    assert not out.exists(), "one frozen run; refuse output overwrite"
    report = json.loads(a.data_report.read_text())
    audit = json.loads(a.data_audit.read_text())
    data_hash = sha(a.data)
    assert report["gate"] == audit["gate"] == "passed"
    assert report["npz_sha256"] == audit["data_sha256"] == data_hash
    out.mkdir(parents=True)
    manifest = dict(
        schema="pdc.qwen35-frame-learning-run.v1",
        sha256={
            str(p): sha(p)
            for p in [a.data, a.data_report, a.data_audit, a.prereg, Path(__file__), Path("pil/p3.py")]
        },
        proposals_per_arm=200,
        seed=20261027,
        torch_threads=4,
        input_embedding_and_backbone="fixed",
        fresh_confirmation=False,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    raw = np.load(a.data)
    D = torch.tensor(raw["native_sources"])
    r = torch.tensor(raw["protected_residuals"])
    teacher = torch.tensor(raw["teacher_indices"])
    truth = torch.tensor(raw["truth_indices"])
    outside_lse = torch.tensor(raw["outside_lse"])
    outside_gram = torch.tensor(raw["outside_unit_gram"])
    outside_active = int(raw["outside_active_count"])
    vocab = int(raw["vocabulary_size"])
    assert D.shape == (100, 65, 2560) and r.shape == (208, 2560)
    assert len(raw["pool_tokens"]) >= 2261
    results = {}
    began = time.perf_counter()
    for arm in ["decode", "admission"]:
        model = Frame(raw)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
        guard = CertifiedP3(
            model,
            optimizer,
            r[:, None, :],
            teacher,
            eta=0.9,
            m_floor=0,
            exact_floor=None,
            alpha_min=2**-20,
            renorm=False,
        )
        assert bool((guard._margins() > 0).all())
        with torch.no_grad():
            previous = oracle(model, D, r[:100], teacher[:100])
        initial = verdicts(*previous)
        old_mask = previous[0].amin(1) < 0
        original_nonzero = model.U.detach().norm(dim=1) > 0
        stats = geometry(model, outside_gram, outside_active, vocab)
        np.savez(out / f"{arm}_initial.npz", U=model.U.detach().numpy(), bias=model.bias.detach().numpy())
        records = []
        for step in range(200):
            # Check only already-admitted contexts; the empty-set premise is vacuous.
            def extra(old_mask=old_mask, model=model):
                if not bool(old_mask.any()):
                    return True
                values = oracle(model, D[old_mask], r[:100][old_mask], teacher[:100][old_mask])[0]
                return bool((values.amin(1) < 0).all())

            guard.candidate_guard = extra
            logits = r[:100] @ model.U.T + model.bias
            ce = F.cross_entropy(torch.cat((logits, outside_lse[:, None]), 1), truth)
            if arm == "admission":
                penalty = F.softplus(oracle(model, D, r[:100], teacher[:100])[0].amin(1)).mean()
                loss = ce + penalty
                penalty_value = float(penalty.detach())
            else:
                penalty_value = float(F.softplus(previous[0].amin(1)).mean())
                loss = ce
            rec = guard.step(loss)
            with torch.no_grad():
                if rec["alpha"] > 0:
                    current = oracle(model, D, r[:100], teacher[:100])
                    stats = geometry(model, outside_gram, outside_active, vocab)
                else:
                    current = previous
                now = current[0].amin(1) < 0
                assert bool((~old_mask | now).all())
                assert bool(guard._decided().all()) and bool((guard._margins() > 0).all())
                assert (
                    rec["protect_n"] == 208 and rec["premise_hold_rate"] == 1 and rec["enforced_flips"] == 0
                )
                assert bool((model.U.norm(dim=1)[original_nonzero] > 0).all())
                rec.update(
                    ce_pre=float(ce.detach()),
                    admission_penalty_pre=penalty_value,
                    admitted=int(now.sum()),
                    admission_mask=now.tolist(),
                    geometry=stats,
                    protected_native_matches=208,
                )
            records.append(rec)
            (out / f"{arm}_steps.json").write_text(json.dumps(records, indent=2) + "\n")
            previous = current
            old_mask = now
            if step % 10 == 0 or step == 199:
                print(
                    arm,
                    step + 1,
                    "admitted",
                    rec["admitted"],
                    "alpha",
                    rec["alpha"],
                    "elapsed_s",
                    round(time.perf_counter() - began, 1),
                    flush=True,
                )
        final = verdicts(*previous)
        predictions = model.forward(r[:, None, :])["logits"].argmax(1).detach().numpy()
        predicted_tokens = raw["pool_tokens"][predictions]
        native_tokens = raw["native_teacher_tokens"]
        assert np.array_equal(predicted_tokens, native_tokens)
        hop2 = {}
        for tag in ["rederive", "crystallize", "generated", "tool"]:
            mask = raw["protected_arms"] == tag
            correct = int((predicted_tokens[mask] == raw["protected_truth_tokens"][mask]).sum())
            hop2[tag] = dict(cases=int(mask.sum()), correct=correct, baseline_unchanged=True)
        np.savez(out / f"{arm}_final.npz", U=model.U.detach().numpy(), bias=model.bias.detach().numpy())
        np.savez(
            out / f"{arm}_oracle.npz",
            violation=previous[0].numpy(),
            margin=previous[1].numpy(),
            two_delta=previous[2].numpy(),
        )
        results[arm] = dict(
            initial=initial,
            final=final,
            initial_admitted=sum(x["admitted"] for x in initial),
            final_admitted=sum(x["admitted"] for x in final),
            summary=guard.summary(),
            geometry=stats,
            protected_native_matches=208,
            hop2_host=hop2,
        )
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    rise = results["admission"]["final_admitted"] - results["admission"]["initial_admitted"]
    contrast = results["admission"]["final_admitted"] - results["decode"]["final_admitted"]
    results["learning_gate"] = dict(
        passed=rise >= 5 and contrast >= 2,
        gain_count=rise,
        control_advantage_count=contrast,
        final_status="requires_independent_endpoint_and_fieldrun_hop2_chart_audit",
    )
    results["elapsed_seconds"] = time.perf_counter() - began
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
