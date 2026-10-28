"""Run the preregistered PDC R4 controls and fixed-size held-out source LPs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pil.pdc_admission import affine_basis, constraints, margin_admission, solve_affine


def sweep(writes, candidates, outdir, name, positions=None):
    residuals = writes.sum(axis=1)
    records = []
    for x0 in range(len(writes)) if positions is None else positions:
        basis = affine_basis(residuals, writes[x0])
        attempts = []
        witness = None
        for j in np.argsort(-np.linalg.norm(writes[x0], axis=1)):
            j = int(j)
            matrix, rhs, tokens = constraints(residuals, candidates, writes[x0], x0, [j], 1., basis)
            verdict = solve_affine(matrix, rhs)
            if verdict["status"] == "feasible":
                coefficients = verdict.pop("coefficients").reshape(len(tokens), basis.shape[1])
                physical = coefficients @ basis.T
                full_logits = np.column_stack([residuals, np.ones(len(residuals))]) @ physical.T
                faithful_margin = min(float(full_logits[n, tokens.index(int(c[0]))]
                    - full_logits[n, tokens.index(int(t))])
                    for n, c in enumerate(candidates) for t in c[1:])
                if faithful_margin < 1 - 1e-7:
                    raise RuntimeError("physical frame failed independent faithfulness check")
                # Margin oracle consumes rows indexed by local token index.
                local_cands = [tokens.index(int(t)) for t in candidates[x0]]
                admission = margin_admission(writes[x0], physical[:, :-1], physical[:, -1],
                    local_cands[0], [j], local_cands)
                if admission["admitted"] and not admission["full_strict_target"]:
                    raise RuntimeError("admission premises failed to preserve the full decision")
                path = outdir / f"{name}_frame_{x0:03d}.npz"
                np.savez_compressed(path, basis=basis, coefficients=coefficients,
                    tokens=tokens, coalition=[j], position=x0)
                witness = {"frame_path": str(path),
                    "frame_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "faithful_margin": faithful_margin, "source": j, "admission": admission}
            attempts.append({"source": j, **verdict})
            if witness is not None:
                break
        status = "reducible" if witness is not None else "irreducible" if all(
            a["status"] == "infeasible" for a in attempts) else "unknown"
        row = {"position": x0, "status": status, "attempts": attempts, "witness": witness}
        records.append(row)
        admission_status = None if witness is None else witness["admission"]["admitted"]
        print(f"{name} {x0}: {status}; admission={admission_status}", flush=True)
        (outdir / f"{name}_progress.json").write_text(json.dumps(records, indent=2) + "\n")
    return {"n": len(records), "reducible": sum(r["status"] == "reducible" for r in records),
        "irreducible": sum(r["status"] == "irreducible" for r in records),
        "unknown": sum(r["status"] == "unknown" for r in records),
        "witness_admitted": sum(r["witness"] is not None and r["witness"]["admission"]["admitted"]
            for r in records),
        "records": records}


def controls(outdir):
    positive = np.array([[[(-1 if n % 2 else 1)*(1+(n % 4)/4), 0], [0, (n % 7)/7]]
        for n in range(64)], dtype=float)
    candidates = [[n % 2, 1-n % 2] for n in range(64)]
    frame, bias = np.array([[1., 0.], [-1., 0.]]), np.zeros(2)
    native = [margin_admission(positive[n], frame, bias, n % 2, [0], [0, 1]) for n in range(64)]
    if not all(r["admitted"] for r in native):
        raise RuntimeError("positive admission control failed: apply R4 stop rule")
    np.savez_compressed(outdir / "positive_native.npz", writes=positive, frame=frame, bias=bias,
        candidates=candidates)
    pos_result = sweep(positive, candidates, outdir, "positive")
    pos_result["native_admitted"] = len(native)
    pos_result["native_admissions"] = native
    if pos_result["reducible"] != 64:
        raise RuntimeError("positive LP control failed: apply R4 stop rule")
    refusal = np.array([[[1., 0.], [0., 1.]], [[1., 0.], [0., 0.]], [[0., 1.], [0., 0.]]])
    refusal_cands = [[0, 1], [1, 0], [1, 0]]
    np.savez_compressed(outdir / "refusal_native.npz", writes=refusal,
        candidates=refusal_cands, frame=[[2., 2.], [0., 0.]], bias=[-3., 0.])
    neg_result = sweep(refusal, refusal_cands, outdir, "refusal", positions=[0])
    if neg_result["irreducible"] != 1 or not all(a.get("dual_exact_rational_check")
        for a in neg_result["records"][0]["attempts"]):
        raise RuntimeError("refusal control failed")
    return {"positive": pos_result, "refusal": neg_result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = controls(args.out)
    if args.dump is not None:
        raw = [json.loads(line) for line in args.dump.read_text().splitlines() if line.strip()]
        if len(raw) != 80 or any(r["cands"][0] != r["pred"] for r in raw):
            raise ValueError("expected exactly 80 records, each with decoded token first")
        writes = np.array([r["d"] for r in raw], dtype=np.float64)
        candidates = [r["cands"] for r in raw]
        report["heldout"] = sweep(writes, candidates, args.out, "heldout")
        report["dump_sha256"] = hashlib.sha256(args.dump.read_bytes()).hexdigest()
    prereg = Path("docs/notes/pdc_r4_prereg.md")
    report["prereg_sha256"] = hashlib.sha256(prereg.read_bytes()).hexdigest()
    (args.out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {k: v for k, v in values.items() if k not in ["records", "native_admissions"]}
        for name, values in report.items() if isinstance(values, dict)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
