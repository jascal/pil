"""Train the certificate-aware TPR on a rosetta compiled-TPR dataset and export it as weighted Datalog facts.

Pre-registered in rosetta docs/compiled-tpr-prereg.md.
Reads <rosetta_out>/<TASK>/dataset.json (token ids + split),
runs Hugging Face GPT-2 on the TRAIN split only, trains the `cert`-objective TPR (d_F = 32, fit seed 0;
certified_substitutes.py), and writes <rosetta_out>/<TASK>/tpr/:

    filler.facts   token_id<TAB>filler_index
    w.facts        v_token<TAB>filler<TAB>role<TAB>round(2^16 * <U_v, W (e_F (x) e_R)>)
    bias.facts     v_token<TAB>round(2^16 * <U_v, b>)
    meta.json      candidates, train_pairs_seen, provenance

    /path/to/venv-with-transformers/bin/python experiments/compile_tpr.py /path/to/rosetta_out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_models  # noqa: E402
from certified_substitutes import TPR, train  # noqa: E402

SCALE = 2**16
D_F, FIT_SEED = 32, 0


def pairs_of(task, ctx, fidx, offset=0):
    """Python twin of rosetta's Datalog role parse: (filler_index, role). `offset` = 1 skips a BOS."""
    if task == "SVO":
        return [(fidx[ctx[1 + offset]], 0), (fidx[ctx[2 + offset]], 1), (fidx[ctx[4 + offset]], 2)]
    n = len(ctx)
    return [(fidx[t], n - 1 - p) for p, t in enumerate(ctx) if p >= offset]


def forward_all(rows, model, device, batch):
    """Decode inputs and argmax decisions for every row (fp32, pinned revision)."""
    us, decisions = [], []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            chunk = rows[s : s + batch]
            length = max(len(x["ctx"]) for x in chunk)
            ids = torch.zeros(len(chunk), length, dtype=torch.long)
            att = torch.zeros(len(chunk), length, dtype=torch.long)
            for i, x in enumerate(chunk):
                ids[i, : len(x["ctx"])] = torch.tensor(x["ctx"])
                att[i, : len(x["ctx"])] = 1
            u = hf_models.decode_inputs(model, ids.to(device), att.to(device))
            us.append(u)
            decisions.append((u @ hf_models.unembedding(model).T).argmax(-1))
    return torch.cat(us), torch.cat(decisions)


def compile_task(task_dir: Path, model, device, batch=256):
    data = json.loads((task_dir / "dataset.json").read_text())
    task, meta = data["task"], data["meta"]
    offset = int(meta.get("offset", 0))
    model.to(device)
    all_u, all_dec = forward_all(data["rows"], model, device, batch)
    U = hf_models.unembedding(model).cpu()
    model.to("cpu")  # free the GPU for training (fp32 1.5B models do not fit alongside the trainer)
    torch.cuda.empty_cache()
    U = U.to(device)
    # HF fp32 references for EVERY row (train/dev/test and the frozen parity set): rosetta's fallback
    # source and the parity comparison against fieldrun.
    (task_dir / "hf_refs.json").write_text(
        json.dumps({str(r["id"]): int(t) for r, t in zip(data["rows"], all_dec.tolist(), strict=True)}) + "\n"
    )
    is_train = torch.tensor([r["part"] == "train" for r in data["rows"]], device=all_u.device)
    train_rows = [r for r in data["rows"] if r["part"] == "train"]
    fillers = meta["fillers"]
    fidx = {t: i for i, t in enumerate(fillers)}
    n_fill, n_role = len(fillers), meta["n_role"]
    pairs = [pairs_of(task, r["ctx"], fidx, offset) for r in train_rows]
    width = max(len(p) for p in pairs)
    f = torch.zeros(len(pairs), width, dtype=torch.long)
    r = torch.zeros_like(f)
    m = torch.zeros(len(pairs), width)
    for i, ps in enumerate(pairs):
        for j, (a, s) in enumerate(ps):
            f[i, j], r[i, j], m[i, j] = a, s, 1
    u, dec = all_u[is_train], all_dec[is_train]
    tpr = TPR(n_fill, n_role, D_F, U.shape[1]).to(device)
    train(tpr, f.to(device), r.to(device), m.to(device), u, U, "cert", FIT_SEED)
    candidates = sorted(set(fillers) | set(dec.tolist()))
    C = torch.tensor(candidates, device=device)
    with torch.no_grad():
        ef, er = tpr.ef.weight, tpr.er.weight  # (n_fill, d_F), (n_role, n_role)
        bound = torch.einsum("fi,rj->frij", ef, er).reshape(n_fill * n_role, -1)
        P = bound @ tpr.W.weight.T  # (n_fill*n_role, hidden): W (e_F (x) e_R)
        w = torch.round(SCALE * (U[C] @ P.T)).long().cpu().numpy()  # (|C|, n_fill*n_role)
        bias = torch.round(SCALE * (U[C] @ tpr.W.bias)).long().cpu().numpy()
    out = task_dir / "tpr"
    out.mkdir(exist_ok=True)
    (out / "filler.facts").write_text("".join(f"{t}\t{i}\n" for i, t in enumerate(fillers)))
    with (out / "w.facts").open("w") as fh:
        for ci, v in enumerate(candidates):
            row = w[ci]
            for k in range(n_fill * n_role):
                fh.write(f"{v}\t{k // n_role}\t{k % n_role}\t{row[k]}\n")
    (out / "bias.facts").write_text(
        "".join(f"{v}\t{b}\n" for v, b in zip(candidates, bias.tolist(), strict=True))
    )
    seen = sorted({p for ps in pairs for p in ps})
    meta_out = dict(
        task=task,
        model=getattr(model, "name_or_path", None) or model.config._name_or_path,
        offset=offset,
        scale=SCALE,
        d_f=D_F,
        fit_seed=FIT_SEED,
        objective="cert",
        candidates=candidates,
        train_pairs_seen=seen,
        n_train=len(train_rows),
        hf_train_decisions_in_fillers=float(np.isin(dec.cpu().numpy(), fillers).mean()),
        dataset_sha256=hashlib.sha256((task_dir / "dataset.json").read_bytes()).hexdigest(),
    )
    (out / "meta.json").write_text(json.dumps(meta_out) + "\n")
    print(
        f"{task}: train {len(train_rows)}, candidates {len(candidates)}, "
        f"pairs seen {len(seen)}/{n_fill * n_role}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rosetta_out")
    ap.add_argument("--tasks", nargs="+", default=["SVO", "COPY"])
    ap.add_argument("--model", default="gpt2", choices=sorted(hf_models.REVISIONS))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="bug check: 30 training steps; NOT results")
    args = ap.parse_args()
    if args.smoke:
        import certified_substitutes

        certified_substitutes.STEPS = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, model = hf_models.load(args.model, device)
    for task in args.tasks:
        compile_task(Path(args.rosetta_out) / task, model, device, args.batch)


if __name__ == "__main__":
    main()
