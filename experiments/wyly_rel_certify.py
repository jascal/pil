"""CERTIFY the learned relational rule: the hard-routed induction head == a 3-line Datalog program.

The decompilation loop's standard of proof (rosetta: a Soufflé query certifies cdecide == the model)
applied, for the first time, to a rule that was LEARNED by gradient in pil rather than mined from a
frozen transformer. Claim: the battery-trained WylyRel induction head, evaluated HARD (one-hot argmax
routing -- a gather, no floats in the decision), computes exactly

    hit(w, p)  :- tok(w, p, t), qtok(w, t), p < L-1.           // token-equality match
    best(w, m) :- qtok(w, _), m = max p : { hit(w, p) }.       // most recent site
    pred(w, t) :- best(w, p), p1 = p + 1, tok(w, p1, t).       // route the successor

over the induction test set. The certificate is EXACT AGREEMENT, window by window, between the
tensor hard-route argmax and Soufflé's pred relation -- tagged per the workspace discipline:
agreement 100% => `proved` over the stated domain (this test set); anything less is reported as the
exact count.

Also reported (empirical, no certificate): the Wyly-LM v2 REAL-TEXT relational path's hard decisions
vs the same Datalog rule on the copy-subset windows -- how much of the learned real-text head is
explained by the pure induction program. (Needs data/wyly_v2_state.pt from wyly_lm_v2.py.)

Requires `souffle` on PATH. Run: cd pil && .venv/bin/python experiments/wyly_rel_certify.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import torch
from wyly_rel_battery import DEV, STEPS, L, WylyRel, accuracy, gen_induction, train

DATA = Path(__file__).resolve().parent.parent / "data"

PROGRAM = """
.decl tok(w:number, p:number, t:number)
.input tok
.decl qtok(w:number, t:number)
.input qtok
.decl hit(w:number, p:number)
hit(w, p) :- tok(w, p, t), qtok(w, t), p < {maxp}.
.decl best(w:number, m:number)
best(w, m) :- qtok(w, _), m = max p : {{ hit(w, p) }}.
.decl pred(w:number, t:number)
pred(w, t) :- best(w, p), p1 = p + 1, tok(w, p1, t).
.output pred
"""


def souffle_pred(x: torch.Tensor, ell: int) -> dict[int, int]:
    """run the induction Datalog program over windows x -> {window: predicted token}."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "prog.dl").write_text(PROGRAM.format(maxp=ell - 1))
        xc = x.cpu()
        with open(td / "tok.facts", "w") as f:
            for w in range(len(xc)):
                row = xc[w].tolist()
                f.writelines(f"{w}\t{p}\t{t}\n" for p, t in enumerate(row))
        with open(td / "qtok.facts", "w") as f:
            f.writelines(f"{w}\t{int(xc[w, -1])}\n" for w in range(len(xc)))
        subprocess.run(["souffle", "-F", str(td), "-D", str(td), str(td / "prog.dl")],
                       check=True, capture_output=True)
        out = {}
        for line in (td / "pred.csv").read_text().splitlines():
            w, t = line.split("\t")
            out[int(w)] = int(t)
        return out


def hard_preds(model, x, bs=512, rel_only=False):
    with torch.no_grad():
        ps = []
        for i in range(0, len(x), bs):
            out = (model.relational(x[i:i + bs], hard=True) if rel_only
                   else model(x[i:i + bs], hard=True))
            ps.append(out.argmax(1))
        return torch.cat(ps)


def main():
    # ---- part 1: the certificate (battery induction head, exact domain) ----
    g = torch.Generator().manual_seed(0)
    xtr, ytr, vocab, _ = gen_induction(16384, g)
    xte, yte, _, _ = gen_induction(2048, g)
    xtr, ytr, xte, yte = xtr.to(DEV), ytr.to(DEV), xte.to(DEV), yte.to(DEV)
    torch.manual_seed(0)
    model = train(WylyRel(vocab).to(DEV), xtr, ytr, STEPS, "sgd", 1.0)
    soft, hard = accuracy(model, xte, yte), accuracy(model, xte, yte, hard=True)
    tensor_pred = hard_preds(model, xte)
    dl = souffle_pred(xte, L)
    dl_pred = torch.tensor([dl[w] for w in range(len(xte))], device=DEV)
    agree = int((tensor_pred == dl_pred).sum())
    dl_correct = float((dl_pred == yte).float().mean())
    print(f"battery induction head: soft {soft:.3f}, hard-route {hard:.3f}")
    print(f"Datalog program accuracy (sanity): {dl_correct:.3f}")
    print(f"CERTIFICATE: tensor-hard == Soufflé-pred on {agree}/{len(xte)} test windows "
          f"({agree / len(xte):.1%})")
    tag = "proved (exact agreement on the stated domain)" if agree == len(xte) else \
        f"empirical ({len(xte) - agree} disagreements -- NOT promoted to proved)"
    print(f"tag: {tag}\n")

    # ---- part 2: how much of the REAL-TEXT relational head is this program? (empirical) ----
    state = DATA / "wyly_v2_state.pt"
    if not state.exists():
        print("(real-text part skipped: run wyly_lm_v2.py first to produce wyly_v2_state.pt)")
        return
    import wyly_lm_v2 as v2
    from wyly_data import load_windows_tied
    ids, y, cls, uv, tr, te = load_windows_tied(v2.V, DEV, v2.DATA)
    m2 = v2.WylyV2(torch.zeros(len(uv), v2.K), cls.cpu(), ids.shape[1])
    m2.load_state_dict(torch.load(state, map_location="cpu"))
    m2 = m2.to(DEV)
    cs = v2.copy_subset(ids, y, cls, te)
    rel = hard_preds(m2, ids[cs], rel_only=True)             # class-space argmax
    rel_tok = cls[rel]                                       # -> compact-vocab token ids
    dl = souffle_pred(ids[cs], ids.shape[1])
    dl_pred = torch.tensor([dl[w] for w in range(len(cs))], device=DEV)
    agree = float((rel_tok == dl_pred).float().mean())
    dl_acc = float((dl_pred == cls[y[cs]]).float().mean())
    rel_acc = float((rel_tok == cls[y[cs]]).float().mean())
    print(f"real-text v2 relational path, copy-subset ({len(cs)} windows):")
    print(f"  Datalog induction program accuracy: {dl_acc:.3f}")
    print(f"  v2 relational-path (hard) accuracy: {rel_acc:.3f}")
    print(f"  agreement (empirical, no certificate): {agree:.3f}")
    print("read: agreement ~1 = the learned real-text head IS the induction program on this subset;")
    print("lower = it learned something else too (or the mix carries the decision).")


if __name__ == "__main__":
    main()
