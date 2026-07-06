"""CERTIFY the derived-extractor REGISTRY: every extractor kind the two-layer package ships is
Soufflé-checked against its tensor mirror, window by window, on real corpus samples.

  recent-member   most recent token in a declared member set        (max aggregate)
  recent-unique   most recent member occurring exactly once         (count + max aggregates)
  bracket-depth   the balance counter, clamped to [0, cap]          (sum aggregate + clamps)

(bracket-mate is certified separately in wyly_mate_certify.py: 256/256.)
100% agreement => `proved` over the stated domain; anything less reported exactly.
Requires souffle on PATH. Run: cd pil && .venv/bin/python experiments/wyly_derived_certify.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
os.environ.setdefault("WYLY_TAG", "pythia70m")
os.environ.setdefault("WYLY_DS", "isabelle")

P_RECENT = """
.decl tok(w:number, p:number, t:number)
.input tok
.decl member(t:number)
.input member
.decl hit(w:number, p:number)
hit(w, p) :- tok(w, p, t), member(t){uniq}.
.decl best(w:number, m:number)
best(w, m) :- hit(w, _), m = max p : {{ hit(w, p) }}.
.decl feat(w:number, t:number)
feat(w, t) :- best(w, p), p2 = p + {succ}, tok(w, p2, t).
.output feat
"""
UNIQ_CLAUSE = ", n = count : { tok(w, _, t) }, n = 1"

P_DEPTH = """
.decl tok(w:number, p:number, t:number)
.input tok
.decl opener(t:number)
.input opener
.decl closer(t:number)
.input closer
.decl delta(w:number, p:number, d:number)
delta(w, p, 1) :- tok(w, p, t), opener(t).
delta(w, p, -1) :- tok(w, p, t), closer(t).
.decl total(w:number, s:number)
total(w, s) :- tok(w, _, _), s = sum d : { delta(w, _, d) }.
total(w, 0) :- tok(w, _, _), !delta(w, _, _).
.decl feat(w:number, v:number)
feat(w, v) :- total(w, s), s >= 0, s <= CAP, v = s.
feat(w, 0) :- total(w, s), s < 0.
feat(w, CAP) :- total(w, s), s > CAP.
.output feat
"""


def run_souffle(prog, facts):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "prog.dl").write_text(prog)
        for name, rows in facts.items():
            (td / f"{name}.facts").write_text(
                "".join("\t".join(str(x) for x in r) + "\n" for r in rows))
        subprocess.run(["souffle", "-F", str(td), "-D", str(td), str(td / "prog.dl")],
                       check=True, capture_output=True)
        out = {}
        for line in (td / "feat.csv").read_text().splitlines():
            w, t = line.split("\t")
            out[int(w)] = int(t)
        return out


def check(name, tensor_vals, dl):
    n = len(tensor_vals)
    agree = sum(int(int(tensor_vals[i]) == dl.get(i, -1)) for i in range(n))
    print(f"  {name}: {agree}/{n} {'PROVED on this domain' if agree == n else 'MISMATCH'}")
    return agree == n


def main():
    import wyly_lm_v5 as v5

    from pil.tokens import TokenSpace
    ids, _y, _cls, uv, _tr, _te = v5.load_ds()
    ts = TokenSpace.from_file(REPO.parent / "rosetta" / "models" / "pythia70m"
                              / "bundle.tokenizer.json")
    vocab = len(uv)
    g = torch.Generator().manual_seed(0)
    w = ids[torch.randperm(len(ids), generator=g)[:192].to(ids.device)]
    tokf = [(wi, p, t) for wi in range(len(w)) for p, t in enumerate(w[wi].tolist())]

    cap_set = torch.zeros(vocab, dtype=torch.bool, device=ids.device)
    for i in range(vocab):
        d = ts.token_str(int(uv[i])).strip()
        if len(d) > 1 and d[0].isupper() and d.isalpha():
            cap_set[i] = True
    mem = [(int(i),) for i in torch.where(cap_set)[0].tolist()]
    ok = True
    tm = v5.recent_member_feature(w, cap_set)
    ok &= check("recent-member", tm.cpu().tolist(),
                run_souffle(P_RECENT.format(uniq="", succ=0), {"tok": tokf, "member": mem}))
    tu = v5.recent_member_feature(w, cap_set, unique=True)
    ok &= check("recent-unique", tu.cpu().tolist(),
                run_souffle(P_RECENT.format(uniq=UNIQ_CLAUSE, succ=0),
                            {"tok": tokf, "member": mem}))
    tsc = v5.at_pos(w, v5.recent_member_pos(w, cap_set), succ=1)
    ok &= check("recent-member succ=1 (ROLE COMPOSITION)", tsc.cpu().tolist(),
                run_souffle(P_RECENT.format(uniq="", succ=1), {"tok": tokf, "member": mem}))
    opl, cll = v5.bracket_sets(uv, ts)
    is_open = torch.zeros(vocab, dtype=torch.bool, device=ids.device)
    is_close = torch.zeros(vocab, dtype=torch.bool, device=ids.device)
    is_open[torch.tensor(opl, device=ids.device)] = True
    is_close[torch.tensor(cll, device=ids.device)] = True
    td = v5.depth_feature(w, is_open, is_close, cap=8)
    ok &= check("bracket-depth", td.cpu().tolist(),
                run_souffle(P_DEPTH.replace("CAP", "8"),
                            {"tok": tokf, "opener": [(t,) for t in opl],
                             "closer": [(t,) for t in cll]}))
    print("REGISTRY VERDICT:", "all PROVED" if ok else "NOT ALL PROVED -- do not promote")


if __name__ == "__main__":
    main()
