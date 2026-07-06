"""CERTIFY the derived predicate behind the two-layer package (15-full): the mate extractor.

Claim: the tensor `mate_feature` (prefix-depth + reverse-cummin, wyly_lm_v5) computes exactly the
Datalog program below -- the innermost UNCLOSED opener: an opener at p is unclosed iff no later
prefix-depth drops below its own; the mate is the max such p, and the feature is its token.

    delta(w,p,+1|-1|0)   from opener/closer membership (extensional, shipped in the package)
    pref(w,p,s)          recursive prefix sum
    viol(w,p)            exists q >= p with pref(w,q) < pref(w,p)
    unclosed(w,p)        opener(p) and not viol(p)          -- stratified negation
    mate(w,m)            m = max p : unclosed                -- Soufflé aggregate
    matetok(w,t)         the derived feature value

The certificate is EXACT AGREEMENT window-by-window (including the no-mate/-1 case) on a sample
of the isabelle dataset -- the corpus whose judge admitted the mate gate. 100% => `proved` over
the stated domain; anything less reported as the exact count. Requires souffle on PATH.

Run: cd pil && .venv/bin/python experiments/wyly_mate_certify.py
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

PROGRAM = """
.decl tok(w:number, p:number, t:number)
.input tok
.decl opener(t:number)
.input opener
.decl closer(t:number)
.input closer
.decl delta(w:number, p:number, d:number)
delta(w, p, 1) :- tok(w, p, t), opener(t).
delta(w, p, -1) :- tok(w, p, t), closer(t).
delta(w, p, 0) :- tok(w, p, t), !opener(t), !closer(t).
.decl pref(w:number, p:number, s:number)
pref(w, 0, d) :- delta(w, 0, d).
pref(w, p, s) :- pref(w, p1, s1), p = p1 + 1, delta(w, p, d), s = s1 + d.
.decl viol(w:number, p:number)
viol(w, p) :- tok(w, p, t), opener(t), pref(w, p, s), pref(w, q, s2), q >= p, s2 < s.
.decl unclosed(w:number, p:number)
unclosed(w, p) :- tok(w, p, t), opener(t), !viol(w, p).
.decl mate(w:number, m:number)
mate(w, m) :- unclosed(w, _), m = max p : { unclosed(w, p) }.
.decl matetok(w:number, t:number)
matetok(w, t) :- mate(w, p), tok(w, p, t).
.output matetok
"""


def souffle_mate(x: torch.Tensor, openers: list[int], closers: list[int]) -> dict[int, int]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "prog.dl").write_text(PROGRAM)
        xc = x.cpu()
        with open(td / "tok.facts", "w") as f:
            for w in range(len(xc)):
                f.writelines(f"{w}\t{p}\t{t}\n" for p, t in enumerate(xc[w].tolist()))
        (td / "opener.facts").write_text("".join(f"{t}\n" for t in openers))
        (td / "closer.facts").write_text("".join(f"{t}\n" for t in closers))
        subprocess.run(["souffle", "-F", str(td), "-D", str(td), str(td / "prog.dl")],
                       check=True, capture_output=True)
        out = {}
        for line in (td / "matetok.csv").read_text().splitlines():
            w, t = line.split("\t")
            out[int(w)] = int(t)
        return out


def main():
    import wyly_lm_v5 as v5

    from pil.tokens import TokenSpace
    ids, _y, _cls, uv, _tr, _te = v5.load_ds()
    ts = TokenSpace.from_file(REPO.parent / "rosetta" / "models" / "pythia70m"
                              / "bundle.tokenizer.json")
    opl, cll = v5.bracket_sets(uv, ts)
    vocab = len(uv)
    is_open = torch.zeros(vocab, dtype=torch.bool, device=ids.device)
    is_close = torch.zeros(vocab, dtype=torch.bool, device=ids.device)
    is_open[torch.tensor(opl, device=ids.device)] = True
    is_close[torch.tensor(cll, device=ids.device)] = True

    g = torch.Generator().manual_seed(0)
    smp = torch.randperm(len(ids), generator=g)[:256].to(ids.device)
    w = ids[smp]
    tensor_mate = v5.mate_feature(w, is_open, is_close).cpu()
    dl = souffle_mate(w, opl, cll)
    agree = 0
    for i in range(len(w)):
        t = int(tensor_mate[i])
        d = dl.get(i, -1)
        agree += int(t == d)
    n = len(w)
    print(f"mate extractor: tensor vs Datalog agree {agree}/{n} on {n} isabelle windows "
          f"({len(opl)} openers, {len(cll)} closers)")
    print("VERDICT:", "PROVED on this domain" if agree == n else "MISMATCH -- do not promote")


if __name__ == "__main__":
    main()
