"""The keystone measurement for 'realize the theorem in serving': replay the held-out test
windows through the PACKAGE RUNTIMES and check the served expert hits the numbers the learner
measured inside core_cover_sw.

Three comparisons per dataset:
  1. python reference runtime (rosetta serve_package.decide) over ALL test windows, token-level
     -- agreement vs teacher decisions must match the run's saved core_sw (exact semantics port);
  2. the C++ spoke (sgiandubh rosetta_package.h, same decide dispatch) checked for per-window
     PARITY with the python runtime on a round-trip-stable subset via HTTP;
  3. abstention rate (cover) must match too -- the bound is part of the semantics.

Usage: WYLY_DS=wikitext .venv/bin/python experiments/wyly_serve_eval.py   (or WYLY_DS=code)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

os.environ.setdefault("WYLY_TAG", "pythia70m")
os.environ.setdefault("WYLY_LIB", "mined")
os.environ.setdefault("WYLY_JUDGE", "cover")
os.environ.setdefault("WYLY_ONLINE", "1")
os.environ.setdefault("WYLY_COVER", "sw")

import wyly_lm_v5 as v5  # noqa: E402  (env must be set first)
from serve_package import decide, load_package  # noqa: E402

DS = os.environ.get("WYLY_DS", "wikitext")
PKG = REPO / "data" / ("wyly_expert_package_v5" if DS == "wikitext"
                       else f"wyly_expert_package_v5_{DS}")
SUF = os.environ.get("WYLY_EVAL_SUFFIX", "_cov_ol_sw")
STATE = REPO / "data" / f"wyly_v5_mined_pythia70m_{DS}{SUF}.pt"


def main():
    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    ref = torch.load(STATE, map_location="cpu")["core_sw"]
    idioms, ngrams, m = load_package(PKG / "manifest.json")
    assert m.get("cover") == "support-weighted", "package must declare the sw cover"
    uvl = uv.tolist()

    t0 = time.time()
    n_agree = n_fired = 0
    py_answers = []
    te_l = te.tolist()
    for wi in te_l:
        ctx = [uvl[c] for c in ids[wi].tolist()]
        d = decide(ctx, idioms, ngrams, m)
        ans = d["answer"] if d else -1
        py_answers.append(ans)
        if d:
            n_fired += 1
            if ans == uvl[yv[wi]]:
                n_agree += 1
    n = len(te_l)
    agree, cover = n_agree / n, n_fired / n
    print(f"[{DS}] SERVED (python runtime, {n} test windows, {time.time() - t0:.0f}s): "
          f"agree {agree:.3f} @ {cover:.1%}")
    print(f"[{DS}] LEARNER core_sw (saved state):                    "
          f"agree {ref['agree']:.3f} @ {ref['cover']:.1%}")
    gap = agree - ref["agree"]
    print(f"[{DS}] gap {gap:+.4f} -> {'PARITY' if abs(gap) < 0.002 else 'DIVERGENCE'}")

    # C++ spoke parity on a round-trip-stable subset over HTTP
    sys.path.insert(0, str(REPO))
    from pil.tokens import TokenSpace
    ts = TokenSpace.from_file(PKG / "bundle.tokenizer.json")
    sg = REPO.parent / "sgiandubh" / "build" / "sgiandubh"
    proc = subprocess.Popen([str(sg), "--rosetta-package", str(PKG), "8191"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2.0)
        checked = match = 0
        for wi, py_ans in zip(te_l, py_answers, strict=True):
            if checked >= 200:
                break
            toks = [uvl[c] for c in ids[wi].tolist()]
            text = ts.decode(toks)
            if ts.encode(text) != toks:
                continue
            req = urllib.request.Request(
                "http://localhost:8191/v1/chat/completions",
                json.dumps({"messages": [{"role": "user", "content": text}]}).encode(),
                {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                a = json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
            cpp_ans = (ts.encode(a["answer"]) if a.get("kind") != "abstain" else [-1])
            py_tok = ts.token_str(py_ans) if py_ans >= 0 else None
            ok = ((a.get("kind") == "abstain" and py_ans < 0)
                  or (py_tok is not None and a.get("answer") == py_tok
                      and cpp_ans == [py_ans]))
            checked += 1
            match += int(ok)
        print(f"[{DS}] C++ SPOKE parity vs python runtime: {match}/{checked} windows exact")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
