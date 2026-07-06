"""The BENCHMARKED DOMAIN EXPERT: periodic table, end to end.

The domain is CLOSED and fully enumerable, so the benchmark states the expert's complete required
scope: for every element, six cloze facts (symbol, atomic number, rounded atomic mass, period,
group/f-block, category). The coverage corpus states every benchmark fact in many surface
templates (the training text IS the coverage contract); the teacher is chosen by a bake-off ON
the benchmark among models this machine runs; the package is built from the winner's decisions by
the standard wyly pipeline and evaluated on the same benchmark by greedy multi-token completion
through the runtime -- plus out-of-domain abstention probes (the bounded-expert virtue).

Subcommands:
  build      -> data/corpus_elements.txt + data/element_bench.json
  bench      -> evaluate a HF causal LM on the benchmark (--model, --dtype fp32|fp16|int8)
  evalpkg    -> evaluate the emitted package (python runtime, greedy loop) + OOD probes
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
EJSON = Path("/home/allans/.claude/jobs/4d2f36a2/tmp/elements.json")

CLOZE = {
    "symbol": "The chemical symbol of {name} is",
    "number": "The atomic number of {name} is",
    "mass": "The atomic mass of {name} in atomic mass units is approximately",
    "period": "{name} is in period",
    "group": "{name} belongs to group",
    "category": "In the periodic table, {name} is classified as a",
}
TEMPLATES = {
    "symbol": ["The chemical symbol of {name} is {a}.",
               "{name} has the chemical symbol {a}.",
               "{a} is the chemical symbol of {name}.",
               "Element {name}: symbol {a}.",
               "The symbol for {name} is {a}."],
    "number": ["The atomic number of {name} is {a}.",
               "{name} has atomic number {a}.",
               "Element number {a} is {name}.",
               "{name}: atomic number {a}.",
               "With atomic number {a}, {name} sits in the periodic table."],
    "mass": ["The atomic mass of {name} in atomic mass units is approximately {a}.",
             "{name} has an atomic mass of approximately {a} atomic mass units.",
             "{name}: atomic mass approximately {a}.",
             "Approximately {a} atomic mass units is the atomic mass of {name}."],
    "period": ["{name} is in period {a}.",
               "{name} sits in period {a} of the periodic table.",
               "Period {a} contains {name}.",
               "{name}: period {a}."],
    "group": ["{name} belongs to group {a}.",
              "{name} is found in group {a}.",
              "Group {a} contains {name}.",
              "{name}: group {a}."],
    "category": ["In the periodic table, {name} is classified as a {a}.",
                 "{name} is classified as a {a}.",
                 "{name} is a {a}.",
                 "As a {a}, {name} shows the family's typical behavior."],
}


def facts():
    els = json.load(open(EJSON))["elements"]
    out = []
    for e in els:
        if e["number"] > 118:
            continue
        grp = str(e["group"]) if e.get("group") else "the f-block"
        for prop, ans in [("symbol", e["symbol"]), ("number", str(e["number"])),
                          ("mass", str(round(e["atomic_mass"], 1))),
                          ("period", str(e["period"])), ("group", grp),
                          ("category", e["category"])]:
            out.append({"name": e["name"], "prop": prop, "answer": ans})
    return out


def cmd_build():
    fs = facts()
    bench = [{"prompt": CLOZE[f["prop"]].format(name=f["name"]), "answer": f["answer"],
              "prop": f["prop"], "name": f["name"]} for f in fs]
    (DATA / "element_bench.json").write_text(json.dumps(bench, indent=0))
    rng = random.Random(0)
    lines = []
    for _rep in range(25):                                    # counts support: 25 passes
        for f in fs:
            t = rng.choice(TEMPLATES[f["prop"]])
            lines.append(t.format(name=f["name"], a=f["answer"]))
    rng.shuffle(lines)
    (DATA / "corpus_elements.txt").write_text(" ".join(lines))
    print(f"benchmark: {len(bench)} clozes; corpus: "
          f"{len(' '.join(lines)) // 1000} KB ({len(lines)} fact statements)")


def norm(s):
    return s.strip().strip(".,;:").lower()


def cmd_bench(model, dtype, batch, bench_file):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    bench = json.load(open(bench_file))
    tok = AutoTokenizer.from_pretrained(model)
    kw = {}
    if dtype == "int8":
        from transformers import BitsAndBytesConfig
        kw = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True),
              "device_map": "cuda:0"}
    else:
        kw = {"torch_dtype": torch.float16 if dtype == "fp16" else torch.float32}
    m = AutoModelForCausalLM.from_pretrained(model, **kw)
    if dtype != "int8":
        m = m.cuda()
    m = m.eval()
    ok = 0
    per = {}
    for i in range(0, len(bench), batch):
        chunk = bench[i:i + batch]
        prompts = [b["prompt"] for b in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=10, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        for b, row in zip(chunk, g, strict=True):
            txt = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            hit = norm(txt).startswith(norm(b["answer"])) or norm(b["answer"]) in norm(
                txt.split(".")[0])
            ok += int(hit)
            per.setdefault(b["prop"], [0, 0])
            per[b["prop"]][0] += int(hit)
            per[b["prop"]][1] += 1
    print(f"BENCH {model} [{dtype}]: {ok}/{len(bench)} = {ok / len(bench):.3f}")
    for p, (h, n) in sorted(per.items()):
        print(f"   {p:>9}: {h}/{n} = {h / n:.3f}")


def cmd_evalpkg(pkg, bench_file):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))
    from serve_package import decide, load_package

    from pil.tokens import TokenSpace
    bench = json.load(open(bench_file))
    ts = TokenSpace.from_file(Path(pkg) / "bundle.tokenizer.json")
    idioms, ngrams, m = load_package(Path(pkg) / "manifest.json")
    ok = 0
    per = {}
    abstain = 0
    import os
    floor = float(os.environ.get("WYLY_CONF_FLOOR", "0"))
    for b in bench:
        ctx = list(ts.encode(" " + b["prompt"]))
        out = []
        for _ in range(10):
            d = decide(ctx, idioms, ngrams, m)
            if d is None or (floor and d.get("confidence", 1.0) < floor and not out):
                d = None
            if d is None:
                break
            ctx.append(d["answer"])
            out.append(d["answer"])
        txt = ts.decode(out) if out else ""
        if not out:
            abstain += 1
        hit = norm(txt).startswith(norm(b["answer"])) or norm(b["answer"]) in norm(
            txt.split(".")[0])
        ok += int(hit)
        per.setdefault(b["prop"], [0, 0])
        per[b["prop"]][0] += int(hit)
        per[b["prop"]][1] += 1
    print(f"PACKAGE {pkg}: {ok}/{len(bench)} = {ok / len(bench):.3f} "
          f"(abstained on {abstain})")
    for p, (h, n) in sorted(per.items()):
        print(f"   {p:>9}: {h}/{n} = {h / n:.3f}")
    ood = ["The capital of France is", "The square root of 144 is",
           "The author of Hamlet is", "The boiling point of ethanol is"]
    st = 0
    for q in ood:
        ctx = list(ts.encode(" " + q))
        d = decide(ctx, idioms, ngrams, m)
        st += int(d is None or (floor and d.get("confidence", 1.0) < floor))
    print(f"OOD probes: abstained {st}/{len(ood)} (the bounded-expert contract)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "bench", "evalpkg"])
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pkg", default=str(DATA / "wyly_expert_package_v5_elements"))
    ap.add_argument("--bench-file", default=str(DATA / "element_bench.json"))
    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build()
    elif a.cmd == "bench":
        cmd_bench(a.model, a.dtype, a.batch, a.bench_file)
    else:
        cmd_evalpkg(a.pkg, a.bench_file)
