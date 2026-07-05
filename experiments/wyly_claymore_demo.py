"""Per-dataset experts under claymore (design direction 6): the deployment-shaped matrix.

Two LEARNED next-token experts -- the wikitext package and the code package, both emitted by the
self-compiling learner (wyly_lm_v5 WYLY_EMIT=1; counts + judge-admitted induction/relation rules
with confidence fields) -- served as separate sgiandubh HTTP spokes and federated under the
claymore hub. The bound IS the router: a code query fires the code expert (the wiki expert abstains
or ranks lower), a prose query fires the wiki expert, and a gibberish query makes every spoke
abstain, so the hub REFUSES in code. Spoke confidences (the support-weighted cover's currency) feed
claymore's ranking directly.

Note: claymore's hub-side relevance gate is WORD-level Jaccard over passage answers; BPE splits
identifiers (ne12 -> ne+12), so subword citations never overlap whole-word queries -- the demo runs
with --min-relevance 0 and routes purely by abstention (the design's own philosophy: the bound IS
the router). A subword-aware relevance gate is noted as claymore future work.

Run: cd pil && .venv/bin/python experiments/wyly_claymore_demo.py
     (needs ../sgiandubh/build/sgiandubh and ../claymore/build/claymore built)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SG = REPO.parent / "sgiandubh" / "build" / "sgiandubh"
CM = REPO.parent / "claymore" / "build" / "claymore"
WIKI = REPO / "data" / "wyly_expert_package_v5"
CODE = REPO / "data" / "wyly_expert_package_v5_code"


def post(url, payload, timeout=20):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(url, text):
    out = post(url + "/v1/chat/completions",
               {"messages": [{"role": "user", "content": text}]})
    content = out["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"answer": content, "kind": "hub"}


def probes():
    """round-trip-stable probes: a wiki window; a code window whose LAST token the wiki package has
    never seen as a predecessor (so the wiki spoke must abstain -- the bound routes); and gibberish
    made of two DIFFERENT never-seen tokens (a doubled token would legitimately fire the relation
    rule -- repeated-token continuation is real next-token behavior, not gibberish)."""
    sys.path.insert(0, str(REPO))
    import torch

    from pil.tokens import TokenSpace
    ts = TokenSpace.from_file(WIKI / "bundle.tokenizer.json")
    prevs = {}
    for name, pkg in [("wiki", WIKI), ("code", CODE)]:
        man = json.loads((pkg / "manifest.json").read_text())
        prevs[name] = {r["ctx"][0] for r in man["rules"] if r.get("kind") == "ngram"}
    out = {}
    for name, dfile, avoid in [("wiki", "wyly_nexttoken_wikitext_L256.pt", "code"),
                               ("code", "wyly_nexttoken_code_L256.pt", "wiki")]:
        d = torch.load(REPO / "data" / dfile, map_location="cpu")
        for wi in range(int(0.9 * len(d["kept_ids"])), len(d["kept_ids"])):
            orig = d["kept_ids"][wi][-24:].tolist()
            text = ts.decode(orig)
            if (ts.encode(text) == orig and text.strip()
                    and orig[-1] in prevs[name] and orig[-1] not in prevs[avoid]):
                out[name] = text
                break
    unseen = [t for t in range(300, 50304)
              if t not in prevs["wiki"] and t not in prevs["code"]][:400]
    for a in unseen:
        for b in unseen:
            if a != b:
                txt = ts.decode([a, b])
                if ts.encode(txt) == [a, b]:
                    out["gibberish"] = txt
                    return out
    return out


def main():
    spokes_cfg = {"spokes": [
        {"name": "wiki-expert", "url": "http://localhost:8181",
         "domain": "English encyclopedic prose next-token continuation (wikitext)"},
        {"name": "code-expert", "url": "http://localhost:8182",
         "domain": "C C++ source code next-token continuation (llama.cpp)"}],
        "mode": "deterministic", "top_k": 2}
    cfg = REPO / "data" / "wyly_spokes.json"
    cfg.write_text(json.dumps(spokes_cfg, indent=1))
    procs = []
    try:
        procs.append(subprocess.Popen([str(SG), "--rosetta-package", str(WIKI), "8181"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen([str(SG), "--rosetta-package", str(CODE), "8182"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen([str(CM), str(cfg), "9099", "--min-relevance", "0"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(2.5)
        ps = probes()
        print("PER-DATASET EXPERTS UNDER CLAYMORE -- two learned spokes, hub routes by abstention\n")
        for name in ["wiki", "code", "gibberish"]:
            q = ps[name]
            wiki_a = ask("http://localhost:8181", q)
            code_a = ask("http://localhost:8182", q)
            hub_a = ask("http://localhost:9099", q)
            print(f"[{name}] query: {q[:70]!r}")
            print(f"    wiki spoke: {wiki_a.get('kind'):>9}  "
                  f"{wiki_a.get('answer', '')!r:>12}  conf={wiki_a.get('confidence', '-')}")
            print(f"    code spoke: {code_a.get('kind'):>9}  "
                  f"{code_a.get('answer', '')!r:>12}  conf={code_a.get('confidence', '-')}")
            hub_txt = hub_a.get("answer", "")
            print(f"    HUB: {hub_txt[:110]!r}")
            print()
    finally:
        for p in procs:
            p.terminate()
    print("read: the hub answers from whichever expert is confident in-domain and REFUSES when all")
    print("spokes abstain -- the bound is the router; confidences come from the support-weighted")
    print("cover and rank the survivors.")


if __name__ == "__main__":
    main()
