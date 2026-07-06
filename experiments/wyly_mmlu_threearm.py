"""THREE-ARM MMLU (college chemistry, 100 Qs): the composed unit evaluated.
  A  LLM direct        (llama-server chat, pick the letter)
  B  LLM + expert      (claymore tools mode -- the hub decomposes, the certified spoke answers)
  C  expert alone      (option scoring: walk each option through the sw cover, sum confidences)
Sub-queries in arm B are visible in the hub's verbose log for failure attribution.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

LETTERS = "ABCD"


def chat(url, content, timeout=180, max_tokens=400):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps({"messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tokens, "temperature": 0}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def extract_letter(txt):
    m = re.search(r"\b([ABCD])\b", txt.strip()[:200])
    return m.group(1) if m else None


def main():
    import os
    bf = os.environ.get("THREEARM_BENCH", "")
    if bf:
        qs = json.load(open(bf))
    else:
        import pandas as pd
        df = pd.read_parquet("/home/allans/.claude/jobs/4d2f36a2/tmp/mmlu_chem2.parquet")
        qs = df.to_dict("records")
    print(f"{len(qs)} MMLU college-chemistry questions")

    def prompt_of(q):
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["choices"]))
        return (f"Question: {q['question']}\n{opts}\n"
                "Answer with the single letter of the correct choice.")

    import os
    arms = {}
    arm_filter = os.environ.get("THREEARM_ONLY", "")
    pairs = [("A: LLM direct", "http://localhost:8090"),
             ("B: LLM+expert (hub)", "http://localhost:9099")]
    if arm_filter:
        pairs = [p_ for p_ in pairs if p_[0].startswith(arm_filter)]
    for name, url in pairs:
        ok = n = 0
        for q in qs:
            try:
                txt = chat(url, prompt_of(q))
            except Exception:
                txt = ""
            let = extract_letter(txt)
            ok += int(let == LETTERS[q["answer"]])
            n += 1
        arms[name] = (ok, n)
        print(f"{name}: {ok}/{n} = {ok / n:.3f}")

    if arm_filter and arm_filter != "C":
        print("THREE-ARM (partial):", {k: f"{a}/{b}" for k, (a, b) in arms.items()})
        return
    from serve_package import decide, load_package

    from pil.tokens import TokenSpace
    PKG = REPO / "data" / "wyly_expert_package_v5_elements"
    ts = TokenSpace.from_file(PKG / "bundle.tokenizer.json")
    idioms, ngrams, m = load_package(PKG / "manifest.json")
    ok = n = 0
    for q in qs:
        best, bl = -1e9, None
        for i, opt in enumerate(q["choices"]):
            ctx = list(ts.encode(" " + q["question"].strip()))
            score, steps = 0.0, 0
            for t in ts.encode(" " + str(opt)):
                d = decide(ctx, idioms, ngrams, m)
                if d is not None and d["answer"] == t:
                    score += d.get("confidence", 0.0)
                elif d is None:
                    score -= 0.1
                ctx.append(t)
                steps += 1
            sc = score / max(steps, 1)
            if sc > best:
                best, bl = sc, LETTERS[i]
        ok += int(bl == LETTERS[q["answer"]])
        n += 1
    arms["C: expert option-scoring"] = (ok, n)
    print(f"C: expert option-scoring: {ok}/{n} = {ok / n:.3f}")
    print("\nTHREE-ARM:", {k: f"{a}/{b}" for k, (a, b) in arms.items()})


if __name__ == "__main__":
    main()
