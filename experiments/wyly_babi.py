"""bAbI qa1 for the wyly expert: the GENERALIZATION benchmark. The corpus is the TRAIN split
rendered with inline answers; the benchmark is the TEST split (unseen stories) -- tables cannot
memorize the answers, only the TEMPLATE + binding rules generalize. build/evalpkg mirror
wyly_element_expert.py; the teacher bench reuses its `bench` with --bench-file.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
T = Path("/home/allans/.claude/jobs/4d2f36a2/tmp")


def render(story, upto=None):
    out = []
    for i, (typ, txt, ans) in enumerate(zip(story["type"], story["text"],
                                            story["answer"], strict=True)):
        if upto is not None and i == upto:
            out.append(f"Q: {txt} A:")
            break
        if int(typ) == 1:
            out.append(f"Q: {txt} A: {ans}.")
        else:
            out.append(txt)
    return " ".join(out)


def main():
    import pandas as pd
    tr = pd.read_parquet(T / "babi_qa1_train.parquet")
    te = pd.read_parquet(T / "babi_qa1_test.parquet")
    lines = [render(r["story"]) for _, r in tr.iterrows()]
    import os
    passes = int(os.environ.get("BABI_PASSES", "1"))
    corpus = " ".join(lines * passes)
    (REPO / "data" / "corpus_babi.txt").write_text(corpus)
    bench = []
    for _, r in te.iterrows():
        st = r["story"]
        for i, typ in enumerate(st["type"]):
            if int(typ) == 1:
                bench.append({"prompt": render(st, upto=i),
                              "answer": st["answer"][i], "prop": "qa1"})
    (REPO / "data" / "babi_bench.json").write_text(json.dumps(bench, indent=0))
    print(f"corpus {len(corpus) // 1000} KB ({len(lines)} stories x{passes}); "
          f"bench {len(bench)} test questions")


if __name__ == "__main__":
    main()
