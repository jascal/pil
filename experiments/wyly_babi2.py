"""bAbI qa2/qa3 from the classic archive (native format): corpus (90% of train, inline
answers), judge queries (held-out 10% of train stories), benchmark (test)."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
T = Path("/home/allans/.claude/jobs/4d2f36a2/tmp/tasks_1-20_v1-2/en-10k")
TASK = os.environ.get("BABI_TASK", "qa2")
NAMES = {"qa2": "qa2_two-supporting-facts", "qa3": "qa3_three-supporting-facts"}


def parse(path):
    stories, cur = [], []
    for line in path.read_text().splitlines():
        num, rest = line.split(" ", 1)
        if num == "1" and cur:
            stories.append(cur)
            cur = []
        if "\t" in rest:
            q, a, _ = rest.split("\t")
            cur.append(("q", q.strip(), a.strip()))
        else:
            cur.append(("s", rest.strip(), None))
    if cur:
        stories.append(cur)
    return stories


def render(story, upto=None, inline=True):
    parts = []
    for i, (kind, text, ans) in enumerate(story):
        if upto is not None and i == upto:
            parts.append(f"Q: {text} A:")
            break
        if kind == "s":
            parts.append(text)
        elif inline:
            parts.append(f"Q: {text} A: {ans}.")
    return " ".join(parts)


def main():
    name = NAMES[TASK]
    train = parse(T / f"{name}_train.txt")
    test = parse(T / f"{name}_test.txt")
    ncut = int(0.9 * len(train))
    corpus = " ".join(render(st) for st in train[:ncut])
    (REPO / "data" / f"corpus_babi_{TASK}.txt").write_text(corpus)
    vq = []
    for st in train[ncut:]:
        for i, (kind, _t, ans) in enumerate(st):
            if kind == "q":
                vq.append({"prompt": render(st, upto=i), "answer": ans})
    (REPO / "data" / f"wyly_queries_babi_{TASK}.json").write_text(json.dumps(vq, indent=0))
    bench = []
    for st in test:
        for i, (kind, _t, ans) in enumerate(st):
            if kind == "q":
                bench.append({"prompt": render(st, upto=i), "answer": " " + ans,
                              "prop": TASK, "name": TASK})
    bench = bench[:1000]
    (REPO / "data" / f"babi_{TASK}_bench.json").write_text(json.dumps(bench, indent=0))
    print(f"{TASK}: corpus {len(corpus) // 1000} KB ({ncut} stories); "
          f"{len(vq)} judge queries; bench {len(bench)}")


if __name__ == "__main__":
    main()
