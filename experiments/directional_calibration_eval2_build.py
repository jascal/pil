"""Build the signed #131 EVAL2 decisions, indices 150:200 per bAbI task.

Run only after the C0/C1/C2 development selection is recorded. Uses the same
parallel-decisions prompt compiler and option order as #129/#130.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from parallel_decisions import Decider
from transformers import AutoTokenizer

from experiments.restricted_decisions_build import CHOICES, TASKS, compile_one, schema_for

MODELS = {"0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "3b": "Qwen/Qwen2.5-3B-Instruct"}


def selected_items(data: Path) -> list[dict]:
    out = []
    for task, fname in TASKS.items():
        rows = json.loads((data / fname).read_text())
        rows = [r for r in rows if r["answer"].strip() in CHOICES]
        perm = np.random.default_rng(0).permutation(len(rows))
        if len(perm) < 200:
            raise ValueError(f"{task}: fewer than 200 eligible contexts")
        for idx in perm[150:200]:
            row = rows[int(idx)]
            out.append({"task": task, "context": row["prompt"].rsplit("A:", 1)[0].strip(),
                        "gold": row["answer"].strip()})
    return out


def build(data: Path, out: Path, seen_meta: Path, library: bool = False):
    items = selected_items(data)
    seen = json.loads(seen_meta.read_text())
    seen_contexts = {m["context"] for m in seen.values() if m["order"] == "canonical"}
    if len(items) != 150 or any(it["context"] in seen_contexts for it in items):
        raise RuntimeError("EVAL2 count or context-disjointness failed")
    tokenizers = {tag: AutoTokenizer.from_pretrained(model) for tag, model in MODELS.items()}
    meta = {}
    lines = []
    for i, it in enumerate(items):
        compiled = {tag: compile_one(tok, it["context"], CHOICES) for tag, tok in tokenizers.items()}
        one, three = compiled["0.5b"], compiled["3b"]
        if one["ids"] != three["ids"] or one["option_ids"] != three["option_ids"]:
            raise RuntimeError(f"tokenizer mismatch at EVAL2 index {i}")
        if one["collision"]:
            raise RuntimeError(f"field collision at EVAL2 index {i}")
        sid = f"eval2-canonical-{it['task']}-{i}"
        lines.append({"sid": sid, "ids": one["ids"]})
        meta[sid] = {"split": "eval2", "order": "canonical", "task": it["task"], "gold": it["gold"],
                     "options": CHOICES, "option_ids": one["option_ids"], "suffix": one["suffix"],
                     "context": it["context"]}
    for tag in MODELS:
        target = out / tag
        target.mkdir(parents=True, exist_ok=True)
        (target / "eval2_canonical.ids.jsonl").write_text("".join(json.dumps(row) + "\n" for row in lines))
        (target / "meta.json").write_text(json.dumps(meta) + "\n")
    print("EVAL2: 150 decisions, 50/task, distinct contexts, identical 0.5B/3B token IDs")
    if library:
        decider = Decider(MODELS["3b"])
        result = {}
        for sid, item in meta.items():
            r = decider.decide(item["context"], schema_for(CHOICES))
            result[sid] = {"value": r["location"].value, "probability": r["location"].probability}
        (out / "3b" / "library_decisions.json").write_text(json.dumps(result) + "\n")
        print(f"3B library decisions: {len(result)}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("runs/directional_eval2"))
    p.add_argument("--seen-meta", type=Path, default=Path("runs/ree/meta.json"))
    p.add_argument("--library-decisions", action="store_true")
    a = p.parse_args()
    build(a.data, a.out, a.seen_meta, a.library_decisions)


if __name__ == "__main__":
    main()
