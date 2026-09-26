"""Build the decision prompts for ``docs/notes/restricted_early_exit_prereg.md`` (SIGNED).

Runs in the library venv. Uses ``rorshopping/parallel-decisions`` @ 45820320 as the source of truth for prompt
text and field compilation (PIN B): token ids = ``tokenizer.encode(build_prompt(context, schema)) +
field.suffix_tokens``, exactly what its Torch engine prefills and scores. One placeholder token (the first
option's first token) is appended so fieldrun ``--source-dump --tail 1`` dumps the decision position (PIN D).

    runs/pdvenv/bin/python experiments/restricted_decisions_build.py --data data --out runs/ree \
        [--library-decisions]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from parallel_decisions import Schema
from parallel_decisions.prompts import build_prompt
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CHOICES = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]      # PIN C, alphabetical
DESCRIPTION = "Where the person or object asked about is, according to the story"
TASKS = {"qa1": "babi_bench.json", "qa2": "babi_qa2_bench.json", "qa3": "babi_qa3_bench.json"}
N_CAL, N_EVAL = 50, 100


def schema_for(order: list[str]) -> Schema:
    return Schema({"location": {"type": "enum", "choices": order, "description": DESCRIPTION}})


def items(data: Path):
    """Per task: (context, gold) with contexts shuffled by seed 0; CAL = first 50, EVAL = next 100."""
    out = {"cal": [], "eval": []}
    for task, fname in TASKS.items():
        rows = json.loads((data / fname).read_text())
        rows = [r for r in rows if r["answer"].strip() in CHOICES]
        perm = np.random.default_rng(0).permutation(len(rows))
        for split, sl in (("cal", slice(0, N_CAL)), ("eval", slice(N_CAL, N_CAL + N_EVAL))):
            for i in perm[sl]:
                r = rows[int(i)]
                context = r["prompt"].rsplit("A:", 1)[0].strip()
                out[split].append({"task": task, "context": context, "gold": r["answer"].strip()})
    return out


def compile_one(tok, context: str, order: list[str]) -> dict:
    schema = schema_for(order)
    (cf,) = schema.compile(tok)
    prompt_ids = [int(t) for t in tok.encode(build_prompt(context, schema))]
    first = [c[0] for c in cf.candidate_ids]
    return {"ids": prompt_ids + list(cf.suffix_tokens) + [first[0]], "options": order, "option_ids": first,
            "collision": bool(cf.collision), "suffix": cf.suffix}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/ree")
    p.add_argument("--model", default=MODEL, help="tokenizer and (with --library-decisions) the Torch model")
    p.add_argument("--library-decisions", action="store_true",
                   help="also run the library's Torch engine on EVAL")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    data = items(Path(a.data))

    meta, collisions = {}, 0
    plans = {"cal": ("canonical",), "eval": ("canonical", "gold_first", "gold_last")}
    for split, orders in plans.items():
        for order_name in orders:
            rows = []
            for i, it in enumerate(data[split]):
                if order_name == "canonical":
                    order = CHOICES
                else:
                    rest = [c for c in CHOICES if c != it["gold"]]
                    order = [it["gold"], *rest] if order_name == "gold_first" else [*rest, it["gold"]]
                c = compile_one(tok, it["context"], order)
                collisions += c["collision"]
                sid = f"{split}-{order_name}-{it['task']}-{i}"
                rows.append({"sid": sid, "ids": c["ids"]})
                meta[sid] = {"split": split, "order": order_name, "task": it["task"], "gold": it["gold"],
                             "options": c["options"], "option_ids": c["option_ids"], "suffix": c["suffix"],
                             "context": it["context"]}
            path = out / f"{split}_{order_name}.ids.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"[build] {path}: {len(rows)} decisions")
    (out / "meta.json").write_text(json.dumps(meta) + "\n")
    print(f"[build] collisions: {collisions}")
    if collisions:
        raise SystemExit("ABORT (PIN B): a compiled field collides; "
                         "the restricted decode is not single-position")

    if a.library_decisions:
        from parallel_decisions import Decider
        decider = Decider(a.model)
        lib = {}
        for i, it in enumerate(data["eval"]):
            res = decider.decide(it["context"], schema_for(CHOICES))
            lib[f"eval-canonical-{it['task']}-{i}"] = {"value": res["location"].value,
                                                        "probability": res["location"].probability}
        (out / "library_decisions.json").write_text(json.dumps(lib) + "\n")
        print(f"[build] library decisions: {len(lib)}")


if __name__ == "__main__":
    main()
