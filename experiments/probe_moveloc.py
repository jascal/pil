"""Component-level probe: WHY does moveloc miss on deployment queries? Pure token-level python
re-implementation of the moveloc chain on the 1000 real test prompts, vs gold."""
import json
import sys

sys.path.insert(0, "/home/allans/code/pil")
from pil.tokens import TokenSpace

ts = TokenSpace.from_file("/home/allans/code/pil/data/qwen3b.tokenizer.json")
bench = json.load(open("/home/allans/code/pil/data/babi_bench.json"))
LOCS = {"kitchen", "bathroom", "garden", "office", "hallway", "bedroom"}

def tok_str(t): return ts.token_str(t)

stats = {"ok": 0, "recent_cap_wrong": 0, "no_prevocc": 0, "next_loc_wrong": 0, "other": 0}
examples = []
for b in bench[:1000]:
    toks = ts.encode(" " + b["prompt"])
    strs = [tok_str(t) for t in toks]
    # 1) recent cap = queried entity
    cap = None
    for i in range(len(strs) - 1, -1, -1):
        w = strs[i].strip()
        if len(w) > 1 and w[0].isupper() and w.isalpha() and w != "Where":
            cap = i
            break
    gold_entity = b["prompt"].split("Where is ")[-1].split("?")[0].strip()
    if cap is None or strs[cap].strip() != gold_entity:
        stats["recent_cap_wrong"] += 1
        continue
    # 2) filtered prev-occ: last earlier occurrence of the same token, no ?/Q/A within 3 after
    q = -1
    for i in range(cap):
        if toks[i] == toks[cap]:
            nxt3 = [strs[j].strip() for j in range(i + 1, min(i + 4, len(strs)))]
            if not any(x in {"?", "Q", "A", "Q:", "A:"} for x in nxt3):
                q = i
    if q < 0:
        stats["no_prevocc"] += 1
        if len(examples) < 3:
            examples.append(("no_prevocc", b["prompt"][-120:]))
        continue
    # 3) next location token after q
    ans = None
    for j in range(q + 1, len(strs)):
        if strs[j].strip() in LOCS:
            ans = strs[j].strip()
            break
    if ans == b["answer"].strip():
        stats["ok"] += 1
    else:
        stats["next_loc_wrong"] += 1
        if len(examples) < 8:
            ctx = "".join(strs[max(0, q - 2):min(len(strs), q + 12)])
            examples.append(("next_loc_wrong", f"gold={b['answer']} got={ans} | ...{ctx}..."))
print(stats)
print(f"UPPER BOUND of the moveloc FORM: {stats['ok'] / 10:.1f}%")
for e in examples:
    print(" ", e)
