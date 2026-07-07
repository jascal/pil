"""Validate the SELF-GROUNDED estate design: mine entity/value sets from the corpus with the
proposed procedures, then run the register on the 1000 test prompts."""
import json
import sys
from collections import Counter

sys.path.insert(0, "/home/allans/code/pil")
from pil.tokens import TokenSpace

ts = TokenSpace.from_file("/home/allans/code/pil/data/qwen3b.tokenizer.json")
corpus = open("/home/allans/code/pil/data/corpus_babi.txt").read()
toks = ts.encode(corpus[:400000])
strs = [ts.token_str(t) for t in toks]
AVOID = {"?", "Q", "A", "Q:", "A:"}

# entity set: cap tokens NOT predominantly in question contexts
cap_occ, cap_clean = Counter(), Counter()
for i, w in enumerate(strs):
    ww = w.strip()
    if len(ww) > 1 and ww[0].isupper() and ww.isalpha():
        cap_occ[ww] += 1
        nxt3 = {strs[j].strip() for j in range(i + 1, min(i + 4, len(strs)))}
        if not nxt3 & AVOID:
            cap_clean[ww] += 1
E = {w for w, n in cap_occ.items() if n >= 10 and cap_clean[w] / n >= 0.5}
print("entities mined:", sorted(E))

# value set: successors of 'the' within 6 after an entity occurrence
val = Counter()
ent_pos = [i for i, w in enumerate(strs) if w.strip() in E]
entset_pos = set(ent_pos)
for i in ent_pos:
    for j in range(i + 1, min(i + 7, len(strs) - 1)):
        if strs[j].strip() == "the":
            val[strs[j + 1].strip()] += 1
            break
tot = sum(val.values())
V, acc = set(), 0
for w, n in val.most_common():
    if acc / tot >= 0.95:
        break
    V.add(w)
    acc += n
print("values mined:", sorted(V), f"(95% mass, from {len(val)} candidates)")

# the REGISTER on the 1000 test prompts
bench = json.load(open("/home/allans/code/pil/data/babi_bench.json"))
ok = miss = 0
for b in bench:
    ptoks = ts.encode(" " + b["prompt"])
    pstrs = [ts.token_str(t).strip() for t in ptoks]
    ent = None
    for i in range(len(pstrs) - 1, -1, -1):
        if pstrs[i] in E:
            ent = pstrs[i]
            break
    reg = {}
    i = 0
    for i2, w in enumerate(pstrs):
        if w in E:
            nxt3 = {pstrs[j] for j in range(i2 + 1, min(i2 + 4, len(pstrs)))}
            if nxt3 & AVOID:
                continue
            for j in range(i2 + 1, min(i2 + 8, len(pstrs))):
                if pstrs[j] in V:
                    reg[w] = pstrs[j]
                    break
    got = reg.get(ent)
    if got == b["answer"].strip():
        ok += 1
    else:
        miss += 1
print(f"ESTATE (self-grounded) on 1000 unseen test prompts: {ok}/1000 = {ok / 1000:.3f}")
