"""CANONICALIZATION enricher: mine a template inventory + slot sets from the corpus and write
a "canon" section into a package manifest. Templates are word-level patterns with {E} (entity)
and {V} (value) slots; templates are clustered into PROPERTIES by shared (E, V) pairs, and each
property's canonical representative is its highest-support VALUE-TERMINAL template (usable as a
next-token query prefix). Self-grounded: no labels.
Run: wyly_canon.py <corpus.txt> <package_dir>"""
import json
import re
import sys
from collections import Counter, defaultdict


def mine(corpus_path, min_support=30):
    text = open(corpus_path).read()
    sents = re.split(r"(?<=[.!?])\s+", text)
    words_all = re.findall(r"[A-Za-z']+", text)
    capc = Counter(w for w in words_all if len(w) > 1 and w[0].isupper() and w.isalpha())
    lowc = Counter(w for w in words_all if w.islower())
    ents = {w for w, n in capc.items()
            if n >= 20 and lowc[w.lower()] < n * 0.2}         # NAMES: rarely seen lowercase
    pat_count, pat_pairs = Counter(), defaultdict(set)
    for s in sents[:60000]:
        toks = re.findall(r"[\w.']+|[:%]", s.rstrip("."))
        if not (2 <= len(toks) <= 16):
            continue
        epos = [i for i, w in enumerate(toks) if w in ents]
        if len(epos) != 1:
            continue
        vpos = [i for i, w in enumerate(toks)
                if i != epos[0] and (re.fullmatch(r"\d+(\.\d+)?", w)
                                     or (w[0].isupper() and 1 <= len(w) <= 2))]
        cands = [None] + vpos
        for vp in cands:
            pat = tuple("{E}" if i == epos[0] else ("{V}" if i == vp else w)
                        for i, w in enumerate(toks))
            if vp is not None:
                pat_count[pat] += 1
                pat_pairs[pat].add((toks[epos[0]], toks[vp]))
    pats = [p for p, n in pat_count.most_common(60) if n >= min_support]
    # cluster templates into properties by (E,V)-pair overlap
    groups = []
    for p in pats:
        placed = False
        for g in groups:
            rep_pairs = pat_pairs[g[0]]
            ov = len(pat_pairs[p] & rep_pairs) / max(1, min(len(pat_pairs[p]), len(rep_pairs)))
            if ov >= 0.5:
                g.append(p)
                placed = True
                break
        if not placed:
            groups.append([p])
    canon = []
    for g in groups:
        term = [p for p in g if p[-1] == "{V}"]
        if not term:
            continue
        rep = max(term, key=lambda p: pat_count[p])
        prefix = " ".join(rep[:-1]).replace(" :", ":").replace(" %", "%")
        content = sorted({w.lower().rstrip("s") for p in g for w in p
                          if w not in ("{E}", "{V}") and w.isalpha()
                          and w.lower() not in {"the", "of", "is", "a", "an", "in", "for",
                                                "as", "with", "units"}})
        canon.append({"prefix": prefix, "keywords": content,
                      "support": sum(pat_count[p] for p in g)})
    return {"entities": sorted(ents), "templates": canon}


def main():
    corpus, pkg = sys.argv[1], sys.argv[2]
    c = mine(corpus)
    mp = f"{pkg}/manifest.json"
    m = json.load(open(mp))
    m["canon"] = c
    json.dump(m, open(mp, "w"))
    print(f"canon: {len(c['entities'])} entities, {len(c['templates'])} property templates")
    for t in c["templates"]:
        print(f"  [{t['support']}] {t['prefix']}  (kw: {', '.join(t['keywords'][:5])})")


if __name__ == "__main__":
    main()
