"""estate2 probe: the WORLD-STATE FOLD for qa2/qa3, self-grounded, pure python at word level.
Mines entity/location/object/verb classes and take-vs-drop semantics from the corpus's inline
answers, then runs the fold on the unseen test benchmark. Validates the FORM before any tensor
realization. Run: probe_estate2.py [qa2|qa3]"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TASK = sys.argv[1] if len(sys.argv) > 1 else "qa2"
AV = {"?", "Q", "A", "Q:", "A:"}

corpus = (REPO / "data" / f"corpus_babi_{TASK}.txt").read_text()
words = re.findall(r"[\w']+|[.?:]", corpus)

# --- mining (same procedures as estate, extended) ---
cap_occ, cap_clean = Counter(), Counter()
for i, w in enumerate(words):
    if len(w) > 1 and w[0].isupper() and w.isalpha():
        cap_occ[w] += 1
        if not set(words[i + 1:i + 6]) & AV and not set(words[max(0, i - 2):i]) & AV:
            cap_clean[w] += 1
E = {w for w, n in cap_occ.items() if n >= 10 and cap_clean[w] / n >= 0.5}

loc_hist = Counter()                                        # locations: "to the X"
for i, w in enumerate(words[:-2]):
    if w == "to" and words[i + 1] == "the":
        loc_hist[words[i + 2]] += 1
tot = sum(loc_hist.values())
L, acc = set(), 0
for w, n in loc_hist.most_common():
    if acc / tot >= 0.95:
        break
    L.add(w)
    acc += n

obj_hist = Counter()                                        # objects: "the X" NOT after "to",
for i, w in enumerate(words[:-1]):                          # not a location
    if w == "the" and (i == 0 or words[i - 1] != "to") and words[i + 1] not in L:
        obj_hist[words[i + 1]] += 1
O = {w for w, n in obj_hist.most_common(8) if n >= 20}

verb_hist = Counter()                                       # verbs: E _v_ ... "the"
for i, w in enumerate(words[:-2]):
    if w in E and words[i + 1] not in AV:
        verb_hist[words[i + 1]] += 1
def move_dest(ws, i):
    """destination of "E v [back] to the X" starting at verb index i+1; None if not a move."""
    for k in (2, 3):
        if i + k + 2 < len(ws) and ws[i + k] == "to" and ws[i + k + 1] == "the":
            return ws[i + k + 2]
    return None


PARTS = {"up", "down"}
MOVE, GRAB = set(), set()                                   # movement: followed by "to the L"
for v, n in verb_hist.most_common(20):
    if n < 20:
        continue
    ctx = [i for i, w in enumerate(words[:-6]) if w in E and words[i + 1] == v]
    to_the = sum(1 for i in ctx if move_dest(words, i) is not None)
    the_obj = sum(1 for i in ctx if words[i + 2] == "the" and words[i + 3] in O)
    part_obj = sum(1 for i in ctx if words[i + 2] in PARTS and words[i + 3] == "the"
                   and words[i + 4] in O)
    if to_the > len(ctx) * 0.4:
        MOVE.add(v)
    elif (the_obj + part_obj) > len(ctx) * 0.4:
        GRAB.add(v)


def grab_obj(ws, i):
    """object of "E v [part] the O" with verb at i+1; None if not a grab-shaped event."""
    if ws[i + 1] not in GRAB:
        return None
    if i + 3 < len(ws) and ws[i + 2] == "the" and ws[i + 3] in O:
        return ws[i + 3]
    if i + 4 < len(ws) and ws[i + 2] in PARTS and ws[i + 3] == "the" and ws[i + 4] in O:
        return ws[i + 4]
    return None

# take vs drop: UNIVERSAL self-grounding -- greedy flip per verb, scored by how often the
# world-state fold reproduces the corpus's own inline answers (works for any question form).
def run_fold(ws, TAKE, DROP):
    loc, holder, oloc, ohist = {}, {}, {}, {}
    answers = []
    def setoloc(o, place):
        if oloc.get(o) != place:
            ohist.setdefault(o, []).append(oloc.get(o))
            oloc[o] = place
    i = 0
    while i < len(ws):
        w = ws[i]
        if w in E and i + 1 < len(ws):
            v = ws[i + 1]
            dest = move_dest(ws, i) if v in MOVE else None
            if dest is not None:
                loc[w] = dest
                for o, h in holder.items():
                    if h == w:
                        setoloc(o, dest)
            elif v in TAKE and grab_obj(ws, i) is not None:
                obj = grab_obj(ws, i)
                holder[obj] = w
                if w in loc:
                    setoloc(obj, loc[w])
            elif v in DROP and grab_obj(ws, i) is not None:
                holder.pop(grab_obj(ws, i), None)
        if w == "Where" and i + 4 < len(ws):
            if ws[i + 1] == "is" and ws[i + 2] == "the" and ws[i + 3] in O:
                answers.append((oloc.get(ws[i + 3]), i))
            elif ws[i + 1] == "was" and ws[i + 2] == "the" and ws[i + 3] in O \
                    and ws[i + 4] == "before" and i + 6 < len(ws):
                o, ref = ws[i + 3], ws[i + 6]
                hist = (ohist.get(o, []) + [oloc.get(o)])[1:]
                got = next((hist[k - 1] for k in range(len(hist) - 1, 0, -1)
                            if hist[k] == ref), None)
                answers.append((got, i))
        i += 1
    return answers


def gold_at(ws, qi):
    for j in range(qi, min(qi + 14, len(ws) - 1)):
        if ws[j] == "?":
            return ws[j + 3] if j + 2 < len(ws) and ws[j + 2] == ":" else ws[j + 2]
    return None


def score(TAKE, DROP, sample):
    ok = tot = 0
    for got, qi in run_fold(sample, TAKE, DROP):
        g = gold_at(sample, qi)
        if g:
            ok += int(got == g)
            tot += 1
    return ok / max(tot, 1)


sample = words[:120000]
TAKE = set(GRAB)
DROP = set()
for _ in range(2):                                          # coordinate descent, 2 sweeps
    for v in sorted(GRAB):
        as_take = score(TAKE | {v}, DROP - {v}, sample)
        as_drop = score(TAKE - {v}, DROP | {v}, sample)
        if as_drop > as_take:
            TAKE.discard(v)
            DROP.add(v)
        else:
            DROP.discard(v)
            TAKE.add(v)
print(f"E={sorted(E)}\nL={sorted(L)}\nO={sorted(O)}\nMOVE={sorted(MOVE)}")
print(f"TAKE={sorted(TAKE)} DROP={sorted(DROP)}")
print("fold self-agreement on corpus sample:", round(score(TAKE, DROP, sample), 3))

# --- the world-state fold on the unseen benchmark ---
bench = json.load(open(REPO / "data" / f"babi_{TASK}_bench.json"))
ok, fails = 0, []
for b in bench:
    pw = re.findall(r"[\w']+|[.?:]", b["prompt"])
    loc, holder, oloc, ohist = {}, {}, {}, {}
    def setoloc(o, place, oloc=oloc, ohist=ohist):
        if oloc.get(o) != place:
            ohist.setdefault(o, []).append(oloc.get(o))
            oloc[o] = place
    for i, w in enumerate(pw):
        if w in E and i + 1 < len(pw):
            v = pw[i + 1]
            dest = move_dest(pw, i) if v in MOVE else None
            if dest is not None:
                place = dest
                loc[w] = place
                for o, h in holder.items():
                    if h == w:
                        setoloc(o, place)
            elif v in TAKE and grab_obj(pw, i) is not None:
                obj = grab_obj(pw, i)
                holder[obj] = w
                if w in loc:
                    setoloc(obj, loc[w])
            elif v in DROP and grab_obj(pw, i) is not None:
                holder.pop(grab_obj(pw, i), None)
    # query
    got = None
    if TASK == "qa2":
        ms = re.findall(r"Where is the (\w+)", b["prompt"])
        m = ms[-1] if ms else None
        if m:
            got = oloc.get(m)
    else:
        ms = re.findall(r"Where was the (\w+) before the (\w+)", b["prompt"])
        m = ms[-1] if ms else None
        if m:
            o, ref = m
            hist = (ohist.get(o, []) + [oloc.get(o)])[1:]    # full location history
            got = next((hist[k - 1] for k in range(len(hist) - 1, 0, -1)
                        if hist[k] == ref), None)
    hit = got == b["answer"].strip()
    ok += int(hit)
    if not hit and len(fails) < 5:
        fails.append((got, b["answer"].strip(), b["prompt"][-260:]))
print(f"\nESTATE2 world-state fold on {TASK} (1000 unseen): {ok}/1000 = {ok / 1000:.3f}")
for g, a, ptail in fails:
    print(f"\n got={g} gold={a} | ...{ptail}")

if len(sys.argv) > 2 and sys.argv[2] == "--emit-sets":
    out = {"entities": sorted(E), "locations": sorted(L), "objects": sorted(O),
           "move_verbs": sorted(MOVE), "take_verbs": sorted(TAKE),
           "drop_verbs": sorted(DROP), "particles": sorted(PARTS)}
    op = REPO / "data" / f"wyly_estate2_{TASK}.json"
    op.write_text(json.dumps(out, indent=1))
    print("sets ->", op)
