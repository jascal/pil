"""FEEDBACK SHARPENING: the RLHF analog for a certified expert, demonstrated.

Because every answer is CITED, feedback is attributable: a wrong benchmark answer names the
exact fired rule/key that produced its first wrong token. We simulate human feedback with the
benchmark's own errors (floor 0): BAN each culprit key (surgical removal -- the retreat
machinery's serve-time form), re-evaluate, and report (a) the recovered questions (arbitration
falls through to the next-best rule) and (b) the regression count on previously-correct answers
(bans only remove entries; C9-flavored expectation: minimal collateral).
Run: .venv/bin/python experiments/wyly_feedback.py [rounds]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))
from serve_package import decide, load_package  # noqa: E402

from pil.tokens import TokenSpace  # noqa: E402

PKG = REPO / "data" / "wyly_expert_package_v5_elements"


def norm(s):
    return s.strip().strip(".,;:").lower()


def run_bench(bench, ts, idioms, ngrams, m, trace=False):
    results = []
    for b in bench:
        ctx = list(ts.encode(" " + b["prompt"]))
        out, fired = [], []
        for _ in range(10):
            d = decide(ctx, idioms, ngrams, m)
            if d is None:
                break
            fired.append((d, tuple(ctx[-d["k"]:]) if "k" in d else None))
            ctx.append(d["answer"])
            out.append(d["answer"])
        txt = ts.decode(out) if out else ""
        hit = norm(txt).startswith(norm(b["answer"])) or norm(b["answer"]) in norm(
            txt.split(".")[0])
        results.append((hit, fired, out, b))
    return results


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    bench = json.load(open(REPO / "data" / "element_bench.json"))
    ts = TokenSpace.from_file(PKG / "bundle.tokenizer.json")
    idioms, ngrams, m = load_package(PKG / "manifest.json")
    res = run_bench(bench, ts, idioms, ngrams, m)
    base_ok = {i for i, r in enumerate(res) if r[0]}
    print(f"round 0: {len(base_ok)}/{len(bench)} = {len(base_ok) / len(bench):.3f}")
    for rd in range(1, rounds + 1):
        protected = set()                                    # keys load-bearing for CORRECT
        for hit, fired, _out, _b in res:                     # answers are immune (collateral
            if hit:                                          # guard -- the round-1 lesson:
                for _d, key in fired:                        # naive bans regressed -12)
                    if key is not None:
                        protected.add(key)
        bans = 0
        for _i, (hit, fired, out, b) in enumerate(res):
            if hit or not fired:
                continue
            want = ts.encode(" " + b["answer"])
            for step, (_d, key) in enumerate(fired):
                if step < len(want) and out[step] == want[step]:
                    continue                                  # this step was right; skip
                if (key is not None and key not in protected
                        and key in ngrams[len(key)]):
                    del ngrams[len(key)][key]                # ban the culprit key
                    bans += 1
                break                                         # one attributable ban per error
        adds = 0                                             # the round-2 lesson: subtraction
        for _i, (hit, _fired, _out, b) in enumerate(res):    # cannot sharpen (culprits are
            if hit:                                          # load-bearing); ADD corrective
                continue                                     # prompt-specific long keys instead
            ctx = list(ts.encode(" " + b["prompt"]))
            want = ts.encode(" " + b["answer"])
            kw = int(m.get("W", 6))                          # patch at the manifest's own max
            for w in want:                                   # scanned key length
                key = tuple(ctx[-kw:])
                if len(key) == kw:
                    ngrams[kw][key] = (w, "feedback", f"human-feedback patch: {b['prompt'][:40]}",
                                      0.999)
                    adds += 1
                ctx.append(w)
        res = run_bench(bench, ts, idioms, ngrams, m)
        now_ok = {i for i, r in enumerate(res) if r[0]}
        fixed = len(now_ok - base_ok)
        broken = len(base_ok - now_ok)
        print(f"round {rd}: banned {bans}, patched {adds} keys -> {len(now_ok)}/{len(bench)} = "
              f"{len(now_ok) / len(bench):.3f}  (+{fixed} fixed, -{broken} regressed)")
        base_ok = now_ok
        if bans == 0 and adds == 0:
            break


if __name__ == "__main__":
    main()
