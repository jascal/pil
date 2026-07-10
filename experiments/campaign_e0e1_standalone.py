"""Standalone-learner ablations E0 / E1 (no host LLM scaffold in training labels/geometry).

E0 — rules only:  LABELS=corpus, CONCEPT_INIT=random, CONCEPTS=0
E1 — concepts+rules: same + CONCEPTS=1 (ConceptSpace from counts; random soft geometry)

Still uses the host *tokenizer* (token ids are the package format) — not teacher logits/embeds.
Query judge + bench remain dataset gold. Packages go to wyly_expert_package_v5_{ds}_e0/_e1.

Run:
  cd pil && .venv/bin/python -u experiments/campaign_e0e1_standalone.py
  cd pil && STANDALONE_ARMS=e0 BABI_TASKS=babi2 .venv/bin/python -u experiments/campaign_e0e1_standalone.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Shared: bAbI multi-rule path without teacher scaffold
BASE = {
    "WYLY_TAG": "qwen3b",          # tokenizer/embed auto only if CONCEPT_INIT=grounded
    "WYLY_LIB": "mined",
    "WYLY_EMIT": "1",
    "WYLY_PARITY": "1",
    "WYLY_VAL_REGION": "1",
    "WYLY_ONLINE": "1",
    "WYLY_FOLDS": "0",
    "WYLY_TPOINTER": "0",
    "WYLY_DX": "0",
    "WYLY_LABELS": "corpus",       # stream next-token; no teacher file
    "WYLY_CONCEPT_INIT": "random", # no host embed PCA
    "WYLY_EPISODES": os.environ.get("WYLY_EPISODES", "3"),
    "WYLY_WAKE_STEPS": os.environ.get("WYLY_WAKE_STEPS", "80"),
    "PYTHONUNBUFFERED": "1",
}
# Explicitly strip embed so grounded_init cannot be reached by accident
if "WYLY_EMBED" in os.environ:
    del os.environ["WYLY_EMBED"]

ARMS = {
    "e0": {"WYLY_CONCEPTS": "0", "WYLY_PKG_SUFFIX": "_e0", "label": "E0 rules-only"},
    "e1": {"WYLY_CONCEPTS": "1", "WYLY_PKG_SUFFIX": "_e1", "label": "E1 concepts+rules"},
}

TASKS = {
    "babi2": {"ds": "babi2", "bench": REPO / "data" / "babi_qa2_bench.json", "qa": "qa2"},
    "babi3x": {"ds": "babi3x", "bench": REPO / "data" / "babi_qa3_bench.json", "qa": "qa3"},
}


def bench_package(pkg: Path, bench_path: Path) -> dict:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))
    from pil.tokens import TokenSpace
    from serve_package import decide, load_package

    man = json.loads((pkg / "manifest.json").read_text())
    kinds = {}
    for r in man["rules"]:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    e2 = [d for d in man.get("derived", []) if d.get("kind") == "estate2"]
    concepts = man.get("concepts")
    idi, ngr, mp_ = load_package(pkg / "manifest.json")
    ts = TokenSpace.from_file(pkg / "bundle.tokenizer.json")
    bench = json.loads(bench_path.read_text())
    ok = tot = abstain = 0
    for item in bench:
        toks = ts.encode(" " + item["prompt"])
        d = decide(toks, idi, ngr, mp_)
        gold = item["answer"].strip().lower()
        if not d or d.get("answer") is None:
            abstain += 1
            tot += 1
            continue
        pred = ts.token_str(int(d["answer"])).strip().lower()
        hit = pred == gold or gold.startswith(pred) or pred.startswith(gold)
        ok += int(hit)
        tot += 1
    return {
        "agree": ok / max(tot, 1),
        "ok": ok,
        "tot": tot,
        "abstain": abstain,
        "kinds": kinds,
        "estate2_modes": [d.get("mode") for d in e2],
        "n_rules": man["n_rules"],
        "origin": man.get("origin"),
        "labels": man.get("labels"),
        "concept_init": man.get("concept_init"),
        "n_concept_groups": len(concepts) if concepts else 0,
    }


def run_one(arm: str, task: str) -> dict:
    cfg_t = TASKS[task]
    cfg_a = ARMS[arm]
    env = os.environ.copy()
    env.update(BASE)
    env.update(cfg_a)
    env["WYLY_DS"] = cfg_t["ds"]
    # ensure no embed
    env.pop("WYLY_EMBED", None)
    print(f"\n{'='*60}\n{cfg_a['label']} × {cfg_t['qa']} ({cfg_t['ds']})\n{'='*60}",
          flush=True)
    script = f"""
import os, sys
sys.path.insert(0, {str(REPO / 'experiments')!r})
import wyly_lm_v2 as v2
v2.EPISODES = int(os.environ.get('WYLY_EPISODES', '3'))
v2.WAKE_STEPS = int(os.environ.get('WYLY_WAKE_STEPS', '80'))
import wyly_lm_v5 as v5
print('arm:', 'LABELS='+v5.LABELS, 'CONCEPT_INIT='+v5.CONCEPT_INIT,
      'CONCEPTS='+str(v5.CONCEPTS), 'PKG_SUFFIX='+v5.PKG_SUFFIX,
      'EMBED='+os.environ.get('WYLY_EMBED',''),
      'QUERIES='+str(bool(v5.QUERIES)), 'ESTATE2='+str(bool(v5.ESTATE2)), flush=True)
assert v5.LABELS == 'corpus', v5.LABELS
assert v5.CONCEPT_INIT == 'random', v5.CONCEPT_INIT
assert not os.environ.get('WYLY_EMBED'), 'embed must be unset'
v5.main()
"""
    r = subprocess.run([sys.executable, "-u", "-c", script], cwd=str(REPO), env=env)
    pkg = REPO / "data" / f"wyly_expert_package_v5_{cfg_t['ds']}{cfg_a['WYLY_PKG_SUFFIX']}"
    out = {"arm": arm, "task": task, "qa": cfg_t["qa"], "exit": r.returncode, "pkg": str(pkg)}
    if r.returncode == 0 and pkg.exists():
        b = bench_package(pkg, cfg_t["bench"])
        out.update(b)
        print(f"BENCH {arm}/{cfg_t['qa']}: {b['ok']}/{b['tot']}={b['agree']:.3f} "
              f"estate2={b['estate2_modes']} concepts={b['n_concept_groups']} "
              f"origin={b['origin']} kinds={b['kinds']}", flush=True)
    else:
        print(f"FAILED {arm}/{task} exit={r.returncode}", flush=True)
    return out


def main():
    arms = [a.strip() for a in os.environ.get("STANDALONE_ARMS", "e0,e1").split(",") if a.strip()]
    tasks = [t.strip() for t in os.environ.get("BABI_TASKS", "babi2,babi3x").split(",") if t.strip()]
    results = []
    for arm in arms:
        for task in tasks:
            results.append(run_one(arm, task))
    print("\n" + "=" * 60)
    print("STANDALONE E0 / E1 SCOREBOARD (served, no teacher labels/embeds)")
    print("=" * 60)
    print(f"{'arm':4} {'qa':4} {'served':>8} {'estate2':>12} {'concepts':>8} {'origin':>10}")
    for r in results:
        if "agree" in r:
            print(f"{r['arm']:4} {r['qa']:4} {r['agree']:8.3f} "
                  f"{str(r.get('estate2_modes')):>12} {r.get('n_concept_groups', 0):8d} "
                  f"{r.get('origin', '?'):>10}")
        else:
            print(f"{r['arm']:4} {r['qa']:4} FAILED")
    out = REPO / "data" / "standalone_e0e1_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
