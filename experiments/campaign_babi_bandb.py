"""Band B campaign: multi-rule bAbI packages with estate2 under frozen query defaults.

B1 — babi3x: multi-rule cover + estate2/before (post load-order fix)
B2 — babi2:  multi-rule cover + estate2/is (push 0.778 residual)
B3 — defaults live in wyly_lm_v5 (_BABI cover-sw, auto QUERIES/ESTATE2, QUERY_PACK)

Short wake (counts + admit matter more than SGD for estate2). Emits packages and benches.

Run:
  cd pil && .venv/bin/python -u experiments/campaign_babi_bandb.py
  cd pil && BABI_TASKS=babi3x .venv/bin/python -u experiments/campaign_babi_bandb.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Shared campaign knobs (before importing v5)
os.environ.setdefault("WYLY_TAG", "qwen3b")
os.environ.setdefault("WYLY_LIB", "mined")
os.environ.setdefault("WYLY_EMIT", "1")
os.environ.setdefault("WYLY_PARITY", "1")
os.environ.setdefault("WYLY_VAL_REGION", "1")
os.environ.setdefault("WYLY_ONLINE", "1")
# FOLDS off: packed query batch + multi-rule admit is enough; FOLDS×estate2 is hours.
os.environ.setdefault("WYLY_FOLDS", "0")
# Keep CX for estate2; skip concept growth / transform-pointer by default (speed).
os.environ.setdefault("WYLY_CONCEPTS", "0")
os.environ.setdefault("WYLY_TPOINTER", "0")
os.environ.setdefault("WYLY_DX", "0")
# POINTER/CX/JUDGE/COVER/QUERIES/ESTATE2/QUERY_PACK auto from v5 bAbI defaults when unset.

TASKS = {
    "babi2": {
        "ds": "babi2",
        "bench": REPO / "data" / "babi_qa2_bench.json",
        "label": "qa2",
    },
    "babi3x": {
        "ds": "babi3x",
        "bench": REPO / "data" / "babi_qa3_bench.json",
        "label": "qa3",
    },
}


def bench_package(pkg: Path, bench_path: Path) -> dict:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))
    from serve_package import decide, load_package

    from pil.tokens import TokenSpace

    man = json.loads((pkg / "manifest.json").read_text())
    kinds = {}
    for r in man["rules"]:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    e2 = [d for d in man.get("derived", []) if d.get("kind") == "estate2"]
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
        "estate2": e2,
        "n_rules": man["n_rules"],
    }


def run_task(name: str) -> dict:
    cfg = TASKS[name]
    # Reset module-level state by re-importing in a subprocess for isolation.
    env = os.environ.copy()
    env["WYLY_DS"] = cfg["ds"]
    # Short wake via env consumed after import in child
    env["WYLY_EPISODES"] = env.get("WYLY_EPISODES", "3")
    env["WYLY_WAKE_STEPS"] = env.get("WYLY_WAKE_STEPS", "80")
    env["PYTHONUNBUFFERED"] = "1"
    print(f"\n{'='*60}\nTASK {name} ({cfg['label']}) DS={cfg['ds']}\n{'='*60}", flush=True)
    import subprocess
    script = f"""
import os, sys
from pathlib import Path
sys.path.insert(0, {str(REPO / 'experiments')!r})
import wyly_lm_v2 as v2
v2.EPISODES = int(os.environ.get('WYLY_EPISODES', '3'))
v2.WAKE_STEPS = int(os.environ.get('WYLY_WAKE_STEPS', '80'))
import wyly_lm_v5 as v5
print('defaults:', 'DS='+v5.DS, 'JUDGE='+v5.JUDGE, 'SWCOVER='+str(v5.SWCOVER),
      'POINTER='+str(v5.POINTER), 'CX='+str(v5.CX), 'QUERIES='+str(bool(v5.QUERIES)),
      'ESTATE2='+str(bool(v5.ESTATE2)), 'QUERY_PACK='+str(v5.QUERY_PACK),
      'LIB='+v5.LIB, 'EPISODES', v2.EPISODES, 'WAKE', v2.WAKE_STEPS, flush=True)
v5.main()
"""
    r = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=str(REPO), env=env, check=False,
    )
    pkg = REPO / "data" / f"wyly_expert_package_v5_{cfg['ds']}"
    result = {"task": name, "label": cfg["label"], "exit": r.returncode, "pkg": str(pkg)}
    if r.returncode == 0 and pkg.exists() and cfg["bench"].exists():
        b = bench_package(pkg, cfg["bench"])
        result.update(b)
        print(f"BENCH {cfg['label']}: {b['ok']}/{b['tot']} = {b['agree']:.3f} "
              f"abstain={b['abstain']} estate2={bool(b['estate2'])} kinds={b['kinds']}",
              flush=True)
    else:
        print(f"TASK {name} failed exit={r.returncode} pkg_exists={pkg.exists()}", flush=True)
    return result


def main():
    want = os.environ.get("BABI_TASKS", "babi2,babi3x").split(",")
    want = [w.strip() for w in want if w.strip()]
    results = []
    for name in want:
        if name not in TASKS:
            print(f"unknown task {name}; known {list(TASKS)}", flush=True)
            continue
        results.append(run_task(name))
    print("\n" + "=" * 60)
    print("BAND B SCOREBOARD")
    print("=" * 60)
    for r in results:
        if "agree" in r:
            e2m = ",".join(d.get("mode", "?") for d in r.get("estate2") or [])
            print(f"  {r['label']:4}  served={r['agree']:.3f}  "
                  f"({r['ok']}/{r['tot']})  estate2=[{e2m}]  "
                  f"n_rules={r.get('n_rules')}  kinds={r.get('kinds')}")
        else:
            print(f"  {r['label']:4}  FAILED exit={r.get('exit')}")
    out = REPO / "data" / "bandb_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
