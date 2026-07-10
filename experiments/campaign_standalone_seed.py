"""P0+P1 standalone seed campaign: SOFT=0 + WordCodec alphabet on bAbI.

No host tokenizer, no teacher labels/embeds, no soft SGD — counts + sleep + estate2.
Reuses the e0e1 campaign shell.

  build:  .venv/bin/python experiments/build_word_babi.py
  run:    .venv/bin/python -u experiments/campaign_standalone_seed.py

Packages: data/wyly_expert_package_v5_babi{2,3x}_word_standalone
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BASE = {
    "WYLY_TAG": "standalone",
    "WYLY_LIB": "mined",
    "WYLY_EMIT": "1",
    "WYLY_PARITY": "1",
    "WYLY_VAL_REGION": "1",
    "WYLY_ONLINE": "1",
    "WYLY_FOLDS": "0",
    "WYLY_CONCEPTS": "0",
    "WYLY_TPOINTER": "0",
    "WYLY_DX": "0",
    "WYLY_LABELS": "corpus",
    "WYLY_CONCEPT_INIT": "random",
    "WYLY_SOFT": "0",
    "WYLY_ALPHABET": "word",
    "WYLY_ORIGIN": "standalone",
    "WYLY_PKG_SUFFIX": "_standalone",
    "WYLY_EPISODES": os.environ.get("WYLY_EPISODES", "3"),
    "WYLY_WAKE_STEPS": os.environ.get("WYLY_WAKE_STEPS", "80"),
    "PYTHONUNBUFFERED": "1",
}

TASKS = {
    "babi2_word": {
        "ds": "babi2_word",
        "bench": REPO / "data" / "babi_qa2_bench.json",
        "qa": "qa2",
        "alphabet": REPO / "data" / "alphabet_babi2_word.json",
    },
    "babi3x_word": {
        "ds": "babi3x_word",
        "bench": REPO / "data" / "babi_qa3_bench.json",
        "qa": "qa3",
        "alphabet": REPO / "data" / "alphabet_babi3x_word.json",
    },
}


def ensure_data():
    need = []
    for t in TASKS.values():
        pt = REPO / "data" / f"wyly_nexttoken_{t['ds']}_L256.pt"
        if not pt.exists() or not t["alphabet"].exists():
            need.append(t["ds"])
    if need:
        print(f"building word datasets for {need}...", flush=True)
        subprocess.check_call([sys.executable, str(REPO / "experiments" / "build_word_babi.py")],
                              cwd=str(REPO))


def bench_package(pkg: Path, bench_path: Path, alphabet_path: Path) -> dict:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))
    from serve_package import decide, load_package

    from pil.alphabet import WordCodec

    man = json.loads((pkg / "manifest.json").read_text())
    kinds = {}
    for r in man["rules"]:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    e2 = [d for d in man.get("derived", []) if d.get("kind") == "estate2"]
    codec = WordCodec.from_file(alphabet_path if alphabet_path.exists()
                                else pkg / "alphabet.json")
    idi, ngr, mp_ = load_package(pkg / "manifest.json")
    bench = json.loads(bench_path.read_text())
    ok = tot = abstain = 0
    for item in bench:
        toks = codec.encode(item["prompt"])
        d = decide(toks, idi, ngr, mp_)
        gold = item["answer"].strip().lower()
        if not d or d.get("answer") is None:
            abstain += 1
            tot += 1
            continue
        pred = codec.token_str(int(d["answer"])).strip().lower()
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
        "alphabet": man.get("alphabet"),
        "soft_student": man.get("soft_student"),
        "labels": man.get("labels"),
        "alphabet_hash": man.get("alphabet_hash"),
    }


def run_one(task: str) -> dict:
    cfg = TASKS[task]
    env = os.environ.copy()
    env.update(BASE)
    env["WYLY_DS"] = cfg["ds"]
    env["WYLY_ALPHABET_PATH"] = str(cfg["alphabet"])
    env.pop("WYLY_EMBED", None)
    env.pop("WYLY_TOKENIZER", None)
    print(f"\n{'='*60}\nSTANDALONE SEED × {cfg['qa']} ({cfg['ds']})\n{'='*60}",
          flush=True)
    script = f"""
import os, sys
sys.path.insert(0, {str(REPO / 'experiments')!r})
import wyly_lm_v2 as v2
v2.EPISODES = int(os.environ.get('WYLY_EPISODES', '3'))
v2.WAKE_STEPS = int(os.environ.get('WYLY_WAKE_STEPS', '80'))
import wyly_lm_v5 as v5
print('config:', 'SOFT='+str(v5.SOFT), 'ALPHABET='+v5.ALPHABET,
      'LABELS='+v5.LABELS, 'origin='+v5.provenance_origin(),
      'QUERIES='+str(bool(v5.QUERIES)), 'ESTATE2='+str(bool(v5.ESTATE2)), flush=True)
assert not v5.SOFT
assert v5.ALPHABET == 'word'
assert v5.LABELS == 'corpus'
assert v5.provenance_origin() == 'standalone'
v5.main()
"""
    r = subprocess.run([sys.executable, "-u", "-c", script], cwd=str(REPO), env=env)
    pkg = REPO / "data" / f"wyly_expert_package_v5_{cfg['ds']}_standalone"
    out = {"task": task, "qa": cfg["qa"], "exit": r.returncode, "pkg": str(pkg)}
    if r.returncode == 0 and pkg.exists():
        b = bench_package(pkg, cfg["bench"], cfg["alphabet"])
        out.update(b)
        print(f"BENCH {cfg['qa']}: {b['ok']}/{b['tot']}={b['agree']:.3f} "
              f"origin={b['origin']} alphabet={b['alphabet']} soft={b['soft_student']} "
              f"estate2={b['estate2_modes']} kinds={b['kinds']}", flush=True)
    else:
        print(f"FAILED {task} exit={r.returncode}", flush=True)
    return out


def main():
    ensure_data()
    tasks = [t.strip() for t in os.environ.get("BABI_TASKS", "babi2_word,babi3x_word").split(",")
             if t.strip()]
    results = []
    for t in tasks:
        if t not in TASKS:
            print(f"unknown task {t}", flush=True)
            continue
        results.append(run_one(t))
    print("\n" + "=" * 60)
    print("STANDALONE SEED SCOREBOARD (SOFT=0 + WordCodec, no host LLM)")
    print("=" * 60)
    print(f"{'qa':4} {'served':>8} {'ok/tot':>10} {'origin':>12} {'alphabet':>8} {'soft':>5}")
    for r in results:
        if "agree" in r:
            print(f"{r['qa']:4} {r['agree']:8.3f} {r['ok']}/{r['tot']:<4} "
                  f"{r.get('origin','?'):>12} {r.get('alphabet','?'):>8} "
                  f"{str(r.get('soft_student')):>5}")
        else:
            print(f"{r.get('qa','?'):4} FAILED")
    out = REPO / "data" / "standalone_seed_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
