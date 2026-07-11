"""Cross-domain residual template transfer (standalone).

Same ``ResidualFamily`` propose/admit code on two domains:

  1. **SCAN/simple** — nfold + prefix_body + structural (existing residual leaves)
  2. **listops** — synthetic item language with x2/x3 n-fold and ``and`` concat;
     only short composites in train for some atoms (hold out bare leaves) so residual
     n-fold is required for test composition.

Ablations
  - hardcode: apply all proposed residuals (no val admit)
  - leaf_admit: greedy val-marginal on residual candidates (SCAN residual campaign)
  - template_admit: meta-admit whole template_ids, then apply their candidates

Diagnostics: frac of proposed residuals with pattern_agnostic=True (reusable patterns).

Standalone: synthetic + SCAN corpus, SOFT=0, no teacher.

Run:
  cd pil && .venv/bin/python -u experiments/campaign_residual_transfer.py
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.residual_template import (  # noqa: E402
    MapDict,
    ResidualFamily,
    listops_domain_atoms,
    scan_domain_atoms,
)

sys.path.insert(0, str(REPO / "experiments"))
import campaign_scan_learned as learned  # noqa: E402
import campaign_scan_prims as prims_camp  # noqa: E402
import campaign_scan_standalone as base  # noqa: E402

SPLITS_SCAN = ["simple"]  # hard residual stress; length/addprim already closed
LEAF_ALWAYS = frozenset({"dir"})
UNARY = ("twice", "thrice", "around", "opposite")
BINARY = ("and", "after")


def log(m=""):
    print(m, flush=True)


# --- listops synthetic domain ------------------------------------------------

def _listops_expand(tokens: list[str], leaves: MapDict) -> list[str] | None:
    """Expand listops: and (concat), x2/x3 (n-fold), exact leaf lookup."""
    if not tokens:
        return []
    for i, t in enumerate(tokens):
        if t == "and":
            left = _listops_expand(tokens[:i], leaves)
            right = _listops_expand(tokens[i + 1 :], leaves)
            if left is None or right is None:
                return None
            return left + right
    key = tuple(tokens)
    if key in leaves:
        return list(leaves[key])
    if len(tokens) >= 2 and tokens[-1] == "x2":
        b = _listops_expand(tokens[:-1], leaves)
        return None if b is None else b + b
    if len(tokens) >= 2 and tokens[-1] == "x3":
        b = _listops_expand(tokens[:-1], leaves)
        return None if b is None else b + b + b
    return None


def build_listops(
    *,
    seed: int = 0,
    n_train: int = 800,
    n_test: int = 400,
    holdout_bare: frozenset[str] = frozenset({"c", "d"}),
) -> dict:
    """Synthetic listops with systematic bare-leaf holdout (residual stress).

    Items a,b always appear bare in train; c,d only as ``c x2`` / ``d x3`` short
    maps so residual n-fold must recover bare leaves for composed test cmds.
    """
    rng = random.Random(seed)
    items = ["a", "b", "c", "d"]
    # bare leaves as MapDict: (item,) -> [ITEM]
    bare: MapDict = {(x,): [x.upper()] for x in items}

    def gold(tokens: list[str]) -> list[str] | None:
        return _listops_expand(tokens, bare)

    # Induction maps: composites always (sources for nfold residual).
    # Expand base: bare leaves only for non-holdout items — composites are NOT
    # stored as exact maps, so expand must peel x2/x3 and needs residual bare leaves.
    induction: MapDict = {}
    expand_base: MapDict = {}
    for x in items:
        if x not in holdout_bare:
            expand_base[(x,)] = list(bare[(x,)])
        induction[(x, "x2")] = list(bare[(x,)]) * 2
        induction[(x, "x3")] = list(bare[(x,)]) * 3

    def sample_cmd(force_holdout: bool) -> list[str]:
        # compose: item (x2|x3) (and item (x2|x3)?)* — always use folds so peel needs bare
        n_parts = rng.randint(1, 3)
        parts = []
        for j in range(n_parts):
            if force_holdout and j == 0:
                x = rng.choice(list(holdout_bare))
            elif force_holdout:
                x = rng.choice(items)
            else:
                x = rng.choice([i for i in items if i not in holdout_bare])
            parts.append([x, rng.choice(["x2", "x3"])])
        out: list[str] = []
        for i, p in enumerate(parts):
            if i:
                out.append("and")
            out.extend(p)
        return out

    train_cmds, train_out = [], []
    # inject induction composites as train evidence (not used as expand exact maps)
    for src, tgt in induction.items():
        train_cmds.append(list(src))
        train_out.append(list(tgt))
    for src, tgt in expand_base.items():
        train_cmds.append(list(src))
        train_out.append(list(tgt))
    # train mixes easy and holdout compositions so val marginal sees residual need
    while len(train_cmds) < n_train:
        cmd = sample_cmd(force_holdout=(rng.random() < 0.5))
        g = gold(cmd)
        if g is None:
            continue
        train_cmds.append(cmd)
        train_out.append(g)

    test_cmds, test_out = [], []
    while len(test_cmds) < n_test:
        cmd = sample_cmd(force_holdout=True)
        g = gold(cmd)
        if g is None:
            continue
        test_cmds.append(cmd)
        test_out.append(g)

    return {
        "induction": induction,
        "expand_base": expand_base,
        "train_in": train_cmds,
        "train_out": train_out,
        "test_in": test_cmds,
        "test_out": test_out,
        "holdout_bare": sorted(holdout_bare),
        "gold_leaves": bare,
    }


def listops_score(leaves: MapDict, pairs: list[tuple[list[str], list[str]]]) -> float:
    ok = tot = 0
    for cmd, gold in pairs:
        pred = _listops_expand(cmd, leaves)
        if pred is not None and pred == gold:
            ok += 1
        tot += 1
    return ok / max(tot, 1)


def run_listops() -> dict:
    data = build_listops()
    induction = data["induction"]
    expand_base = data["expand_base"]
    test_pairs = list(zip(data["test_in"], data["test_out"], strict=True))
    tr = list(zip(data["train_in"], data["train_out"], strict=True))
    rng = random.Random(0)
    rng.shuffle(tr)
    nval = max(1, len(tr) // 5)
    val_pairs = tr[:nval]

    # Propose residuals from induction composites ∪ expand_base (like SCAN short maps)
    short_fit: MapDict = {**induction, **expand_base}

    def val_score(maps: MapDict) -> float:
        # expand uses only bare leaves (strip composite keys for scoring)
        leaves = {k: v for k, v in maps.items() if len(k) == 1}
        return listops_score(leaves, val_pairs)

    family = ResidualFamily(listops_domain_atoms())
    cands = family.propose(short_fit)
    # hardcode: expand_base + all residual proposals
    res_map = family.propose_map(short_fit)
    maps_hard = {**expand_base, **res_map}
    maps_leaf, log_leaf = family.admit(short_fit, val_score, thresh=1e-4)
    # admit returns full short_fit∪residuals — keep only bare leaves for expand
    leaves_leaf = {k: v for k, v in maps_leaf.items() if len(k) == 1}
    en_t, maps_tmpl, log_tmpl = family.admit_templates(short_fit, val_score, thresh=1e-4)
    leaves_tmpl = {k: v for k, v in maps_tmpl.items() if len(k) == 1}

    acc_base = listops_score(expand_base, test_pairs)
    acc_hard = listops_score(maps_hard, test_pairs)
    acc_leaf = listops_score(leaves_leaf, test_pairs)
    acc_tmpl = listops_score(leaves_tmpl, test_pairs)
    diag = family.diagnostics(
        short_fit,
        admitted_src=[c.src for c in cands if c.src in leaves_leaf and c.src not in expand_base],
    )

    log("\n=== listops residual transfer ===")
    log(f"  induction={len(induction)} expand_base={len(expand_base)} "
        f"proposed={len(cands)} holdout={data['holdout_bare']}")
    log(f"  base (no residual)  {acc_base:.3f}")
    log(f"  hardcode residuals  {acc_hard:.3f}")
    log(f"  leaf admit          {acc_leaf:.3f}")
    log(f"  template admit      {acc_tmpl:.3f}  enabled={sorted(en_t)}")
    log(f"  frac_proposed_agnostic {diag['frac_proposed_agnostic']:.2f}")

    return {
        "domain": "listops",
        "n_test": len(test_pairs),
        "n_induction": len(induction),
        "n_expand_base": len(expand_base),
        "acc_base": acc_base,
        "acc_hardcode": acc_hard,
        "acc_leaf_admit": acc_leaf,
        "acc_template_admit": acc_tmpl,
        "enabled_templates": sorted(en_t),
        "proposed": [
            {"src": " ".join(c.src), "template_id": c.template_id,
             "agnostic": c.pattern_agnostic}
            for c in cands
        ],
        "diagnostics": diag,
        "admit_log_leaf": log_leaf[:20],
        "admit_log_template": log_tmpl[:20],
        "origin": "standalone",
    }


# --- SCAN simple via same ResidualFamily -------------------------------------

def run_scan_simple() -> dict:
    ensure_scan()
    blob, codec = base.load_split("simple")
    tr_in, tr_out = blob["train_in"], blob["train_out"]
    te_in, te_out = blob["test_in"], blob["test_out"]
    fit, val = prims_camp.split_fit_val(tr_in, tr_out, val_frac=0.1, seed=0)
    fit_in = [c for c, _ in fit]
    fit_out = [a for _, a in fit]
    test_pairs = list(zip(te_in, te_out, strict=True))
    full_comb = set(UNARY) | set(BINARY)

    short = base.mine_scan_grammar(fit_in, fit_out, codec, residual_leaves=False)
    family = ResidualFamily(scan_domain_atoms())
    cands = family.propose(short)

    def score_maps(maps: MapDict) -> float:
        return learned.score_exact(val, codec, maps, full_comb | set(LEAF_ALWAYS))

    maps_hard = {**short, **family.propose_map(short)}
    maps_leaf, log_leaf = family.admit(short, score_maps, thresh=1e-4)
    en_t, maps_tmpl, log_tmpl = family.admit_templates(short, score_maps, thresh=1e-4)

    # Fix full combinator set for residual ablations (isolates residual gain; one
    # combinator admit would dominate runtime and confound residual measurement).
    en = full_comb | set(LEAF_ALWAYS)

    def stack_acc(maps: MapDict) -> float:
        return learned.score_exact(test_pairs, codec, maps, en)

    acc_base = stack_acc(short)
    acc_hard = stack_acc(maps_hard)
    acc_leaf = stack_acc(maps_leaf)
    acc_tmpl = stack_acc(maps_tmpl)
    acc_pc = base.eval_prim_comp(
        base.mine_scan_grammar(tr_in, tr_out, codec, residual_leaves=True),
        te_in, te_out, codec,
    )[0]

    diag = family.diagnostics(
        short,
        admitted_src=[c.src for c in cands if c.src in maps_leaf and c.src not in short],
    )

    log("\n=== SCAN/simple residual transfer ===")
    log(f"  short={len(short)} proposed={len(cands)}")
    log(f"  prim_compose (full+res) {acc_pc:.3f}")
    log(f"  stack short only        {acc_base:.3f}")
    log(f"  hardcode residuals      {acc_hard:.3f}")
    log(f"  leaf admit              {acc_leaf:.3f}")
    log(f"  template admit          {acc_tmpl:.3f}  enabled={sorted(en_t)}")
    log(f"  frac_proposed_agnostic  {diag['frac_proposed_agnostic']:.2f}")

    return {
        "domain": "scan_simple",
        "n_test": len(te_in),
        "n_short_fit": len(short),
        "prim_compose": acc_pc,
        "acc_base": acc_base,
        "acc_hardcode": acc_hard,
        "acc_leaf_admit": acc_leaf,
        "acc_template_admit": acc_tmpl,
        "enabled_templates": sorted(en_t),
        "proposed": [
            {"src": " ".join(c.src), "template_id": c.template_id,
             "agnostic": c.pattern_agnostic}
            for c in cands
        ],
        "diagnostics": diag,
        "admit_log_leaf": log_leaf[:20],
        "origin": "standalone",
    }


def ensure_scan():
    if not (REPO / "data" / "scan_simple.pt").exists():
        subprocess.check_call(
            [sys.executable, str(REPO / "experiments" / "build_scan.py")], cwd=str(REPO))


def main():
    results = []
    results.append(run_listops())
    if os.environ.get("SKIP_SCAN", "") != "1":
        results.append(run_scan_simple())

    log("\n" + "=" * 72)
    log("RESIDUAL TEMPLATE TRANSFER SCOREBOARD")
    log("=" * 72)
    log(f"{'domain':14} {'base':>7} {'hard':>7} {'leafAd':>7} {'tmplAd':>7} "
        f"{'%agn':>7} {'templates':>20}")
    for r in results:
        agn = r["diagnostics"]["frac_proposed_agnostic"]
        tmpls = ",".join(r.get("enabled_templates") or [])[:20]
        log(f"{r['domain']:14} {r['acc_base']:7.3f} {r['acc_hardcode']:7.3f} "
            f"{r['acc_leaf_admit']:7.3f} {r['acc_template_admit']:7.3f} "
            f"{agn:7.2f} {tmpls:>20}")
    out = REPO / "data" / "residual_transfer_scoreboard.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {out}")
    log("\nReading: same ResidualFamily code on listops (x2/x3) and SCAN (twice/thrice). "
        "pattern_agnostic frac ≈ reusable template share; structural seeds are "
        "domain-specific. Ablation hardcode vs leaf/template admit.")


if __name__ == "__main__":
    main()
