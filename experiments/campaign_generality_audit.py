"""Honest generality audit: pattern-class agnostic vs data-induced provenance.

``frac_*_agnostic`` measures agnostic *pattern class* (nfold/prefix_body/relation_atom),
NOT whether structure was discovered from data. CFQ's ``relation_atom`` is a pure
content pass-through (``pattern_agnostic=True``) yet has zero induced structure —
so agnostic≈1.00 overstates generality. This campaign reports the additive honest
metrics ``frac_*_induced`` / ``provenance_*`` from ``ResidualFamily.diagnostics``.

Domains:
  1. listops (induce_only markers)
  2. SCAN hand pack (supplied nfold + prefix)
  3. SCAN induce_only (markers + prefixes recovered from short maps)
  4. CFQ mcd1 (relation_atom vote pass-through)

Also checks that SCAN induce_only recovers ``I_TURN_LEFT``/``I_TURN_RIGHT`` and
does not regress test exact-match vs the hand pack (within 0.001).

Run:
  cd pil && .venv/bin/python -u experiments/campaign_generality_audit.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.residual_template import (  # noqa: E402
    CFQ_TEMPLATES,
    MapDict,
    ResidualFamily,
    cfq_domain_atoms,
    induce_prefix_tokens,
    listops_domain_atoms,
    resolve_prefix_tokens,
    scan_domain_atoms,
)

sys.path.insert(0, str(REPO / "experiments"))
import campaign_cfq_residual as cfq  # noqa: E402
import campaign_residual_transfer as xfer  # noqa: E402
import campaign_scan_learned as learned  # noqa: E402
import campaign_scan_prims as prims_camp  # noqa: E402
import campaign_scan_standalone as base  # noqa: E402

UNARY = xfer.UNARY
BINARY = xfer.BINARY
LEAF_ALWAYS = xfer.LEAF_ALWAYS
EXPECTED_SCAN_PREFIXES = frozenset({"I_TURN_LEFT", "I_TURN_RIGHT"})
ACC_TOL = 0.001


def log(m: str = "") -> None:
    print(m, flush=True)


def _jsonable(obj: object) -> object:
    """Make frozenset/tuple keys JSON-serializable."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                sk = " ".join(str(x) for x in k)
            else:
                sk = str(k)
            out[sk] = _jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return obj


# --- listops -----------------------------------------------------------------

def run_listops() -> dict:
    data = xfer.build_listops()
    induction = data["induction"]
    expand_base = data["expand_base"]
    tr = list(zip(data["train_in"], data["train_out"], strict=True))
    rng = random.Random(0)
    rng.shuffle(tr)
    nval = max(1, len(tr) // 5)
    val_pairs = tr[:nval]
    short_fit: MapDict = {**induction, **expand_base}

    def val_score(maps: MapDict) -> float:
        leaves = {k: v for k, v in maps.items() if len(k) == 1}
        return xfer.listops_score(leaves, val_pairs)

    family = ResidualFamily(listops_domain_atoms(induce_only=True))
    cands = family.propose(short_fit)
    maps_leaf, admit_log = family.admit(short_fit, val_score, thresh=1e-4)
    leaves_leaf = {k: v for k, v in maps_leaf.items() if len(k) == 1}
    diag = family.diagnostics(
        short_fit,
        admitted_src=[
            c.src for c in cands if c.src in leaves_leaf and c.src not in expand_base
        ],
    )
    log("\n=== listops ===")
    log(f"  proposed={len(cands)} admitted={diag['n_admitted']}")
    log(f"  frac_admitted_agnostic={diag['frac_admitted_agnostic']:.3f}  "
        f"frac_admitted_induced={diag['frac_admitted_induced']:.3f}")
    log(f"  provenance_admitted={diag['provenance_admitted']}")
    return {
        "domain": "listops",
        "diagnostics": diag,
        "n_proposed": len(cands),
        "admit_log": admit_log[:20],
    }


# --- SCAN --------------------------------------------------------------------

def _scan_setup() -> tuple:
    xfer.ensure_scan()
    blob, codec = base.load_split("simple")
    tr_in, tr_out = blob["train_in"], blob["train_out"]
    te_in, te_out = blob["test_in"], blob["test_out"]
    fit, val = prims_camp.split_fit_val(tr_in, tr_out, val_frac=0.1, seed=0)
    fit_in = [c for c, _ in fit]
    fit_out = [a for _, a in fit]
    test_pairs = list(zip(te_in, te_out, strict=True))
    short = base.mine_scan_grammar(fit_in, fit_out, codec, residual_leaves=False)
    full_comb = set(UNARY) | set(BINARY)
    en = full_comb | set(LEAF_ALWAYS)
    return short, val, test_pairs, codec, en


def run_scan(tag: str, *, induce_only: bool) -> dict:
    short, val, test_pairs, codec, en = _scan_setup()
    domain = scan_domain_atoms(induce_only=induce_only)
    family = ResidualFamily(domain)
    cands = family.propose(short)

    def score_maps(maps: MapDict) -> float:
        return learned.score_exact(val, codec, maps, en)

    maps_leaf, admit_log = family.admit(short, score_maps, thresh=1e-4)
    test_acc = learned.score_exact(test_pairs, codec, maps_leaf, en)
    diag = family.diagnostics(
        short,
        admitted_src=[c.src for c in cands if c.src in maps_leaf and c.src not in short],
    )

    induced_prefs = (
        induce_prefix_tokens(short) if domain.auto_induce_prefixes else frozenset()
    )
    resolved_prefs = resolve_prefix_tokens(short, domain)
    prefixes_ok = EXPECTED_SCAN_PREFIXES.issubset(resolved_prefs)

    log(f"\n=== SCAN ({tag}) induce_only={induce_only} ===")
    log(f"  short={len(short)} proposed={len(cands)} admitted={diag['n_admitted']}")
    log(f"  test_acc={test_acc:.4f}")
    log(f"  frac_admitted_agnostic={diag['frac_admitted_agnostic']:.3f}  "
        f"frac_admitted_induced={diag['frac_admitted_induced']:.3f}")
    log(f"  provenance_admitted={diag['provenance_admitted']}")
    log(f"  domain.prefix_tokens={sorted(domain.prefix_tokens)} "
        f"auto_induce_prefixes={domain.auto_induce_prefixes}")
    log(f"  induced_prefix_tokens={sorted(induced_prefs)}")
    log(f"  resolved_prefix_tokens={sorted(resolved_prefs)}")
    log(f"  expected_prefixes_recovered={prefixes_ok}")

    return {
        "domain": tag,
        "induce_only": induce_only,
        "diagnostics": diag,
        "n_proposed": len(cands),
        "test_acc": test_acc,
        "induced_prefix_tokens": sorted(induced_prefs),
        "resolved_prefix_tokens": sorted(resolved_prefs),
        "expected_prefixes": sorted(EXPECTED_SCAN_PREFIXES),
        "prefixes_recovered": prefixes_ok,
        "admit_log": admit_log[:20],
    }


# --- CFQ ---------------------------------------------------------------------

def run_cfq() -> dict:
    cfq.ensure_data(["mcd1"])
    blob = cfq.load_split("mcd1")
    tr_q, tr_a = blob["train_q"], blob["train_sparql"]
    idx = list(range(len(tr_q)))
    random.Random(0).shuffle(idx)
    # modest fit/val for diagnostics-only audit (reuse run_split slicing pattern)
    nval = min(int(os.environ.get("CFQ_MAX_VAL", "2000")), max(1, len(idx) // 10))
    val_i, fit_i = idx[:nval], idx[nval:]
    fit_q = [tr_q[i] for i in fit_i]
    fit_a = [tr_a[i] for i in fit_i]
    val_pairs = [(tr_q[i], tr_a[i]) for i in val_i]

    base_maps = cfq.mine_base_atoms(fit_q, fit_a, max_ns=1, min_support=2)
    multi_votes = cfq.mine_multi_votes(fit_q, fit_a, min_ns=2)
    multi_votes = {k: v for k, v in multi_votes.items() if k not in base_maps}
    top_k = int(os.environ.get("CFQ_RES_TOPK", "80"))
    if len(multi_votes) > top_k:
        multi_votes = dict(sorted(multi_votes.items(), key=lambda kv: -kv[1])[:top_k])

    domain = cfq_domain_atoms(multi_word_path_votes=multi_votes, atom_min_support=3)
    fam = ResidualFamily(domain, templates=CFQ_TEMPLATES)
    cands = fam.propose(base_maps)

    val_admit = val_pairs
    max_va = int(os.environ.get("CFQ_ADMIT_VAL", "400"))
    if len(val_admit) > max_va:
        val_admit = val_admit[:max_va]

    def val_score(maps: MapDict) -> float:
        return cfq.mean_set_f1(val_admit, maps)

    maps_adm, admit_log = fam.admit(
        base_maps, val_score, thresh=1e-4, max_rules=16, celf=False,
    )
    diag = fam.diagnostics(
        base_maps,
        admitted_src=[
            c.src for c in cands if c.src in maps_adm and c.src not in base_maps
        ],
    )
    log("\n=== CFQ/mcd1 ===")
    log(f"  base_atoms={len(base_maps)} proposed={len(cands)} "
        f"admitted={diag['n_admitted']}")
    log(f"  frac_admitted_agnostic={diag['frac_admitted_agnostic']:.3f}  "
        f"frac_admitted_induced={diag['frac_admitted_induced']:.3f}")
    log(f"  provenance_admitted={diag['provenance_admitted']}")
    return {
        "domain": "cfq",
        "diagnostics": diag,
        "n_proposed": len(cands),
        "n_base_atoms": len(base_maps),
        "admit_log": admit_log[:20],
    }


# --- main --------------------------------------------------------------------

def main() -> None:
    log("=" * 72)
    log("GENERALITY AUDIT: pattern-class agnostic vs honest induced provenance")
    log("=" * 72)

    results: list[dict] = []
    results.append(run_listops())
    scan_hand = run_scan("scan_hand", induce_only=False)
    results.append(scan_hand)
    scan_ind = run_scan("scan_induce_only", induce_only=True)
    results.append(scan_ind)
    results.append(run_cfq())

    # SCAN accuracy + prefix recovery gate
    acc_hand = float(scan_hand["test_acc"])
    acc_ind = float(scan_ind["test_acc"])
    prefixes_ok = bool(scan_ind["prefixes_recovered"])
    acc_regressed = (acc_hand - acc_ind) > ACC_TOL

    log("\n" + "=" * 72)
    log("SCAN accuracy comparison (hand pack vs induce_only)")
    log("=" * 72)
    log(f"  scan_hand          test_acc={acc_hand:.4f}")
    log(f"  scan_induce_only   test_acc={acc_ind:.4f}  "
        f"delta={acc_ind - acc_hand:+.4f}")
    log(f"  induced prefixes recovered "
        f"{EXPECTED_SCAN_PREFIXES}?  {'PASS' if prefixes_ok else 'FAIL'}  "
        f"got={scan_ind['resolved_prefix_tokens']}")

    if acc_regressed or not prefixes_ok:
        log("")
        log("!" * 72)
        log("WARNING: SCAN induce_only did not fully recover / regressed accuracy")
        if not prefixes_ok:
            log(f"  - missing expected prefixes among resolved set "
                f"(expected {sorted(EXPECTED_SCAN_PREFIXES)}, "
                f"got {scan_ind['resolved_prefix_tokens']})")
        if acc_regressed:
            log(f"  - test_acc dropped by more than {ACC_TOL}: "
                f"hand={acc_hand:.4f} induce_only={acc_ind:.4f}")
        log("!" * 72)

    scan_check = {
        "acc_hand": acc_hand,
        "acc_induce_only": acc_ind,
        "acc_delta": acc_ind - acc_hand,
        "prefixes_recovered": prefixes_ok,
        "acc_regressed": acc_regressed,
        "warning": acc_regressed or not prefixes_ok,
    }
    for r in results:
        if r["domain"] in ("scan_hand", "scan_induce_only"):
            r["scan_check"] = scan_check

    # scoreboard
    log("\n" + "=" * 72)
    log("GENERALITY SCOREBOARD")
    log("=" * 72)
    log(
        "NOTE: CFQ shows frac_admitted_agnostic≈1.00 but frac_admitted_induced≈0.00 "
        "(the correction). SCAN frac_admitted_induced rises from hand pack to "
        "induce_only pack."
    )
    log(
        f"{'domain':18} {'agn(OLD)':>10} {'ind(HONEST)':>12}  provenance_admitted"
    )
    log("-" * 72)
    for r in results:
        d = r["diagnostics"]
        agn = d["frac_admitted_agnostic"]
        ind = d["frac_admitted_induced"]
        prov = d["provenance_admitted"]
        log(f"{r['domain']:18} {agn:10.3f} {ind:12.3f}  {prov}")

    out_dir = REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "generality_audit.json"
    payload = {
        "results": results,
        "scan_check": scan_check,
        "note": (
            "frac_*_agnostic is pattern-CLASS, not provenance; "
            "frac_*_induced is the honest induced-vs-supplied generality number."
        ),
    }
    out.write_text(json.dumps(_jsonable(payload), indent=2, default=str))
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
