"""khop-in-real-text probe (B1): does wikitext hold recoverable 2-hop chains?

Pre-registration (SIGNED, fixed thresholds — do not tune):
  /home/allans/code/PIL_KHOP_REALTEXT_PREREG.md

Scope note (why [lo,hi] does not port; bridge-set membership replaces it):
  docs/notes/khop.md

The certified 2-hop chaining (mir_khop2 / _khop_predict) is reused EXACTLY, with the ONE
mechanical change: the synthetic range gate on the query is replaced by set-membership on the
bridge B discovered by hop 1 (checked AFTER hop 1). Bridge set is mined from FIT bigram edges.

A DEAD (registered negative) result is a complete, publishable outcome — do not tune.

Usage: .venv/bin/python experiments/campaign_khop_realtext.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# Registered constants (fixed by the SIGNED prereg — do not tune after seeing results).
STRIDE = 12
EDGE_MINSUPP = 2
MIN_IN_DEGREE = 2
MIN_OUT_DEGREE = 2
RECOVERY_BAR = 0.005
ABLATION_SEED = 0
MIN_BRIDGE_SET = 2
MIN_FIRED = 200
OUTPUT = REPO / "data" / "khop_realtext_pilot.json"


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable; no dataset load)
# ---------------------------------------------------------------------------


def mine_bridge_set(
    key_pop: torch.Tensor,
    val_pop: torch.Tensor,
    edge_minsupp: int = EDGE_MINSUPP,
    min_in_degree: int = MIN_IN_DEGREE,
    min_out_degree: int = MIN_OUT_DEGREE,
) -> torch.Tensor:
    """Mine bridge tokens from FIT bigram edges (A, C).

    An edge (A, C) is observed only if its support >= edge_minsupp.
    For each token B:
      in_deg(B)  = # distinct A with observed (A, B)
      out_deg(B) = # distinct C with observed (B, C)
    bridge_set = {B : in_deg >= min_in_degree and out_deg >= min_out_degree}.

    Returns a sorted 1-D long tensor of bridge token ids.
    """
    key_pop = key_pop.long().reshape(-1)
    val_pop = val_pop.long().reshape(-1)
    if key_pop.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    pairs, counts = torch.stack([key_pop, val_pop]).unique(dim=1, return_counts=True)
    obs = counts >= edge_minsupp
    if not bool(obs.any()):
        return torch.empty(0, dtype=torch.long)
    src = pairs[0][obs]
    dst = pairs[1][obs]
    # Each remaining (src, dst) row is a unique edge (unique already collapsed counts).
    # in_deg(B) = # rows with dst == B; out_deg(B) = # rows with src == B.
    uniq_dst, in_deg = dst.unique(return_counts=True)
    uniq_src, out_deg = src.unique(return_counts=True)
    # Align degrees on the token universe that appears as either endpoint.
    universe = torch.cat([uniq_src, uniq_dst]).unique()
    # searchsorted maps each universe token into the sorted uniq_* tables.
    idx_in = torch.searchsorted(uniq_dst, universe)
    idx_in = idx_in.clamp(max=max(len(uniq_dst) - 1, 0))
    hit_in = (len(uniq_dst) > 0) & (uniq_dst[idx_in] == universe)
    in_vals = torch.where(hit_in, in_deg[idx_in], torch.zeros_like(universe))

    idx_out = torch.searchsorted(uniq_src, universe)
    idx_out = idx_out.clamp(max=max(len(uniq_src) - 1, 0))
    hit_out = (len(uniq_src) > 0) & (uniq_src[idx_out] == universe)
    out_vals = torch.where(hit_out, out_deg[idx_out], torch.zeros_like(universe))

    keep = (in_vals >= min_in_degree) & (out_vals >= min_out_degree)
    return universe[keep].long().sort().values


def khop_bridge_gate(
    ids_batch: torch.Tensor,
    bridge_set_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized 2-hop with bridge-set membership gate (the ONE change vs mir_khop2).

    Hop 1 always runs (find p1, bridge b = ctx[p1+1]). The rule fires only if b is in the
    mined bridge set; then hop 2 runs with the certified bridge-site exclusion excl = p1+1.
    Everything else is an exact copy of mir_khop2 with L generalized to W = ids_batch.shape[1].

    Returns (pred, b, p1, ok) where ok is the fire mask.
    """
    ids_batch = ids_batch.long()
    n, w = ids_batch.shape
    ar = torch.arange(n, device=ids_batch.device)
    m1 = ids_batch[:, :-1] == ids_batch[:, -1:]
    p1 = (m1.float() * torch.arange(1, w, device=ids_batch.device)).argmax(1)
    b = ids_batch[ar, (p1 + 1).clamp(max=w - 1)]
    ok1 = m1.any(1) & torch.isin(b, bridge_set_tensor)  # <-- the ONE gate swap
    m2 = ids_batch[:, :-1] == b.unsqueeze(1)
    excl = p1 + 1
    ok_ex = excl <= w - 2
    m2[ar[ok_ex], excl[ok_ex]] = False
    ok = ok1 & m2.any(1)
    p2 = (m2.float() * torch.arange(1, w, device=ids_batch.device)).argmax(1)
    pred = torch.where(ok, ids_batch[ar, (p2 + 1).clamp(max=w - 1)], torch.full_like(b, -1))
    return pred, b, p1, ok


def khop_bridge_ablated(
    ids_batch: torch.Tensor,
    bridge_set_tensor: torch.Tensor,
    b_real: torch.Tensor,
    p1_real: torch.Tensor,
    seed: int = ABLATION_SEED,
) -> torch.Tensor:
    """Same rule form as khop_bridge_gate hop 2, but with a wrong bridge (p1/excl fixed).

    For every row, replace bridge b with a deterministic different member of bridge_set,
    then re-run hop 2 with the same exclusion. On no match, returns -1 (counts as incorrect
    against the real rule's fixed fired-subset denominator).
    """
    ids_batch = ids_batch.long()
    n, w = ids_batch.shape
    ar = torch.arange(n, device=ids_batch.device)
    bset = bridge_set_tensor.long()
    k = len(bset)
    if k < 2:
        raise ValueError(f"bridge ablation requires len(bridge_set) >= 2, got {k}")
    idx_true = torch.searchsorted(
        bset, b_real.long().clamp(min=int(bset.min()), max=int(bset.max()))
    )
    idx_true = idx_true.clamp(max=k - 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    offset = torch.randint(1, k, (n,), generator=g).to(ids_batch.device)  # 1..K-1, never 0
    idx_wrong = (idx_true + offset) % k  # guaranteed != idx_true
    b_wrong = bset[idx_wrong]
    m2 = ids_batch[:, :-1] == b_wrong.unsqueeze(1)
    excl = p1_real + 1
    ok_ex = excl <= w - 2
    m2[ar[ok_ex], excl[ok_ex]] = False
    ok2 = m2.any(1)
    p2 = (m2.float() * torch.arange(1, w, device=ids_batch.device)).argmax(1)
    return torch.where(ok2, ids_batch[ar, (p2 + 1).clamp(max=w - 1)], torch.full_like(b_real, -1))


def score_recovery(
    pred: torch.Tensor,
    ablated_pred: torch.Tensor,
    baseline_agree: torch.Tensor,
    gold: torch.Tensor,
) -> dict[str, Any]:
    """Section-8 arithmetic on a fixed fired subset (all tensors already subsetted, length n).

    baseline_agree is a boolean (or 0/1) tensor: True iff 1-hop OR memorizer matches gold.
    """
    pred = pred.long()
    ablated_pred = ablated_pred.long()
    gold = gold.long()
    base = baseline_agree.bool()
    n = int(pred.numel())
    if n == 0:
        return {
            "n": 0,
            "agree_2hop": 0.0,
            "agree_2hop_ablated": 0.0,
            "agree_baseline": 0.0,
            "recovery": 0.0,
            "recovery_ablated": 0.0,
            "one_sided_count": 0,
            "reverse_count": 0,
            "one_sided_frac": 0.0,
            "reverse_frac": 0.0,
        }
    hit_2 = pred == gold
    hit_ab = ablated_pred == gold
    agree_2hop = float(hit_2.float().mean())
    agree_2hop_ablated = float(hit_ab.float().mean())
    agree_baseline = float(base.float().mean())
    recovery = agree_2hop - agree_baseline
    recovery_ablated = agree_2hop_ablated - agree_baseline
    one_sided_count = int((hit_2 & ~base).sum())
    reverse_count = int((~hit_2 & base).sum())
    return {
        "n": n,
        "agree_2hop": agree_2hop,
        "agree_2hop_ablated": agree_2hop_ablated,
        "agree_baseline": agree_baseline,
        "recovery": recovery,
        "recovery_ablated": recovery_ablated,
        "one_sided_count": one_sided_count,
        "reverse_count": reverse_count,
        "one_sided_frac": one_sided_count / n,
        "reverse_frac": reverse_count / n,
    }


def verdict_from_recovery(recovery: float, recovery_ablated: float, bar: float = RECOVERY_BAR) -> str:
    """FIRES iff recovery clears the bar AND ablated recovery does not (guard holds)."""
    guard_holds = recovery_ablated < bar
    if recovery >= bar and guard_holds:
        return "FIRES"
    return "DEAD"


def fit_bigram_population(
    ids: torch.Tensor,
    yv: torch.Tensor,
    tr: torch.Tensor,
    stride: int = STRIDE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FIT-only bigram edge population (mirrors fit_frame k=1 construction)."""
    w = ids[tr[::stride]]
    key_pop = torch.cat([w[:, :-1].reshape(-1), ids[tr][:, -1]])
    val_pop = torch.cat([w[:, 1:].reshape(-1), yv[tr]])
    return key_pop, val_pop


def git_head(repo: Path = REPO) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Pilot (expensive: loads wikitext + fits tables)
# ---------------------------------------------------------------------------


def run_pilot() -> dict[str, Any]:
    os.environ.setdefault("WYLY_TAG", "pythia70m")
    os.environ.setdefault("WYLY_DS", "wikitext")
    import wyly_lm_v5 as v5  # noqa: PLC0415 — deferred so module import stays cheap
    from wyly_lm_v5 import fit_kgram, mir_kgram  # noqa: PLC0415

    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]  # teacher-label targets in the same dense integer space as ids
    w_len = int(ids.shape[1])
    tr_size, te_size = int(len(tr)), int(len(te))
    # Vocab for k-gram radix encoding = size of the dense integer alphabet ids ranges over
    # (len(uv) after load_ds unique remapping) — matches wyly_lm_v5.main(), NOT v5.V
    # (label-class count, top-4096 only). Using v5.V aliases k>=2 suffix keys.
    vocab = int(len(uv))
    tag = os.environ.get("WYLY_TAG", "pythia70m")
    ds = os.environ.get("WYLY_DS", "wikitext")

    print(
        f"loaded ids={tuple(ids.shape)} tr={tr_size} te={te_size} W={w_len} vocab={vocab}",
        flush=True,
    )
    if abs(tr_size - 65031) > 500 or abs(te_size - 11477) > 500:
        print(
            f"WARN: split sizes differ a lot from expected tr=65031 te=11477 "
            f"(got tr={tr_size} te={te_size}) — data build may have changed",
            flush=True,
        )

    # --- mine bridge set on FIT only ---
    key_pop, val_pop = fit_bigram_population(ids, yv, tr, stride=STRIDE)
    bridge_set_tensor = mine_bridge_set(
        key_pop, val_pop,
        edge_minsupp=EDGE_MINSUPP,
        min_in_degree=MIN_IN_DEGREE,
        min_out_degree=MIN_OUT_DEGREE,
    )
    bridge_set_tensor = bridge_set_tensor.to(ids.device)
    bridge_set_size = int(len(bridge_set_tensor))
    print(
        f"bridge_set size={bridge_set_size} "
        f"(EDGE_MINSUPP={EDGE_MINSUPP} MIN_IN={MIN_IN_DEGREE} MIN_OUT={MIN_OUT_DEGREE} "
        f"STRIDE={STRIDE})",
        flush=True,
    )

    if bridge_set_size < MIN_BRIDGE_SET:
        msg = f"STOP: bridge set too small for ablation (size={bridge_set_size})"
        print(msg, flush=True)
        result = _stop_result(
            "STOP", msg, bridge_set_size, tr_size, te_size, w_len, vocab, tag, ds
        )
        _write_json(result)
        return result

    # --- 2-hop on held-out te ---
    pred, b, p1, ok = khop_bridge_gate(ids[te], bridge_set_tensor)
    fired = ok
    n_fired = int(fired.sum())
    print(f"fired subset n={n_fired} / te={te_size}", flush=True)
    if n_fired < MIN_FIRED:
        msg = f"STOP: fired subset too small (n={n_fired})"
        print(msg, flush=True)
        result = _stop_result(
            "STOP", msg, bridge_set_size, tr_size, te_size, w_len, vocab, tag, ds,
            n_fired=n_fired,
        )
        _write_json(result)
        return result

    # --- baselines on te (FIT-trained tables; file defaults stride=12,minsupp=2,mindet=0.5) ---
    tab1, _ = fit_kgram(ids, yv, tr, 1, vocab)
    onehop_fn = mir_kgram(tab1, 1, vocab)
    tab2, _ = fit_kgram(ids, yv, tr, 2, vocab)
    tab3, _ = fit_kgram(ids, yv, tr, 3, vocab)
    mem2_fn = mir_kgram(tab2, 2, vocab)
    mem3_fn = mir_kgram(tab3, 3, vocab)

    onehop_pred = onehop_fn(ids[te])
    mem2_pred = mem2_fn(ids[te])
    mem3_pred = mem3_fn(ids[te])
    # longest-suffix-wins (file preference order: k=3 then k=2)
    memorizer_pred = torch.where(mem3_pred != -1, mem3_pred, mem2_pred)
    gold = yv[te]
    baseline_agree = (onehop_pred == gold) | (memorizer_pred == gold)

    # --- bridge ablation ---
    ablated_pred = khop_bridge_ablated(
        ids[te], bridge_set_tensor, b, p1, seed=ABLATION_SEED
    )

    # --- metrics on fired subset ---
    sc = score_recovery(
        pred[fired], ablated_pred[fired], baseline_agree[fired], gold[fired]
    )
    agree_1hop = float((onehop_pred[fired] == gold[fired]).float().mean())
    agree_memorizer = float((memorizer_pred[fired] == gold[fired]).float().mean())
    agree_baseline = sc["agree_baseline"]
    recovery = sc["recovery"]
    recovery_ablated = sc["recovery_ablated"]
    verdict = verdict_from_recovery(recovery, recovery_ablated, RECOVERY_BAR)
    guard_holds = recovery_ablated < RECOVERY_BAR
    ablation_fate = "VANISHED" if guard_holds else "SURVIVED"

    result: dict[str, Any] = {
        "n_fired": n_fired,
        "bridge_set_size": bridge_set_size,
        "edge_minsupp": EDGE_MINSUPP,
        "min_in_degree": MIN_IN_DEGREE,
        "min_out_degree": MIN_OUT_DEGREE,
        "stride": STRIDE,
        "recovery": recovery,
        "recovery_ablated": recovery_ablated,
        "agree_2hop": sc["agree_2hop"],
        "agree_2hop_ablated": sc["agree_2hop_ablated"],
        "agree_baseline": agree_baseline,
        "agree_1hop": agree_1hop,
        "agree_memorizer": agree_memorizer,
        "one_sided_count": sc["one_sided_count"],
        "reverse_count": sc["reverse_count"],
        "one_sided_frac": sc["one_sided_frac"],
        "reverse_frac": sc["reverse_frac"],
        "verdict": verdict,
        "ablation_seed": ABLATION_SEED,
        "ablation_fate": ablation_fate,
        "recovery_bar": RECOVERY_BAR,
        "te_size": te_size,
        "tr_size": tr_size,
        "W": w_len,
        "vocab": vocab,
        "wyly_tag": tag,
        "wyly_ds": ds,
        "git_commit": git_head(),
    }
    _write_json(result)
    _print_scoreboard(result)
    return result


def _stop_result(
    verdict: str,
    msg: str,
    bridge_set_size: int,
    tr_size: int,
    te_size: int,
    w_len: int,
    vocab: int,
    tag: str,
    ds: str,
    n_fired: int = 0,
) -> dict[str, Any]:
    return {
        "n_fired": n_fired,
        "bridge_set_size": bridge_set_size,
        "edge_minsupp": EDGE_MINSUPP,
        "min_in_degree": MIN_IN_DEGREE,
        "min_out_degree": MIN_OUT_DEGREE,
        "stride": STRIDE,
        "recovery": None,
        "recovery_ablated": None,
        "agree_2hop": None,
        "agree_2hop_ablated": None,
        "agree_baseline": None,
        "agree_1hop": None,
        "agree_memorizer": None,
        "one_sided_count": None,
        "reverse_count": None,
        "verdict": verdict,
        "stop_message": msg,
        "ablation_seed": ABLATION_SEED,
        "te_size": te_size,
        "tr_size": tr_size,
        "W": w_len,
        "vocab": vocab,
        "wyly_tag": tag,
        "wyly_ds": ds,
        "git_commit": git_head(),
    }


def _write_json(result: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT}", flush=True)


def _print_scoreboard(r: dict[str, Any]) -> None:
    print("=" * 60, flush=True)
    print("khop-in-real-text pilot scoreboard", flush=True)
    print("=" * 60, flush=True)
    keys = [
        "n_fired", "bridge_set_size", "edge_minsupp", "min_in_degree", "min_out_degree",
        "stride", "agree_2hop", "agree_2hop_ablated", "agree_baseline", "agree_1hop",
        "agree_memorizer", "recovery", "recovery_ablated", "one_sided_count",
        "reverse_count", "one_sided_frac", "reverse_frac", "ablation_fate",
        "ablation_seed", "te_size", "tr_size", "W", "vocab", "wyly_tag", "wyly_ds",
        "git_commit", "verdict",
    ]
    for k in keys:
        if k in r:
            print(f"  {k}: {r[k]}", flush=True)
    print(
        f"VERDICT: {r['verdict']} recovery={r['recovery']:.6f} "
        f"recovery_ablated={r['recovery_ablated']:.6f} n={r['n_fired']}",
        flush=True,
    )
    print(
        f"  ablation recovery {r['ablation_fate']} "
        f"(guard holds iff recovery_ablated < {RECOVERY_BAR})",
        flush=True,
    )


def main() -> int:
    result = run_pilot()
    if result.get("verdict") == "STOP":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
