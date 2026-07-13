"""Fast unit tests for the khop-in-real-text pilot (hand-built fixtures only).

No wikitext / load_ds(). Imports pure helpers from campaign_khop_realtext.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from campaign_khop_realtext import (  # noqa: E402
    khop_bridge_ablated,
    khop_bridge_gate,
    mine_bridge_set,
    score_recovery,
    verdict_from_recovery,
)

W = 16  # hand-built window length (not the production W=256)


def _window(filler: int, query: int, overrides: dict[int, int] | None = None) -> list[int]:
    ctx = [filler] * W
    ctx[-1] = query
    for i, v in (overrides or {}).items():
        ctx[i] = v
    return ctx


# ---------------------------------------------------------------------------
# (a) Genuine 2-hop chain: real recovery; ablation destroys it
# ---------------------------------------------------------------------------


def test_genuine_chain_recovers_and_ablation_destroys():
    """A→B→C with B in bridge_set; C never follows A directly in the window."""
    filler, a, b, c, b_other = 0, 10, 20, 30, 21
    # Hop1: rightmost earlier A at pos 8, successor B at 9 → bridge=B.
    # Hop2: B at pos 2 (excl=9), successor C at 3 → pred=C.
    # A never precedes C: no A,C adjacent pair.
    ctx = _window(filler, a, {
        2: b, 3: c,
        8: a, 9: b,
    })
    ids = torch.tensor([ctx], dtype=torch.long)
    bridge_set = torch.tensor([b, b_other], dtype=torch.long)

    pred, b_out, p1, ok = khop_bridge_gate(ids, bridge_set)
    assert bool(ok[0]), "genuine chain should fire"
    assert int(b_out[0]) == b
    assert int(p1[0]) == 8
    assert int(pred[0]) == c

    # Ablation: wrong bridge (must be the other set member); no B_OTHER→C in window.
    ablated = khop_bridge_ablated(ids, bridge_set, b_out, p1, seed=0)
    assert int(ablated[0]) != c, (
        "ablation must not recover C on a genuine chain "
        f"(got {int(ablated[0])})"
    )


def test_bridge_site_exclusion_still_holds():
    """Bridge-site exclusion is load-bearing: excl=p1+1 must not re-match as hop2."""
    filler, a, b, c_true, c_wrong = 0, 10, 20, 30, 99
    # Without exclusion, hop2 would re-match the bridge site (pos 9) and take c_wrong.
    # With exclusion, hop2 takes earlier B→c_true.
    ctx = _window(filler, a, {
        2: b, 3: c_true,
        8: a, 9: b, 10: c_wrong,
    })
    ids = torch.tensor([ctx], dtype=torch.long)
    bridge_set = torch.tensor([b, 21], dtype=torch.long)
    pred, _b, p1, ok = khop_bridge_gate(ids, bridge_set)
    assert bool(ok[0])
    assert int(p1[0]) == 8
    assert int(pred[0]) == c_true


# ---------------------------------------------------------------------------
# (b) Mislabeled 1-hop: ablation recovery SURVIVES (memorization guard case)
# ---------------------------------------------------------------------------


def test_mislabeled_1hop_survives_ablation():
    """Wrong bridge still yields gold C — recovery would survive ablation (flag as memo)."""
    filler, a, b, c, b_other = 0, 10, 20, 30, 21
    # Real chain A→B→C AND ablated chain A→B_OTHER→C both present.
    ctx = _window(filler, a, {
        1: b_other, 2: c,   # wrong-bridge path also ends at C
        4: b, 5: c,         # true-bridge path ends at C
        8: a, 9: b,
    })
    ids = torch.tensor([ctx], dtype=torch.long)
    bridge_set = torch.tensor([b, b_other], dtype=torch.long)

    pred, b_out, p1, ok = khop_bridge_gate(ids, bridge_set)
    assert bool(ok[0])
    assert int(pred[0]) == c

    ablated = khop_bridge_ablated(ids, bridge_set, b_out, p1, seed=0)
    assert int(ablated[0]) == c, (
        "mislabeled case: ablated hop2 must still land on gold C "
        f"(got {int(ablated[0])})"
    )


# ---------------------------------------------------------------------------
# (c) Scoring arithmetic
# ---------------------------------------------------------------------------


def test_score_recovery_arithmetic():
    # 4 fired rows with known outcomes:
    # i0: 2hop right, baseline wrong  → one-sided
    # i1: 2hop right, baseline right  → both
    # i2: 2hop wrong, baseline right  → reverse
    # i3: 2hop wrong, baseline wrong  → neither
    # ablated: only i1 still right → agree_ablated = 0.25
    pred = torch.tensor([5, 5, 9, 9])
    ablated = torch.tensor([-1, 5, -1, -1])
    gold = torch.tensor([5, 5, 5, 5])
    baseline_agree = torch.tensor([False, True, True, False])

    sc = score_recovery(pred, ablated, baseline_agree, gold)
    assert sc["n"] == 4
    assert abs(sc["agree_2hop"] - 0.5) < 1e-9          # i0, i1
    assert abs(sc["agree_baseline"] - 0.5) < 1e-9      # i1, i2
    assert abs(sc["agree_2hop_ablated"] - 0.25) < 1e-9  # i1 only
    assert abs(sc["recovery"] - 0.0) < 1e-9
    assert abs(sc["recovery_ablated"] - (-0.25)) < 1e-9
    assert sc["one_sided_count"] == 1
    assert sc["reverse_count"] == 1
    assert abs(sc["one_sided_frac"] - 0.25) < 1e-9
    assert abs(sc["reverse_frac"] - 0.25) < 1e-9


def test_verdict_fires_only_when_recovery_and_guard():
    assert verdict_from_recovery(0.01, 0.0) == "FIRES"       # recovery ok, ablation vanished
    assert verdict_from_recovery(0.01, 0.01) == "DEAD"       # guard fails (survived)
    assert verdict_from_recovery(0.001, 0.0) == "DEAD"       # recovery below bar
    assert verdict_from_recovery(0.005, 0.004) == "FIRES"    # bar inclusive on recovery
    assert verdict_from_recovery(0.005, 0.005) == "DEAD"     # ablated must be STRICTLY below


# ---------------------------------------------------------------------------
# (d) Bridge-set mining degrees / support
# ---------------------------------------------------------------------------


def test_mine_bridge_set_degrees_and_support():
    """Hand-built FIT edges with known in/out degrees at support>=2.

    Edges (each listed with multiplicity for support):
      A1→B   x2   A2→B   x2   → in_deg(B)=2
      B→C1   x2   B→C2   x2   → out_deg(B)=2  ⇒ B qualifies
      X→Y    x2               → in_deg(Y)=1, out_deg(X)=1  (if only these)
      P→Q    x1               → support 1, dropped
      R→S    x2   R→T    x2   → out_deg(R)=2 but in_deg(R)=0 ⇒ R out
      U→R    x2   V→R    x2   → in_deg(R)=2, out_deg(R)=0 still? wait we gave R out above

    Clean design:
      B: in from {A1,A2} each supp>=2; out to {C1,C2} each supp>=2 → KEEP
      Z: in from {A1} only (one source) even if many times; out to two → DROP (in_deg=1)
      W: in from two sources; out to one target → DROP (out_deg=1)
      lone: only one edge at supp 1 → DROP
    """
    # Build key/val with multiplicities.
    pairs = []
    # B: in_deg=2, out_deg=2
    pairs += [(1, 100)] * 2 + [(2, 100)] * 2          # A1→B, A2→B
    pairs += [(100, 201)] * 2 + [(100, 202)] * 2      # B→C1, B→C2
    # Z: in_deg=1, out_deg=2 → reject
    pairs += [(3, 300)] * 3                             # only A3→Z
    pairs += [(300, 301)] * 2 + [(300, 302)] * 2        # Z→...
    # W: in_deg=2, out_deg=1 → reject
    pairs += [(4, 400)] * 2 + [(5, 400)] * 2            # →W
    pairs += [(400, 401)] * 3                           # only one out
    # support-1 edge (ignored entirely)
    pairs += [(9, 999)] * 1

    key = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    val = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    bset = mine_bridge_set(key, val, edge_minsupp=2, min_in_degree=2, min_out_degree=2)
    got = set(int(x) for x in bset.tolist())
    assert 100 in got, f"B=100 should qualify, got {got}"
    assert 300 not in got, "Z in_deg=1 must not qualify"
    assert 400 not in got, "W out_deg=1 must not qualify"
    assert 999 not in got, "support-1 edge must not create a bridge"
    assert got == {100}, f"only B should qualify, got {got}"


def test_mine_bridge_set_empty_and_minsupp():
    key = torch.tensor([1, 1, 2], dtype=torch.long)
    val = torch.tensor([3, 3, 3], dtype=torch.long)
    # (1,3) supp=2, (2,3) supp=1 → only edge 1→3 observed; in_deg(3)=1, out_deg(1)=1
    bset = mine_bridge_set(key, val, edge_minsupp=2, min_in_degree=2, min_out_degree=2)
    assert len(bset) == 0
    empty = mine_bridge_set(torch.tensor([]), torch.tensor([]))
    assert len(empty) == 0


def test_gate_requires_bridge_membership():
    """Hop1 can find a bridge, but gate must abstain if b not in bridge_set."""
    filler, a, b, c = 0, 10, 20, 30
    ctx = _window(filler, a, {2: b, 3: c, 8: a, 9: b})
    ids = torch.tensor([ctx], dtype=torch.long)
    # bridge_set has other tokens only — real B not a member
    bridge_set = torch.tensor([21, 22], dtype=torch.long)
    pred, b_out, _p1, ok = khop_bridge_gate(ids, bridge_set)
    assert int(b_out[0]) == b
    assert not bool(ok[0])
    assert int(pred[0]) == -1
