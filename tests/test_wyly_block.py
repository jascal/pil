"""Unit tests for multi-block Wyly foundation (BlockState, carry, admit_layer)."""
from __future__ import annotations

import torch

from pil.wyly_block import (
    BlockStack,
    WylyBlock,
    scan_stack_spec,
)


def _const_pred(val: int, conf: float = 1.0):
    def pred_fn(ids: torch.Tensor) -> torch.Tensor:
        return torch.full((len(ids),), val, dtype=torch.long, device=ids.device)

    def conf_fn(ids: torch.Tensor):
        a = pred_fn(ids)
        c = torch.full((len(ids),), conf, device=ids.device)
        return a, c

    return pred_fn, conf_fn


def test_scan_stack_spec_three_families():
    spec = scan_stack_spec()
    assert spec.n_blocks() == 3
    families = [b.family for b in spec.blocks]
    assert families == ["prims", "unary", "binary"]
    stack = BlockStack.from_spec(spec)
    assert len(stack.blocks) == 3
    assert stack.blocks[0].name == "prim_lexicon"


def test_admit_layer_freeze_upstream():
    """B1 admit scores frozen B0 rules + trial (stack mode)."""
    b0 = WylyBlock(0, "b0", family="prims", admit_mode="stack")
    b1 = WylyBlock(1, "b1", family="unary", depends_on=[0], admit_mode="stack")
    p0, c0 = _const_pred(1, 1.0)
    p1, c1 = _const_pred(2, 2.0)
    p2, c2 = _const_pred(3, 3.0)
    b0.add_candidate("r0", p0, c0)
    b1.add_candidate("r1a", p1, c1)
    b1.add_candidate("r1b", p2, c2)
    # force-admit r0
    b0.rules.append((b0._qualify("r0"), p0))
    stack = BlockStack([b0, b1], carry="merge")

    # score = number of rules (so greedy admits both B1 candidates)
    def score_fn(rules):
        return float(len(rules))

    adm = stack.admit_layer(1, score_fn, thresh=0.5, max_rules=2)
    assert len(adm) == 2
    assert len(b1.rules) == 2
    assert stack.admit_log[-1]["n_frozen"] == 1


def test_carry_merge_accumulates_residual():
    b0 = WylyBlock(0, "b0")
    b1 = WylyBlock(1, "b1", depends_on=[0])
    b0.set_residual(prims=["walk"])
    b1.set_residual(unary=["twice"])
    p, c = _const_pred(-1, -1e9)  # abstain
    b0.add_candidate("x", p, c)
    b1.add_candidate("y", p, c)
    stack = BlockStack([b0, b1], carry="merge")
    ids = torch.zeros(3, 2, dtype=torch.long)
    stack.forward(ids)
    carried = stack.last_carried[-1]
    assert carried.residual.get("prims") == ["walk"]
    assert carried.residual.get("unary") == ["twice"]


def test_carry_gated_prefers_upstream_when_weak():
    b0 = WylyBlock(0, "b0")
    b1 = WylyBlock(1, "b1", depends_on=[0])
    p0, c0 = _const_pred(7, 5.0)  # strong B0
    p1, c1 = _const_pred(-1, -1e9)  # B1 abstains
    b0.rules.append(("b0/r", p0))
    b0.conf_fns["b0/r"] = c0
    b1.rules.append(("b1/r", p1))
    b1.conf_fns["b1/r"] = c1
    stack = BlockStack([b0, b1], carry="gated", gate_conf=0.0)
    ids = torch.zeros(2, 1, dtype=torch.long)
    states = stack.forward(ids)
    # raw B1 still abstains
    assert (states[1].pred < 0).all()
    # carried state should recover B0 pred where B1 weak
    carried = stack.last_carried[-1]
    assert (carried.pred == 7).all()


def test_per_block_marginals_monotonic_when_rules_help():
    b0 = WylyBlock(0, "b0")
    b1 = WylyBlock(1, "b1")
    b0.rules.append(("b0/a", _const_pred(1)[0]))
    b1.rules.append(("b1/b", _const_pred(2)[0]))
    stack = BlockStack([b0, b1])

    def score_fn(rules):
        return float(len(rules))

    rows = stack.per_block_marginals(score_fn)
    assert rows[0]["score"] == 1.0
    assert rows[1]["score"] == 2.0
    assert rows[1]["marginal"] == 1.0
