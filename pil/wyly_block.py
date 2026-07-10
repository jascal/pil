"""WylyBlock: one stacked layer of feature extraction + rule admit + local cover (Phase A).

Each block owns:
  - feature extractors (estate2-style folds, or upstream-derived tensors)
  - candidate proposers → local sleep_admit_cover
  - admitted rules + conf callables
  - provenance (block_id, depends_on)

Blocks chain via ``BlockState`` (symbolic residual): downstream blocks may read upstream
feature tensors and predictions without opaque neural residuals.

Standalone-compatible: no teacher/embed/soft assumptions; callers supply corpus labels,
WordCodec ids, and query batches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

# (name, predict_fn) where predict_fn(ids) -> LongTensor [B] class-space answers (-1 = miss)
RuleFn = Callable[[torch.Tensor], torch.Tensor]
# conf_fn(ids) -> (val [B], conf [B])
ConfFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
# feat_fn(ids, upstream_state|None) -> LongTensor [B] (-1 = absent)
FeatFn = Callable[..., torch.Tensor]


@dataclass
class BlockState:
    """Symbolic residual passed block → block (and optionally into final cover)."""

    block_id: int
    name: str
    # feature name -> [B] long (-1 miss); may include predictions from this block
    features: dict[str, torch.Tensor] = field(default_factory=dict)
    # rule name -> [B] predicted class-space token (-1 miss)
    predictions: dict[str, torch.Tensor] = field(default_factory=dict)
    # final arbitrated prediction for this block (-1 abstain)
    pred: torch.Tensor | None = None
    conf: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSpec:
    """Named feature extractor registered on a block."""

    name: str
    fn: FeatFn
    # if True, fn receives (ids, upstream: BlockState|None); else fn(ids) only
    needs_upstream: bool = False
    # package emission payload (kind, params, ...) — opaque to the block core
    emit: Any = None


class WylyBlock:
    """One depth unit: features + candidates + local admitted rules + SW cover."""

    def __init__(
        self,
        block_id: int,
        name: str,
        *,
        depends_on: list[int] | None = None,
        device: str | torch.device = "cpu",
    ):
        self.block_id = int(block_id)
        self.name = name
        self.depends_on = list(depends_on or [])
        self.device = device
        self.features: list[FeatureSpec] = []
        self.candidates: list[tuple[str, RuleFn]] = []
        self.conf_fns: dict[str, ConfFn] = {}
        self.emit_info: dict[str, Any] = {}
        self.rule_size: dict[str, Callable[[], int]] = {}
        self.rules: list[tuple[str, RuleFn]] = []
        self.rules2: list[tuple[str, RuleFn]] = []
        self.log: list[str] = []
        # last admit stats
        self.last_admit: list[dict] = []

    # --- registration --------------------------------------------------------

    def add_feature(self, name: str, fn: FeatFn, *, needs_upstream: bool = False,
                    emit: Any = None) -> None:
        self.features.append(FeatureSpec(name, fn, needs_upstream, emit))

    def add_candidate(
        self,
        name: str,
        predict_fn: RuleFn,
        conf_fn: ConfFn | None = None,
        emit: Any = None,
        size_fn: Callable[[], int] | None = None,
    ) -> None:
        # prefix block id for global uniqueness in multi-block covers
        full = self._qualify(name)
        self.candidates.append((full, predict_fn))
        if conf_fn is not None:
            self.conf_fns[full] = conf_fn
        if emit is not None:
            self.emit_info[full] = emit
        if size_fn is not None:
            self.rule_size[full] = size_fn

    def _qualify(self, name: str) -> str:
        if name.startswith(f"b{self.block_id}/"):
            return name
        return f"b{self.block_id}/{name}"

    # --- forward -------------------------------------------------------------

    def extract(self, ids: torch.Tensor, upstream: BlockState | None = None) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for spec in self.features:
            if spec.needs_upstream:
                out[spec.name] = spec.fn(ids, upstream)
            else:
                out[spec.name] = spec.fn(ids)
        return out

    def predict_cover(
        self,
        ids: torch.Tensor,
        *,
        counts_row_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
        conf_fns: dict[str, ConfFn] | None = None,
        alpha: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Support-weighted cover over this block's admitted rules (+ optional counts).

        Returns (pred [B], conf [B]). pred = -1 where nothing fired.
        """
        conf_map = conf_fns if conf_fns is not None else self.conf_fns
        B = len(ids)
        pred = torch.full((B,), -1, dtype=torch.long, device=ids.device)
        conf = torch.full((B,), -1e9, device=ids.device, dtype=torch.float32)

        def consider(a: torch.Tensor, c: torch.Tensor) -> None:
            nonlocal pred, conf
            m = (a >= 0) & (c > conf)
            pred = torch.where(m, a, pred)
            conf = torch.where(m, c, conf)

        for name, fn in self.rules:
            if name in conf_map:
                a, c = conf_map[name](ids)
            else:
                a = fn(ids)
                c = torch.full((B,), 0.0, device=ids.device)
            consider(a, c)
        if counts_row_fn is not None:
            a, c = counts_row_fn(ids)
            consider(a, c)
        return pred, conf

    def forward(
        self,
        ids: torch.Tensor,
        upstream: BlockState | None = None,
        *,
        counts_row_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> BlockState:
        feats = self.extract(ids, upstream)
        pred, conf = self.predict_cover(ids, counts_row_fn=counts_row_fn)
        preds = {n: fn(ids) for n, fn in self.rules}
        # expose features + this block's pred under stable names for downstream
        feats = dict(feats)
        feats[f"b{self.block_id}_pred"] = pred
        return BlockState(
            block_id=self.block_id,
            name=self.name,
            features=feats,
            predictions=preds,
            pred=pred,
            conf=conf,
            meta={"n_rules": len(self.rules), "depends_on": list(self.depends_on)},
        )

    # --- admission (local) ---------------------------------------------------

    def admit_greedy(
        self,
        score_fn: Callable[[list[tuple[str, RuleFn]]], float],
        *,
        thresh: float = 5e-4,
        max_rules: int = 8,
    ) -> list[str]:
        """Greedy forward selection: add candidate with best marginal under score_fn.

        ``score_fn(rules)`` returns a scalar agreement (higher is better) — typically
        query_agree or cover agree on held-out deployment queries.
        """
        admitted: list[str] = []
        base = score_fn(self.rules)
        self.last_admit = []
        pool = list(self.candidates)
        for _ in range(max_rules):
            best = (thresh, None, None)  # marg, name, fn
            for name, fn in pool:
                if any(n == name for n, _ in self.rules):
                    continue
                trial = self.rules + [(name, fn)]
                sc = score_fn(trial)
                marg = sc - base
                self.last_admit.append({"name": name, "marginal": marg, "score": sc})
                if marg > best[0]:
                    best = (marg, name, fn)
            if best[1] is None:
                break
            self.rules.append((best[1], best[2]))
            admitted.append(best[1])
            self.log.append(f"b{self.block_id} ADMITTED {best[1]} (+{best[0]:.4f})")
            base = best[0] + base  # absolute score = base + marg; recompute cleanly:
            base = score_fn(self.rules)
            # remove from pool
            pool = [(n, f) for n, f in pool if n != best[1]]
        return admitted

    def summary(self) -> dict:
        return {
            "block_id": self.block_id,
            "name": self.name,
            "depends_on": list(self.depends_on),
            "n_features": len(self.features),
            "n_candidates": len(self.candidates),
            "admitted": [n for n, _ in self.rules],
            "n_rules2": len(self.rules2),
        }


class BlockStack:
    """Ordered chain of WylyBlocks with global final arbitration."""

    def __init__(self, blocks: list[WylyBlock] | None = None):
        self.blocks: list[WylyBlock] = list(blocks or [])

    def add(self, block: WylyBlock) -> None:
        self.blocks.append(block)

    def forward(
        self,
        ids: torch.Tensor,
        *,
        counts_row_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
        counts_only_on_block0: bool = True,
    ) -> list[BlockState]:
        states: list[BlockState] = []
        upstream: BlockState | None = None
        for i, blk in enumerate(self.blocks):
            cfn = counts_row_fn if (i == 0 or not counts_only_on_block0) else None
            st = blk.forward(ids, upstream, counts_row_fn=cfn)
            states.append(st)
            upstream = st
        return states

    def final_pred(self, states: list[BlockState]) -> tuple[torch.Tensor, torch.Tensor]:
        """Last non-abstaining block wins; else earliest counts-backed pred.

        Prefer deepest block with pred >= 0 and highest conf among firings.
        """
        if not states:
            raise ValueError("empty stack")
        B = len(states[0].pred)
        device = states[0].pred.device
        pred = torch.full((B,), -1, dtype=torch.long, device=device)
        conf = torch.full((B,), -1e9, device=device, dtype=torch.float32)
        for st in states:
            m = (st.pred >= 0) & (st.conf > conf)
            pred = torch.where(m, st.pred, pred)
            conf = torch.where(m, st.conf, conf)
        return pred, conf

    def all_rules(self) -> list[tuple[str, RuleFn]]:
        out: list[tuple[str, RuleFn]] = []
        for b in self.blocks:
            out.extend(b.rules)
        return out

    def all_conf_fns(self) -> dict[str, ConfFn]:
        out: dict[str, ConfFn] = {}
        for b in self.blocks:
            out.update(b.conf_fns)
        return out

    def summary(self) -> list[dict]:
        return [b.summary() for b in self.blocks]
