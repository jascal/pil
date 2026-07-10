"""WylyBlock: one stacked layer of feature extraction + rule admit + local cover.

Each block owns:
  - feature extractors (estate2-style folds, or upstream-derived tensors)
  - candidate proposers → local greedy admit (or stack-marginal admit)
  - admitted rules + conf callables
  - provenance (block_id, depends_on, family, residual)

Blocks chain via ``BlockState`` (symbolic residual): downstream blocks may read upstream
feature tensors, predictions, and a free-form ``residual`` dict without opaque neural
residuals. ``BlockStack`` supports carry modes:

  - ``replace`` — only the latest block's state is upstream (default prior behavior)
  - ``merge``   — features/residual accumulate across depth
  - ``gated``   — upstream fields pass only where conf is low (residual for unsolved slots)

Standalone-compatible: no teacher/embed/soft assumptions; callers supply corpus labels,
WordCodec ids, and query batches.

Minimal 2-block usage (see ``experiments/campaign_wyly_blocks.py``)::

    from pil.wyly_block import WylyBlock, BlockStack

    b0 = WylyBlock(0, "lexicon", family="prims")
    b0.add_candidate("walk->WALK", pred_fn, conf_fn)
    b0.admit_greedy(score_fn)

    b1 = WylyBlock(1, "compose", depends_on=[0], family="unary")
    b1.add_feature("b0_pred", lambda ids, up: up.pred, needs_upstream=True)
    b1.add_candidate("twice", twice_fn, twice_conf)
    b1.admit_greedy(lambda rules: score_stack(b0.rules + rules))

    stack = BlockStack([b0, b1], carry="gated")
    states = stack.forward(ids, counts_row_fn=counts_fn)
    pred, conf = stack.final_pred(states)

3-block SCAN setup (primitives → unary combinators → binary composition)::

    from pil.wyly_block import WylyBlock, BlockStack, StackSpec, scan_stack_spec

    # Conceptual families (actual SCAN expand is symbolic; see campaign_scan_multiblock):
    #   B0 family=prims     — walk/jump/look/run, turn left/right  (+ leaf dir syntax)
    #   B1 family=unary     — twice, thrice, around, opposite
    #   B2 family=binary    — and, after
    #
    # Admission: freeze upstream; score candidates by *stack* marginal (joint when
    # operators synergize — staged unary-then-binary can stall; see SCAN notes).

    spec = scan_stack_spec()          # StackSpec with 3 families
    stack = BlockStack.from_spec(spec)
    # register candidates per block, then:
    # stack.admit_layer(1, score_fn, thresh=1e-4)  # freeze B0, admit B1
    # stack.admit_layer(2, score_fn, thresh=1e-4)  # freeze B0+B1, admit B2

Cross-links: ``docs/notes/wyly_multilayer.md``, ``docs/notes/scan_standalone.md``,
``experiments/campaign_scan_multiblock.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

# (name, predict_fn) where predict_fn(ids) -> LongTensor [B] class-space answers (-1 = miss)
RuleFn = Callable[[torch.Tensor], torch.Tensor]
# conf_fn(ids) -> (val [B], conf [B])
ConfFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
# feat_fn(ids, upstream_state|None) -> LongTensor [B] (-1 = absent)
FeatFn = Callable[..., torch.Tensor]

CarryMode = Literal["replace", "merge", "gated"]


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
    # free-form symbolic residual (enabled combinators, world state, plan, …)
    residual: dict[str, Any] = field(default_factory=dict)
    # provenance / admit diagnostics for this forward
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


@dataclass
class BlockSpec:
    """Configuration for one block (families, admit policy, dependencies)."""

    block_id: int
    name: str
    family: str = "default"
    depends_on: list[int] = field(default_factory=list)
    # max rules admitted in this block during stack.admit_layer
    max_rules: int = 16
    # local = score only this block's rules; stack = score freeze-upstream + trial
    admit_mode: Literal["local", "stack"] = "stack"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StackSpec:
    """Ordered multi-block configuration (number of blocks + rule families)."""

    blocks: list[BlockSpec]
    carry: CarryMode = "merge"
    gate_conf: float = 0.0  # for carry=gated: pass residual where conf <= gate_conf
    name: str = "stack"

    def n_blocks(self) -> int:
        return len(self.blocks)


def scan_stack_spec() -> StackSpec:
    """Canonical 3-block SCAN family layout (prims → unary → binary)."""
    return StackSpec(
        name="scan_3block",
        carry="merge",
        blocks=[
            BlockSpec(0, "prim_lexicon", family="prims", depends_on=[], max_rules=16),
            BlockSpec(1, "unary", family="unary", depends_on=[0], max_rules=8),
            BlockSpec(2, "binary", family="binary", depends_on=[0, 1], max_rules=4),
        ],
    )


class WylyBlock:
    """One depth unit: features + candidates + local admitted rules + SW cover."""

    def __init__(
        self,
        block_id: int,
        name: str,
        *,
        depends_on: list[int] | None = None,
        family: str = "default",
        device: str | torch.device = "cpu",
        max_rules: int = 16,
        admit_mode: Literal["local", "stack"] = "stack",
    ):
        self.block_id = int(block_id)
        self.name = name
        self.family = family
        self.depends_on = list(depends_on or [])
        self.device = device
        self.max_rules = int(max_rules)
        self.admit_mode = admit_mode
        self.features: list[FeatureSpec] = []
        self.candidates: list[tuple[str, RuleFn]] = []
        self.conf_fns: dict[str, ConfFn] = {}
        self.emit_info: dict[str, Any] = {}
        self.rule_size: dict[str, Callable[[], int]] = {}
        self.rules: list[tuple[str, RuleFn]] = []
        self.rules2: list[tuple[str, RuleFn]] = []
        self.log: list[str] = []
        # last admit stats (per-candidate marginals)
        self.last_admit: list[dict] = []
        # symbolic residual seed merged into every BlockState from this block
        self.residual_seed: dict[str, Any] = {}

    @classmethod
    def from_spec(cls, spec: BlockSpec, *, device: str | torch.device = "cpu") -> WylyBlock:
        return cls(
            spec.block_id,
            spec.name,
            depends_on=spec.depends_on,
            family=spec.family,
            device=device,
            max_rules=spec.max_rules,
            admit_mode=spec.admit_mode,
        )

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

    def set_residual(self, **kwargs: Any) -> None:
        """Update symbolic residual seed (merged into forward BlockState.residual)."""
        self.residual_seed.update(kwargs)

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
        _ = alpha  # reserved for Laplace SW; conf_fns already encode support
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
        residual = dict(self.residual_seed)
        if upstream is not None and upstream.residual:
            # default: local residual wins on key clash; stack merge may re-merge
            residual = {**upstream.residual, **residual}
        return BlockState(
            block_id=self.block_id,
            name=self.name,
            features=feats,
            predictions=preds,
            pred=pred,
            conf=conf,
            residual=residual,
            meta={
                "n_rules": len(self.rules),
                "depends_on": list(self.depends_on),
                "family": self.family,
                "admitted": [n for n, _ in self.rules],
            },
        )

    # --- admission (local) ---------------------------------------------------

    def admit_greedy(
        self,
        score_fn: Callable[[list[tuple[str, RuleFn]]], float],
        *,
        thresh: float = 5e-4,
        max_rules: int | None = None,
    ) -> list[str]:
        """Greedy forward selection: add candidate with best marginal under score_fn.

        ``score_fn(rules)`` returns a scalar agreement (higher is better) — typically
        query_agree or cover agree on held-out deployment queries.
        """
        cap = self.max_rules if max_rules is None else int(max_rules)
        admitted: list[str] = []
        base = score_fn(self.rules)
        self.last_admit = []
        pool = list(self.candidates)
        for _ in range(cap):
            best = (thresh, None, None)  # marg, name, fn
            for name, fn in pool:
                if any(n == name for n, _ in self.rules):
                    continue
                trial = self.rules + [(name, fn)]
                sc = score_fn(trial)
                marg = sc - base
                self.last_admit.append({
                    "name": name, "marginal": marg, "score": sc,
                    "block_id": self.block_id, "family": self.family,
                })
                if marg > best[0]:
                    best = (marg, name, fn)
            if best[1] is None:
                break
            self.rules.append((best[1], best[2]))
            admitted.append(best[1])
            self.log.append(f"b{self.block_id} ADMITTED {best[1]} (+{best[0]:.4f})")
            base = score_fn(self.rules)
            pool = [(n, f) for n, f in pool if n != best[1]]
        return admitted

    def summary(self) -> dict:
        return {
            "block_id": self.block_id,
            "name": self.name,
            "family": self.family,
            "depends_on": list(self.depends_on),
            "admit_mode": self.admit_mode,
            "n_features": len(self.features),
            "n_candidates": len(self.candidates),
            "admitted": [n for n, _ in self.rules],
            "n_rules2": len(self.rules2),
            "last_admit_top": sorted(
                self.last_admit, key=lambda r: -r.get("marginal", 0.0),
            )[:8],
        }


class BlockStack:
    """Ordered chain of WylyBlocks with global final arbitration and carry modes."""

    def __init__(
        self,
        blocks: list[WylyBlock] | None = None,
        *,
        carry: CarryMode = "replace",
        gate_conf: float = 0.0,
        name: str = "stack",
    ):
        self.blocks: list[WylyBlock] = list(blocks or [])
        self.carry: CarryMode = carry
        self.gate_conf = float(gate_conf)
        self.name = name
        self.admit_log: list[dict] = []

    @classmethod
    def from_spec(cls, spec: StackSpec, *, device: str | torch.device = "cpu") -> BlockStack:
        blocks = [WylyBlock.from_spec(bs, device=device) for bs in spec.blocks]
        return cls(blocks, carry=spec.carry, gate_conf=spec.gate_conf, name=spec.name)

    def add(self, block: WylyBlock) -> None:
        self.blocks.append(block)

    def _merge_upstream(
        self,
        prev: BlockState | None,
        cur: BlockState,
    ) -> BlockState:
        """Produce the state that the *next* block will see as upstream."""
        if prev is None or self.carry == "replace":
            return cur
        if self.carry == "merge":
            feats = {**prev.features, **cur.features}
            residual = {**prev.residual, **cur.residual}
            return BlockState(
                block_id=cur.block_id,
                name=cur.name,
                features=feats,
                predictions={**prev.predictions, **cur.predictions},
                pred=cur.pred if cur.pred is not None else prev.pred,
                conf=cur.conf if cur.conf is not None else prev.conf,
                residual=residual,
                meta={**prev.meta, **cur.meta, "carry": "merge"},
            )
        # gated: pass previous features where current conf is weak / abstain
        assert cur.pred is not None and cur.conf is not None
        assert prev.pred is not None and prev.conf is not None
        weak = (cur.pred < 0) | (cur.conf <= self.gate_conf)
        # tensor features: where weak, keep prev; else cur
        feats = dict(cur.features)
        for k, v in prev.features.items():
            if k in feats and feats[k].shape == v.shape:
                feats[k] = torch.where(weak, v, feats[k])
            elif k not in feats:
                feats[k] = v
        # pred: prefer cur when strong; else prev
        pred = torch.where(weak, prev.pred, cur.pred)
        conf = torch.where(weak, prev.conf, cur.conf)
        residual = {**prev.residual, **cur.residual}
        residual["gated_frac"] = float(weak.float().mean()) if weak.numel() else 0.0
        return BlockState(
            block_id=cur.block_id,
            name=cur.name,
            features=feats,
            predictions={**prev.predictions, **cur.predictions},
            pred=pred,
            conf=conf,
            residual=residual,
            meta={**cur.meta, "carry": "gated", "gated_frac": residual["gated_frac"]},
        )

    def forward(
        self,
        ids: torch.Tensor,
        *,
        counts_row_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
        counts_only_on_block0: bool = True,
    ) -> list[BlockState]:
        """Run each block; return per-block states (pre-carry) and set ``stack.last_states``.

        ``self.last_carried`` holds the carried state after each block (what next sees).
        """
        states: list[BlockState] = []
        carried: list[BlockState] = []
        upstream: BlockState | None = None
        for i, blk in enumerate(self.blocks):
            cfn = counts_row_fn if (i == 0 or not counts_only_on_block0) else None
            st = blk.forward(ids, upstream, counts_row_fn=cfn)
            states.append(st)
            up_next = self._merge_upstream(upstream, st)
            carried.append(up_next)
            upstream = up_next
        self.last_states = states
        self.last_carried = carried
        return states

    def final_pred(self, states: list[BlockState] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Deepest confident non-abstaining block wins (max conf among firings)."""
        if states is None:
            states = getattr(self, "last_states", None) or []
        if not states:
            raise ValueError("empty stack")
        B = len(states[0].pred)  # type: ignore[arg-type]
        device = states[0].pred.device  # type: ignore[union-attr]
        pred = torch.full((B,), -1, dtype=torch.long, device=device)
        conf = torch.full((B,), -1e9, device=device, dtype=torch.float32)
        for st in states:
            assert st.pred is not None and st.conf is not None
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

    def rules_through(self, block_id: int) -> list[tuple[str, RuleFn]]:
        """Admitted rules from blocks with id <= block_id (freeze-upstream prefix)."""
        out: list[tuple[str, RuleFn]] = []
        for b in self.blocks:
            if b.block_id <= block_id:
                out.extend(b.rules)
        return out

    def admit_layer(
        self,
        block_id: int,
        score_fn: Callable[[list[tuple[str, RuleFn]]], float],
        *,
        thresh: float = 5e-4,
        max_rules: int | None = None,
    ) -> list[str]:
        """Admit into one block. ``stack`` mode scores freeze-upstream ∪ trial rules.

        ``score_fn`` always receives the full rule list to evaluate (caller defines
        how rules map to a scalar). For ``admit_mode=stack`` we pass
        ``rules_through(block_id-1) + trial_block_rules``; for ``local`` only the
        block's trial rules.
        """
        blk = next(b for b in self.blocks if b.block_id == block_id)
        frozen = self.rules_through(block_id - 1) if block_id > 0 else []

        def wrapped(trial_rules: list[tuple[str, RuleFn]]) -> float:
            if blk.admit_mode == "local":
                return score_fn(trial_rules)
            return score_fn(frozen + trial_rules)

        admitted = blk.admit_greedy(wrapped, thresh=thresh, max_rules=max_rules)
        self.admit_log.append({
            "block_id": block_id,
            "family": blk.family,
            "admitted": list(admitted),
            "n_frozen": len(frozen),
            "last_admit": list(blk.last_admit[-20:]),
        })
        return admitted

    def per_block_marginals(
        self,
        score_fn: Callable[[list[tuple[str, RuleFn]]], float],
    ) -> list[dict]:
        """Cumulative score after each block (prefix rules) for diagnostics."""
        rows = []
        acc_rules: list[tuple[str, RuleFn]] = []
        prev = score_fn([])
        for b in self.blocks:
            acc_rules = acc_rules + list(b.rules)
            sc = score_fn(acc_rules)
            rows.append({
                "block_id": b.block_id,
                "name": b.name,
                "family": b.family,
                "n_rules": len(b.rules),
                "score": sc,
                "marginal": sc - prev,
                "admitted": [n for n, _ in b.rules],
            })
            prev = sc
        return rows

    def summary(self) -> dict:
        return {
            "name": self.name,
            "carry": self.carry,
            "gate_conf": self.gate_conf,
            "blocks": [b.summary() for b in self.blocks],
            "admit_log": list(self.admit_log),
        }
