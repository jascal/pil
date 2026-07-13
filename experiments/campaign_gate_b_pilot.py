"""Gate (b) pilot -- beam-energy vs M=1 confidence (decisive measure-first).

Pre-registration: /home/allans/code/PIL_GATE_B_PILOT_PREREG.md (SIGNED; thresholds fixed;
CORRECTION v2/v3 — propagation-aligned whole-board beam + pinned turnstile margin).

Implements DECIDE_ENERGY as M steps of whole-board naked+hidden-single propagation
(branch on true local fixpoint, prune contradictions), ranked by the pinned PIC
turnstile margin (+1 forced / 0 guessed) with weakest-link lex trajectory, then
count-table det_rank + (rule_id, WYLY_SEED) hash tie-break. conf_acc is beam_acc(M=1)
by construction. Ground-truth propagation_depth and the beam share propagate_one_pass
(oracle-free). Unconstrained wikitext arm is contrast-only (no hard prune; reported
not gated).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# v5 binds recipe constants at import time -- setdefaults before the import (same as #100).
WYLY_ENV: dict[str, str] = {
    "WYLY_TAG": "pythia70m",
    "WYLY_DS": "sudoku",
    "WYLY_LIB": "mined",
    "WYLY_JUDGE": "cover",
    "WYLY_ONLINE": "1",
    "WYLY_COVER": "sw",
    "WYLY_CONCEPTS": "1",
    "WYLY_POINTER": "1",
    "WYLY_TPOINTER": "1",
    "WYLY_DX": "1",
    "WYLY_CX": "1",
    "WYLY_FOLDS": "3",
    "WYLY_LABELS": "corpus",
}
for _k, _v in WYLY_ENV.items():
    os.environ.setdefault(_k, _v)

import campaign_legality_certificate as cert  # noqa: E402
import campaign_sudoku_forced_move as osc  # noqa: E402
import campaign_uniform_reveal as unif  # noqa: E402
import wyly_lm_v5 as v5  # noqa: E402

OUTPUT = REPO / "data" / "gate_b_pilot.json"
CORPUS = REPO / "data" / "corpus_sudoku.txt"
WIKITEXT_PT = REPO / "data" / "wyly_nexttoken_wikitext_L256.pt"

# Registered constants -- do not tune.
BEAM_WIDTH = 8
WYLY_SEED = int(os.environ.get("WYLY_SEED", "0"))
RULE_ID = "decide_energy_v1"
M_SWEEP = [1, 2, 3, 4, 5]
SEPARATION_THRESHOLD = 0.15
BEAM_ACC_FLOOR = 0.50

COUNT_TABLE_LEAKAGE_NOTE = (
    "CountTable is built from the FULL corpus (not held out from the target population), so a "
    "target cell's own grid contributes a +1 self-count to its own cell/digit entry among "
    "(n_grids) grids -- a mild, disclosed inclusion, not a blocking issue (biases the M=1 "
    "baseline slightly UP, i.e. conservative against a false FIRES)."
)

UNCONSTRAINED_NOTE = (
    "pure margin energy, no legality hard-prune (candidates always non-empty via fallback); "
    "bigram count-table chained through the path's own committed choices (no ground-truth "
    "multi-step continuation exists in this dataset); reported as contrast only, per the "
    "pre-reg NOT gated."
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return str(obj)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CountTable (sudoku) -- det_rank determinism source
# ---------------------------------------------------------------------------

def build_sudoku_count_table(
    corpus_path: Path = CORPUS,
) -> tuple[list[list[list[int]]], int]:
    """table[r][c][d] (r,c in 0..8, d in 1..9) = count of corpus SOLUTION grids where cell
    (r,c) == digit d, via cert.parse_corpus_solutions(corpus_path). Returns (table, n_grids).
    table[r][c] is a length-10 list (index 0 unused, always 0).
    """
    solutions = cert.parse_corpus_solutions(corpus_path)
    table = [[[0] * 10 for _ in range(9)] for _ in range(9)]
    for grid in solutions:
        for r in range(9):
            for c in range(9):
                d = int(grid[r][c])
                if 1 <= d <= 9:
                    table[r][c][d] += 1
    return table, len(solutions)


def det_rank_compare(table: list[list[list[int]]], a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    """a, b are (r, c, v) tuples. Integer cross-multiply comparator (NO floats):
    rate(a) = table[a.r][a.c][a.v] / total(a.r,a.c); total(r,c) = sum(table[r][c][1..9]).
    Compare rate(a) vs rate(b) via ca*tb vs cb*ta (cross multiply). Higher rate = MORE
    deterministic = ranks BETTER (should sort first / 'smaller' in an ascending-argmin scheme).
    Returns -1 if a is MORE deterministic (ranks first), +1 if b is, 0 if exactly tied
    (including the 0/0 case -- treat as tied). Guard total==0 safely (never divide, only
    cross-multiply).
    """
    ar, ac, av = a
    br, bc, bv = b
    ca = int(table[ar][ac][av])
    cb = int(table[br][bc][bv])
    ta = sum(int(table[ar][ac][d]) for d in range(1, 10))
    tb = sum(int(table[br][bc][d]) for d in range(1, 10))
    # 0/0 and any total==0: cross-multiply yields equality when counts are 0
    left = ca * tb
    right = cb * ta
    if left > right:
        return -1
    if left < right:
        return 1
    return 0


def stable_hash_step(rule_id: str, r: int, c: int, v: int, wyly_seed: int) -> int:
    payload = f"{rule_id}:{r}:{c}:{v}:{wyly_seed}".encode()
    return int(hashlib.sha256(payload).hexdigest(), 16)


def stable_hash_ctx(rule_id: str, ctx: int, v: int, wyly_seed: int) -> int:
    payload = f"{rule_id}:{ctx}:{v}:{wyly_seed}".encode()
    return int(hashlib.sha256(payload).hexdigest(), 16)


# ---------------------------------------------------------------------------
# Shared oracle-free propagation primitive + ground-truth d(cell)
# ---------------------------------------------------------------------------

def propagate_one_pass(
    board: list[list[int]],
) -> tuple[list[tuple[int, int, int]] | None, bool]:
    """One synchronous, oracle-free naked+hidden-single pass over the WHOLE board.

    Returns (forced, contradiction):
      contradiction=True  -> some empty cell has 0 legal candidates; forced is None.
      contradiction=False -> forced = the list of (r, c, v) cells forceable RIGHT NOW,
        computed SIMULTANEOUSLY from the board as passed in (none of `forced`'s own fills
        are applied to each other's candidate computation within this call). May be empty:
        a TRUE LOCAL FIXPOINT — no cell is forceable by single-cell legality right now.

    A cell (r, c) with legal-candidate set C (via osc.cell_candidates) is forced to v if:
      (a) naked single:  len(C) == 1, or
      (b) hidden single: some v in C is uniquely placeable in its row OR col OR box.
        If more than one v qualifies, take the smallest such v deterministically.
    Row/col/box per-value candidate COUNTS are computed once (Counters); each cell's
    hidden-single check is O(1) against those counts.
    """
    empties = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
    cand: dict[tuple[int, int], set[int]] = {}
    for (r, c) in empties:
        cs = osc.cell_candidates(board, r, c)
        if not cs:
            return None, True
        cand[(r, c)] = cs

    row_count = [Counter() for _ in range(9)]
    col_count = [Counter() for _ in range(9)]
    box_count = [Counter() for _ in range(9)]
    for (r, c), cs in cand.items():
        b = 3 * (r // 3) + (c // 3)
        for v in cs:
            row_count[r][v] += 1
            col_count[c][v] += 1
            box_count[b][v] += 1

    forced: list[tuple[int, int, int]] = []
    for (r, c), cs in cand.items():
        if len(cs) == 1:
            forced.append((r, c, next(iter(cs))))
            continue
        b = 3 * (r // 3) + (c // 3)
        for v in sorted(cs):
            if row_count[r][v] == 1 or col_count[c][v] == 1 or box_count[b][v] == 1:
                forced.append((r, c, v))
                break
    return forced, False


def propagation_depth(board: list[list[int]], r: int, c: int) -> int | None:
    """Minimum single-cell-legality propagation passes until (r, c) is forced.

    d=1 iff (r, c) is itself forceable (naked OR hidden single) directly from the raw board
    (pass 1, applied to the untouched board) -- checked via propagate_one_pass's own forced
    list, NOT merely `len(candidates) == 1` (which misses hidden singles). Otherwise, iterate
    synchronous passes and return 1 + the pass index at which (r, c) is first FILLED (a robust
    board-state check, not a recomputed-candidate-count check, which can miss a hidden-single
    resolution in a later pass too).

    Pre-filled targets (puzzle givens already on the board) are d=1: they are not empty cells
    for propagate_one_pass to force, and decide_energy immediately commits board[r][c]. Without
    this guard the post-pass filled-check would mislabel them as d=2/never.
    """
    if board[r][c] != 0:
        return 1

    forced0, contradiction0 = propagate_one_pass(board)
    if contradiction0:
        return None
    if (r, c) in {(rr, cc) for rr, cc, _ in forced0}:
        return 1

    grid = [row[:] for row in board]
    for (rr, cc, v) in forced0:
        grid[rr][cc] = v
    pass_num = 1
    while True:
        pass_num += 1
        forced, contradiction = propagate_one_pass(grid)
        if contradiction or not forced:
            return None  # true fixpoint (or contradiction) without forcing the target
        for (rr, cc, v) in forced:
            grid[rr][cc] = v
        if grid[r][c] != 0:
            return pass_num


# ---------------------------------------------------------------------------
# DECIDE_ENERGY beam (constrained / sudoku arm)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    """One filled cell on a beam path. margin is the pinned turnstile: +1 forced, 0 guessed."""

    r: int
    c: int
    v: int
    margin: int  # {0, 1} only under the pinned EnergyStep definition


@dataclass
class BeamPath:
    steps: list[Step] = field(default_factory=list)
    board: list[list[int]] = field(default_factory=list)

    def extend(self, r: int, c: int, v: int, margin: int) -> BeamPath:
        new_board = [row[:] for row in self.board]
        new_board[r][c] = v
        return BeamPath(
            steps=self.steps + [Step(r=r, c=c, v=v, margin=margin)], board=new_board
        )


def compare_paths(
    a: BeamPath,
    b: BeamPath,
    count_table: list[list[list[int]]],
    wyly_seed: int,
    rule_id: str,
) -> int:
    """Ascending comparator: -1 if a ranks better (sorts first), +1 if b does, 0 if tied.

    Ladder structure (worst-margin step first): margin ladder (pinned {0,1}), then det_rank
    via integer cross-multiply det_rank_compare position-by-position, then stable hash.
    """
    sa, sb = a.steps, b.steps
    n = min(len(sa), len(sb))
    if n == 0:
        if len(sa) != len(sb):
            return -1 if len(sa) < len(sb) else 1
        return 0

    order_a = sorted(range(len(sa)), key=lambda i: sa[i].margin)
    order_b = sorted(range(len(sb)), key=lambda i: sb[i].margin)

    # 1. margin ladder (higher margin = more forced = ranks better; +1 beats 0)
    for ia, ib in zip(order_a, order_b, strict=False):
        ma, mb = sa[ia].margin, sb[ib].margin
        if ma != mb:
            return -1 if ma > mb else 1

    # 2. detrank ladder -- cross-multiply, position-by-position in the same order
    for ia, ib in zip(order_a, order_b, strict=False):
        cmp = det_rank_compare(
            count_table,
            (sa[ia].r, sa[ia].c, sa[ia].v),
            (sb[ib].r, sb[ib].c, sb[ib].v),
        )
        if cmp != 0:
            return cmp

    # 3. (rule_id, WYLY_SEED) hash tie-break
    for ia, ib in zip(order_a, order_b, strict=False):
        ha = stable_hash_step(rule_id, sa[ia].r, sa[ia].c, sa[ia].v, wyly_seed)
        hb = stable_hash_step(rule_id, sb[ib].r, sb[ib].c, sb[ib].v, wyly_seed)
        if ha != hb:
            return -1 if ha < hb else 1

    if len(sa) != len(sb):
        return -1 if len(sa) < len(sb) else 1
    return 0


def decide_energy(
    board: list[list[int]],
    r0: int,
    c0: int,
    M: int,
    beam_width: int,
    count_table: list[list[list[int]]],
    wyly_seed: int = WYLY_SEED,
    rule_id: str = RULE_ID,
) -> dict[str, Any]:
    """M-step propagation-aligned beam; commit the value path* assigns to cell0.

    Each outer step runs propagate_one_pass on every surviving path:
      - contradiction -> PRUNE (E=+inf)
      - forced non-empty -> apply ALL forced fills synchronously, margin=+1 each
        (even if cell0 is already filled — lets wrong guesses get pruned later)
      - true local fixpoint + cell0 filled -> carry path forward unchanged
      - true local fixpoint + cell0 still empty -> BRANCH on the global min-candidate
        empty cell (tie-break row-major); each legal value is a child with margin=0

    Branching triggers ONLY on a true local fixpoint, never merely because cell0 was
    untouched by a pass that made progress elsewhere (required for d-alignment).
    """
    paths: list[BeamPath] = [BeamPath(steps=[], board=[row[:] for row in board])]
    prune_events = 0
    total_steps = 0
    surviving_counts: list[int] = []

    for _step in range(M):
        total_steps += 1
        next_paths: list[BeamPath] = []
        any_pruned = False
        for p in paths:
            forced, contradiction = propagate_one_pass(p.board)
            if contradiction:
                any_pruned = True
                continue  # PRUNE this path entirely (E = +inf)
            if forced:
                # Progress (may or may not include cell0) — apply ALL forced fills
                # synchronously, margin=+1 each. Keep checking even if cell0 is already
                # filled so a wrong earlier guess can be pruned by a later pass.
                new_board = [row[:] for row in p.board]
                new_steps = list(p.steps)
                for (r, c, v) in forced:
                    new_board[r][c] = v
                    new_steps.append(Step(r=r, c=c, v=v, margin=1))
                next_paths.append(BeamPath(steps=new_steps, board=new_board))
                continue
            # forced == [] and not contradiction -> TRUE LOCAL FIXPOINT for this path.
            if p.board[r0][c0] != 0:
                # cell0 already committed; nothing more forceable; carry forward.
                next_paths.append(p)
                continue
            # cell0 still ambiguous and genuinely stuck — BRANCH on global min-cand pivot.
            pivot = min(
                ((r, c) for r in range(9) for c in range(9) if p.board[r][c] == 0),
                key=lambda rc: (len(osc.cell_candidates(p.board, rc[0], rc[1])), rc),
            )
            pr, pc = pivot
            for v in sorted(osc.cell_candidates(p.board, pr, pc)):
                nb = [row[:] for row in p.board]
                nb[pr][pc] = v
                next_paths.append(
                    BeamPath(
                        steps=p.steps + [Step(r=pr, c=pc, v=v, margin=0)],
                        board=nb,
                    )
                )
        if any_pruned:
            prune_events += 1
        if not next_paths:
            paths = []
            break
        next_paths.sort(
            key=cmp_to_key(
                lambda aa, bb: compare_paths(aa, bb, count_table, wyly_seed, rule_id)
            )
        )
        paths = next_paths[:beam_width]
        surviving_counts.append(len(paths))

    if not paths:
        return {
            "committed_value": None,
            "anomaly_total_contradiction": True,
            "n_steps": M,
            "prune_events": prune_events,
            "total_steps": total_steps,
            "surviving_counts": surviving_counts,
        }
    commit = paths[0].board[r0][c0]
    committed_value = int(commit) if commit != 0 else None
    return {
        "committed_value": committed_value,
        "anomaly_total_contradiction": False,
        "n_steps": M,
        "prune_events": prune_events,
        "total_steps": total_steps,
        "surviving_counts": surviving_counts,
    }


# ---------------------------------------------------------------------------
# Unconstrained arm (wikitext bigrams; NO hard prune)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UStep:
    ctx: int
    v: int
    margin: int


@dataclass
class UPath:
    steps: list[UStep] = field(default_factory=list)
    ctx: int = 0

    def extend(self, ctx: int, v: int, margin: int) -> UPath:
        return UPath(steps=self.steps + [UStep(ctx=ctx, v=v, margin=margin)], ctx=v)


def build_wikitext_bigrams(
    path: Path = WIKITEXT_PT,
) -> tuple[dict[int, dict[int, int]], list[int], list[int], int, list[int]]:
    """Return (bigram_counts, last_toks, targets, V_EFF, default_top9)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    ids = d["kept_ids"]
    target = d["target"]
    last_tok = ids[:, -1]
    bigram_counts: dict[int, dict[int, int]] = {}
    global_counts: Counter[int] = Counter()
    n = int(last_tok.shape[0])
    last_list: list[int] = []
    targ_list: list[int] = []
    for i in range(n):
        ctx = int(last_tok[i].item())
        tgt = int(target[i].item())
        last_list.append(ctx)
        targ_list.append(tgt)
        bucket = bigram_counts.setdefault(ctx, {})
        bucket[tgt] = bucket.get(tgt, 0) + 1
        global_counts[tgt] += 1
    v_eff = int(max(int(ids.max().item()), int(target.max().item()))) + 1
    default_top9 = [tok for tok, _ in global_counts.most_common(9)]
    if not default_top9:
        default_top9 = list(range(min(9, v_eff)))
    return bigram_counts, last_list, targ_list, v_eff, default_top9


def get_bigram_candidates(
    bigram_counts: dict[int, dict[int, int]],
    ctx: int,
    default_top9: list[int],
) -> dict[int, int]:
    """Candidate multiset for ctx. Never empty -- falls back to global top-9.

    NO legality hard-prune on the unconstrained arm: a step can never produce a
    zero-candidate contradiction (candidates always non-empty via this fallback).
    """
    cands = bigram_counts.get(ctx, {})
    if cands:
        return cands
    return {t: 1 for t in default_top9}


def bigram_det_rank_compare(
    bigram_counts: dict[int, dict[int, int]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> int:
    """a, b are (ctx, v). Cross-multiply rates; total 0 -> use 1 for safety (spec)."""
    ca = int(bigram_counts.get(a[0], {}).get(a[1], 0))
    cb = int(bigram_counts.get(b[0], {}).get(b[1], 0))
    ta = sum(bigram_counts.get(a[0], {}).values()) or 1
    tb = sum(bigram_counts.get(b[0], {}).values()) or 1
    left = ca * tb
    right = cb * ta
    if left > right:
        return -1
    if left < right:
        return 1
    return 0


def compare_upaths(
    a: UPath,
    b: UPath,
    bigram_counts: dict[int, dict[int, int]],
    wyly_seed: int,
    rule_id: str,
) -> int:
    sa, sb = a.steps, b.steps
    n = min(len(sa), len(sb))
    if n == 0:
        if len(sa) != len(sb):
            return -1 if len(sa) < len(sb) else 1
        return 0
    order_a = sorted(range(len(sa)), key=lambda i: sa[i].margin)
    order_b = sorted(range(len(sb)), key=lambda i: sb[i].margin)
    for ia, ib in zip(order_a, order_b, strict=False):
        ma, mb = sa[ia].margin, sb[ib].margin
        if ma != mb:
            return -1 if ma > mb else 1
    for ia, ib in zip(order_a, order_b, strict=False):
        cmp = bigram_det_rank_compare(
            bigram_counts,
            (sa[ia].ctx, sa[ia].v),
            (sb[ib].ctx, sb[ib].v),
        )
        if cmp != 0:
            return cmp
    for ia, ib in zip(order_a, order_b, strict=False):
        ha = stable_hash_ctx(rule_id, sa[ia].ctx, sa[ia].v, wyly_seed)
        hb = stable_hash_ctx(rule_id, sb[ib].ctx, sb[ib].v, wyly_seed)
        if ha != hb:
            return -1 if ha < hb else 1
    if len(sa) != len(sb):
        return -1 if len(sa) < len(sb) else 1
    return 0


def _local_top_vs(
    cands: dict[int, int],
    ctx: int,
    beam_width: int,
    wyly_seed: int,
    rule_id: str,
) -> list[int]:
    """Top-beam_width values at ctx by (-count, stable_hash) — exact sibling order.

    Avoids hashing every candidate: count-sort first, hash only the threshold-tie group.
    """
    if not cands:
        return []
    items = sorted(cands.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
    if len(items) <= beam_width:
        # Still need hash order among equal counts for exactness when sizes <= beam_width
        items.sort(
            key=lambda kv: (
                -int(kv[1]),
                stable_hash_ctx(rule_id, ctx, int(kv[0]), wyly_seed),
            )
        )
        return [int(v) for v, _ in items]
    thresh = int(items[beam_width - 1][1])
    head = [(int(v), int(c)) for v, c in items if int(c) > thresh]
    tied = [(int(v), int(c)) for v, c in items if int(c) == thresh]
    tied.sort(key=lambda kv: stable_hash_ctx(rule_id, ctx, kv[0], wyly_seed))
    need = beam_width - len(head)
    chosen = head + tied[:need]
    return [v for v, _ in chosen]


def decide_energy_unconstrained(
    ctx0: int,
    M: int,
    beam_width: int,
    bigram_counts: dict[int, dict[int, int]],
    v_eff: int,
    default_top9: list[int],
    wyly_seed: int = WYLY_SEED,
    rule_id: str = RULE_ID,
    _cand_cache: dict[int, dict[int, int]] | None = None,
    _top_cache: dict[tuple[int, int], list[int]] | None = None,
) -> dict[str, Any]:
    """Same ranking math as decide_energy; NO hard prune (cands never empty).

    Performance: per parent, only the top-`beam_width` local candidates are expanded.
    This is exact for global top-`beam_width` selection: any child ranked worse than
    beam_width-th among its siblings cannot enter the global top-beam_width (those
    better siblings share the same parent prefix and dominate it under compare_upaths).
    Local sibling ranking uses raw bigram counts (valid: same ctx => same denominator)
    then stable_hash, matching compare_upaths on equal-prefix paths.
    """
    paths: list[UPath] = [UPath(steps=[], ctx=ctx0)]
    prune_events = 0  # always 0 by construction (no legality prune)
    total_steps = 0
    surviving_counts: list[int] = []
    if _cand_cache is None:
        _cand_cache = {}
    if _top_cache is None:
        _top_cache = {}

    for _step in range(M):
        total_steps += 1
        children: list[UPath] = []
        for p in paths:
            if p.ctx not in _cand_cache:
                _cand_cache[p.ctx] = get_bigram_candidates(
                    bigram_counts, p.ctx, default_top9
                )
            cands = _cand_cache[p.ctx]
            # cands is never empty (fallback); no prune path
            margin = v_eff - len(cands)
            cache_key = (p.ctx, beam_width)
            if cache_key not in _top_cache:
                _top_cache[cache_key] = _local_top_vs(
                    cands, p.ctx, beam_width, wyly_seed, rule_id
                )
            for v in _top_cache[cache_key]:
                children.append(p.extend(p.ctx, v, margin))
        if not children:
            paths = []
            break
        children.sort(
            key=cmp_to_key(
                lambda aa, bb: compare_upaths(aa, bb, bigram_counts, wyly_seed, rule_id)
            )
        )
        paths = children[:beam_width]
        surviving_counts.append(len(paths))

    if not paths:
        return {
            "committed_value": None,
            "anomaly_total_contradiction": True,
            "n_steps": M,
            "prune_events": prune_events,
            "total_steps": total_steps,
            "surviving_counts": surviving_counts,
        }
    pstar = paths[0]
    return {
        "committed_value": int(pstar.steps[0].v) if pstar.steps else None,
        "anomaly_total_contradiction": False,
        "n_steps": M,
        "prune_events": prune_events,
        "total_steps": total_steps,
        "surviving_counts": surviving_counts,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _beam_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    total_prune = sum(int(r["prune_events"]) for r in results)
    total_steps = sum(int(r["total_steps"]) for r in results)
    surv_sum = 0
    surv_n = 0
    for r in results:
        for s in r.get("surviving_counts", []):
            surv_sum += int(s)
            surv_n += 1
    return {
        "mean_surviving_paths": osc.fraction(surv_sum, surv_n) if surv_n else 0.0,
        "prune_bite_rate": osc.fraction(total_prune, total_steps),
        "n_calls": len(results),
        "sum_prune_events": total_prune,
        "sum_total_steps": total_steps,
    }


def _acc_on(mask: list[bool], commits: list[int | None], golds: list[int]) -> tuple[float, int]:
    n = 0
    hits = 0
    for m, c, g in zip(mask, commits, golds, strict=True):
        if not m:
            continue
        n += 1
        if c is not None and int(c) == int(g):
            hits += 1
    return osc.fraction(hits, n), n


def _verdict_from(
    per_m: dict[int, dict[str, Any]],
    diagonal_acc: dict[int, float],
    diagonal_n: dict[int, int],
) -> dict[str, Any]:
    """Apply registered FIRES / FAILS / BOUNDARY predicates exactly."""
    sep_met_ms: list[int] = []
    floor_met_ms: list[int] = []
    for m in M_SWEEP:
        row = per_m[m]
        if row["n_stratum"] > 0 and row["separation"] >= SEPARATION_THRESHOLD:
            sep_met_ms.append(m)
        if row["n_stratum"] > 0 and row["beam_acc"] >= BEAM_ACC_FLOOR:
            floor_met_ms.append(m)

    both_met = [m for m in M_SWEEP if m in sep_met_ms and m in floor_met_ms]

    # Rising signature on d==M diagonal for M=2..5
    diag_points = [(m, diagonal_acc[m], diagonal_n[m]) for m in (2, 3, 4, 5)]
    nonempty = [(m, acc) for m, acc, n in diag_points if n > 0]
    if len(nonempty) < 2:
        signature_holds = False
        signature_note = (
            f"signature NOT holding: fewer than two non-empty d==M diagonal points "
            f"(nonempty M values: {[m for m, _ in nonempty]})"
        )
    else:
        mono = all(nonempty[i][1] <= nonempty[i + 1][1] for i in range(len(nonempty) - 1))
        signature_holds = mono
        signature_note = (
            "signature holds: beam_acc on d==M is monotonically non-decreasing across "
            f"non-empty diagonal points {[(m, round(a, 6)) for m, a in nonempty]}"
            if mono
            else (
                "signature does NOT hold: beam_acc on d==M is not monotonically "
                f"non-decreasing across {[(m, round(a, 6)) for m, a in nonempty]}"
            )
        )

    deciding_m = both_met[0] if both_met else None
    deciding = None
    if deciding_m is not None:
        row = per_m[deciding_m]
        deciding = {
            "M": deciding_m,
            "stratum": f"d in [2,{deciding_m}]",
            "beam_acc": row["beam_acc"],
            "conf_acc": row["conf_acc"],
            "separation": row["separation"],
            "n_stratum": row["n_stratum"],
        }

    fires = bool(both_met) and signature_holds
    fails = (not sep_met_ms) or (not floor_met_ms)

    if fires:
        verdict = "FIRES"
        explanation = (
            f"FIRES: at M={deciding_m} on stratum d in [2,{deciding_m}], "
            f"SEPARATION={per_m[deciding_m]['separation']:.6f} >= {SEPARATION_THRESHOLD} and "
            f"beam_acc={per_m[deciding_m]['beam_acc']:.6f} >= {BEAM_ACC_FLOOR}; "
            f"and {signature_note}."
        )
    elif fails:
        verdict = "FAILS"
        explanation = (
            f"FAILS: registered negative. "
            f"sep_met_Ms={sep_met_ms}, floor_met_Ms={floor_met_ms}. "
            f"SEPARATION never reached {SEPARATION_THRESHOLD} on any M and/or beam_acc never "
            f"reached {BEAM_ACC_FLOOR} on any d in [2,M] stratum. {signature_note}."
        )
    else:
        verdict = "BOUNDARY"
        explanation = (
            f"BOUNDARY: neither FIRES nor FAILS strictly. "
            f"sep+floor both met at Ms={both_met} but signature_holds={signature_holds}; "
            f"or partial conditions only (sep_met={sep_met_ms}, floor_met={floor_met_ms}). "
            f"{signature_note}."
        )

    return {
        "verdict": verdict,
        "deciding": deciding,
        "signature_holds": signature_holds,
        "signature_note": signature_note,
        "sep_met_ms": sep_met_ms,
        "floor_met_ms": floor_met_ms,
        "both_met_ms": both_met,
        "explanation": explanation,
        "diagonal_acc": {str(m): diagonal_acc[m] for m in (2, 3, 4, 5)},
        "diagonal_n": {str(m): diagonal_n[m] for m in (2, 3, 4, 5)},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    log("=" * 72)
    log("GATE (b) PILOT -- beam-energy vs M=1 confidence (DECIDE_ENERGY)")
    log("=" * 72)
    log(
        f"Registered: BEAM_WIDTH={BEAM_WIDTH} M_SWEEP={M_SWEEP} "
        f"DELTA={SEPARATION_THRESHOLD} FLOOR={BEAM_ACC_FLOOR} "
        f"RULE_ID={RULE_ID} WYLY_SEED={WYLY_SEED}"
    )
    log()
    log("WYLY_* env used for this run:")
    env_used = {k: os.environ.get(k, "") for k in WYLY_ENV}
    for k, v in env_used.items():
        log(f"  {k}={v}")
    log(
        f"  (module-bound) TAG={v5.TAG} DS={v5.DS} LIB={v5.LIB} JUDGE={v5.JUDGE} "
        f"LABELS={v5.LABELS} STATE={v5.STATE.name}"
    )
    log()

    # Provenance: cover state cache must exist (STOP if missing -- no regeneration).
    state_path = Path(v5.STATE)
    if not state_path.exists():
        log(f"ERROR: v5.STATE not found on disk: {state_path}")
        log("STOP -- will not fall back to osc.run_cover_regeneration().")
        return 1
    cover_state = torch.load(state_path, map_location="cpu", weights_only=False)
    log(f"loaded cover state cache: {state_path.name}")

    count_table, n_grids = build_sudoku_count_table(CORPUS)
    log(f"CountTable: n_grids={n_grids} from {CORPUS}")
    log(f"leakage note: {COUNT_TABLE_LEAKAGE_NOTE}")
    log()

    codec = v5.load_codec()
    puz_pat, sol_pat = osc.header_patterns(codec)
    value_map = osc.build_value_map(codec)
    ids, y, cls, uv, tr, te = v5.load_ds()
    n_kept = int(ids.shape[0])
    log(f"load_ds: N={n_kept} tr={len(tr)} te={len(te)} vocab_compact={len(uv)}")

    ids_cpu = ids.detach().cpu()
    y_cpu = y.detach().cpu()
    cls_cpu = cls.detach().cpu()
    uv_cpu = uv.detach().cpu() if torch.is_tensor(uv) else uv
    L = int(ids_cpu.shape[1])

    # ---- Build full classifiable board-valid population (tr+te) ----
    log("--- scanning all rows (uniform-reveal reconstruction) ---")
    records: list[dict[str, Any]] = []
    n_none = 0
    n_invalid = 0
    n_nongold = 0
    for i in range(n_kept):
        seq = osc.raw_window_from_compact(ids_cpu[i], uv_cpu)
        info = unif.reconstruct_block_for_target(seq, sol_pat, puz_pat, value_map, L=L)
        if info is None:
            n_none += 1
            continue
        if not info["board_valid"]:
            n_invalid += 1
            continue
        gold_tid = int(uv_cpu[int(cls_cpu[int(y_cpu[i])].item())].item())
        tv = value_map.get(gold_tid, ...)
        if tv is ... or tv is None:
            n_nongold += 1
            continue
        tv = int(tv)
        r, c = int(info["r"]), int(info["c"])
        board = info["board"]
        solution = info["solution"]
        d = propagation_depth(board, r, c)
        records.append({
            "row": int(i),
            "r": r,
            "c": c,
            "gold": tv,
            "board": board,
            "solution": solution,  # kept for provenance/debug; not used by d or beam
            "d": d,
        })
    n_all_valid = len(records)
    log(
        f"population_all_valid={n_all_valid}  (skipped none={n_none} invalid={n_invalid} "
        f"nongold={n_nongold})"
    )

    d_hist: Counter[str] = Counter()
    n_d1 = 0
    n_target = 0
    n_never = 0
    for rec in records:
        d = rec["d"]
        if d == 1:
            n_d1 += 1
            d_hist["d=1"] += 1
        elif d is None:
            n_never += 1
            n_target += 1
            d_hist["never"] += 1
        else:
            n_target += 1
            d_hist[f"d={d}"] += 1

    log(f"d=1 (sanity, excluded from target): {n_d1}")
    log(f"target population (non-forced-at-step-1): {n_target}")
    log(f"d-histogram (target + never): {dict(sorted(d_hist.items()))}")
    log()

    # ---- Beam M=1..5 on every valid cell (needed for gated + sanity strata) ----
    log("--- beam sweep on full valid population ---")
    # commits[m][i] for each record
    commits: dict[int, list[int | None]] = {m: [] for m in M_SWEEP}
    beam_results: dict[int, list[dict[str, Any]]] = {m: [] for m in M_SWEEP}
    n_anomaly = 0

    for idx, rec in enumerate(records):
        if idx > 0 and idx % 5000 == 0:
            log(f"  ... beam progress {idx}/{n_all_valid}")
        for m in M_SWEEP:
            res = decide_energy(
                rec["board"], rec["r"], rec["c"], m, BEAM_WIDTH, count_table, WYLY_SEED,
            )
            commits[m].append(res["committed_value"])
            beam_results[m].append(res)
            if res.get("anomaly_total_contradiction"):
                n_anomaly += 1

    if n_anomaly:
        log(f"WARNING: anomaly_total_contradiction count (across all M calls) = {n_anomaly}")
    golds = [int(rec["gold"]) for rec in records]
    ds = [rec["d"] for rec in records]

    # ---- Per-M gated + sanity tables ----
    constrained_per_m: dict[int, dict[str, Any]] = {}
    diagonal_acc: dict[int, float] = {}
    diagonal_n: dict[int, int] = {}

    for m in M_SWEEP:
        # gated stratum: d in [2, M]
        mask_stratum = [
            (d is not None and 2 <= int(d) <= m) for d in ds
        ]
        beam_acc, n_stratum = _acc_on(mask_stratum, commits[m], golds)
        conf_acc, _ = _acc_on(mask_stratum, commits[1], golds)
        separation = beam_acc - conf_acc

        # diagonal d==M
        mask_diag = [(d is not None and int(d) == m) for d in ds]
        b_diag, n_diag = _acc_on(mask_diag, commits[m], golds)
        diagonal_acc[m] = b_diag
        diagonal_n[m] = n_diag

        # sanity d=1
        mask_d1 = [d == 1 for d in ds]
        b_d1, n_d1_s = _acc_on(mask_d1, commits[m], golds)
        c_d1, _ = _acc_on(mask_d1, commits[1], golds)

        # sanity d>M or never (d=1 already excluded by d>m / None)
        mask_beyond = [
            (d is None) or (isinstance(d, int) and d > m) for d in ds
        ]
        b_bey, n_bey = _acc_on(mask_beyond, commits[m], golds)
        c_bey, _ = _acc_on(mask_beyond, commits[1], golds)

        stats = _beam_stats(beam_results[m])
        constrained_per_m[m] = {
            "beam_acc": beam_acc,
            "conf_acc": conf_acc,
            "separation": separation,
            "n_stratum": n_stratum,
            "beam_acc_on_d_eq_M": b_diag,
            "n_d_eq_M": n_diag,
            "beam_stats": stats,
            "sanity_d1": {
                "beam_acc": b_d1,
                "conf_acc": c_d1,
                "n": n_d1_s,
            },
            "sanity_d_gt_M_or_never": {
                "beam_acc": b_bey,
                "conf_acc": c_bey,
                "n": n_bey,
            },
        }
        log(
            f"  M={m}: n_stratum={n_stratum} beam_acc={beam_acc:.6f} "
            f"conf_acc={conf_acc:.6f} SEPARATION={separation:+.6f} "
            f"d==M n={n_diag} beam_acc_diag={b_diag:.6f} "
            f"prune_bite={stats['prune_bite_rate']:.6f}"
        )

    verdict_block = _verdict_from(constrained_per_m, diagonal_acc, diagonal_n)
    log()
    log(f"signature: {verdict_block['signature_note']}")
    log()

    # ---- Unconstrained arm ----
    log("--- unconstrained arm (wikitext bigrams; reported NOT gated) ---")
    unconstrained: dict[str, Any]
    if not WIKITEXT_PT.exists():
        log(f"WARNING: {WIKITEXT_PT} missing; skipping unconstrained arm")
        unconstrained = {
            "note": UNCONSTRAINED_NOTE,
            "skipped": True,
            "reason": f"{WIKITEXT_PT.name} not found",
        }
    else:
        bigrams, last_list, targ_list, v_eff, default_top9 = build_wikitext_bigrams(WIKITEXT_PT)
        residual_idx: list[int] = []
        for i, (ctx, tgt) in enumerate(zip(last_list, targ_list, strict=True)):
            cands = get_bigram_candidates(bigrams, ctx, default_top9)
            if len(cands) <= 1:
                continue
            # argmax by count; ties broken by smaller token id for determinism
            top = max(cands.keys(), key=lambda t: (cands[t], -t))
            if top != tgt:
                residual_idx.append(i)
        log(
            f"wikitext: N={len(last_list)} V_EFF={v_eff} residual_ambiguous={len(residual_idx)}"
        )

        u_commits: dict[int, list[int | None]] = {m: [] for m in M_SWEEP}
        u_results: dict[int, list[dict[str, Any]]] = {m: [] for m in M_SWEEP}
        u_golds = [targ_list[i] for i in residual_idx]
        # Shared caches across residual rows (same bigram table / seed / beam_width).
        _u_cand_cache: dict[int, dict[int, int]] = {}
        _u_top_cache: dict[tuple[int, int], list[int]] = {}
        for j, i in enumerate(residual_idx):
            if j > 0 and j % 10000 == 0:
                log(f"  ... unconstrained beam progress {j}/{len(residual_idx)}")
            ctx = last_list[i]
            for m in M_SWEEP:
                res = decide_energy_unconstrained(
                    ctx, m, BEAM_WIDTH, bigrams, v_eff, default_top9, WYLY_SEED,
                    _cand_cache=_u_cand_cache, _top_cache=_u_top_cache,
                )
                u_commits[m].append(res["committed_value"])
                u_results[m].append(res)

        u_per_m: dict[str, Any] = {}
        for m in M_SWEEP:
            mask_all = [True] * len(residual_idx)
            b_acc, n_s = _acc_on(mask_all, u_commits[m], u_golds)
            c_acc, _ = _acc_on(mask_all, u_commits[1], u_golds)
            stats = _beam_stats(u_results[m])
            u_per_m[str(m)] = {
                "beam_acc": b_acc,
                "conf_acc": c_acc,
                "separation": b_acc - c_acc,
                "n_stratum": n_s,
                "beam_stats": stats,
            }
            log(
                f"  unconst M={m}: n={n_s} beam_acc={b_acc:.6f} conf_acc={c_acc:.6f} "
                f"SEPARATION={b_acc - c_acc:+.6f}"
            )
        unconstrained = {
            "note": UNCONSTRAINED_NOTE,
            "skipped": False,
            "n_population": len(residual_idx),
            "v_eff": v_eff,
            "per_m": u_per_m,
        }

    wall = time.time() - t0

    # ---- Scoreboard ----
    log()
    log("=" * 72)
    log("SCOREBOARD -- constrained (gated)")
    log("=" * 72)
    log(
        f"{'M':>3}  {'n':>7}  {'beam_acc':>10}  {'conf_acc':>10}  {'SEPARATION':>12}  "
        f"{'d==M n':>7}  {'beam_d==M':>10}"
    )
    for m in M_SWEEP:
        row = constrained_per_m[m]
        log(
            f"{m:>3}  {row['n_stratum']:>7}  {row['beam_acc']:>10.6f}  "
            f"{row['conf_acc']:>10.6f}  {row['separation']:>+12.6f}  "
            f"{row['n_d_eq_M']:>7}  {row['beam_acc_on_d_eq_M']:>10.6f}"
        )
    log()
    log("Sanity strata (not gated):")
    for m in M_SWEEP:
        row = constrained_per_m[m]
        s1 = row["sanity_d1"]
        sb = row["sanity_d_gt_M_or_never"]
        log(
            f"  M={m} d=1: n={s1['n']} beam={s1['beam_acc']:.4f} conf={s1['conf_acc']:.4f} | "
            f"d>M/never: n={sb['n']} beam={sb['beam_acc']:.4f} conf={sb['conf_acc']:.4f}"
        )
    log()
    v = verdict_block["verdict"]
    deciding = verdict_block["deciding"]
    if deciding:
        decide_str = (
            f"M={deciding['M']} stratum={deciding['stratum']} "
            f"sep={deciding['separation']:.6f} beam_acc={deciding['beam_acc']:.6f} "
            f"n={deciding['n_stratum']}"
        )
    else:
        decide_str = "no M met both SEPARATION and floor"
    log(f"VERDICT: {v} -- {decide_str}")
    log(verdict_block["explanation"])
    log(f"wall_time_s={wall:.2f}")
    log()

    # ---- JSON report ----
    d_histogram_out: dict[str, int] = {
        "d=1_sanity": n_d1,
        "never": n_never,
        "population_all_valid": n_all_valid,
        "population_target": n_target,
    }
    for k, cnt in sorted(d_hist.items()):
        if k not in ("d=1", "never"):
            d_histogram_out[k] = cnt
        elif k == "never":
            pass  # already
        else:
            pass

    report: dict[str, Any] = {
        "provenance": {
            "dataset": "sudoku via v5.load_ds() (wyly_nexttoken_sudoku_L256.pt)",
            "corpus_path": str(CORPUS),
            "count_table_n_grids": n_grids,
            "count_table_leakage_note": COUNT_TABLE_LEAKAGE_NOTE,
            "count_table_prior_key": "P(digit | cell (r,c)) over full corpus solutions",
            "cover_state_path": str(state_path),
            "cover_state_cache": _json_safe(cover_state),
            "beam_width": BEAM_WIDTH,
            "wyly_seed": WYLY_SEED,
            "rule_id": RULE_ID,
            "m_sweep": M_SWEEP,
            "separation_threshold": SEPARATION_THRESHOLD,
            "beam_acc_floor": BEAM_ACC_FLOOR,
            "mechanism": (
                "CORRECTION v2+v3: propagation-aligned whole-board beam via shared "
                "oracle-free propagate_one_pass (naked+hidden singles, one synchronous "
                "pass per outer step). Branch ONLY on a true local fixpoint (nothing "
                "forceable anywhere), pivot = global min-candidate empty cell. Prune "
                "paths where propagate_one_pass returns contradiction. Energy = pinned "
                "PIC turnstile margin (+1 forced / 0 guessed), weakest-link lexicographic "
                "trajectory comparison, then det_rank (integer cross-multiply on the "
                "train/corpus count table) then (rule_id, WYLY_SEED) hash. Ground-truth "
                "propagation_depth uses the SAME propagate_one_pass primitive (no "
                "solution oracle). conf_acc = beam_acc(M=1) by construction."
            ),
            "conf_acc_on_gated_stratum_note": (
                "For many cells in the gated stratum d in [2,M], conf_acc=beam_acc(1) "
                "can be at or near 0: a finite d>=2 requires pass 1 to make SOME global "
                "progress, and with an M=1 budget that single pass is often spent on "
                "progress elsewhere, leaving cell0 uncommitted (None) when cell0 itself "
                "is not among the pass-1 forced set (multi-hop d=2 cascades). This is "
                "correct aligned behavior, not a bug — report real numbers regardless."
            ),
            "uniform_reveal_module": "experiments/campaign_uniform_reveal.py",
            "legality_module": "experiments/campaign_sudoku_forced_move.py",
            "certificate_module": "experiments/campaign_legality_certificate.py",
            "git_commit": git_commit_hash(),
            "wall_time_s": wall,
            "wyly_env": env_used,
            "n_anomaly_total_contradiction": n_anomaly,
            "population_truncated": False,
            "correction": "v3 pinned turnstile margin + v2 propagation-aligned beam",
        },
        "d_histogram": d_histogram_out,
        "constrained": {
            "per_m": {str(m): constrained_per_m[m] for m in M_SWEEP},
        },
        "unconstrained": unconstrained,
        "verdict": {
            "verdict": verdict_block["verdict"],
            "deciding": deciding,
            "signature_holds": verdict_block["signature_holds"],
            "signature_note": verdict_block["signature_note"],
            "sep_met_ms": verdict_block["sep_met_ms"],
            "floor_met_ms": verdict_block["floor_met_ms"],
            "both_met_ms": verdict_block["both_met_ms"],
            "explanation": verdict_block["explanation"],
            "diagonal_acc": verdict_block["diagonal_acc"],
            "diagonal_n": verdict_block["diagonal_n"],
        },
    }

    OUTPUT.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    log(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
