"""Sudoku oracle for rosetta's generic beam_engine.

Wraps the EXISTING Gate (b) pilot primitives (propagate_one_pass, cell_candidates,
_det_rank_fallback) — does not reimplement their logic. Child order and deep-copy
discipline match decide_energy bit-for-bit.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

import campaign_gate_b_pilot as gbp  # noqa: E402
import campaign_sudoku_forced_move as osc  # noqa: E402
from beam_engine import Path as BeamPath  # noqa: E402
from beam_engine import Step  # noqa: E402


class SudokuOracle:
    """Oracle over a 9x9 board targeting cell (r0, c0) with a corpus count table."""

    def __init__(
        self,
        board0: list[list[int]],
        r0: int,
        c0: int,
        count_table: list[list[list[int]]],
        wyly_seed: int,
    ) -> None:
        self.board0 = [row[:] for row in board0]  # own copy; NEVER mutated
        self.r0 = r0
        self.c0 = c0
        self.count_table = count_table
        self.wyly_seed = wyly_seed

    def initial_state(self) -> list[list[int]]:
        return [row[:] for row in self.board0]

    def expand(
        self, board: list[list[int]]
    ) -> list[tuple[list[list[int]], list[Step]]]:
        forced, contradiction = gbp.propagate_one_pass(board)
        if contradiction:
            return []  # PRUNE
        if forced:
            # PROPAGATE: apply ALL forced fills synchronously, margin=1.0 each,
            # in the exact order propagate_one_pass returned them.
            new_board = [row[:] for row in board]
            steps: list[Step] = []
            for r, c, v in forced:
                new_board[r][c] = v
                steps.append(
                    Step(key=(r, c, v), margin=1.0, counts=self._counts(r, c, v))
                )
            return [(new_board, steps)]
        # forced == [] and not contradiction -> TRUE LOCAL FIXPOINT
        if board[self.r0][self.c0] != 0:
            return [(board, [])]  # CARRY
        # BRANCH on global min-candidate empty cell (tie-break row-major)
        pivot = min(
            ((r, c) for r in range(9) for c in range(9) if board[r][c] == 0),
            key=lambda rc: (len(osc.cell_candidates(board, rc[0], rc[1])), rc),
        )
        pr, pc = pivot
        children: list[tuple[list[list[int]], list[Step]]] = []
        for v in sorted(osc.cell_candidates(board, pr, pc)):
            nb = [row[:] for row in board]
            nb[pr][pc] = v
            step = Step(key=(pr, pc, v), margin=0.0, counts=self._counts(pr, pc, v))
            children.append((nb, [step]))
        return children

    def extract_commit(self, path: BeamPath) -> int | None:
        v = path.state[self.r0][self.c0]
        return int(v) if v != 0 else None

    def fallback(self) -> int | None:
        # REUSE pilot fallback UNCHANGED — hardcodes rule_id "det_rank_fallback"
        # internally (independent of the beam engine's rule_id).
        return gbp._det_rank_fallback(
            self.board0, self.r0, self.c0, self.count_table, self.wyly_seed
        )

    def _counts(self, r: int, c: int, v: int) -> tuple[int, int]:
        t = self.count_table
        return (int(t[r][c][v]), sum(int(t[r][c][d]) for d in range(1, 10)))
