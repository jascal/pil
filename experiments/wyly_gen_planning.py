"""Generators for the CONSTRAINT/PLANNING wing (seed-0 deterministic, reproducible):

  chess   PGN movetext of legal games (python-chess; greedy-capture + randomness) -- the
          legality constraint is total: every token continuation is move-legal.
  sudoku  puzzle -> solution grids (backtracking generator) -- pure CSP text.
  sched   synthetic hospital rosters satisfying no-double-booking / rest constraints.
  robot   gridworld path traces: obstacle map + BFS shortest path as a move sequence.

OOD caveat (named): sched/robot formats are synthetic and outside the Pile; chess PGN and sudoku
grids occur on the web the Pile sampled. The study measures teacher-imitation crystallization on
constraint-structured text, not Pile-fidelity.

Run: cd pil && .venv/bin/python experiments/wyly_gen_planning.py
"""

import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def gen_chess(target=12_000_000):
    import chess
    rng = random.Random(0)
    out = []
    total = 0
    g = 0
    while total < target:
        board = chess.Board()
        moves = []
        while not board.is_game_over() and len(moves) < 160:
            legal = list(board.legal_moves)
            caps = [m for m in legal if board.is_capture(m)]
            pool = caps if caps and rng.random() < 0.6 else legal
            mv = rng.choice(pool)
            moves.append(board.san(mv))
            board.push(mv)
        txt = " ".join(f"{i // 2 + 1}. {m}" if i % 2 == 0 else m
                       for i, m in enumerate(moves)) + f" {board.result()}\n"
        out.append(txt)
        total += len(txt)
        g += 1
    (DATA / "corpus_chess.txt").write_text("".join(out))
    print(f"chess: {g} games, {total // 1000} KB")


def gen_sudoku(target=8_000_000):
    rng = random.Random(0)

    def solve(grid, cells):
        if not cells:
            return True
        r, c = cells[0]
        vals = list(range(1, 10))
        rng.shuffle(vals)
        for v in vals:
            if all(grid[r][j] != v for j in range(9)) \
                    and all(grid[i][c] != v for i in range(9)) \
                    and all(grid[3 * (r // 3) + i][3 * (c // 3) + j] != v
                            for i in range(3) for j in range(3)):
                grid[r][c] = v
                if solve(grid, cells[1:]):
                    return True
                grid[r][c] = 0
        return False

    out, total, n = [], 0, 0
    while total < target:
        grid = [[0] * 9 for _ in range(9)]
        solve(grid, [(r, c) for r in range(9) for c in range(9)])
        full = [row[:] for row in grid]
        holes = rng.sample(range(81), 45)
        for h in holes:
            grid[h // 9][h % 9] = 0
        p = "PUZZLE\n" + "\n".join(" ".join(str(x) if x else "." for x in row)
                                   for row in grid)
        s = "SOLUTION\n" + "\n".join(" ".join(str(x) for x in row) for row in full)
        txt = p + "\n" + s + "\n\n"
        out.append(txt)
        total += len(txt)
        n += 1
    (DATA / "corpus_sudoku.txt").write_text("".join(out))
    print(f"sudoku: {n} puzzles, {total // 1000} KB")


def gen_sched(target=8_000_000):
    rng = random.Random(0)
    docs = ["Adams", "Baker", "Chen", "Diaz", "Evans", "Fischer", "Garcia", "Hoffman",
            "Iyer", "Jones", "Kim", "Lopez", "Murphy", "Nguyen", "Okafor", "Patel"]
    wards = ["ICU", "ER", "Surgery", "Pediatrics", "Oncology", "Cardiology", "Radiology"]
    shifts = [("08:00", "16:00"), ("16:00", "00:00"), ("00:00", "08:00")]
    out, total, week = [], 0, 0
    while total < target:
        week += 1
        lines = [f"WEEK {week} ROSTER"]
        rest = {d: -2 for d in docs}
        for day in range(1, 8):
            lines.append(f"Day {day}:")
            busy = set()
            for ward in wards:
                for si, (a, b) in enumerate(shifts):
                    cands = [d for d in docs if d not in busy and rest[d] < day]
                    if not cands:
                        continue
                    d = rng.choice(cands)
                    busy.add(d)
                    if si == 2:
                        rest[d] = day + 1              # rest day after a night shift
                    lines.append(f"  {ward} {a}-{b}: Dr. {d}")
            lines.append(f"  Off duty: {', '.join(sorted(set(docs) - busy))}")
        txt = "\n".join(lines) + "\n\n"
        out.append(txt)
        total += len(txt)
    (DATA / "corpus_sched.txt").write_text("".join(out))
    print(f"sched: {week} weeks, {total // 1000} KB")


def gen_robot(target=8_000_000):
    from collections import deque
    rng = random.Random(0)
    out, total, n = [], 0, 0
    while total < target:
        w = h = 12
        walls = {(rng.randrange(h), rng.randrange(w)) for _ in range(26)}
        free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in walls]
        s, g = rng.sample(free, 2)
        prev = {s: None}
        dq = deque([s])
        while dq and g not in prev:
            cur = dq.popleft()
            for dr, dc, mv in ((-1, 0, "U"), (1, 0, "D"), (0, -1, "L"), (0, 1, "R")):
                nxt = (cur[0] + dr, cur[1] + dc)
                if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in walls \
                        and nxt not in prev:
                    prev[nxt] = (cur, mv)
                    dq.append(nxt)
        if g not in prev:
            continue
        path = []
        cur = g
        while prev[cur] is not None:
            cur, mv = prev[cur]
            path.append(mv)
        path.reverse()
        grid = [["#" if (r, c) in walls else "." for c in range(w)] for r in range(h)]
        grid[s[0]][s[1]], grid[g[0]][g[1]] = "S", "G"
        txt = (f"MAP {n}\n" + "\n".join("".join(row) for row in grid)
               + f"\nSTART {s[0]},{s[1]} GOAL {g[0]},{g[1]}\nPLAN "
               + " ".join(path) + f"\nSTEPS {len(path)}\n\n")
        out.append(txt)
        total += len(txt)
        n += 1
    (DATA / "corpus_robot.txt").write_text("".join(out))
    print(f"robot: {n} maps, {total // 1000} KB")


if __name__ == "__main__":
    gen_chess()
    gen_sudoku()
    gen_sched()
    gen_robot()
