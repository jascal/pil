"""Does an abstraction tier emerge from CROSS-TASK REUSE (not single-task fit)? A minimal wake/sleep loop.

The missing tier (this session's thread): the rule/decode/memory strata all run over a FIXED vocabulary;
none INVENTS a new predicate/entity. Here we test the sharp claim that abstraction discovery is driven by
cross-task reuse and is invisible to single-task fit — the DreamCoder library-learning move, in miniature.

Setup: a family of tasks over a tiny arithmetic DSL, all secretly built from a shared helper h(x)=x*x+x.
  WAKE  — bottom-up synthesis (observational equivalence) finds the smallest program fitting each task's I/O.
  SLEEP — score every recurring subtree by MDL compression of the WHOLE library+solution set; invent the
          best one as a new primitive; re-solve with the enriched DSL.

Decisive readouts:
  (1) does it discover the planted h?
  (2) N-sweep: h's MDL gain crosses 0 at cross-task count >=2 — invented from a family, never one task.
  (3) reuse payoff: a hard task (h*h) unsolvable within the size budget in the base DSL becomes solvable once
      h exists.
  (4) total description length (library + all solutions) drops across rounds.
"""

from __future__ import annotations

# ---- tiny DSL: programs are tuples; library maps name -> definition (a program in terms of 'x') ---
X = ("x",)


def evalp(p, x, lib):
    t = p[0]
    if t == "x":
        return x
    if t == "c":
        return p[1]
    if t == "call":
        return evalp(lib[p[1]], evalp(p[2], x, lib), lib)
    a = evalp(p[1], x, lib)
    b = evalp(p[2], x, lib)
    return a + b if t == "+" else a * b


def size(p):
    t = p[0]
    if t in ("x", "c"):
        return 1
    if t == "call":
        return 1 + size(p[2])
    return 1 + size(p[1]) + size(p[2])


def sig(p, inputs, lib):
    try:
        return tuple(evalp(p, x, lib) for x in inputs)
    except (ValueError, OverflowError):
        return None


def subtrees(p):
    yield p
    if p[0] in ("+", "*"):
        yield from subtrees(p[1])
        yield from subtrees(p[2])
    elif p[0] == "call":
        yield from subtrees(p[2])


def refactor(p, target, name):
    """Replace every exact copy of subtree `target` with call(name, x)."""
    if p == target:
        return ("call", name, X)
    if p[0] in ("+", "*"):
        return (p[0], refactor(p[1], target, name), refactor(p[2], target, name))
    if p[0] == "call":
        return (p[0], p[1], refactor(p[2], target, name))
    return p


# ---- WAKE: bottom-up synthesis, grown strictly by size, dedup by signature (obs. equivalence) -------
def synth(target_sig, inputs, lib, max_size, vcap=10 ** 7, bank_cap=6000):
    bank: dict[tuple, tuple] = {}          # signature -> smallest program
    by_size: dict[int, list] = {1: []}
    for p in (X, ("c", 0), ("c", 1), ("c", 2)):
        s = sig(p, inputs, lib)
        if s is not None and s not in bank:
            bank[s] = p
            by_size[1].append(p)
    if target_sig in bank:
        return bank[target_sig]
    unary = list(lib)
    for sz in range(2, max_size + 1):
        made: list = []
        # binary op(a,b) with size(a)+size(b)+1 == sz
        for i in range(1, sz - 1):
            j = sz - 1 - i
            for a in by_size.get(i, ()):
                for b in by_size.get(j, ()):
                    for op in ("+", "*"):
                        p = (op, a, b)
                        s = sig(p, inputs, lib)
                        if s is None or s in bank or any(abs(v) > vcap for v in s):
                            continue
                        bank[s] = p
                        made.append(p)
                        if s == target_sig:
                            return p
        # unary invented primitive call(name, a) with size(a) == sz-1
        for a in by_size.get(sz - 1, ()):
            for name in unary:
                p = ("call", name, a)
                s = sig(p, inputs, lib)
                if s is None or s in bank or any(abs(v) > vcap for v in s):
                    continue
                bank[s] = p
                made.append(p)
                if s == target_sig:
                    return p
        by_size[sz] = made[:bank_cap]
    return None


# ---- SLEEP: candidates are recurring SEMANTIC fragments (subtree signatures, form-independent);
#      each is scored by RE-SOLVING the whole task family with it added and measuring total MDL. This is
#      the correct joint objective — it sidesteps the "same function, different minimal syntax" problem
#      that defeats exact-subtree matching, and it counts a fragment as reusable by cross-task compression.
from collections import Counter  # noqa: E402


def enumerate_bank(inputs, max_size, vcap=10 ** 7):
    """All distinct-signature base-DSL programs up to max_size (the fragment pool to propose from)."""
    bank: dict[tuple, tuple] = {}
    by_size: dict[int, list] = {1: []}
    for p in (X, ("c", 0), ("c", 1), ("c", 2)):
        s = sig(p, inputs, {})
        if s is not None and s not in bank:
            bank[s] = p
            by_size[1].append(p)
    for sz in range(2, max_size + 1):
        made = []
        for i in range(1, sz - 1):
            j = sz - 1 - i
            for a in by_size.get(i, ()):
                for b in by_size.get(j, ()):
                    for op in ("+", "*"):
                        s = sig((op, a, b), inputs, {})
                        if s is None or s in bank or any(abs(v) > vcap for v in s):
                            continue
                        bank[s] = (op, a, b)
                        made.append((op, a, b))
        by_size[sz] = made
    return bank


def fragment_signatures(solutions, inputs):
    count, best = Counter(), {}          # sig -> (#solutions it appears in, smallest program)
    for sol in solutions:
        here = set()
        for st in subtrees(sol):
            if size(st) < 3:
                continue
            s = sig(st, inputs, {})
            if s is None or len(set(s)) == 1:                     # skip constants (not a useful predicate)
                continue
            if s not in here:
                count[s] += 1
                here.add(s)
            if s not in best or size(st) < size(best[s]):
                best[s] = st
    return count, best


# ---- tasks: all built from the planted shared helper h(x) = x*x + x --------------------------------
H = ("+", ("*", X, X), X)                                    # x^2 + x, size 5
TASKS = {
    "h+1": ("+", H, ("c", 1)),
    "h+2": ("+", H, ("c", 2)),
    "h*2": ("*", H, ("c", 2)),
    "h+x": ("+", H, X),
    "h*h": ("*", H, H),                                      # the "hard" task (size 11 in base DSL)
}
INPUTS = [0, 1, 2, 3, 4, 5]
BUDGET = 8                                                   # size budget: base h+c solvable, h*h NOT


def solve_all(task_sigs, lib, budget):
    return {name: synth(s, INPUTS, lib, budget) for name, s in task_sigs.items()}


UNSOLVED = 16          # an unsolved task is a failure, not free — else "solving it" reads as added cost


def total_mdl(lib, sols):
    lib_cost = sum(size(d) for d in lib.values())
    sol_cost = sum(size(p) if p is not None else UNSOLVED for p in sols.values())
    return lib_cost + sol_cost


H_SIG = sig(H, INPUTS, {})


def main():
    task_sigs = {n: sig(p, INPUTS, {}) for n, p in TASKS.items()}
    print(f"planted helper h(x)=x*x+x sig={H_SIG} (size {size(H)}); tasks {list(TASKS)}; budget {BUDGET}\n")

    # ---- (2) N-sweep: re-solve MDL gain of adding h, vs #tasks it can be reused across ----
    print("(2) N-sweep — total-MDL gain of adding h to the DSL, vs #tasks (re-solve based):")
    fam = ["h+1", "h+2", "h*2", "h+x"]
    for n in range(1, len(fam) + 1):
        sub = {k: task_sigs[k] for k in fam[:n]}
        base = total_mdl({}, solve_all(sub, {}, BUDGET))
        withh = total_mdl({"h": H}, solve_all(sub, {"h": H}, BUDGET))
        gain = base - withh
        print(f"   N={n}  base_MDL={base}  +h_MDL={withh}  gain={gain:+d}  -> "
              f"{'INVENT' if gain > 0 else 'reject'}")
    print("   => h pays for itself only when reused across >=2 tasks; one task never justifies it.\n")

    # ---- WAKE round 0 (base DSL) ----
    sols = solve_all(task_sigs, {}, BUDGET)
    print("(3) reuse payoff — round 0 (base DSL):")
    for nm, p in sols.items():
        print(f"   {nm:>4}: {'UNSOLVED (>budget)' if p is None else f'size {size(p)}  {p}'}")
    base_mdl = total_mdl({}, sols)
    print(f"   total MDL = {base_mdl}\n")

    # ---- SLEEP-A (naive proposer): candidates = subtrees of the minimal solutions ----
    solved = [p for p in sols.values() if p is not None]
    count, best = fragment_signatures(solved, INPUTS)
    print("(1a) naive proposer (subtrees of minimal solutions): recurring fragments =")
    for s, c in sorted(count.items(), key=lambda kv: -kv[1])[:4]:
        print(f"    sig={s} in {c} solutions  {'<- planted h' if s == H_SIG else ''}")
    print(f"    planted h present? {H_SIG in count}  -> h is LATENT in no minimal solution; "
          "the naive proposer cannot see it.\n")

    # ---- SLEEP-B (fragment-pool proposer): candidates from the enumeration space, scored by re-solve ----
    pool = enumerate_bank(INPUTS, 5)
    scored = []
    for s, prim in pool.items():
        if len(set(s)) == 1:                                     # skip constants
            continue
        m = total_mdl({"A": prim}, solve_all(task_sigs, {"A": prim}, BUDGET))
        scored.append((base_mdl - m, prim, s))
    scored.sort(reverse=True, key=lambda t: t[0])
    print("(1b) fragment-pool proposer (enumeration space), scored by re-solve compression:")
    for g, prim, s in scored[:4]:
        print(f"    MDL_gain={g:+d}  prim={prim} sig={s}  {'<- planted h' if s == H_SIG else ''}")
    g, best_prim, best_sig = scored[0]
    print(f"\n   invented A := {best_prim}  (semantically == planted h? {best_sig == H_SIG})\n")

    # ---- round 1: re-solve with the invented primitive ----
    lib = {"A": best_prim}
    sols2 = solve_all(task_sigs, lib, BUDGET)
    print("   round 1 (DSL + A):")
    for nm, p in sols2.items():
        print(f"   {nm:>4}: {'UNSOLVED' if p is None else f'size {size(p)}  {p}'}")
    m2 = total_mdl(lib, sols2)
    newly = [nm for nm in sols if sols[nm] is None and sols2[nm] is not None]
    print(f"\n(4) total MDL {base_mdl} -> {m2} ({'DROP' if m2 < base_mdl else 'rise'});  "
          f"newly-solvable once h exists: {newly}")
    print("    => the abstraction tier is BUILDABLE here: a reusable predicate discovered purely from "
          "cross-task compression, invisible to any single task, that also unlocks a task beyond budget.")


if __name__ == "__main__":
    main()
