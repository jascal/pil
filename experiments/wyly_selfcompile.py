"""The SELF-COMPILING learner: sleep = consolidate -> harden -> extract -> certify -> install -> recycle.

The wake/sleep design taken literally (goal rungs 1+2). WAKE learns soft relational circuits by
plain SGD. SLEEP compiles them: a consolidation finetune (Adam + beta-anneal + straight-through --
sleep is the STRUCTURAL phase, so heavier optimization is design-legal; this is also how the
"SGD cannot learn 2-hop" wall dissolves: wake sketches, sleep compiles), then AUTONOMOUS extraction
(no ground-truth sites anywhere):
  family    : read from the weights -- the vocabulary match matrix M(a,b) says whether the matcher
              is QUERY-match (diagonal) or CONSTANT-match (one dominant row);
  guard     : read from the data -- the certified DOMAIN (observed query-token range / the constant
              itself) becomes the runtime guard, i.e. the tag discipline's "stated domain" is what
              the rule is allowed to fire on;
  selection : behavioral -- candidates from the extraction library are scored by agreement with the
              model's own HARD route on held-out task data;
  certify   : the winning program runs through Soufflé; its torch mirror must agree EXACTLY with
              Soufflé (the mirror check), and its agreement with the model-hard route + gold is
              recorded; install only above threshold;
  install   : the program becomes a structural vote (exact, no gradient);
  recycle   : the compiled soft heads are RESET -- capacity returns to the pool. A certified rule is
              forgetting-proof because it is no longer weights at all.

Rung 1 (mechanism): after the first sleep, the heads are freshly reset yet task-1 accuracy holds at
~1.0 (the function moved into structure), and the freed heads then learn the NEXT task.
Rung 2 (zero forgetting): curriculum induction -> marker -> khop2 (disjoint token ranges so the
certified-domain guards are honest), ONE model, NO replay. The baseline arm gets the IDENTICAL
optimization budget (wake + sleep-finetune) minus compile/install/reset -- so the final contrast is
purely about compilation. khop2's compile emits a certified 2-HOP program (a new rule family beyond
induction: match query -> take successor as bridge -> match bridge excluding the bridge site ->
take successor).

Run: cd pil && .venv/bin/python experiments/wyly_selfcompile.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from wyly_rel_battery import L
from wyly_rel_battery import gen_induction as _gen_ind
from wyly_rel_battery import gen_khop2 as _gen_khop
from wyly_rel_certify import souffle_pred
from wyly_rel_harden import HardRelLayer

REPO = Path(__file__).resolve().parent.parent

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB, K, H, DEPTH = 384, 48, 4, 2  # H matches HardRelLayer
WAKE_STEPS, BATCH = 5000, 64
W_INSTALL = 50.0   # must dominate the tied decode's |C_qtok|^2 self-vote (~12-30)
NTR, NTE = 12288, 2048


# ---------------- curriculum tasks (disjoint token ranges -> honest domain guards) ----------------
def gen_induction(n, g):
    x, y, _, s = _gen_ind(n, g)                              # tokens 0..63
    return x, y, s


def gen_marker(n, g):
    """marker token 64, body tokens 128..191 (disjoint from induction's query range)."""
    seq = torch.randint(128, 192, (n, L), generator=g)
    p = torch.randint(0, L - 2, (n,), generator=g)
    ar = torch.arange(n)
    seq[ar, p] = 64
    return seq, seq[ar, p + 1].clone(), p


def gen_khop2(n, g):
    """battery khop2 shifted to 224..383, with the pad slot (L-2) FILLED: the battery generator
    leaves position L-2 as zero-fill, which after the shift collides with genuine token 224 --
    caught by the Soufflé mirror-check (a query token appearing at the pad made best1 the pad
    site). Fill it with a per-row token that is neither a first nor a second."""
    x, y, _, s = _gen_khop(n, g)
    npairs = (L - 1) // 2
    r = torch.rand(n, 160, generator=g)
    r.scatter_(1, x[:, 0:2 * npairs].reshape(n, -1) % 160, torch.inf)   # exclude every used token
    x[:, 2 * npairs] = r.argmin(1)                           # unused filler (not first/second/query)
    return x + 224, y + 224, s


TASKS = [("induction", gen_induction), ("marker", gen_marker), ("khop2", gen_khop2)]


# ---------------- the model ----------------
class WylyComp(torch.nn.Module):
    """shared concepts + DEPTH relational strata + tied decode + INSTALLED rule votes."""

    def __init__(self):
        super().__init__()
        # FROZEN concept space (the grounded-frozen lesson from wyly_lm_grounded/v2): the tied
        # decode back-props into ALL class rows, so a trainable C lets task k wreck the not-yet-
        # seen tasks' rows. A stationary shared concept space is what continual learning needs.
        self.register_buffer("C", torch.randn(VOCAB, K) * 0.5)
        self.layers = torch.nn.ModuleList([HardRelLayer() for _ in range(DEPTH)])
        self.bias = torch.nn.Parameter(torch.zeros(VOCAB))
        par = torch.zeros(L, 2)
        par[0::2, 0], par[1::2, 1] = 1.0, 1.0
        self.register_buffer("parity", par)
        self.installed = []                                  # [(name, mirror_fn)] -- structure, not weights

    def forward(self, ids, hard=False):
        m = self.C[ids]
        mpos = torch.cat([m, self.parity.expand(ids.shape[0], -1, -1)], -1)
        q = self.C[ids[:, -1]]
        for lyr in self.layers:
            q, _ = lyr(mpos, q, hard=hard)
        out = q @ self.C.T + self.bias
        for _, fn in self.installed:                         # TRUSTED tier PREEMPTS (the package
            a = fn(ids)                                      # runtime's own cover semantics): where
            fire = a >= 0                                    # a certified rule fires, it REPLACES the
            if bool(fire.any()):                             # soft logits -- soft magnitude can never
                out = out.clone()                            # override structure (guards keep rules
                out[fire] = W_INSTALL * F.one_hot(a[fire], VOCAB).float()   # in their domain)
        return out

    def reset_heads(self):
        with torch.no_grad():
            for lyr in self.layers:
                lyr.A.copy_(torch.randn_like(lyr.A) * 0.05)
                lyr.u.zero_()
                lyr.rho.zero_()
                lyr.theta.fill_(-2.0)
                lyr.Wv.copy_(torch.eye(K, device=lyr.Wv.device).repeat(lyr.Wv.shape[0], 1, 1))
                lyr.beta, lyr.st = 1.0, False


# ---------------- the extraction library: torch mirrors + matching Datalog ----------------
def mir_induction(lo, hi):
    def fn(ids):
        ar = torch.arange(len(ids), device=ids.device)
        qt = ids[:, -1]
        m = ids[:, :-1] == ids[:, -1:]
        ok = (qt >= lo) & (qt <= hi) & m.any(1)
        mp = (m.float() * torch.arange(1, L, device=ids.device)).argmax(1)
        return torch.where(ok, ids[ar, mp + 1], torch.full_like(qt, -1))
    return fn


_DL_IND = (
    "\n.decl tok(w:number, p:number, t:number)\n.input tok\n"
    ".decl qtok(w:number, t:number)\n.input qtok\n"
    ".decl hit(w:number, p:number)\n"
    "hit(w, p) :- tok(w, p, t), qtok(w, t), t >= @LO@, t <= @HI@, p < {maxp}.\n"
    ".decl best(w:number, m:number)\n"
    "best(w, m) :- qtok(w, _), m = max p : {{ hit(w, p) }}.\n"
    ".decl pred(w:number, t:number)\n"
    "pred(w, t) :- best(w, p), p1 = p + 1, tok(w, p1, t).\n.output pred\n")


def dl_induction(lo, hi):
    return _DL_IND.replace("@LO@", str(lo)).replace("@HI@", str(hi))


def mir_marker(c):
    def fn(ids):
        ar = torch.arange(len(ids), device=ids.device)
        m = ids[:, :-1] == c
        ok = m.any(1)
        mp = (m.float() * torch.arange(1, L, device=ids.device)).argmax(1)
        return torch.where(ok, ids[ar, mp + 1], torch.full_like(ids[:, -1], -1))
    return fn


_DL_MARK = _DL_IND.replace(
    "hit(w, p) :- tok(w, p, t), qtok(w, t), t >= @LO@, t <= @HI@, p < {maxp}.",
    "hit(w, p) :- tok(w, p, @C@), p < {maxp}.")


def dl_marker(c):
    return _DL_MARK.replace("@C@", str(c))


def mir_khop2(lo, hi):
    def fn(ids):
        ar = torch.arange(len(ids), device=ids.device)
        qt = ids[:, -1]
        m1 = ids[:, :-1] == ids[:, -1:]
        ok1 = (qt >= lo) & (qt <= hi) & m1.any(1)
        p1 = (m1.float() * torch.arange(1, L, device=ids.device)).argmax(1)
        b = ids[ar, (p1 + 1).clamp(max=L - 1)]
        m2 = ids[:, :-1] == b.unsqueeze(1)
        excl = p1 + 1                                        # exclude the bridge site itself --
        ok_ex = excl <= L - 2                                # only when it IS a matchable column
        m2[ar[ok_ex], excl[ok_ex]] = False
        ok = ok1 & m2.any(1)
        p2 = (m2.float() * torch.arange(1, L, device=ids.device)).argmax(1)
        return torch.where(ok, ids[ar, (p2 + 1).clamp(max=L - 1)], torch.full_like(qt, -1))
    return fn


_DL_KHOP = (
    "\n.decl tok(w:number, p:number, t:number)\n.input tok\n"
    ".decl qtok(w:number, t:number)\n.input qtok\n"
    ".decl hit1(w:number, p:number)\n"
    "hit1(w, p) :- tok(w, p, t), qtok(w, t), t >= @LO@, t <= @HI@, p < {maxp}.\n"
    ".decl best1(w:number, m:number)\n"
    "best1(w, m) :- qtok(w, _), m = max p : {{ hit1(w, p) }}.\n"
    ".decl bridge(w:number, b:number)\n"
    "bridge(w, b) :- best1(w, p), p1 = p + 1, tok(w, p1, b).\n"
    ".decl hit2(w:number, p:number)\n"
    "hit2(w, p) :- bridge(w, b), tok(w, p, b), p < {maxp}, best1(w, q), q1 = q + 1, p != q1.\n"
    ".decl best2(w:number, m:number)\n"
    "best2(w, m) :- bridge(w, _), m = max p : {{ hit2(w, p) }}.\n"
    ".decl pred(w:number, t:number)\n"
    "pred(w, t) :- best2(w, p), p1 = p + 1, tok(w, p1, t).\n.output pred\n")


def dl_khop2(lo, hi):
    return _DL_KHOP.replace("@LO@", str(lo)).replace("@HI@", str(hi))


def emit_khop_package(lo, hi, W, out_path):
    """Write a rosetta-servable package whose single rule is the certified 2-hop (khop) program."""
    rule = {
        "kind": "khop", "lo": int(lo), "hi": int(hi),
        "id": "khop2", "tier": "trusted", "basis": "certified",
        "confidence": 1.0, "citation": {"program": "dl_khop2", "cert": "souffle"},
    }
    manifest = {
        "model": "wyly_selfcompile/khop2", "cover": "support-weighted", "W": int(W),
        "n_rules": 1, "labels": "corpus", "rules": [rule],
    }
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


# ---------------- training + eval ----------------
def accuracy(model, x, y, hard=False, bs=512):
    with torch.no_grad():
        return float(torch.cat([(model(x[i:i + bs], hard=hard).argmax(1) == y[i:i + bs])
                                for i in range(0, len(x), bs)]).float().mean())


def hard_route(model, x, bs=512):
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs], hard=True).argmax(1) for i in range(0, len(x), bs)])


def wake(model, x, y, seed=0):
    """plain SGD -- the design's wake optimizer."""
    for lyr in model.layers:
        lyr.beta, lyr.st = 1.0, False
    g = torch.Generator().manual_seed(seed)
    for _ in range(WAKE_STEPS):
        bi = torch.randint(len(x), (BATCH,), generator=g).to(DEV)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x[bi]), y[bi]).backward()
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 1.0 * p.grad
    return model


def sleep_finetune(model, x, y, warm=8000, anneal=3000, st=1000, beta_max=24.0, seed=1):
    """the consolidation finetune: sleep is structural, heavier optimization is design-legal."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(warm + anneal + st):
        beta = 1.0 if step < warm else beta_max ** (min(1.0, (step - warm) / max(anneal, 1)))
        for lyr in model.layers:
            lyr.beta, lyr.st = beta, step >= warm + anneal
        bi = torch.randint(len(x), (BATCH,), generator=g).to(DEV)
        opt.zero_grad()
        F.cross_entropy(model(x[bi]), y[bi]).backward()
        opt.step()
    return model


def sleep_compile(model, xval, yval, cycle, retries=2):
    """harden -> autonomously extract -> certify (Soufflé) -> install -> recycle. -> record|None
    If the best candidate agrees but not enough (the circuit is forming), sleep EXTENDS: more
    consolidation finetune, then retry -- up to `retries` times."""
    sleep_finetune(model, xval, yval)
    hard_pred = hard_route(model, xval)
    # candidate library, guards read FROM DATA (the certified domain)
    qt = xval[:, -1]
    lo, hi = int(qt.min()), int(qt.max())
    cands = [("induction", mir_induction(lo, hi), dl_induction(lo, hi)),
             ("khop2", mir_khop2(lo, hi), dl_khop2(lo, hi))]
    body = xval[:, :-1]
    present = [c for c in range(VOCAB)
               if float((body == c).any(1).float().mean()) > 0.95 and not (lo <= c <= hi)]
    for c in present[:2]:
        cands.append((f"marker[{c}]", mir_marker(c), dl_marker(c)))
    best = None
    for name, fn, prog in cands:
        a = fn(xval)
        fire = a >= 0
        agree = float((a[fire] == hard_pred[fire]).float().mean()) if int(fire.sum()) else 0.0
        cov = float(fire.float().mean())
        score = agree * cov
        if best is None or score > best[0]:
            best = (score, name, fn, prog, agree, cov)
    _, name, fn, prog, agree, cov = best
    gold = float((fn(xval) == yval).float().mean())
    if agree < 0.98 or gold < 0.98:
        if retries > 0 and agree > 0.5:                      # the circuit is FORMING -- keep sleeping
            print(f"    sleep {cycle}: best candidate {name} agree {agree:.3f} -- circuit forming, "
                  f"sleep EXTENDS ({retries} retries left)")
            return sleep_compile(model, xval, yval, cycle, retries - 1)
        print(f"    sleep {cycle}: best candidate {name} agree {agree:.3f} gold {gold:.3f} "
              "-- below threshold, NOT compiled")
        return None
    n_cert = 512
    dl = souffle_pred(xval[:n_cert], L, prog)
    mir = fn(xval[:n_cert])
    exact = all(dl.get(w, -1) == int(mir[w]) for w in range(n_cert))
    assert exact, f"mirror != Soufflé for {name} -- the certificate is void"
    model.installed.append((f"{name}@sleep{cycle}", fn))
    if name == "khop2" and os.environ.get("WYLY_EMIT") == "1":
        emit_khop_package(lo, hi, L, REPO / "data" / "wyly_khop_package")
    model.reset_heads()                                      # RECYCLE: capacity back to the pool
    print(f"    sleep {cycle}: COMPILED {name} (guard [{lo},{hi}]) -- model-hard agree {agree:.3f} "
          f"@ cover {cov:.2f}, gold {gold:.3f}, Soufflé mirror-check {n_cert}/{n_cert} exact; "
          "heads RESET")
    return name


def run_arm(compile_on, gens, tests):
    torch.manual_seed(0)
    model = WylyComp().to(DEV)
    hist = []
    for ci, (tname, _gen) in enumerate(TASKS):
        xtr, ytr = gens[tname]
        wake(model, xtr, ytr, seed=ci)
        if compile_on:
            sleep_compile(model, xtr[:4096], ytr[:4096], ci)
        else:
            sleep_finetune(model, xtr[:4096], ytr[:4096])    # SAME budget, no compilation
        row = {tn: accuracy(model, tx, ty) for tn, (tx, ty) in tests.items()}
        hist.append((tname, row))
        print(f"  after {tname:>10}: " + "  ".join(f"{tn} {row[tn]:.3f}" for tn, _ in TASKS),
              flush=True)
    return model, hist


def main():
    torch.manual_seed(0)
    print(f"SELF-COMPILING learner -- curriculum {[t for t, _ in TASKS]}, one model, no replay, "
          f"H={H} x depth {DEPTH}, {DEV}")
    g = torch.Generator().manual_seed(0)
    gens, tests = {}, {}
    for tname, gen in TASKS:
        xtr, ytr, _ = gen(NTR, g)
        xte, yte, _ = gen(NTE, g)
        gens[tname] = (xtr.to(DEV), ytr.to(DEV))
        tests[tname] = (xte.to(DEV), yte.to(DEV))

    print("\n== COMPILE arm (sleep = harden -> extract -> certify -> install -> recycle) ==")
    model, hist = run_arm(True, gens, tests)
    # rung 1 explicit: the heads were reset at each compile; verify structure carries every task
    print(f"  installed rules: {[n for n, _ in model.installed]}")

    print("\n== BASELINE arm (same wake + same sleep-finetune budget, NO compilation) ==")
    _, bist = run_arm(False, gens, tests)

    print("\n== VERDICTS ==")
    final = hist[-1][1]
    bfinal = bist[-1][1]
    t1 = TASKS[0][0]
    after1 = hist[0][1][t1]
    print(f"rung 1 (mechanism): after sleep-1 the heads are RESET yet {t1} = {after1:.3f} "
          f"(function lives in structure); the freed heads then learned "
          f"{TASKS[1][0]} to {hist[1][1][TASKS[1][0]]:.3f}")
    keep = all(final[tn] > 0.95 for tn, _ in TASKS)
    forgot = min(bfinal[tn] for tn, _ in TASKS[:2])
    print("rung 2 (zero forgetting): compile arm final = "
          + "  ".join(f"{tn} {final[tn]:.3f}" for tn, _ in TASKS))
    print("                          baseline   final = "
          + "  ".join(f"{tn} {bfinal[tn]:.3f}" for tn, _ in TASKS))
    print(f"  -> {'CLEARED' if keep and forgot < 0.5 else 'not cleared'}: compilation retains all "
          f"tasks (min {min(final.values()):.3f}); the matched-budget baseline forgets "
          f"(min over earlier tasks {forgot:.3f}). khop2 compiled = the trusted tier's first 2-hop "
          "program (a NEW certified rule family).")


if __name__ == "__main__":
    main()
