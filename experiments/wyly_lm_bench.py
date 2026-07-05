"""Wyly-LM corrected benchmark: ONE protocol, every comparison redone + the disciplined proposer study.

Replaces the cross-protocol table the endgame review was built on (see WYLY_LM_ENDGAME_REVIEW_FABLE.md
sections 1-2): temporal split only, one concept budget, seed 0, and every rules row reports its own
RULES-ABLATED top-1 (all rules deactivated on the same trained model) so the rules' marginal
contribution is measured, never inferred across rows.

Two questions, two tables:
  (1) PROTOCOL TABLE -- bigram floor, concepts-only (CTX=4 and CTX=12: quantifies the context confound),
      transformer ceiling (lr-tuned; the shared Adam-0.01 recipe diverges at 2 layers).
  (2) PROPOSER STUDY (Fable rec 2) -- the sparse k-slot rule arm under five proposal disciplines:
        random          uniform literal indices + shout head (dec += 2)      [as published]
        random-dm       data-measure birth (body sampled from a mispredicted window's active literals,
                        decode-neutral head)                                 [rule-learner lesson]
        correlation     top-KSLOT marginally-discriminative literals + shout [as published; redundant
                        with the linear path BY CONSTRUCTION -- review sec 2(i)]
        interaction     conjunction-level scoring: best PAIR by E_pos[la*lb]-E_neg[la*lb] (one Gram
                        difference), extended greedily by the literal keeping the RUNNING conjunction
                        discriminative + shout head
        disciplined     interaction + decode-neutral birth + held-out ADMISSION GATE (a cohort that
                        doesn't improve val top-1 is deactivated -- the MDL-judge shape)
      Falsifiable verdict (printed): if no arm's rules-marginal exceeds +0.01, this task is declared
      conjunction-free at this scale and proposer claims move to the long-range battery.

Run: cd pil && .venv/bin/python experiments/wyly_lm_bench.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import wyly_lm as w0
from wyly_data import load_windows

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V, K0, KSLOT = 2048, 192, 4
RMAX, RGROW = 1536, 128
WAKE_STEPS, LR, EPISODES = 500, 0.5, 6
NVAL = 3000                                       # held out from TRAIN for the admission gate


class WylyBench(torch.nn.Module):
    """token->concept table + k-slot sparse-gather conjunction rules + linear head (CTX param)."""

    def __init__(self, vocab, ctx=4):
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn(vocab, K0) * 0.5)
        self.ctx = ctx
        self.D = K0 * ctx + ctx
        self.register_buffer("idx", torch.randint(0, 2 * self.D, (RMAX, KSLOT)))
        self.dec = torch.nn.Parameter(torch.zeros(self.D + RMAX, V))
        self.bias = torch.nn.Parameter(torch.zeros(V))
        self.register_buffer("active", torch.zeros(RMAX, dtype=torch.bool))
        self.register_buffer("born", torch.full((RMAX,), -1, dtype=torch.long))   # episode of birth

    def base(self, ids):
        c = self.C[ids]
        cur = ids[:, -1:]
        cc = [c[:, -1 - i] for i in range(self.ctx)]
        ones = torch.ones(ids.shape[0], 1, device=ids.device)
        eqs = [(ids[:, -1 - i:-i] == cur).any(1, keepdim=True).float() if i else ones
               for i in range(self.ctx)]
        return torch.cat(cc + eqs, -1)

    def literals(self, f):
        return torch.cat([torch.sigmoid(f), 1 - torch.sigmoid(f)], -1)

    def forward(self, ids):
        f = self.base(ids)
        fire = self.literals(f)[:, self.idx].prod(-1) * self.active
        return torch.cat([f, fire], -1) @ self.dec + self.bias

    def sgd(self, lr):
        for p in [self.C, self.dec, self.bias]:
            if p.grad is not None:
                p.data -= lr * p.grad


def flogits(model, ids, idxs, bs=256):
    return torch.cat([model(ids[idxs[i:i + bs]]) for i in range(0, len(idxs), bs)])


def top1(model, ids, y, idxs):
    with torch.no_grad():
        return float((flogits(model, ids, idxs).argmax(1) == y[idxs]).float().mean())


def train_steps(model, ids, y, idxb, g, steps):
    for _ in range(steps):
        bi = idxb[torch.randint(len(idxb), (128,), generator=g)]
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(ids[bi]), y[bi]).backward()
        model.sgd(LR)


# ---------------- proposers ----------------
def wrong_stats(model, ids, y, s):
    """literals + mispredicted-target candidates on a sample s (shared by all proposers)."""
    lit = model.literals(model.base(ids[s]))
    pred = flogits(model, ids, s).argmax(1)
    wrong = pred != y[s]
    return lit, wrong


def grow_random(model, ids, y, s, g, ep, shout=True):
    nr = torch.where(~model.active)[0][:RGROW]
    model.idx[nr] = torch.randint(0, 2 * model.D, (len(nr), KSLOT), device=model.idx.device)
    model.active[nr], model.born[nr] = True, ep


def grow_random_dm(model, ids, y, s, g, ep, shout=False):
    """data-measure birth: body = KSLOT literals ACTIVE (>0.6) on a real mispredicted window;
    decode-neutral head (no shout). The rule fires on at least the window that birthed it."""
    lit, wrong = wrong_stats(model, ids, y, s)
    wi = torch.where(wrong)[0]
    if not len(wi):
        return grow_random(model, ids, y, s, g, ep)
    free = torch.where(~model.active)[0][:RGROW]
    picks = wi[torch.randint(len(wi), (len(free),), generator=g)]
    for r, w in zip(free.tolist(), picks.tolist(), strict=False):
        strong = torch.where(lit[w] > 0.6)[0]
        if len(strong) < KSLOT:
            strong = lit[w].topk(KSLOT).indices
        sel = strong[torch.randperm(len(strong), generator=g)[:KSLOT]]
        model.idx[r] = sel
        model.active[r], model.born[r] = True, ep


def grow_correlation(model, ids, y, s, g, ep, shout=True):
    """[as published] top-KSLOT MARGINALLY-discriminative literals for a mispredicted target."""
    lit, wrong = wrong_stats(model, ids, y, s)
    if wrong.sum() < 10:
        return grow_random(model, ids, y, s, g, ep)
    wt, wc = y[s][wrong].unique(return_counts=True)
    cand_t = wt[wc.argsort(descending=True)[:RGROW]]
    free = torch.where(~model.active)[0][:len(cand_t)]
    for r, t in zip(free.tolist(), cand_t.tolist(), strict=False):
        pos = y[s] == t
        if int(pos.sum()) < 3:
            continue
        score = lit[pos].mean(0) - lit[~pos].mean(0)
        model.idx[r] = score.topk(KSLOT).indices
        if shout:
            model.dec.data[model.D + r, t] += 2.0
        model.active[r], model.born[r] = True, ep


def grow_interaction(model, ids, y, s, g, ep, shout=True, min_pos=20, max_neg=4000):
    """conjunction-level scoring: best pair by one Gram-difference, greedy conditional extension."""
    lit, wrong = wrong_stats(model, ids, y, s)
    if wrong.sum() < 10:
        return grow_random(model, ids, y, s, g, ep)
    wt, wc = y[s][wrong].unique(return_counts=True)
    wt, wc = wt[wc >= min_pos], wc[wc >= min_pos]
    if not len(wt):
        return grow_random(model, ids, y, s, g, ep)
    cand_t = wt[wc.argsort(descending=True)[:RGROW]]
    free = torch.where(~model.active)[0][:len(cand_t)]
    for r, t in zip(free.tolist(), cand_t.tolist(), strict=False):
        pos = y[s] == t
        lp, ln = lit[pos], lit[~pos]
        if len(ln) > max_neg:
            ln = ln[torch.randperm(len(ln), generator=g)[:max_neg]]
        sc2 = lp.T @ lp / len(lp) - ln.T @ ln / len(ln)
        sc2.fill_diagonal_(-1e9)
        flat = int(sc2.argmax())
        sel = [flat // sc2.shape[1], flat % sc2.shape[1]]
        gp, gn = lp[:, sel[0]] * lp[:, sel[1]], ln[:, sel[0]] * ln[:, sel[1]]
        for _ in range(KSLOT - 2):
            sc = (gp.unsqueeze(1) * lp).mean(0) - (gn.unsqueeze(1) * ln).mean(0)
            sc[torch.tensor(sel, device=sc.device)] = -1e9
            c = int(sc.argmax())
            sel.append(c)
            gp, gn = gp * lp[:, c], gn * ln[:, c]
        model.idx[r] = torch.tensor(sel, device=model.idx.device)
        if shout:
            model.dec.data[model.D + r, t] += 2.0
        model.active[r], model.born[r] = True, ep


def grow_disciplined(model, ids, y, s, g, ep, shout=False):
    return grow_interaction(model, ids, y, s, g, ep, shout=False)


# ---------------- harness ----------------
def run_rules(ids, y, tr, te, grow, gate=False, s_size=6000, make_model=None):
    """episodic SGD + proposer growth; optional held-out ADMISSION GATE per birth cohort."""
    torch.manual_seed(0)
    make = make_model if make_model is not None else WylyBench
    model = make(ids.max().item() + 1).to(DEV)
    g = torch.Generator().manual_seed(0)
    val = tr[torch.randperm(len(tr), generator=g)[:NVAL]]
    fit = tr[~torch.isin(tr, val)]
    trc = fit[ids[fit, -1].argsort()]                       # non-stationary episode stream
    for ep, ch in enumerate(torch.chunk(trc, EPISODES)):
        train_steps(model, ids, y, ch, g, WAKE_STEPS)
        if grow is not None and int((~model.active).sum()) >= RGROW:
            s = ch[torch.randint(len(ch), (s_size,), generator=g)]
            grow(model, ids, y, s, g, ep)
        train_steps(model, ids, y, ch, g, WAKE_STEPS // 2)
        if gate:                                            # admit only cohorts that help held-out val
            cohort = (model.born == ep) & model.active
            if int(cohort.sum()):
                with_c = top1(model, ids, y, val)
                model.active[cohort] = False
                without_c = top1(model, ids, y, val)
                if with_c - without_c > 0:
                    model.active[cohort] = True             # admitted
    acc = top1(model, ids, y, te)
    saved = model.active.clone()
    model.active.zero_()
    acc0 = top1(model, ids, y, te)
    model.active.copy_(saved)
    return acc, acc0, int(saved.sum())


def run_concepts(ids, y, tr, te, ctx):
    torch.manual_seed(0)
    model = WylyBench(ids.max().item() + 1, ctx=ctx).to(DEV)
    g = torch.Generator().manual_seed(0)
    trc = tr[ids[tr, -1].argsort()]
    for ch in torch.chunk(trc, EPISODES):
        train_steps(model, ids, y, ch, g, WAKE_STEPS + WAKE_STEPS // 2)
    return top1(model, ids, y, te)


def run_transformer(ids, y, tr, te, nlayer, lr):
    torch.manual_seed(0)
    model = w0.TinyTransformer(ids.max().item() + 1, V, nlayer=nlayer).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(5000):
        bi = tr[torch.randint(len(tr), (128,), generator=g)]
        opt.zero_grad()
        F.cross_entropy(model(ids[bi]), y[bi]).backward()
        opt.step()
    return top1(model, ids, y, te)


def run_bigram(ids, y, tr, te, vocab):
    torch.manual_seed(0)
    big = torch.nn.Embedding(vocab, V).to(DEV)
    opt = torch.optim.Adam(big.parameters(), lr=0.01)
    g = torch.Generator().manual_seed(0)
    for _ in range(3000):
        bi = tr[torch.randint(len(tr), (256,), generator=g)]
        opt.zero_grad()
        F.cross_entropy(big(ids[bi][:, -1]), y[bi]).backward()
        opt.step()
    return top1(lambda x: big(x[:, -1]), ids, y, te)


def main():
    ids, y, vocab, tr, te = load_windows(V, DEV)
    print(f"Wyly-LM corrected benchmark -- TEMPORAL split only, vocab {vocab}, V {V}, {DEV}, "
          f"{ids.shape[0]} windows (train {len(tr)} / test {len(te)})\n")

    print("== (1) PROTOCOL TABLE ==")
    print(f"{'row':>34}{'top-1':>9}")
    print(f"{'bigram lookup (Adam)':>34}{run_bigram(ids, y, tr, te, vocab):>9.3f}", flush=True)
    for ctx in [4, 12]:
        print(f"{f'concepts-only SGD (CTX={ctx})':>34}{run_concepts(ids, y, tr, te, ctx):>9.3f}",
              flush=True)
    for nl, lr in [(1, 1e-3), (2, 1e-3)]:
        print(f"{f'transformer {nl}L (Adam lr {lr:g})':>34}{run_transformer(ids, y, tr, te, nl, lr):>9.3f}",
              flush=True)

    print("\n== (2) PROPOSER STUDY (CTX=4, sparse k-slot rules) ==")
    print(f"{'proposer':>16}{'top-1':>9}{'ablated':>9}{'marginal':>10}{'rules':>7}")
    arms = [
        ("random", grow_random, False),
        ("random-dm", grow_random_dm, False),
        ("correlation", grow_correlation, False),
        ("interaction", grow_interaction, False),
        ("disciplined", grow_disciplined, True),
    ]
    best = ("", -1.0)
    for name, grow, gate in arms:
        acc, acc0, nr = run_rules(ids, y, tr, te, grow, gate=gate)
        marg = acc - acc0
        if marg > best[1]:
            best = (name, marg)
        print(f"{name:>16}{acc:>9.3f}{acc0:>9.3f}{marg:>+10.3f}{nr:>7}", flush=True)

    print(f"\nVERDICT: best rules-marginal = {best[1]:+.3f} ({best[0]}); threshold +0.010.")
    if best[1] > 0.010:
        print("-> conjunctions PAY on this task; the proposer ladder is real signal here.")
    else:
        print("-> task declared CONJUNCTION-FREE at this scale (expected: transformer <= bigram here;")
        print("   see review sec 1.6). Proposer claims move to the long-range battery "
              "(wyly_rel_battery.py).")


if __name__ == "__main__":
    main()
