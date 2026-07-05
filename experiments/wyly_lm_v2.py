"""Wyly-LM v2: the assembled model on REAL TEXT at L=256 -- the first Wyly asked to beat the bigram.

Everything validated separately in the arc, composed (see WYLY_LM_ENDGAME_REVIEW_FABLE.md sec 8):
  counts     : bigram LOOKUP path -- online count table over the training stream (memorization done
               STRUCTURALLY, not by gradient; plain SGD provably underfits lookup tables, and a count
               table is what the rosetta packages ship anyway). log1p(counts[last]) into the mix.
  linear     : the grounded-concept linear path (the +0.023 arm): concepts of the last CTX positions
               + eq flags -> class logits. Concepts = z-scored PCA-K of pythia-70m's embedding rows
               (fieldrun bundle), FROZEN -- ground once, run standalone.
  relational : WylyRel heads over the FULL 256-token window (T3 gate x bilinear match on raw frozen
               codes x successor routing, tied decode q @ C[cls]^T) -- the 0.999-certified induction
               machinery from wyly_rel_battery.py, now with recency/parity positional literals.
  conj       : T3-era k-slot conjunction rules over the linear path's literals, grown by the
               interaction proposer under the held-out ADMISSION JUDGE (expected to admit little --
               that is the judge working, and it is part of the assembled design).
Continuous parts train with plain SGD inside wake episodes over the temporal stream; counts update
online; the judge runs in sleep.

THE LADDER (falsifiable, in order):
  (a) beat the Adam-bigram floor on this protocol -- never achieved anywhere in the arc;
  (b) relational marginal (full vs relational-ablated) > +0.02, plus the COPY-SUBSET diagnostic
      (test windows where the induction pattern exists: some earlier position holds the current
      token and its successor IS the target) -- where rules must pay;
  (c) stretch: pythia-70m's own final-residual decode reference is 0.189 (full-document context,
      300B-token training -- reported for orientation, not as a matched comparison).

Run: cd pil && .venv/bin/python experiments/wyly_lm_v2.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from wyly_data import load_windows_tied
from wyly_lm_grounded import grounded_init

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = Path(__file__).resolve().parent.parent / "data" / "wyly_nexttoken_wikitext_L256.pt"
V, K, H, DEPTH, CTX = 4096, 192, 4, 2, 4
RMAX, KSLOT, RGROW = 512, 4, 64
EPISODES, WAKE_STEPS, LR, BATCH = 8, 1200, 0.3, 64
NVAL = 3000


class RelLayer(torch.nn.Module):
    """battery-certified relational rule head; kp = K + 4 positional literals (parity, pos/L, recency)."""

    def __init__(self, kp):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(H, kp, K) * 0.05)
        self.u = torch.nn.Parameter(torch.zeros(H, kp))
        self.rho = torch.nn.Parameter(torch.zeros(H, 2 * K))
        self.theta = torch.nn.Parameter(torch.full((H,), -2.0))
        self.Wv = torch.nn.Parameter(torch.eye(K).repeat(H, 1, 1))

    def forward(self, mpos, q, hard=False):
        mq = torch.sigmoid(q)
        lit = torch.cat([mq, 1 - mq], -1)
        gate = torch.sigmoid(4.0 * (lit @ self.rho.T - self.theta))
        score = torch.einsum("blk,hkj,bj->bhl", mpos, self.A, q) / K ** 0.5
        score = score + torch.einsum("blk,hk->bhl", mpos, self.u)
        score = score[:, :, :-1]
        alpha = (F.one_hot(score.argmax(-1), score.shape[-1]).to(score.dtype) if hard
                 else torch.softmax(score, -1))
        v = torch.einsum("bhl,blk->bhk", alpha, mpos[:, 1:, :K])
        return q + torch.einsum("bhk,hkj->bj", v * gate.unsqueeze(-1), self.Wv), score


class WylyV2(torch.nn.Module):
    def __init__(self, ground, cls, ell):
        super().__init__()
        vocab, ncls = ground.shape[0], len(cls)
        self.register_buffer("C", ground)                    # FROZEN grounded concepts
        self.register_buffer("cls", cls)
        self.register_buffer("counts", torch.zeros(vocab, ncls))
        pos = torch.arange(ell, dtype=torch.float32)
        pf = torch.stack([pos % 2, 1 - pos % 2, pos / ell, torch.exp(-(ell - 1 - pos) / 32)], 1)
        self.register_buffer("posf", pf)                     # positional literals
        self.layers = torch.nn.ModuleList([RelLayer(K + 4) for _ in range(DEPTH)])
        self.D = K * CTX + CTX
        self.W = torch.nn.Parameter(torch.zeros(self.D, ncls))          # grounded linear path
        self.register_buffer("idx", torch.randint(0, 2 * self.D, (RMAX, KSLOT)))
        self.Wconj = torch.nn.Parameter(torch.zeros(RMAX, ncls))
        self.register_buffer("active", torch.zeros(RMAX, dtype=torch.bool))
        self.register_buffer("born", torch.full((RMAX,), -1, dtype=torch.long))
        self.wc = torch.nn.Parameter(torch.tensor(1.0))                 # path mixers
        self.wr = torch.nn.Parameter(torch.tensor(0.1))
        self.wi = torch.nn.Parameter(torch.tensor(1.0))
        self.bias = torch.nn.Parameter(torch.zeros(ncls))
        lut = torch.full((vocab,), -1, dtype=torch.long)
        lut[cls] = torch.arange(ncls)
        self.register_buffer("lut", lut)

    def base(self, ids):
        c = self.C[ids]
        cur = ids[:, -1:]
        cc = [c[:, -1 - i] for i in range(CTX)]
        ones = torch.ones(ids.shape[0], 1, device=ids.device)
        eqs = [(ids[:, -1 - i:-i] == cur).any(1, keepdim=True).float() if i else ones
               for i in range(CTX)]
        return torch.cat(cc + eqs, -1)

    def literals(self, f):
        return torch.cat([torch.sigmoid(f), 1 - torch.sigmoid(f)], -1)

    def relational(self, ids, hard=False):
        m = self.C[ids]
        mpos = torch.cat([m, self.posf.expand(ids.shape[0], -1, -1)], -1)
        q = self.C[ids[:, -1]]
        for lyr in self.layers:
            q, _ = lyr(mpos, q, hard=hard)
        return q @ self.C[self.cls].T                        # tied decode over class rows

    def induction(self, ids):
        """the CERTIFIED rule (wyly_rel_certify.py), installed STRUCTURALLY -- no gradient, exact:
        hit(p) :- tok(p)==tok(last), p<L-1; best = max hit; vote the successor tok(best+1)."""
        with torch.no_grad():
            m = ids[:, :-1] == ids[:, -1:]
            has = m.any(1)
            mp = (m.float() * torch.arange(1, ids.shape[1], device=ids.device)).argmax(1)
            c = self.lut[ids[torch.arange(len(ids), device=ids.device), mp + 1]]
            ok = has & (c >= 0)
            add = torch.zeros(len(ids), self.counts.shape[1], device=ids.device)
            add[torch.where(ok)[0], c[ok]] = 1.0
            return add

    def forward(self, ids, use_rel=True, use_counts=True, use_conj=True, use_ind=True, hard=False):
        f = self.base(ids)
        out = f @ self.W + self.bias
        if use_counts:
            out = out + self.wc * torch.log1p(self.counts[ids[:, -1]])
        if use_ind:
            out = out + self.wi * self.induction(ids)
        if use_rel:
            out = out + self.wr * self.relational(ids, hard=hard)
        if use_conj and self.active.any():
            fire = self.literals(f)[:, self.idx].prod(-1) * self.active
            out = out + fire @ self.Wconj
        return out

    def update_counts(self, ids, y, lut):
        with torch.no_grad():
            prev = torch.cat([ids[:, :-1].reshape(-1), ids[:, -1]])
            nxt = torch.cat([lut[ids[:, 1:].reshape(-1)], y])
            ok = nxt >= 0
            self.counts.view(-1).index_add_(
                0, prev[ok] * self.counts.shape[1] + nxt[ok],
                torch.ones(int(ok.sum()), device=ids.device))

    def sgd(self, lr):
        for p in self.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad


def flogits(model, ids, idxs, bs=192, **kw):
    return torch.cat([model(ids[idxs[i:i + bs]], **kw) for i in range(0, len(idxs), bs)])


def top1(model, ids, y, idxs, **kw):
    with torch.no_grad():
        return float((flogits(model, ids, idxs, **kw).argmax(1) == y[idxs]).float().mean())


def grow_interaction(model, ids, y, s, g, ep, min_pos=20, max_neg=4000):
    """conjunction-level scoring (Gram difference + greedy conditional extension), decode-neutral."""
    with torch.no_grad():
        lit = model.literals(model.base(ids[s]))
        wrong = flogits(model, ids, s).argmax(1) != y[s]
        if wrong.sum() < 10:
            return
        wt, wc_ = y[s][wrong].unique(return_counts=True)
        wt = wt[wc_ >= min_pos]
        wc_ = wc_[wc_ >= min_pos]
        if not len(wt):
            return
        cand_t = wt[wc_.argsort(descending=True)[:RGROW]]
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
            model.active[r], model.born[r] = True, ep


def copy_subset(ids, y, cls, idxs):
    """test windows where the induction pattern exists: ids[p]==ids[-1] and ids[p+1]==target."""
    yv = cls[y[idxs]]
    w = ids[idxs]
    hit = ((w[:, :-1] == w[:, -1:]) & (w[:, 1:] == yv.unsqueeze(1))).any(1)
    return idxs[hit]


def adam_bigram(ids, y, tr, te, vocab, ncls):
    torch.manual_seed(0)
    big = torch.nn.Embedding(vocab, ncls).to(DEV)
    opt = torch.optim.Adam(big.parameters(), lr=0.01)
    g = torch.Generator().manual_seed(0)
    for _ in range(4000):
        bi = tr[torch.randint(len(tr), (256,), generator=g)]
        opt.zero_grad()
        F.cross_entropy(big(ids[bi][:, -1]), y[bi]).backward()
        opt.step()
    with torch.no_grad():
        return float(torch.cat([(big(ids[te[i:i + 512]][:, -1]).argmax(1) == y[te[i:i + 512]])
                                for i in range(0, len(te), 512)]).float().mean())


def main():
    ids, y, cls, uv, tr, te = load_windows_tied(V, DEV, DATA)
    vocab, ncls, n, ell = len(uv), len(cls), ids.shape[0], ids.shape[1]
    print(f"Wyly-LM v2 -- REAL TEXT, L={ell}, vocab {vocab}, classes {ncls}, {n} windows, {DEV}")
    cs = copy_subset(ids, y, cls, te)
    print(f"copy-subset (induction pattern present): {len(cs)}/{len(te)} test windows "
          f"({len(cs) / len(te):.0%})\n")

    floor = adam_bigram(ids, y, tr, te, vocab, ncls)
    print(f"Adam-bigram floor (the ladder's rung a): {floor:.3f}", flush=True)

    ground = grounded_init(uv).to(DEV)
    ground = ground / ground.shape[1] ** 0.5                 # unit-ish rows for the tied decode
    torch.manual_seed(0)
    model = WylyV2(ground, cls, ell).to(DEV)
    lut = torch.full((vocab,), -1, dtype=torch.long, device=DEV)
    lut[cls] = torch.arange(ncls, device=DEV)
    g = torch.Generator().manual_seed(0)
    val = tr[torch.randperm(len(tr), generator=g)[:NVAL]]
    fit = tr[~torch.isin(tr, val)]
    print(f"\n{'episode':>8}{'test top-1':>12}{'copy-subset':>13}{'conj rules':>12}")
    for ep, ch in enumerate(torch.chunk(fit, EPISODES)):     # temporal stream, corpus order
        model.update_counts(ids[ch], y[ch], lut)             # online lookup path
        for _ in range(WAKE_STEPS):
            bi = ch[torch.randint(len(ch), (BATCH,), generator=g)]
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(ids[bi]), y[bi]).backward()
            model.sgd(LR)
        if int((~model.active).sum()) >= RGROW:              # sleep: propose + judge
            s = ch[torch.randint(len(ch), (min(6000, len(ch)),), generator=g)]
            grow_interaction(model, ids, y, s, g, ep)
            cohort = (model.born == ep) & model.active
            if int(cohort.sum()):
                for _ in range(200):
                    bi = ch[torch.randint(len(ch), (BATCH,), generator=g)]
                    model.zero_grad(set_to_none=True)
                    F.cross_entropy(model(ids[bi]), y[bi]).backward()
                    model.sgd(LR)
                with_c = top1(model, ids, y, val)
                model.active[cohort] = False
                if with_c - top1(model, ids, y, val) > 0:
                    model.active[cohort] = True
        print(f"{ep:>8}{top1(model, ids, y, te):>12.3f}{top1(model, ids, y, cs):>13.3f}"
              f"{int(model.active.sum()):>12}", flush=True)

    full = top1(model, ids, y, te)
    norel = top1(model, ids, y, te, use_rel=False)
    nocnt = top1(model, ids, y, te, use_counts=False)
    noconj = top1(model, ids, y, te, use_conj=False)
    noind = top1(model, ids, y, te, use_ind=False)
    hard = top1(model, ids, y, te, hard=True)
    cs_full, cs_norel = top1(model, ids, y, cs), top1(model, ids, y, cs, use_rel=False)
    cs_noind = top1(model, ids, y, cs, use_ind=False)
    print(f"\n{'arm':>34}{'top-1':>9}")
    for nm, a in [("FULL (assembled)", full), ("- relational", norel), ("- counts", nocnt),
                  ("- conjunctions", noconj), ("- certified induction rule", noind),
                  ("FULL, hard-route", hard), ("copy-subset FULL", cs_full),
                  ("copy-subset - relational", cs_norel),
                  ("copy-subset - certified rule", cs_noind)]:
        print(f"{nm:>34}{a:>9.3f}")
    print(f"\nLADDER: (a) full {full:.3f} vs Adam-bigram {floor:.3f} -> "
          f"{'CLEARED' if full > floor else 'not cleared'} ({full - floor:+.3f})")
    print(f"        (b) relational marginal {full - norel:+.3f} overall, "
          f"{cs_full - cs_norel:+.3f} on the copy subset (threshold +0.02)")
    print(f"        (b') certified-rule marginal {full - noind:+.3f} overall, "
          f"{cs_full - cs_noind:+.3f} on the copy subset (learn -> certify -> INSTALL)")
    print("        (c) pythia-70m full-context decode reference: 0.189 (orientation only)")
    out = DATA.parent / "wyly_v2_state.pt"
    torch.save(model.state_dict(), out)                      # for wyly_rel_certify.py
    print(f"\nstate saved: {out}")


if __name__ == "__main__":
    main()
