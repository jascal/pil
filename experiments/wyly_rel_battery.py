"""Relational Wyly + the long-range synthetic battery: can RULES actually replace ATTENTION?

The wikitext-12tok arc could not test this (no super-linear headroom: transformer <= bigram there;
WYLY_LM_ENDGAME_REVIEW_FABLE.md sec 1.6), and its rule language could not even EXPRESS the induction
head (eq-flags detect "seen before" but nothing retrieves what FOLLOWED the match; review sec 4).
This experiment adds the missing constructs and tests them where attention's work is the ground truth.

RELATIONAL RULE HEAD (the language extension, all PIC-native):
  gate  : T3 threshold gate over the query stream's literals, sigma(beta*(rho . lit - theta)) --
          the repo-validated gate family (parity fix in the rule-learner thread), subsumes k-AND.
  match : BILINEAR matcher over RAW position codes, score_i = [c_i,parity]^T A q / sqrt(K) + u.[c_i]
          -- i-orca C7: token equality needs a bilinear reader GIVEN near-orthonormal features; raw
          codes keep that geometry (sigma-squashed memberships do NOT -- v1 of this file matched on
          sigmoids and sat at chance; the positive-orthant mean swamps the equality signal).
  route : SUCCESSOR ROUTING -- alpha = softmax(score); v = sum_i alpha_i * c_{i+1}; the rule writes
          the MATCHED POSITION'S SUCCESSOR code into the query stream residually, q' = q + gate*(v@Wv)
          with Wv init = I. This is the value transport eq-flags lacked; one head = one discrete
          attention head in rule form. Position codes carry an even/odd parity flag (a positional
          literal, as in the Datalog side's offset predicates).
  decode: TIED -- logits = q_final @ C^T (route a token's code -> decode reads it by similarity; the
          OV/unembedding tying). Gives the matcher direct gradient from step one.
  Depth 2 = two relational strata (hop-2 reads what hop-1 routed).

BATTERY (ctx L=128, ground truth known, laptop-scale):
  induction : ... A B ... A -> B      (THE canonical attention circuit; last token seen once before)
  khop2     : pairs (a b) meaning f(a)=b, seconds disjoint from firsts except one bridge;
              query a -> f(f(a)) -- needs two chained hops (depth test)
  marker    : one MARKER token in context; target = its successor; the query token is uninformative
              (tests content match on a UNARY concept, not query equality)

CERTIFICATION (what "rules, not soft attention" means here):
  hard-route eval: replace softmax routing with one-hot argmax at eval -- accuracy must hold.
  match fidelity: fraction of test rows where some head's argmax == the true source position.
SGD row = the design's optimizer (plain SGD, lr piloted); Adam row = reference (review Q5: any gap =
optimizer, not the rule form). The transformer reference gets ~4x WylyRel's step budget + an lr pilot;
if it still hasn't formed induction heads at that budget, that is reported, not hidden -- the
relational head's inductive bias (successor routing built into the form) is exactly the point.

Run: cd pil && .venv/bin/python experiments/wyly_rel_battery.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, K, H, DEPTH = 128, 48, 4, 2
NTR, NTE, BATCH, STEPS, TR_STEPS, PILOT = 16384, 2048, 64, 8000, 30000, 500


# ---------------- task generators (all vectorized, seed-deterministic) ----------------
def gen_induction(n, g):
    """random stream; the last token occurred exactly once before, at p; target = seq[p+1]. W=64."""
    w = 64
    seq = torch.randint(0, w, (n, L), generator=g)
    q = torch.randint(0, w, (n,), generator=g)
    clash = seq == q.unsqueeze(1)
    seq[clash] = (seq[clash] + 1) % w                       # q occurs nowhere else
    p = torch.randint(0, L - 2, (n,), generator=g)
    ar = torch.arange(n)
    seq[ar, p] = q
    seq[:, -1] = q
    return seq, seq[ar, p + 1].clone(), w, p


def gen_khop2(n, g):
    """(a b) pairs = f; seconds disjoint from firsts except ONE bridge f(a)=b with b a first;
    query a at the end; target = f(f(a)). W=160, 63 pairs. True sites: hop1 2*i0, hop2 2*i1."""
    w, npairs = 160, (L - 1) // 2
    r = torch.rand(n, w, generator=g)
    firsts = r.argsort(1)[:, :npairs]                       # distinct per row
    r2 = torch.rand(n, w, generator=g)
    r2.scatter_(1, firsts, torch.inf)                       # seconds from the complement
    seconds = r2.argsort(1)[:, :npairs]
    ij = torch.rand(n, npairs, generator=g).argsort(1)[:, :2]
    i0, i1 = ij[:, 0], ij[:, 1]
    ar = torch.arange(n)
    seconds[ar, i0] = firsts[ar, i1]                        # the bridge: f(a)=b, b also a first
    target = seconds[ar, i1].clone()
    seq = torch.zeros(n, L, dtype=torch.long)
    seq[:, 0:2 * npairs:2] = firsts
    seq[:, 1:2 * npairs:2] = seconds
    seq[:, -1] = firsts[ar, i0]                             # query a
    return seq, target, w, torch.stack([2 * i0, 2 * i1], 1)


def gen_marker(n, g):
    """one MARKER token (id w) somewhere; target = its successor; last token is random junk. W=65."""
    w = 64
    seq = torch.randint(0, w, (n, L), generator=g)
    p = torch.randint(0, L - 2, (n,), generator=g)
    ar = torch.arange(n)
    seq[ar, p] = w
    return seq, seq[ar, p + 1].clone(), w + 1, p


# ---------------- the relational Wyly ----------------
class RelLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kp = K + 2                                          # raw codes + even/odd parity flags
        self.A = torch.nn.Parameter(torch.randn(H, kp, K) * 0.05)    # bilinear matcher (small init:
        self.u = torch.nn.Parameter(torch.zeros(H, kp))              #  routing starts near-uniform)
        self.rho = torch.nn.Parameter(torch.zeros(H, 2 * K))         # T3 gate over query literals
        self.theta = torch.nn.Parameter(torch.full((H,), -2.0))      # gate starts open
        self.Wv = torch.nn.Parameter(torch.eye(K).repeat(H, 1, 1))   # identity write

    def forward(self, mpos, q, hard=False):
        """mpos (B,L,K+2) raw position codes+parity; q (B,K) query stream -> (q', score)."""
        mq = torch.sigmoid(q)
        lit = torch.cat([mq, 1 - mq], -1)
        gate = torch.sigmoid(4.0 * (lit @ self.rho.T - self.theta))          # (B,H) T3 gate
        score = torch.einsum("blk,hkj,bj->bhl", mpos, self.A, q) / K ** 0.5
        score = score + torch.einsum("blk,hk->bhl", mpos, self.u)
        score = score[:, :, :-1]                            # cannot match the query position itself
        if hard:
            alpha = F.one_hot(score.argmax(-1), score.shape[-1]).to(score.dtype)
        else:
            alpha = torch.softmax(score, -1)
        succ = mpos[:, 1:, :K]                              # RAW successor codes, aligned with sites
        v = torch.einsum("bhl,blk->bhk", alpha, succ)
        write = torch.einsum("bhk,hkj->bj", v * gate.unsqueeze(-1), self.Wv)
        return q + write, score


class WylyRel(torch.nn.Module):
    """raw token codes + DEPTH stacked relational rule heads + TIED decode (q @ C^T)."""

    def __init__(self, vocab):
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn(vocab, K) * 0.5)
        self.layers = torch.nn.ModuleList([RelLayer() for _ in range(DEPTH)])
        self.bias = torch.nn.Parameter(torch.zeros(vocab))
        par = torch.zeros(L, 2)
        par[0::2, 0], par[1::2, 1] = 1.0, 1.0
        self.register_buffer("parity", par)

    def forward(self, ids, hard=False, want_scores=False):
        m = self.C[ids]                                     # (B,L,K) RAW codes (near-orthogonal, C7)
        mpos = torch.cat([m, self.parity.expand(ids.shape[0], -1, -1)], -1)
        q = self.C[ids[:, -1]]
        scores = []
        for lyr in self.layers:
            q, sc = lyr(mpos, q, hard=hard)
            scores.append(sc)
        out = q @ self.C.T + self.bias                      # tied decode
        return (out, scores) if want_scores else out


class NonRel(torch.nn.Module):
    """the OLD architecture (concepts of last 4 positions + eq flags): expected ~chance here."""

    def __init__(self, vocab, ctx=4):
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn(vocab, K) * 0.5)
        self.ctx = ctx
        self.dec = torch.nn.Parameter(torch.zeros(K * ctx + ctx, vocab))
        self.bias = torch.nn.Parameter(torch.zeros(vocab))

    def forward(self, ids):
        m = torch.sigmoid(self.C[ids])
        cur = ids[:, -1:]
        cc = [m[:, -1 - i] for i in range(self.ctx)]
        ones = torch.ones(ids.shape[0], 1, device=ids.device)
        eqs = [(ids[:, -1 - i:-i] == cur).any(1, keepdim=True).float() if i else ones
               for i in range(self.ctx)]
        return torch.cat(cc + eqs, -1) @ self.dec + self.bias


class PreLNTransformer(torch.nn.Module):
    """the attention reference: 2-layer pre-LN causal transformer, full L-token window."""

    def __init__(self, vocab, nlayer=2, ke=128):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, ke)
        self.pos = torch.nn.Parameter(torch.randn(L, ke) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(ke, 4, ke * 4, batch_first=True, dropout=0.0,
                                                 norm_first=True)
        self.tr = torch.nn.TransformerEncoder(layer, nlayer, enable_nested_tensor=False)
        self.dec = torch.nn.Linear(ke, vocab)

    def forward(self, ids):
        x = self.emb(ids) + self.pos[:ids.shape[1]]
        mask = torch.triu(torch.ones(ids.shape[1], ids.shape[1], device=ids.device), 1).bool()
        return self.dec(self.tr(x, mask=mask)[:, -1])


# ---------------- training ----------------
def accuracy(model, x, y, hard=False, bs=512):
    with torch.no_grad():
        hits = 0
        for i in range(0, len(x), bs):
            out = model(x[i:i + bs], hard=hard) if isinstance(model, WylyRel) else model(x[i:i + bs])
            hits += int((out.argmax(1) == y[i:i + bs]).sum())
        return hits / len(x)


def train(model, x, y, steps, opt_name, lr, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = (torch.optim.Adam(model.parameters(), lr=lr) if opt_name == "adam"
           else torch.optim.SGD(model.parameters(), lr=lr))
    for _ in range(steps):
        bi = torch.randint(len(x), (BATCH,), generator=g).to(x.device)
        opt.zero_grad()
        F.cross_entropy(model(x[bi]), y[bi]).backward()
        opt.step()
    return model


def pilot_lr(make, x, y, opt_name, lrs, seed=0):
    """short pilot on train acc -- same tuning courtesy for every arm."""
    best = (None, -1.0)
    for lr in lrs:
        torch.manual_seed(seed)
        m = train(make().to(DEV), x, y, PILOT, opt_name, lr, seed)
        a = accuracy(m, x[:2048], y[:2048])
        if a > best[1]:
            best = (lr, a)
    return best[0]


def match_fidelity(model, x, sites, bs=512):
    """fraction of rows where some head's argmax hits the true source position, per layer."""
    with torch.no_grad():
        hit = torch.zeros(len(model.layers))
        for i in range(0, len(x), bs):
            _, scores = model(x[i:i + bs], want_scores=True)
            s = sites[i:i + bs]
            for li, sc in enumerate(scores):
                tgt = s[:, li] if s.dim() == 2 else s
                hit[li] += ((sc.argmax(-1) == tgt.unsqueeze(1)).any(1)).sum().cpu()
        return (hit / len(x)).tolist()


def main():
    torch.manual_seed(0)
    print(f"RELATIONAL Wyly battery -- ctx L={L}, K={K}, H={H} heads, depth {DEPTH}, {DEV}")
    print("rule head = T3 gate x bilinear match (raw codes) x successor routing, tied decode\n")
    for name, gen in [("induction", gen_induction), ("khop2", gen_khop2), ("marker", gen_marker)]:
        g = torch.Generator().manual_seed(0)
        xtr, ytr, vocab, _ = gen(NTR, g)
        xte, yte, _, sites = gen(NTE, g)
        xtr, ytr, xte, yte = xtr.to(DEV), ytr.to(DEV), xte.to(DEV), yte.to(DEV)
        sites = sites.to(DEV)
        print(f"== {name}  (vocab {vocab}, chance {1 / vocab:.3f}) ==")
        rows = []
        m = train(NonRel(vocab).to(DEV), xtr, ytr, STEPS // 2, "adam", 1e-2)
        rows.append(("old form (ctx4+eq, Adam)", accuracy(m, xte, yte), None, None))
        lr = pilot_lr(lambda v=vocab: PreLNTransformer(v), xtr, ytr, "adam", [3e-4, 1e-3, 3e-3])
        m = train(PreLNTransformer(vocab).to(DEV), xtr, ytr, TR_STEPS, "adam", lr)
        rows.append((f"transformer 2L ({TR_STEPS // 1000}k, Adam {lr:g})",
                     accuracy(m, xte, yte), None, None))
        for opt_name, lrs in [("sgd", [1.0, 0.5, 0.2]), ("adam", [1e-3, 3e-3])]:
            lr = pilot_lr(lambda v=vocab: WylyRel(v), xtr, ytr, opt_name, lrs)
            torch.manual_seed(0)
            m = train(WylyRel(vocab).to(DEV), xtr, ytr, STEPS, opt_name, lr)
            rows.append((f"WylyRel ({opt_name} {lr:g})", accuracy(m, xte, yte),
                         accuracy(m, xte, yte, hard=True), match_fidelity(m, xte, sites)))
        print(f"{'model':>30}{'top-1':>8}{'hard-route':>12}   match-fidelity/layer")
        for nm, acc, hardacc, fid in rows:
            h = f"{hardacc:.3f}" if hardacc is not None else "-"
            f_ = " ".join(f"{v:.2f}" for v in fid) if fid else "-"
            print(f"{nm:>30}{acc:>8.3f}{h:>12}   {f_}", flush=True)
        print()
    print("read: WylyRel ~ (or >) transformer >> old form/chance = the relational rule head DOES the")
    print("work attention does here; hard-route ~ soft certifies the head is a DISCRETE relational")
    print("rule (a gather, not soft attention); match-fidelity = the learned matcher points at the")
    print("true source site (relational incidence is learned by gradient on the bilinear form --")
    print("the discrete-proposer problem does not arise for match rules). khop2 needs BOTH strata")
    print("(depth pays exactly when the task composes -- review Q3/Q5).")


if __name__ == "__main__":
    main()
