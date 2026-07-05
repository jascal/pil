"""Wyly-LM: replace the WHOLE transformer from RAW TOKENS (the actual endgame -- Wyly IS the LLM).

Prior experiments read a representation pythia already computed (residual / early layer / even its embedding).
That is Wyly as a parasite on a transformer, not Wyly AS the LLM. Here nothing comes from pythia: raw token
window ids -> Wyly (learned token-concept codes replacing the embedding+FFN + relational rules over concepts
+ eq_atoms across positions replacing ATTENTION) -> next token, trained end-to-end. Benchmarked against a
transformer trained on the SAME raw window->next-token task (the attention ceiling) and n-gram floors.

THE question: how far does concept+rule (NO attention) get vs a transformer from raw tokens, and do the
RELATIONAL RULES bridge the n-gram-floor -> transformer-ceiling gap (do rules replace attention)?

Run: cd pil && .venv/bin/python experiments/wyly_lm.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

DATA = Path(os.environ.get(  # canonical dataset -- see experiments/extract_nexttoken.py
    "WYLY_DATA", Path(__file__).resolve().parent.parent / "data" / "wyly_nexttoken_pythia70m.pt"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, R, KE, STEPS = 256, 512, 128, 5000                             # concepts, rules, token-embed dim, steps


class Bigram(torch.nn.Module):
    """flat retrieval floor: last token -> next-token distribution (a learned lookup)."""
    def __init__(self, vocab, nclass):
        super().__init__()
        self.table = torch.nn.Embedding(vocab, nclass)

    def forward(self, ids):
        return self.table(ids[:, -1])


class WylyLM(torch.nn.Module):
    """raw tokens -> learned token-concept codes + relational rules (eq_atoms + cross-position) -> next token.
    use_rules=False = concepts-only (n-gram in concept space); True = full relational rule layer."""
    def __init__(self, vocab, nclass, use_rules=True, ctx=4):
        super().__init__()
        self.C = torch.nn.Embedding(vocab, K)                     # token -> K concept logits
        self.ctx = ctx                                            # recent positions to read
        self.use_rules = use_rules
        feat = K * ctx + ctx                                      # last ctx concepts + eq flags
        if use_rules:
            self.req = torch.nn.Parameter(torch.randn(R, feat) * 0.3 - 2.0)
            self.dec = torch.nn.Linear(feat + R, nclass)          # [features, firings] -> vocab
        else:
            self.dec = torch.nn.Linear(feat, nclass)

    def features(self, ids):
        m = torch.sigmoid(self.C(ids))                            # (B, L, K) per-position concept memberships
        cur = ids[:, -1:]
        cc = [m[:, -1 - i] for i in range(self.ctx)]              # concepts of the last ctx positions
        eqs = [(ids[:, -1 - i:-i] == cur).any(1, keepdim=True).float() if i else (cur == cur).float()
               for i in range(self.ctx)]                          # eq_atoms: earlier position matches cur
        return torch.cat(cc + eqs, -1)

    def forward(self, ids):
        f = self.features(ids)
        if not self.use_rules:
            return self.dec(f)
        req = torch.sigmoid(self.req)
        fire = (1 - req.unsqueeze(0) + req.unsqueeze(0) * f.unsqueeze(1)).prod(-1)   # soft-AND rules
        return self.dec(torch.cat([f, fire], -1))


class TinyTransformer(torch.nn.Module):
    """attention ceiling: a small causal transformer on the SAME raw window -> next token."""
    def __init__(self, vocab, nclass, nlayer=2, ctx=12):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, KE)
        self.pos = torch.nn.Parameter(torch.randn(ctx, KE) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(KE, 4, KE * 4, batch_first=True, dropout=0.0)
        self.tr = torch.nn.TransformerEncoder(layer, nlayer)
        self.dec = torch.nn.Linear(KE, nclass)

    def forward(self, ids):
        x = self.emb(ids) + self.pos[:ids.shape[1]]
        mask = torch.triu(torch.ones(ids.shape[1], ids.shape[1], device=ids.device), 1).bool()
        return self.dec(self.tr(x, mask=mask)[:, -1])


def train_top1(model, ids, y, tr, te, seed=0):
    torch.manual_seed(seed)
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        idx = tr[torch.randint(len(tr), (128,), generator=g)]
        opt.zero_grad()
        F.cross_entropy(model(ids[idx]), y[idx]).backward()
        opt.step()
    with torch.no_grad():
        correct = 0
        for i in range(0, len(te), 512):                          # batch eval (soft-AND is memory-heavy)
            b = te[i:i + 512]
            correct += int((model(ids[b]).argmax(1) == y[b]).sum())
        return correct / len(te)


def main():
    d = torch.load(DATA)
    ids = d["kept_ids"]                                           # (N, L) raw token window -- NO pythia repr
    target = d["target"]
    topv = torch.bincount(target).argsort(descending=True)[:2048]  # restrict targets (8GB GPU) to top-2048
    keep = torch.isin(target, topv)
    ids, target = ids[keep], target[keep]
    uv, ids = ids.reshape(-1).unique(return_inverse=True)         # compact input vocab (shrinks the tables)
    ids = ids.reshape(target.shape[0], -1).to(DEV)
    uniq, y = target.unique(return_inverse=True)
    y = y.to(DEV)
    vocab = len(uv)
    nclass, n = len(uniq), ids.shape[0]
    tr = torch.arange(int(0.85 * n), device=DEV)
    te = torch.arange(int(0.85 * n), n, device=DEV)
    print(f"Wyly-LM (Wyly IS the LLM) -- raw tokens only, window L={ids.shape[1]}, vocab {vocab}, "
          f"nclass {nclass}, {DEV}")
    print("reference: pythia-70m final-residual->vocab was 0.189 (the transformer we aim to replace)\n")
    rows = [
        ("bigram (flat floor)", Bigram(vocab, nclass)),
        ("Wyly concepts-only (no rules)", WylyLM(vocab, nclass, use_rules=False)),
        ("Wyly + relational RULES", WylyLM(vocab, nclass, use_rules=True)),
        ("transformer 1-layer (attn)", TinyTransformer(vocab, nclass, nlayer=1)),
        ("transformer 2-layer (attn)", TinyTransformer(vocab, nclass, nlayer=2)),
    ]
    print(f"{'model':>32}{'top-1':>9}")
    res = {}
    for name, m in rows:
        a = train_top1(m, ids, y, tr, te)
        res[name] = a
        print(f"{name:>32}{a:>9.3f}", flush=True)
    conc = res["Wyly concepts-only (no rules)"]
    rule = res["Wyly + relational RULES"]
    tr2 = res["transformer 2-layer (attn)"]
    print(f"\nrules over concepts-only: {rule - conc:+.3f}   (do relational rules add / replace attention?)")
    print(f"Wyly-rules vs transformer-2L: {rule:.3f} vs {tr2:.3f}  (gap {tr2 - rule:+.3f})")
    print("read: this is the WHOLE-LLM-replacement test -- Wyly from raw tokens vs a transformer on the same")
    print("task. rules >> concepts-only = relational rules capture cross-token (replace attention);")
    print("Wyly-rules near the transformer = a concept+rule machine can stand in for it.")


if __name__ == "__main__":
    main()
