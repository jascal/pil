"""Benchmark the PIC rule learner on standard AI/NN problems, in real LLM token spaces.

Tasks (all expressed in the pythia/GPT-NeoX vocabulary via ``pil.tokens.TokenSpace``,
matching the LLMs modeled in ../rosetta):

  parity8 / parity16   the classic NN-hardness probe; parity16 is evaluated on *unseen*
                       bit patterns (memorization impossible: 2^16 patterns, ~17% seen)
  modadd               a + b = c (mod 97), the grokking benchmark, 50% of all pairs held out;
                       tokens are the actual pythia tokens for "0".."96", "+", "="
  induction            A B ... A -> B with position-varying A (the recursive PIC-LP clause,
                       learned as ground rules via ambiguity-driven specialization)
  wikitext             next-token LM on wikitext-2 (pythia tokenizer), vs a bigram floor and
                       an MLP; plus decision agreement with pythia-160m itself on the rosetta
                       package's logit-cache contexts (same tokenizer, same windows)

Baseline: an MLP (embedding -> ReLU hidden -> logits) trained on identical data with a
comparable parameter budget. The rule program reports its *hard* (Boolean-gate, Datalog-
semantics) accuracy -- the number the exported Soufflé program reproduces exactly.

Usage:
  python experiments/rule_learner_bench.py --task parity8
  python experiments/rule_learner_bench.py --task all --out results/rule_learner_bench.txt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pil.rule_learner import PICRuleLearner, RuleLearnerConfig
from pil.rules import RuleProgram, RuleProgramConfig, program_summary
from pil.schemas import SchemaBank, arithmetic_library, copy_library, propose_schemas
from pil.tokens import TokenSpace

ROSETTA_PKG = Path("/home/allans/code/rosetta/models/pythia160m")
SCRATCH = Path("/tmp/claude-1000/-home-allans-code-pil/9c868468-870b-45a8-b78c-fb33d0f1ee47/scratchpad")


# ---------------------------------------------------------------- baselines
class MLP(nn.Module):
    def __init__(self, vocab: int, window: int, n_cand: int, emb: int = 32, hidden: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.net = nn.Sequential(
            nn.Linear(window * emb, hidden), nn.ReLU(), nn.Linear(hidden, n_cand)
        )

    def forward(self, x):
        z = self.emb(x).flatten(1)
        return self.net(z)


def train_mlp(X, y, n_cand, vocab, epochs=60, bs=256, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    mlp = MLP(vocab, X.shape[1], n_cand)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0], generator=g)
        for i in range(0, X.shape[0], bs):
            idx = perm[i : i + bs]
            loss = F.cross_entropy(mlp(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return mlp


@torch.no_grad()
def mlp_acc(mlp, X, y):
    outs = [mlp(X[i : i + 4096]).argmax(-1) for i in range(0, X.shape[0], 4096)]
    return float((torch.cat(outs) == y).float().mean())


def used_param_count(mlp: MLP, X: torch.Tensor) -> int:
    """MLP params, counting only embedding rows the task actually uses."""
    used = X.unique().numel()
    emb_dim = mlp.emb.embedding_dim
    dense = sum(p.numel() for p in mlp.net.parameters())
    return used * emb_dim + dense


def rule_param_count(prog: RuleProgram, X: torch.Tensor | None = None) -> int:
    """Hardened-program size: active literals + head/frame/bias floats + used lookup rows."""
    lits = sum(
        int((torch.sigmoid(s.rel) > 0.5).sum().item()) for s in prog.strata if s.rel.numel()
    )
    heads = prog.n_rules * prog.cfg.frame_dim
    frame = prog.U.numel() + prog.bias.numel()
    lkp = 0
    if X is not None:
        for o in prog.lookup:
            lkp += X[:, int(o)].unique().numel() * prog.cfg.frame_dim
    return lits + heads + frame + lkp


# ---------------------------------------------------------------- tasks
def bench_parity(n_bits: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    N = 6000 if n_bits <= 8 else 12000
    bits = torch.randint(0, 2, (N, n_bits), generator=g)
    X, y = bits + 10, bits.sum(1).remainder(2) + 10
    # strictly-unseen-pattern test set
    codes = (bits * (2 ** torch.arange(n_bits))).sum(1)
    tb = torch.randint(0, 2, (4000, n_bits), generator=g)
    tc = (tb * (2 ** torch.arange(n_bits))).sum(1)
    fresh = ~torch.isin(tc, codes) if n_bits > 10 else torch.ones(len(tc), dtype=torch.bool)
    Xte, yte = tb[fresh] + 10, tb[fresh].sum(1).remainder(2) + 10

    cfg = RuleProgramConfig(
        vocab_size=50304, window=n_bits, frame_dim=16, candidates=[10, 11], seed=seed
    )
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=8 if n_bits <= 8 else 10, init_rules=96, births_per_phase=64,
        epochs_per_phase=25, max_rules=512, recency_gamma=1.0, seed=seed, verbose=False,
    )
    learner = PICRuleLearner(prog, lc)
    t0 = time.time()
    learner.fit(X, y)
    t_rules = time.time() - t0

    yte_i = prog.target_index(yte)
    hard = learner._acc(Xte, yte_i, hard=True)

    t0 = time.time()
    ycand = prog.target_index(y)
    mlp = train_mlp(X, ycand, 2, 50304, seed=seed)
    t_mlp = time.time() - t0
    return {
        "task": f"parity{n_bits}",
        "eval": f"unseen patterns (n={len(yte_i)})",
        "rule_hard_acc": hard,
        "mlp_acc": mlp_acc(mlp, Xte, yte_i),
        "rules": prog.n_rules,
        "rule_params": rule_param_count(prog),
        "mlp_params": used_param_count(mlp, X),
        "rule_s": round(t_rules), "mlp_s": round(t_mlp),
        "summary": program_summary(prog),
    }


def bench_modadd(p: int = 97, seed: int = 0) -> dict:
    ts = TokenSpace.from_rosetta_package(ROSETTA_PKG)
    tok = [ts.encode(str(r))[0] for r in range(p)]
    plus, eq = ts.encode("+")[0], ts.encode("=")[0]
    tok_t = torch.tensor(tok)

    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    X = torch.stack([tok_t[a], torch.full_like(a, plus), tok_t[b], torch.full_like(a, eq)], dim=1)
    y = tok_t[(a + b).remainder(p)]

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    n_tr = (p * p) // 2
    tr, te = perm[:n_tr], perm[n_tr:]

    cfg = RuleProgramConfig(
        vocab_size=50304, window=4, frame_dim=64, candidates=tok, seed=seed
    )
    prog = RuleProgram(cfg)
    # background-knowledge channel (ILP-style): arithmetic + copy schema library over the
    # numeric values of the residue tokens; *selection* is learned (data hit-rate), the
    # weight by SGD. This is what generalizes past the memorized table (the grokking gap).
    values = {int(t): r for r, t in enumerate(tok)}
    library = arithmetic_library(0, 2, p) + copy_library(4)
    accepted, scores = propose_schemas(
        library, X[tr], prog.target_index(y[tr]), prog.candidate_ids, 50304, values,
        min_hit_rate=0.2,
    )
    prog.attach_schemas(SchemaBank(accepted, 50304, values))
    lc = RuleLearnerConfig(
        n_phases=6, init_rules=128, births_per_phase=64, epochs_per_phase=25,
        max_rules=1024, recency_gamma=1.0, thresh_fraction=0.5, seed=seed, verbose=False,
    )
    learner = PICRuleLearner(prog, lc)
    t0 = time.time()
    learner.fit(X[tr], y[tr])
    t_rules = time.time() - t0
    yte_i = prog.target_index(y[te])
    hard = learner._acc(X[te], yte_i, hard=True)

    t0 = time.time()
    mlp = train_mlp(X[tr], prog.target_index(y[tr]), p, 50304, epochs=400, seed=seed)
    t_mlp = time.time() - t0
    return {
        "task": f"modadd(p={p}, 50% train)",
        "eval": f"held-out pairs (n={len(te)})",
        "rule_hard_acc": hard,
        "schemas_adopted": {s.name: round(r, 3) for s, r in zip(accepted, scores, strict=True)},
        "mlp_acc": mlp_acc(mlp, X[te], yte_i),
        "rules": prog.n_rules,
        "rule_params": rule_param_count(prog),
        "mlp_params": used_param_count(mlp, X),
        "rule_s": round(t_rules), "mlp_s": round(t_mlp),
        "summary": program_summary(prog),
    }


def bench_induction(seed: int = 0) -> dict:
    """A B ... A -> B: trigger token repeats; predict its successor. Position varies."""
    ts = TokenSpace.from_rosetta_package(ROSETTA_PKG)
    alphabet = torch.tensor([ts.encode(f" {w}")[0] for w in
                             ["cat", "dog", "sun", "sea", "red", "blue", "old", "new"]])
    A, W, N = len(alphabet), 10, 12000
    g = torch.Generator().manual_seed(seed)
    seq = alphabet[torch.randint(0, A, (N, W), generator=g)]
    j = torch.randint(0, W - 2, (N,), generator=g)
    trig = alphabet[torch.randint(0, A, (N,), generator=g)]
    succ = alphabet[torch.randint(0, A, (N,), generator=g)]
    rows = torch.arange(N)
    seq[rows, j], seq[rows, j + 1], seq[rows, W - 1] = trig, succ, trig
    # scrub accidental extra copies of the trigger (keep the pattern unambiguous)
    for o in range(W - 1):
        protected = (j == o) | (j + 1 == o)
        clash = (seq[:, o] == trig) & ~protected
        while bool(clash.any()):
            seq[clash, o] = alphabet[torch.randint(0, A, (int(clash.sum()),), generator=g)]
            clash = (seq[:, o] == trig) & ~protected
    X, y = seq, succ

    n_tr = int(N * 0.85)
    cfg = RuleProgramConfig(
        vocab_size=50304, window=W, frame_dim=32, candidates=alphabet.tolist(),
        eq_atoms=True, seed=seed,
    )
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=10, init_rules=256, births_per_phase=128, epochs_per_phase=20,
        max_rules=1024, recency_gamma=1.0, ambiguity_split_threshold=0.7,
        max_splits_per_phase=32, seed=seed, verbose=False,
    )
    learner = PICRuleLearner(prog, lc)
    t0 = time.time()
    learner.fit(X[:n_tr], y[:n_tr])
    t_rules = time.time() - t0
    yte_i = prog.target_index(y[n_tr:])
    hard = learner._acc(X[n_tr:], yte_i, hard=True)

    t0 = time.time()
    mlp = train_mlp(X[:n_tr], prog.target_index(y[:n_tr]), A, 50304, epochs=120, seed=seed)
    t_mlp = time.time() - t0
    return {
        "task": "induction (A B .. A -> B)",
        "eval": f"held-out (n={N - n_tr}), chance={1/A:.3f}",
        "rule_hard_acc": hard,
        "mlp_acc": mlp_acc(mlp, X[n_tr:], yte_i),
        "rules": prog.n_rules,
        "rule_params": rule_param_count(prog),
        "mlp_params": used_param_count(mlp, X),
        "rule_s": round(t_rules), "mlp_s": round(t_mlp),
        "summary": program_summary(prog),
    }


def bench_wikitext(seed: int = 0, n_train: int = 60000, n_val: int = 8000, top_v: int = 512) -> dict:
    ts = TokenSpace.from_rosetta_package(ROSETTA_PKG)
    text = (SCRATCH / "wikitext2_train.txt").read_text()
    ids = torch.tensor(ts.encode(text[:2_000_000]), dtype=torch.long)
    W = 8
    # candidate set: the top_v most frequent next tokens
    freq = torch.bincount(ids, minlength=50304)
    cands = freq.topk(top_v).indices.sort().values

    starts = torch.arange(0, len(ids) - W - 1)
    tgt = ids[starts + W]
    keep = torch.isin(tgt, cands)
    starts, tgt = starts[keep], tgt[keep]
    coverage = float(keep.float().mean())
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(len(starts), generator=g)[: n_train + n_val]
    Xall = ids[starts[sel].unsqueeze(1) + torch.arange(W)]
    yall = tgt[sel]
    Xtr, ytr, Xva, yva = Xall[:n_train], yall[:n_train], Xall[n_train:], yall[n_train:]

    # bigram floor: most frequent next token given the last context token
    big: dict[int, torch.Tensor] = {}
    last = Xtr[:, -1]
    for c in last.unique().tolist():
        m = last == c
        big[c] = ytr[m].mode().values
    global_mode = ytr.mode().values
    big_pred = torch.stack([big.get(int(c), global_mode) for c in Xva[:, -1]])
    bigram_acc = float((big_pred == yva).float().mean())

    cfg = RuleProgramConfig(
        vocab_size=50304, window=W, frame_dim=48, candidates=cands.tolist(),
        lookup_offsets=(W - 2, W - 1), seed=seed,   # bigram/trigram-marginal backbone
    )
    prog = RuleProgram(cfg)
    lc = RuleLearnerConfig(
        n_phases=6, init_rules=512, births_per_phase=256, epochs_per_phase=4,
        max_rules=1536, batch_size=512, ambiguity_split_threshold=0.4,
        max_splits_per_phase=24, probe_size=1024, seed=seed, verbose=True,
    )
    learner = PICRuleLearner(prog, lc)
    t0 = time.time()
    learner.fit(Xtr, ytr)
    t_rules = time.time() - t0
    yva_i = prog.target_index(yva)
    hard = learner._acc(Xva, yva_i, hard=True)

    t0 = time.time()
    mlp = train_mlp(Xtr, prog.target_index(ytr), top_v, 50304, epochs=8, seed=seed)
    t_mlp = time.time() - t0

    # LLM comparability: agreement with pythia-160m's own argmax on the rosetta
    # logit-cache contexts (same tokenizer, same 8-token windows)
    llm = json.loads((ROSETTA_PKG / "logit_cache.json").read_text())
    ctxs, llm_arg = [], []
    for key, topk in llm.items():
        ctxs.append([int(t) for t in key.split(",")])
        llm_arg.append(topk[0][0])
    Xllm = torch.tensor(ctxs)
    llm_arg = torch.tensor(llm_arg)
    in_cand = torch.isin(llm_arg, cands)
    with torch.no_grad():
        Lh = learner._logits_in_chunks(Xllm, hard=True)
        ours = prog.candidate_ids[Lh.argmax(-1)]
    llm_agree = float((ours[in_cand] == llm_arg[in_cand]).float().mean())

    return {
        "task": f"wikitext-2 (pythia tok, top-{top_v} cands, cover {coverage:.2f})",
        "eval": f"val positions (n={n_val})",
        "rule_hard_acc": hard,
        "mlp_acc": mlp_acc(mlp, Xva, yva_i),
        "bigram_floor": bigram_acc,
        "pythia160m_agreement_on_cached_ctxs": llm_agree,
        "rules": prog.n_rules,
        "rule_params": rule_param_count(prog, Xtr),
        "mlp_params": used_param_count(mlp, Xtr),
        "rule_s": round(t_rules), "mlp_s": round(t_mlp),
        "summary": program_summary(prog),
    }


TASKS = {
    "parity8": lambda: bench_parity(8),
    "parity16": lambda: bench_parity(16),
    "modadd": bench_modadd,
    "induction": bench_induction,
    "wikitext": bench_wikitext,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=[*TASKS, "all"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    names = list(TASKS) if args.task == "all" else [args.task]
    for name in names:
        print(f"=== {name} ===", flush=True)
        r = TASKS[name]()
        rows.append(r)
        for k, v in r.items():
            if k != "summary":
                print(f"  {k}: {v}")
        print(f"  summary: {r['summary']}", flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
