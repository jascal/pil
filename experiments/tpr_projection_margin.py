"""TPR projection as margin widening (pre-registered: docs/notes/tpr_projection_margin_prereg.md).

    # 1. dump (needs `transformers`; e.g. run with lm-sae's venv — pil itself stays numpy+torch):
    /path/to/venv-with-transformers/bin/python experiments/tpr_projection_margin.py dump \
        --out runs/tpr_margin/dump.npz
    # 2. evaluate (pil venv):
    .venv/bin/python experiments/tpr_projection_margin.py evaluate --dump runs/tpr_margin/dump.npz \
        --out runs/tpr_margin/summary.json

Site: GPT-2 small decode input u = ln_f(h_L) at the final position of a list-copy prompt;
logits = U u (tied, no bias).
Arms: real u; oracle TPR û(σ); cleanup û(σ̂) (linear unbind -> snap -> rebind); projection onto span(W);
rank-matched PCA.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NOUNS = """apple table river horse window garden pencil mirror candle bottle rabbit forest island castle
bridge jacket carpet basket kitchen engine ladder pepper rubber lion trouble kindness coffee
chair house money water music paper stone bread cloud train plane ship tree flower school church
phone glass metal wood fire snow rain wind salt sugar milk cheese butter egg fish bird dog cat
cow pig sheep mouse snake bear wolf fox deer duck eagle shark whale frog bee rose grass sand
rock gold silver iron steel coal oil glove shirt shoe hat coat ring clock lamp bed door wall
roof floor road street park farm field hill lake sea ocean beach desert valley cave star moon
sun planet book letter map flag box bag cup plate knife""".split()
N_LIST, PS = 5, (1, 2, 3, 4)


def roles_of(p):
    """Pair roles: list slot i -> (i, p) in [0, 20); query slot j < p -> 20 + offset(p) + j in [20, 30)."""
    offset = {1: 0, 2: 1, 3: 3, 4: 6}[p]
    return [i * len(PS) + (p - 1) for i in range(N_LIST)], [20 + offset + j for j in range(p)]


N_ROLES = 30


def cmd_dump(args):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # optional dependency, dump only

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token, tok.padding_side = tok.eos_token, "right"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    nouns = [w for w in NOUNS if len(tok.encode(" " + w)) == 1][:100]
    ids = [tok.encode(" " + w)[0] for w in nouns]
    rng = random.Random(args.seed)
    rows = []
    while len(rows) < args.n:
        words = rng.sample(range(len(nouns)), N_LIST)
        p = rng.choice(PS)
        text = ("Here is a list of words: " + ", ".join(nouns[w] for w in words) + ". Again: "
                + ", ".join(nouns[w] for w in words[:p]) + ",")
        rows.append((text, words, p))
    us, argmax = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            enc = tok([r[0] for r in chunk], return_tensors="pt", padding=True).to(device)
            out = model.transformer(**enc)
            last = enc["attention_mask"].sum(1) - 1
            u = out.last_hidden_state[torch.arange(len(last), device=device), last]  # already ln_f-applied
            us.append(u.float().cpu())
            argmax.append(model.lm_head(u).argmax(-1).cpu())
    np.savez_compressed(args.out, u=torch.cat(us).numpy(), argmax=torch.cat(argmax).numpy(),
                        words=np.array([r[1] for r in rows]), p=np.array([r[2] for r in rows]),
                        filler_ids=np.array(ids), U=model.lm_head.weight.detach().float().cpu().numpy(),
                        nouns=np.array(nouns))
    print(f"wrote {args.out}: {len(rows)} contexts")


def structure(words, p):
    """Padded (filler, role, mask) of width N_LIST + max(p)."""
    n = len(words)
    width = N_LIST + max(PS)
    f = np.zeros((n, width), dtype=np.int64)
    r = np.zeros((n, width), dtype=np.int64)
    m = np.zeros((n, width), dtype=np.float32)
    for i in range(n):
        list_roles, query_roles = roles_of(int(p[i]))
        for j, (w, role) in enumerate(zip(words[i], list_roles, strict=True)):
            f[i, j], r[i, j], m[i, j] = w, role, 1
        for j, role in enumerate(query_roles):
            f[i, N_LIST + j], r[i, N_LIST + j], m[i, N_LIST + j] = words[i][j], role, 1
    return torch.tensor(f), torch.tensor(r), torch.tensor(m)


class TPR(nn.Module):
    def __init__(self, n_fill, d, d_f):
        super().__init__()
        self.f = nn.Embedding(n_fill, d_f)
        self.r = nn.Embedding(N_ROLES, N_ROLES)
        self.W = nn.Linear(d_f * N_ROLES, d)
        nn.init.normal_(self.f.weight, std=0.3)
        nn.init.normal_(self.r.weight, std=0.3)

    def forward(self, f, r, m):
        bound = torch.einsum("bpi,bpj->bpij", self.f(f), self.r(r)) * m[..., None, None]
        return self.W(bound.sum(1).flatten(1))


def fit(module, inputs, target, steps, lr=3e-3, wd=1e-5):
    opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(module(*inputs), target)
        loss.backward()
        opt.step()
    return float(loss.detach())


class Unbind(nn.Module):
    """Linear readouts: list filler per slot, and the query length p (cleanup's estimate of σ)."""

    def __init__(self, d, n_fill):
        super().__init__()
        self.slots = nn.Linear(d, N_LIST * n_fill)
        self.plen = nn.Linear(d, len(PS))
        self.n_fill = n_fill

    def forward(self, x):
        return self.slots(x).view(len(x), N_LIST, self.n_fill), self.plen(x)

    def fit(self, x, words, p, steps=800, lr=3e-3, wd=1e-4):
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        for _ in range(steps):
            opt.zero_grad()
            slots, plen = self(x)
            loss = F.cross_entropy(slots.flatten(0, 1), words.flatten()) + F.cross_entropy(plen, p - 1)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def predict(self, x):
        slots, plen = self(x)
        return slots.argmax(-1), plen.argmax(-1) + 1


@torch.no_grad()
def decode_metrics(vec, U, gold, real_logp, real_argmax, chunk=512):
    from pil.geometry import margin_to_worst

    margins, agree, acc, kls = [], [], [], []
    for s in range(0, len(vec), chunk):
        L = vec[s:s + chunk] @ U.T
        margins.append(margin_to_worst(L, gold[s:s + chunk]))
        am = L.argmax(-1)
        agree.append(am == real_argmax[s:s + chunk])
        acc.append(am == gold[s:s + chunk])
        lp = L.log_softmax(-1)
        rp = real_logp(s, s + chunk)
        kls.append((rp.exp() * (rp - lp)).sum(-1))
    mg = torch.cat(margins)
    return dict(gold_acc=float(torch.cat(acc).float().mean()),
                agree_real=float(torch.cat(agree).float().mean()),
                kl_from_real=float(torch.cat(kls).mean()), mean_margin=float(mg.mean()),
                median_margin=float(mg.median()),
                retrievable={str(g): float((mg >= g).float().mean()) for g in (0, 1, 2)})


def cmd_evaluate(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    torch.manual_seed(args.seed)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    z = np.load(args.dump)
    u = torch.tensor(z["u"], device=dev)
    U = torch.tensor(z["U"], device=dev)
    words, p = torch.tensor(z["words"], device=dev), torch.tensor(z["p"], device=dev)
    filler_ids = torch.tensor(z["filler_ids"], device=dev)
    gold = filler_ids[words[torch.arange(len(p)), p]]
    n_fill = len(filler_ids)
    n_train = int(len(u) * 2 / 3)
    tr, te = slice(0, n_train), slice(n_train, len(u))
    f, r, m = (t.to(dev) for t in structure(words.cpu().numpy(), p.cpu().numpy()))
    real_logits_te = lambda a, b: (u[te][a:b] @ U.T).log_softmax(-1)  # noqa: E731
    real_argmax_te = torch.tensor(z["argmax"], device=dev)[te]
    results = {"n_train": n_train, "n_test": len(u) - n_train, "arms": {}}

    def record(name, vec, extra=None):
        res = decode_metrics(vec, U, gold[te], real_logits_te, real_argmax_te)
        res.update(extra or {})
        if args.certify:
            from substitution_certificate import coverage_report  # T5(a) per-context certificate (Soufflé)

            res["certificate"] = coverage_report(u[te].double().cpu().numpy(), vec.double().cpu().numpy(),
                                                 U.double().cpu().numpy())
            c = res["certificate"]
            print(f"{'':12s} T5(a) certified / {c['n']}: uniform {c['uniform']['certified']}  "
                  f"hybrid(K={c['hybrid_k']}) {c['hybrid']['certified']}  "
                  f"pairwise {c['pairwise']['certified']}",
                  flush=True)
        results["arms"][name] = res
        print(f"{name:12s} acc={res['gold_acc']:.3f} agree={res['agree_real']:.3f} "
              f"kl={res['kl_from_real']:.3f} "
              f"margin={res['mean_margin']:.2f} retr@1={res['retrievable']['1']:.3f}", flush=True)

    record("real", u[te])
    for d_f in (8, 32):
        tpr = TPR(n_fill, u.shape[1], d_f).to(dev)
        loss = fit(tpr, (f[tr], r[tr], m[tr]), u[tr], args.steps)
        with torch.no_grad():
            oracle = tpr(f[te], r[te], m[te])
        fvu = float(((u[te] - oracle) ** 2).sum() / ((u[te] - u[te].mean(0)) ** 2).sum())
        record(f"oracle-dF{d_f}", oracle, {"fvu": fvu, "train_mse": loss})
        if d_f != 8:
            continue
        unbind = Unbind(u.shape[1], n_fill).to(dev)
        unbind.fit(u[tr], words[tr], p[tr])
        w_hat, p_hat = unbind.predict(u[te])
        sigma_ok = ((w_hat == words[te]).all(-1) & (p_hat == p[te]))
        fh, rh, mh = (t.to(dev) for t in structure(w_hat.cpu().numpy(), p_hat.cpu().numpy()))
        with torch.no_grad():
            clean = tpr(fh, rh, mh)
        record("cleanup", clean, {"sigma_exact": float(sigma_ok.float().mean()),
                                  "slot_acc": float((w_hat == words[te]).float().mean()),
                                  "p_acc": float((p_hat == p[te]).float().mean())})
        # Linear projection onto span(W) (affine: through the TPR bias) and a rank-matched PCA control.
        Wm = tpr.W.weight.detach()
        Q, _ = torch.linalg.qr(Wm)
        rank = int(torch.linalg.matrix_rank(Wm))
        Q = Q[:, :rank]
        b = tpr.W.bias.detach()
        proj = b + (u[te] - b) @ Q @ Q.T
        mu = u[tr].mean(0)
        _, _, Vh = torch.linalg.svd(u[tr] - mu, full_matrices=False)
        P = Vh[:rank].T
        pca = mu + (u[te] - mu) @ P @ P.T
        # T6(a) diagnostic: share of ||U_gold - U_rival||^2 inside span(W), real logits' top-5 rivals.
        with torch.no_grad():
            Lr = u[te] @ U.T
            Lr.scatter_(-1, gold[te][:, None], float("-inf"))
            rivals = Lr.topk(5, dim=-1).indices
            diff = U[gold[te]][:, None, :] - U[rivals]
            inside = ((diff @ Q) ** 2).sum(-1) / (diff ** 2).sum(-1)
        record("proj-tpr", proj, {"rank": rank, "readout_diff_share_in_span": float(inside.mean())})
        record("proj-pca", pca, {"rank": rank})
    if args.post_hoc_kl:
        # POST-HOC (not pre-registered): same TPR form (d_F = 8), trained on KL(real || TPR) of the decode
        # instead of MSE. Separates "the TPR form cannot carry the decision" from "MSE spends capacity on
        # decode-irrelevant directions". Fitting to the host's logits is no longer DISCOVER's
        # representational test.
        tpr = TPR(n_fill, u.shape[1], 8).to(dev)
        opt = torch.optim.Adam(tpr.parameters(), lr=3e-3)
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        for _ in range(args.steps):
            idx = torch.randint(0, n_train, (512,), device=dev, generator=gen)
            target = (u[idx] @ U.T).log_softmax(-1)
            pred = (tpr(f[idx], r[idx], m[idx]) @ U.T).log_softmax(-1)
            loss = (target.exp() * (target - pred)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            fitted = tpr(f[te], r[te], m[te])
        fvu = float(((u[te] - fitted) ** 2).sum() / ((u[te] - u[te].mean(0)) ** 2).sum())
        record("posthoc-kl-dF8", fitted, {"fvu": fvu, "post_hoc": True})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(tag="empirical", prereg="docs/notes/tpr_projection_margin_prereg.md",
                                   dump=str(args.dump), **results), indent=2) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, default=12000)
    d.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--dump", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--steps", type=int, default=3000)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--device", help="cuda/cpu (default: cuda if available)")
    e.add_argument("--post-hoc-kl", action="store_true", help="add the post-hoc KL-trained TPR arm")
    e.add_argument("--certify", action="store_true",
                   help="issue the T5(a) per-context substitution certificate for every arm (needs souffle)")
    args = ap.parse_args()
    {"dump": cmd_dump, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
