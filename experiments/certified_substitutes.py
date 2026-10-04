"""Certified compressed substitutes for GPT-2's last layer.

Pre-registered: docs/notes/certified_substitutes_prereg.md.

    # 1. dump (needs `transformers`; run with a venv that has it):
    /path/to/venv/bin/python experiments/certified_substitutes.py dump --out runs/cert_sub
    # 2. run (GPU recommended; needs souffle):
    /path/to/venv/bin/python experiments/certified_substitutes.py run --dumps runs/cert_sub \
        --out runs/cert_sub/summary.json

Each family maps the symbolic structure sigma to an estimate u_hat of GPT-2's decode input.
Every test context gets
the uniform / hybrid / pairwise certificates (i-orca PIC_Binding, kernel-checked) as Soufflé verdicts.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from substitution_certificate import certificate_facts, certify_all_souffle  # noqa: E402
from tpr_projection_margin import N_LIST, NOUNS, PS, roles_of  # noqa: E402

OCCUPATIONS = """doctor lawyer teacher nurse pilot farmer baker chef judge poet painter singer dancer actor
writer author editor banker soldier sailor driver student scientist engineer artist captain
priest waiter guard coach player officer agent tourist manager clerk tailor butcher miner hunter
spy king queen prince""".split()
VERBS = """helped called thanked liked hated pushed followed visited praised blamed hired warned chased
loved watched kissed attacked admired avoided""".split()
SEEDS = dict(stimulus=21, split=22, fit=(0, 1, 2))
N_CONTEXTS = 12000
K_RIVALS = 32
STEPS = 3000


# ---------------------------------------------------------------- dump


def build_rows(task, tok, rng):
    if task == "LIST":
        nouns = [w for w in NOUNS if len(tok.encode(" " + w)) == 1][:100]
        rows = []
        while len(rows) < N_CONTEXTS:
            words = rng.sample(range(len(nouns)), N_LIST)
            p = rng.choice(PS)
            text = (
                "Here is a list of words: "
                + ", ".join(nouns[w] for w in words)
                + ". Again: "
                + ", ".join(nouns[w] for w in words[:p])
                + ","
            )
            list_roles, query_roles = roles_of(p)
            pairs = list(zip(words, list_roles, strict=True)) + list(zip(words[:p], query_roles, strict=True))
            rows.append((text, pairs))
        return rows, len(nouns), 30
    occ = [w for w in OCCUPATIONS if len(tok.encode(" " + w)) == 1][:40]
    verbs = [w for w in VERBS if w.endswith("ed") and len(tok.encode(" " + w)) == 1][:16]
    combos = [
        (s, v, o) for s in range(len(occ)) for o in range(len(occ)) if s != o for v in range(len(verbs))
    ]
    rng.shuffle(combos)
    rows = [
        (
            f"The {occ[s]} {verbs[v]} the {occ[o]}. The {occ[o]} was {verbs[v]} by the",
            [(s, 0), (len(occ) + v, 1), (o, 2)],
        )
        for s, v, o in combos[:N_CONTEXTS]
    ]
    return rows, len(occ) + len(verbs), 3


def cmd_dump(args):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # optional dependency, dump only

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token, tok.padding_side = tok.eos_token, "right"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in ("LIST", "SVO"):
        rng = random.Random(SEEDS["stimulus"] + (0 if task == "LIST" else 1000))
        rows, n_fill, n_role = build_rows(task, tok, rng)
        width = max(len(p) for _, p in rows)
        f = np.zeros((len(rows), width), dtype=np.int64)
        r = np.zeros_like(f)
        m = np.zeros(f.shape, dtype=np.float32)
        for i, (_, pairs) in enumerate(rows):
            for j, (a, s) in enumerate(pairs):
                f[i, j], r[i, j], m[i, j] = a, s, 1
        us = []
        with torch.no_grad():
            for s in range(0, len(rows), 256):
                enc = tok([t for t, _ in rows[s : s + 256]], return_tensors="pt", padding=True).to(device)
                last = enc["attention_mask"].sum(1) - 1
                h = model.transformer(**enc).last_hidden_state  # ln_f applied: the decode input
                us.append(h[torch.arange(len(last), device=device), last].float().cpu())
        np.savez_compressed(
            out / f"{task}.npz",
            u=torch.cat(us).numpy(),
            f=f,
            r=r,
            m=m,
            n_fill=n_fill,
            n_role=n_role,
            U=model.lm_head.weight.detach().float().cpu().numpy(),
        )
        print(f"wrote {out / task}.npz: {len(rows)} contexts, {n_fill} fillers, {n_role} roles", flush=True)


# ---------------------------------------------------------------- families


class Additive(nn.Module):
    def __init__(self, n_fill, n_role, k, d=768):
        super().__init__()
        self.ef, self.er = nn.Embedding(n_fill, k), nn.Embedding(n_role, k)
        self.W = nn.Linear(k, d)

    def forward(self, f, r, m):
        return self.W(((self.ef(f) + self.er(r)) * m[..., None]).sum(1))


class TPR(nn.Module):
    def __init__(self, n_fill, n_role, d_f, d=768):
        super().__init__()
        self.ef, self.er = nn.Embedding(n_fill, d_f), nn.Embedding(n_role, n_role)
        nn.init.normal_(self.ef.weight, std=0.3)
        nn.init.normal_(self.er.weight, std=0.3)
        self.W = nn.Linear(d_f * n_role, d)

    def forward(self, f, r, m):
        bound = torch.einsum("bpi,bpj->bpij", self.ef(f), self.er(r)) * m[..., None, None]
        return self.W(bound.sum(1).flatten(1))


class PairCode(nn.Module):
    def __init__(self, n_fill, n_role, k, d=768):
        super().__init__()
        self.n_role = n_role
        self.c = nn.Embedding(n_fill * n_role, k)
        self.W = nn.Linear(k, d)

    def forward(self, f, r, m):
        return self.W((self.c(f * self.n_role + r) * m[..., None]).sum(1))


def params(family, n_fill, n_role, k_or_df, n_pairs, d=768):
    if family == "additive":
        return k_or_df * (n_fill + n_role) + d * k_or_df + d
    if family == "tpr":
        return k_or_df * n_fill + n_role * n_role + d * k_or_df * n_role + d
    return k_or_df * n_pairs + d * k_or_df + d  # paircode: only codes of SEEN pairs count


def largest_k(family, budget, n_fill, n_role, n_pairs):
    k = 1
    while params(family, n_fill, n_role, k + 1, n_pairs) <= budget:
        k += 1
    return k


def build(family, size, n_fill, n_role):
    return {"additive": Additive, "tpr": TPR, "paircode": PairCode}[family](n_fill, n_role, size)


# ---------------------------------------------------------------- training


def train(model, f, r, m, u, U, objective, seed, steps=None, batch=512, lr=3e-3):
    steps = steps or STEPS
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, len(u), (batch,), generator=g).to(u.device)
        uh = model(f[idx], r[idx], m[idx])
        mse = F.mse_loss(uh, u[idx])
        if objective == "mse":
            loss = mse
        else:
            with torch.no_grad():
                L = u[idx] @ U.T
                top = L.topk(K_RIVALS + 1, dim=1)
                t = top.indices[:, 0]
                rivals = top.indices[:, 1:]
                margin = top.values[:, 0] - top.values[:, 1]
            Lh = uh @ U.T
            slack = Lh.gather(1, t[:, None]) - Lh.gather(1, rivals)
            hinge = F.relu(1 - slack).mean()
            delta = ((u[idx] - uh) @ U.T).abs().max(1).values
            tail = F.relu(2 * delta - margin).mean()
            loss = 0.1 * mse + hinge + 0.1 * tail
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    summary = dict(
        tag="empirical", prereg="docs/notes/certified_substitutes_prereg.md", seeds=SEEDS, tasks={}
    )
    for task in ("LIST", "SVO"):
        z = np.load(Path(args.dumps) / f"{task}.npz")
        n_fill, n_role = int(z["n_fill"]), int(z["n_role"])
        n = len(z["u"])
        order = np.random.default_rng(SEEDS["split"]).permutation(n)
        n_tr, n_va = int(0.6 * n), int(0.1 * n)
        tr, te = order[:n_tr], order[n_tr + n_va :]
        seen = {
            (int(a), int(s)) for i in tr for a, s, k in zip(z["f"][i], z["r"][i], z["m"][i], strict=True) if k
        }
        keep = [
            i
            for i in te
            if all(
                (int(a), int(s)) in seen for a, s, k in zip(z["f"][i], z["r"][i], z["m"][i], strict=True) if k
            )
        ]
        dropped = len(te) - len(keep)
        te = np.array(keep)
        T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
        f, r, m = T(z["f"], torch.long), T(z["r"], torch.long), T(z["m"])
        u, U = T(z["u"]), T(z["U"])
        tr_t, te_t = torch.tensor(tr, device=device), torch.tensor(te, device=device)
        budgets = {"B8": params("tpr", n_fill, n_role, 8, 0), "B32": params("tpr", n_fill, n_role, 32, 0)}
        task_out = dict(
            n_fill=n_fill,
            n_role=n_role,
            pairs_seen=len(seen),
            n_train=len(tr),
            n_test=len(te),
            test_dropped_unseen_pair=dropped,
            budgets=budgets,
            cells={},
        )
        print(
            f"[{time.time() - t0:7.1f}s] {task}: fillers {n_fill} roles {n_role} pairs {len(seen)} "
            f"train {len(tr)} test {len(te)} (dropped {dropped}) budgets {budgets}",
            flush=True,
        )
        for bname, budget in budgets.items():
            sizes = {
                "additive": largest_k("additive", budget, n_fill, n_role, len(seen)),
                "tpr": 8 if bname == "B8" else 32,
                "paircode": largest_k("paircode", budget, n_fill, n_role, len(seen)),
            }
            for family, size in sizes.items():
                for objective in ("mse", "cert"):
                    per_seed = []
                    for seed in SEEDS["fit"]:
                        model = build(family, size, n_fill, n_role).to(device)
                        train(model, f[tr_t], r[tr_t], m[tr_t], u[tr_t], U, objective, seed)
                        with torch.no_grad():
                            uh = model(f[te_t], r[te_t], m[te_t]).double().cpu().numpy()
                        facts = certificate_facts(z["u"][te].astype(np.float64), uh, z["U"])
                        cert = certify_all_souffle(facts)
                        agree = facts["t_hat"] == facts["t"]
                        if any((c & ~agree).any() for c in cert.values()):
                            raise AssertionError("certified context changed its argmax: checker unsound")
                        per_seed.append(
                            {k: float(v.mean()) for k, v in cert.items()} | {"agree": float(agree.mean())}
                        )
                    cell = {
                        k: dict(
                            mean=float(np.mean([s[k] for s in per_seed])),
                            range=[float(min(s[k] for s in per_seed)), float(max(s[k] for s in per_seed))],
                        )
                        for k in per_seed[0]
                    }
                    key = f"{bname}/{family}/{objective}"
                    task_out["cells"][key] = dict(
                        size=size, params=params(family, n_fill, n_role, size, len(seen)), **cell
                    )
                    print(
                        f"[{time.time() - t0:7.1f}s] {task} {key:24s} size {size:4d} "
                        f"params {task_out['cells'][key]['params']:7d}  "
                        f"pairwise {cell['pairwise']['mean']:.3f}  "
                        f"uniform {cell['uniform']['mean']:.3f}  hybrid {cell['hybrid']['mean']:.3f}",
                        flush=True,
                    )
        summary["tasks"][task] = task_out
    summary["verdicts"] = verdicts(summary)
    summary["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["verdicts"], indent=2))


# ---------------------------------------------------------------- pre-registered decision rules


def family_score(cells, budget, family, metric):
    """Better objective's seed-mean coverage (fixed in the pre-registration)."""
    return max(cells[f"{budget}/{family}/{o}"][metric]["mean"] for o in ("mse", "cert"))


def winner(summary, metric, margin=0.03):
    wins = {fam: 0 for fam in ("additive", "tpr", "paircode")}
    table = {}
    for task, t in summary["tasks"].items():
        for b in t["budgets"]:
            scores = {fam: family_score(t["cells"], b, fam, metric) for fam in wins}
            table[f"{task}/{b}"] = scores
            for fam, s in scores.items():
                if all(s >= o + margin for g, o in scores.items() if g != fam):
                    wins[fam] += 1
    champ = [fam for fam, w in wins.items() if w >= 3]
    return dict(scores=table, wins=wins, winner=champ[0] if champ else "no clear winner")


def verdicts(summary):
    h1 = winner(summary, "pairwise")
    h3 = winner(summary, "uniform")
    h2_cells = 0
    detail = {}
    for task, t in summary["tasks"].items():
        for b in t["budgets"]:
            gain = (
                t["cells"][f"{b}/tpr/cert"]["pairwise"]["mean"]
                - t["cells"][f"{b}/tpr/mse"]["pairwise"]["mean"]
            )
            detail[f"{task}/{b}"] = gain
            h2_cells += gain >= 0.10
    best = {}
    for task, t in summary["tasks"].items():
        for b in t["budgets"]:
            key = max(
                (k for k in t["cells"] if k.startswith(b + "/")),
                key=lambda k: t["cells"][k]["pairwise"]["mean"],
            )
            best[f"{task}/{b}"] = dict(
                cell=key, pairwise=t["cells"][key]["pairwise"]["mean"], params=t["cells"][key]["params"]
            )
    return dict(
        H1_pairwise=dict(h1, pass_paircode=h1["winner"] == "paircode"),
        H2_objective=dict(gain_tpr_cert_minus_mse=detail, cells_ge_0_10=int(h2_cells), passes=h2_cells >= 3),
        H3_uniform=h3,
        best_per_budget=best,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--smoke", action="store_true", help="bug check: stimulus seed 999, 600 contexts")
    rr = sub.add_parser("run")
    rr.add_argument("--dumps", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument("--smoke", action="store_true", help="bug check: 30 steps; numbers are NOT results")
    args = ap.parse_args()
    global N_CONTEXTS, STEPS
    if args.smoke:
        SEEDS["stimulus"], N_CONTEXTS, STEPS = 999, 600, 30
    {"dump": cmd_dump, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
