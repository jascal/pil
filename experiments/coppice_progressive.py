"""Coppice end-to-end: progressive learning on a UNIFIED substrate (geometric + token-identity concepts).

The synthesis. One substrate:
  - GEOMETRIC concepts: Kg hyperplanes on residuals (memberships per position)
  - TOKEN-IDENTITY concepts (DEGENERATE): raw token ids -> exact eq_atom token_eq = (current == some earlier)
  - RULES: soft-conjunctions (AND) over [geometric memberships, token_eq] -> per-task heads
Learned PROGRESSIVELY with frozen-core (wake/sleep) consolidation over a sequence:
  A lexical (func, cap) -> geometric concepts
  B relational (local_repeat)         -> token eq_atom (geometry alone CANNOT, per ground_multipos.py)
  C compositional (rep_func = repeat AND func) -> a RULE combining B's eq_atom with A's frozen func-concept

Learners: coppice (unified, frozen-core progressive) vs geometric-only (no token_eq -> should FAIL B/C) vs
monolithic (one net retrained per phase, no consolidation -> should FORGET A) vs joint (all-at-once ceiling).
Metric: after A->B->C, accuracy retained on EACH task family (balanced, chance 0.5).

Run: cd pil && .venv/bin/python experiments/coppice_progressive.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "coppice_pythia70m.pt"
KG, NR = 16, 30                         # geometric concepts, rules
TASKS = ["func", "cap", "local_repeat", "rep_func"]
PHASES = {"A": ["func", "cap"], "B": ["local_repeat"], "C": ["rep_func"]}
RULE_GROUP = {"A": range(0, 12), "B": range(12, 21), "C": range(21, 30)}


class Coppice(torch.nn.Module):
    """Unified substrate + soft-AND rules + per-task heads. use_token_eq gates the token concept."""
    def __init__(self, d, use_token_eq=True):
        super().__init__()
        self.use_token_eq = use_token_eq
        self.Ug = torch.nn.Parameter(torch.randn(KG, d) * (1.0 / d ** 0.5))
        self.bg = torch.nn.Parameter(torch.zeros(KG))
        self.F = KG + (1 if use_token_eq else 0) + 1        # geom memb + [token_eq] + geom_match
        self.req = torch.nn.Parameter(torch.randn(NR, self.F) * 0.3 - 1.0)
        self.heads = torch.nn.Parameter(torch.zeros(len(TASKS), NR + 1))

    def features(self, R, ids, tau=0.6):
        m = torch.sigmoid((R @ self.Ug.T - self.bg) / tau)         # (B, L, KG)
        cur = m[:, -1]                                             # (B, KG)
        gmatch = (m[:, :-1] * cur.unsqueeze(1)).sum(-1).max(1).values.unsqueeze(1) / KG
        feats = [cur, gmatch]
        if self.use_token_eq:
            teq = (ids[:, :-1] == ids[:, -1:]).any(1, keepdim=True).float()   # exact eq_atom
            feats.insert(1, teq)
        return torch.cat(feats, -1)                               # (B, F)

    def fire(self, R, ids):
        f = self.features(R, ids)
        req = torch.sigmoid(self.req)
        term = 1 - req.unsqueeze(0) + req.unsqueeze(0) * f.unsqueeze(1)       # (B, NR, F)
        return term.prod(-1)                                      # (B, NR) soft-AND

    def logit(self, ti, R, ids):
        fr = self.fire(R, ids)
        return fr @ self.heads[ti, :-1] + self.heads[ti, -1]


def acc(model, ti, R, ids, y, te):
    with torch.no_grad():
        return float(((model.logit(ti, R[te], ids[te]) > 0).float() == y[te]).float().mean())


def fit(model, tasks, data, steps, mask=None, lr=0.02, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        losses = []
        for t in tasks:
            R, ids, y, tr, _ = data[t]
            idx = tr[torch.randint(len(tr), (256,), generator=g)]
            lg = model.logit(TASKS.index(t), R[idx], ids[idx])
            losses.append(F.binary_cross_entropy_with_logits(lg, y[idx]))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        if mask is not None:
            for n, p in model.named_parameters():
                if p.grad is not None and n in mask:
                    p.grad *= mask[n]
        opt.step()


def balanced_task(R, ids, y, seed):
    g = torch.Generator().manual_seed(seed)
    pos = torch.where(y == 1)[0]
    neg = torch.where(y == 0)[0][torch.randperm((y == 0).sum(), generator=g)[:len(pos)]]
    keep = torch.cat([pos, neg])[torch.randperm(2 * len(pos), generator=g)]
    Rk = R[keep].clone()
    mu, sd = Rk.reshape(-1, Rk.shape[-1]).mean(0), Rk.reshape(-1, Rk.shape[-1]).std(0) + 1e-6
    Rk = (Rk - mu) / sd
    cut = int(0.7 * len(keep))
    return Rk, ids[keep], y[keep].float(), torch.arange(cut), torch.arange(cut, len(keep))


def rule_mask(model, groups):
    """freeze everything except the rules in `groups` and their heads (concepts frozen)."""
    m = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    idxs = [i for grp in groups for i in RULE_GROUP[grp]]
    m["req"][idxs] = 1
    for grp in groups:
        for t in PHASES[grp]:
            m["heads"][TASKS.index(t)] = 1
    return m


def evaluate(name, data, seed):
    d0 = data[TASKS[0]][0]
    dim = d0.shape[-1]
    if name == "monolithic":                                      # one net, retrained per phase, NO freeze
        model = Coppice(dim, use_token_eq=True)
        for ph in ["A", "B", "C"]:
            fit(model, PHASES[ph], data, 2500, seed=seed)
        return {t: acc(model, TASKS.index(t), *data[t][:3], data[t][4]) for t in TASKS}
    if name == "joint":
        model = Coppice(dim, use_token_eq=True)
        fit(model, TASKS, data, 6000, seed=seed)
        return {t: acc(model, TASKS.index(t), *data[t][:3], data[t][4]) for t in TASKS}
    # coppice (unified) / geometric-only: frozen-core progressive
    model = Coppice(dim, use_token_eq=(name == "coppice"))
    fit(model, PHASES["A"], data, 3000, mask=rule_mask_full(model, "A"), seed=seed)  # A trains concepts
    fit(model, PHASES["B"], data, 2500, mask=rule_mask(model, ["B"]), seed=seed)
    fit(model, PHASES["C"], data, 2500, mask=rule_mask(model, ["C"]), seed=seed)      # C reuses frozen A+B
    return {t: acc(model, TASKS.index(t), *data[t][:3], data[t][4]) for t in TASKS}


def rule_mask_full(model, grp):
    """phase A: train concepts (Ug/bg) + A-rules + A-heads."""
    m = rule_mask(model, [grp])
    m["Ug"][:] = 1
    m["bg"][:] = 1
    return m


def main():
    d = torch.load(DATA)
    R = d["r"].float()
    ids = d["kept_ids"]
    print(f"Coppice progressive: A(func,cap)->B(local_repeat)->C(rep_func)  R {tuple(R.shape)}")
    print("unified substrate = geometric concepts + token-identity eq_atom; frozen-core consolidation\n")
    rows = {}
    for name in ["coppice", "geometric-only", "monolithic", "joint"]:
        seeds = [{} for _ in range(2)]
        for si, s in enumerate((0, 1)):
            data = {t: balanced_task(R, ids, d["labels"][t], s) for t in TASKS}
            seeds[si] = evaluate(name, data, s)
        rows[name] = {t: sum(sd[t] for sd in seeds) / len(seeds) for t in TASKS}
    print(f"{'learner':>15}" + "".join(f"{t:>13}" for t in TASKS) + f"{'mean':>8}")
    for name, r in rows.items():
        mean = sum(r.values()) / len(r)
        print(f"{name:>15}" + "".join(f"{r[t]:>13.3f}" for t in TASKS) + f"{mean:>8.3f}", flush=True)
    print("\nread: coppice (unified) should learn ALL of A/B/C and RETAIN them; geometric-only should FAIL B "
          "(local_repeat) and C (rep_func) -- no token eq_atom; monolithic should FORGET A after B/C (no "
          "consolidation); joint = ceiling. If so: token-identity-as-degenerate-concept unifies symbolic+"
          "geometric, and Coppice's frozen-core is a working progressive learner over the unified substrate.")


if __name__ == "__main__":
    main()
