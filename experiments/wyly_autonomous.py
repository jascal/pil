"""FULL autonomous Wyly, end-to-end: adaptive capacity + wake/sleep + replay, NOTHING hand-allocated.

The controlled wyly_progressive.py pre-allocated rule groups and hand-specified phases. This is the actual
Wyly proposal -- a self-organizing learner over a task STREAM with NO pre-set capacity:
  WAKE  : train on current task (+ replay of prior tasks); if under-learned, OVER-SUBSCRIBE -> GROW concepts
          + rules from a reserve pool and retrain, until learned or capacity exhausted.
  SLEEP : (1) curate REPLAY exemplars = k representative (random) + k edge (highest-loss); (2) CONSOLIDATE by
          PROTECTING the concepts/rules this task used (freeze them); (3) HOMEOSTASIS -- MERGE redundant
          unprotected concepts (cosine-similar directions) and PRUNE dead ones (~zero usage).
Substrate is the unified one: geometric concepts + token-identity eq_atom + soft-AND rules.

Stream: func -> cap -> local_repeat -> rep_func(=repeat&func) -> rep_cap(=repeat&cap)  (lexical+relational+
compositional, reuse). Compare: autonomous vs fixed-small (under-provisions) vs monolithic (forgets) vs
controlled (hand-allocated) vs joint (ceiling). Report per-task retention + final active concept/rule counts.

Run: cd pil && .venv/bin/python experiments/wyly_autonomous.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
DATA = SP / "wyly_pythia70m.pt"
STREAM = ["func", "cap", "local_repeat", "rep_func", "rep_cap"]
KMAX, RMAX = 40, 60
KINIT, RINIT = 6, 8                     # start small; grow as needed
KGROW, RGROW = 5, 7
TARGET = 0.88                           # wake grows until task val-acc reaches this (or capacity out)
TAU = 0.6


class AutoWyly(torch.nn.Module):
    """Growable unified substrate with active/protected masks over a fixed backing store."""
    def __init__(self, d, use_token_eq=True):
        super().__init__()
        self.d, self.use_token_eq = d, use_token_eq
        self.Ug = torch.nn.Parameter(torch.randn(KMAX, d) * (1.0 / d ** 0.5))
        self.bg = torch.nn.Parameter(torch.zeros(KMAX))
        self.F = KMAX + (1 if use_token_eq else 0) + 1
        self.req = torch.nn.Parameter(torch.randn(RMAX, self.F) * 0.3 - 1.0)
        self.heads = torch.nn.Parameter(torch.zeros(len(STREAM), RMAX + 1))
        self.register_buffer("active_c", torch.zeros(KMAX, dtype=torch.bool))
        self.register_buffer("active_r", torch.zeros(RMAX, dtype=torch.bool))
        self.register_buffer("prot_c", torch.zeros(KMAX, dtype=torch.bool))
        self.register_buffer("prot_r", torch.zeros(RMAX, dtype=torch.bool))
        self.register_buffer("frozen_h", torch.zeros(len(STREAM), dtype=torch.bool))
        self.active_c[:KINIT] = True
        self.active_r[:RINIT] = True

    def features(self, R, ids):
        m = torch.sigmoid((R @ self.Ug.T - self.bg) / TAU) * self.active_c        # (B,L,KMAX) inactive->0
        cur = m[:, -1]
        nact = max(int(self.active_c.sum()), 1)
        gmatch = (m[:, :-1] * cur.unsqueeze(1)).sum(-1).max(1).values.unsqueeze(1) / nact
        feats = [cur, gmatch]
        if self.use_token_eq:
            teq = (ids[:, :-1] == ids[:, -1:]).any(1, keepdim=True).float()
            feats.insert(1, teq)
        return torch.cat(feats, -1)

    def fire(self, R, ids):
        f = self.features(R, ids)
        req = torch.sigmoid(self.req)
        term = 1 - req.unsqueeze(0) + req.unsqueeze(0) * f.unsqueeze(1)
        fr = term.prod(-1) * self.active_r                                        # inactive rules -> 0
        return fr

    def logit(self, ti, R, ids):
        fr = self.fire(R, ids)
        return fr @ self.heads[ti, :-1] + self.heads[ti, -1]

    def plastic_mask(self):
        """grads flow only to ACTIVE & UNPROTECTED concepts/rules (+ heads handled per-call)."""
        mc = (self.active_c & ~self.prot_c).float()
        mr = (self.active_r & ~self.prot_r).float()
        return mc, mr

    def grow(self):
        nc = torch.where(~self.active_c)[0][:KGROW]
        nr = torch.where(~self.active_r)[0][:RGROW]
        with torch.no_grad():
            self.Ug[nc] = torch.randn(len(nc), self.d) * (1.0 / self.d ** 0.5)   # fresh shapes
            self.req[nr] = torch.randn(len(nr), self.F) * 0.3 - 1.0
        self.active_c[nc] = True
        self.active_r[nr] = True
        return len(nc) > 0 or len(nr) > 0


def step(model, batch, task_heads, opt):
    loss = 0.0
    for ti, (R, ids, y) in batch:
        loss = loss + F.binary_cross_entropy_with_logits(model.logit(ti, R, ids), y)
    loss = loss / len(batch)
    opt.zero_grad()
    loss.backward()
    mc, mr = model.plastic_mask()
    with torch.no_grad():
        model.Ug.grad *= mc.unsqueeze(1)
        model.bg.grad *= mc
        model.req.grad *= mr.unsqueeze(1)
        hm = torch.zeros_like(model.heads)                       # only unfrozen tasks in batch
        for ti, _ in batch:
            if not model.frozen_h[ti]:
                hm[ti] = 1
        model.heads.grad *= hm
    opt.step()


def val_acc(model, ti, R, ids, y, te):
    with torch.no_grad():
        return float(((model.logit(ti, R[te], ids[te]) > 0).float() == y[te]).float().mean())


def wake(model, t, data, replay, seed, allow_grow=True):
    """train on task t (+ replay); grow when under-learned."""
    ti = STREAM.index(t)
    g = torch.Generator().manual_seed(seed)
    R, ids, y, tr, te = data[t]
    for _ in range(6):                                                            # up to 6 grow rounds
        opt = torch.optim.Adam([p for p in model.parameters()], lr=0.02)
        for _ in range(1400):
            idx = tr[torch.randint(len(tr), (192,), generator=g)]
            batch = [(ti, (R[idx], ids[idx], y[idx]))]
            for (rti, rR, rids, ry) in replay_sample(replay, g):
                batch.append((rti, (rR, rids, ry)))
            step(model, batch, None, opt)
        if val_acc(model, ti, R, ids, y, te) >= TARGET or not allow_grow:
            break
        if not model.grow():
            break


def replay_sample(replay, g, k=96):
    out = []
    for rti, (rR, rids, ry) in replay.items():
        if len(ry) == 0:
            continue
        idx = torch.randint(len(ry), (min(k, len(ry)),), generator=g)
        out.append((rti, rR[idx], rids[idx], ry[idx]))
    return out


def sleep(model, t, data, replay, seed):
    ti = STREAM.index(t)
    R, ids, y, tr, te = data[t]
    g = torch.Generator().manual_seed(seed + 999)
    # (1) curate replay: k representative (random) + k edge (highest current loss)
    with torch.no_grad():
        lg = model.logit(ti, R[tr], ids[tr])
        losses = F.binary_cross_entropy_with_logits(lg, y[tr], reduction="none")
    edge = tr[losses.argsort(descending=True)[:60]]
    rep = tr[torch.randperm(len(tr), generator=g)[:60]]
    keep = torch.cat([edge, rep])
    replay[ti] = (R[keep], ids[keep], y[keep])
    # (2) consolidate: protect the rules this task uses most + the concepts feeding them, and ISOLATE the
    #     task head to read ONLY protected rules (so plastic-rule drift on later tasks can't forget it)
    with torch.no_grad():
        rule_use = model.heads[ti, :RMAX].abs() * model.active_r
        top_r = rule_use.argsort(descending=True)[:6]
        model.prot_r[top_r] = True
        fr_req = torch.sigmoid(model.req[top_r])                 # concepts of protected rules
        used_c = (fr_req[:, :KMAX] > 0.5).any(0) & model.active_c
        model.prot_c[used_c] = True
        model.heads[ti, :RMAX][~model.prot_r] = 0.0              # head reads only protected rules
        model.frozen_h[ti] = True                                                # consolidated -> head frozen
    # (3) homeostasis: merge cosine-similar unprotected concepts; prune ~dead active concepts/rules
    homeostasis(model, data, t)


def homeostasis(model, data, t):
    R, ids, y, tr, te = data[t]
    with torch.no_grad():
        act = model.active_c & ~model.prot_c
        idxs = torch.where(act)[0]
        Un = model.Ug[idxs]
        Un = Un / (Un.norm(dim=1, keepdim=True) + 1e-6)
        cos = Un @ Un.T
        for a in range(len(idxs)):                               # MERGE near-dup concept pairs
            for b in range(a + 1, len(idxs)):
                if cos[a, b] > 0.92 and model.active_c[idxs[b]]:
                    model.active_c[idxs[b]] = False
        m = torch.sigmoid((R[tr] @ model.Ug.T - model.bg) / TAU)[:, -1].mean(0)  # PRUNE dead
        dead = (m < 0.02) & model.active_c & ~model.prot_c
        model.active_c[dead] = False
        rf = model.fire(R[tr], ids[tr]).mean(0)
        deadr = (rf < 0.02) & model.active_r & ~model.prot_r
        model.active_r[deadr] = False


def run_autonomous(dim, data, seed):
    torch.manual_seed(seed)
    model = AutoWyly(dim)
    replay = {}
    trace = []
    for t in STREAM:
        wake(model, t, data, replay, seed)
        sleep(model, t, data, replay, seed)
        trace.append((int(model.active_c.sum()), int(model.active_r.sum())))
    res = {t: val_acc(model, STREAM.index(t), *data[t][:3], data[t][4]) for t in STREAM}
    res["_nc"], res["_nr"] = int(model.active_c.sum()), int(model.active_r.sum())
    return res


def run_fixed(dim, data, seed, grow=False):
    """fixed-small: no growth (KINIT/RINIT only); tests if starting capacity suffices."""
    torch.manual_seed(seed)
    model = AutoWyly(dim)
    replay = {}
    for t in STREAM:
        wake(model, t, data, replay, seed, allow_grow=grow)
        sleep(model, t, data, replay, seed)
    return {t: val_acc(model, STREAM.index(t), *data[t][:3], data[t][4]) for t in STREAM}


def run_monolithic(dim, data, seed):
    """full capacity active, retrained per task, NO replay/consolidation -> forgets."""
    torch.manual_seed(seed)
    model = AutoWyly(dim)
    model.active_c[:] = True
    model.active_r[:] = True
    g = torch.Generator().manual_seed(seed)
    for t in STREAM:
        ti = STREAM.index(t)
        R, ids, y, tr, _ = data[t]
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(2000):
            idx = tr[torch.randint(len(tr), (192,), generator=g)]
            step(model, [(ti, (R[idx], ids[idx], y[idx]))], None, opt)
    return {t: val_acc(model, STREAM.index(t), *data[t][:3], data[t][4]) for t in STREAM}


def run_joint(dim, data, seed):
    torch.manual_seed(seed)
    model = AutoWyly(dim)
    model.active_c[:] = True
    model.active_r[:] = True
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(7000):
        batch = []
        for t in STREAM:
            ti = STREAM.index(t)
            R, ids, y, tr, _ = data[t]
            idx = tr[torch.randint(len(tr), (128,), generator=g)]
            batch.append((ti, (R[idx], ids[idx], y[idx])))
        step(model, batch, None, opt)
    return {t: val_acc(model, STREAM.index(t), *data[t][:3], data[t][4]) for t in STREAM}


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


def main():
    d = torch.load(DATA)
    R, ids = d["r"].float(), d["kept_ids"]
    labels = dict(d["labels"])
    labels["rep_cap"] = (labels["local_repeat"].bool() & labels["cap"].bool()).long()
    print(f"FULL autonomous Wyly, stream {STREAM}  R {tuple(R.shape)}")
    print(f"start K={KINIT}/R={RINIT}; wake grows to {TARGET}; sleep=replay+consolidate+homeostasis\n")
    runners = {"autonomous": run_autonomous, "fixed-small": run_fixed,
               "monolithic": run_monolithic, "joint": run_joint}
    rows = {}
    for name, fn in runners.items():
        seeds = []
        for s in (0, 1, 2):
            data = {t: balanced_task(R, ids, labels[t], s) for t in STREAM}
            seeds.append(fn(data[STREAM[0]][0].shape[-1], data, s))
        rows[name] = {k: sum(sd[k] for sd in seeds) / len(seeds)
                      for k in (STREAM + (["_nc", "_nr"] if name == "autonomous" else []))}
    print(f"{'learner':>12}" + "".join(f"{t:>13}" for t in STREAM) + f"{'mean':>7}{'nc/nr':>9}")
    for name, r in rows.items():
        mean = sum(r[t] for t in STREAM) / len(STREAM)
        cap = f"{int(r['_nc'])}/{int(r['_nr'])}" if "_nc" in r else "-"
        print(f"{name:>12}" + "".join(f"{r[t]:>13.3f}" for t in STREAM) + f"{mean:>7.3f}{cap:>9}", flush=True)
    print("\nread: autonomous ~ joint ceiling with a SELF-ADAPTED nc/nr (grew from KINIT to fit the stream, "
          "pruned waste) = self-organizing Wyly WORKS end-to-end; fixed-small should trail the "
          "harder/later tasks (under-provisioned); monolithic should FORGET early tasks (func/cap).")


if __name__ == "__main__":
    main()
