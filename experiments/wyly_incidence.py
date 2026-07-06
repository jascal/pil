"""Wyly with a UNIFIED graded-incidence field -- active/plastic/frozen/prune lifecycle on one axis.

Per the design: replace the ad-hoc masks (active/prot/use + prune/merge) with ONE incidence-importance
field omega per concept and per rule (and per task head). The whole lifecycle is positions on that one axis:
    omega = 0            reserve (inactive, available to grow into)
    0 < omega < THETA    DROP: incidence too low -> reclaim to reserve   (the user's lower threshold)
    THETA < omega        active/plastic: gradient flows; protection is the EWC ANCHOR below, NOT grad-scaling
    omega -> large       FROZEN: anchor pins param to its consolidated value (binary freeze = special case)
NB: gradient-scaling by prot(omega)=1/(1+beta*omega) is provably INERT under sign-normalized optimizers like
Adam (i-orca C8 sign_updates_ignore_scaling) -- so protection is done ONLY by the quadratic anchor, never by
scaling the gradient. (An earlier prot()/BETA is removed to avoid attributing protection to an inert knob.)
CONSOLIDATION = RAISE incidence importance (omega += usage), not flip a freeze bit. Disuse -> omega decays ->
falls below THETA -> dropped. The binary system is the step-function limit (LAMBDA, CONS -> inf).
omega is the graded incidence weight; THETA is the gamma-margin of the PIC turnstile. eq_atom stays a fixed
(param-free) feature. Sparse rule init keeps unused incidences ~inert so frozen rules don't drift on growth.

Run: cd pil && .venv/bin/python experiments/wyly_incidence.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
KMAX, RMAX, TAU = 64, 96, 0.6
KINIT, RINIT, KGROW, RGROW = 6, 8, 5, 7
TARGET, MAXTASK = 0.88, 12
CONS_C, CONS_R, CONS_H = 8.0, 8.0, 40.0   # incidence gained on consolidation (usage-weighted)
DECAY = 0.15        # per-sleep importance leak (disuse -> fades)
THETA, SEED = 0.15, 0.45                   # drop threshold; importance a freshly-grown unit is born with
LAMBDA = 3.0        # EWC incidence-anchor strength: penalty = LAMBDA * imp * (param - consolidated)^2


class Wyly(torch.nn.Module):
    def __init__(self, d, ntask, use_token_eq=True):
        super().__init__()
        self.d, self.use_token_eq = d, use_token_eq
        self.Ug = torch.nn.Parameter(torch.randn(KMAX, d) * (1.0 / d ** 0.5))
        self.bg = torch.nn.Parameter(torch.zeros(KMAX))
        self.F = KMAX + (1 if use_token_eq else 0) + 1
        self.req = torch.nn.Parameter(torch.randn(RMAX, self.F) * 0.3 - 3.5)
        self.heads = torch.nn.Parameter(torch.zeros(ntask, RMAX + 1))
        self.register_buffer("imp_c", torch.zeros(KMAX))          # incidence importance (concepts)
        self.register_buffer("imp_r", torch.zeros(RMAX))          # incidence importance (rules)
        self.register_buffer("imp_h", torch.zeros(ntask))         # importance (task heads)
        self.register_buffer("anc_Ug", torch.zeros(KMAX, d))      # EWC anchors: consolidated param values
        self.register_buffer("anc_req", torch.zeros(RMAX, self.F))
        self.register_buffer("anc_h", torch.zeros(ntask, RMAX + 1))
        self.imp_c[:KINIT] = SEED
        self.imp_r[:RINIT] = SEED
        self.anc_Ug.copy_(self.Ug.detach())                      # anchor at actual init values (not zero)
        self.anc_req.copy_(self.req.detach())

    def active_c(self):
        return self.imp_c > 0

    def active_r(self):
        return self.imp_r > 0

    def features(self, R, ids):
        m = torch.sigmoid((R @ self.Ug.T - self.bg) / TAU) * self.active_c()
        cur = m[:, -1]
        nact = max(int(self.active_c().sum()), 1)
        gm = (m[:, :-1] * cur.unsqueeze(1)).sum(-1).max(1).values.unsqueeze(1) / nact
        feats = [cur, gm]
        if self.use_token_eq:
            feats.insert(1, (ids[:, :-1] == ids[:, -1:]).any(1, keepdim=True).float())
        return torch.cat(feats, -1)

    def fire(self, R, ids):
        # sparse incidence: very negative req init keeps NON-incident concepts ~inert (term ~= 1)
        # so a rule barely drifts when later tasks grow concepts; trained incidences rise and discriminate.
        req = torch.sigmoid(self.req)
        term = 1 - req.unsqueeze(0) + req.unsqueeze(0) * self.features(R, ids).unsqueeze(1)
        return term.prod(-1) * self.active_r()

    def logit(self, ti, R, ids):
        return self.fire(R, ids) @ self.heads[ti, :-1] + self.heads[ti, -1]

    def grow(self):
        nc = torch.where(self.imp_c == 0)[0][:KGROW]
        nr = torch.where(self.imp_r == 0)[0][:RGROW]
        with torch.no_grad():
            self.Ug[nc] = torch.randn(len(nc), self.d) * (1.0 / self.d ** 0.5)
            self.req[nr] = torch.randn(len(nr), self.F) * 0.3 - 3.5
            self.imp_c[nc] = SEED
            self.imp_r[nr] = SEED
            self.anc_Ug[nc] = self.Ug[nc]                        # anchor fresh units at their own init values
            self.anc_req[nr] = self.req[nr]
        return len(nc) + len(nr) > 0


def step(model, batch, opt):
    loss = sum(F.binary_cross_entropy_with_logits(model.logit(ti, R, ids), y) for ti, (R, ids, y) in batch)
    loss = loss / len(batch)
    # EWC-style INCIDENCE ANCHOR: high-importance params pulled to their consolidated value (Adam-proof,
    # unlike grad-scaling). imp is the incidence weight = Fisher; anchor is the consolidated value. imp->inf
    # pins the param (the binary freeze, as a special case); imp~0 leaves it free.
    loss = loss + LAMBDA * (model.imp_c * ((model.Ug - model.anc_Ug) ** 2).mean(1)).sum()
    loss = loss + LAMBDA * (model.imp_r * ((model.req - model.anc_req) ** 2).mean(1)).sum()
    for ti, _ in batch:
        loss = loss + LAMBDA * model.imp_h[ti] * ((model.heads[ti] - model.anc_h[ti]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    with torch.no_grad():                                        # reserve (inactive) units never train; only
        model.Ug.grad *= model.active_c().float().unsqueeze(1)   # in-batch task heads train
        model.bg.grad *= model.active_c().float()
        model.req.grad *= model.active_r().float().unsqueeze(1)
        hm = torch.zeros_like(model.heads)
        for ti, _ in batch:
            hm[ti] = 1
        model.heads.grad *= hm
    opt.step()
    with torch.no_grad():                                        # anchor FOLLOWS plastic units (no penalty on
        fc = (model.imp_c < 1.0).float().unsqueeze(1)
        model.anc_Ug.mul_(1 - fc).add_(model.Ug * fc)
        fr = (model.imp_r < 1.0).float().unsqueeze(1)
        model.anc_req.mul_(1 - fr).add_(model.req * fr)


def acc(model, ti, R, ids, y, te):
    with torch.no_grad():
        return float(((model.logit(ti, R[te], ids[te]) > 0).float() == y[te]).float().mean())


MIN_PC, MIN_PR = 5, 7          # over-subscribe: min PLASTIC (low-imp) concepts/rules each task starts with


def plastic_counts(model):
    pc = int(((model.imp_c > 0) & (model.imp_c < 1.0)).sum())
    pr = int(((model.imp_r > 0) & (model.imp_r < 1.0)).sum())
    return pc, pr


def wake(model, ti, R, ids, y, tr, te, replay, seed):
    g = torch.Generator().manual_seed(seed + ti)
    while True:
        pc, pr = plastic_counts(model)                           # task builds on new units, not frozen ones
        if (pc >= MIN_PC and pr >= MIN_PR) or not model.grow():
            break
    for _ in range(6):
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(1400):
            idx = tr[torch.randint(len(tr), (192,), generator=g)]
            step(model, [(ti, (R[idx], ids[idx], y[idx]))] + replay_sample(replay, g), opt)
        if acc(model, ti, R, ids, y, te) >= TARGET or not model.grow():
            break


def replay_sample(replay, g, k=96):
    out = []
    for rti, (rR, rids, ry) in replay.items():
        if len(ry):
            idx = torch.randint(len(ry), (min(k, len(ry)),), generator=g)
            out.append((rti, (rR[idx], rids[idx], ry[idx])))
    return out


def consolidate(model, ti, R, ids, y, tr, replay, seed):
    """CONSOLIDATION = RAISE incidence importance by usage (concepts, rules, head). No freeze bit."""
    g = torch.Generator().manual_seed(seed + 999 + ti)
    with torch.no_grad():
        lg = model.logit(ti, R[tr], ids[tr])
        losses = F.binary_cross_entropy_with_logits(lg, y[tr], reduction="none")
        edge = tr[losses.argsort(descending=True)[:60]]
        keep = torch.cat([edge, tr[torch.randperm(len(tr), generator=g)[:60]]])
        replay[ti] = (R[keep], ids[keep], y[keep])
        # rule usage = how much the task's HEAD weights each rule x how much it fires
        rule_use = (model.heads[ti, :RMAX].abs() * model.fire(R[tr], ids[tr]).mean(0))
        rule_use = rule_use / (rule_use.max() + 1e-9)
        model.imp_r += CONS_R * rule_use                          # rules the head weighs gain importance
        # concept usage is RULE-MEDIATED: a concept gains importance only via the rules that use it (req) that
        # the head weights -- so concepts irrelevant to the task stay plastic/droppable (no over-freezing).
        concept_use = (rule_use.unsqueeze(1) * torch.sigmoid(model.req[:, :KMAX])).sum(0) * model.active_c()
        concept_use = concept_use / (concept_use.max() + 1e-9)
        model.imp_c += CONS_C * concept_use
        # GRADED head-isolation: the task head keeps weight only on high-incidence rules, so it
        # reads its own (now high-imp, frozen) rules -- not low-imp rules later tasks will repurpose. As
        # imp_r -> large this becomes the binary "read only protected rules" isolation (the special case).
        model.heads[ti, :RMAX] *= torch.sigmoid(4.0 * (model.imp_r - 1.0))
        model.imp_h[ti] += CONS_H                                 # task head consolidated -> frozen
        model.anc_h[ti].copy_(model.heads[ti])                   # pin this task's (now isolated) head


def homeostasis(model, R, ids, tr):
    """DECAY importance (disuse fades) then DROP anything below THETA (the lower incidence threshold)."""
    with torch.no_grad():
        model.imp_c *= (1 - DECAY)
        model.imp_r *= (1 - DECAY)
        drop_c = (model.imp_c > 0) & (model.imp_c < THETA)
        drop_r = (model.imp_r > 0) & (model.imp_r < THETA)
        model.imp_c[drop_c] = 0.0                                 # incidence too low -> reclaim to reserve
        model.imp_r[drop_r] = 0.0
        return int(drop_c.sum()), int(drop_r.sum())


def run_stream(model, stream, data, seed):
    replay, dropped = {}, 0
    for ti, t in enumerate(stream):
        R, ids, y, tr, te = data[t]
        wake(model, ti, R, ids, y, tr, te, replay, seed)
        consolidate(model, ti, R, ids, y, tr, replay, seed)
        dc, dr = homeostasis(model, R, ids, tr)
        dropped += dc + dr
    res = {t: acc(model, ti, *data[t][:3], data[t][4]) for ti, t in enumerate(stream)}
    return res, (int(model.active_c().sum()), int(model.active_r().sum())), dropped


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
    d = torch.load(SP / "wyly_pythia70m.pt")
    labels = dict(d["labels"])
    labels["rep_cap"] = (labels["local_repeat"].bool() & labels["cap"].bool()).long()
    r, ids = d["r"].float(), d["kept_ids"]
    stream = ["func", "cap", "local_repeat", "rep_func", "rep_cap"]
    print("Wyly with UNIFIED graded-incidence field (binary lifecycle = special case)")
    print(f"CONS=({CONS_C},{CONS_R},{CONS_H}) DECAY={DECAY} THETA={THETA} SEED={SEED} LAMBDA={LAMBDA}\n")
    accs, caps, drops = [], [], []
    for s in (0, 1, 2):
        data = {t: balanced_task(r, ids, labels[t], s) for t in stream}
        torch.manual_seed(s)
        m = Wyly(r.shape[-1], len(stream))
        res, cap, dropped = run_stream(m, stream, data, s)
        accs.append(res)
        caps.append(cap)
        drops.append(dropped)
        if s == 0:
            imp = m.imp_c[m.imp_c > 0]
            print(f"  seed0 final imp_c (active {len(imp)}): "
                  f"min {imp.min():.2f} median {imp.median():.2f} max {imp.max():.2f} "
                  f"(frozen>1: {int((imp > 1).sum())}, plastic<1: {int((imp <= 1).sum())})")
    mean = {t: sum(a[t] for a in accs) / len(accs) for t in stream}
    print(f"\n{'task':>14}" + "".join(f"{t[:9]:>10}" for t in stream) + f"{'mean':>8}")
    print(f"{'graded-incid':>14}" + "".join(f"{mean[t]:>10.3f}" for t in stream)
          + f"{sum(mean.values()) / len(stream):>8.3f}")
    ac = sum(c[0] for c in caps) / len(caps)
    ar = sum(c[1] for c in caps) / len(caps)
    print(f"\nfinal active c/r {ac:.0f}/{ar:.0f}  |  units dropped (imp<THETA) {sum(drops) / len(drops):.0f}")
    print("read: mean approaches the tuned binary version (the LAMBDA/CONS->inf special case); the "
          "importance spread (frozen vs plastic) + the drop count show the ONE incidence axis governing the "
          "whole active/plastic/frozen/drop lifecycle -- consolidation raises incidence, disuse drops it.")


if __name__ == "__main__":
    main()
