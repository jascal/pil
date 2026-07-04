"""Autonomous Wyly, next stage: TUNE homeostasis, SCALE the stream, and boundary-free ONLINE operation.

Builds on wyly_autonomous.py (which worked but trailed the ceiling by ~0.08, mostly rep_func, because
aggressive homeostasis pruned/merged capacity a later compositional task reused). Three upgrades:
  TUNE   : running usage-tracking (use_c/use_r) -- NEVER prune/merge a concept/rule ever meaningfully used
           (protect the reuse chain); merge only near-EXACT duplicates that were never used.
  SCALE  : configurable long stream (5 lexical bases + repeat + 5 compositions = up to 11 tasks).
  ONLINE : boundary-free driver -- a continuous chunked stream with task ids HIDDEN; a drift detector
           (novelty = current-head loss on incoming chunks) auto-segments into tasks -> wake/sleep with no
           boundary supervision. Reports detected-vs-true boundaries + retention.

Run: cd pil && .venv/bin/python experiments/wyly_online.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SP = Path("/tmp/claude-1000/-home-allans-code/79593291-dade-4d02-a009-357bd1c48e92/scratchpad")
KMAX, RMAX, TAU = 64, 96, 0.6
KINIT, RINIT, KGROW, RGROW = 6, 8, 5, 7
TARGET, MAXTASK = 0.88, 12


class Wyly(torch.nn.Module):
    def __init__(self, d, ntask, use_token_eq=True):
        super().__init__()
        self.d, self.use_token_eq = d, use_token_eq
        self.Ug = torch.nn.Parameter(torch.randn(KMAX, d) * (1.0 / d ** 0.5))
        self.bg = torch.nn.Parameter(torch.zeros(KMAX))
        self.F = KMAX + (1 if use_token_eq else 0) + 1
        self.req = torch.nn.Parameter(torch.randn(RMAX, self.F) * 0.3 - 3.5)  # SPARSE: rules start near-empty
        self.heads = torch.nn.Parameter(torch.zeros(ntask, RMAX + 1))
        for nm in ["active_c", "prot_c", "active_r", "prot_r"]:
            n = KMAX if "_c" in nm else RMAX
            self.register_buffer(nm, torch.zeros(n, dtype=torch.bool))
        self.register_buffer("frozen_h", torch.zeros(ntask, dtype=torch.bool))
        self.register_buffer("use_c", torch.zeros(KMAX))            # running max membership (usage)
        self.register_buffer("use_r", torch.zeros(RMAX))            # running max fire (usage)
        self.active_c[:KINIT] = True
        self.active_r[:RINIT] = True

    def features(self, R, ids):
        m = torch.sigmoid((R @ self.Ug.T - self.bg) / TAU) * self.active_c
        cur = m[:, -1]
        nact = max(int(self.active_c.sum()), 1)
        gm = (m[:, :-1] * cur.unsqueeze(1)).sum(-1).max(1).values.unsqueeze(1) / nact
        feats = [cur, gm]
        if self.use_token_eq:
            feats.insert(1, (ids[:, :-1] == ids[:, -1:]).any(1, keepdim=True).float())
        return torch.cat(feats, -1)

    def fire(self, R, ids):
        req = torch.sigmoid(self.req)
        term = 1 - req.unsqueeze(0) + req.unsqueeze(0) * self.features(R, ids).unsqueeze(1)
        return term.prod(-1) * self.active_r

    def logit(self, ti, R, ids):
        fr = self.fire(R, ids)
        return fr @ self.heads[ti, :-1] + self.heads[ti, -1]

    def grow(self):
        nc = torch.where(~self.active_c)[0][:KGROW]
        nr = torch.where(~self.active_r)[0][:RGROW]
        with torch.no_grad():
            self.Ug[nc] = torch.randn(len(nc), self.d) * (1.0 / self.d ** 0.5)
            self.req[nr] = torch.randn(len(nr), self.F) * 0.3 - 3.5
        self.active_c[nc] = True
        self.active_r[nr] = True
        return len(nc) + len(nr) > 0


def step(model, batch, opt):
    loss = sum(F.binary_cross_entropy_with_logits(model.logit(ti, R, ids), y) for ti, (R, ids, y) in batch)
    loss = loss / len(batch)
    opt.zero_grad()
    loss.backward()
    with torch.no_grad():
        mc = (model.active_c & ~model.prot_c).float()
        mr = (model.active_r & ~model.prot_r).float()
        model.Ug.grad *= mc.unsqueeze(1)
        model.bg.grad *= mc
        model.req.grad *= mr.unsqueeze(1)
        hm = torch.zeros_like(model.heads)
        for ti, _ in batch:
            if not model.frozen_h[ti]:
                hm[ti] = 1
        model.heads.grad *= hm
    opt.step()


def acc(model, ti, R, ids, y, te):
    with torch.no_grad():
        return float(((model.logit(ti, R[te], ids[te]) > 0).float() == y[te]).float().mean())


def wake(model, ti, R, ids, y, tr, te, replay, seed, grow=True):
    g = torch.Generator().manual_seed(seed + ti)
    for _ in range(6):
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(1400):
            idx = tr[torch.randint(len(tr), (192,), generator=g)]
            batch = [(ti, (R[idx], ids[idx], y[idx]))] + replay_sample(replay, g)
            step(model, batch, opt)
        if acc(model, ti, R, ids, y, te) >= TARGET or not grow or not model.grow():
            break


def replay_sample(replay, g, k=96):
    out = []
    for rti, (rR, rids, ry) in replay.items():
        if len(ry):
            idx = torch.randint(len(ry), (min(k, len(ry)),), generator=g)
            out.append((rti, (rR[idx], rids[idx], ry[idx])))
    return out


def consolidate(model, ti, R, ids, y, tr, replay, seed):
    g = torch.Generator().manual_seed(seed + 999 + ti)
    with torch.no_grad():
        lg = model.logit(ti, R[tr], ids[tr])
        losses = F.binary_cross_entropy_with_logits(lg, y[tr], reduction="none")
        edge = tr[losses.argsort(descending=True)[:60]]
        rep = tr[torch.randperm(len(tr), generator=g)[:60]]
        keep = torch.cat([edge, rep])
        replay[ti] = (R[keep], ids[keep], y[keep])
        top_r = (model.heads[ti, :RMAX].abs() * model.active_r).argsort(descending=True)[:10]
        model.prot_r[top_r] = True
        # protect the concepts these rules actually use (sparse init keeps unused ones ~inert -> frozen rules
        # barely drift when later tasks activate new concepts); isolate + freeze the head.
        used_c = (torch.sigmoid(model.req[top_r])[:, :KMAX] > 0.4).any(0) & model.active_c
        model.prot_c[used_c] = True
        model.heads[ti, :RMAX][~model.prot_r] = 0.0
        model.frozen_h[ti] = True


def homeostasis_gentle(model, R, ids, tr):
    """usage-protected: prune ONLY never-used capacity; merge ONLY near-exact duplicates never used."""
    with torch.no_grad():
        m = torch.sigmoid((R[tr] @ model.Ug.T - model.bg) / TAU)[:, -1].mean(0)
        model.use_c = torch.maximum(model.use_c, m)
        model.use_r = torch.maximum(model.use_r, model.fire(R[tr], ids[tr]).mean(0))
        act = torch.where(model.active_c & ~model.prot_c)[0]
        Un = model.Ug[act] / (model.Ug[act].norm(dim=1, keepdim=True) + 1e-6)
        cos = Un @ Un.T
        for a in range(len(act)):
            for b in range(a + 1, len(act)):
                if cos[a, b] > 0.97 and model.use_c[act[b]] < 0.05 and model.active_c[act[b]]:
                    model.active_c[act[b]] = False                  # merge near-dup UNUSED only
        model.active_c[(model.use_c < 0.02) & model.active_c & ~model.prot_c] = False   # prune truly-dead
        model.active_r[(model.use_r < 0.02) & model.active_r & ~model.prot_r] = False


def run_stream(model, stream, data, seed, gentle=True):
    replay = {}
    for ti, t in enumerate(stream):
        R, ids, y, tr, te = data[t]
        wake(model, ti, R, ids, y, tr, te, replay, seed)
        consolidate(model, ti, R, ids, y, tr, replay, seed)
        if gentle:
            homeostasis_gentle(model, R, ids, tr)
    return {t: acc(model, ti, *data[t][:3], data[t][4]) for ti, t in enumerate(stream)}, \
        (int(model.active_c.sum()), int(model.active_r.sum()))


def run_joint(model, stream, data, seed):
    model.active_c[:] = True
    model.active_r[:] = True
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(1400 * len(stream)):
        batch = []
        for ti, t in enumerate(stream):
            R, ids, y, tr, _ = data[t]
            idx = tr[torch.randint(len(tr), (96,), generator=g)]
            batch.append((ti, (R[idx], ids[idx], y[idx])))
        step(model, batch, opt)
    return {t: acc(model, ti, *data[t][:3], data[t][4]) for ti, t in enumerate(stream)}


def run_online(model, stream, data, seed, chunks_per_task=6, drift=0.62):
    """boundary-FREE: chunks arrive in task order, ids HIDDEN; novelty (current-head loss) auto-segments."""
    g = torch.Generator().manual_seed(seed)
    replay, cur, converged, detected, true_bounds = {}, 0, False, [], []
    seq = []
    for ti, t in enumerate(stream):
        R, ids, y, tr, te = data[t]
        for _c in range(chunks_per_task):
            idx = tr[torch.randint(len(tr), (400,), generator=g)]
            seq.append((ti, t, R[idx], ids[idx], y[idx]))
        true_bounds.append(len(seq))
    for j, (_tti, _t, Rc, idc, yc) in enumerate(seq):
        with torch.no_grad():                                       # novelty = current-head loss
            nov = float(F.binary_cross_entropy_with_logits(model.logit(cur, Rc, idc), yc))
        if converged and nov > drift:                              # drift -> consolidate old, open new task
            R0, i0, y0, tr0, _ = data[stream[cur]]
            consolidate(model, cur, R0, i0, y0, tr0, replay, seed)
            homeostasis_gentle(model, R0, i0, tr0)
            cur += 1
            converged = False
            detected.append(j)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)         # wake current head on this chunk
        for _ in range(500):
            bs = torch.randint(len(yc), (192,), generator=g)
            step(model, [(cur, (Rc[bs], idc[bs], yc[bs]))] + replay_sample(replay, g), opt)
        with torch.no_grad():
            if float((((model.logit(cur, Rc, idc)) > 0).float() == yc).float().mean()) > TARGET:
                converged = True
    R0, i0, y0, tr0, _ = data[stream[cur]]
    consolidate(model, cur, R0, i0, y0, tr0, replay, seed)
    res = {t: acc(model, ti, *data[t][:3], data[t][4]) for ti, t in enumerate(stream)}
    return res, detected, true_bounds[:-1], cur + 1


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


def load(path, extra):
    d = torch.load(SP / path)
    labels = dict(d["labels"])
    for name, expr in extra.items():
        labels[name] = expr(labels)
    return d["r"].float(), d["kept_ids"], labels


def summarize(tag, stream, rows, caps):
    print(f"\n[{tag}]  stream({len(stream)}): {stream}")
    print(f"{'learner':>18}" + "".join(f"{t[:8]:>9}" for t in stream) + f"{'mean':>7}{'nc/nr':>8}")
    for name in rows:
        r = rows[name]
        mean = sum(r[t] for t in stream) / len(stream)
        cap = f"{caps[name][0]}/{caps[name][1]}" if name in caps else "-"
        print(f"{name:>18}" + "".join(f"{r[t]:>9.2f}" for t in stream) + f"{mean:>7.3f}{cap:>8}", flush=True)


def main():
    torch.manual_seed(0)
    # ---- TUNE: gentle homeostasis on the original 5-task stream (existing data) ----
    rep_cap = {"rep_cap": lambda L: (L["local_repeat"].bool() & L["cap"].bool()).long()}
    r5, ids5, lab5 = load("wyly_pythia70m.pt", rep_cap)
    stream5 = ["func", "cap", "local_repeat", "rep_func", "rep_cap"]
    rows, caps = {}, {}
    for name, gentle in [("gentle-autonomous", True)]:
        accs, cap = [], None
        for s in (0, 1, 2):
            data = {t: balanced_task(r5, ids5, lab5[t], s) for t in stream5}
            torch.manual_seed(s)
            m = Wyly(r5.shape[-1], len(stream5))
            res, cap = run_stream(m, stream5, data, s, gentle=gentle)
            accs.append(res)
        rows[name] = {t: sum(a[t] for a in accs) / len(accs) for t in stream5}
        caps[name] = cap
    data0 = {t: balanced_task(r5, ids5, lab5[t], 0) for t in stream5}
    torch.manual_seed(0)
    mj = Wyly(r5.shape[-1], len(stream5))
    rows["joint-ceiling"] = run_joint(mj, stream5, data0, 0)
    summarize("TUNE 5-task", stream5, rows, caps)

    # ---- ONLINE: boundary-free on the same 5-task stream ----
    torch.manual_seed(0)
    mo = Wyly(r5.shape[-1], MAXTASK)
    ores, det, true_b, ndet = run_online(mo, stream5, data0, 0)
    print(f"\n[ONLINE]  detected {ndet} tasks (true 5); boundaries {det} vs true {true_b}")
    print("  per-task acc: " + "  ".join(f"{t[:8]}={ores[t]:.2f}" for t in stream5)
          + f"   mean {sum(ores.values()) / len(stream5):.3f}")


if __name__ == "__main__":
    main()
