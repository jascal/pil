"""Close the Coppice~PackNet tie: race them on the UNIFIED substrate + compositional chain.

The earlier tie (0.939) was on lexical tasks with a geometric-only substrate. The end-to-end result showed
the unified substrate (token eq_atom) is more capable than geometric-only -- but that is a SUBSTRATE win, not
a Coppice-vs-PackNet win. To honestly close the tie we give BOTH learners the SAME unified substrate
(geometric concepts + token eq_atom + rules; concepts trained+frozen in phase A for all), isolating the
CONSOLIDATION algorithm: Coppice's pre-allocated frozen-core vs PackNet's train-all -> prune -> release.
Chain: A(func,cap) -> B(local_repeat) -> C(rep_func = repeat AND func). Does Coppice BEAT PackNet (frozen-core
reuse preserves a concept PackNet prunes) or TIE (parity confirmed, substrate advantage shared)?

Run: cd pil && .venv/bin/python experiments/normative_unified.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import coppice_progressive as CP  # noqa: E402

TASKS, PHASES, NR = CP.TASKS, CP.PHASES, CP.NR
KEEP = {"A": 12, "B": 9}                 # rules PackNet keeps per phase (matches Coppice group sizes)


def zero_mask(model):
    return {n: torch.zeros_like(p) for n, p in model.named_parameters()}


def phase_mask(model, rule_idxs, tasks, train_concepts=False):
    m = zero_mask(model)
    m["req"][list(rule_idxs)] = 1
    for t in tasks:
        m["heads"][TASKS.index(t)] = 1
    if train_concepts:
        m["Ug"][:] = 1
        m["bg"][:] = 1
    return m


def importance(model, tasks):
    ti = [TASKS.index(t) for t in tasks]
    return model.heads.detach()[ti][:, :NR].abs().sum(0)


def run_packnet(dim, data, seed):
    """train-all-on-A -> prune top-KEEP by importance -> fine-tune -> freeze -> release for B -> C."""
    torch.manual_seed(seed)
    model = CP.Coppice(dim, use_token_eq=True)
    free = set(range(NR))
    # phase A: train concepts + ALL rules + A-heads
    mA = phase_mask(model, free, PHASES["A"], train_concepts=True)
    CP.fit(model, PHASES["A"], data, 3000, mask=mA, seed=seed)
    for ph in ["A", "B"]:
        keptn = KEEP[ph]
        imp = importance(model, PHASES[ph])
        avail = torch.tensor(sorted(free))
        keep = avail[imp[avail].argsort(descending=True)[:keptn]].tolist()
        released = [i for i in free if i not in keep]
        with torch.no_grad():                                    # re-init released rules for the next task
            model.req[released] = torch.randn(len(released), model.F) * 0.3 - 1.0
        # fine-tune this phase's heads on the KEPT rules (concepts + kept rules frozen structurally)
        CP.fit(model, PHASES[ph], data, 1200, mask=phase_mask(model, keep, PHASES[ph]), seed=seed)
        free = set(released)
        nxt = "B" if ph == "A" else "C"
        CP.fit(model, PHASES[nxt], data, 2500, mask=phase_mask(model, free, PHASES[nxt]), seed=seed)
    return {t: CP.acc(model, TASKS.index(t), *data[t][:3], data[t][4]) for t in TASKS}


def run_ewc(dim, data, seed, lam=40.0):
    """all rules plastic; anchor to post-phase weights with diagonal-Fisher penalty."""
    torch.manual_seed(seed)
    model = CP.Coppice(dim, use_token_eq=True)
    anchors = []
    for ph in ["A", "B", "C"]:
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        g = torch.Generator().manual_seed(seed)
        for _ in range(2800):
            loss = 0.0
            for t in PHASES[ph]:
                R, ids, y, tr, _ = data[t]
                idx = tr[torch.randint(len(tr), (256,), generator=g)]
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    model.logit(TASKS.index(t), R[idx], ids[idx]), y[idx])
            for th0, fish in anchors:
                loss = loss + lam * (fish * (model.req - th0) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():                                    # crude diagonal Fisher on req from grad^2
            gr = model.req.grad
            fish = (gr ** 2).clone() if gr is not None else torch.zeros_like(model.req)
            anchors.append((model.req.detach().clone(), fish / (fish.mean() + 1e-9)))
    return {t: CP.acc(model, TASKS.index(t), *data[t][:3], data[t][4]) for t in TASKS}


def main():
    d = torch.load(CP.DATA)
    R, ids = d["r"].float(), d["kept_ids"]
    print(f"CLOSE THE TIE -- unified substrate, compositional chain A->B->C  R {tuple(R.shape)}")
    print("both Coppice & PackNet get geometric concepts + token eq_atom; isolate the consolidation algo\n")
    rows = {}
    for name in ["coppice", "packnet", "ewc", "joint"]:
        seeds = []
        for s in (0, 1, 2, 3, 4):
            data = {t: CP.balanced_task(R, ids, d["labels"][t], s) for t in TASKS}
            dim = data[TASKS[0]][0].shape[-1]
            if name == "coppice":
                seeds.append(CP.evaluate("coppice", data, s))
            elif name == "packnet":
                seeds.append(run_packnet(dim, data, s))
            elif name == "ewc":
                seeds.append(run_ewc(dim, data, s))
            else:
                seeds.append(CP.evaluate("joint", data, s))
        rows[name] = {t: sum(sd[t] for sd in seeds) / len(seeds) for t in TASKS}
    print(f"{'learner':>10}" + "".join(f"{t:>13}" for t in TASKS) + f"{'mean':>8}")
    for name, r in rows.items():
        print(f"{name:>10}" + "".join(f"{r[t]:>13.3f}" for t in TASKS) + f"{sum(r.values()) / len(r):>8.3f}",
              flush=True)
    cm, pm = (sum(rows[x].values()) / 4 for x in ("coppice", "packnet"))
    print(f"\ncoppice {cm:.3f}  packnet {pm:.3f}  diff {cm - pm:+.3f}")
    print("read: diff ~ 0 = the tie HOLDS on the unified substrate + compositional chain -- Coppice's "
          "consolidation is at PARITY with PackNet (substrate advantage shared; value = interpretability). "
          "diff > 0 = Coppice's frozen-core reuse genuinely beats prune-and-release on compositional tasks "
          "(inspect C=rep_func: PackNet may prune a concept C needs).")


if __name__ == "__main__":
    main()
