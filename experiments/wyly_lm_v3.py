"""Wyly-LM v3: AUTONOMOUS discovery on real text -- the sleep loop finds, admits and installs the rules.

Rung 3 of the self-compiling-learner goal. Identical to wyly_lm_v2.py EXCEPT: no hand-installed
induction rule. Instead, after each wake episode the SLEEP loop proposes exact-program candidates
from the extraction library (induction at L=1,2,3 -- the same semantics the rosetta package runtime
executes), and the ADMISSION JUDGE decides: a candidate is installed iff it improves held-out val
top-1 by more than the threshold. Each admitted rule records the sleep cycle that admitted it and
its measured marginal -- provenance says when the model learned it. The +0.042 copy-subset gain that
v2 got from a hand-installed rule must now arrive from the machine.

Rung 4: after every sleep the updated rosetta package is emitted (count ngrams + the admitted
induction rules, each citing its sleep cycle); the trusted tier grows 0 -> N across the run, and the
final package is served by the sgiandubh spoke (same probes as before, plus the grown tier).

Run: cd pil && .venv/bin/python experiments/wyly_lm_v3.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wyly_lm_v2 as v2
from wyly_data import load_windows_tied
from wyly_export_package import build_manifest
from wyly_lm_grounded import grounded_init

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "data" / "wyly_expert_package_v3"
DEV, V = v2.DEV, v2.V
ADMIT = 0.002                                                # the judge's val-marginal threshold


def mir_induction_L(lm):
    """exact torch mirror of the package runtime's induction kind at suffix length L=lm."""
    def fn(ids):
        n = ids.shape[1]
        suf = ids[:, -lm:]
        m = torch.ones(len(ids), n - lm, dtype=torch.bool, device=ids.device)
        for o in range(lm):
            m &= ids[:, o:o + n - lm] == suf[:, o:o + 1]
        has = m.any(1)
        mp = (m.float() * torch.arange(1, n - lm + 1, device=ids.device)).argmax(1)
        ans = ids[torch.arange(len(ids), device=ids.device), (mp + lm).clamp(max=n - 1)]
        return torch.where(has, ans, torch.full_like(ans, -1))
    return fn


class WylyV3(v2.WylyV2):
    """v2 minus the hand-installed rule; rules arrive only via the sleep loop's judge."""

    def __init__(self, ground, cls, ell):
        super().__init__(ground, cls, ell)
        self.rules = []                                      # [(name, mirror_fn)]
        self.rw = torch.nn.ParameterList()                   # learned per-rule weights

    def forward(self, ids, use_rules=True, **kw):
        out = super().forward(ids, use_ind=False, **kw)
        if use_rules:
            for (_name, fn), w in zip(self.rules, self.rw, strict=False):
                a = fn(ids)
                c = self.lut[a.clamp(min=0)]
                fire = (a >= 0) & (c >= 0)
                out[fire] = out[fire] + w * F.one_hot(c[fire], out.shape[1]).float()
        return out

    def install(self, name, fn):
        self.rules.append((name, fn))
        self.rw.append(torch.nn.Parameter(torch.tensor(1.0, device=self.C.device)))


def sleep_admit(model, ids, y, val, cycle, candidates, log):
    """the judge: temp-install each not-yet-installed candidate; admit the best if it pays."""
    base = v2.top1(model, ids, y, val)
    best = (ADMIT, None)
    for name, fn in candidates:
        if any(n == name for n, _ in model.rules):
            continue
        model.install(name, fn)
        marg = v2.top1(model, ids, y, val) - base
        model.rules.pop()
        model.rw = torch.nn.ParameterList(list(model.rw)[:-1])
        log.append(f"    sleep {cycle}: candidate {name} val marginal {marg:+.4f}")
        if marg > best[0]:
            best = (marg, (name, fn))
    if best[1]:
        name, fn = best[1]
        model.install(name, fn)
        log.append(f"    sleep {cycle}: ADMITTED {name} (val marginal {best[0]:+.4f})")
        return {"kind": "induction", "tier": "trusted", "basis": "causal",
                "L": int(name.split("L=")[1]),
                "citation": [f"admitted by the sleep judge at cycle {cycle} "
                             f"(held-out val marginal {best[0]:+.4f}); exact program, semantics = "
                             "the package runtime's induction kind; family certified in "
                             "wyly_rel_harden (Soufflé, proved on its domain)"]}
    return None


def main():
    ids, y, cls, uv, tr, te = load_windows_tied(V, DEV, v2.DATA)
    vocab, ell = len(uv), ids.shape[1]
    cs = v2.copy_subset(ids, y, cls, te)
    print(f"Wyly-LM v3 (autonomous sleep loop) -- L={ell}, vocab {vocab}, {DEV}; "
          f"copy-subset {len(cs)}/{len(te)}")
    ground = grounded_init(uv).to(DEV)
    ground = ground / ground.shape[1] ** 0.5
    torch.manual_seed(0)
    model = WylyV3(ground, cls, ell).to(DEV)
    lut = model.lut
    g = torch.Generator().manual_seed(0)
    val = tr[torch.randperm(len(tr), generator=g)[:v2.NVAL]]
    fit = tr[~torch.isin(tr, val)]
    candidates = [(f"induction L={lm}", mir_induction_L(lm)) for lm in (1, 2, 3)]
    sys.path.insert(0, str(REPO))
    from pil.tokens import TokenSpace
    ts = TokenSpace.from_file(v2.DATA.parent.parent.parent / "rosetta" / "models" / "pythia70m"
                              / "bundle.tokenizer.json")
    admitted, tier_timeline, log = [], [], []
    PKG.mkdir(parents=True, exist_ok=True)
    print(f"\n{'episode':>8}{'test top-1':>12}{'copy-subset':>13}{'trusted tier':>14}")
    for ep, ch in enumerate(torch.chunk(fit, v2.EPISODES)):
        model.update_counts(ids[ch], y[ch], lut)
        for _ in range(v2.WAKE_STEPS):                       # WAKE: plain SGD
            bi = ch[torch.randint(len(ch), (v2.BATCH,), generator=g)]
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(ids[bi]), y[bi]).backward()
            model.sgd(v2.LR)
        rec = sleep_admit(model, ids, y, val, ep, candidates, log)   # SLEEP: the judge
        if rec:
            admitted.append(rec)
        man = build_manifest(model.counts, cls, uv, ts, 1, 0.0, induction_rules=admitted,
                             model="wyly-v3-selfcompiled")
        (PKG / "manifest.json").write_text(json.dumps(man))          # the GROWING package
        tier_timeline.append(len(admitted))
        print(f"{ep:>8}{v2.top1(model, ids, y, te):>12.3f}{v2.top1(model, ids, y, cs):>13.3f}"
              f"{len(admitted):>14}", flush=True)
    shutil.copy(v2.DATA.parent.parent.parent / "rosetta" / "models" / "pythia70m"
                / "bundle.tokenizer.json", PKG / "bundle.tokenizer.json")
    print("\n" + "\n".join(log))
    full = v2.top1(model, ids, y, te)
    norules = v2.top1(model, ids, y, te, use_rules=False)
    cs_full = v2.top1(model, ids, y, cs)
    cs_norules = v2.top1(model, ids, y, cs, use_rules=False)
    print(f"\nrung 3 -- autonomous discovery: {len(admitted)} rules admitted by the judge "
          f"(tier timeline {tier_timeline})")
    print(f"  test {full:.3f} (rules ablated {norules:.3f}, marginal {full - norules:+.3f})")
    print(f"  copy-subset {cs_full:.3f} (rules ablated {cs_norules:.3f}, "
          f"marginal {cs_full - cs_norules:+.3f}; v2's hand-installed gain was +0.042)")
    print(f"  vs floor 0.148: {'CLEARED' if full > 0.148 else 'not cleared'} ({full - 0.148:+.3f})")
    print(f"\nrung 4 -- the growing package: {PKG}")
    print(f"  trusted tier {tier_timeline[0]} -> {tier_timeline[-1]}; every rule cites its sleep "
          "cycle + measured marginal. Serve: ../sgiandubh/build/sgiandubh --rosetta-package "
          f"{PKG} 8099")


if __name__ == "__main__":
    main()
