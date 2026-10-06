"""POST-HOC diagnostic (not pre-registered): is the cal->test gap a property of this one split, or systematic?
A cheap word-blind gate (logistic on one-hot S, V, O; trained on the study's train split) stands in for the
transformer gate; cert16 is taken as all-true (the study's students had cert16 0.998-1.000)."""

import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from certified_student import HELD_OUT, WORDS
from ltt_certificate import SEEDS, grid_from_train, ltt_tau

z = np.load(sys.argv[1])
sid = torch.tensor(z["sid"])
dec = torch.tensor(z["dec"])
n = len(sid)
seen = torch.tensor([i for i, w in enumerate(WORDS) if w not in HELD_OUT])
order = np.random.default_rng(SEEDS["split"]).permutation(n)
tr, ca, te = order[:2400], order[2400:4200], order[4200:]
sc = (dec[:, seen] == dec[:, seen[:1]]).all(1)
V = int(sid.max()) + 1
X = torch.cat([F.one_hot(sid[:, 0, p], V).float() for p in (3, 4, 6)], 1)
torch.manual_seed(0)
W = torch.zeros(3 * V, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
o = torch.optim.Adam([W, b], lr=0.05)
for _ in range(500):
    l = F.binary_cross_entropy_with_logits(X[tr] @ W + b, sc[tr].float()) + 1e-3 * W.pow(2).sum()
    o.zero_grad()
    l.backward()
    o.step()
s = (X @ W + b).detach()
grid = grid_from_train(s[tr])
ones = torch.ones(1800, dtype=torch.bool)


def run(ca, te):
    ch, _ = ltt_tau(grid, s[ca], ones, sc[ca])
    if ch is None:
        return None, None, 0.0
    iss = s[te] >= ch[0]
    return ch[2] / ch[1], float((~sc[te][iss]).float().mean()), float(iss.float().mean())


c, t, cov = run(ca, te)
print(f"pre-registered split: cal FC {c:.3f}  test FC {t:.3f}  coverage {cov:.3f}")
c, t, cov = run(te, ca)
print(f"swapped (cal<->test): cal FC {c:.3f}  test FC {t:.3f}  coverage {cov:.3f}")
rng = np.random.default_rng(1)
hold = order[2400:]
res = []
for _ in range(300):
    p = rng.permutation(hold)
    res.append(run(p[:1800], p[1800:]))
r = np.array([(a, b_, cv) for a, b_, cv in res if a is not None])
print(
    f"300 random cal/test re-splits: issued {len(r)}; mean cal FC {r[:, 0].mean():.3f}, "
    f"mean test FC {r[:, 1].mean():.3f}, "
    f"test FC > 0.10 in {np.mean(r[:, 1] > 0.10):.3f}, mean coverage {r[:, 2].mean():.3f}"
)
# population-level: the 'true' FC at each grid tau over the whole 3600 holdout
