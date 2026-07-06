"""Does crystallization track domain structure? One row per corpus at the pythia-70m rung:
structure proxy = gzip(compressed/raw) of the DETOKENIZED window stream (uniform method across
all corpora, independent of raw-file availability); crystallization = certified core_sw / soft
student. Also: teacher gold top-1 (teacher==target on the kept windows), copy%, counts-only
determinism. Spearman rank correlation at the end."""
import gzip
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.tokens import TokenSpace  # noqa: E402

ts = TokenSpace.from_file(REPO.parent / "rosetta" / "models" / "pythia70m" /
                          "bundle.tokenizer.json")
DSS = ["wikitext", "wt103", "de", "legal", "math", "py", "code", "isabelle"]
rows = []
for ds in DSS:
    tsfx = "" if ds == "wikitext" else f"_{ds}"
    d = torch.load(REPO / "data" / f"wyly_nexttoken_{ds}_L256.pt", map_location="cpu")
    t = torch.load(REPO / "data" / f"wyly_teacher_pythia70m{tsfx}_L256.pt",
                   map_location="cpu")["teacher"]
    st = torch.load(REPO / "data" / f"wyly_v5_mined_pythia70m_{ds}_cov_ol_sw.pt",
                    map_location="cpu")
    ids, tgt = d["kept_ids"], d["target"]
    gold = float((t == tgt).float().mean())
    # copy%: teacher decision token appears in its own window (the copy-pattern incidence)
    smp = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))[:8000]
    copy = float((ids[smp] == t[smp, None]).any(1).float().mean())
    # structure proxy: gzip ratio of the detokenized stream (lower = more structured/repetitive)
    txt = ts.decode(ids[smp[:800]].reshape(-1).tolist()).encode()
    gz = len(gzip.compress(txt, 9)) / len(txt)
    student, core = st["full"], st["core_sw"]["agree"]
    rows.append((ds, gz, gold, copy, student, core, core / student))

rows.sort(key=lambda r: r[1])
print(f"{'corpus':>10} {'gzip':>6} {'gold':>6} {'copy%':>6} {'student':>8} {'core_sw':>8} "
      f"{'crystal':>8}")
for r in rows:
    print(f"{r[0]:>10} {r[1]:>6.3f} {r[2]:>6.3f} {r[3]:>6.1%} {r[4]:>8.3f} {r[5]:>8.3f} "
          f"{r[6]:>8.1%}")


def spearman(a, b):
    def rk(v):
        return {x: i for i, x in enumerate(sorted(v))}
    ra, rb = rk(a), rk(b)
    n = len(a)
    d2 = sum((ra[x] - rb[y]) ** 2 for x, y in zip(a, b, strict=True))
    return 1 - 6 * d2 / (n * (n * n - 1))


gzs = [r[1] for r in rows]
print(f"\nSpearman rank correlations vs gzip structure (n={len(rows)}; more structured = LOWER "
      f"gzip):")
for name, col in [("crystallization", 6), ("core_sw", 5), ("gold", 2), ("copy%", 3)]:
    v = [r[col] for r in rows]
    print(f"  gzip vs {name:>16}: rho = {spearman(gzs, v):+.3f}")
