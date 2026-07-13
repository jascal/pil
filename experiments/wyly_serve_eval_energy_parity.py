"""DECIDE_ENERGY Slice 1 parity: energy-beam M=1/beam_width=1 vs support-weighted
on the real production package + held-out test windows.

No training, no GPU, no C++ spoke. Loads the on-disk expert package and the
cached dataset, then asserts bit-exact decide() agreement on every test window
and a zero gap on agree/cover.

Usage: .venv/bin/python experiments/wyly_serve_eval_energy_parity.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

os.environ.setdefault("WYLY_TAG", "pythia70m")
os.environ.setdefault("WYLY_LIB", "mined")
os.environ.setdefault("WYLY_JUDGE", "cover")
os.environ.setdefault("WYLY_ONLINE", "1")
os.environ.setdefault("WYLY_COVER", "sw")

import wyly_lm_v5 as v5  # noqa: E402  (env must be set first)
from serve_package import decide, load_package  # noqa: E402

DS = os.environ.get("WYLY_DS", "wikitext")
PKG = REPO / "data" / ("wyly_expert_package_v5" if DS == "wikitext"
                       else f"wyly_expert_package_v5_{DS}")


def _strip_cert(d):
    if d is None:
        return None
    out = dict(d)
    out.pop("cert_kind", None)
    return out


def main():
    pkg_path = PKG / "manifest.json"
    if not pkg_path.exists():
        print(f"FAIL: missing production package {pkg_path}")
        sys.exit(1)

    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    idioms, ngrams, m_sw = load_package(pkg_path)
    assert m_sw.get("cover") == "support-weighted", (
        f"expected support-weighted package, got {m_sw.get('cover')!r}")
    m_e = dict(m_sw)
    m_e["cover"] = "energy-beam"
    m_e["M"] = 1
    m_e["beam_width"] = 1
    uvl = uv.tolist()
    te_l = te.tolist()

    t0 = time.time()
    n_agree_sw = n_fired_sw = 0
    n_agree_e = n_fired_e = 0
    n = len(te_l)
    for wi in te_l:
        ctx = [uvl[c] for c in ids[wi].tolist()]
        d_sw = decide(ctx, idioms, ngrams, m_sw)
        d_e = decide(ctx, idioms, ngrams, m_e)
        if d_sw is None and d_e is None:
            continue
        if d_e is not None and d_e.get("cert_kind") != "per-token":
            print(f"FAIL: energy commit missing cert_kind=per-token at wi={wi}")
            print(f"  ctx={ctx!r}")
            print(f"  energy={d_e!r}")
            sys.exit(1)
        if _strip_cert(d_e) != d_sw:
            print(f"FAIL: bit-exact divergence at wi={wi}")
            print(f"  ctx={ctx!r}")
            print(f"  support-weighted={d_sw!r}")
            print(f"  energy-beam={d_e!r}")
            sys.exit(1)
        gold = uvl[yv[wi]]
        if d_sw:
            n_fired_sw += 1
            if d_sw["answer"] == gold:
                n_agree_sw += 1
        if d_e:
            n_fired_e += 1
            if d_e["answer"] == gold:
                n_agree_e += 1

    agree_sw, cover_sw = n_agree_sw / n, n_fired_sw / n
    agree_e, cover_e = n_agree_e / n, n_fired_e / n
    gap_agree = agree_e - agree_sw
    gap_cover = cover_e - cover_sw
    elapsed = time.time() - t0
    print(f"[{DS}] energy-beam M=1 parity over {n} test windows ({elapsed:.0f}s)")
    print(f"[{DS}] support-weighted: agree {agree_sw:.6f} @ cover {cover_sw:.4%}")
    print(f"[{DS}] energy-beam:      agree {agree_e:.6f} @ cover {cover_e:.4%}")
    print(f"[{DS}] gap agree {gap_agree:+.6f}  gap cover {gap_cover:+.6f}")
    if gap_agree != 0.0 or gap_cover != 0.0:
        print(f"FAIL: non-zero gap (agree={gap_agree}, cover={gap_cover})")
        sys.exit(1)
    print(f"[{DS}] PASS: bit-exact parity, gap 0.0")


if __name__ == "__main__":
    main()
