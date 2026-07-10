"""Build word-alphabet datasets + corpus-mined queries for standalone bAbI (P1+P2).

No host tokenizer. Vocab fit on the *fit* region of the corpus only (first ~90% of
inline Q/A by temporal order). Judge queries are mined from the held-out tail.

Outputs per dataset tag (e.g. babi2_word)::

  data/alphabet_{ds}.json
  data/wyly_nexttoken_{ds}_L256.pt
  data/wyly_queries_mined_{ds}.json (+ .meta.json)

Run: cd pil && .venv/bin/python experiments/build_word_babi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.alphabet import WordCodec  # noqa: E402
from pil.query_mine import mine_inline_qa  # noqa: E402

SPECS = [
    {
        "ds": "babi2_word",
        "corpus": "corpus_babi_qa2.txt",
        "L": 256,
        "n": 40_000,
    },
    {
        "ds": "babi3x_word",
        "corpus": "corpus_babi_qa3.txt",
        "L": 512,
        "n": 40_000,
    },
]


def build_one(corpus_name: str, ds: str, L: int, n: int,
              holdout_frac: float = 0.1, max_queries: int = 1000) -> None:
    text = (REPO / "data" / corpus_name).read_text()
    queries, fit_end = mine_inline_qa(
        text, holdout_frac=holdout_frac, max_queries=max_queries)
    qpath = REPO / "data" / f"wyly_queries_mined_{ds}.json"
    qpath.write_text(__import__("json").dumps(queries, indent=0))
    meta = {
        "source": corpus_name,
        "n_queries": len(queries),
        "fit_end_char": fit_end,
        "corpus_chars": len(text),
        "holdout_frac": holdout_frac,
        "query_source": "corpus_mined",
    }
    qpath.with_suffix(".meta.json").write_text(
        __import__("json").dumps(meta, indent=2))

    # Windows/tables only from fit region (no judge-story leak into counts)
    fit_text = text[:fit_end] if fit_end > 0 else text
    codec = WordCodec.from_corpus(fit_text)
    codec_path = REPO / "data" / f"alphabet_{ds}.json"
    codec.save(codec_path)

    ids = torch.tensor(codec.encode(fit_text), dtype=torch.long)
    if len(ids) < L + 1:
        raise SystemExit(f"{corpus_name}: fit region too short ({len(ids)} tokens)")
    stride = max(1, (len(ids) - L) // (n * 2))
    windows = ids.unfold(0, L + 1, stride)
    if len(windows) > n:
        idx = torch.linspace(0, len(windows) - 1, n).long()
        windows = windows[idx]
    out = {
        "kept_ids": windows[:, :L].contiguous(),
        "target": windows[:, L].contiguous(),
        "L": L,
        "alphabet": codec_path.name,
        "kind": "word",
        "fit_end_char": fit_end,
        "query_source": "corpus_mined",
    }
    alias = REPO / "data" / f"wyly_nexttoken_{ds}_L256.pt"
    torch.save(out, alias)
    if L != 256:
        torch.save(out, REPO / "data" / f"wyly_nexttoken_{ds}_L{L}.pt")
    print(
        f"{ds}: vocab={codec.vocab_size} fit_tokens={len(ids)} windows={len(windows)} "
        f"L={L} queries={len(queries)} fit_end={fit_end}/{len(text)} "
        f"codec={codec_path.name} hash={codec.hash()} -> {alias.name}"
    )


def main():
    for s in SPECS:
        build_one(s["corpus"], s["ds"], s["L"], s["n"])


if __name__ == "__main__":
    main()
