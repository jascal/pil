"""Build word-alphabet next-token datasets + codecs for standalone bAbI (P1).

No host tokenizer. Vocab fit on the train corpus only.

  data/alphabet_babi2_word.json
  data/wyly_nexttoken_babi2_word_L256.pt
  data/alphabet_babi3x_word.json
  data/wyly_nexttoken_babi3x_word_L512.pt

Run: cd pil && .venv/bin/python experiments/build_word_babi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.alphabet import WordCodec  # noqa: E402

SPECS = [
    {"ds": "babi2_word", "corpus": "corpus_babi_qa2.txt", "L": 256, "n": 40_000},
    {"ds": "babi3x_word", "corpus": "corpus_babi_qa3.txt", "L": 512, "n": 40_000},
]


def build_one(corpus_name: str, ds: str, L: int, n: int) -> None:
    text = (REPO / "data" / corpus_name).read_text()
    codec = WordCodec.from_corpus(text)
    codec_path = REPO / "data" / f"alphabet_{ds}.json"
    # alphabet file named alphabet_babi2_word.json
    codec_path = REPO / "data" / f"alphabet_{ds}.json"
    codec.save(codec_path)
    ids = torch.tensor(codec.encode(text), dtype=torch.long)
    if len(ids) < L + 1:
        raise SystemExit(f"{corpus_name}: too short ({len(ids)} tokens)")
    # stride > 1 reduces speed/diversity; keep corpus order sample
    stride = max(1, (len(ids) - L) // (n * 2))
    windows = ids.unfold(0, L + 1, stride)
    if len(windows) > n:
        # evenly spaced selection in corpus order
        idx = torch.linspace(0, len(windows) - 1, n).long()
        windows = windows[idx]
    out = {
        "kept_ids": windows[:, :L].contiguous(),
        "target": windows[:, L].contiguous(),
        "L": L,
        "alphabet": str(codec_path.name),
        "kind": "word",
    }
    pt = REPO / "data" / f"wyly_nexttoken_{ds}_L{L}.pt"
    # load_ds expects wyly_nexttoken_{DS}_L256.pt for non-wikitext — DS tag embeds L only for
    # legacy; actual path uses L256 in name even when L=512 for babi3x historically.
    # For word sets we write both the true-L name and the L256 alias load_ds expects.
    torch.save(out, pt)
    alias = REPO / "data" / f"wyly_nexttoken_{ds}_L256.pt"
    if alias != pt:
        torch.save(out, alias)
    print(f"{ds}: vocab={codec.vocab_size} stream={len(ids)} windows={len(windows)} "
          f"L={L} -> {pt.name} (+ alias {alias.name}) codec={codec_path.name} "
          f"hash={codec.hash()}")


def main():
    for s in SPECS:
        build_one(s["corpus"], s["ds"], s["L"], s["n"])


if __name__ == "__main__":
    main()
