"""Build data/wyly_nexttoken_pythia70m.pt -- the raw-token next-token dataset for the wyly_lm* arc.

Provenance: the arc's original file (``layers_pythia70m.pt``) was produced in a scratch session by
running pythia-70m over a text slice and storing per-layer residuals PLUS the raw 12-token windows
(``kept_ids``) and next tokens (``target``). The residuals are irrelevant to the Wyly-LM experiments
(which read raw tokens only) and the source text was not preserved, so:

  --from-layers PATH   slim an existing layers_*.pt into the canonical file EXACTLY (keeps every
                       published number reproducible; drops the ~300 MB of residuals).
  --from-text PATH     rebuild an equivalent dataset from any UTF-8 text file, tokenized with the
                       exact pure-python pythia BPE (pil.tokens.TokenSpace) and the pythia-70m
                       tokenizer shipped in the sibling rosetta package. Windows stride by 1 token;
                       numbers from a rebuilt corpus are equivalent-protocol, not bit-identical.

Run: cd pil && .venv/bin/python experiments/extract_nexttoken.py --from-layers <path>
     cd pil && .venv/bin/python experiments/extract_nexttoken.py --from-text corpus.txt [--n 40000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "wyly_nexttoken_pythia70m.pt"
TOKENIZER = REPO.parent / "rosetta" / "models" / "pythia70m" / "bundle.tokenizer.json"


def from_layers(path: Path) -> dict:
    d = torch.load(path, map_location="cpu")
    return {"kept_ids": d["kept_ids"].long(), "target": d["target"].long(), "L": int(d["L"])}


def from_text(path: Path, tokenizer: Path, window: int, n: int | None) -> dict:
    sys.path.insert(0, str(REPO))
    from pil.tokens import TokenSpace

    ts = TokenSpace.from_file(tokenizer)
    ids = torch.tensor(ts.encode(path.read_text()), dtype=torch.long)
    if len(ids) < window + 1:
        raise SystemExit(f"corpus too short: {len(ids)} tokens < window {window}+1")
    windows = ids.unfold(0, window + 1, 1)  # stride-1 windows of L+1
    if n is not None and len(windows) > n:
        windows = windows[torch.randperm(len(windows), generator=torch.Generator().manual_seed(0))[:n]]
    return {"kept_ids": windows[:, :window].contiguous(), "target": windows[:, window].contiguous(),
            "L": window}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-layers", type=Path, help="existing layers_*.pt to slim exactly")
    src.add_argument("--from-text", type=Path, help="UTF-8 text file to tokenize into windows")
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--n", type=int, default=40000, help="windows to keep from --from-text (seed-0 sample)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    d = from_layers(a.from_layers) if a.from_layers else from_text(a.from_text, a.tokenizer, a.window, a.n)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, a.out)
    print(f"wrote {a.out}  kept_ids {tuple(d['kept_ids'].shape)}  target {tuple(d['target'].shape)}  "
          f"L={d['L']}")


if __name__ == "__main__":
    main()
