"""Corpus-induced alphabets for standalone Wyly packages (no host LLM tokenizer).

WordCodec: whitespace/punct word stream matching the estate probe convention
(``[\\w']+|[.?:]`` — same as ``probe_estate2.py``). Vocab is fit only on the training
corpus; the codec JSON is shipped with the package so serve/bench need no host BPE.

Future codecs (corpus BPE, byte/adaptive dictionary) should implement the same surface:
``encode``, ``token_str``, ``from_file``, ``save``, ``hash`` — see roadmap Phase 1 L2/L3.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Same scanner as experiments/probe_estate2.py — estate member sets are word strings.
_WORD_RE = re.compile(r"[\w']+|[.?:]")

# Reserved ids
PAD_ID = 0
UNK_ID = 1
SPECIAL = {"<pad>": PAD_ID, "<unk>": UNK_ID}


class WordCodec:
    """Discrete word/punct alphabet. TokenSpace-compatible surface (encode / token_str)."""

    def __init__(self, stoi: dict[str, int], itos: list[str] | None = None):
        self.stoi = dict(stoi)
        if itos is None:
            itos = [""] * (max(stoi.values()) + 1)
            for s, i in stoi.items():
                itos[i] = s
        self.itos = itos
        self.vocab_size = len(itos)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _WORD_RE.findall(text)

    @classmethod
    def from_corpus(cls, text: str, min_count: int = 1) -> WordCodec:
        from collections import Counter

        counts = Counter(_WORD_RE.findall(text))
        stoi = dict(SPECIAL)
        for w, c in counts.most_common():
            if c < min_count or w in stoi:
                continue
            stoi[w] = len(stoi)
        itos = [""] * len(stoi)
        for s, i in stoi.items():
            itos[i] = s
        return cls(stoi, itos)

    @classmethod
    def from_file(cls, path: str | Path) -> WordCodec:
        d = json.loads(Path(path).read_text())
        return cls(d["stoi"], d.get("itos"))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps({
            "kind": "word",
            "stoi": self.stoi,
            "itos": self.itos,
            "vocab_size": self.vocab_size,
        }, indent=0))

    def encode(self, text: str) -> list[int]:
        # leading space is a TokenSpace convention; WordCodec ignores it
        t = text[1:] if text.startswith(" ") else text
        return [self.stoi.get(w, UNK_ID) for w in self.tokenize(t)]

    def token_str(self, tid: int) -> str:
        if 0 <= int(tid) < len(self.itos):
            return self.itos[int(tid)]
        return "<unk>"

    def encode_ids(self, text: str) -> list[int]:
        return self.encode(text)

    def hash(self) -> str:
        import hashlib
        blob = json.dumps({"stoi": self.stoi}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
