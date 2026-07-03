"""Tokenizer compatibility layer: the *same* tokens as the LLMs modeled in rosetta.

The PIC rule learner's propositions are token ids in a real LLM vocabulary, so learned
programs are directly comparable to (and exportable next to) the rosetta/fieldrun packages.
This module reads a HuggingFace ``tokenizer.json`` (rosetta packages ship it as
``bundle.tokenizer.json``; HF snapshots ship it as ``tokenizer.json``) and implements
byte-level BPE encode/decode in pure Python -- no new dependencies, verified by exact
round-trip against the package's own pre-tokenized ``corpus.json``.

Supported: ByteLevel-BPE tokenizers (GPT-2 / GPT-NeoX-pythia / Qwen2 / llama-3 style):
NFC normalizer, ByteLevel pre-tokenizer with the GPT-2 regex, BPE merges, added tokens.
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte<->unicode bijection (printable stand-ins for raw bytes)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _class_run(text: str, start: int) -> int:
    """End index of the maximal same-class run (letters | numbers | other-non-space) at ``start``."""
    n = len(text)
    ch = text[start]
    if _is_letter(ch):
        pred = _is_letter
    elif _is_number(ch):
        pred = _is_number
    else:

        def pred(c: str) -> bool:
            return not (c.isspace() or _is_letter(c) or _is_number(c))

    j = start + 1
    while j < n and pred(text[j]):
        j += 1
    return j


def _gpt2_pretokenize(text: str) -> list[str]:
    """Scanner equivalent to the GPT-2 ByteLevel regex (ordered alternation):

        's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

    (Python ``re`` lacks ``\\p{..}``; the scanner uses unicodedata categories instead.
    Note `` ``-style unicode spaces are *not* ``\\s`` for this pattern's purposes in
    HF's implementation, but they are `Z`-category and non-letter/number, so they land in
    the punctuation class either way only if ``str.isspace()`` disagrees; corpus round-trip
    is the arbiter.)
    """
    pieces: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        # 1) contractions
        if ch == "'":
            for c in _CONTRACTIONS:
                if text.startswith(c, i):
                    pieces.append(c)
                    i += len(c)
                    break
            else:
                j = _class_run(text, i)
                pieces.append(text[i:j])
                i = j
            continue
        # 2-4)  ` ?<class-run>` -- a single literal space may prefix a class run
        if ch == " " and i + 1 < n and not text[i + 1].isspace():
            j = _class_run(text, i + 1)
            pieces.append(text[i:j])
            i = j
            continue
        # 5-6) whitespace runs: `\s+(?!\S)` then `\s+`
        if ch.isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j == n:
                pieces.append(text[i:j])  # trailing run: `\s+(?!\S)` takes it all
                i = j
            elif j - i > 1:
                pieces.append(text[i : j - 1])  # run minus final char (lookahead sees \S)
                i = j - 1  # final whitespace re-enters the loop and glues to the next piece
            else:
                pieces.append(ch)  # lone non-gluing whitespace (e.g. '\n' before a word)
                i = j
            continue
        # plain class run with no leading space
        j = _class_run(text, i)
        pieces.append(text[i:j])
        i = j
    return pieces


def _llama3_pretokenize(text: str, max_digits: int) -> list[str]:
    """Scanner for the GPT-4/cl100k-style split regex used by Qwen2 / llama-3:

        (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\\r\\n\\p{L}\\p{N}]?\\p{L}+ | \\p{N}{1,k}
        |  ?[^\\s\\p{L}\\p{N}]+[\\r\\n]* | \\s*[\\r\\n]+ | \\s+(?!\\S) | \\s+

    with ``k = max_digits`` (3 for llama-3, 1 for Qwen2).
    """

    def is_punct(c: str) -> bool:
        return not (c.isspace() or _is_letter(c) or _is_number(c))

    pieces: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        # 1) case-insensitive contractions
        if ch == "'":
            low = text[i : i + 3].lower()
            hit = next((c for c in _CONTRACTIONS if low.startswith(c)), None)
            if hit is not None:
                pieces.append(text[i : i + len(hit)])
                i += len(hit)
                continue
        # 2) [^\r\n L N]? \p{L}+
        if _is_letter(ch):
            j = i + 1
            while j < n and _is_letter(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue
        if ch not in "\r\n" and not _is_number(ch) and i + 1 < n and _is_letter(text[i + 1]):
            j = i + 2
            while j < n and _is_letter(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue
        # 3) \p{N}{1,k}
        if _is_number(ch):
            j = i + 1
            while j < n and j - i < max_digits and _is_number(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue
        # 4)  ?[^\s L N]+ [\r\n]*
        if is_punct(ch) or (ch == " " and i + 1 < n and is_punct(text[i + 1])):
            j = i + 1 if is_punct(ch) else i + 2
            while j < n and is_punct(text[j]):
                j += 1
            while j < n and text[j] in "\r\n":
                j += 1
            pieces.append(text[i:j])
            i = j
            continue
        # 5) \s*[\r\n]+  -- whitespace prefix ending at the run's last newline
        if ch.isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            run = text[i:j]
            last_nl = max(run.rfind("\r"), run.rfind("\n"))
            if last_nl >= 0:
                pieces.append(run[: last_nl + 1])
                i += last_nl + 1
                continue
            # 6-7) \s+(?!\S) then \s+
            if j == n:
                pieces.append(run)
                i = j
            elif j - i > 1:
                pieces.append(text[i : j - 1])
                i = j - 1
            else:
                pieces.append(ch)
                i = j
            continue
        # lone non-letter char followed by non-letter (e.g. stray byte before digit)
        pieces.append(ch)
        i += 1
    return pieces


class TokenSpace:
    """A vocabulary + byte-level BPE codec loaded from a ``tokenizer.json``.

    ``TokenSpace.from_rosetta_package("/path/to/rosetta/models/pythia160m")`` binds the
    learner to the exact token space of a modeled LLM.
    """

    def __init__(self, tokenizer_json: dict):
        model = tokenizer_json["model"]
        if model.get("type") != "BPE":
            raise ValueError(f"unsupported tokenizer model type {model.get('type')!r} (want BPE)")
        self.vocab: dict[str, int] = dict(model["vocab"])
        merges = model["merges"]
        # merges may be "a b" strings or [a, b] pairs depending on tokenizers version
        pairs = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in merges]
        self.merge_ranks: dict[tuple[str, str], int] = {p: r for r, p in enumerate(pairs)}
        self.added: dict[str, int] = {}
        for at in tokenizer_json.get("added_tokens", []):
            self.added[at["content"]] = at["id"]
            self.vocab.setdefault(at["content"], at["id"])
        self.id_to_token: dict[int, str] = {i: t for t, i in self.vocab.items()}
        norm = tokenizer_json.get("normalizer") or {}
        self.nfc = norm.get("type") == "NFC"
        self._pretok = self._detect_pretokenizer(tokenizer_json.get("pre_tokenizer") or {})
        self.b2u = _bytes_to_unicode()
        self.u2b = {u: b for b, u in self.b2u.items()}
        # longest-first added-token matching (the multi-space pythia tokens)
        self._added_sorted = sorted(self.added, key=len, reverse=True)

    @staticmethod
    def _detect_pretokenizer(pre: dict):
        """Pick the scanner matching the tokenizer's split regex family."""
        if pre.get("type") == "ByteLevel":
            return _gpt2_pretokenize
        if pre.get("type") == "Sequence":
            for sub in pre.get("pretokenizers", []):
                if sub.get("type") == "Split":
                    pat = sub.get("pattern", {}).get("Regex", "")
                    max_digits = 3 if "\\p{N}{1,3}" in pat else 1

                    def scan(text: str, k: int = max_digits) -> list[str]:
                        return _llama3_pretokenize(text, k)

                    return scan
        return _gpt2_pretokenize

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> TokenSpace:
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def from_rosetta_package(cls, package_dir: str | Path) -> TokenSpace:
        return cls.from_file(Path(package_dir) / "bundle.tokenizer.json")

    @property
    def vocab_size(self) -> int:
        return max(self.id_to_token) + 1

    # -- BPE ---------------------------------------------------------------
    def _bpe(self, piece: str) -> list[str]:
        word = list(piece)
        if len(word) < 2:
            return word
        while True:
            best, best_rank = None, None
            for a, b in zip(word, word[1:], strict=False):
                r = self.merge_ranks.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = (a, b), r
            if best is None:
                return word
            a, b = best
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
            if len(word) == 1:
                return word

    def encode(self, text: str) -> list[int]:
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        return self._encode_with_added(text)

    def _encode_with_added(self, text: str) -> list[int]:
        # added tokens match on raw text, longest first, before pre-tokenization
        for tok in self._added_sorted:
            idx = text.find(tok)
            if idx != -1:
                left = self._encode_with_added(text[:idx]) if idx else []
                right = self._encode_with_added(text[idx + len(tok) :])
                return left + [self.added[tok]] + right
        ids: list[int] = []
        for piece in self._pretok(text):
            mapped = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            for sub in self._bpe(mapped):
                if sub not in self.vocab:
                    raise KeyError(f"BPE produced unknown token {sub!r}")
                ids.append(self.vocab[sub])
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        buf = bytearray()
        for i in ids:
            tok = self.id_to_token[i]
            if tok in self.added:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(tok)
            else:
                buf.extend(self.u2b[u] for u in tok)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def token_str(self, i: int) -> str:
        """Legible single-token string (for export / display)."""
        return self.decode([i])
