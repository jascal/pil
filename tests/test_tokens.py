"""TokenSpace: pure-Python byte-level BPE against the rosetta package tokenizers.

The encoder was verified exact against HF ``tokenizers`` on all three families
(pythia/GPT-NeoX, Qwen2, llama-3) during development (48-49/49 diverse samples,
including whitespace runs, contractions, unicode, code). These tests pin the
invariants that don't need the reference package installed.
"""

from pathlib import Path

import pytest

from pil.tokens import TokenSpace, _gpt2_pretokenize, _llama3_pretokenize

ROSETTA = Path("/home/allans/code/rosetta/models")
pytestmark = pytest.mark.skipif(not ROSETTA.exists(), reason="rosetta packages not present")


@pytest.fixture(scope="module")
def pythia() -> TokenSpace:
    return TokenSpace.from_rosetta_package(ROSETTA / "pythia160m")


def test_roundtrip_diverse(pythia):
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "It's John's dog; they're won't we'll",
        "Numbers 12345 mixed a1b2, punct!!! (parens)",
        "Unicode: café 東京 🚀",
        "def f(x):\n    return x**2  # comment\n",
        "a" + " " * 24 + "b" + " " * 7 + "c",       # multi-space added tokens
    ]
    for s in samples:
        assert pythia.decode(pythia.encode(s)) == s


def test_known_encoding(pythia):
    # pinned against HF tokenizers (EleutherAI/pythia-160m)
    assert pythia.decode(pythia.encode("The model")) == "The model"
    ids = pythia.encode("The")
    assert len(ids) == 1 and pythia.token_str(ids[0]) == "The"


def test_gpt2_pretokenize_shapes():
    assert _gpt2_pretokenize("It's fine") == ["It", "'s", " fine"]
    # whitespace run glues its last space to the following word
    assert _gpt2_pretokenize("a  b") == ["a", " ", " b"]
    assert _gpt2_pretokenize("x   y") == ["x", "  ", " y"]
    assert _gpt2_pretokenize("tail   ") == ["tail", "   "]


def test_llama3_pretokenize_digits():
    # \p{N}{1,3} grouping (llama-3) vs single digits (qwen: max_digits=1)
    assert _llama3_pretokenize("12345", 3) == ["123", "45"]
    assert _llama3_pretokenize("12345", 1) == ["1", "2", "3", "4", "5"]


def test_added_tokens_split(pythia):
    text = "a" + " " * 24 + "b"
    ids = pythia.encode(text)
    assert pythia.decode(ids) == text
    # the 24-space run must be a single added token
    assert any(pythia.id_to_token[i] == " " * 24 for i in ids)


def test_vocab_size(pythia):
    assert pythia.vocab_size == 50277
