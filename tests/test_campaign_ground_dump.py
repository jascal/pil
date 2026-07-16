"""Hermetic tests for campaign_ground_dump alignment / pooling / checkpoints.

No fieldrun binary or live model run required. Uses the real Qwen2.5-0.5B tokenizer
file on disk for offset-correct alignment tests; residual tensors are fabricated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import campaign_ground_dump as cgd  # noqa: E402

TOKENIZER_PATH = Path(
    "/home/allans/code/fieldrun/bundles/Qwen2.5-0.5B-Instruct/"
    "Qwen2.5-0.5B-Instruct.tokenizer.json"
)

MANASSE_TEXT = "Manasse ist ein einzigartiger Parfümeur."
MANASSE_TOKENS = ["Manasse", "ist", "ein", "einzigartiger", "Parfümeur", "."]


@pytest.fixture(scope="module")
def tok():
    if not TOKENIZER_PATH.is_file():
        pytest.skip(f"tokenizer missing: {TOKENIZER_PATH}")
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER_PATH))


# ---------------------------------------------------------------------------
# 1. Required populated-alignment test (real tokenizer + fabricated residuals)
# ---------------------------------------------------------------------------


def test_manasse_populated_alignment(tok):
    """Hand-verified alignment for dev-s1 against real Qwen offsets."""
    enc = tok.encode(MANASSE_TEXT, add_special_tokens=False)
    ids = list(enc.ids)
    tokens = list(enc.tokens)
    L = len(ids)
    assert L == 13, f"expected 13 subwords, got {L}: {tokens}"
    assert tokens[:2] == ["Man", "asse"]
    # uniquelyartiger: ein zig art iger at 4..7
    assert 4 + 3 < L

    # Fabricated D: dim=4, nb=3; residual identity via cumsum of one-hot-ish blocks
    # Easier: make full residual r[pos] = full(dim, float(pos)) by putting all mass in block 0
    dim = 4
    nb = 3
    blocks = ["embed", "L0.attn", "L0.mlp"]  # n_layer=1; only need consistent labels for ckpts
    # Use richer blocks for checkpoint test separately; here use simple cum path
    # dumped positions 1..L-2 = 1..11
    dumped_positions = list(range(1, L - 1))
    assert dumped_positions == list(range(1, 12))
    n_dumped = len(dumped_positions)
    D = np.zeros((n_dumped, nb, dim), dtype=np.float32)
    for i, p in enumerate(dumped_positions):
        # All contribution in last block so cumsum to end = pos * ones
        D[i, -1, :] = float(p)

    dump = cgd.SentenceDump(
        sid="dev-s1",
        positions=dumped_positions,
        curs=[ids[p] for p in dumped_positions],
        targets=[ids[p + 1] for p in dumped_positions],
        D=D,
        blocks=blocks,
    )
    # checkpoints: single f=1.0 → L0.mlp
    block_idxs, _labs, n_layer = cgd.checkpoint_block_indices(blocks, [1.0])
    assert n_layer == 1
    assert block_idxs == [2]

    rec = {"sent_id": "dev-s1", "text": MANASSE_TEXT, "tokens": MANASSE_TOKENS}
    result = cgd.align_sentence(rec, dump, tok, block_idxs, dim=dim, n_ck=1)
    assert result.sanity_ok and result.drop_reason is None
    assert len(result.rows) == 6

    by_word = {r.word: r for r in result.rows}

    # Manasse: subwords 0,1 → only 1 dumped → last_pos=1, mean uses only 1
    m = by_word["Manasse"]
    assert m.subword_indices == [0, 1]
    assert m.dumped_positions == [1]
    assert m.last_pos == 1
    assert m.covered is True
    np.testing.assert_allclose(m.resid_last[0].astype(np.float32), np.full(dim, 1.0), atol=1e-3)
    np.testing.assert_allclose(m.resid_mean[0].astype(np.float32), np.full(dim, 1.0), atol=1e-3)

    # ist → index 2
    assert by_word["ist"].last_pos == 2
    assert by_word["ist"].covered is True
    np.testing.assert_allclose(by_word["ist"].resid_last[0].astype(np.float32), np.full(dim, 2.0), atol=1e-3)

    # ein → index 3
    assert by_word["ein"].last_pos == 3
    np.testing.assert_allclose(by_word["ein"].resid_last[0].astype(np.float32), np.full(dim, 3.0), atol=1e-3)

    # einzigartiger → 4,5,6,7 last=7; mean of 4..7
    e = by_word["einzigartiger"]
    assert e.subword_indices == [4, 5, 6, 7], e.subword_indices
    assert e.last_pos == 7
    assert e.covered is True
    np.testing.assert_allclose(e.resid_last[0].astype(np.float32), np.full(dim, 7.0), atol=1e-3)
    expected_mean = np.full(dim, np.mean([4.0, 5.0, 6.0, 7.0]), dtype=np.float32)
    np.testing.assert_allclose(e.resid_mean[0].astype(np.float32), expected_mean, atol=1e-3)

    # Parfümeur → 8,9,10,11 last=11
    p = by_word["Parfümeur"]
    assert p.subword_indices == [8, 9, 10, 11], p.subword_indices
    assert p.last_pos == 11
    np.testing.assert_allclose(p.resid_last[0].astype(np.float32), np.full(dim, 11.0), atol=1e-3)

    # "." → index 12, never dumped
    dot = by_word["."]
    assert dot.subword_indices == [12]
    assert dot.covered is False
    assert dot.last_pos == -1
    np.testing.assert_allclose(dot.resid_last[0].astype(np.float32), 0.0, atol=1e-6)
    np.testing.assert_allclose(dot.resid_mean[0].astype(np.float32), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Straddle detection on synthetic spans
# ---------------------------------------------------------------------------


def test_straddle_detection_synthetic():
    # Words: "ab" [0,2), "cd" [2,4)  text="abcd"
    word_spans = [(0, 2), (2, 4)]
    # Subwords: "a" [0,1), "bc" [1,3) straddles, "d" [3,4)
    content = [(0, 1), (1, 3), (3, 4)]
    word_subs, straddling = cgd.assign_subwords_to_words(content, word_spans)
    assert straddling == [1]
    assert word_subs[0] == [0]
    assert word_subs[1] == [2]


def test_whitespace_only_subword_excluded():
    word_spans = [(0, 3)]  # "foo"
    content = [None, (0, 3)]  # whitespace-only then full word
    word_subs, straddling = cgd.assign_subwords_to_words(content, word_spans)
    assert straddling == []
    assert word_subs[0] == [1]


# ---------------------------------------------------------------------------
# 3. Checkpoint / pooling arithmetic
# ---------------------------------------------------------------------------


def test_checkpoint_block_indices_and_full_residual():
    # 24 layers like Qwen0.5B: embed + 24*(attn,mlp) = 49
    blocks = ["embed"]
    for i in range(24):
        blocks.append(f"L{i}.attn")
        blocks.append(f"L{i}.mlp")
    assert len(blocks) == 49
    fracs = [0.25, 0.5, 0.75, 1.0]
    idxs, labels, n_layer = cgd.checkpoint_block_indices(blocks, fracs)
    assert n_layer == 24
    # layer_idx(f) = min(23, int(f*24))
    # 0.25 → int(6)=6 → L6.mlp; 0.5 → 12; 0.75 → 18; 1.0 → min(23,24)=23 → L23.mlp
    assert labels == ["L6.mlp", "L12.mlp", "L18.mlp", "L23.mlp"]
    for lab, bi in zip(labels, idxs, strict=True):
        assert blocks[bi] == lab

    # D with distinct block contributions; f=1.0 cum == full sum
    n, dim = 3, 5
    D = np.random.randn(n, len(blocks), dim).astype(np.float32)
    cum = cgd.cumulative_residuals(D, idxs)
    full = D.sum(axis=1)
    np.testing.assert_allclose(cum[:, -1, :], full, rtol=1e-5, atol=1e-5)
    # partial checkpoint is prefix sum
    bi0 = idxs[0]
    np.testing.assert_allclose(cum[:, 0, :], D[:, : bi0 + 1, :].sum(axis=1), rtol=1e-5, atol=1e-5)


def test_gsd_word_spans_glued_punct():
    text = "Bröchten, bitte"
    tokens = ["Bröchten", ",", "bitte"]
    spans = cgd.gsd_word_spans(text, tokens)
    assert spans == [(0, 8), (8, 9), (10, 15)]
    assert text[spans[0][0] : spans[0][1]] == "Bröchten"
    assert text[spans[1][0] : spans[1][1]] == ","


def test_trim_content_span_leading_space():
    text = "Manasse ist"
    # " ist" at (7,11)
    assert cgd.trim_content_span(text, 7, 11) == (8, 11)
    assert text[8:11] == "ist"


def test_gsd_span_mismatch_raises():
    with pytest.raises(AssertionError, match="GSD word/text mismatch"):
        cgd.gsd_word_spans("hello world", ["hello", "x"])
