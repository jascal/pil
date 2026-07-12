"""Pure-function tests for the code-legality residual probe (no v5.main / no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_code_legality_probe as X  # noqa: E402  # isort: skip
import wyly_lm_v5 as v5  # noqa: E402  # isort: skip


# ---------------------------------------------------------------------------
# Fixtures: hand-built compact vocab + duck-typed TokenSpace
# ---------------------------------------------------------------------------

# Compact ids: 0=pad/id-like, 1='(', 2=')', 3='[', 4=']', 5='{', 6='}',
# 7=' (', 8=' )', 9=identifier
TOK_STR = {
    10: "x",
    11: "(",
    12: ")",
    13: "[",
    14: "]",
    15: "{",
    16: "}",
    17: " (",
    18: " )",
    19: "foo",
}


class _FakeTS:
    def token_str(self, t: int) -> str:
        return TOK_STR[int(t)]


def _uv_tensor() -> torch.Tensor:
    # uv[compact] = raw token id; compact indices 0..9 map to raw 10..19
    return torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.long)


def _hand_maps() -> tuple[dict[int, str], dict[int, str]]:
    """opener/closer char maps keyed by compact id (no tokenizer needed)."""
    opener_char = {1: "(", 3: "[", 5: "{", 7: "("}
    closer_char = {2: ")", 4: "]", 6: "}", 8: ")"}
    return opener_char, closer_char


def _is_open_close(vocab: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    is_open = torch.zeros(vocab, dtype=torch.bool)
    is_close = torch.zeros(vocab, dtype=torch.bool)
    for i in (1, 3, 5, 7):
        is_open[i] = True
    for i in (2, 4, 6, 8):
        is_close[i] = True
    return is_open, is_close


# ---------------------------------------------------------------------------
# bracket_char_maps composes with real v5.bracket_sets
# ---------------------------------------------------------------------------

def test_bracket_char_maps_composes_with_real_bracket_sets():
    uv = _uv_tensor()
    ts = _FakeTS()
    op, cl = v5.bracket_sets(uv, ts)
    opener_char, closer_char = X.bracket_char_maps(op, cl, uv, ts)
    # bare + space-prefixed paren variants both present
    assert set(opener_char.values()) == {"(", "[", "{"}
    assert set(closer_char.values()) == {")", "]", "}"}
    # two compact ids map to ')' (bare + space-prefixed)
    assert sum(1 for c in closer_char.values() if c == ")") == 2
    assert sum(1 for c in opener_char.values() if c == "(") == 2


# ---------------------------------------------------------------------------
# mate_recoverable_mask
# ---------------------------------------------------------------------------

def test_mate_recoverable_matching_closer_any_variant():
    opener_char, closer_char = _hand_maps()
    # mate_opener compact id 1 = '('; gold bare ')' compact id 2; wrong pred
    mate = torch.tensor([1])
    gold = torch.tensor([2])
    pred = torch.tensor([9])  # wrong
    mask = X.mate_recoverable_mask(mate, gold, pred, opener_char, closer_char)
    assert bool(mask[0].item()) is True


def test_mate_not_recoverable_when_gold_not_closer():
    opener_char, closer_char = _hand_maps()
    mate = torch.tensor([1])
    gold = torch.tensor([9])  # identifier, not a closer
    pred = torch.tensor([2])
    mask = X.mate_recoverable_mask(mate, gold, pred, opener_char, closer_char)
    assert bool(mask[0].item()) is False


def test_mate_not_recoverable_when_pred_equals_gold():
    """Classifier's pred != gold check is load-bearing (not only residual prefilter)."""
    opener_char, closer_char = _hand_maps()
    mate = torch.tensor([1])
    gold = torch.tensor([2])
    pred = torch.tensor([2])  # same as gold — not an error row
    mask = X.mate_recoverable_mask(mate, gold, pred, opener_char, closer_char)
    assert bool(mask[0].item()) is False


def test_mate_recoverable_either_closer_surface_variant():
    """Char-collapsing: bare ')' and ' )' both match mate-forced ')'."""
    opener_char, closer_char = _hand_maps()
    mate = torch.tensor([1, 1])  # '('
    gold = torch.tensor([2, 8])  # bare ')' and space-prefixed ' )'
    pred = torch.tensor([9, 9])
    mask = X.mate_recoverable_mask(mate, gold, pred, opener_char, closer_char)
    assert bool(mask[0].item()) is True
    assert bool(mask[1].item()) is True


def test_mate_not_recoverable_no_unclosed_opener():
    opener_char, closer_char = _hand_maps()
    mate = torch.tensor([-1])
    gold = torch.tensor([2])
    pred = torch.tensor([9])
    mask = X.mate_recoverable_mask(mate, gold, pred, opener_char, closer_char)
    assert bool(mask[0].item()) is False


def test_mate_feature_feeds_recoverable_on_hand_window():
    """End-to-end with real mate_feature on a tiny window ending with unclosed '('."""
    is_open, is_close = _is_open_close()
    opener_char, closer_char = _hand_maps()
    # window: pad... then '(' at end → innermost unclosed opener is '('
    ids = torch.zeros(1, 8, dtype=torch.long)
    ids[0, -1] = 1  # compact '('
    mate = v5.mate_feature(ids, is_open, is_close)
    assert int(mate[0].item()) == 1
    gold = torch.tensor([2])  # ')'
    pred = torch.tensor([9])
    mask = X.mate_recoverable_mask(mate, gold, pred, opener_char, closer_char)
    assert bool(mask[0].item()) is True


# ---------------------------------------------------------------------------
# depth_flag_row
# ---------------------------------------------------------------------------

def test_depth_flag_closer_at_depth_zero():
    is_open, is_close = _is_open_close()
    assert X.depth_flag_row(2, is_close, depth=0) is True  # ')' at depth 0


def test_depth_flag_closer_at_positive_depth():
    is_open, is_close = _is_open_close()
    assert X.depth_flag_row(2, is_close, depth=2) is False


def test_depth_flag_non_closer_pred():
    is_open, is_close = _is_open_close()
    assert X.depth_flag_row(9, is_close, depth=0) is False
    assert X.depth_flag_row(-1, is_close, depth=0) is False  # abstain


def test_depth_flag_fraction_aggregate():
    is_open, is_close = _is_open_close()
    pred = torch.tensor([2, 2, 9, -1])
    depths = torch.tensor([0, 1, 0, 0])
    frac = X.depth_flag_fraction(pred, depths, is_close)
    # only row 0 flagged
    assert frac == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# calibration_deciles
# ---------------------------------------------------------------------------

def test_calibration_deciles_monotonic_on_sorted_confidence():
    conf = torch.arange(1, 101, dtype=torch.float64)  # 1..100 sorted
    dec = X.calibration_deciles(conf)
    assert dec.shape == conf.shape
    # non-decreasing buckets
    diffs = dec[1:] - dec[:-1]
    assert bool((diffs >= 0).all().item())
    assert int(dec.min().item()) >= 1
    assert int(dec.max().item()) <= 10


# ---------------------------------------------------------------------------
# verdict / abort helpers reused from sudoku harness
# ---------------------------------------------------------------------------

def test_verdict_thresholds_untuned():
    import campaign_sudoku_forced_move as sudoku

    assert sudoku.verdict_from_gate(0.30) == "FIRES"
    assert sudoku.verdict_from_gate(0.29) == "MIDDLE (escalate)"
    assert sudoku.verdict_from_gate(0.10) == "MIDDLE (escalate)"
    assert sudoku.verdict_from_gate(0.09) == "DEAD"
    assert sudoku.abort_if_residual_too_small(0.019) is not None
    assert sudoku.abort_if_residual_too_small(0.02) is None


# ---------------------------------------------------------------------------
# Live path: skip-guarded (not part of required-green baseline)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO / "data" / "wyly_nexttoken_code_L256.pt").exists(),
    reason="code window corpus missing",
)
def test_live_probe_data_present_smoke():
    """Existence guard only — does NOT call v5.main() / run_cover_regeneration."""
    assert (REPO / "data" / "wyly_nexttoken_code_L256.pt").exists()
    # Module bound env is the code recipe when this test module imported the campaign.
    assert X.WYLY_ENV["WYLY_DS"] == "code"
    assert X.WYLY_ENV["WYLY_LABELS"] == "corpus"
