"""Pure-function tests for slice #90 legality feature + naked-vs-hidden pinning.

No v5.main(), no GPU. Hand-built fixtures only (style of tests/test_sudoku_forced_move.py).

Scoping note on real-sample parity: the >=500-row parity check against #89's classify_forced
is asserted inside experiments/campaign_legality_pinning.py (campaign e2e on the real dataset).
This pytest file covers fixture-level parity + freeze/strata/conf pure logic as a fast
regression suite without corpus/codec/GPU.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_legality_pinning as X  # noqa: E402  # isort: skip
from experiments import campaign_sudoku_forced_move as osc  # noqa: E402  # isort: skip


# ---------------------------------------------------------------------------
# Hand-built grids (same classification cases as #89 tests)
# ---------------------------------------------------------------------------

def _empty() -> list[list[int]]:
    return [[0] * 9 for _ in range(9)]


def _naked_not_hidden_grid() -> tuple[list[list[int]], int, int, int]:
    """Naked single at (0,0)=1 that is NOT a hidden single (digit still legal elsewhere)."""
    g = _empty()
    for c, d in enumerate(range(3, 10), start=2):
        g[0][c] = d
    g[1][0] = 2
    return g, 0, 0, 1


def _hidden_not_naked_grid() -> tuple[list[list[int]], int, int, int]:
    """Hidden single at (0,0)=1 with candidates {1,2} (not naked)."""
    g = _empty()
    for c, d in enumerate(range(3, 10), start=2):
        g[0][c] = d
    g[5][1] = 1  # block 1 from (0,1) via col 1
    return g, 0, 0, 1


def _non_forced_grid() -> tuple[list[list[int]], int, int, int]:
    g = _empty()
    return g, 4, 4, 5


def _as_batch(
    grids: list[list[list[int]]],
    rows: list[int],
    cols: list[int],
    true_vs: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(grids, dtype=torch.long),
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(cols, dtype=torch.long),
        torch.tensor(true_vs, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# 1. Feature parity vs #89 oracle (hand-built fixtures)
# ---------------------------------------------------------------------------

def test_feature_parity_hand_built_fixture():
    """Tensor is_naked / is_hidden / forced_value agree with #89 on a handful of grids."""
    cases = [
        _naked_not_hidden_grid(),
        _hidden_not_naked_grid(),
        _non_forced_grid(),
    ]
    # also a classic naked-from-row case
    g4 = _empty()
    for c, d in enumerate(range(2, 10), start=1):
        g4[0][c] = d
    cases.append((g4, 0, 0, 1))

    grids, rows, cols, tvs = _as_batch(
        [c[0] for c in cases],
        [c[1] for c in cases],
        [c[2] for c in cases],
        [c[3] for c in cases],
    )
    result = X.check_parity_batch(grids, rows, cols, tvs)
    assert result["n_checked"] == len(cases)
    assert result["n_mismatches"] == 0, result["mismatches"]
    assert result["passed"] is True


def test_real_sample_parity_scoped_to_campaign():
    """Document that >=500-row real-dataset parity lives in the campaign script, not pytest."""
    src = (REPO / "experiments" / "campaign_legality_pinning.py").read_text()
    assert "PARITY_MIN_ROWS" in src
    assert "check_parity_batch" in src
    assert "classify_forced" in src or "parity" in src.lower()


# ---------------------------------------------------------------------------
# 2. naked / hidden / forced_value correctness on hand-built grids
# ---------------------------------------------------------------------------

def test_naked_not_hidden_tensor():
    g, r, c, tv = _naked_not_hidden_grid()
    # oracle preconditions
    assert osc.cell_candidates(g, r, c) == {1}
    assert osc.is_naked_single(g, r, c) is True
    assert osc.is_hidden_single(g, r, c, tv) is False

    grids, rows, cols, tvs = _as_batch([g], [r], [c], [tv])
    feat = X.legality_feature_tensor(grids, rows, cols, tvs)
    assert bool(feat["is_naked"][0]) is True
    assert bool(feat["is_hidden"][0]) is False
    assert int(feat["forced_value"][0]) == 1
    assert int(feat["n_candidates"][0]) == 1


def test_hidden_not_naked_tensor():
    g, r, c, tv = _hidden_not_naked_grid()
    assert osc.cell_candidates(g, r, c) == {1, 2}
    assert osc.is_naked_single(g, r, c) is False
    assert osc.is_hidden_single(g, r, c, tv) is True

    grids, rows, cols, tvs = _as_batch([g], [r], [c], [tv])
    feat = X.legality_feature_tensor(grids, rows, cols, tvs)
    assert bool(feat["is_naked"][0]) is False
    assert bool(feat["is_hidden"][0]) is True
    assert int(feat["forced_value"][0]) == 1
    assert int(feat["n_candidates"][0]) == 2


def test_non_forced_tensor():
    g, r, c, tv = _non_forced_grid()
    assert osc.is_naked_single(g, r, c) is False
    assert osc.is_hidden_single(g, r, c, tv) is False

    grids, rows, cols, tvs = _as_batch([g], [r], [c], [tv])
    feat = X.legality_feature_tensor(grids, rows, cols, tvs)
    assert bool(feat["is_naked"][0]) is False
    assert bool(feat["is_hidden"][0]) is False
    assert int(feat["forced_value"][0]) == -1
    assert int(feat["n_candidates"][0]) == 9


# ---------------------------------------------------------------------------
# 3. Validity-floor path
# ---------------------------------------------------------------------------

def test_validity_floor_underpowered():
    """Absolute union-recovered count < 50 -> UNDERPOWERED -- ESCALATE, no freeze."""
    # 40 union-recovered out of 100 errors: fractions fine, absolute count too low
    n_union = 40
    n_naked = 38
    n_err = 100
    rec_n = n_naked / n_err
    rec_u = n_union / n_err
    decision = X.freeze_from_recovery(rec_n, rec_u, n_union)
    assert decision["underpowered"] is True
    assert decision["message"] == "UNDERPOWERED -- ESCALATE"
    assert decision["freeze_decided"] is False
    assert decision["inventory"] is None


def test_validity_floor_message_printed_verbatim(capsys):
    decision = X.freeze_from_recovery(0.9, 0.95, n_recovered_union=10)
    print(decision["message"], flush=True)
    out = capsys.readouterr().out
    assert "UNDERPOWERED -- ESCALATE" in out


# ---------------------------------------------------------------------------
# 4. Freeze-rule truth table
# ---------------------------------------------------------------------------

def test_freeze_naked_only():
    """recovered_naked >= 0.90 * recovered_union -> NAKED-ONLY."""
    # 95/100 naked, 100/100 union: 0.95 >= 0.90 * 1.0
    d = X.freeze_from_recovery(0.95, 1.0, n_recovered_union=100)
    assert d["underpowered"] is False
    assert d["freeze_decided"] is True
    assert d["inventory"] == "NAKED-ONLY"
    assert d["naked_suffices"] is True


def test_freeze_naked_union_hidden():
    """recovered_naked < 0.90 * recovered_union -> NAKED UNION HIDDEN."""
    # 0.70 naked, 1.00 union: 0.70 < 0.90
    d = X.freeze_from_recovery(0.70, 1.0, n_recovered_union=200)
    assert d["underpowered"] is False
    assert d["freeze_decided"] is True
    assert d["inventory"] == "NAKED UNION HIDDEN"
    assert d["naked_suffices"] is False


def test_freeze_boundary_exactly_90pct():
    """Exactly at 0.90 * union -> NAKED-ONLY (>=)."""
    d = X.freeze_from_recovery(0.90, 1.0, n_recovered_union=100)
    assert d["inventory"] == "NAKED-ONLY"


# ---------------------------------------------------------------------------
# 5. Reveal-third stratification arithmetic
# ---------------------------------------------------------------------------

def test_stratify_reveal_third_arithmetic():
    """cell_index // 27 bucketing + recovered fractions/counts per third."""
    # thirds: 0 (idx 0-26), 1 (27-53), 2 (54-80)
    cell_index = [
        0, 13, 26,   # third 0
        27, 40, 53,  # third 1
        54, 67, 80,  # third 2
    ]
    # all treated as error rows
    is_naked = [True, True, False, True, False, False, True, True, True]
    is_hidden = [False, True, True, False, True, False, False, False, True]
    # forced_value == gold for all except last of each third pattern:
    # third0: rows 0,1 recovered naked; row2 union only (hidden)
    # -> naked rec: 2/3, union: 3/3
    forced_eq = [True, True, True, True, True, False, True, True, True]
    # third1: idx3 naked+eq, idx4 hidden+eq, idx5 neither eq
    # -> naked 1/3, union 2/3
    # third2: all three naked+eq (idx8 also hidden)
    # -> naked 3/3, union 3/3

    strata = X.stratify_by_reveal_third(
        cell_index, is_naked, is_hidden, forced_eq, error_mask=None,
    )
    assert set(strata.keys()) == {0, 1, 2}

    s0 = strata[0]
    assert s0["n_error"] == 3
    assert s0["n_recovered_naked"] == 2
    assert s0["n_recovered_union"] == 3
    assert s0["recovered_naked"] == pytest.approx(2 / 3)
    assert s0["recovered_union"] == pytest.approx(1.0)

    s1 = strata[1]
    assert s1["n_error"] == 3
    assert s1["n_recovered_naked"] == 1
    assert s1["n_recovered_union"] == 2
    assert s1["recovered_naked"] == pytest.approx(1 / 3)
    assert s1["recovered_union"] == pytest.approx(2 / 3)

    s2 = strata[2]
    assert s2["n_error"] == 3
    assert s2["n_recovered_naked"] == 3
    assert s2["n_recovered_union"] == 3
    assert s2["recovered_naked"] == pytest.approx(1.0)
    assert s2["recovered_union"] == pytest.approx(1.0)


def test_stratify_respects_error_mask():
    cell_index = [0, 27, 54]
    is_naked = [True, True, True]
    is_hidden = [False, False, False]
    forced_eq = [True, True, True]
    error_mask = [True, False, True]  # middle (third 1) not an error
    strata = X.stratify_by_reveal_third(
        cell_index, is_naked, is_hidden, forced_eq, error_mask=error_mask,
    )
    assert strata[0]["n_error"] == 1
    assert strata[1]["n_error"] == 0
    assert strata[2]["n_error"] == 1


def test_prenamed_reading_condition():
    assert X.prenamed_reading_holds(0.95, 0.50, n_error_third0=10) is True
    assert X.prenamed_reading_holds(0.95, 0.95, n_error_third0=10) is False
    assert X.prenamed_reading_holds(0.80, 0.50, n_error_third0=10) is False
    # empty first third must not vacuously fire the pre-named reading
    assert X.prenamed_reading_holds(0.95, 0.0, n_error_third0=0) is False
    # backward-compatible: omitting n_error_third0 keeps the pure ratio check
    assert X.prenamed_reading_holds(0.95, 0.50) is True


# ---------------------------------------------------------------------------
# 6. CONF_FN interface
# ---------------------------------------------------------------------------

def test_conf_fn_returns_forced_value_or_minus_one():
    """Registered proposer-style fn returns forced_value where it fires, else -1."""
    cases = [
        _naked_not_hidden_grid(),
        _hidden_not_naked_grid(),
        _non_forced_grid(),
    ]
    grids, rows, cols, tvs = _as_batch(
        [c[0] for c in cases],
        [c[1] for c in cases],
        [c[2] for c in cases],
        [c[3] for c in cases],
    )
    out = X.conf_fn_forced_digits(grids, rows, cols, tvs)
    assert out.shape == (3,)
    assert int(out[0]) == 1   # naked fires
    assert int(out[1]) == 1   # hidden fires
    assert int(out[2]) == -1  # non-forced does not fire


def test_compute_recovery_counts():
    is_naked = [True, False, True, False]
    is_hidden = [False, True, True, False]
    forced = [5, 5, 5, 5]
    gold = [5, 5, 7, 5]  # row2 forced != gold
    err = [True, True, True, False]
    rec = X.compute_recovery(is_naked, is_hidden, forced, gold, error_mask=err)
    # err rows 0,1,2:
    # 0: naked & match -> naked+union
    # 1: hidden & match -> union
    # 2: naked&hidden but forced!=gold -> neither
    assert rec["n_error"] == 3
    assert rec["n_recovered_naked"] == 1
    assert rec["n_recovered_union"] == 2
    assert rec["recovered_naked"] == pytest.approx(1 / 3)
    assert rec["recovered_union"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# 7. No-natural-text / no-wikitext structural guard
# ---------------------------------------------------------------------------

def test_no_natural_text_guard():
    """Source must not reference wikitext or other natural-language domains.

    The required FRAMING_NOTE uses the words 'natural-text' by design (Condition 3
    framing). Those occurrences are exempt; everything else must not match.
    """
    path = REPO / "experiments" / "campaign_legality_pinning.py"
    src = path.read_text()
    assert re.search(r"wikitext", src, flags=re.IGNORECASE) is None

    framing = X.FRAMING_NOTE
    assert "natural-text" in framing  # required verbatim phrase
    stripped = src
    # remove all occurrences of the required framing constant/value
    stripped = stripped.replace(framing, "")
    # also remove the FRAMING_NOTE assignment artifacts if any residual fragments
    assert re.search(r"natural", stripped, flags=re.IGNORECASE) is None


def test_framing_note_verbatim():
    assert X.FRAMING_NOTE == (
        "domain-solving numbers, NOT natural-text; no natural-text comparison in this slice."
    )


def test_import_does_not_call_main():
    assert hasattr(X, "legality_feature_tensor")
    assert hasattr(X, "freeze_from_recovery")
    assert X.OUTPUT.name == "legality_pinning.json"


def test_honesty_note_mentions_computed_not_certified():
    assert "computed-not-certified" in X.HONESTY_NOTE
    assert "LABELS=corpus" in X.HONESTY_NOTE


# ---------------------------------------------------------------------------
# 8. Reveal-index distribution (measurement/reporting amendment)
# ---------------------------------------------------------------------------

def test_reveal_index_distribution_mixed_thirds():
    """min/max/median + histogram fractions on a small mixed set of cell_index values."""
    # thirds: 0 [0-26], 1 [27-53], 2 [54-80]
    cell_index = [0, 13, 26, 27, 40, 54, 67, 80]
    dist = X.reveal_index_distribution(cell_index)
    assert dist["n"] == 8
    assert dist["min"] == 0
    assert dist["max"] == 80
    # median of sorted [0,13,26,27,40,54,67,80] = avg(27,40) = 33.5
    assert dist["median"] == pytest.approx(33.5)
    assert set(dist["histogram"].keys()) == {"0", "1", "2"}
    assert dist["histogram"]["0"]["count"] == 3
    assert dist["histogram"]["0"]["fraction"] == pytest.approx(3 / 8)
    assert dist["histogram"]["1"]["count"] == 2
    assert dist["histogram"]["1"]["fraction"] == pytest.approx(2 / 8)
    assert dist["histogram"]["2"]["count"] == 3
    assert dist["histogram"]["2"]["fraction"] == pytest.approx(3 / 8)


def test_reveal_index_distribution_all_last_third():
    """Skewed case matching real data: every index in third 2 ([54-80])."""
    cell_index = [59, 60, 67, 72, 80]
    dist = X.reveal_index_distribution(cell_index)
    assert dist["n"] == 5
    assert dist["min"] == 59
    assert dist["max"] == 80
    assert dist["median"] == pytest.approx(67.0)
    assert dist["histogram"]["0"]["count"] == 0
    assert dist["histogram"]["0"]["fraction"] == pytest.approx(0.0)
    assert dist["histogram"]["1"]["count"] == 0
    assert dist["histogram"]["1"]["fraction"] == pytest.approx(0.0)
    assert dist["histogram"]["2"]["count"] == 5
    assert dist["histogram"]["2"]["fraction"] == pytest.approx(1.0)


def test_reveal_index_note_verbatim():
    assert X.REVEAL_INDEX_NOTE == (
        "the sudoku dataset presents ONLY late-reveal solution cells (index ~59-80); "
        "early/mid cells never appear as prediction targets, so the register is measured "
        "only on the near-solved sub-task and its capability on early/mid cells is "
        "UNTESTED by this data."
    )
    # also present verbatim in the JSON honesty note field
    assert X.REVEAL_INDEX_NOTE in X.HONESTY_NOTE
