"""Unit tests for the windowed count-table deprel labeler campaign."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.campaign_windowed_labeler import (  # noqa: E402
    UNARY_LEVEL_COUNT,
    WindowedDeprelLabeler,
    _coarsen,
    _structure_tuple,
    _token_shape,
    apply_preregistered_read,
    apply_through_line_case_read,
    diagnostic_dev_accuracy_vs_t_curve,
    diagnostic_windowed_fires_vs_unary,
    diagnostic_windowed_hit_rate,
    fit_windowed_backoff,
    full_windowed_keys,
    partition_deprel_accuracy,
    select_mt_on_dev,
    windowed_level_key,
)
from experiments.german_r3_codex import (  # noqa: E402
    _punctuation,
    _relation_key_levels,
    resolve_governor_index,
)

# ---------------------------------------------------------------------------
# 1. Windowed key builder
# ---------------------------------------------------------------------------


def test_windowed_level_key_shapes_and_boundaries() -> None:
    """level(m): boundaries, fixed gov ±1, structure matches unary convention."""
    tokens = ("Das", "Kind", "sieht", "den", "Hund")
    pos = ("DET", "NOUN", "VERB", "DET", "NOUN")
    # token 0 → +2 (VERB), token 1 → +1 (VERB), token 2 root, etc.
    offsets = (2, 1, 0, 1, -2)

    # Edge token index 0, radius 1: left cell is <BOS>, right is (NOUN, shape of Kind).
    key_m1 = windowed_level_key(tokens, pos, 0, offsets[0], radius=1)
    own_pos, dep_window, gov_window, structure = key_m1
    assert own_pos == "DET"
    assert len(dep_window) == 2  # ±1 excluding self
    assert dep_window[0] == "<BOS>"
    assert dep_window[1] == (pos[1], _token_shape(tokens[1]))
    # Governor of token 0 with offset +2 is index 2 (VERB).
    assert resolve_governor_index(0, 2, len(tokens)) == 2
    assert gov_window == (
        (pos[1], _token_shape(tokens[1])),  # g-1
        (pos[2], _token_shape(tokens[2])),  # g
        (pos[3], _token_shape(tokens[3])),  # g+1
    )
    assert structure == _structure_tuple(tokens, pos, 0, offsets[0])
    assert structure[0] == "RIGHT"
    assert structure[1] == 2
    assert structure[2] == "VERB"

    # Edge token last index, radius 2: right cells hit <EOS>.
    key_last = windowed_level_key(tokens, pos, 4, offsets[4], radius=2)
    _own, dep_last, gov_last, struct_last = key_last
    assert len(dep_last) == 4  # ±2 excluding self
    assert dep_last[-1] == "<EOS>"
    assert dep_last[-2] == "<EOS>"  # index 5 and 6 both out of range for size 5
    # Governor of token 4 with offset -2 is index 2.
    assert resolve_governor_index(4, -2, len(tokens)) == 2
    # gov_window is FIXED ±1 regardless of m (here m=2).
    assert gov_last == (
        (pos[1], _token_shape(tokens[1])),
        (pos[2], _token_shape(tokens[2])),
        (pos[3], _token_shape(tokens[3])),
    )
    assert struct_last[0] == "LEFT"

    # ROOT attachment: gov_window uses <ROOT> markers; structure is SELF.
    key_root = windowed_level_key(tokens, pos, 2, 0, radius=1)
    _o, _d, gov_root, struct_root = key_root
    assert gov_root == ("<BOS>", "<ROOT>", "<EOS>")
    assert struct_root == ("SELF", 0, pos[2])  # unary self-loop governor_pos

    # Governor window identical at m=1 and m=3 for the same (i, offset).
    k1 = windowed_level_key(tokens, pos, 1, offsets[1], radius=1)
    k3 = windowed_level_key(tokens, pos, 1, offsets[1], radius=3)
    assert k1[2] == k3[2]
    assert len(k1[1]) == 2
    assert len(k3[1]) == 6

    # Shape never carries raw form; punctuation convention matches shipped.
    assert _token_shape(".") == (False, False, _punctuation("."))
    assert _token_shape("Hund") == (True, False, False)


def test_full_keys_append_shipped_unary_levels() -> None:
    tokens = ("A", "B", "C", "D")
    pos = ("DET", "NOUN", "VERB", "NOUN")
    index, offset, m = 1, 1, 2
    full = full_windowed_keys(tokens, pos, index, offset, m)
    assert len(full) == m + UNARY_LEVEL_COUNT
    # Windowed: level(2), level(1)
    assert full[0] == windowed_level_key(tokens, pos, index, offset, 2)
    assert full[1] == windowed_level_key(tokens, pos, index, offset, 1)
    # Trailing levels are exactly the shipped unary hierarchy.
    unary = _relation_key_levels(tokens, pos, index, offset)
    assert full[m:] == unary
    assert len(unary) == UNARY_LEVEL_COUNT


# ---------------------------------------------------------------------------
# 2. Backoff-by-T
# ---------------------------------------------------------------------------


def test_backoff_by_t_skips_undersupport_windowed_level() -> None:
    """A windowed level with count < T is skipped; count ≥ T is used."""
    # Two windowed levels (M=2) + 6 unary pads.  Level-0 key appears once;
    # level-1 key appears enough times with a different label.
    rare_key = ("RARE",)
    common_key = ("COMMON",)
    unary_pad = tuple(f"u{i}" for i in range(UNARY_LEVEL_COUNT))

    def pack(level0: object, level1: object) -> tuple:
        return (level0, level1, *unary_pad)

    rows_t5 = [
        (pack(rare_key, common_key), "from_rare"),
        *[(pack(("other",), common_key), "from_common") for _ in range(5)],
    ]
    # T=5: rare_key count=1 < 5 → not entered; common_key count=6 ≥ 5 → entered.
    tables = fit_windowed_backoff(rows_t5, hierarchy_radius=2, minsupp_windowed=5)
    query = pack(rare_key, common_key)
    label, source = tables.predict_with_source(query)
    assert label == "from_common"
    assert source == 1  # fell through T-gated level 0

    # With T=1 the rare level is kept.
    tables_t1 = fit_windowed_backoff(rows_t5, hierarchy_radius=2, minsupp_windowed=1)
    label_t1, source_t1 = tables_t1.predict_with_source(query)
    assert label_t1 == "from_rare"
    assert source_t1 == 0


# ---------------------------------------------------------------------------
# 3. REQUIRED populated-table regression (first-pass position-0 bug)
# ---------------------------------------------------------------------------


def test_populated_table_predict_uses_each_token_own_index() -> None:
    """Populated multi-level tables: non-zero indices must not get position-0 labels.

    Recreates the first-pass bug scenario: a broken implementation that built
    keys as if every token were at index 0 (e.g. by slicing head_offsets to
    length 1 and re-enumerating) would return the wrong label for at least one
    non-zero-index token.  Empty-table stubs are NOT used — both windowed and
    unary levels carry real counts that would fire under a wrong key.
    """
    tokens = ("Das", "Kind", "sieht", "den", "Hund")
    pos = ("DET", "NOUN", "VERB", "DET", "NOUN")
    # Distinct non-zero positions 1 and 4 with different governors/offsets.
    offsets = (2, 1, 0, 1, -2)
    m = 1

    # Build the *correct* full keys for positions 0, 1, and 4.
    keys_0 = full_windowed_keys(tokens, pos, 0, offsets[0], m)
    keys_1 = full_windowed_keys(tokens, pos, 1, offsets[1], m)
    keys_4 = full_windowed_keys(tokens, pos, 4, offsets[4], m)
    # Sanity: positions 1 and 4 differ at the windowed level (and unary).
    assert keys_1[0] != keys_4[0]
    assert keys_1[m] != keys_4[m]
    # Position-0 key also differs from both (so a pos-0 bug is detectable).
    assert keys_0[0] != keys_1[0]
    assert keys_0[0] != keys_4[0]

    # Populate BOTH windowed (level 0) and unary (level m..) with enough
    # support that T=1 keeps every entry.  Distinct labels per position key.
    label_for = {
        keys_0: "det",
        keys_1: "nsubj",
        keys_4: "obj",
    }
    # Repeat each row so minsupp is comfortably satisfied at every level.
    rows: list[tuple[tuple, str]] = []
    for keys, label in label_for.items():
        for _ in range(5):
            rows.append((keys, label))
    # Also add rows that would be produced if someone incorrectly keyed every
    # token as position 0 with that token's offset — those get a distinct
    # poison label so a position-0 bug is unambiguous.
    poison = "POISON_POS0"
    for index, offset in enumerate(offsets):
        if index == 0:
            continue
        wrong_keys = full_windowed_keys(tokens, pos, 0, offset, m)
        # Only poison if this wrong key is not one of our real keys.
        if wrong_keys not in label_for:
            for _ in range(5):
                rows.append((wrong_keys, poison))

    labeler = WindowedDeprelLabeler(hierarchy_radius=m, minsupp_windowed=1)
    labeler.fit_from_rows(rows)

    predicted, sources = labeler.predict_with_source(tokens, pos, offsets)
    assert len(predicted) == len(tokens)

    # Position 1 and 4 must get their OWN labels, not position 0's "det"
    # and not the poison label a pos-0-keyed bug would prefer.
    assert predicted[1] == "nsubj", (
        f"token 1 got {predicted[1]!r} (source={sources[1]}); "
        "a position-0 keying bug would mislabel non-zero indices"
    )
    assert predicted[4] == "obj", (
        f"token 4 got {predicted[4]!r} (source={sources[4]}); "
        "a position-0 keying bug would mislabel non-zero indices"
    )
    assert predicted[0] == "det"
    assert predicted[1] != predicted[4]
    assert poison not in (predicted[1], predicted[4])
    # Served from windowed level (index 0) given we populated it.
    assert sources[1] == 0
    assert sources[4] == 0

    # Direct table path: each token's own full_keys yield its label.
    tables = labeler.tables
    assert tables is not None
    assert tables.predict(keys_1) == "nsubj"
    assert tables.predict(keys_4) == "obj"
    # Cross-check: predicting keys_0 under a wrong "always index 0" would
    # not equal keys_1's label.
    assert tables.predict(keys_0) == "det"
    assert tables.predict(keys_0) != tables.predict(keys_1)


def test_predict_signature_matches_predict_deprels_pattern() -> None:
    signature = inspect.signature(WindowedDeprelLabeler.predict)
    assert set(signature.parameters) == {
        "self",
        "tokens",
        "predicted_pos",
        "head_offsets",
    }
    assert "gold" not in " ".join(signature.parameters)


# ---------------------------------------------------------------------------
# 4. DEV-vs-TEST isolation
# ---------------------------------------------------------------------------


def test_select_mt_on_dev_signature_has_no_test_parameter() -> None:
    """Sweep/selection only ever consumes DEV-labeled inputs."""
    params = inspect.signature(select_mt_on_dev).parameters
    for name in params:
        assert "test" not in name.lower(), f"DEV sweep must not take {name}"
    # Required parameters include train + dev, not test.
    assert "dev" in params
    assert "dev_heads" in params
    assert "dev_pos" in params
    assert "train" in params
    assert "test" not in params
    assert "test_heads" not in params


# ---------------------------------------------------------------------------
# 5. Partition metric
# ---------------------------------------------------------------------------


def test_partition_metric_known_split() -> None:
    gold = (
        "nsubj",
        "obj",
        "obl:arg",
        "det",
        "amod",
        "case",
        "root",
        "punct",
    )
    predicted = (
        "nsubj",
        "iobj",
        "obl:arg",
        "det",
        "amod",
        "nmod",
        "root",
        "punct",
    )
    result = partition_deprel_accuracy(predicted, gold)
    assert result["pairwise_sensitive_n"] == 3
    assert result["pairwise_sensitive_accuracy"] == pytest.approx(2 / 3)
    assert result["local_n"] == 4
    assert result["local_accuracy"] == pytest.approx(3 / 4)


def test_coarsen_strips_subtype() -> None:
    assert _coarsen("nsubj:pass") == "nsubj"
    assert _coarsen("obl:arg") == "obl"
    assert _coarsen("det") == "det"


# ---------------------------------------------------------------------------
# 6. §5 read verdicts
# ---------------------------------------------------------------------------


def test_read_verdict_fires() -> None:
    read = apply_preregistered_read(
        unary_las=0.50,
        windowed_las=0.54,
        unary_case=0.75,
        windowed_case=0.78,
        partition_gains={"pairwise_sensitive": 0.08, "local": 0.02},
        dev_M=2,
        dev_T=10,
    )
    assert read["verdict"] == "FIRES"
    assert read["deprel_las_gain"] == pytest.approx(0.04)
    assert read["dev_M"] == 2
    assert read["dev_T"] == 10
    assert "FIRES" in read["text"]


def test_read_verdict_halted() -> None:
    read = apply_preregistered_read(
        unary_las=0.50,
        windowed_las=0.48,
        unary_case=0.75,
        windowed_case=0.74,
        partition_gains={"pairwise_sensitive": -0.01, "local": 0.0},
        dev_M=1,
        dev_T=5,
    )
    assert read["verdict"] == "HALTED"
    assert read["deprel_las_gain"] < 0
    assert "HALTED" in read["text"]


def test_read_verdict_in_between_small_gain() -> None:
    read = apply_preregistered_read(
        unary_las=0.50,
        windowed_las=0.515,
        unary_case=0.75,
        windowed_case=0.77,
        partition_gains={"pairwise_sensitive": 0.03, "local": 0.01},
        dev_M=1,
        dev_T=20,
    )
    assert read["verdict"] == "IN-BETWEEN"
    assert 0.0 < read["deprel_las_gain"] < 0.03
    assert "IN-BETWEEN" in read["text"]


def test_read_verdict_in_between_diffuse_partition() -> None:
    # LAS gain high and case improves, but partition is diffuse.
    read = apply_preregistered_read(
        unary_las=0.50,
        windowed_las=0.55,
        unary_case=0.75,
        windowed_case=0.80,
        partition_gains={"pairwise_sensitive": 0.01, "local": 0.04},
        dev_M=3,
        dev_T=50,
    )
    assert read["verdict"] == "IN-BETWEEN"
    assert "diffuse" in read["text"]


def test_read_verdict_in_between_no_case_improvement() -> None:
    read = apply_preregistered_read(
        unary_las=0.50,
        windowed_las=0.55,
        unary_case=0.78,
        windowed_case=0.78,
        partition_gains={"pairwise_sensitive": 0.08, "local": 0.02},
        dev_M=2,
        dev_T=10,
    )
    assert read["verdict"] == "IN-BETWEEN"
    assert "case-cascade" in read["text"]


def test_through_line_case_plateau_language() -> None:
    deliverable = apply_through_line_case_read(0.91)
    plateau = apply_through_line_case_read(0.80)
    diagnose = apply_through_line_case_read(0.74)

    assert deliverable["verdict"] == "deliverable_met"
    assert plateau["verdict"] == "plateau"
    assert "plateau" in plateau["text"]
    assert "irreducible" not in plateau["text"].lower()
    assert "required" not in plateau["text"].lower()
    assert diagnose["verdict"] == "diagnose"


# ---------------------------------------------------------------------------
# 7. Diagnostics on synthetic inputs
# ---------------------------------------------------------------------------


def test_diagnostic_windowed_hit_rate() -> None:
    # M=2: sources 0,1 are windowed; 2+ and None are unary/fallback.
    sources: list[int | None] = [0, 1, 2, None, 0, 3]
    result = diagnostic_windowed_hit_rate(sources, hierarchy_radius=2)
    assert result["tokens"] == 6
    assert result["windowed_hits"] == 3
    assert result["unary_or_fallback"] == 3
    assert result["windowed_hit_rate"] == pytest.approx(0.5)


def test_diagnostic_dev_accuracy_vs_t_curve_shape() -> None:
    # Mild convergence toward unary as T rises (not a large drop).
    sweep = [
        {"M": 1, "T": 5, "dev_deprel_only_accuracy": 0.700},
        {"M": 1, "T": 10, "dev_deprel_only_accuracy": 0.698},
        {"M": 1, "T": 20, "dev_deprel_only_accuracy": 0.697},
        {"M": 1, "T": 50, "dev_deprel_only_accuracy": 0.696},
        {"M": 2, "T": 5, "dev_deprel_only_accuracy": 0.702},
        {"M": 2, "T": 10, "dev_deprel_only_accuracy": 0.699},
        {"M": 2, "T": 20, "dev_deprel_only_accuracy": 0.697},
        {"M": 2, "T": 50, "dev_deprel_only_accuracy": 0.696},
    ]
    result = diagnostic_dev_accuracy_vs_t_curve(
        sweep, m_grid=(1, 2), t_grid=(5, 10, 20, 50), unary_dev_baseline=0.696
    )
    assert "1" in result["by_M"] and "2" in result["by_M"]
    assert len(result["by_M"]["1"]) == 4
    assert result["by_M"]["1"][0]["T"] == 5
    assert result["suspect"] is False
    assert result["unary_dev_baseline"] == pytest.approx(0.696)

    # Falling curve is flagged suspect.
    falling = [
        {"M": 1, "T": 5, "dev_deprel_only_accuracy": 0.80},
        {"M": 1, "T": 10, "dev_deprel_only_accuracy": 0.70},
        {"M": 1, "T": 20, "dev_deprel_only_accuracy": 0.60},
        {"M": 1, "T": 50, "dev_deprel_only_accuracy": 0.50},
    ]
    bad = diagnostic_dev_accuracy_vs_t_curve(
        falling, m_grid=(1,), t_grid=(5, 10, 20, 50)
    )
    assert bad["suspect"] is True
    assert "FALLS" in bad["text"]


def test_diagnostic_windowed_fires_vs_unary_on_subset() -> None:
    # Tokens 0,2 served by windowed (src < M=2); 1,3 by unary/fallback.
    windowed = ("nsubj", "det", "obj", "punct")
    unary = ("nmod", "det", "iobj", "punct")
    gold = ("nsubj", "det", "obj", "punct")
    sources: list[int | None] = [0, 2, 1, None]
    result = diagnostic_windowed_fires_vs_unary(
        windowed, unary, gold, sources, hierarchy_radius=2
    )
    # Subset S = indices 0 and 2.
    assert result["subset_n"] == 2
    assert result["windowed_accuracy_on_S"] == pytest.approx(1.0)  # both hit
    assert result["unary_accuracy_on_S"] == pytest.approx(0.0)  # both miss


def test_fit_windowed_backoff_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_windowed_backoff([], hierarchy_radius=1, minsupp_windowed=5)


def test_count_table_hand_construction_still_backs_off() -> None:
    """Hand-built CountTable at windowed level with entry still respects source index."""
    # Minimal: one windowed + 6 unary levels; only unary level 0 of the unary
    # block has an entry for a specific key chain.
    m = 1
    windowed_key = ("NOUN", (("<BOS>",),), ("<BOS>", "<ROOT>", "<EOS>"), ("SELF", 0, "VERB"))
    unary_keys = tuple(f"u{i}" for i in range(UNARY_LEVEL_COUNT))
    full = (windowed_key, *unary_keys)
    # Put support only on the last unary level so we fall all the way through.
    rows = [(full, "root") for _ in range(3)]
    tables = fit_windowed_backoff(rows, hierarchy_radius=m, minsupp_windowed=5)
    # Windowed level has count 3 < T=5 → no entry; unary levels have minsupp=1.
    label, source = tables.predict_with_source(full)
    assert label == "root"
    assert source is not None and source >= m
