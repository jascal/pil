"""Pure-function tests for the indent-forcedness probe (no v5.main / no cover load)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Bind code recipe before importing modules that read WYLY_* at import time.
_WYLY_ENV = {
    "WYLY_TAG": "pythia70m",
    "WYLY_DS": "code",
    "WYLY_LIB": "mined",
    "WYLY_JUDGE": "cover",
    "WYLY_ONLINE": "1",
    "WYLY_COVER": "sw",
    "WYLY_LABELS": "corpus",
}
for _k, _v in _WYLY_ENV.items():
    os.environ.setdefault(_k, _v)

from experiments import campaign_indent_forcedness as X  # noqa: E402  # isort: skip
import wyly_lm_v5 as v5  # noqa: E402  # isort: skip


# ---------------------------------------------------------------------------
# Hand-built compact vocab fixtures (no real codec)
# Compact ids: 0='{', 1='}', 2=identifier-ish, 3='\n'-bearing, 4=spaces run
# ---------------------------------------------------------------------------

TOK_STR = {
    0: "{",
    1: "}",
    2: "foo",
    3: "\n",
    4: "    ",
    5: "();",
    6: "\t",
}


def _token_str_fn(compact_id: int) -> str:
    return TOK_STR[int(compact_id)]


def _is_open_close(vocab: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    is_open = torch.zeros(vocab, dtype=torch.bool)
    is_close = torch.zeros(vocab, dtype=torch.bool)
    is_open[0] = True  # '{'
    is_close[1] = True  # '}'
    return is_open, is_close


# ---------------------------------------------------------------------------
# 1. Case A — opener then fresh line, default unit
# ---------------------------------------------------------------------------


def test_case_a_default_unit_at_depth_one():
    """Context ends after '{\\n'; no prior indent step → default 4 spaces at depth 1."""
    is_open, is_close = _is_open_close()
    # tokens: '{' then '\n' → cumdepth [1, 1], d_end=1
    ids_row = torch.tensor([0, 3], dtype=torch.long)
    context_text = "{\n"
    predicted, meta = X.predict_structural_indent(
        context_text, ids_row, is_open, is_close, _token_str_fn
    )
    assert meta["case"] == "A"
    assert meta["d_end"] == 1
    assert meta["used_default"] is True
    assert meta["unit_char"] == " "
    assert meta["unit_size"] == 4
    assert predicted == "    "


# ---------------------------------------------------------------------------
# 2. Case C same-depth continuation
# ---------------------------------------------------------------------------


def test_case_c_same_depth_reproduces_completed_line_indent():
    """Context ends non-ws at depth 1 with d_end == d_line_start → '\\n' + line indent."""
    is_open, is_close = _is_open_close()
    # '{' '\n' '    ' 'foo' '();'  — no closer; depths stay 1 after opener
    ids_row = torch.tensor([0, 3, 4, 2, 5], dtype=torch.long)
    context_text = "{\n    foo();"
    predicted, meta = X.predict_structural_indent(
        context_text, ids_row, is_open, is_close, _token_str_fn
    )
    assert meta["case"] == "C-same-depth"
    assert meta["d_end"] == 1
    assert meta["d_line_start"] == 1
    assert predicted == "\n    "


# ---------------------------------------------------------------------------
# 3. Case C structural / causally-available dedent
# ---------------------------------------------------------------------------


def test_case_c_structural_dedent_on_completed_closer_line():
    """After a closer-only completed line, d_end=0 != d_line_start=1 → structural '\\n'."""
    is_open, is_close = _is_open_close()
    # '{' '\n' '    ' 'foo' '();' '\n' '}'
    ids_row = torch.tensor([0, 3, 4, 2, 5, 3, 1], dtype=torch.long)
    context_text = "{\n    foo();\n}"
    predicted, meta = X.predict_structural_indent(
        context_text, ids_row, is_open, is_close, _token_str_fn
    )
    assert meta["case"] == "C-structural"
    assert meta["d_end"] == 0
    assert meta["d_line_start"] == 1
    assert predicted == "\n"
    # Must NOT reproduce the previous body line's 4-space indent
    assert predicted != "\n    "


# ---------------------------------------------------------------------------
# 4. Naive baseline blindness on the same context
# ---------------------------------------------------------------------------


def test_naive_baseline_blindness_on_dedent():
    """Naive still re-emits prior body indent; does not structurally dedent."""
    context_text = "{\n    foo();\n}"
    naive = X.predict_naive_indent(context_text)
    assert naive == "\n    "
    # Structural rule (from test 3) would predict "\n" — lift over naive
    is_open, is_close = _is_open_close()
    ids_row = torch.tensor([0, 3, 4, 2, 5, 3, 1], dtype=torch.long)
    structural, _ = X.predict_structural_indent(
        context_text, ids_row, is_open, is_close, _token_str_fn
    )
    assert structural == "\n"
    assert structural != naive


# ---------------------------------------------------------------------------
# 5. Case B1 mid-run continuation
# ---------------------------------------------------------------------------


def test_case_b1_mid_run_continuation_remaining_spaces():
    """Target 8 spaces at depth 2 with 4 already present → remaining 4 spaces."""
    is_open, is_close = _is_open_close()
    # Two openers then newline then 4 spaces already typed
    # depths: { +1, { +1 → d_end=2 after second opener; \n keeps 2
    ids_row = torch.tensor([0, 0, 3, 4], dtype=torch.long)
    context_text = "{{\n    "  # 4 spaces already after newline; target = 8 default
    # unit default (no positive indent step among non-blank lines before the partial run)
    # non-blank lines: "{{" only before trailing partial — actually split:
    # lines = ['{{', '    '] — second line is all-ws so blank for unit inference
    # only one non-blank → used_default, unit_size=4, d_end=2 → target 8 spaces
    predicted, meta = X.predict_structural_indent(
        context_text, ids_row, is_open, is_close, _token_str_fn
    )
    assert meta["case"] == "B1"
    assert meta["d_end"] == 2
    assert meta["used_default"] is True
    assert meta["target_indent_chars"] == " " * 8
    assert predicted == "    "  # remaining 4


def test_case_b1_with_explicit_depth_two_and_partial_indent():
    """Same B1 geometry with an earlier indent step so unit is inferred, not default."""
    is_open, is_close = _is_open_close()
    # Build context with a clear 4-space step, then depth-2 fresh line with 4 of 8 typed.
    # Window ids still need d_end=2 and a newline before the partial run.
    context_text = "{\n    outer\n        "  # 4 already of target 8 if unit=4 depth=2
    # For unit inference: non-blank lines '{', '    outer' → increment 4
    # But d_end depends on ids. Use two opens.
    ids_row = torch.tensor([0, 3, 4, 2, 3, 4], dtype=torch.long)
    # Wait: only one '{'. Need d_end=2. Use two opens at start.
    ids_row = torch.tensor([0, 0, 3, 4, 2, 3, 4], dtype=torch.long)
    context_text = "{{\n    outer\n    "
    predicted, meta = X.predict_structural_indent(
        context_text, ids_row, is_open, is_close, _token_str_fn
    )
    assert meta["case"] == "B1"
    assert meta["d_end"] == 2
    assert meta["unit_size"] == 4
    assert meta["used_default"] is False
    assert predicted == "    "


# ---------------------------------------------------------------------------
# 6. Case B2 mid-line padding
# ---------------------------------------------------------------------------


def test_case_b2_mid_line_padding_declines():
    """Spaces not preceded by newline (and not window start) → empty prediction."""
    is_open, is_close = _is_open_close()
    ids_row = torch.tensor([2, 5], dtype=torch.long)
    context_text = "foo();   "
    predicted, meta = X.predict_structural_indent(
        context_text, ids_row, is_open, is_close, _token_str_fn
    )
    assert meta["case"] == "B2"
    assert predicted == ""


# ---------------------------------------------------------------------------
# 7. Unit inference
# ---------------------------------------------------------------------------


def test_unit_inference_two_space_steps():
    context = "main\n  if (x)\n    body\n"
    unit_char, unit_size, used_default = X.infer_indent_unit(context)
    assert unit_char == " "
    assert unit_size == 2
    assert used_default is False


def test_unit_inference_tab_steps():
    context = "main\n\tif (x)\n\t\tbody\n"
    unit_char, unit_size, used_default = X.infer_indent_unit(context)
    assert unit_char == "\t"
    assert unit_size == 1
    assert used_default is False


def test_unit_inference_default_when_no_positive_increment():
    context = "foo\nbar\nbaz"
    unit_char, unit_size, used_default = X.infer_indent_unit(context)
    assert unit_char == " "
    assert unit_size == 4
    assert used_default is True


def test_unit_inference_default_single_nonblank_line():
    unit_char, unit_size, used_default = X.infer_indent_unit("only_one_line")
    assert unit_char == " "
    assert unit_size == 4
    assert used_default is True


def test_unit_inference_mode_tie_breaks_to_smallest():
    # increments 2, 4, 2, 4 → both appear twice → smallest (2) wins
    context = "a\n  b\n      c\n        d\n            e"
    # lengths: 0, 2, 6, 8, 12 → incs: 2, 4, 2, 4
    unit_char, unit_size, used_default = X.infer_indent_unit(context)
    assert used_default is False
    assert unit_size == 2
    assert unit_char == " "


# ---------------------------------------------------------------------------
# Verdict thresholds (local 0.5 / 0.2)
# ---------------------------------------------------------------------------


def test_verdict_for_thresholds():
    assert X.verdict_for(0.5) == "FIRES"
    assert X.verdict_for(0.49) == "MIDDLE"
    assert X.verdict_for(0.2) == "MIDDLE"
    assert X.verdict_for(0.199) == "DEAD"
    assert X.verdict_for(0.0) == "DEAD"


def test_leading_ws_helper():
    assert X.leading_ws("    foo") == "    "
    assert X.leading_ws("\t\tbar") == "\t\t"
    assert X.leading_ws("nope") == ""
    assert X.leading_ws("  ") == "  "


def test_naive_indent_chars_most_recent_completed():
    assert X.naive_indent_chars("{\n    foo();\n}") == "    "
    assert X.naive_indent_chars("no_newline") == ""
    assert X.naive_indent_chars("a\n  b") == "a".replace("a", "")  # lines[-2] = 'a' → ''
    assert X.naive_indent_chars("a\n  b") == ""
    assert X.naive_indent_chars("a\n    b\n") == "    "


def test_structural_depths_matches_depth_feature_end():
    """d_end must match v5.depth_feature on the same 1-row batch."""
    is_open, is_close = _is_open_close()
    ids_row = torch.tensor([0, 3, 4, 2, 5, 3, 1], dtype=torch.long)
    d_end, d_line_start = X.structural_depths(ids_row, is_open, is_close, _token_str_fn)
    batch = ids_row.unsqueeze(0)
    feat = int(v5.depth_feature(batch, is_open, is_close, cap=8)[0].item())
    assert d_end == feat
    assert d_end == 0
    assert d_line_start == 1


# ---------------------------------------------------------------------------
# Live path: skip-guarded existence smoke only (no cover regeneration)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (REPO / "data" / "wyly_v5_mined_pythia70m_code_cov_ol_sw_corp.pt").exists(),
    reason="code cover state file missing",
)
def test_live_state_present_smoke():
    """Existence guard only — does NOT call v5.main() / run_cover_regeneration."""
    assert (REPO / "data" / "wyly_v5_mined_pythia70m_code_cov_ol_sw_corp.pt").exists()
    assert X.WYLY_ENV["WYLY_DS"] == "code"
    assert X.WYLY_ENV["WYLY_LABELS"] == "corpus"
    assert X.WYLY_ENV["WYLY_TAG"] == "pythia70m"
