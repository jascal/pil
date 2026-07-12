"""Fast unit tests for the slice #98 pure surface classifier."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

from campaign_code_residual_anatomy import classify_surface  # noqa: E402


def test_bracket():
    assert classify_surface("(") == "bracket"


def test_terminator_separator():
    assert classify_surface(";") == "terminator_sep"


def test_whitespace_indent():
    assert classify_surface("\n    ") == "whitespace_indent"


def test_identifier():
    assert classify_surface("foo") == "identifier"


def test_keyword():
    assert classify_surface("return") == "keyword"


def test_literal():
    assert classify_surface("42") == "literal"


def test_operator():
    assert classify_surface("==") == "operator"
