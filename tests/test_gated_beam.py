"""Hard-term gate coverage for Rosetta's generic beam and text serving path."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))

from beam_engine import Path, Step, beam_decode  # noqa: E402
from serve_package import serve_energy, serve_sw  # noqa: E402


def _ngram(out: int, conf: float, cite: str, counts: tuple[int, int]) -> tuple:
    return (out, "observational", cite, conf, 1, "teacher", counts)


def test_zero_margin_text_beam_is_bit_exact_to_corner() -> None:
    ngrams = defaultdict(dict)
    ngrams[1][(1,)] = _ngram(2, 0.8, "first", (8, 10))
    ngrams[1][(2,)] = _ngram(3, 0.7, "second", (7, 10))

    corner = serve_energy((1,), [], ngrams, 1, M=1, beam_width=1)
    support_weighted = serve_sw((1,), [], ngrams, 1)
    assert support_weighted is not None
    support_weighted["cert_kind"] = "per-token"
    assert corner == support_weighted
    assert corner is not None
    assert corner["cert_kind"] == "per-token"

    for M in (2, 3, 4, 5):
        beam = serve_energy((1,), [], ngrams, 1, M=M, beam_width=8)
        assert beam == corner
        assert beam["cert_kind"] == "per-token"


class _ForcedCommitOracle:
    """One forced propagation step followed by carry steps."""

    def initial_state(self) -> dict[str, int | None]:
        return {"committed": None}

    def expand(self, state: dict) -> list[tuple[dict, list[Step]]]:
        if state["committed"] is not None:
            return [(state, [])]
        child = {"committed": 42}
        return [
            (child, [Step(key=("forced",), margin=1.0, counts=(1, 1))])
        ]

    def extract_commit(self, path: Path) -> int | None:
        return path.state["committed"]

    def fallback(self) -> int:
        return -1


def test_hard_term_signal_is_nonzero_on_committed_winner() -> None:
    result = beam_decode(
        _ForcedCommitOracle(),
        M=2,
        beam_width=4,
        rule_id="hard-term-test",
        wyly_seed=0,
    )
    assert result["committed_value"] == 42
    assert result["max_committed_margin"] > 0.0


class _MarginSequenceOracle:
    """Produce one exact winning trajectory with caller-supplied margins."""

    def __init__(self, margins: tuple[float, ...]) -> None:
        self.margins = margins

    def initial_state(self) -> int:
        return 0

    def expand(self, state: int) -> list[tuple[int, list[Step]]]:
        if state >= len(self.margins):
            return [(state, [])]
        step = Step(
            key=("sequence", state),
            margin=self.margins[state],
            counts=(state + 1, state + 1),
        )
        return [(state + 1, [step])]

    def extract_commit(self, path: Path) -> int:
        return path.state

    def fallback(self) -> int:
        return -1


def test_max_committed_margin_uses_full_winner_path_and_empty_default() -> None:
    mixed = beam_decode(
        _MarginSequenceOracle((0.0, 1.0, 0.0)),
        M=3,
        beam_width=2,
        rule_id="margin-sequence-test",
        wyly_seed=0,
    )
    assert mixed["committed_value"] == 3
    assert mixed["max_committed_margin"] == 1.0

    empty = beam_decode(
        _MarginSequenceOracle(()),
        M=2,
        beam_width=2,
        rule_id="empty-margin-test",
        wyly_seed=0,
    )
    assert empty["committed_value"] == 0
    assert empty["max_committed_margin"] == 0.0
