"""Mine deployment-shaped (prompt, answer) pairs from corpus inline Q/A (P2).

bAbI corpora are rendered with inline answers::

    ... Mary went to the garden. Q: Where is the football? A: garden. ...

Judge queries need the prompt ending at ``A:`` (no answer) and a held-out split so
admission is not scored on the same story spans used for count tables.

Temporal holdout: last ``holdout_frac`` of Q/A events (by corpus order) become the
judge set; earlier events define the fit region end for window building.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Inline Q/A as rendered by experiments/wyly_babi2.py
_QA_RE = re.compile(r"Q:\s*(.*?)\s*A:\s*([A-Za-z]+)\.")


def mine_inline_qa(
    text: str,
    *,
    holdout_frac: float = 0.1,
    max_queries: int = 1000,
    max_prompt_chars: int = 8000,
) -> tuple[list[dict], int]:
    """Return (held_out_queries, fit_end_char).

    Each query is ``{"prompt": "... Q: ... A:", "answer": "garden"}``.
    ``fit_end_char`` is the corpus character index after the last fit-region Q/A —
    windows for tables should be built only from ``text[:fit_end_char]``.
    """
    matches = list(_QA_RE.finditer(text))
    if not matches:
        return [], 0
    events: list[dict] = []
    for m in matches:
        a_pos = text.find("A:", m.start(), m.end())
        if a_pos < 0:
            continue
        prompt_end = a_pos + 2
        prompt_start = max(0, prompt_end - max_prompt_chars)
        if prompt_start > 0:
            # snap forward to a sentence boundary when possible
            dot = text.find(". ", prompt_start, prompt_end)
            if dot != -1 and dot + 2 < prompt_end:
                prompt_start = dot + 2
        prompt = text[prompt_start:prompt_end].strip()
        if not prompt.endswith("A:"):
            continue
        events.append({
            "prompt": prompt,
            "answer": m.group(2).strip(),
            "char_start": prompt_start,
            "char_end": m.end(),
        })
    if not events:
        return [], 0
    cut = max(1, int(len(events) * (1.0 - holdout_frac)))
    if cut >= len(events):
        cut = len(events) - 1
    fit_events, held = events[:cut], events[cut:]
    fit_end_char = fit_events[-1]["char_end"] if fit_events else 0
    if len(held) > max_queries:
        # evenly spaced subsample in corpus order (stable)
        step = (len(held) - 1) / max(max_queries - 1, 1)
        held = [held[int(round(i * step))] for i in range(max_queries)]
    # strip span fields for JSON consumers
    queries = [{"prompt": e["prompt"], "answer": e["answer"]} for e in held]
    return queries, fit_end_char


def mine_and_save(
    corpus_path: str | Path,
    out_path: str | Path,
    **kwargs,
) -> dict:
    text = Path(corpus_path).read_text()
    queries, fit_end = mine_inline_qa(text, **kwargs)
    out_path = Path(out_path)
    out_path.write_text(json.dumps(queries, indent=0))
    meta = {
        "source": str(corpus_path),
        "n_queries": len(queries),
        "fit_end_char": fit_end,
        "corpus_chars": len(text),
        "holdout_frac": kwargs.get("holdout_frac", 0.1),
        "query_source": "corpus_mined",
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
