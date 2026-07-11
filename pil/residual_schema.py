"""Bridge: residual candidates → PIC schemas (first integration slice).

Implements the step-2 *bridge* from ``docs/notes/residual_as_schema.md``: an
admitted residual leaf becomes a :class:`~pil.schemas.Schema`, so the existing
:class:`~pil.schemas.SchemaBank` / :func:`~pil.schemas.propose_schemas` machinery
selects, weights, and exports it — instead of a parallel symbolic admit loop.

**Scope of this slice — token-presence residuals.** A candidate whose firing word
``src[0]`` appears anywhere in the context window fires its single target token
``tgt[0]`` (a Freebase ``ns:`` path for ``relation_atom``, or any single-token
leaf). The tensor ``predict`` and the exported Datalog clause are verified
equivalent (``tests/test_residual_schema.py`` — including a souffle round-trip).

**Still open (see the note, do not overclaim):**
  - CFQ *set-F1* selection for bag prediction — ``propose_schemas`` here scores
    single-token exact-match, which is the honest reuse, not the set metric.
  - n-fold *unit* schemas (multi-token ``tgt``; needs the value table) — skipped.
  - soft-NLL weight training of admitted residual weights.

Symbols must already share a vocabulary: ``stoi`` maps both question words and
target paths to token ids. Building a CFQ ``stoi`` is a later slice; the bridge
itself is vocabulary-agnostic.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from pil.residual_template import ResidualCandidate
from pil.schemas import Schema


def _presence_schema(word_id: int, path_id: int, name: str) -> Schema:
    """A schema that fires ``path_id`` when ``word_id`` is present in the window.

    ``predict`` ignores ``values``/``v2t`` (presence is not an arithmetic
    predicate). The Datalog body matches the word token at *any* position
    (``tok(I,_,word_id)``) and assigns the constant target token to ``C`` — the
    exporter joins ``candtok(V,C)`` and the learned weight.
    """

    def predict(x: Tensor, values: Tensor, v2t: Tensor) -> Tensor:  # noqa: ARG001
        present = (x == word_id).any(dim=1)
        path = torch.full((x.shape[0],), path_id, dtype=torch.long, device=x.device)
        return torch.where(present, path, torch.full_like(path, -1))

    return Schema(name=name, predict=predict, datalog=f"tok(I,_,{word_id}), C={path_id}")


def residual_candidates_to_schemas(
    candidates: Sequence[ResidualCandidate],
    stoi: Mapping[str, int],
) -> tuple[list[Schema], list[dict]]:
    """Bridge residual leaves to token-presence schemas.

    A candidate bridges iff it has a firing word (``src[0]``), a *single* target
    token (``tgt[0]``), and both symbols are in ``stoi``. Everything else is
    skipped with a reason (returned, not swallowed) — this slice is single-token
    presence only. Duplicate ``(word, path)`` pairs collapse to one schema.

    Returns ``(schemas, skipped)`` where each skipped entry is
    ``{"src", "template_id", "reason"}``.
    """
    schemas: list[Schema] = []
    skipped: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for c in candidates:
        if not c.src or len(c.tgt) != 1:
            skipped.append({
                "src": c.src, "template_id": c.template_id,
                "reason": "not-single-token-target",
            })
            continue
        word, path = c.src[0], c.tgt[0]
        if word not in stoi or path not in stoi:
            skipped.append({
                "src": c.src, "template_id": c.template_id,
                "reason": "unknown-symbol",
            })
            continue
        key = (stoi[word], stoi[path])
        if key in seen:
            continue
        seen.add(key)
        schemas.append(_presence_schema(
            key[0], key[1], name=f"{c.template_id}/{word}|{path}",
        ))
    return schemas, skipped
