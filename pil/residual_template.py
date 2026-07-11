"""Domain-agnostic residual templates (mine short maps → propose leaves → admit).

Residual families recover *missing base maps* from short composite examples so
combinators / joins can compose systematically.

**Main line: induce, don't hand-author markers.**
  - ``induce_nfold_markers`` discovers suffix→k from short maps when
    ``(x, m) → unit*k`` holds consistently — ``DomainAtoms.nfold_markers`` may be empty.
  - ``RewriteSynthesizer`` enumerates a tiny rewrite DSL (repeat_k / strip_prefix /
    strip_suffix) aligned to short maps; subsumes hand nfold+prefix for bare-leaf recovery.

Built-in templates remain as explicit pattern classes; synthesizer is an optional proposer.

``ResidualFamily.admit`` uses **CELF-style lazy greedy** (priority queue of stale upper
bounds) so score_fn is not re-run for every candidate every round. Admit log records
admissions only (no O(rounds×cands) duplicate marginal rows).

Standalone: corpus maps only, no teacher/soft SGD. Provenance on every candidate.

See ``docs/notes/residual_templates.md``, ``experiments/campaign_residual_transfer.py``.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# --- types -------------------------------------------------------------------

MapDict = dict[tuple[str, ...], list[str]]
ScoreFn = Callable[[MapDict], float]  # higher is better (e.g. val exact-match)


@dataclass(frozen=True)
class ResidualCandidate:
    """One proposed residual map with template provenance."""

    src: tuple[str, ...]
    tgt: tuple[str, ...]
    template_id: str
    domain: str
    # True if the *pattern* is domain-agnostic (nfold/prefix_body); structural may be False
    pattern_agnostic: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def as_map(self) -> MapDict:
        return {self.src: list(self.tgt)}


@dataclass
class DomainAtoms:
    """Domain atoms. Prefer *empty* nfold_markers + auto induction when possible.

    Parameters
    ----------
    name:
        Domain tag (``scan``, ``listops``, …).
    nfold_markers:
        Optional human seeds for suffix→k. With ``auto_induce_markers=True`` (default),
        corpus-induced markers fill gaps; supplied keys override induction on conflict.
    prefix_tokens:
        Leading tgt tokens for prefix_body (empty → induce if auto_induce_prefixes).
    structural_seeds:
        Always-proposed residual maps when missing (often domain-specific).
    enabled_templates:
        Which proposers to run; include ``rewrite_synth`` for DSL enumeration.
    auto_induce_markers:
        Discover nfold markers from short maps (main line toward generality).
    auto_induce_prefixes:
        Discover frequent leading tgt tokens as prefix candidates.
    """

    name: str
    nfold_markers: dict[str, int] = field(default_factory=dict)
    prefix_tokens: frozenset[str] = field(default_factory=frozenset)
    structural_seeds: MapDict = field(default_factory=dict)
    enabled_templates: tuple[str, ...] | None = None  # None = all applicable
    auto_induce_markers: bool = True
    auto_induce_prefixes: bool = False

    def wants(self, template_id: str) -> bool:
        if self.enabled_templates is None:
            return True
        return template_id in self.enabled_templates


# --- marker / prefix induction -----------------------------------------------

def is_exact_nfold(tgt: Sequence[str], n: int) -> list[str] | None:
    """If tgt is unit repeated n times, return unit; else None."""
    if n < 2 or not tgt or len(tgt) % n != 0:
        return None
    unit_len = len(tgt) // n
    unit = list(tgt[:unit_len])
    if all(list(tgt[i * unit_len:(i + 1) * unit_len]) == unit for i in range(n)):
        return unit
    return None


def induce_nfold_markers(
    short_maps: MapDict,
    *,
    min_support: int = 1,
    max_k: int = 8,
) -> dict[str, int]:
    """Induce suffix→k from short maps: (x, m) → unit*k consistently.

    No human nfold_markers required. Votes per marker; majority k wins if support ≥ min.
    """
    votes: dict[str, Counter[int]] = defaultdict(Counter)
    for src, tgt in short_maps.items():
        if len(src) != 2 or not tgt:
            continue
        m = src[1]
        for k in range(2, min(max_k, len(tgt)) + 1):
            if is_exact_nfold(tgt, k) is not None:
                votes[m][k] += 1
                break  # prefer smallest k that fits
    out: dict[str, int] = {}
    for m, ctr in votes.items():
        k, c = ctr.most_common(1)[0]
        if c >= min_support:
            out[m] = k
    return out


def induce_prefix_tokens(
    short_maps: MapDict,
    *,
    min_support: int = 2,
) -> frozenset[str]:
    """Frequent leading tgt tokens on len-2 maps with |tgt|≥2."""
    counts: Counter[str] = Counter()
    for src, tgt in short_maps.items():
        if len(src) == 2 and len(tgt) >= 2:
            counts[tgt[0]] += 1
    return frozenset(t for t, c in counts.items() if c >= min_support)


def resolve_nfold_markers(short_maps: MapDict, domain: DomainAtoms) -> dict[str, int]:
    """Merge induced + supplied markers (supplied overrides on conflict)."""
    markers: dict[str, int] = {}
    if domain.auto_induce_markers:
        markers.update(induce_nfold_markers(short_maps))
    markers.update(domain.nfold_markers)
    return markers


def resolve_prefix_tokens(short_maps: MapDict, domain: DomainAtoms) -> frozenset[str]:
    prefs = set(domain.prefix_tokens)
    if domain.auto_induce_prefixes:
        prefs |= set(induce_prefix_tokens(short_maps))
    return frozenset(prefs)


# --- abstract templates ------------------------------------------------------

class ResidualTemplate:
    """One rewrite pattern. Subclasses implement ``propose`` only."""

    id: str = "base"
    pattern_agnostic: bool = True

    def propose(self, short_maps: MapDict, domain: DomainAtoms) -> list[ResidualCandidate]:
        raise NotImplementedError


class NFoldTemplate(ResidualTemplate):
    """If ``(x, marker)`` maps to exact n-fold of unit, propose bare ``(x,) → unit``.

    Markers from ``resolve_nfold_markers`` (induced and/or supplied).
    """

    id = "nfold"
    pattern_agnostic = True

    def propose(self, short_maps: MapDict, domain: DomainAtoms) -> list[ResidualCandidate]:
        markers = resolve_nfold_markers(short_maps, domain)
        if not markers:
            return []
        out: list[ResidualCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for src, tgt in short_maps.items():
            if len(src) != 2 or src[1] not in markers:
                continue
            n = markers[src[1]]
            unit = is_exact_nfold(tgt, n)
            if unit is None:
                continue
            leaf = (src[0],)
            if leaf in short_maps or leaf in seen:
                continue
            seen.add(leaf)
            out.append(ResidualCandidate(
                src=leaf,
                tgt=tuple(unit),
                template_id=self.id,
                domain=domain.name,
                pattern_agnostic=True,
                meta={
                    "from": src, "n": n, "marker": src[1],
                    "marker_induced": src[1] not in domain.nfold_markers,
                },
            ))
        return out


class PrefixBodyTemplate(ResidualTemplate):
    """If ``(x, d)`` maps to ``prefix + body`` with known prefix token, propose ``(x,) → body``."""

    id = "prefix_body"
    pattern_agnostic = True

    def propose(self, short_maps: MapDict, domain: DomainAtoms) -> list[ResidualCandidate]:
        prefs = resolve_prefix_tokens(short_maps, domain)
        if not prefs:
            return []
        out: list[ResidualCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for src, tgt in short_maps.items():
            if len(src) != 2 or not tgt:
                continue
            if tgt[0] not in prefs or len(tgt) < 2:
                continue
            leaf = (src[0],)
            body = tuple(tgt[1:])
            if not body or leaf in short_maps or leaf in seen:
                continue
            seen.add(leaf)
            out.append(ResidualCandidate(
                src=leaf,
                tgt=body,
                template_id=self.id,
                domain=domain.name,
                pattern_agnostic=True,
                meta={"from": src, "prefix": tgt[0]},
            ))
        return out


class StructuralSeedTemplate(ResidualTemplate):
    """Domain-supplied structural maps (often domain-specific)."""

    id = "structural"
    pattern_agnostic = False

    def propose(self, short_maps: MapDict, domain: DomainAtoms) -> list[ResidualCandidate]:
        out: list[ResidualCandidate] = []
        for src, tgt in domain.structural_seeds.items():
            if src in short_maps:
                continue
            out.append(ResidualCandidate(
                src=src,
                tgt=tuple(tgt),
                template_id=self.id,
                domain=domain.name,
                pattern_agnostic=False,
                meta={"seed": True},
            ))
        return out


class RewriteSynthesizer(ResidualTemplate):
    """Tiny enumerative synthesizer over a rewrite DSL aligned to short maps.

    Ops (applied to recover a *shorter src leaf* from a composite map)::

      repeat_k     — tgt is unit*k  ⇒  leaf (x,) → unit   [same as nfold]
      strip_prefix — tgt = p + body with |p|=1  ⇒  leaf (x,) → body
      strip_suffix — tgt = body + s with |s|=1  ⇒  leaf (x,) → body  (rare)

    Subsumes hand nfold/prefix for bare-leaf recovery without a marker lexicon.
    """

    id = "rewrite_synth"
    pattern_agnostic = True

    def propose(self, short_maps: MapDict, domain: DomainAtoms) -> list[ResidualCandidate]:
        out: list[ResidualCandidate] = []
        seen: set[tuple[str, ...]] = set()
        markers = resolve_nfold_markers(short_maps, domain)

        for src, tgt in short_maps.items():
            if len(src) != 2 or not tgt:
                continue
            leaf = (src[0],)
            if leaf in short_maps or leaf in seen:
                continue

            # repeat_k for k=2..8
            for k in range(2, min(8, len(tgt)) + 1):
                unit = is_exact_nfold(tgt, k)
                if unit is None:
                    continue
                # prefer markers that match induced k when present
                meta = {"op": "repeat_k", "k": k, "from": src, "marker": src[1]}
                if src[1] in markers:
                    meta["marker_k"] = markers[src[1]]
                seen.add(leaf)
                out.append(ResidualCandidate(
                    src=leaf, tgt=tuple(unit), template_id=self.id,
                    domain=domain.name, pattern_agnostic=True, meta=meta,
                ))
                break

            if leaf in seen:
                continue
            # strip_prefix
            if len(tgt) >= 2:
                body = tuple(tgt[1:])
                if body:
                    seen.add(leaf)
                    out.append(ResidualCandidate(
                        src=leaf, tgt=body, template_id=self.id,
                        domain=domain.name, pattern_agnostic=True,
                        meta={"op": "strip_prefix", "prefix": tgt[0], "from": src},
                    ))
        return out


# Built-in library. rewrite_synth is opt-in via enabled_templates (overlaps nfold).
DEFAULT_TEMPLATES: tuple[ResidualTemplate, ...] = (
    NFoldTemplate(),
    PrefixBodyTemplate(),
    StructuralSeedTemplate(),
)

SYNTH_TEMPLATES: tuple[ResidualTemplate, ...] = (
    RewriteSynthesizer(),
    StructuralSeedTemplate(),
)


# --- domain packs ------------------------------------------------------------

def scan_domain_atoms(*, induce_only: bool = False) -> DomainAtoms:
    """SCAN pack. ``induce_only=True`` leaves nfold_markers empty (pure induction)."""
    return DomainAtoms(
        name="scan",
        nfold_markers={} if induce_only else {"twice": 2, "thrice": 3},
        prefix_tokens=frozenset({"I_TURN_LEFT", "I_TURN_RIGHT"}),
        structural_seeds={
            ("turn", "left"): ["I_TURN_LEFT"],
            ("turn", "right"): ["I_TURN_RIGHT"],
        },
        auto_induce_markers=True,
    )


def listops_domain_atoms(*, induce_only: bool = True) -> DomainAtoms:
    """Listops pack — default **induce-only** markers (no hand x2/x3)."""
    return DomainAtoms(
        name="listops",
        nfold_markers={} if induce_only else {"x2": 2, "x3": 3},
        prefix_tokens=frozenset(),
        structural_seeds={},
        auto_induce_markers=True,
    )


# --- family: propose + CELF admit --------------------------------------------

@dataclass
class ResidualFamily:
    """Mine short maps → template propose → CELF greedy admit.

    Parameters
    ----------
    domain:
        Domain atoms (markers / seeds / induction flags).
    templates:
        Template library (default = nfold, prefix_body, structural).
    """

    domain: DomainAtoms
    templates: Sequence[ResidualTemplate] = DEFAULT_TEMPLATES

    def active_templates(self) -> list[ResidualTemplate]:
        return [t for t in self.templates if self.domain.wants(t.id)]

    def propose(self, short_maps: MapDict) -> list[ResidualCandidate]:
        """Propose residual candidates; first template wins on duplicate src."""
        by_src: dict[tuple[str, ...], ResidualCandidate] = {}
        for tmpl in self.active_templates():
            for cand in tmpl.propose(short_maps, self.domain):
                if cand.src in short_maps or cand.src in by_src:
                    continue
                by_src[cand.src] = cand
        return list(by_src.values())

    def propose_map(self, short_maps: MapDict) -> MapDict:
        """Convenience: all proposed residuals as a map dict (no admit)."""
        out: MapDict = {}
        for c in self.propose(short_maps):
            out[c.src] = list(c.tgt)
        return out

    def admit(
        self,
        short_maps: MapDict,
        score_fn: ScoreFn,
        *,
        thresh: float = 1e-4,
        max_rules: int = 32,
        candidates: list[ResidualCandidate] | None = None,
        celf: bool = True,
    ) -> tuple[MapDict, list[dict[str, Any]]]:
        """Greedy val-marginal admit. CELF lazy greedy by default.

        Admit log contains **one row per admission** (not per candidate×round).
        Final ``log[-1]`` may include a ``stopped`` summary with n_remaining.
        """
        if celf:
            return self._admit_celf(
                short_maps, score_fn, thresh=thresh, max_rules=max_rules,
                candidates=candidates,
            )
        return self._admit_naive(
            short_maps, score_fn, thresh=thresh, max_rules=max_rules,
            candidates=candidates,
        )

    def _admit_naive(
        self,
        short_maps: MapDict,
        score_fn: ScoreFn,
        *,
        thresh: float,
        max_rules: int,
        candidates: list[ResidualCandidate] | None,
    ) -> tuple[MapDict, list[dict[str, Any]]]:
        pool = list(candidates if candidates is not None else self.propose(short_maps))
        admitted: MapDict = {k: list(v) for k, v in short_maps.items()}
        log: list[dict[str, Any]] = []
        base = score_fn(admitted)
        remaining = list(pool)
        n_score = 1
        for _ in range(max_rules):
            best: tuple[float, ResidualCandidate | None] = (thresh, None)
            for cand in remaining:
                if cand.src in admitted:
                    continue
                trial = dict(admitted)
                trial[cand.src] = list(cand.tgt)
                sc = score_fn(trial)
                n_score += 1
                marg = sc - base
                if marg > best[0]:
                    best = (marg, cand)
            if best[1] is None:
                break
            cand = best[1]
            admitted[cand.src] = list(cand.tgt)
            remaining = [c for c in remaining if c.src != cand.src]
            base = score_fn(admitted)
            n_score += 1
            log.append({
                "event": "admit",
                "src": " ".join(cand.src),
                "template_id": cand.template_id,
                "pattern_agnostic": cand.pattern_agnostic,
                "domain": cand.domain,
                "marginal": best[0],
                "score": base,
                "meta": dict(cand.meta),
            })
        log.append({"event": "stop", "n_score_calls": n_score, "n_remaining": len(remaining)})
        return admitted, log

    def _admit_celf(
        self,
        short_maps: MapDict,
        score_fn: ScoreFn,
        *,
        thresh: float,
        max_rules: int,
        candidates: list[ResidualCandidate] | None,
    ) -> tuple[MapDict, list[dict[str, Any]]]:
        """CELF: priority queue of marginal upper bounds; re-score only the current top.

        Submodular diminishing returns ⇒ a stale upper bound that is still best
        after re-evaluation is optimal for this round (Leskovec et al.).
        """
        pool = list(candidates if candidates is not None else self.propose(short_maps))
        admitted: MapDict = {k: list(v) for k, v in short_maps.items()}
        log: list[dict[str, Any]] = []
        base = score_fn(admitted)
        n_score = 1
        # heap entries: (-marginal_upper, seq, cand_id, cand, bound_at_base)
        # bound_at_base tracks which base score the upper bound was computed under
        heap: list[tuple[float, int, int, ResidualCandidate, float]] = []
        seq = 0
        for i, cand in enumerate(pool):
            if cand.src in admitted:
                continue
            trial = dict(admitted)
            trial[cand.src] = list(cand.tgt)
            sc = score_fn(trial)
            n_score += 1
            marg = sc - base
            heapq.heappush(heap, (-marg, seq, i, cand, base))
            seq += 1

        for _ in range(max_rules):
            if not heap:
                break
            admitted_this = False
            while heap:
                neg_u, _, cid, cand, bound_base = heapq.heappop(heap)
                if cand.src in admitted:
                    continue
                # re-score if bound is stale (base moved)
                if bound_base != base:
                    trial = dict(admitted)
                    trial[cand.src] = list(cand.tgt)
                    sc = score_fn(trial)
                    n_score += 1
                    marg = sc - base
                    heapq.heappush(heap, (-marg, seq, cid, cand, base))
                    seq += 1
                    continue
                marg = -neg_u
                if marg <= thresh:
                    # best bound below thresh → stop
                    heap = []  # force exit outer
                    break
                # admit
                admitted[cand.src] = list(cand.tgt)
                base = score_fn(admitted)
                n_score += 1
                log.append({
                    "event": "admit",
                    "src": " ".join(cand.src),
                    "template_id": cand.template_id,
                    "pattern_agnostic": cand.pattern_agnostic,
                    "domain": cand.domain,
                    "marginal": marg,
                    "score": base,
                    "meta": dict(cand.meta),
                    "n_score_calls": n_score,
                })
                admitted_this = True
                break
            if not admitted_this:
                break
        log.append({
            "event": "stop",
            "n_score_calls": n_score,
            "n_admitted": sum(1 for e in log if e.get("event") == "admit"),
            "n_heap_left": len(heap),
        })
        return admitted, log

    def admit_templates(
        self,
        short_maps: MapDict,
        score_fn: ScoreFn,
        *,
        thresh: float = 1e-4,
    ) -> tuple[set[str], MapDict, list[dict[str, Any]]]:
        """Meta-admit: enable whole template_ids by marginal, then apply their cands."""
        all_cands = self.propose(short_maps)
        by_tid: dict[str, list[ResidualCandidate]] = {}
        for c in all_cands:
            by_tid.setdefault(c.template_id, []).append(c)

        enabled: set[str] = set()
        maps: MapDict = {k: list(v) for k, v in short_maps.items()}
        log: list[dict[str, Any]] = []
        base = score_fn(maps)
        remaining = list(by_tid.keys())
        while remaining:
            best = (thresh, None)
            for tid in remaining:
                trial = dict(maps)
                for c in by_tid[tid]:
                    if c.src not in trial:
                        trial[c.src] = list(c.tgt)
                sc = score_fn(trial)
                marg = sc - base
                if marg > best[0]:
                    best = (marg, tid)
            if best[1] is None:
                break
            tid = best[1]
            enabled.add(tid)
            for c in by_tid[tid]:
                if c.src not in maps:
                    maps[c.src] = list(c.tgt)
            remaining.remove(tid)
            base = score_fn(maps)
            log.append({
                "event": "admit_template",
                "template_id": tid, "marginal": best[0], "score": base,
                "n_cands": len(by_tid[tid]),
            })
        log.append({"event": "stop", "enabled": sorted(enabled)})
        return enabled, maps, log

    def diagnostics(
        self,
        short_maps: MapDict,
        admitted_src: Iterable[tuple[str, ...]] | None = None,
    ) -> dict[str, Any]:
        """Coverage stats: agnostic vs domain-specific; induced marker report."""
        cands = self.propose(short_maps)
        adm = set(admitted_src) if admitted_src is not None else {c.src for c in cands}
        proposed = cands
        admitted_c = [c for c in proposed if c.src in adm]
        induced = induce_nfold_markers(short_maps)

        def frac(xs: list[ResidualCandidate], pred) -> float:
            if not xs:
                return 0.0
            return sum(1 for c in xs if pred(c)) / len(xs)

        return {
            "domain": self.domain.name,
            "n_short_maps": len(short_maps),
            "n_proposed": len(proposed),
            "n_admitted": len(admitted_c),
            "proposed_by_template": _count_by(proposed, lambda c: c.template_id),
            "admitted_by_template": _count_by(admitted_c, lambda c: c.template_id),
            "frac_proposed_agnostic": frac(proposed, lambda c: c.pattern_agnostic),
            "frac_admitted_agnostic": frac(admitted_c, lambda c: c.pattern_agnostic),
            "induced_nfold_markers": dict(induced),
            "supplied_nfold_markers": dict(self.domain.nfold_markers),
            "active_templates": [t.id for t in self.active_templates()],
        }


def _count_by(xs: list[ResidualCandidate], key_fn) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in xs:
        k = key_fn(c)
        out[k] = out.get(k, 0) + 1
    return out


def apply_residuals(short_maps: MapDict, residuals: MapDict) -> MapDict:
    """Merge residual maps into short maps (short maps win on key clash)."""
    out = dict(residuals)
    out.update(short_maps)
    return out
