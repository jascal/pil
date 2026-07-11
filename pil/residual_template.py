"""Domain-agnostic residual templates (mine short maps → propose leaves → admit).

Residual families recover *missing base maps* from short composite examples so
combinators / joins can compose systematically. The **patterns** are reusable;
domains only supply markers, prefix tokens, and optional structural seeds.

Built-in templates (domain-agnostic patterns)::

  nfold        — ``(x, marker_n) → unit*n``  ⇒  propose ``(x,) → unit``
  prefix_body  — ``(x, d) → prefix + body``  ⇒  propose ``(x,) → body``
                 when tgt starts with a known prefix token
  structural   — domain-supplied seed maps always proposed if missing

Domain packs (SCAN, listops, …) instantiate markers; the same
``ResidualFamily.propose`` / ``admit`` code runs unchanged.

Standalone: corpus maps only, no teacher/soft SGD. Every candidate carries
``template_id`` + provenance for certification.

See ``docs/notes/residual_templates.md``, ``experiments/campaign_residual_transfer.py``.
"""

from __future__ import annotations

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
    """Domain-supplied atoms / markers that instantiate abstract templates.

    Parameters
    ----------
    name:
        Domain tag (``scan``, ``listops``, …).
    nfold_markers:
        Suffix token → fold count, e.g. SCAN ``{"twice": 2, "thrice": 3}`` or
        listops ``{"x2": 2, "x3": 3}``.
    prefix_tokens:
        Target tokens that count as a leading "direction/prefix" before a body,
        e.g. SCAN ``{"I_TURN_LEFT", "I_TURN_RIGHT"}``. Empty disables prefix_body.
    structural_seeds:
        Always-proposed residual maps when missing from the short map (domain-specific).
    enabled_templates:
        Which abstract templates to run; default = all built-ins the domain can use.
    """

    name: str
    nfold_markers: dict[str, int] = field(default_factory=dict)
    prefix_tokens: frozenset[str] = field(default_factory=frozenset)
    structural_seeds: MapDict = field(default_factory=dict)
    enabled_templates: tuple[str, ...] | None = None  # None = all applicable

    def wants(self, template_id: str) -> bool:
        if self.enabled_templates is None:
            return True
        return template_id in self.enabled_templates


# --- abstract templates ------------------------------------------------------

class ResidualTemplate:
    """One rewrite pattern. Subclasses implement ``propose`` only."""

    id: str = "base"
    pattern_agnostic: bool = True

    def propose(self, short_maps: MapDict, domain: DomainAtoms) -> list[ResidualCandidate]:
        raise NotImplementedError


class NFoldTemplate(ResidualTemplate):
    """If ``(x, marker)`` maps to exact n-fold of unit, propose bare ``(x,) → unit``."""

    id = "nfold"
    pattern_agnostic = True

    def propose(self, short_maps: MapDict, domain: DomainAtoms) -> list[ResidualCandidate]:
        if not domain.nfold_markers:
            return []
        out: list[ResidualCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for src, tgt in short_maps.items():
            if len(src) != 2 or src[1] not in domain.nfold_markers:
                continue
            n = domain.nfold_markers[src[1]]
            if n < 2 or not tgt or len(tgt) % n != 0:
                continue
            unit_len = len(tgt) // n
            unit = tgt[:unit_len]
            if not all(tgt[i * unit_len:(i + 1) * unit_len] == unit for i in range(n)):
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
                meta={"from": src, "n": n, "marker": src[1]},
            ))
        return out


class PrefixBodyTemplate(ResidualTemplate):
    """If ``(x, d)`` maps to ``prefix + body`` with known prefix token, propose ``(x,) → body``."""

    id = "prefix_body"
    pattern_agnostic = True

    def propose(self, short_maps: MapDict, domain: DomainAtoms) -> list[ResidualCandidate]:
        if not domain.prefix_tokens:
            return []
        out: list[ResidualCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for src, tgt in short_maps.items():
            if len(src) != 2 or not tgt:
                continue
            if tgt[0] not in domain.prefix_tokens or len(tgt) < 2:
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


# Built-in library (order: nfold before prefix_body so both can propose same leaf;
# ResidualFamily dedupes by src keeping first).
DEFAULT_TEMPLATES: tuple[ResidualTemplate, ...] = (
    NFoldTemplate(),
    PrefixBodyTemplate(),
    StructuralSeedTemplate(),
)


# --- domain packs ------------------------------------------------------------

def scan_domain_atoms() -> DomainAtoms:
    """SCAN action domain: twice/thrice n-fold, I_TURN_* prefixes, turn L/R seeds."""
    return DomainAtoms(
        name="scan",
        nfold_markers={"twice": 2, "thrice": 3},
        prefix_tokens=frozenset({"I_TURN_LEFT", "I_TURN_RIGHT"}),
        structural_seeds={
            ("turn", "left"): ["I_TURN_LEFT"],
            ("turn", "right"): ["I_TURN_RIGHT"],
        },
    )


def listops_domain_atoms() -> DomainAtoms:
    """Synthetic listops: x2/x3 n-fold, optional PAD prefix strip (usually unused)."""
    return DomainAtoms(
        name="listops",
        nfold_markers={"x2": 2, "x3": 3},
        prefix_tokens=frozenset(),  # no dir-prefix in base listops
        structural_seeds={},
    )


# --- family: propose + admit -------------------------------------------------

@dataclass
class ResidualFamily:
    """Mine short maps → template propose → optional greedy admit.

    Parameters
    ----------
    domain:
        Domain atoms (markers / seeds).
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
    ) -> tuple[MapDict, list[dict[str, Any]]]:
        """Greedy val-marginal admit of residual candidates into a copy of short_maps.

        ``score_fn(maps)`` scores the full map dict (base short + admitted residual).
        Returns (admitted_full_maps, admit_log).
        """
        pool = list(candidates if candidates is not None else self.propose(short_maps))
        admitted: MapDict = {k: list(v) for k, v in short_maps.items()}
        log: list[dict[str, Any]] = []
        base = score_fn(admitted)
        remaining = list(pool)
        for _ in range(max_rules):
            best: tuple[float, ResidualCandidate | None] = (thresh, None)
            for cand in remaining:
                if cand.src in admitted:
                    continue
                trial = dict(admitted)
                trial[cand.src] = list(cand.tgt)
                sc = score_fn(trial)
                marg = sc - base
                log.append({
                    "src": " ".join(cand.src),
                    "template_id": cand.template_id,
                    "pattern_agnostic": cand.pattern_agnostic,
                    "domain": cand.domain,
                    "marginal": marg,
                    "score": sc,
                    "meta": dict(cand.meta),
                })
                if marg > best[0]:
                    best = (marg, cand)
            if best[1] is None:
                break
            cand = best[1]
            admitted[cand.src] = list(cand.tgt)
            remaining = [c for c in remaining if c.src != cand.src]
            base = score_fn(admitted)
        return admitted, log

    def admit_templates(
        self,
        short_maps: MapDict,
        score_fn: ScoreFn,
        *,
        thresh: float = 1e-4,
    ) -> tuple[set[str], MapDict, list[dict[str, Any]]]:
        """Meta-admit: enable whole template_ids by marginal, then apply all their cands.

        Returns (enabled_template_ids, maps_with_those_residuals, log).
        """
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
                log.append({
                    "template_id": tid, "marginal": marg, "score": sc,
                    "n_cands": len(by_tid[tid]),
                })
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
        return enabled, maps, log

    def diagnostics(
        self,
        short_maps: MapDict,
        admitted_src: Iterable[tuple[str, ...]] | None = None,
    ) -> dict[str, Any]:
        """Coverage stats: agnostic vs domain-specific among proposed / admitted."""
        cands = self.propose(short_maps)
        adm = set(admitted_src) if admitted_src is not None else {c.src for c in cands}
        proposed = cands
        admitted_c = [c for c in proposed if c.src in adm]
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
