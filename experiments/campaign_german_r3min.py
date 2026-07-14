"""German R3-minimal — 4-arm case/aux_verb ladder (parse-free → heuristic → gold-oracle).

Ladder per task on GSD test (read exactly once for final scoring):
  1. parse-free baseline (loaded from data/german_r1.json / data/german_r2.json)
  2. non-parse heuristic (Arm C case / Arm D aux_verb) — serve-honest, no gold attachment
  3. gold-oracle upper bound (Arm A case / Arm B aux_verb) — NOT serve-honest

recovery_fraction = heuristic_gain / oracle_gain  (\"N/A\" if oracle_gain <= 0)

Does NOT import or reference any *_codex* files (parallel RACED lane).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
import campaign_german_r1 as r1  # noqa: E402
import campaign_german_r2 as r2  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GDATA = Path("/home/allans/code/germandata")
TASKS = GDATA / "tasks"
REGS = GDATA / "registers"
OUTPUT = REPO / "data" / "german_r3min.json"
BASELINE_R1 = REPO / "data" / "german_r1.json"
BASELINE_R2 = REPO / "data" / "german_r2.json"

SUBJECT_DEPRELS = frozenset({"nsubj", "nsubj:pass", "expl", "expl:pv"})
# Case-bearing nominal UPOS — oracle government only targets these (not ADV/PUNCT/VERB/…).
CASE_BEARING_UPOS = frozenset({"NOUN", "PROPN", "PRON", "DET", "ADJ"})
CONJ_FORMS = frozenset({"und", "oder", "aber", "denn", "sondern", "bzw", "beziehungsweise"})

# Full-cascade deprel→case rule names (Arm A); order = priority for primary rules.
FULL_ORACLE_RULES = (
    "prep", "cop", "subj", "obj", "iobj", "obl_arg", "nmod_gen", "verbgov", "inherit", "none",
)
# Direct nominative subjects under the full cascade (rule SUBJ).
SUBJ_CASE_DEPRELS = frozenset({"nsubj", "nsubj:pass", "csubj"})
# Expected partial-oracle Arm A accuracy (previous 2-branch rule set); regression-checked.
PARTIAL_ORACLE_ACC_EXPECTED = 0.8262318968772888

HONEST_SCOPE = (
    "R3-minimal 4-arm ladder. Arm A (case gold-oracle) and Arm B (aux_verb gold-oracle) "
    "are UPPER BOUNDS that use gold head_offset + gold deprel at score time — NOT serve-honest. "
    "Arm A FULL cascade (first-match-wins on case-bearing UPOS {NOUN,PROPN,PRON,DET,ADJ}): "
    "(1) PREP reverse-child ADP/case → preposition_case; "
    "(2) COP reverse-child deprel=cop on UPOS in {NOUN,PROPN,PRON} → Nom "
    "(predicate ADJ/DET with cop are uninflected, gold case '-', so excluded); "
    "(3) SUBJ deprel in {nsubj,nsubj:pass,csubj} → Nom; "
    "(4) OBJ deprel=obj → Acc; "
    "(5) IOBJ deprel=iobj → Dat (literal; near-zero coverage on GSD — treebank barely uses iobj); "
    "(6) OBL_ARG deprel=obl:arg → Dat "
    "[DATA-DRIVEN ADDITION beyond literal coordinator wording: GSD annotates bare dative "
    "objects as obl:arg, not iobj; TRAIN obl:arg is ~99.3% Dat]; "
    "(7) NMOD_GEN bare nmod (no ADP/case child; PREP did not fire) → Gen; "
    "(8) VERBGOV residual: governor VERB/AUX → verb_government majority object case "
    "(expect weak precision ~0.64 — a second limiter beyond attachment); "
    "(9) INHERIT fixpoint: case-bearing dependents inherit governor's candidate set "
    "(multi-hop agreement chains); "
    "(10) FALLBACK to R1 form-decidable registers + pred_case_off soft fallback. "
    "Partial oracle (prep + verb_gov with subject exclusion + one-hop inherit) is retained "
    "for regression (expected ~0.8262). Arm B never uses the ambiguous verb's own gold "
    "deprel/UPOS/label (circularity ban); only head_offset + SURFACE morphology of structural "
    "neighbors (participle_shape / zu-infinitive). Arms C/D are PARSE-FREE: Arm C verb-aware "
    "NP-span stop; Arm D clause-bounded participle/zu-infinitive search. No new DEV/TEST-tuned "
    "thresholds; R1's DEV-only fit_context_tier is reused. Baselines from data/german_r1.json "
    "(morph_case.registers_off) and data/german_r2.json (aux_verb.full)."
)


# ---------------------------------------------------------------------------
# Data loading (head_deprel + baselines)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_head_deprel_split(split: str) -> list[dict]:
    """Load head_deprel split; returns list of {sent_id, tokens, head_offset, deprel}."""
    path = TASKS / "head_deprel" / f"{split}.jsonl"
    out: list[dict] = []
    for rec in load_jsonl(path):
        tokens = rec["tokens"]
        ho = list(rec["targets"]["head_offset"])
        deprel = list(rec["targets"]["deprel"])
        n = len(tokens)
        if len(ho) != n or len(deprel) != n:
            raise RuntimeError(
                f"head_deprel/{split}: length mismatch on {rec.get('sent_id')}"
            )
        out.append({
            "sent_id": rec["sent_id"],
            "text": rec.get("text", ""),
            "tokens": tokens,
            "head_offset": ho,
            "deprel": deprel,
        })
    return out


def index_by_sent_id(rows: list[dict]) -> dict[str, dict]:
    return {r["sent_id"]: r for r in rows}


def assert_tokens_match(a: list[str], b: list[str], sent_id: str, left: str, right: str) -> None:
    if a != b:
        raise RuntimeError(
            f"token mismatch on {sent_id}: {left} vs {right} "
            f"(len {len(a)} vs {len(b)})"
        )


def join_head_deprel(
    aligned: list[dict],
    hd_rows: list[dict],
    *,
    label: str,
) -> list[dict]:
    """Join aligned pos/case sentences to head_deprel by sent_id; assert tokens match."""
    by_sid = index_by_sent_id(hd_rows)
    out: list[dict] = []
    for s in aligned:
        sid = s["sent_id"]
        if sid not in by_sid:
            raise RuntimeError(f"{label}: missing head_deprel for sent_id={sid}")
        hd = by_sid[sid]
        assert_tokens_match(s["tokens"], hd["tokens"], sid, "aligned", "head_deprel")
        out.append({
            **s,
            "head_offset": hd["head_offset"],
            "deprel": hd["deprel"],
        })
    return out


def load_baselines() -> tuple[float, float]:
    """Load parse-free baselines from R1/R2 artifacts (do not recompute from scratch)."""
    r1_art = json.loads(BASELINE_R1.read_text(encoding="utf-8"))
    r2_art = json.loads(BASELINE_R2.read_text(encoding="utf-8"))
    case_base = float(r1_art["scores"]["morph_case"]["registers_off"])
    aux_base = float(r2_art["scores"]["aux_verb"]["full"])
    return case_base, aux_base


def data_hash() -> str:
    """sha256 over task jsonl files used by this campaign (fixed order)."""
    h = hashlib.sha256()
    for task in ("pos", "morph_case", "morph_gnn", "head_deprel", "aux_verb"):
        for split in ("train", "dev", "test"):
            path = TASKS / task / f"{split}.jsonl"
            h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Governor / children helpers (head_offset structure)
# ---------------------------------------------------------------------------

def governor_index(i: int, head_offset: list[int]) -> int | None:
    """Governor index for token i; None if root (head_offset[i] == 0 → points to self)."""
    g = i + head_offset[i]
    if g == i:
        return None
    return g


def build_children(n: int, head_offset: list[int]) -> list[list[int]]:
    """children[g] = dependents of g (invert head_offset links; exclude self-loops)."""
    children: list[list[int]] = [[] for _ in range(n)]
    for j in range(n):
        g = j + head_offset[j]
        if j != g and 0 <= g < n:
            children[g].append(j)
    return children


# ---------------------------------------------------------------------------
# Surface morphology shared by Arms B/D
# ---------------------------------------------------------------------------

def zu_infinitive_shape(tokens: list[str], k: int) -> bool:
    """True if tokens[k] is surface zu-infinitive: ends -en (len>=4) and prev token is 'zu'."""
    if k < 0 or k >= len(tokens):
        return False
    low = tokens[k].lower()
    if not (len(low) >= 4 and low.endswith("en")):
        return False
    return k > 0 and tokens[k - 1].lower() == "zu"


def is_participle_or_zu_inf(tokens: list[str], k: int) -> bool:
    return r2.participle_shape(tokens[k]) or zu_infinitive_shape(tokens, k)


# ---------------------------------------------------------------------------
# Arm A — CASE gold-oracle government (upper bound; NOT serve-honest)
# ---------------------------------------------------------------------------

def _prep_case_from_adp_child(
    children_i: list[int],
    tokens: list[str],
    upos: list[str],
    deprel: list[str],
    strict_prep: dict[str, str],
    two_way_prep: dict[str, list[str]],
) -> set[str] | None:
    """If any ADP/case child is a known preposition, return its case candidate set."""
    for j in children_i:
        if upos[j] == "ADP" and deprel[j] == "case":
            low = tokens[j].lower()
            if low in strict_prep:
                return {strict_prep[low]}
            if low in two_way_prep:
                return set(two_way_prep[low])
    return None


def _has_adp_case_child(
    children_i: list[int],
    upos: list[str],
    deprel: list[str],
) -> bool:
    """True if any reverse-child is ADP with deprel=case (regardless of prep register)."""
    for j in children_i:
        if upos[j] == "ADP" and deprel[j] == "case":
            return True
    return False


def _has_cop_child(children_i: list[int], deprel: list[str]) -> bool:
    return any(deprel[j] == "cop" for j in children_i)


def oracle_case_gov_sentence_partial(
    tokens: list[str],
    upos: list[str],
    head_offset: list[int],
    deprel: list[str],
    strict_prep: dict[str, str],
    two_way_prep: dict[str, list[str]],
    verb_stem_index: list[tuple[str, str, dict]],
) -> tuple[list[set[str] | None], list[set[str] | None], dict[str, Any]]:
    """PARTIAL oracle (previous rule set): prep + verb_gov(excl. subjects) + one-hop inherit.

    Retained for regression against Arm A = 0.8262. Prefer oracle_case_gov_sentence_full
    for the current Arm A upper bound.
    """
    n = len(tokens)
    children = build_children(n, head_offset)
    prep_gov: list[set[str] | None] = [None] * n
    verb_gov: list[set[str] | None] = [None] * n
    b1_cands: list[set[str] | None] = [None] * n
    b2_cands: list[set[str] | None] = [None] * n

    # Pass 1 — primary heads (case-bearing nominals only)
    for i in range(n):
        if upos[i] not in CASE_BEARING_UPOS:
            continue

        # Branch 1 — reverse search for ADP/case children
        cand1 = _prep_case_from_adp_child(
            children[i], tokens, upos, deprel, strict_prep, two_way_prep,
        )
        if cand1 is not None:
            prep_gov[i] = cand1
            b1_cands[i] = cand1
            continue

        # Branch 2 — verb government via own governor (exclude subjects)
        g = i + head_offset[i]
        if (
            g != i
            and 0 <= g < n
            and upos[g] in ("VERB", "AUX")
            and deprel[i] not in SUBJECT_DEPRELS
        ):
            matched = r1.match_verb_stem(tokens[g], verb_stem_index)
            if matched is not None:
                _lemma, entry = matched
                case, _conf = r1.verb_majority_case(entry)
                if case is not None:
                    cand = {case}
                    verb_gov[i] = cand
                    b2_cands[i] = cand

    # Pass 2 — one-hop inheritance to case-bearing dependents of primary heads
    for i in range(n):
        if upos[i] not in CASE_BEARING_UPOS:
            continue
        if prep_gov[i] is not None or verb_gov[i] is not None:
            continue
        g = governor_index(i, head_offset)
        if g is None:
            continue
        if prep_gov[g] is not None:
            prep_gov[i] = set(prep_gov[g])
            b1_cands[i] = set(prep_gov[g])
        elif verb_gov[g] is not None:
            verb_gov[i] = set(verb_gov[g])
            b2_cands[i] = set(verb_gov[g])

    n_b1 = sum(1 for c in b1_cands if c is not None)
    n_b2 = sum(1 for c in b2_cands if c is not None)
    diag = {
        "n_branch1": n_b1,
        "n_branch2": n_b2,
        "n_neither": n - n_b1 - n_b2,
        "b1_cands": b1_cands,
        "b2_cands": b2_cands,
    }
    return prep_gov, verb_gov, diag


# Backward-compatible alias: existing tests call oracle_case_gov_sentence (partial).
oracle_case_gov_sentence = oracle_case_gov_sentence_partial


def oracle_case_gov_sentence_full(
    tokens: list[str],
    upos: list[str],
    head_offset: list[int],
    deprel: list[str],
    strict_prep: dict[str, str],
    two_way_prep: dict[str, list[str]],
    verb_stem_index: list[tuple[str, str, dict]],
) -> tuple[list[set[str] | None], list[set[str] | None], dict[str, Any]]:
    """FULL deprel→case gold-oracle cascade (first-match-wins; NOT serve-honest).

    Priority: PREP → COP → SUBJ → OBJ → IOBJ → OBL_ARG → NMOD_GEN → VERBGOV
    → INHERIT (fixpoint) → FALLBACK (handled downstream by merge/arbitrate).

    PREP candidates go to prep_gov; all other structural rules go to verb_gov
    (shared merge pipeline). Inheritance copies the governor's channel as-is.
    """
    n = len(tokens)
    children = build_children(n, head_offset)
    prep_gov: list[set[str] | None] = [None] * n
    verb_gov: list[set[str] | None] = [None] * n
    # Per-rule candidate streams (same length as tokens; None if rule did not fire here)
    rule_cands: dict[str, list[set[str] | None]] = {
        r: [None] * n for r in FULL_ORACLE_RULES if r != "none"
    }
    fired: list[str | None] = [None] * n  # which rule resolved each position

    def _resolve(i: int, rule: str, cand: set[str], *, via_prep_channel: bool) -> None:
        fired[i] = rule
        rule_cands[rule][i] = set(cand)
        if via_prep_channel:
            prep_gov[i] = set(cand)
        else:
            verb_gov[i] = set(cand)

    # ---- Rules 1–8: primary resolution on case-bearing nominals ----
    for i in range(n):
        if upos[i] not in CASE_BEARING_UPOS:
            continue

        # 1. PREP
        cand_prep = _prep_case_from_adp_child(
            children[i], tokens, upos, deprel, strict_prep, two_way_prep,
        )
        if cand_prep is not None:
            _resolve(i, "prep", cand_prep, via_prep_channel=True)
            continue

        # 2. COP (predicate nominal with reverse-child copula).
        # Restrict to true nominals: predicate ADJ/DET also host cop children but
        # are uninflected (gold case "-"); forcing Nom there is always wrong.
        if upos[i] in ("NOUN", "PROPN", "PRON") and _has_cop_child(children[i], deprel):
            _resolve(i, "cop", {"Nom"}, via_prep_channel=False)
            continue

        # 3. SUBJ
        if deprel[i] in SUBJ_CASE_DEPRELS:
            _resolve(i, "subj", {"Nom"}, via_prep_channel=False)
            continue

        # 4. OBJ
        if deprel[i] == "obj":
            _resolve(i, "obj", {"Acc"}, via_prep_channel=False)
            continue

        # 5. IOBJ (literal; near-zero coverage on GSD)
        if deprel[i] == "iobj":
            _resolve(i, "iobj", {"Dat"}, via_prep_channel=False)
            continue

        # 6. OBL_ARG — data-driven: GSD bare dative objects use obl:arg, not iobj
        if deprel[i] == "obl:arg":
            _resolve(i, "obl_arg", {"Dat"}, via_prep_channel=False)
            continue

        # 7. NMOD_GEN — bare adnominal genitive (PREP already skipped: no known prep)
        #    Also require no ADP/case reverse-child at all (prepositional nmod even if
        #    the prep is out-of-register should not get Gen from this rule).
        if deprel[i] == "nmod" and not _has_adp_case_child(children[i], upos, deprel):
            _resolve(i, "nmod_gen", {"Gen"}, via_prep_channel=False)
            continue

        # 8. VERBGOV residual (no hand-enumerated exclusion list; 1–7 already claimed
        #    obj/iobj/obl:arg/subj/…). Weak precision (~0.64) is a second limiter.
        g = i + head_offset[i]
        if g != i and 0 <= g < n and upos[g] in ("VERB", "AUX"):
            matched = r1.match_verb_stem(tokens[g], verb_stem_index)
            if matched is not None:
                _lemma, entry = matched
                case, _conf = r1.verb_majority_case(entry)
                if case is not None:
                    _resolve(i, "verbgov", {case}, via_prep_channel=False)
                    continue

    # ---- Rule 9: INHERIT fixpoint (multi-hop agreement chains) ----
    for _ in range(n):  # safety cap
        changed = False
        for i in range(n):
            if upos[i] not in CASE_BEARING_UPOS:
                continue
            if fired[i] is not None:
                continue
            g = governor_index(i, head_offset)
            if g is None or fired[g] is None:
                continue
            # Governor resolved by 1–8 or prior inherit; copy its set + channel as-is
            if prep_gov[g] is not None:
                _resolve(i, "inherit", prep_gov[g], via_prep_channel=True)
                changed = True
            elif verb_gov[g] is not None:
                _resolve(i, "inherit", verb_gov[g], via_prep_channel=False)
                changed = True
        if not changed:
            break

    n_resolved = sum(1 for f in fired if f is not None)
    diag = {
        "rule_cands": rule_cands,
        "fired": fired,
        "n_resolved": n_resolved,
        "n_none": n - n_resolved,
    }
    return prep_gov, verb_gov, diag


def flatten_oracle_case_gov_partial(
    joined: list[dict],
    strict_prep: dict[str, str],
    two_way_prep: dict[str, list[str]],
    verb_stem_index: list[tuple[str, str, dict]],
) -> tuple[list[set[str] | None], list[set[str] | None], dict[str, Any]]:
    """Flatten partial-oracle gov to stream order (regression path)."""
    prep_all: list[set[str] | None] = []
    verb_all: list[set[str] | None] = []
    n_b1 = n_b2 = n_neither = 0
    b1_all: list[set[str] | None] = []
    b2_all: list[set[str] | None] = []
    for s in joined:
        pg, vg, diag = oracle_case_gov_sentence_partial(
            s["tokens"], s["upos"], s["head_offset"], s["deprel"],
            strict_prep, two_way_prep, verb_stem_index,
        )
        prep_all.extend(pg)
        verb_all.extend(vg)
        n_b1 += diag["n_branch1"]
        n_b2 += diag["n_branch2"]
        n_neither += diag["n_neither"]
        b1_all.extend(diag["b1_cands"])
        b2_all.extend(diag["b2_cands"])
    return prep_all, verb_all, {
        "n_branch1": n_b1,
        "n_branch2": n_b2,
        "n_neither": n_neither,
        "b1_cands": b1_all,
        "b2_cands": b2_all,
    }


def flatten_oracle_case_gov_full(
    joined: list[dict],
    strict_prep: dict[str, str],
    two_way_prep: dict[str, list[str]],
    verb_stem_index: list[tuple[str, str, dict]],
) -> tuple[list[set[str] | None], list[set[str] | None], dict[str, Any]]:
    """Flatten full-cascade oracle gov to stream order (Arm A going forward)."""
    prep_all: list[set[str] | None] = []
    verb_all: list[set[str] | None] = []
    rule_all: dict[str, list[set[str] | None]] = {
        r: [] for r in FULL_ORACLE_RULES if r != "none"
    }
    fired_all: list[str | None] = []
    n_resolved = 0
    n_none = 0
    for s in joined:
        pg, vg, diag = oracle_case_gov_sentence_full(
            s["tokens"], s["upos"], s["head_offset"], s["deprel"],
            strict_prep, two_way_prep, verb_stem_index,
        )
        prep_all.extend(pg)
        verb_all.extend(vg)
        for r, cands in diag["rule_cands"].items():
            rule_all[r].extend(cands)
        fired_all.extend(diag["fired"])
        n_resolved += diag["n_resolved"]
        n_none += diag["n_none"]
    return prep_all, verb_all, {
        "rule_cands": rule_all,
        "fired": fired_all,
        "n_resolved": n_resolved,
        "n_none": n_none,
    }


# Alias used by older call sites / tests that still name flatten_oracle_case_gov
flatten_oracle_case_gov = flatten_oracle_case_gov_full


def branch_precision(
    cands: list[set[str] | None],
    gold_labels: list[str],
) -> dict[str, float | int]:
    """Pre-arbitration: fraction of branch-fired positions where gold ∈ candidate set."""
    hits = 0
    n = 0
    for cand, gold in zip(cands, gold_labels, strict=True):
        if cand is None:
            continue
        n += 1
        if gold in cand:
            hits += 1
    return {
        "n_fired": n,
        "precision": (hits / n) if n else 0.0,
        "n_hit": hits,
    }


def full_rule_diagnostics(
    rule_cands: dict[str, list[set[str] | None]],
    fired: list[str | None],
    gold_labels: list[str],
) -> dict[str, dict[str, float | int]]:
    """Per-rule n_fired, pre-arb precision, and coverage of case-bearing tokens."""
    n_case_bearing = sum(1 for g in gold_labels if g != "-")
    out: dict[str, dict[str, float | int]] = {}
    for rule in FULL_ORACLE_RULES:
        if rule == "none":
            n_fired = 0
            n_hit = 0
            n_fired_cb = 0
            for tag, gold in zip(fired, gold_labels, strict=True):
                if tag is not None:
                    continue
                n_fired += 1
                # "precision" for none is undefined; leave 0 / n_hit=0
                if gold != "-":
                    n_fired_cb += 1
            out["none"] = {
                "n_fired": n_fired,
                "n_hit": n_hit,
                "precision": 0.0,
                "n_fired_case_bearing": n_fired_cb,
                "coverage_case_bearing": (
                    (n_fired_cb / n_case_bearing) if n_case_bearing else 0.0
                ),
                "n_case_bearing_total": n_case_bearing,
            }
            continue
        cands = rule_cands[rule]
        prec = branch_precision(cands, gold_labels)
        n_fired_cb = 0
        for cand, gold in zip(cands, gold_labels, strict=True):
            if cand is not None and gold != "-":
                n_fired_cb += 1
        out[rule] = {
            "n_fired": prec["n_fired"],
            "n_hit": prec["n_hit"],
            "precision": prec["precision"],
            "n_fired_case_bearing": n_fired_cb,
            "coverage_case_bearing": (
                (n_fired_cb / n_case_bearing) if n_case_bearing else 0.0
            ),
            "n_case_bearing_total": n_case_bearing,
        }
    return out


# ---------------------------------------------------------------------------
# Arm C — CASE non-parse heuristic (verb-aware span-stop; NO gold attachment)
# ---------------------------------------------------------------------------

def np_span_indices_verb_aware(
    tokens: list[str],
    start: int,
    prep_forms: set[str],
    conj_forms: set[str],
    verb_stem_index: list[tuple[str, str, dict]],
    safety_cap: int = r1.GOV_SPAN_SAFETY_CAP,
) -> list[int]:
    """Like r1.np_span_indices but also stops at a known verb-stem match (parse-free)."""
    indices: list[int] = []
    end = min(start + safety_cap, len(tokens))
    for j in range(start, end):
        t = tokens[j]
        if r1.is_punct_token(t):
            break
        lj = t.lower()
        if lj in prep_forms or lj in conj_forms:
            break
        if r1.match_verb_stem(t, verb_stem_index) is not None:
            break
        indices.append(j)
    return indices


def prep_gov_case_sets_verb_aware(
    tokens: list[str],
    strict_prep: dict[str, str],
    two_way_prep: dict[str, list[str]],
    prep_forms: set[str],
    conj_forms: set[str],
    verb_stem_index: list[tuple[str, str, dict]],
    safety_cap: int = r1.GOV_SPAN_SAFETY_CAP,
) -> list[set[str] | None]:
    """Prep government over verb-aware NP spans (parse-free; no head_deprel)."""
    sources: list[list[set[str]]] = [[] for _ in tokens]
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in strict_prep:
            case_set = {strict_prep[low]}
        elif low in two_way_prep:
            case_set = set(two_way_prep[low])
        else:
            continue
        for j in np_span_indices_verb_aware(
            tokens, i + 1, prep_forms, conj_forms, verb_stem_index, safety_cap,
        ):
            sources[j].append(set(case_set))
    return [r1._intersect_str_sets(srcs) for srcs in sources]


def verb_gov_case_sets_verb_aware(
    tokens: list[str],
    stem_index: list[tuple[str, str, dict]],
    prep_forms: set[str],
    conj_forms: set[str],
    safety_cap: int = r1.GOV_SPAN_SAFETY_CAP,
) -> list[set[str] | None]:
    """Verb government over verb-aware NP spans (parse-free; no head_deprel)."""
    sources: list[list[set[str]]] = [[] for _ in tokens]
    for i, tok in enumerate(tokens):
        matched = r1.match_verb_stem(tok, stem_index)
        if matched is None:
            continue
        _lemma, entry = matched
        case, _conf = r1.verb_majority_case(entry)
        if case is None:
            continue
        case_set = {case}
        for j in np_span_indices_verb_aware(
            tokens, i + 1, prep_forms, conj_forms, stem_index, safety_cap,
        ):
            sources[j].append(set(case_set))
    return [r1._intersect_str_sets(srcs) for srcs in sources]


def flatten_heuristic_case_gov(
    split: list[dict],
    strict_prep: dict[str, str],
    two_way_prep: dict[str, list[str]],
    prep_forms: set[str],
    conj_forms: set[str],
    verb_stem_index: list[tuple[str, str, dict]],
) -> tuple[list[set[str] | None], list[set[str] | None]]:
    """Flatten Arm C heuristic gov. Takes only surface tokens — no head_deprel."""
    prep_all: list[set[str] | None] = []
    verb_all: list[set[str] | None] = []
    for s in split:
        toks = s["tokens"]
        pg = prep_gov_case_sets_verb_aware(
            toks, strict_prep, two_way_prep, prep_forms, conj_forms, verb_stem_index,
        )
        vg = verb_gov_case_sets_verb_aware(
            toks, verb_stem_index, prep_forms, conj_forms,
        )
        prep_all.extend(pg)
        verb_all.extend(vg)
    return prep_all, verb_all


def stream_with_gov(
    stream: dict[str, Any],
    prep_gov: list[set[str] | None],
    verb_gov: list[set[str] | None],
) -> dict[str, Any]:
    """Shallow copy of stream with overridden prep_gov / verb_gov lists."""
    if len(prep_gov) != stream["n"] or len(verb_gov) != stream["n"]:
        raise RuntimeError(
            f"gov length mismatch: prep={len(prep_gov)} verb={len(verb_gov)} "
            f"stream_n={stream['n']}"
        )
    out = dict(stream)
    out["prep_gov"] = prep_gov
    out["verb_gov"] = verb_gov
    return out


# ---------------------------------------------------------------------------
# Arm B — AUX_VERB gold-oracle (upper bound; head_offset + neighbor surface only)
# ---------------------------------------------------------------------------

def aux_oracle_fires(tokens: list[str], idx: int, head_offset: list[int]) -> bool:
    """AUX iff governor or any child is surface participle / zu-infinitive.

    Uses ONLY head_offset structure (no deprel, no gold UPOS/labels of any token).
    Neighbor classification is pure surface morphology.
    """
    n = len(tokens)
    gov = idx + head_offset[idx]
    if gov != idx and 0 <= gov < n and is_participle_or_zu_inf(tokens, gov):
        return True
    for j in range(n):
        if j != idx and j + head_offset[j] == idx and is_participle_or_zu_inf(tokens, j):
            return True
    return False


# ---------------------------------------------------------------------------
# Arm D — AUX_VERB non-parse clause-bounded search (NO gold attachment)
# ---------------------------------------------------------------------------

def clause_bounds(tokens: list[str], idx: int) -> tuple[int, int]:
    """Punctuation-delimited clause window around idx (inclusive left/right)."""
    left = 0
    for j in range(idx - 1, -1, -1):
        if r1.is_punct_token(tokens[j]):
            left = j + 1
            break
    right = len(tokens) - 1
    for j in range(idx + 1, len(tokens)):
        if r1.is_punct_token(tokens[j]):
            right = j - 1
            break
    return left, right


def aux_heuristic_fires(tokens: list[str], idx: int) -> bool:
    """Parse-free: any other token in the punctuation clause matches participle/zu-inf.

    Never reads head_deprel. Known FN risk: parentheticals that split aux from participle.
    """
    left, right = clause_bounds(tokens, idx)
    for j in range(left, right + 1):
        if j == idx:
            continue
        if is_participle_or_zu_inf(tokens, j):
            return True
    return False


# ---------------------------------------------------------------------------
# AUX form-majority fallback (shared by Arms B and D)
# ---------------------------------------------------------------------------

def fit_aux_form_fallback(
    train: list[dict],
) -> tuple[dict[str, str], str]:
    """TRAIN-only majority AUX/VERB per surface form; global majority if unseen."""
    per_form: dict[str, Counter] = defaultdict(Counter)
    global_c: Counter = Counter()
    for s in train:
        for idx, lab in zip(s["indices"], s["labels"], strict=True):
            form = s["tokens"][idx]
            per_form[form][lab] += 1
            global_c[lab] += 1
    form_maj = {f: c.most_common(1)[0][0] for f, c in per_form.items()}
    gmaj = global_c.most_common(1)[0][0] if global_c else "VERB"
    return form_maj, gmaj


def aux_predict_label(
    form: str,
    fires: bool,
    form_maj: dict[str, str],
    global_maj: str,
) -> str:
    if fires:
        return "AUX"
    return form_maj.get(form, global_maj)


def score_aux_instances(
    split: list[dict],
    fire_fn,
    form_maj: dict[str, str],
    global_maj: str,
    *,
    hd_by_sid: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Score aux_verb instances. fire_fn(tokens, idx, head_offset|None) -> bool."""
    n_total = 0
    n_correct = 0
    n_fire = 0
    n_fire_correct = 0
    for s in split:
        tokens = s["tokens"]
        head_offset = None
        if hd_by_sid is not None:
            hd = hd_by_sid[s["sent_id"]]
            assert_tokens_match(tokens, hd["tokens"], s["sent_id"], "aux_verb", "head_deprel")
            head_offset = hd["head_offset"]
        for idx, gold in zip(s["indices"], s["labels"], strict=True):
            fires = fire_fn(tokens, idx, head_offset)
            pred = aux_predict_label(tokens[idx], fires, form_maj, global_maj)
            n_total += 1
            if pred == gold:
                n_correct += 1
            if fires:
                n_fire += 1
                if pred == gold:
                    n_fire_correct += 1
    acc = n_correct / n_total if n_total else 0.0
    fire_prec = n_fire_correct / n_fire if n_fire else 0.0
    return {
        "accuracy": acc,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_fire": n_fire,
        "fire_rate": n_fire / n_total if n_total else 0.0,
        "precision_when_fired": fire_prec,
    }


# ---------------------------------------------------------------------------
# Verdict / recovery helpers (pure; unit-tested)
# ---------------------------------------------------------------------------

def case_r3_gating(arm_a_acc: float) -> str:
    """CONFIRMED iff Arm A >= 0.88; INCOMPLETE if < 0.85; else PARTIAL."""
    if arm_a_acc >= 0.88:
        return "CONFIRMED"
    if arm_a_acc < 0.85:
        return "INCOMPLETE"
    return "PARTIAL"


def recovery_fraction(heur_gain: float, oracle_gain: float) -> float | str:
    """heuristic_gain / oracle_gain; \"N/A\" if oracle_gain <= 0 (no exception)."""
    if oracle_gain <= 0:
        return "N/A"
    return heur_gain / oracle_gain


def format_recovery(val: float | str) -> str:
    if isinstance(val, str):
        return val
    return f"{val:.4f}"


# ---------------------------------------------------------------------------
# Main campaign
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("German R3-minimal — 4-arm case/aux_verb ladder")
    print("=" * 72)
    print(HONEST_SCOPE)
    print()

    case_baseline, aux_baseline = load_baselines()
    print(
        f"Baselines (loaded from disk, not recomputed): "
        f"morph_case.registers_off={case_baseline:.4f}  "
        f"aux_verb.full={aux_baseline:.4f}"
    )

    # ---- TRAIN + DEV only (test not touched yet) ----
    print("\nLoading TRAIN + DEV (gold tasks only; test not read yet)...")
    train = r1.load_aligned_split("train")
    dev = r1.load_aligned_split("dev")
    aux_train = r2.load_task_split("aux_verb", "train")
    print(
        f"  train sentences={len(train)}  dev sentences={len(dev)}  "
        f"aux_verb train instances="
        f"{sum(len(s['indices']) for s in aux_train)}"
    )

    # Registers (pre-built; not mined from test)
    art_reg = r1.load_register("article_paradigm")
    adj_reg = r1.load_register("adjective_endings")
    pro_reg = r1.load_register("pronoun_paradigm")
    prep_reg = r1.load_register("preposition_case")
    verb_reg = r1.load_register("verb_government")
    det_reg = r1.load_register("det_pron_ambiguous")

    art_f2c = r1.invert_article_paradigm(art_reg["table"])
    pro_f2c = r1.invert_pronoun_paradigm(pro_reg["table"])
    end2c = r1.invert_adjective_endings(adj_reg["table"])
    strict_prep = r1.strict_prepositions(prep_reg["table"])
    two_way_prep = r1.two_way_prepositions(prep_reg["table"])
    prep_forms = r1.all_preposition_forms(prep_reg["table"])
    verb_table = verb_reg["table"]
    verb_stem_index = r1.build_verb_stem_index(verb_table)
    conj_forms = set(CONJ_FORMS)

    extra_forms = (
        list(art_f2c.keys())
        + list(pro_f2c.keys())
        + list(strict_prep.keys())
        + list(two_way_prep.keys())
        + list(prep_forms)
        + list(det_reg["table"].keys())
        + list(verb_table.keys())
    )
    vocab = r1.build_vocab(train, extra_forms)
    labs = r1.label_maps(train)
    case2i = labs["case2i"]
    n_case = len(r1.CASE_LABELS)
    base = len(vocab) + 1
    form_maj_pos, form_pos_supp = r1.train_form_majority_pos(train)

    print(
        f"  vocab_size={len(vocab)}  strict_preps={len(strict_prep)}  "
        f"two_way={len(two_way_prep)}  verb_stems={len(verb_stem_index)}"
    )

    tr_stream = r1.build_stream(
        train, vocab, strict_prep, two_way_prep, prep_forms, conj_forms, verb_stem_index,
    )
    dv_stream = r1.build_stream(
        dev, vocab, strict_prep, two_way_prep, prep_forms, conj_forms, verb_stem_index,
    )
    y_case_tr = r1.flatten_labels(train, "case", case2i)
    y_case_dv = r1.flatten_labels(dev, "case", case2i)
    counts_case = r1.build_counts(tr_stream["surf"], y_case_tr, len(vocab), n_case)
    maj_case = int(counts_case.sum(0).argmax())
    tr_bigram = tr_stream["windows"][:, -2] * base + tr_stream["windows"][:, -1]
    bg_case_keys, bg_case_counts = r1.build_keyed_counts(tr_bigram, y_case_tr, n_case)

    print("\nTuning context-tier (prev1×cur) on DEV for morph_case (inherited R1 grid)...")
    ctx_case, hp_case, acc_case_dv = r1.fit_context_tier(
        tr_stream["windows"], y_case_tr, dv_stream["windows"], y_case_dv,
        counts_case, n_case, base,
    )
    print(f"  case ctx minsupp={hp_case[0]} mindet={hp_case[1]}  dev_acc={acc_case_dv:.4f}")

    form_maj_aux, gmaj_aux = fit_aux_form_fallback(aux_train)
    print(f"  aux form-fallback: n_forms={len(form_maj_aux)}  global_maj={gmaj_aux}")

    # ---- TEST read exactly once ----
    print("\n*** Reading TEST exactly once (final scoring pass only) ***")
    test = r1.load_aligned_split("test")
    hd_test = load_head_deprel_split("test")
    aux_test = r2.load_task_split("aux_verb", "test")
    test_read_once = True

    # Align head_deprel to case/pos test (Arms A/B only)
    test_joined = join_head_deprel(test, hd_test, label="test")
    hd_by_sid = index_by_sent_id(hd_test)
    # Align aux_verb tokens to head_deprel
    for s in aux_test:
        hd = hd_by_sid[s["sent_id"]]
        assert_tokens_match(s["tokens"], hd["tokens"], s["sent_id"], "aux_verb", "head_deprel")

    te_stream = r1.build_stream(
        test, vocab, strict_prep, two_way_prep, prep_forms, conj_forms, verb_stem_index,
    )
    y_case_te = r1.flatten_labels(test, "case", case2i)
    gold_case_str = [r1.CASE_LABELS[int(y)] for y in y_case_te.tolist()]
    te_bigram = te_stream["windows"][:, -2] * base + te_stream["windows"][:, -1]
    print(
        f"  test sentences={len(test)}  tokens={te_stream['n']}  "
        f"aux_verb instances={sum(len(s['indices']) for s in aux_test)}"
    )

    # Parse-free case baseline (registers_off) — should match artifact within 0.01
    pred_case_off = r1.predict_head(
        te_stream["windows"], y_case_te, counts_case, n_case,
        [], {}, ctx_case, base,
        registers_on=False, fill_maj=maj_case, key_mode={"ctx": "ctx"},
    )
    acc_case_off = r1.accuracy(pred_case_off, y_case_te)
    print(
        f"  recomputed registers_off={acc_case_off:.4f}  "
        f"artifact baseline={case_baseline:.4f}  "
        f"delta={acc_case_off - case_baseline:+.4f}"
    )
    if abs(acc_case_off - case_baseline) > 0.01:
        raise RuntimeError(
            f"registers_off mismatch: recomputed={acc_case_off} "
            f"artifact={case_baseline} (tol=0.01)"
        )

    case_form_src = r1.form_case_candidate_sets(
        te_stream["tokens"], art_f2c, pro_f2c, end2c, form_maj_pos, form_pos_supp,
    )

    # ---- Arm C: CASE non-parse heuristic (NO head_deprel) ----
    print("\n--- Arm C: CASE non-parse verb-aware span heuristic ---")
    # Assert: this path never receives head_deprel structures
    heur_prep, heur_verb = flatten_heuristic_case_gov(
        test,  # tokens only; no head_offset/deprel fields used
        strict_prep, two_way_prep, prep_forms, conj_forms, verb_stem_index,
    )
    assert "head_offset" not in test[0] and "deprel" not in test[0], (
        "Arm C must not see head_deprel fields on test sentences"
    )
    stream_c = stream_with_gov(te_stream, heur_prep, heur_verb)
    pred_c = r1.predict_case_or_gnn_registers_on(
        stream_c, case_form_src, case2i, counts_case,
        bg_case_keys, bg_case_counts, te_bigram, n_case,
        soft_fallback=pred_case_off, include_gov=True,
    )
    acc_c = r1.accuracy(pred_c, y_case_te)
    delta_c = acc_c - case_baseline
    per_c = r1.per_class_accuracy(pred_c, y_case_te, r1.CASE_LABELS)
    print(f"  Arm C accuracy={acc_c:.4f}  baseline={case_baseline:.4f}  delta={delta_c:+.4f}")
    print("  Arm C per-class:")
    for lab in r1.CASE_LABELS:
        info = per_c[lab]
        print(f"    {lab:>4}: acc={info['acc']:.4f}  support={info['support']}")
    print("  [empirical] Arm C is parse-free (no head_deprel).")

    # ---- Arm A: CASE gold-oracle (upper bound) — partial (regression) + full ----
    print("\n--- Arm A: CASE gold-oracle rescore (UPPER BOUND; not serve-honest) ---")

    # Partial oracle (previous 2-branch rule set) — regression-checked against 0.8262
    ora_prep_p, ora_verb_p, ora_diag_p = flatten_oracle_case_gov_partial(
        test_joined, strict_prep, two_way_prep, verb_stem_index,
    )
    stream_a_partial = stream_with_gov(te_stream, ora_prep_p, ora_verb_p)
    pred_a_partial = r1.predict_case_or_gnn_registers_on(
        stream_a_partial, case_form_src, case2i, counts_case,
        bg_case_keys, bg_case_counts, te_bigram, n_case,
        soft_fallback=pred_case_off, include_gov=True,
    )
    acc_a_partial = r1.accuracy(pred_a_partial, y_case_te)
    if abs(acc_a_partial - PARTIAL_ORACLE_ACC_EXPECTED) > 1e-4:
        raise RuntimeError(
            f"partial oracle regression failed: got {acc_a_partial:.6f} "
            f"expected {PARTIAL_ORACLE_ACC_EXPECTED:.6f} (tol=1e-4)"
        )
    b1_prec = branch_precision(ora_diag_p["b1_cands"], gold_case_str)
    b2_prec = branch_precision(ora_diag_p["b2_cands"], gold_case_str)
    print(
        f"  Arm A (partial oracle, previous rule set) = {acc_a_partial:.4f} "
        f"(unchanged, regression-checked vs {PARTIAL_ORACLE_ACC_EXPECTED:.4f})"
    )
    print(
        f"    partial branch1 n_fired={ora_diag_p['n_branch1']} "
        f"pre-arb_prec={b1_prec['precision']:.4f}; "
        f"branch2 n_fired={ora_diag_p['n_branch2']} "
        f"pre-arb_prec={b2_prec['precision']:.4f} "
        f"(verbgov residual still weak ~0.64 — second limiter)"
    )

    # Full cascade — Arm A going forward
    ora_prep, ora_verb, ora_diag = flatten_oracle_case_gov_full(
        test_joined, strict_prep, two_way_prep, verb_stem_index,
    )
    stream_a = stream_with_gov(te_stream, ora_prep, ora_verb)
    pred_a = r1.predict_case_or_gnn_registers_on(
        stream_a, case_form_src, case2i, counts_case,
        bg_case_keys, bg_case_counts, te_bigram, n_case,
        soft_fallback=pred_case_off, include_gov=True,
    )
    acc_a = r1.accuracy(pred_a, y_case_te)
    delta_a = acc_a - case_baseline
    delta_a_vs_partial = acc_a - acc_a_partial
    per_a = r1.per_class_accuracy(pred_a, y_case_te, r1.CASE_LABELS)
    case_null = r1.null_floor(y_case_te, case2i["-"])
    rule_diag = full_rule_diagnostics(
        ora_diag["rule_cands"], ora_diag["fired"], gold_case_str,
    )
    print(
        f"  Arm A (full oracle rule set) = {acc_a:.4f}  "
        f"delta over partial = {delta_a_vs_partial:+.4f}  "
        f"delta over baseline (~{case_baseline:.4f}) = {delta_a:+.4f}"
    )
    approaches_90 = acc_a >= 0.90
    print(
        f"  full oracle APPROACHES 0.90? {'YES' if approaches_90 else 'NO'} "
        f"(acc={acc_a:.4f})"
    )
    print(f"  null floor (gold=='-'): {case_null:.4f}")
    print("  Arm A (full) per-class:")
    for lab in r1.CASE_LABELS:
        info = per_a[lab]
        print(f"    {lab:>4}: acc={info['acc']:.4f}  support={info['support']}")
    print(
        "  full-cascade per-rule diagnostics on TEST "
        "(n_fired, pre-arb precision=gold∈cand, coverage of case-bearing tokens):"
    )
    n_cb = next(iter(rule_diag.values()))["n_case_bearing_total"]
    print(f"    case-bearing tokens (gold case != '-'): {n_cb}")
    for rule in FULL_ORACLE_RULES:
        info = rule_diag[rule]
        note = ""
        if rule == "iobj":
            note = (
                "  [literal iobj→Dat; expect ~0 coverage — GSD barely uses iobj]"
            )
        elif rule == "obl_arg":
            note = (
                "  [DATA-DRIVEN addition: GSD bare dative objects are obl:arg, not iobj]"
            )
        elif rule == "verbgov":
            note = "  [residual; expect weak precision ~0.64 — second limiter]"
        print(
            f"    {rule:<10}: n_fired={info['n_fired']:>5}  "
            f"pre-arb_prec={info['precision']:.4f}  "
            f"n_case_bearing={info['n_fired_case_bearing']:>5}  "
            f"coverage={info['coverage_case_bearing']:.4f}"
            f"{note}"
        )
    print(
        "  [empirical, upper-bound (oracle)] Arm A FULL uses gold head_offset/deprel "
        "(+ gold UPOS of gov/children); NOT serve-honest. "
        "obl:arg→Dat is a data-driven addition beyond the literal iobj rule."
    )

    # ---- Arm D: AUX non-parse clause-bounded search (NO head_deprel) ----
    print("\n--- Arm D: AUX_VERB non-parse clause-bounded heuristic ---")

    def _fire_d(tokens: list[str], idx: int, _ho: list[int] | None) -> bool:
        return aux_heuristic_fires(tokens, idx)

    # hd_by_sid deliberately NOT passed — Arm D is parse-free by construction
    score_d = score_aux_instances(
        aux_test, _fire_d, form_maj_aux, gmaj_aux, hd_by_sid=None,
    )
    acc_d = score_d["accuracy"]
    delta_d = acc_d - aux_baseline
    print(f"  Arm D accuracy={acc_d:.4f}  baseline={aux_baseline:.4f}  delta={delta_d:+.4f}")
    print(
        f"  firing: n_fire={score_d['n_fire']}/{score_d['n_total']}  "
        f"rate={score_d['fire_rate']:.4f}  "
        f"precision_when_fired={score_d['precision_when_fired']:.4f}"
    )
    print(
        "  [empirical] Arm D never reads head_deprel (clause-bounded surface search only)."
    )

    # ---- Arm B: AUX gold-oracle ----
    print("\n--- Arm B: AUX_VERB gold-oracle (UPPER BOUND; not serve-honest) ---")

    def _fire_b(tokens: list[str], idx: int, ho: list[int] | None) -> bool:
        assert ho is not None
        return aux_oracle_fires(tokens, idx, ho)

    score_b = score_aux_instances(
        aux_test, _fire_b, form_maj_aux, gmaj_aux, hd_by_sid=hd_by_sid,
    )
    acc_b = score_b["accuracy"]
    delta_b = acc_b - aux_baseline
    print(f"  Arm B accuracy={acc_b:.4f}  baseline={aux_baseline:.4f}  delta={delta_b:+.4f}")
    print(
        f"  firing: n_fire={score_b['n_fire']}/{score_b['n_total']}  "
        f"rate={score_b['fire_rate']:.4f}  "
        f"precision_when_fired={score_b['precision_when_fired']:.4f}"
    )
    print(
        "  CIRCULARITY/COVERAGE: no gold label/deprel/UPOS of the ambiguous verb itself "
        "is used — only head_offset links + neighbor SURFACE shape. "
        f"Coverage (fire rate)={score_b['fire_rate']:.4f}; "
        "when the rule does not fire, TRAIN form-majority memorizer is used. "
        "Still an upper-bound structural oracle (gold attachment structure), not serve-honest."
    )
    print("  [empirical, upper-bound (oracle)]")

    # ---- Ladder scoreboard ----
    case_heur_gain = acc_c - case_baseline
    case_oracle_gain = acc_a - case_baseline
    case_rec = recovery_fraction(case_heur_gain, case_oracle_gain)
    aux_heur_gain = acc_d - aux_baseline
    aux_oracle_gain = acc_b - aux_baseline
    aux_rec = recovery_fraction(aux_heur_gain, aux_oracle_gain)

    print()
    print("=" * 72)
    print("LADDER SCOREBOARD  (GSD test; 3 rungs per task)")
    print("=" * 72)
    hdr = (
        f"{'task':<12} {'baseline':>10} {'heuristic':>10} {'oracle':>10} "
        f"{'heur_gain':>10} {'oracle_gain':>12} {'recovery_frac':>14}"
    )
    print(hdr)
    print("-" * len(hdr))
    print(
        f"{'morph_case':<12} {case_baseline:10.4f} {acc_c:10.4f} {acc_a:10.4f} "
        f"{case_heur_gain:+10.4f} {case_oracle_gain:+12.4f} "
        f"{format_recovery(case_rec):>14}"
    )
    print(
        f"{'aux_verb':<12} {aux_baseline:10.4f} {acc_d:10.4f} {acc_b:10.4f} "
        f"{aux_heur_gain:+10.4f} {aux_oracle_gain:+12.4f} "
        f"{format_recovery(aux_rec):>14}"
    )

    # ---- Verdicts ----
    case_gate = case_r3_gating(acc_a)
    print()
    print("=" * 72)
    print("VERDICTS")
    print("=" * 72)
    print(
        f"CASE R3-GATING: {case_gate}  "
        f"(Arm A={acc_a:.4f}, delta vs baseline={delta_a:+.4f})"
    )
    print(
        "  rule: CONFIRMED iff Arm A >= 0.88; INCOMPLETE if < 0.85; else PARTIAL "
        "[empirical, upper-bound (oracle)]"
    )
    print(
        f"CASE HEURISTIC RECOVERY: {format_recovery(case_rec)}  "
        f"(heur_gain={case_heur_gain:+.4f} / oracle_gain={case_oracle_gain:+.4f})"
    )
    if case_rec == "N/A":
        case_interp = (
            "oracle_gain non-positive — recovery undefined; oracle did not lift baseline."
        )
    elif isinstance(case_rec, float) and case_rec < 0:
        case_interp = (
            "cheap parse-free fix HURTS relative to baseline while gold attachment lifts it; "
            "genuine attachment is required (heuristic does not close the gap)."
        )
    elif isinstance(case_rec, float) and case_rec >= 0.7:
        case_interp = (
            "cheap parse-free fix already closes most of the R3-attachment ceiling; "
            "genuine attachment still needed for the residual."
        )
    elif isinstance(case_rec, float) and case_rec >= 0.3:
        case_interp = (
            "parse-free fix recovers a meaningful fraction but genuine attachment "
            "is still required for the majority of the oracle gap."
        )
    else:
        case_interp = (
            "cheap parse-free fix recovers little of the oracle ceiling; "
            "genuine attachment is required."
        )
    print(f"  interpretation: {case_interp}")

    print()
    print(
        f"AUX_VERB oracle (Arm B): {acc_b:.4f}  delta={delta_b:+.4f}  "
        f"[empirical, upper-bound (oracle)]"
    )
    print(
        f"  circularity/coverage caveat: fire_rate={score_b['fire_rate']:.4f}, "
        f"precision_when_fired={score_b['precision_when_fired']:.4f}; "
        "no gold label of the ambiguous verb itself is used."
    )
    print(
        f"AUX_VERB heuristic (Arm D): {acc_d:.4f}  delta={delta_d:+.4f}  [empirical]"
    )
    print(
        f"AUX HEURISTIC RECOVERY: {format_recovery(aux_rec)}  "
        f"(heur_gain={aux_heur_gain:+.4f} / oracle_gain={aux_oracle_gain:+.4f})"
    )
    if aux_rec == "N/A":
        aux_interp = (
            "oracle_gain non-positive — recovery undefined; oracle did not lift baseline."
        )
    elif isinstance(aux_rec, float) and aux_rec >= 0.7:
        aux_interp = (
            "clause-bounded surface search already closes most of the structural ceiling."
        )
    elif isinstance(aux_rec, float) and aux_rec >= 0.3:
        aux_interp = (
            "clause-bounded search recovers a partial share of the oracle gap; "
            "gold attachment still adds headroom."
        )
    else:
        aux_interp = (
            "clause-bounded search recovers little of the oracle gap; "
            "attachment (or better surface cues) still matter."
        )
    print(f"  interpretation: {aux_interp}")

    dhash = data_hash()
    print()
    print(f"data-hash (sha256 of task jsonl files): {dhash}")
    print(
        f"test-read-once confirmation: test was read exactly once "
        f"(test_read_once={test_read_once}), only for this final scoring pass; "
        f"no test-set access during fitting or DEV threshold tuning. "
        f"Arms A/B used gold head_deprel test in that single scoring pass; "
        f"Arms C/D never touch head_deprel."
    )
    print(HONEST_SCOPE)
    print("=" * 72)

    artifact = {
        "rung": "R3-minimal",
        "verdict_tag_oracle": "empirical, upper-bound (oracle)",
        "verdict_tag_heuristic": "empirical",
        "case_r3_gating": case_gate,
        "case_heuristic_recovery_interpretation": case_interp,
        "aux_heuristic_recovery_interpretation": aux_interp,
        "data_hash": dhash,
        "test_read_once": test_read_once,
        "honest_scope": HONEST_SCOPE,
        "baselines": {
            "morph_case_registers_off": case_baseline,
            "aux_verb_full": aux_baseline,
            "source": {
                "morph_case": str(BASELINE_R1),
                "aux_verb": str(BASELINE_R2),
            },
        },
        "recomputed_registers_off": acc_case_off,
        "ctx_hyperparams": {
            "morph_case": {
                "minsupp": hp_case[0],
                "mindet": hp_case[1],
                "dev_acc": acc_case_dv,
            },
        },
        "ladder": {
            "morph_case": {
                "baseline": case_baseline,
                "heuristic": acc_c,
                "oracle": acc_a,
                "heur_gain": case_heur_gain,
                "oracle_gain": case_oracle_gain,
                "recovery_frac": case_rec,
            },
            "aux_verb": {
                "baseline": aux_baseline,
                "heuristic": acc_d,
                "oracle": acc_b,
                "heur_gain": aux_heur_gain,
                "oracle_gain": aux_oracle_gain,
                "recovery_frac": aux_rec,
            },
        },
        "scores": {
            "morph_case": {
                "baseline": case_baseline,
                "arm_c_heuristic": acc_c,
                "arm_a_oracle": acc_a,
                "arm_a_oracle_partial": acc_a_partial,
                "arm_a_oracle_full": acc_a,
                "delta_c": delta_c,
                "delta_a": delta_a,
                "delta_a_vs_partial": delta_a_vs_partial,
                "approaches_0_90": approaches_90,
                "null_floor": case_null,
                "per_class_arm_c": per_c,
                "per_class_arm_a": per_a,
                "partial_oracle_diagnostics": {
                    "n_branch1": ora_diag_p["n_branch1"],
                    "n_branch2": ora_diag_p["n_branch2"],
                    "n_neither": ora_diag_p["n_neither"],
                    "branch1_precision": b1_prec,
                    "branch2_precision": b2_prec,
                    "regression_expected": PARTIAL_ORACLE_ACC_EXPECTED,
                    "regression_ok": abs(acc_a_partial - PARTIAL_ORACLE_ACC_EXPECTED) <= 1e-4,
                },
                "full_rule_diagnostics": rule_diag,
                "branch_diagnostics": {
                    # legacy keys kept for readers expecting branch1/branch2
                    "n_branch1": rule_diag["prep"]["n_fired"],
                    "n_branch2": rule_diag["verbgov"]["n_fired"],
                    "n_neither": rule_diag["none"]["n_fired"],
                    "branch1_precision": {
                        "n_fired": rule_diag["prep"]["n_fired"],
                        "precision": rule_diag["prep"]["precision"],
                        "n_hit": rule_diag["prep"]["n_hit"],
                    },
                    "branch2_precision": {
                        "n_fired": rule_diag["verbgov"]["n_fired"],
                        "precision": rule_diag["verbgov"]["precision"],
                        "n_hit": rule_diag["verbgov"]["n_hit"],
                    },
                },
            },
            "aux_verb": {
                "baseline": aux_baseline,
                "arm_d_heuristic": acc_d,
                "arm_b_oracle": acc_b,
                "delta_d": delta_d,
                "delta_b": delta_b,
                "arm_d": {
                    "n_total": score_d["n_total"],
                    "n_fire": score_d["n_fire"],
                    "fire_rate": score_d["fire_rate"],
                    "precision_when_fired": score_d["precision_when_fired"],
                },
                "arm_b": {
                    "n_total": score_b["n_total"],
                    "n_fire": score_b["n_fire"],
                    "fire_rate": score_b["fire_rate"],
                    "precision_when_fired": score_b["precision_when_fired"],
                    "circularity_note": (
                        "no gold label/deprel/UPOS of the ambiguous verb itself; "
                        "only head_offset + neighbor surface shape"
                    ),
                },
            },
        },
        "n_test_sentences": len(test),
        "n_test_tokens": te_stream["n"],
        "n_aux_test_instances": score_b["n_total"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
