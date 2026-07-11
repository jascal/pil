"""Synthetic join battery: planted, provably-determinate join topologies.

Builds regimes S (schema-determined), L (lexically-determined), and N
(underdetermined control) as CFQ-format SPARQL so ``parse_sparql_joins`` /
``join_f1`` and the typed assembler can be reused without forks.

Stdlib only — no torch, no third-party imports.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from random import Random

from pil.cfq_edges import parse_sparql_joins

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatteryExample:
    question: str  # space-joined tokens
    sparql: str  # CFQ-format: "SELECT count(*) WHERE {\n<triples>\n}"
    topology: str  # "star" | "chain" | "cycle"
    key: tuple[str, ...]  # sorted predicate multiset


@dataclass
class BatterySplit:
    train: list[BatteryExample]
    val: list[BatteryExample]
    test_iid: list[BatteryExample]
    test_comp: list[BatteryExample]  # empty for regime N


# ---------------------------------------------------------------------------
# Constants / naming
# ---------------------------------------------------------------------------

N_TYPES_S = 24
N_PREDS_S = 33
# Role-pool split of 33 predicates across 9 (topology, position) slots:
#   star  positions 0..2 : 4 preds each  (12)
#   chain positions 0..2 : 4 preds each  (12)
#   cycle positions 0..2 : 3 preds each  (9)
# Total 33. Sized so distinct train keys/sigs fit the admission budget
# (expressibility invariant: n_distinct_sigs <= 100, n_distinct_keys <= 250).
ROLE_POOL_SIZES: dict[str, tuple[int, int, int]] = {
    "star": (4, 4, 4),
    "chain": (4, 4, 4),
    "cycle": (3, 3, 3),
}

# Regime L/N predicate pool size (combinations of 3 distinct preds).
# C(6, 3) = 20 combos (was C(12, 3) = 220) — fits admission budget.
L_POOL_SIZE = 6

TRAIN_S = 8000
VAL_S = 400
TEST_IID_S = 1000
TEST_COMP_S = 1000

TRAIN_N = 4000
VAL_N = 400
TEST_IID_N = 1000


def pred_name(i: int) -> str:
    return f"ns:syn.p{i}"


def type_name(j: int) -> str:
    return f"ns:syn.t{j}"


def word_for_pred(pred: str) -> str:
    """Fixed pred -> surface word: ns:syn.p{i} -> w{i}."""
    assert pred.startswith("ns:syn.p")
    return "w" + pred[len("ns:syn.p") :]


def entity_token(i: int) -> str:
    return f"M{i}"


def entity_word(i: int) -> str:
    return f"e{i}"


# ---------------------------------------------------------------------------
# SPARQL emission helpers
# ---------------------------------------------------------------------------


def _format_sparql(
    edge_lines: list[str],
    type_lines: list[str],
) -> str:
    body = "\n".join(edge_lines + type_lines)
    return f"SELECT count(*) WHERE {{\n{body}\n}}"


def _assert_has_join(sparql: str) -> None:
    jq = parse_sparql_joins(sparql)
    assert jq.n_join_vars >= 1, f"expected >=1 join var, got {jq.n_join_vars}: {sparql!r}"


def _shuffle_tokens(rng: Random, tokens: list[str]) -> str:
    toks = list(tokens)
    rng.shuffle(toks)
    return " ".join(toks)


# ---------------------------------------------------------------------------
# Self-certification (regime S): planted sigs vs schema type-merge
# ---------------------------------------------------------------------------

# Schema: pred -> (subj_type, obj_type)
Schema = dict[str, tuple[str, str]]

# One edge as (subj_token, pred, obj_token); vars start with '?'
EdgeTriple = tuple[str, str, str]


def planted_join_signatures(
    edges: list[EdgeTriple],
    var_types: dict[str, str],
) -> list[tuple]:
    """Direct planted join signatures from generator bookkeeping.

    Same shape as ``parse_sparql_joins``: list of
    ``(var_type, sorted((pred, slot), ...))`` for vars with >=2 incidences.
    """
    incidences: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for subj, pred, obj in edges:
        if subj.startswith("?"):
            incidences[subj].append((pred, "subj"))
        if obj.startswith("?"):
            incidences[obj].append((pred, "obj"))
    sigs: list[tuple] = []
    for var, incs in incidences.items():
        if len(incs) < 2:
            continue
        sig = (var_types[var], tuple(sorted(incs)))
        sigs.append(sig)
    sigs.sort()
    return sigs


def type_merge_signatures(
    edges: list[EdgeTriple],
    schema: Schema,
) -> list[tuple]:
    """Schema type-merge closure over var-occupied slots of ``edges``.

    Groups incidences by the required type of that (pred, slot) under the
    planted schema. Groups with >=2 incidences emit a join signature whose
    surface label is the shared required type.
    """
    # Collect var-slot incidences with their required schema type.
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for subj, pred, obj in edges:
        st, ot = schema[pred]
        if subj.startswith("?"):
            groups[st].append((pred, "subj"))
        if obj.startswith("?"):
            groups[ot].append((pred, "obj"))
    sigs: list[tuple] = []
    for req_type, members in groups.items():
        if len(members) < 2:
            continue
        sigs.append((req_type, tuple(sorted(members))))
    sigs.sort()
    return sigs


def self_certifies(
    edges: list[EdgeTriple],
    var_types: dict[str, str],
    schema: Schema,
) -> bool:
    """True iff type-merge signature multiset equals planted signature multiset."""
    planted = planted_join_signatures(edges, var_types)
    merged = type_merge_signatures(edges, schema)
    return Counter(planted) == Counter(merged)


# ---------------------------------------------------------------------------
# Regime S schema + role pools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeSSchema:
    """Fixed schema + role pools for regime S (deterministic from seed)."""

    schema: Schema  # pred -> (subj_type, obj_type)
    # topology -> list of 3 role pools (each a list of pred names)
    role_pools: dict[str, list[list[str]]]
    # All predicates
    all_preds: list[str]


def build_regime_s_schema(seed: int = 0) -> RegimeSSchema:
    """Partition 33 preds into 9 role pools and assign fixed (subj,obj) types.

    Type plan (uses types t0..t6; remaining t7..t23 unused but reserved):
      - star center type: per-predicate subj_type drawn from {t0, t1, t2}
        so only same-center-type triples self-certify (rejection sampling).
        star obj_type is a dummy entity-side type t10 (objects are entities).
      - chain: chain_0.obj / chain_1.subj share type keyed by pred index mod 2
        (t3 or t4); chain_1.obj / chain_2.subj share the other (t4 or t3 flipped
        per cohort) so compatible triples exist and incompatible ones reject.
        Simpler fixed plan used below.
      - cycle: three distinct wrap types t5,t6,t7 assigned by role.

    Chain fixed plan (always compatible when one pred per role):
      chain_0: subj=t11 (entity-ish), obj=t3
      chain_1: subj=t3, obj=t4
      chain_2: subj=t4, obj=t12
    Cycle fixed plan:
      cycle_0: subj=t5, obj=t6
      cycle_1: subj=t6, obj=t7
      cycle_2: subj=t7, obj=t5

    Star: subj types mixed across {t0,t1,t2} so random 3-role picks often
    mismatch and reject; valid same-type picks certify cleanly.
    """
    rng = Random(seed ^ 0x5C0DE)
    preds = [pred_name(i) for i in range(N_PREDS_S)]
    # Shuffle then cut into role pools for determinism under seed.
    order = list(preds)
    rng.shuffle(order)
    role_pools: dict[str, list[list[str]]] = {}
    idx = 0
    for topo, sizes in ROLE_POOL_SIZES.items():
        pools: list[list[str]] = []
        for sz in sizes:
            pools.append(order[idx : idx + sz])
            idx += sz
        role_pools[topo] = pools
    assert idx == N_PREDS_S

    schema: Schema = {}
    # Star: mixed center types for rejection sampling; obj dummy.
    star_center_types = [type_name(0), type_name(1), type_name(2)]
    for pool in role_pools["star"]:
        for p in pool:
            ct = rng.choice(star_center_types)
            schema[p] = (ct, type_name(10))

    # Chain: fixed compatible types across roles.
    for p in role_pools["chain"][0]:
        schema[p] = (type_name(11), type_name(3))
    for p in role_pools["chain"][1]:
        schema[p] = (type_name(3), type_name(4))
    for p in role_pools["chain"][2]:
        schema[p] = (type_name(4), type_name(12))

    # Cycle: cyclic type wrap.
    for p in role_pools["cycle"][0]:
        schema[p] = (type_name(5), type_name(6))
    for p in role_pools["cycle"][1]:
        schema[p] = (type_name(6), type_name(7))
    for p in role_pools["cycle"][2]:
        schema[p] = (type_name(7), type_name(5))

    assert len(schema) == N_PREDS_S
    return RegimeSSchema(schema=schema, role_pools=role_pools, all_preds=preds)


def _build_star_s(
    p0: str, p1: str, p2: str, schema: Schema, ent_base: int,
) -> tuple[list[EdgeTriple], dict[str, str], str] | None:
    """Build star edges or None if schema center types mismatch / fail certify."""
    st0, _ = schema[p0]
    st1, _ = schema[p1]
    st2, _ = schema[p2]
    if not (st0 == st1 == st2):
        return None
    center = "?x0"
    m0, m1, m2 = entity_token(ent_base), entity_token(ent_base + 1), entity_token(ent_base + 2)
    edges: list[EdgeTriple] = [
        (center, p0, m0),
        (center, p1, m1),
        (center, p2, m2),
    ]
    var_types = {center: st0}
    if not self_certifies(edges, var_types, schema):
        return None
    return edges, var_types, "star"


def _build_chain_s(
    p0: str, p1: str, p2: str, schema: Schema, ent_base: int,
) -> tuple[list[EdgeTriple], dict[str, str], str] | None:
    """Chain: M p0 ?x0 . ?x0 p1 ?x1 . ?x1 p2 M ."""
    _s0, o0 = schema[p0]
    s1, o1 = schema[p1]
    s2, _o2 = schema[p2]
    if o0 != s1 or o1 != s2:
        return None
    m_in = entity_token(ent_base)
    m_out = entity_token(ent_base + 1)
    x0, x1 = "?x0", "?x1"
    edges: list[EdgeTriple] = [
        (m_in, p0, x0),
        (x0, p1, x1),
        (x1, p2, m_out),
    ]
    var_types = {x0: o0, x1: o1}
    if not self_certifies(edges, var_types, schema):
        return None
    return edges, var_types, "chain"


def _build_cycle_s(
    p0: str, p1: str, p2: str, schema: Schema, ent_base: int,
) -> tuple[list[EdgeTriple], dict[str, str], str] | None:
    """Cycle: ?x0 p0 ?x1 . ?x1 p1 ?x2 . ?x2 p2 ?x0 ."""
    del ent_base  # no entities in pure cycle
    s0, o0 = schema[p0]
    s1, o1 = schema[p1]
    s2, o2 = schema[p2]
    if o0 != s1 or o1 != s2 or o2 != s0:
        return None
    x0, x1, x2 = "?x0", "?x1", "?x2"
    edges: list[EdgeTriple] = [
        (x0, p0, x1),
        (x1, p1, x2),
        (x2, p2, x0),
    ]
    # Each var's type = its subj_type when used as subject of its outgoing edge
    # ?x0 subj of p0 -> s0; also obj of p2 -> o2 == s0
    # ?x1 subj of p1 -> s1; also obj of p0 -> o0 == s1
    # ?x2 subj of p2 -> s2; also obj of p1 -> o1 == s2
    var_types = {x0: s0, x1: s1, x2: s2}
    if not self_certifies(edges, var_types, schema):
        return None
    return edges, var_types, "cycle"


def _edges_to_example(
    edges: list[EdgeTriple],
    var_types: dict[str, str],
    topology: str,
    rng: Random,
) -> BatteryExample:
    edge_lines = [f"{s} {p} {o} ." for s, p, o in edges]
    type_lines = [f"{v} a {t} ." for v, t in sorted(var_types.items())]
    sparql = _format_sparql(edge_lines, type_lines)
    _assert_has_join(sparql)
    preds = [p for _, p, _ in edges]
    key = tuple(sorted(preds))
    tokens: list[str] = [word_for_pred(p) for p in preds]
    # Entity surface words for each concrete M-token used.
    for s, _p, o in edges:
        for tok in (s, o):
            if tok.startswith("M"):
                tokens.append(entity_word(int(tok[1:])))
    question = _shuffle_tokens(rng, tokens)
    return BatteryExample(
        question=question, sparql=sparql, topology=topology, key=key,
    )


def enumerate_valid_s_keys(rs: RegimeSSchema) -> dict[str, list[tuple[str, str, str]]]:
    """Enumerate valid (p0,p1,p2) triples per topology under schema constraints."""
    out: dict[str, list[tuple[str, str, str]]] = {"star": [], "chain": [], "cycle": []}
    builders = {
        "star": _build_star_s,
        "chain": _build_chain_s,
        "cycle": _build_cycle_s,
    }
    for topo, builder in builders.items():
        pools = rs.role_pools[topo]
        for triple in product(pools[0], pools[1], pools[2]):
            built = builder(triple[0], triple[1], triple[2], rs.schema, 0)
            if built is not None:
                out[topo].append(triple)
    return out


def _partition_keys(
    all_keys: list[tuple[str, ...]],
    rng: Random,
    holdout_frac: float = 0.25,
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Split combination keys into train-eligible (~75%) and held-out (~25%).

    Guarantees every individual predicate appearing in ``all_keys`` is present
    in at least one train-eligible key (by force-including a covering set).
    """
    keys = list(all_keys)
    rng.shuffle(keys)
    n_hold = max(1, int(round(len(keys) * holdout_frac)))
    # Ensure holdout is not the entire set
    n_hold = min(n_hold, len(keys) - 1) if len(keys) > 1 else 0

    # Cover each predicate with at least one train key.
    pred_to_keys: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for k in keys:
        for p in set(k):
            pred_to_keys[p].append(k)
    all_preds = sorted(pred_to_keys.keys())
    train_forced: set[tuple[str, ...]] = set()
    for p in all_preds:
        # Prefer a key not already forced; pick lex-smallest for determinism
        options = sorted(pred_to_keys[p])
        chosen = None
        for opt in options:
            if opt not in train_forced:
                chosen = opt
                break
        if chosen is None:
            chosen = options[0]
        train_forced.add(chosen)

    remaining = [k for k in keys if k not in train_forced]
    rng.shuffle(remaining)
    # Holdout drawn from remaining only (forced keys stay train-eligible)
    holdout = remaining[:n_hold]
    hold_set = set(holdout)
    train_eligible = [k for k in keys if k not in hold_set]
    # Sanity: forced cover present
    for p in all_preds:
        assert any(p in k for k in train_eligible), f"pred {p} missing from train pool"
    return train_eligible, holdout


def _sample_s_example(
    rng: Random,
    rs: RegimeSSchema,
    valid: dict[str, list[tuple[str, str, str]]],
    key_pool: set[tuple[str, ...]],
    ent_counter: list[int],
) -> BatteryExample:
    """Rejection-sample one S example whose key is in ``key_pool``."""
    # Build flat list of (topo, triple) with key in pool
    candidates: list[tuple[str, tuple[str, str, str]]] = []
    for topo, triples in valid.items():
        for t in triples:
            k = tuple(sorted(t))
            if k in key_pool:
                candidates.append((topo, t))
    if not candidates:
        raise RuntimeError("empty S candidate pool for key_pool")
    builders = {
        "star": _build_star_s,
        "chain": _build_chain_s,
        "cycle": _build_cycle_s,
    }
    # Rejection sampling (schema mismatch / self-cert fail)
    for _ in range(10_000):
        topo, triple = candidates[rng.randrange(len(candidates))]
        ent_base = ent_counter[0]
        ent_counter[0] += 3
        built = builders[topo](triple[0], triple[1], triple[2], rs.schema, ent_base)
        if built is None:
            continue
        edges, var_types, topology = built
        # Self-cert is already checked in builders; assert again for the guarantee.
        assert self_certifies(edges, var_types, rs.schema)
        ex_rng = Random(rng.randrange(1 << 30))
        return _edges_to_example(edges, var_types, topology, ex_rng)
    raise RuntimeError("S rejection sampling failed after 10000 tries")


def generate_regime_s(seed: int = 0) -> BatterySplit:
    """Generate regime S (schema-determined stars/chains/cycles)."""
    rs = build_regime_s_schema(seed)
    valid = enumerate_valid_s_keys(rs)
    # Flatten valid triples to multiset keys (sorted preds).
    all_keys: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for triples in valid.values():
        for t in triples:
            k = tuple(sorted(t))
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    all_keys.sort()
    train_pool, holdout_pool = _partition_keys(all_keys, Random(seed ^ 0xA11))
    train_set = set(train_pool)
    hold_set = set(holdout_pool)

    ent_counter = [0]

    def sample_block(n: int, pool: set[tuple[str, ...]], block_seed: int) -> list[BatteryExample]:
        brng = Random(block_seed)
        out: list[BatteryExample] = []
        for i in range(n):
            out.append(
                _sample_s_example(brng, rs, valid, pool, ent_counter)
            )
            _ = i
        return out

    train = sample_block(TRAIN_S, train_set, seed ^ 0x100)
    val = sample_block(VAL_S, train_set, seed ^ 0x200)
    test_iid = sample_block(TEST_IID_S, train_set, seed ^ 0x300)
    test_comp = sample_block(TEST_COMP_S, hold_set, seed ^ 0x400)

    # Post-generation assertions
    train_keys = {ex.key for ex in train}
    comp_keys = {ex.key for ex in test_comp}
    assert train_keys.isdisjoint(comp_keys) or not (train_keys & comp_keys), (
        "test_comp keys must be disjoint from train keys"
    )
    # Every individual predicate appears in train
    train_preds: set[str] = set()
    for ex in train:
        train_preds.update(ex.key)
    # Force-cover: if any train-pool pred is missing from train samples, inject.
    for p in sorted({pp for k in train_set for pp in k}):
        if p not in train_preds:
            # Find a valid triple in train_set containing p and append one example
            injected = False
            for topo, triples in valid.items():
                for t in triples:
                    k = tuple(sorted(t))
                    if k in train_set and p in k:
                        builders = {
                            "star": _build_star_s,
                            "chain": _build_chain_s,
                            "cycle": _build_cycle_s,
                        }
                        built = builders[topo](t[0], t[1], t[2], rs.schema, ent_counter[0])
                        ent_counter[0] += 3
                        if built is None:
                            continue
                        edges, var_types, topology = built
                        # Deterministic per-key seed (no PYTHONHASHSEED dependence).
                        kseed = seed
                        for tok in k:
                            kseed = (kseed * 1000003 + sum(map(ord, tok))) & 0xFFFFFFFF
                        ex = _edges_to_example(
                            edges, var_types, topology, Random(kseed),
                        )
                        train.append(ex)
                        train_preds.update(ex.key)
                        injected = True
                        break
                if injected:
                    break
            assert injected, f"failed to inject train cover for predicate {p}"

    # Re-check disjointness after inject
    train_keys = {ex.key for ex in train}
    assert not (train_keys & comp_keys), "test_comp keys leaked into train"

    # Every pred that appears anywhere in train-eligible blocks is in train
    for block in (train, val, test_iid):
        for ex in block:
            for p in ex.key:
                assert p in train_preds, f"predicate {p} missing from train"

    # Held-out keys absent from train
    for ex in test_comp:
        assert ex.key not in train_keys

    return BatterySplit(train=train, val=val, test_iid=test_iid, test_comp=test_comp)


# ---------------------------------------------------------------------------
# Regime L / N
# ---------------------------------------------------------------------------


def _l_pool_preds() -> list[str]:
    """Fixed L/N predicate pool: ns:syn.p0 .. p{L_POOL_SIZE-1}."""
    return [pred_name(i) for i in range(L_POOL_SIZE)]


def _build_l_star(preds: tuple[str, str, str]) -> tuple[list[EdgeTriple], dict[str, str]]:
    """Topology A: all-var 3-star (center subj of all three; leaf vars)."""
    pa, pb, pc = preds
    edges: list[EdgeTriple] = [
        ("?x0", pa, "?x1"),
        ("?x0", pb, "?x2"),
        ("?x0", pc, "?x3"),
    ]
    t0 = type_name(0)
    var_types = {f"?x{i}": t0 for i in range(4)}
    return edges, var_types


def _build_l_chain(preds: tuple[str, str, str]) -> tuple[list[EdgeTriple], dict[str, str]]:
    """Topology B: all-var 3-chain."""
    pa, pb, pc = preds
    edges: list[EdgeTriple] = [
        ("?x0", pa, "?x1"),
        ("?x1", pb, "?x2"),
        ("?x2", pc, "?x3"),
    ]
    t0 = type_name(0)
    var_types = {f"?x{i}": t0 for i in range(4)}
    return edges, var_types


def _l_example(
    preds: tuple[str, str, str],
    topology: str,
    *,
    with_disambig: bool,
    rng: Random,
) -> BatteryExample:
    if topology == "star":
        edges, var_types = _build_l_star(preds)
        topo_label = "star"
    elif topology == "chain":
        edges, var_types = _build_l_chain(preds)
        topo_label = "chain"
    else:
        raise ValueError(f"L/N topology must be star|chain, got {topology}")
    edge_lines = [f"{s} {p} {o} ." for s, p, o in edges]
    type_lines = [f"{v} a {t} ." for v, t in sorted(var_types.items())]
    sparql = _format_sparql(edge_lines, type_lines)
    _assert_has_join(sparql)
    key = tuple(sorted(preds))
    tokens = [word_for_pred(p) for p in preds]
    if with_disambig:
        tokens.append("topoA" if topology == "star" else "topoB")
    question = _shuffle_tokens(rng, tokens)
    return BatteryExample(
        question=question, sparql=sparql, topology=topo_label, key=key,
    )


def _all_l_keys() -> list[tuple[str, ...]]:
    """All size-3 distinct-predicate multisets from the L pool (as sorted keys)."""
    pool = _l_pool_preds()
    return [tuple(sorted(c)) for c in combinations(pool, 3)]


def generate_regime_l(seed: int = 0) -> BatterySplit:
    """Regime L: lexically disambiguated star vs chain (types uninformative)."""
    all_keys = _all_l_keys()
    train_pool, holdout_pool = _partition_keys(all_keys, Random(seed ^ 0xB22))
    train_set = set(train_pool)

    def sample_block(
        n: int, pool: list[tuple[str, ...]], block_seed: int,
    ) -> list[BatteryExample]:
        brng = Random(block_seed)
        out: list[BatteryExample] = []
        for _ in range(n):
            key = pool[brng.randrange(len(pool))]
            # Ordered preds for star/chain construction: use sorted key as (pa,pb,pc)
            # so the same multiset maps to a fixed ordered triple for emission.
            ordered = key  # already sorted
            topo = "star" if brng.random() < 0.5 else "chain"
            out.append(
                _l_example(
                    ordered,  # type: ignore[arg-type]
                    topo,
                    with_disambig=True,
                    rng=Random(brng.randrange(1 << 30)),
                )
            )
        return out

    # Ensure both topologies appear for every train-pool key somewhere in the
    # full corpus: seed the train split with both topos for a covering subset,
    # then fill the rest randomly.
    train: list[BatteryExample] = []
    cover_rng = Random(seed ^ 0xC0)
    for key in train_pool:
        for topo in ("star", "chain"):
            train.append(
                _l_example(
                    key,  # type: ignore[arg-type]
                    topo,
                    with_disambig=True,
                    rng=Random(cover_rng.randrange(1 << 30)),
                )
            )
    # Fill remaining train slots
    remaining = TRAIN_S - len(train)
    if remaining > 0:
        train.extend(sample_block(remaining, train_pool, seed ^ 0x100))
    elif remaining < 0:
        # Cover already exceeded target (unlikely with 20 keys * 2 = 40 < 8000)
        train = train[:TRAIN_S]

    val = sample_block(VAL_S, train_pool, seed ^ 0x200)
    test_iid = sample_block(TEST_IID_S, train_pool, seed ^ 0x300)
    test_comp = sample_block(TEST_COMP_S, holdout_pool, seed ^ 0x400)

    # Also ensure holdout keys get both topologies somewhere in test_comp
    # (not required by spec, but keeps L "both topologies per multiset" spirit
    # on the full corpus for keys we care about in train).
    # Spec: "For EACH sampled predicate multiset, BOTH topologies are planted
    # somewhere across the dataset" — cover all keys that appear.
    keys_seen: set[tuple[str, ...]] = set()
    for block in (train, val, test_iid, test_comp):
        for ex in block:
            keys_seen.add(ex.key)
    # For each seen key, ensure both topos exist somewhere
    by_key_topo: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for block in (train, val, test_iid, test_comp):
        for ex in block:
            by_key_topo[ex.key].add(ex.topology)
    extras: list[BatteryExample] = []
    erng = Random(seed ^ 0xE0)
    for k, topos in by_key_topo.items():
        for need in ("star", "chain"):
            if need not in topos:
                extras.append(
                    _l_example(
                        k,  # type: ignore[arg-type]
                        need,
                        with_disambig=True,
                        rng=Random(erng.randrange(1 << 30)),
                    )
                )
    # Append extras to train if their key is train-eligible, else test_comp
    for ex in extras:
        if ex.key in train_set:
            train.append(ex)
        else:
            test_comp.append(ex)

    train_keys = {ex.key for ex in train}
    for ex in test_comp:
        assert ex.key not in train_keys

    train_preds = {p for ex in train for p in ex.key}
    for p in _l_pool_preds():
        # Every pool pred must appear in train if it appears anywhere
        appears = any(p in ex.key for block in (train, val, test_iid, test_comp) for ex in block)
        if appears:
            assert p in train_preds, f"L pred {p} missing from train"

    return BatterySplit(train=train, val=val, test_iid=test_iid, test_comp=test_comp)


def generate_regime_n(seed: int = 0) -> BatterySplit:
    """Regime N: underdetermined control (no topo disambiguator; coin-flip topo)."""
    rng = Random(seed)
    all_keys = _all_l_keys()
    # No compositional holdout for N — all keys train-eligible.
    # Still ensure every pred appears in train via cover.
    train_pool = list(all_keys)
    rng.shuffle(train_pool)

    def sample_block(n: int, block_seed: int) -> list[BatteryExample]:
        brng = Random(block_seed)
        out: list[BatteryExample] = []
        for _ in range(n):
            key = train_pool[brng.randrange(len(train_pool))]
            topo = "star" if brng.random() < 0.5 else "chain"
            out.append(
                _l_example(
                    key,  # type: ignore[arg-type]
                    topo,
                    with_disambig=False,
                    rng=Random(brng.randrange(1 << 30)),
                )
            )
        return out

    train = sample_block(TRAIN_N, seed ^ 0x100)
    val = sample_block(VAL_N, seed ^ 0x200)
    test_iid = sample_block(TEST_IID_N, seed ^ 0x300)
    test_comp: list[BatteryExample] = []

    # Dataset-level assertion: at least one key occurs with BOTH topologies in train.
    by_key: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for ex in train:
        by_key[ex.key].add(ex.topology)
        assert "topoA" not in ex.question.split()
        assert "topoB" not in ex.question.split()
    dual = [k for k, tops in by_key.items() if tops >= {"star", "chain"}]
    if not dual:
        # Force-inject both topos for the first train key
        k0 = train[0].key
        train.append(
            _l_example(
                k0,  # type: ignore[arg-type]
                "star",
                with_disambig=False,
                rng=Random(seed ^ 0xD1),
            )
        )
        train.append(
            _l_example(
                k0,  # type: ignore[arg-type]
                "chain",
                with_disambig=False,
                rng=Random(seed ^ 0xD2),
            )
        )
        by_key = defaultdict(set)
        for ex in train:
            by_key[ex.key].add(ex.topology)
        dual = [k for k, tops in by_key.items() if tops >= {"star", "chain"}]
    assert dual, "regime N train must have at least one key with both topologies"

    # Cover all pool preds in train
    train_preds = {p for ex in train for p in ex.key}
    for p in _l_pool_preds():
        if p not in train_preds:
            # inject a triple containing p
            others = [q for q in _l_pool_preds() if q != p][:2]
            key = tuple(sorted([p, others[0], others[1]]))
            pseed = (seed * 1000003 + sum(map(ord, p))) & 0xFFFFFFFF
            train.append(
                _l_example(
                    key,  # type: ignore[arg-type]
                    "star",
                    with_disambig=False,
                    rng=Random(pseed),
                )
            )
            train_preds.add(p)

    return BatterySplit(train=train, val=val, test_iid=test_iid, test_comp=test_comp)


# ---------------------------------------------------------------------------
# Expressibility invariant
# ---------------------------------------------------------------------------


def world_counts(split: BatterySplit) -> dict[str, int]:
    """n_distinct_sigs / n_distinct_keys observed in split.train."""
    distinct_sigs: set[tuple] = set()
    for ex in split.train:
        for sig in parse_sparql_joins(ex.sparql).signatures:
            distinct_sigs.add(sig)
    return {
        "n_distinct_sigs": len(distinct_sigs),
        "n_distinct_keys": len({ex.key for ex in split.train}),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_regime(regime: str, seed: int = 0) -> BatterySplit:
    """Generate a battery split for regime in {\"S\", \"L\", \"N\"}.

    After the train split is built, enforces the expressibility invariant:
    distinct planted per-var signatures and distinct predicate-multiset keys
    in train must fit the admission budget (max_rules / candidate cap).
    """
    if regime == "S":
        split = generate_regime_s(seed)
    elif regime == "L":
        split = generate_regime_l(seed)
    elif regime == "N":
        split = generate_regime_n(seed)
    else:
        raise ValueError(f"unknown regime {regime!r}; expected 'S', 'L', or 'N'")

    counts = world_counts(split)
    n_sigs = counts["n_distinct_sigs"]
    n_keys = counts["n_distinct_keys"]
    print(
        f"  world_counts regime={regime} seed={seed} "
        f"n_distinct_sigs={n_sigs} n_distinct_keys={n_keys}",
        flush=True,
    )
    assert n_sigs <= 100, (
        f"expressibility violated: n_distinct_sigs={n_sigs} > 100 "
        f"(regime={regime}, seed={seed})"
    )
    assert n_keys <= 250, (
        f"expressibility violated: n_distinct_keys={n_keys} > 250 "
        f"(regime={regime}, seed={seed})"
    )
    return split
