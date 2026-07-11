"""Config-holdout qa1 world generator (slice #80).

Builds a fully synthetic bAbI-qa1-shaped corpus with explicitly held-out
entity×location combinations so residual can reappear for compositional-binding
measurement. stdlib + ``random.Random`` only — no torch.

Story cadence matches ``data/corpus_babi.txt`` / ``campaign_qa1_cond_headroom``
loaders: 2 movements then 1 query, ×5 turns = 10 movements + 5 Q/A spans per story.
Train corpus text is a single-line space-joined concatenation of 2000 stories
(10_000 Q/A spans), matching the fixed-shape contract of ``load_train_stories``.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Any

# ---------------------------------------------------------------------------
# Fixed vocabulary (original bAbI qa1 world — do not invent new atoms)
# ---------------------------------------------------------------------------
ENTITIES: tuple[str, ...] = ("Daniel", "John", "Mary", "Sandra")
LOCATIONS: tuple[str, ...] = (
    "bathroom",
    "bedroom",
    "garden",
    "hallway",
    "kitchen",
    "office",
)
VERBS: tuple[str, ...] = (
    "journeyed",
    "moved",
    "travelled",
    "went",
    "went back",
)
NOVEL_ENTITIES: tuple[str, ...] = ("Albert", "Beatrice", "Colin", "Doreen")

HELD_OUT_PAIRS: tuple[tuple[str, str], ...] = (
    ("Mary", "office"),
    ("John", "garden"),
    ("Daniel", "bathroom"),
    ("Sandra", "kitchen"),
)
HELD_OUT_SET: frozenset[tuple[str, str]] = frozenset(HELD_OUT_PAIRS)

# 4 entities × 6 locations − 4 held-out = 20 train-legal pairs
TRAIN_LEGAL_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (e, loc)
    for e in ENTITIES
    for loc in LOCATIONS
    if (e, loc) not in HELD_OUT_SET
)

N_TRAIN = 2000
N_TEST = 200
TURNS = 5
MOVES_PER_TURN = 2
MIN_PAIR_COUNT = 20
MAX_STORY_ATTEMPTS = 200

MOVE_RE = re.compile(r"^(?P<ent>\w+) (?P<verb>.+?) to the (?P<loc>\w+)\.$")
QA_RE = re.compile(r"Q: (?P<q>.*?) A: (?P<a>\w+)\.")
WHERE_RE = re.compile(r"Where is (?P<ent>\w+)\?")
SENT_SPLIT = re.compile(r"(?<=\.)\s+")


@dataclass(frozen=True)
class GenStory:
    """One generated story: corpus-format single-line text + stable id."""

    text: str  # space-joined movements + Q/A spans (corpus shape)
    story_id: str


# ---------------------------------------------------------------------------
# Low-level sentence builders
# ---------------------------------------------------------------------------
def move_sentence(entity: str, verb: str, loc: str) -> str:
    return f"{entity} {verb} to the {loc}."


def qa_span(entity: str, answer: str) -> str:
    return f"Q: Where is {entity}? A: {answer}."


def parse_story_text(text: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Parse a GenStory.text into (movements, queries).

    movements: list of (entity, verb, loc) in chronological order
    queries: list of (entity, answer) in turn order
    """
    movements: list[tuple[str, str, str]] = []
    queries: list[tuple[str, str]] = []
    # Split into movement vs Q/A by finding Q spans
    pos = 0
    for m in QA_RE.finditer(text):
        chunk = text[pos : m.start()].strip()
        if chunk:
            for sent in SENT_SPLIT.split(chunk):
                sent = sent.strip()
                if not sent:
                    continue
                mm = MOVE_RE.match(sent)
                if mm:
                    movements.append((mm.group("ent"), mm.group("verb"), mm.group("loc")))
        q = m.group("q")
        ent_m = WHERE_RE.search(q)
        ent = ent_m.group("ent") if ent_m else q.replace("?", "").split()[-1]
        queries.append((ent, m.group("a")))
        pos = m.end()
    return movements, queries


def live_location(
    entity: str, movements: list[tuple[str, str, str]]
) -> str | None:
    """Most-recent location of entity after the given movement list."""
    for e, _v, loc in reversed(movements):
        if e == entity:
            return loc
    return None


def assert_entity_moved_before_queries(
    movements: list[tuple[str, str, str]],
    queries: list[tuple[str, str]],
    *,
    story_id: str,
) -> None:
    """Raise if any queried entity has no movement before its query turn.

    Cadence: 2 movements per turn before each query; after turn t the movement
    prefix length is 2*(t+1).
    """
    for t, (ent, _ans) in enumerate(queries):
        prefix = movements[: MOVES_PER_TURN * (t + 1)]
        if live_location(ent, prefix) is None:
            raise AssertionError(
                f"story {story_id!r} turn {t}: entity {ent!r} queried before any movement"
            )


def is_holdout_binding(entity: str, answer: str) -> bool:
    return (entity, answer) in HELD_OUT_SET


# ---------------------------------------------------------------------------
# Story generation
# ---------------------------------------------------------------------------
def _pick_pair(
    rng: Random,
    legal: list[tuple[str, str]],
    pair_counts: Counter[tuple[str, str]] | None,
    prefer_min: int = MIN_PAIR_COUNT,
) -> tuple[str, str]:
    """Sample an (entity, location) pair, biasing toward under-covered pairs."""
    if pair_counts is not None and legal:
        under = [p for p in legal if pair_counts[p] < prefer_min]
        pool = under if under else legal
    else:
        pool = legal
    return pool[rng.randrange(len(pool))]


def _pick_verb(rng: Random) -> str:
    return VERBS[rng.randrange(len(VERBS))]


def _append_move(
    parts: list[str],
    movements: list[tuple[str, str, str]],
    entity: str,
    verb: str,
    loc: str,
    pair_counts: Counter[tuple[str, str]] | None,
) -> None:
    movements.append((entity, verb, loc))
    parts.append(move_sentence(entity, verb, loc))
    if pair_counts is not None:
        pair_counts[(entity, loc)] += 1


def _generate_story_once(
    rng: Random,
    story_id: str,
    *,
    legal_pairs: list[tuple[str, str]],
    pair_counts: Counter[tuple[str, str]] | None,
    force_holdout_turns: set[int] | None,
    entities: tuple[str, ...] = ENTITIES,
    allow_held_out_moves: bool = False,
    novel_map: dict[str, str] | None = None,
) -> GenStory:
    """Generate one story; may raise AssertionError if invariants fail.

    Cadence: for each of 5 turns, emit exactly 2 movement sentences then one
    ``Q: Where is <E>? A: <loc>.`` span. Gold = live location of E after those
    movements (and all prior). The last turn still gets exactly 2 movements
    before the final query (10 movements total).
    """
    force_holdout_turns = force_holdout_turns or set()
    # All pairs for unrestricted moves (holdout free turns)
    all_pairs = [(e, loc) for e in entities for loc in LOCATIONS]
    # held-out partner location per display entity (original names only)
    held_for_entity: dict[str, str] = {e: loc for e, loc in HELD_OUT_PAIRS}

    movements: list[tuple[str, str, str]] = []
    queries: list[tuple[str, str]] = []
    parts: list[str] = []
    state: dict[str, str] = {}  # entity -> last loc

    for turn in range(TURNS):
        if turn in force_holdout_turns and not novel_map:
            # Force a holdout-binding query: move target entity to its held-out loc
            # then a distractor; query the target.
            candidates = [e for e in ENTITIES if e in held_for_entity]
            # Prefer entities not yet at holdout, else any
            rng.shuffle(candidates)
            target = candidates[turn % len(candidates)]
            hloc = held_for_entity[target]
            # Move 1: target -> held-out location
            v1 = _pick_verb(rng)
            _append_move(parts, movements, target, v1, hloc, None)
            state[target] = hloc
            # Move 2: distractor (other entity, any location including holdout)
            others = [e for e in ENTITIES if e != target]
            d_ent = others[rng.randrange(len(others))]
            if allow_held_out_moves:
                d_pairs = [(e, loc) for e, loc in all_pairs if e == d_ent]
            else:
                d_pairs = [(e, loc) for e, loc in legal_pairs if e == d_ent]
            if not d_pairs:
                d_pairs = [(d_ent, LOCATIONS[rng.randrange(len(LOCATIONS))])]
            _e, d_loc = d_pairs[rng.randrange(len(d_pairs))]
            v2 = _pick_verb(rng)
            _append_move(parts, movements, d_ent, v2, d_loc, None)
            state[d_ent] = d_loc
            q_ent = target
            q_ans = state[target]
        else:
            # Two unrestricted (legal) movements
            for _ in range(MOVES_PER_TURN):
                if allow_held_out_moves and novel_map is None:
                    pool = all_pairs
                else:
                    pool = legal_pairs
                ent, loc = _pick_pair(rng, pool, pair_counts)
                verb = _pick_verb(rng)
                _append_move(
                    parts,
                    movements,
                    ent,
                    verb,
                    loc,
                    pair_counts if not allow_held_out_moves else None,
                )
                state[ent] = loc
            # Query an entity that has moved
            moved = [e for e in entities if e in state]
            if not moved:
                raise AssertionError(f"{story_id}: no entity has moved before query")
            q_ent = moved[rng.randrange(len(moved))]
            q_ans = state[q_ent]

        queries.append((q_ent, q_ans))
        parts.append(qa_span(q_ent, q_ans))

    assert_entity_moved_before_queries(movements, queries, story_id=story_id)

    if force_holdout_turns and not novel_map:
        n_hold = sum(1 for e, a in queries if is_holdout_binding(e, a))
        if n_hold < 2:
            raise AssertionError(
                f"{story_id}: need >=2 holdout-answer queries, got {n_hold}"
            )

    text = " ".join(parts)
    return GenStory(text=text, story_id=story_id)


def generate_story(
    rng: Random,
    story_id: str,
    *,
    legal_pairs: list[tuple[str, str]],
    pair_counts: Counter[tuple[str, str]] | None = None,
    force_holdout_turns: set[int] | None = None,
    entities: tuple[str, ...] = ENTITIES,
    allow_held_out_moves: bool = False,
    novel_map: dict[str, str] | None = None,
) -> GenStory:
    """Generate one story with retry cap; raises AssertionError if all attempts fail."""
    last_err: Exception | None = None
    for _ in range(MAX_STORY_ATTEMPTS):
        # snapshot counts so failed attempts do not pollute
        snap = Counter(pair_counts) if pair_counts is not None else None
        try:
            story = _generate_story_once(
                rng,
                story_id,
                legal_pairs=legal_pairs,
                pair_counts=snap,
                force_holdout_turns=force_holdout_turns,
                entities=entities,
                allow_held_out_moves=allow_held_out_moves,
                novel_map=novel_map,
            )
            if pair_counts is not None and snap is not None:
                pair_counts.clear()
                pair_counts.update(snap)
            return story
        except AssertionError as e:
            last_err = e
            continue
    raise AssertionError(
        f"failed to generate {story_id} after {MAX_STORY_ATTEMPTS} attempts: {last_err}"
    )


def _assert_train_excludes_held_out(train: list[GenStory]) -> None:
    """(a) No train movement uses a held-out (entity, location) pair."""
    for s in train:
        movs, _qs = parse_story_text(s.text)
        for ent, _verb, loc in movs:
            if (ent, loc) in HELD_OUT_SET:
                raise AssertionError(
                    f"train story {s.story_id} contains held-out movement "
                    f"({ent!r}, {loc!r})"
                )


def _assert_train_pair_coverage(train: list[GenStory], min_count: int = MIN_PAIR_COUNT) -> Counter:
    """(b) Every non-held-out pair appears >= min_count times in train movements."""
    counts: Counter[tuple[str, str]] = Counter()
    for s in train:
        movs, _ = parse_story_text(s.text)
        for ent, _v, loc in movs:
            counts[(ent, loc)] += 1
    missing = []
    for pair in TRAIN_LEGAL_PAIRS:
        c = counts[pair]
        if c < min_count:
            missing.append((pair, c))
    if missing:
        raise AssertionError(
            f"train pair coverage < {min_count} for {len(missing)} pairs; "
            f"examples: {missing[:5]}"
        )
    # also ensure no held-out pair sneaks in with positive count
    for pair in HELD_OUT_PAIRS:
        if counts[pair] > 0:
            raise AssertionError(f"held-out pair {pair} has train count {counts[pair]}")
    return counts


def _assert_all_stories_entity_before_query(stories: list[GenStory]) -> None:
    """(c) Every entity has moved at least once before its first query in every story."""
    for s in stories:
        movs, qs = parse_story_text(s.text)
        assert_entity_moved_before_queries(movs, qs, story_id=s.story_id)


def join_train_corpus(train: list[GenStory]) -> str:
    """Flatten train stories the same way ``corpus_babi.txt`` is shaped (one line)."""
    return " ".join(s.text for s in train)


def story_prompts_bench_style(story: GenStory) -> list[tuple[str, str]]:
    """Build per-turn (prompt, answer) pairs in ``babi_bench.json`` accumulated shape.

    Each prompt is the full story prefix through that turn's ``A:`` (no gold word).
    """
    text = story.text
    spans = list(QA_RE.finditer(text))
    if len(spans) != TURNS:
        raise AssertionError(
            f"{story.story_id}: expected {TURNS} Q/A spans, got {len(spans)}"
        )
    out: list[tuple[str, str]] = []
    for m in spans:
        span_text = text[m.start() : m.end()]
        a_idx = span_text.rfind("A:")
        if a_idx < 0:
            raise RuntimeError(f"no A: in {span_text!r}")
        # prompt = text from story start through this "A:"
        prompt = text[: m.start() + a_idx + 2].strip()
        if not prompt.endswith("A:"):
            raise AssertionError(f"prompt must end with A:, got ...{prompt[-40:]!r}")
        out.append((prompt, m.group("a")))
    return out


def generate_world(seed: int = 0) -> dict[str, Any]:
    """Return train + three test blocks for the config-holdout instrument.

    Returns
    -------
    dict with keys:
      train : list[GenStory] (2000)
      test_holdout : list[GenStory] (200)  — block-1 residual instrument
      test_iid : list[GenStory] (200)      — sanity / pinned val distribution
      test_names : list[GenStory] (200)    — diagnostic only (novel entity names)
      pair_counts : Counter of train movement (entity, loc) pairs
    """
    rng = Random(seed)
    legal = list(TRAIN_LEGAL_PAIRS)
    pair_counts: Counter[tuple[str, str]] = Counter()

    # --- train: no held-out pairs; cover all 20 legal pairs ---
    train: list[GenStory] = []
    for i in range(N_TRAIN):
        s = generate_story(
            rng,
            f"train:{i}",
            legal_pairs=legal,
            pair_counts=pair_counts,
            force_holdout_turns=None,
            allow_held_out_moves=False,
        )
        train.append(s)

    # If coverage still short (unlikely with bias), top-up by rewriting tail stories
    for _boost in range(50):
        try:
            _assert_train_pair_coverage(train)
            break
        except AssertionError:
            # regenerate a random story with strong under-coverage bias
            idx = rng.randrange(N_TRAIN)
            # zero out old story's contribution
            old_movs, _ = parse_story_text(train[idx].text)
            for ent, _v, loc in old_movs:
                pair_counts[(ent, loc)] -= 1
                if pair_counts[(ent, loc)] <= 0:
                    del pair_counts[(ent, loc)]
            train[idx] = generate_story(
                rng,
                f"train:{idx}",
                legal_pairs=legal,
                pair_counts=pair_counts,
                force_holdout_turns=None,
                allow_held_out_moves=False,
            )
    else:
        _assert_train_pair_coverage(train)  # raise with detail

    _assert_train_excludes_held_out(train)
    pair_counts = _assert_train_pair_coverage(train)
    _assert_all_stories_entity_before_query(train)

    # --- test_holdout: >=2 holdout-answer queries per story ---
    test_holdout: list[GenStory] = []
    for i in range(N_TEST):
        # at least two turns forced to holdout answers
        turns = set(rng.sample(range(TURNS), k=2))
        # occasionally add a third
        if rng.random() < 0.3:
            extra = [t for t in range(TURNS) if t not in turns]
            turns.add(extra[rng.randrange(len(extra))])
        s = generate_story(
            rng,
            f"holdout:{i}",
            legal_pairs=list(
                (e, loc) for e in ENTITIES for loc in LOCATIONS
            ),  # unrestricted
            pair_counts=None,
            force_holdout_turns=turns,
            allow_held_out_moves=True,
        )
        test_holdout.append(s)
    _assert_all_stories_entity_before_query(test_holdout)
    for s in test_holdout:
        _movs, qs = parse_story_text(s.text)
        n_h = sum(1 for e, a in qs if is_holdout_binding(e, a))
        if n_h < 2:
            raise AssertionError(f"{s.story_id} has only {n_h} holdout queries")

    # --- test_iid: no held-out bindings anywhere ---
    test_iid: list[GenStory] = []
    for i in range(N_TEST):
        s = generate_story(
            rng,
            f"iid:{i}",
            legal_pairs=legal,
            pair_counts=None,
            force_holdout_turns=None,
            allow_held_out_moves=False,
        )
        # verify no held-out movement or answer
        movs, qs = parse_story_text(s.text)
        for ent, _v, loc in movs:
            if (ent, loc) in HELD_OUT_SET:
                raise AssertionError(f"iid story {s.story_id} has held-out move")
        for ent, ans in qs:
            if is_holdout_binding(ent, ans):
                raise AssertionError(f"iid story {s.story_id} has holdout answer")
        test_iid.append(s)
    _assert_all_stories_entity_before_query(test_iid)

    # --- test_names: novel entities, same legal-location structure as iid ---
    # Map base entities → novel names; held-out pairs do not apply (diagnostic)
    novel_map = {base: nov for base, nov in zip(ENTITIES, NOVEL_ENTITIES, strict=True)}
    novel_ents = NOVEL_ENTITIES
    # legal pairs in novel-name space: all entity×location (no holdout in novel world)
    novel_legal = [(e, loc) for e in novel_ents for loc in LOCATIONS]
    test_names: list[GenStory] = []
    for i in range(N_TEST):
        s = generate_story(
            rng,
            f"names:{i}",
            legal_pairs=novel_legal,
            pair_counts=None,
            force_holdout_turns=None,
            entities=novel_ents,
            allow_held_out_moves=False,
            novel_map=novel_map,
        )
        movs, qs = parse_story_text(s.text)
        for ent, _v, _loc in movs:
            if ent in ENTITIES:
                raise AssertionError(
                    f"names story {s.story_id} uses original entity {ent!r}"
                )
        for ent, _a in qs:
            if ent in ENTITIES:
                raise AssertionError(
                    f"names story {s.story_id} queries original entity {ent!r}"
                )
        test_names.append(s)
    _assert_all_stories_entity_before_query(test_names)

    # global id uniqueness
    all_ids = (
        [s.story_id for s in train]
        + [s.story_id for s in test_holdout]
        + [s.story_id for s in test_iid]
        + [s.story_id for s in test_names]
    )
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("story id collision across splits")

    return {
        "train": train,
        "test_holdout": test_holdout,
        "test_iid": test_iid,
        "test_names": test_names,
        "pair_counts": pair_counts,
    }


def inject_held_out_into_train_text(train_text: str, pair: tuple[str, str] | None = None) -> str:
    """Test helper: splice one held-out movement into train corpus text.

    Used by unit tests to prove ``_assert_train_excludes_held_out`` is not a no-op.
    """
    ent, loc = pair or HELD_OUT_PAIRS[0]
    bad = move_sentence(ent, "moved", loc)
    # insert after first period
    i = train_text.find(".")
    if i < 0:
        return bad + " " + train_text
    return train_text[: i + 1] + " " + bad + train_text[i + 1 :]


def assert_train_corpus_invariants_from_text(text: str) -> None:
    """Parse a flat train corpus and run holdout-exclusion + pair-coverage asserts.

    Splits into pseudo-stories of 5 Q/A each (same contract as load_train_stories).
    """
    spans = list(QA_RE.finditer(text))
    if len(spans) % TURNS != 0:
        raise AssertionError(f"Q/A count {len(spans)} not divisible by {TURNS}")
    n_stories = len(spans) // TURNS
    stories: list[GenStory] = []
    prev = 0
    for s_i in range(n_stories):
        end = spans[s_i * TURNS + (TURNS - 1)].end()
        chunk = text[prev:end].strip()
        stories.append(GenStory(text=chunk, story_id=f"train:{s_i}"))
        prev = end
    _assert_train_excludes_held_out(stories)
    _assert_train_pair_coverage(stories)
