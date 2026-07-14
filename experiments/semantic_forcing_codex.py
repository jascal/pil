"""Probe A semantic-forcing pilot: QA1 control plus English agreement.

This is measurement instrumentation only.  Hugging Face work is delegated to the
existing lm-sae virtualenv so the pil environment remains untouched.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pil.qa1_battery import (  # noqa: E402
    ENTITIES,
    LOCATIONS,
    VERBS,
    generate_world,
    live_location,
    move_sentence,
    parse_story_text,
)

SEED = 0
LM_PYTHON = Path("/home/allans/code/lm-sae/.venv/bin/python")
OUTPUT_JSON = REPO / "data" / "semantic_forcing_codex.json"
TEACHER_MODEL = "gpt2"
TEACHER_LAYER = 6
BATCH_SIZE = 32

BAR1_MIN_EXCLUSIVE = 0.95
BAR2_MIN = 0.90
IMITATOR_UNPERTURBED_MIN = 0.80
IMITATOR_PROPAGATION_MAX = 0.50
CONSTRAINT_PROPAGATION_MIN = 0.95

WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
MIN_WIKITEXT_FORM_FREQUENCY = 20
PROBE_TRAIN_LEMMAS = 20
EVAL_LEMMAS = 16
FRAMES = ("to", "of", "near", "behind")
BIGRAM_FIT_FRAMES = ("to", "of")
EVAL_FRAMES = ("near", "behind")
DISTANCE_K = 4
NUMBER_NAMES = ("singular", "plural")
VERB_PAIRS = (("is", "are"), ("was", "were"), ("has", "have"), ("does", "do"))

# Hand-vetted common countable nouns.  Corpus frequency, not this ordering, determines admission.
SEED_NOUNS: tuple[tuple[str, str], ...] = (
    ("artist", "artists"),
    ("bird", "birds"),
    ("book", "books"),
    ("boy", "boys"),
    ("cabinet", "cabinets"),
    ("car", "cars"),
    ("cat", "cats"),
    ("chair", "chairs"),
    ("child", "children"),
    ("church", "churches"),
    ("city", "cities"),
    ("company", "companies"),
    ("country", "countries"),
    ("doctor", "doctors"),
    ("dog", "dogs"),
    ("door", "doors"),
    ("family", "families"),
    ("film", "films"),
    ("friend", "friends"),
    ("game", "games"),
    ("girl", "girls"),
    ("group", "groups"),
    ("horse", "horses"),
    ("house", "houses"),
    ("key", "keys"),
    ("king", "kings"),
    ("letter", "letters"),
    ("machine", "machines"),
    ("man", "men"),
    ("member", "members"),
    ("officer", "officers"),
    ("picture", "pictures"),
    ("player", "players"),
    ("queen", "queens"),
    ("river", "rivers"),
    ("road", "roads"),
    ("room", "rooms"),
    ("school", "schools"),
    ("ship", "ships"),
    ("soldier", "soldiers"),
    ("song", "songs"),
    ("station", "stations"),
    ("story", "stories"),
    ("student", "students"),
    ("system", "systems"),
    ("table", "tables"),
    ("teacher", "teachers"),
    ("team", "teams"),
    ("tree", "trees"),
    ("window", "windows"),
    ("woman", "women"),
    ("worker", "workers"),
)


@dataclass(frozen=True)
class ImitatorScore:
    name: str
    unperturbed: float
    propagation: float


@dataclass(frozen=True)
class Verdict:
    verdict: str
    failure: str | None


@dataclass(frozen=True)
class AgreementItem:
    item_id: str
    prefix: str
    subject: str
    attractor: str
    subject_number: int
    attractor_number: int
    correct_verb: str
    wrong_verb: str
    prep: str


T = TypeVar("T")


def fixed_verdict(
    *,
    bar1: float,
    bar2: float,
    constraint_propagation: float,
    imitators: Sequence[ImitatorScore],
) -> Verdict:
    """Apply the signed preregistration's fixed bars without tuning."""
    if bar1 <= BAR1_MIN_EXCLUSIVE:
        return Verdict("DEAD", "soft-not-forcing")
    if bar2 < BAR2_MIN:
        return Verdict("DEAD", "not-derivable")
    strong = [score for score in imitators if score.unperturbed >= IMITATOR_UNPERTURBED_MIN]
    if not strong:
        return Verdict("DEAD", "easy-win")
    if len(strong) != len(imitators):
        return Verdict("DEAD", "vacuous-discriminator")
    stronger_propagation = max(score.propagation for score in imitators)
    if (
        stronger_propagation > IMITATOR_PROPAGATION_MAX
        or constraint_propagation < CONSTRAINT_PROPAGATION_MIN
    ):
        return Verdict("DEAD", "vacuous-discriminator")
    return Verdict("FIRES", None)


def drop_mechanism_noops(
    items: Sequence[T],
    *,
    old_gold: Callable[[T], Any],
    new_gold: Callable[[T], Any],
) -> tuple[list[T], int]:
    """Drop perturbations that do not change the forced gold value."""
    kept = [item for item in items if old_gold(item) != new_gold(item)]
    return kept, len(items) - len(kept)


def agreement_aggregate_python(subject_numbers: Sequence[int]) -> list[int]:
    """Certified integer aggregate: verb number is the subject-number concept."""
    values = torch.as_tensor(list(subject_numbers), dtype=torch.int64)
    if values.numel() and not bool(torch.all((values == 0) | (values == 1))):
        raise ValueError("number labels must be 0 (singular) or 1 (plural)")
    return values.clone().tolist()


SOUFFLE_AGREEMENT = """\
.decl subject_number(item:number, number:number)
.input subject_number
.decl verb_number(item:number, number:number)
.output verb_number
verb_number(item, number) :- subject_number(item, number).
"""


def souffle_agreement(subject_numbers: Sequence[int]) -> list[int]:
    """Independently mirror the identity aggregate in a tiny Soufflé program."""
    labels = [int(value) for value in subject_numbers]
    if any(value not in (0, 1) for value in labels):
        raise ValueError("number labels must be 0 (singular) or 1 (plural)")
    with tempfile.TemporaryDirectory(prefix="semantic-forcing-") as raw_dir:
        directory = Path(raw_dir)
        (directory / "agreement.dl").write_text(SOUFFLE_AGREEMENT)
        (directory / "subject_number.facts").write_text(
            "".join(f"{index}\t{value}\n" for index, value in enumerate(labels))
        )
        subprocess.run(
            [
                "souffle",
                "-F",
                str(directory),
                "-D",
                str(directory),
                str(directory / "agreement.dl"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        derived: dict[int, int] = {}
        output = directory / "verb_number.csv"
        if output.exists():
            for line in output.read_text().splitlines():
                item, number = line.split("\t")
                derived[int(item)] = int(number)
    if len(derived) != len(labels):
        raise RuntimeError(f"Soufflé derived {len(derived)} of {len(labels)} rows")
    return [derived[index] for index in range(len(labels))]


def _concept_recovered_live_location(
    entity: str, movements: Sequence[tuple[str, str, str]]
) -> str | None:
    """Recover QA1 state through grounded entity/location integer concepts."""
    entity_to_id = {name: index for index, name in enumerate(ENTITIES)}
    location_to_id = {name: index for index, name in enumerate(LOCATIONS)}
    if entity not in entity_to_id:
        return None
    state: dict[int, int] = {}
    for moved_entity, _verb, location in movements:
        if moved_entity in entity_to_id and location in location_to_id:
            state[entity_to_id[moved_entity]] = location_to_id[location]
    answer_id = state.get(entity_to_id[entity])
    return None if answer_id is None else LOCATIONS[answer_id]


CROSS_VENV_HELPER = r'''
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

request = json.load(sys.stdin)
mode = request["mode"]
if mode == "mine":
    from datasets import Dataset

    base = Path.home() / ".cache/huggingface/datasets/Salesforce___wikitext/wikitext-103-raw-v1"
    revisions = sorted(base.glob("*"))
    if not revisions:
        raise RuntimeError(f"cached WikiText config not found under {base}")
    shards = sorted(revisions[-1].glob("**/*train*.arrow"))
    if not shards:
        raise RuntimeError(f"cached WikiText train shards not found under {revisions[-1]}")
    forms = sorted(set(request["forms"]), key=lambda word: (-len(word), word))
    pattern = re.compile(r"(?<![A-Za-z])(" + "|".join(map(re.escape, forms)) + r")(?![A-Za-z])", re.I)
    counts = Counter()
    rows = 0
    for shard in shards:
        dataset = Dataset.from_file(str(shard))
        rows += len(dataset)
        for text in dataset["text"]:
            counts.update(match.lower() for match in pattern.findall(text))
    json.dump({"counts": counts, "rows": rows, "shards": [str(path) for path in shards]}, sys.stdout)
elif mode == "teacher":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = request["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True).eval()
    tasks = request["tasks"]
    results = []
    batch_size = int(request.get("batch_size", 32))
    layer = int(request.get("layer", 6))
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        texts = [task["prefix"] for task in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True)
            log_probs = output.logits.log_softmax(dim=-1)
        final_positions = encoded["attention_mask"].sum(dim=1) - 1
        hidden_layer = output.hidden_states[layer]
        for row, task in enumerate(batch):
            answer = {}
            for key in ("candidate_a", "candidate_b"):
                candidate_ids = tokenizer.encode(" " + task[key], add_special_tokens=False)
                if len(candidate_ids) != 1:
                    raise RuntimeError(f"candidate is not one GPT-2 token: {task[key]!r} -> {candidate_ids}")
                answer["logprob_" + key[-1]] = float(
                    log_probs[row, final_positions[row], candidate_ids[0]].item()
                )
            if "subject_start" in task:
                subject_start = int(task["subject_start"])
                subject_end = int(task["subject_end"])
                positions = []
                for token_index, (left, right) in enumerate(offsets[row].tolist()):
                    if right > subject_start and left < subject_end:
                        positions.append(token_index)
                if not positions:
                    raise RuntimeError(f"subject span not tokenized in {task['prefix']!r}")
                subject_position = positions[-1]
                answer["hidden"] = hidden_layer[row, subject_position].float().tolist()
                answer["subject_token_position"] = subject_position
                answer["verb_token_position"] = int(final_positions[row].item()) + 1
                answer["token_distance"] = answer["verb_token_position"] - subject_position
            results.append(answer)
    json.dump({"results": results}, sys.stdout)
else:
    raise RuntimeError(f"unknown mode {mode!r}")
'''


def _cross_venv(request: dict[str, Any], *, timeout: int = 900) -> dict[str, Any]:
    if not LM_PYTHON.exists():
        raise FileNotFoundError(f"required sibling interpreter missing: {LM_PYTHON}")
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    result = subprocess.run(
        [str(LM_PYTHON), "-c", CROSS_VENV_HELPER],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )
    if result.returncode:
        raise RuntimeError(f"cross-venv helper failed:\n{result.stderr}")
    return json.loads(result.stdout)


def _teacher(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    response = _cross_venv(
        {
            "mode": "teacher",
            "model": TEACHER_MODEL,
            "layer": TEACHER_LAYER,
            "batch_size": BATCH_SIZE,
            "tasks": list(tasks),
        }
    )
    return response["results"]


def _fit_qa1_pair_memory(train_stories: Sequence[Any]) -> dict[str, str]:
    counts: Counter[tuple[str, str]] = Counter()
    for story in train_stories:
        movements, queries = parse_story_text(story.text)
        for entity, _verb, location in movements:
            counts[(entity, location)] += 1
        for entity, answer in queries:
            counts[(entity, answer)] += 1
    return {
        entity: max(LOCATIONS, key=lambda location: (counts[(entity, location)], -LOCATIONS.index(location)))
        for entity in ENTITIES
    }


def run_arm1() -> dict[str, Any]:
    """Run the generated QA1 positive control."""
    world = generate_world(seed=SEED)
    pair_memory = _fit_qa1_pair_memory(world["train"])
    target_location = LOCATIONS[0]
    candidates: list[dict[str, Any]] = []
    for story in world["test_iid"]:
        movements, queries = parse_story_text(story.text)
        for turn, (entity, answer) in enumerate(queries):
            prefix_movements = movements[: 2 * (turn + 1)]
            old_gold = live_location(entity, prefix_movements)
            if (
                old_gold == answer == pair_memory[entity]
                and old_gold != target_location
                and len(candidates) < 24
            ):
                perturbed = [*prefix_movements, (entity, VERBS[1], target_location)]
                new_gold = live_location(entity, perturbed)
                concept_gold = _concept_recovered_live_location(entity, perturbed)
                context = " ".join(move_sentence(*movement) for movement in perturbed)
                context += f" {entity} is now in the"
                candidates.append(
                    {
                        "story_id": story.story_id,
                        "turn": turn,
                        "entity": entity,
                        "old_gold": old_gold,
                        "new_gold": new_gold,
                        "concept_gold": concept_gold,
                        "context": context,
                        "imitator": pair_memory[entity],
                    }
                )
        if len(candidates) == 24:
            break
    if len(candidates) < 24:
        raise RuntimeError(f"QA1 positive control found only {len(candidates)} eligible queries")
    engaged, drop_count = drop_mechanism_noops(
        candidates,
        old_gold=lambda row: row["old_gold"],
        new_gold=lambda row: row["new_gold"],
    )
    tasks = [
        {
            "prefix": row["context"],
            "candidate_a": row["new_gold"],
            "candidate_b": row["old_gold"],
        }
        for row in engaged
    ]
    teacher = _teacher(tasks)
    bar1 = sum(result["logprob_a"] > result["logprob_b"] for result in teacher) / len(teacher)
    bar2 = sum(row["concept_gold"] == row["new_gold"] for row in engaged) / len(engaged)
    constraint_propagation = bar2
    unperturbed = sum(row["imitator"] == row["old_gold"] for row in engaged) / len(engaged)
    imitation_propagation = sum(row["imitator"] == row["new_gold"] for row in engaged) / len(engaged)
    imitator = ImitatorScore("joint-entity-location-pair-memory", unperturbed, imitation_propagation)
    verdict = fixed_verdict(
        bar1=bar1,
        bar2=bar2,
        constraint_propagation=constraint_propagation,
        imitators=[imitator],
    )
    return {
        "name": "bAbI moveloc",
        "role": "positive control",
        "tag": "positive-control",
        **asdict(verdict),
        "metrics": {
            "violation_wrongness_bar1": bar1,
            "derivability_bar2": bar2,
            "constraint_propagation": constraint_propagation,
            "discriminator": constraint_propagation - imitation_propagation,
        },
        "imitators": {imitator.name: asdict(imitator)},
        "guards": {
            "mechanism_engaged": True,
            "mechanism_drop_count": drop_count,
            "propagation_denominator": len(engaged),
            "concept_extraction": "entity/location integer IDs followed by most-recent-movement recurrence",
            "generator_baked_easy_win_guard": "tagged positive-control regardless of cleanliness",
        },
        "stimuli": {
            "source": "pil.qa1_battery.generate_world(seed=0), test_iid queries",
            "count": len(engaged),
            "perturbation": f"append a generated-vocabulary movement to {target_location}",
            "planted_wrong_answer": "the pre-perturbation pair-memory answer",
        },
    }


def _mine_wikitext_nouns() -> tuple[list[tuple[str, str]], dict[str, Any]]:
    forms = [form for pair in SEED_NOUNS for form in pair]
    mined = _cross_venv({"mode": "mine", "forms": forms})
    counts = {word: int(count) for word, count in mined["counts"].items()}
    admitted = [
        pair
        for pair in SEED_NOUNS
        if counts.get(pair[0], 0) >= MIN_WIKITEXT_FORM_FREQUENCY
        and counts.get(pair[1], 0) >= MIN_WIKITEXT_FORM_FREQUENCY
    ]
    admitted.sort(key=lambda pair: (-min(counts[pair[0]], counts[pair[1]]), pair))
    needed = PROBE_TRAIN_LEMMAS + EVAL_LEMMAS
    if len(admitted) < needed:
        raise RuntimeError(f"only {len(admitted)} noun pairs pass frequency floor; need {needed}")
    return admitted, {
        "dataset": WIKITEXT_DATASET,
        "config": WIKITEXT_CONFIG,
        "split": "train",
        "cached_rows_scanned": mined["rows"],
        "cached_arrow_shards": mined["shards"],
        "method": (
            "hand-vetted countable singular/plural seed forms; case-insensitive whole-word counts "
            f"in cached train text; retain both forms at frequency >= {MIN_WIKITEXT_FORM_FREQUENCY}"
        ),
        "seed_noun_lemma_pairs": len(SEED_NOUNS),
        "frequency_filtered_noun_lemma_pairs": len(admitted),
        "frequencies": {word: counts[word] for pair in admitted for word in pair},
    }


def _verb_pair(item_index: int) -> tuple[str, str]:
    return VERB_PAIRS[item_index % len(VERB_PAIRS)]


def _make_item(
    *,
    item_id: str,
    subject_pair: tuple[str, str],
    attractor_pair: tuple[str, str],
    subject_number: int,
    attractor_number: int,
    prep: str,
    verb_index: int,
) -> AgreementItem:
    subject = subject_pair[subject_number]
    attractor = attractor_pair[attractor_number]
    singular_verb, plural_verb = _verb_pair(verb_index)
    correct = (singular_verb, plural_verb)[subject_number]
    wrong = (plural_verb, singular_verb)[subject_number]
    prefix = f"The {subject} {prep} the {attractor}"
    return AgreementItem(
        item_id=item_id,
        prefix=prefix,
        subject=subject,
        attractor=attractor,
        subject_number=subject_number,
        attractor_number=attractor_number,
        correct_verb=correct,
        wrong_verb=wrong,
        prep=prep,
    )


def _subject_span(item: AgreementItem) -> tuple[int, int]:
    start = item.prefix.index(item.subject)
    return start, start + len(item.subject)


def _teacher_task(item: AgreementItem, *, hidden: bool) -> dict[str, Any]:
    task: dict[str, Any] = {
        "prefix": item.prefix,
        "candidate_a": item.correct_verb,
        "candidate_b": item.wrong_verb,
    }
    if hidden:
        task["subject_start"], task["subject_end"] = _subject_span(item)
    return task


def _build_agreement_sets(
    train_lexis: Sequence[tuple[str, str]], eval_lexis: Sequence[tuple[str, str]]
) -> tuple[list[AgreementItem], list[AgreementItem], list[AgreementItem], list[AgreementItem]]:
    probe_train: list[AgreementItem] = []
    for lemma_index, subject_pair in enumerate(train_lexis):
        attractor_pair = train_lexis[(lemma_index + 1) % len(train_lexis)]
        for prep_index, prep in enumerate(FRAMES):
            for number in (0, 1):
                probe_train.append(
                    _make_item(
                        item_id=f"probe:{lemma_index}:{prep}:{number}",
                        subject_pair=subject_pair,
                        attractor_pair=attractor_pair,
                        subject_number=number,
                        attractor_number=number,
                        prep=prep,
                        verb_index=lemma_index + prep_index,
                    )
                )
    bigram_fit: list[AgreementItem] = []
    matched: list[AgreementItem] = []
    mismatch: list[AgreementItem] = []
    for lemma_index, subject_pair in enumerate(eval_lexis):
        attractor_pair = eval_lexis[(lemma_index + 1) % len(eval_lexis)]
        for prep_index, prep in enumerate(FRAMES):
            target = bigram_fit if prep in BIGRAM_FIT_FRAMES else matched
            for number in (0, 1):
                target.append(
                    _make_item(
                        item_id=f"matched:{lemma_index}:{prep}:{number}",
                        subject_pair=subject_pair,
                        attractor_pair=attractor_pair,
                        subject_number=number,
                        attractor_number=number,
                        prep=prep,
                        verb_index=lemma_index + prep_index,
                    )
                )
                if prep in EVAL_FRAMES:
                    mismatch.append(
                        _make_item(
                            item_id=f"mismatch:{lemma_index}:{prep}:{number}",
                            subject_pair=subject_pair,
                            attractor_pair=attractor_pair,
                            subject_number=number,
                            attractor_number=1 - number,
                            prep=prep,
                            verb_index=lemma_index + prep_index,
                        )
                    )
    return probe_train, bigram_fit, matched, mismatch


def _flip_subject(item: AgreementItem, lexicon: dict[str, tuple[str, str]]) -> AgreementItem:
    pair = lexicon[item.subject]
    new_number = 1 - item.subject_number
    new_subject = pair[new_number]
    singular_verb, plural_verb = next(
        pair for pair in VERB_PAIRS if item.correct_verb in pair or item.wrong_verb in pair
    )
    new_correct = (singular_verb, plural_verb)[new_number]
    new_wrong = (plural_verb, singular_verb)[new_number]
    prefix = re.sub(rf"^The {re.escape(item.subject)}\b", f"The {new_subject}", item.prefix)
    return AgreementItem(
        item_id=item.item_id + ":flipped",
        prefix=prefix,
        subject=new_subject,
        attractor=item.attractor,
        subject_number=new_number,
        attractor_number=item.attractor_number,
        correct_verb=new_correct,
        wrong_verb=new_wrong,
        prep=item.prep,
    )


def _fit_logistic_probe(features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray | float]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale
    weights = np.zeros(normalized.shape[1], dtype=np.float64)
    bias = 0.0
    learning_rate = 0.08
    l2 = 1e-3
    for _ in range(1200):
        logits = np.clip(normalized @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - labels
        weights -= learning_rate * ((normalized.T @ error) / len(labels) + l2 * weights)
        bias -= learning_rate * float(error.mean())
    return {"mean": mean, "scale": scale, "weights": weights, "bias": bias}


def _probe_predict(model: dict[str, np.ndarray | float], features: np.ndarray) -> np.ndarray:
    normalized = (features - model["mean"]) / model["scale"]
    return (normalized @ model["weights"] + model["bias"] >= 0.0).astype(np.int64)


def _fit_bigram(items: Sequence[AgreementItem]) -> dict[str, int]:
    counts: dict[str, Counter[int]] = {}
    for item in items:
        counts.setdefault(item.attractor, Counter())[item.subject_number] += 1
    return {
        noun: max((0, 1), key=lambda number: (counter[number], -number))
        for noun, counter in counts.items()
    }


def run_arm2() -> dict[str, Any]:
    """Run the natural-text English agreement-attractor arm."""
    admitted, provenance = _mine_wikitext_nouns()
    train_lexis = admitted[:PROBE_TRAIN_LEMMAS]
    eval_lexis = admitted[PROBE_TRAIN_LEMMAS : PROBE_TRAIN_LEMMAS + EVAL_LEMMAS]
    train_words = {word for pair in train_lexis for word in pair}
    eval_words = {word for pair in eval_lexis for word in pair}
    shared_words = train_words & eval_words
    if shared_words:
        raise AssertionError(f"probe/eval subject words overlap: {sorted(shared_words)}")

    probe_train, bigram_fit, matched, mismatch = _build_agreement_sets(train_lexis, eval_lexis)
    word_to_pair = {word: pair for pair in eval_lexis for word in pair}
    flipped_all = [_flip_subject(item, word_to_pair) for item in matched]
    engaged_pairs, mechanism_drop_count = drop_mechanism_noops(
        list(zip(matched, flipped_all, strict=True)),
        old_gold=lambda pair: pair[0].subject_number,
        new_gold=lambda pair: pair[1].subject_number,
    )
    matched = [pair[0] for pair in engaged_pairs]
    flipped = [pair[1] for pair in engaged_pairs]

    all_hidden_items = [*probe_train, *matched, *flipped]
    tasks = [_teacher_task(item, hidden=True) for item in all_hidden_items]
    tasks.extend(_teacher_task(item, hidden=False) for item in mismatch)
    teacher_results = _teacher(tasks)
    hidden_results = teacher_results[: len(all_hidden_items)]
    mismatch_results = teacher_results[len(all_hidden_items) :]

    train_end = len(probe_train)
    matched_end = train_end + len(matched)
    train_features = np.asarray([result["hidden"] for result in hidden_results[:train_end]])
    matched_features = np.asarray(
        [result["hidden"] for result in hidden_results[train_end:matched_end]]
    )
    flipped_features = np.asarray([result["hidden"] for result in hidden_results[matched_end:]])
    train_labels = np.asarray([item.subject_number for item in probe_train], dtype=np.float64)
    probe = _fit_logistic_probe(train_features, train_labels)
    train_predictions = _probe_predict(probe, train_features)
    matched_concepts = _probe_predict(probe, matched_features).tolist()
    flipped_concepts = _probe_predict(probe, flipped_features).tolist()

    python_derived = agreement_aggregate_python(matched_concepts)
    souffle_derived = souffle_agreement(matched_concepts)
    gold = [item.subject_number for item in matched]
    three_way = [
        gold_value == python_value == souffle_value
        for gold_value, python_value, souffle_value in zip(
            gold, python_derived, souffle_derived, strict=True
        )
    ]
    bar2 = sum(three_way) / len(three_way)
    mirror_agreement = sum(
        python_value == souffle_value
        for python_value, souffle_value in zip(python_derived, souffle_derived, strict=True)
    ) / len(python_derived)
    constraint_predictions = agreement_aggregate_python(flipped_concepts)
    new_gold = [item.subject_number for item in flipped]
    constraint_propagation = sum(
        predicted == expected
        for predicted, expected in zip(constraint_predictions, new_gold, strict=True)
    ) / len(new_gold)

    bar1 = sum(
        result["logprob_a"] > result["logprob_b"] for result in mismatch_results
    ) / len(mismatch_results)

    bigram = _fit_bigram(bigram_fit)
    nearest_unperturbed = sum(item.attractor_number == item.subject_number for item in matched) / len(matched)
    nearest_propagation = sum(item.attractor_number == item.subject_number for item in flipped) / len(flipped)
    bigram_unperturbed = sum(bigram[item.attractor] == item.subject_number for item in matched) / len(matched)
    bigram_propagation = sum(bigram[item.attractor] == item.subject_number for item in flipped) / len(flipped)
    imitators = [
        ImitatorScore("nearest-noun", nearest_unperturbed, nearest_propagation),
        ImitatorScore("learned-preceding-noun-bigram", bigram_unperturbed, bigram_propagation),
    ]
    verdict = fixed_verdict(
        bar1=bar1,
        bar2=bar2,
        constraint_propagation=constraint_propagation,
        imitators=imitators,
    )

    eval_distance_results = hidden_results[train_end:]
    distances = [int(result["token_distance"]) for result in eval_distance_results]
    distance_holds = all(distance >= DISTANCE_K for distance in distances)
    attractor_between = all(
        item.prefix.index(item.subject) < item.prefix.index(item.attractor)
        for item in [*matched, *flipped, *mismatch]
    )
    if not distance_holds or not attractor_between:
        raise AssertionError("distance/attractor guard failed")

    provenance.update(
        {
            "probe_train_noun_lemma_pairs": len(train_lexis),
            "eval_noun_lemma_pairs": len(eval_lexis),
            "final_subject_attractor_pairs_used": len(eval_lexis),
            "probe_train_lexis": train_lexis,
            "eval_lexis": eval_lexis,
        }
    )
    return {
        "name": "English subject-verb number agreement with attractor",
        "role": "substantive natural-text arm",
        "tag": "empirical, domain-restricted to canonical templated syntax",
        "naturalness_caveat": (
            "A FIRES certifies agreement forcing on canonical templated syntax, not in the wild."
        ),
        **asdict(verdict),
        "metrics": {
            "violation_wrongness_bar1": bar1,
            "derivability_bar2": bar2,
            "constraint_propagation": constraint_propagation,
            "discriminator": constraint_propagation
            - max(imitator.propagation for imitator in imitators),
            "python_souffle_mirror_agreement": mirror_agreement,
            "probe_train_accuracy": float((train_predictions == train_labels).mean()),
        },
        "imitators": {imitator.name: asdict(imitator) for imitator in imitators},
        "stimuli": {
            "provenance": provenance,
            "frames": list(FRAMES),
            "bigram_fit_frames": list(BIGRAM_FIT_FRAMES),
            "measured_eval_frames": list(EVAL_FRAMES),
            "matched_base_count": len(matched),
            "mismatch_bar1_only_count": len(mismatch),
            "probe_train_sentence_count": len(probe_train),
            "bigram_fit_sentence_count": len(bigram_fit),
            "matched_usage": "bar 2 and pre/post-flip propagation discriminator only",
            "mismatch_usage": "bar 1 violation-wrongness only",
            "verb_targets": [list(pair) for pair in VERB_PAIRS],
        },
        "guards": {
            "distance_k_gpt2_tokens": DISTANCE_K,
            "minimum_observed_token_distance": min(distances),
            "distance_holds_every_eval_item": distance_holds,
            "attractor_strictly_between_every_eval_item": attractor_between,
            "adjacency_excluded": True,
            "mechanism_engaged": mechanism_drop_count == 0,
            "mechanism_drop_count": mechanism_drop_count,
            "propagation_denominator": len(flipped),
            "probe_train_subject_word_count": len(train_words),
            "eval_subject_word_count": len(eval_words),
            "probe_eval_shared_subject_words": len(shared_words),
            "probe_train_eval_lexis_disjoint": len(shared_words) == 0,
            "three_way_gold_python_souffle_agreement": bar2,
            "python_souffle_agree_100_percent": mirror_agreement == 1.0,
            "strong_imitator_floor": IMITATOR_UNPERTURBED_MIN,
            "stronger_imitator_propagation_ceiling": IMITATOR_PROPAGATION_MAX,
        },
        "probe": {
            "teacher_hidden_state": f"GPT-2 hidden state {TEACHER_LAYER} at final subject token",
            "classifier": "binary logistic regression with L2 penalty",
            "aggregate": "integer identity: predicted verb number = probed subject number",
            "souffle_rules": 1,
        },
    }


def _format_arm(arm: dict[str, Any]) -> list[str]:
    metrics = arm["metrics"]
    verdict = arm["verdict"]
    if arm["failure"]:
        verdict += f" ({arm['failure']})"
    lines = [
        f"[{arm['name']}]",
        f"  verdict: {verdict}",
        f"  tag: {arm['tag']}",
        f"  bar 1 violation-wrongness: {metrics['violation_wrongness_bar1']:.6f} (> 0.95)",
        f"  bar 2 derivability: {metrics['derivability_bar2']:.6f} (>= 0.90)",
        f"  constraint propagation: {metrics['constraint_propagation']:.6f} (>= 0.95)",
        f"  discriminator: {metrics['discriminator']:.6f}",
    ]
    for name, score in arm["imitators"].items():
        lines.append(
            f"  imitator {name}: unperturbed={score['unperturbed']:.6f}, "
            f"propagation={score['propagation']:.6f}"
        )
    for key, value in arm["guards"].items():
        lines.append(f"  guard {key}: {value}")
    return lines


def main() -> int:
    scoreboard: dict[str, Any] = {
        "pilot": "Probe A semantic forcing",
        "seed": SEED,
        "teacher": {
            "model": TEACHER_MODEL,
            "forward_pass": "real AutoModelForCausalLM forward passes in lm-sae venv",
            "candidate_scoring": "single-token continuation log-probability",
            "hidden_state_layer": TEACHER_LAYER,
            "reason": "small cached causal LM; all registered English verb candidates are one BPE token",
        },
        "fixed_bars": {
            "bar1_violation_wrongness": "> 0.95",
            "bar2_derivability": ">= 0.90",
            "both_imitators_unperturbed_arm2": ">= 0.80",
            "stronger_imitator_propagation": "<= 0.50",
            "constraint_propagation": ">= 0.95",
        },
        "arms": {},
    }
    arm1 = run_arm1()
    scoreboard["arms"]["arm1_babi_moveloc"] = arm1
    arm2 = run_arm2()
    scoreboard["arms"]["arm2_english_agreement"] = arm2
    harness_broken = arm1["verdict"] != "FIRES"
    scoreboard["run_status"] = "harness broken" if harness_broken else "complete"
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(scoreboard, indent=2, sort_keys=True) + "\n")

    print("SEMANTIC FORCING SCOREBOARD")
    print("\n".join(_format_arm(arm1)))
    print("\n".join(_format_arm(arm2)))
    provenance = arm2["stimuli"]["provenance"]
    print(
        "  provenance: "
        f"{provenance['dataset']} ({provenance['config']}), {provenance['method']}"
    )
    print(
        "  mined counts: "
        f"{provenance['frequency_filtered_noun_lemma_pairs']} frequency-filtered noun lemma pairs; "
        f"{provenance['final_subject_attractor_pairs_used']} final subject-attractor pairs used"
    )
    print(
        "  set split: mismatch is bar-1 only; matched is bar-2/discriminator only "
        f"(matched={arm2['stimuli']['matched_base_count']}, "
        f"mismatch={arm2['stimuli']['mismatch_bar1_only_count']})"
    )
    if harness_broken:
        print("HARNESS BROKEN: Arm 1 positive control did not fire; Arm 2 is reported diagnostically.")
    print(f"JSON: {OUTPUT_JSON}")
    return 2 if harness_broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
