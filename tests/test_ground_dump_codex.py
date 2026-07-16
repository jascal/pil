import json
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.ground_dump_codex as ground_dump  # noqa: E402
from experiments.ground_dump_codex import (  # noqa: E402
    GsdSentence,
    align_subwords_to_words,
    choose_checkpoint_layers,
    encode_with_safe_pad,
    load_batched_source_dump,
    reduce_clean_words,
    validate_sentence_dump,
)

TOKENIZER_PATH = Path(
    "/home/allans/code/fieldrun/bundles/Qwen2.5-0.5B-Instruct/"
    "Qwen2.5-0.5B-Instruct.tokenizer.json"
)
TEXT = "Manasse ist ein einzigartiger Parfümeur."
TOKENS = ("Manasse", "ist", "ein", "einzigartiger", "Parfümeur", ".")


def _source_record(sid: str, position: int, cur: int) -> dict:
    blocks = ["embed", "L0.attn", "L0.mlp", "L1.attn", "L1.mlp"]
    contribution = [[float(position), float(position) + 0.5]] + [[0.0, 0.0]] * 4
    return {
        "sid": sid,
        "pos": position,
        "cur": cur,
        "target": 0,
        "tgt_idx": 0,
        "pred": 0,
        "margin": 0.0,
        "dim": 2,
        "cands": [0],
        "blocks": blocks,
        "d": contribution,
    }


def test_real_tokenizer_alignment_and_last_subword_residual(tmp_path):
    """The compound must use subword/position 7, not either adjacent word."""
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    sentence = GsdSentence("fixture-s1", TEXT, TOKENS)
    encoded = encode_with_safe_pad(tokenizer, sentence, " x")

    # Re-derived from the actual tokenizer at test time: padding leaves the 13 real
    # sentence token ids intact and creates a terminal token after the full stop.
    assert encoded.pad_safety_passed
    assert len(encoded.original_ids) == 13
    assert encoded.dump_ids[:13] == encoded.original_ids

    # A synthetic exact-schema raw dump makes each cumulative residual encode its
    # fieldrun position.  Writing positions in reverse order also exercises loader
    # sorting rather than relying on emission order.
    records = [
        _source_record(sentence.sent_id, position, encoded.dump_ids[position])
        for position in reversed(range(1, 13))
    ]
    dump_path = tmp_path / "fixture.source.jsonl"
    dump_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    dump = load_batched_source_dump(dump_path)[sentence.sent_id]
    validation = validate_sentence_dump(dump, encoded.dump_ids, n=len(encoded.dump_ids) + 8)
    assert validation.passed
    np.testing.assert_array_equal(dump.r, dump.D.sum(axis=1))

    alignment = align_subwords_to_words(
        sentence.sent_id,
        sentence.text,
        sentence.tokens,
        encoded.original_offsets,
        set(dump.positions.tolist()),
    )
    assert alignment.word_to_subwords == (
        (0, 1),
        (2,),
        (3,),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (12,),
    )
    assert all(alignment.clean_words)
    compound_subwords = alignment.word_to_subwords[3]
    assert compound_subwords[-1] == 7
    assert compound_subwords[-1] not in {3, 4, 8}

    layers, _fractions = choose_checkpoint_layers(dump.blocks)
    reduced = reduce_clean_words(dump, alignment, layers)
    compound_row = int(np.flatnonzero(reduced.token_indices == 3)[0])
    assert reduced.last_subword_indices[compound_row] == 7
    # Only embed is nonzero, so both checkpoints must reproduce position 7 exactly.
    expected_last = np.asarray([[7.0, 7.5], [7.0, 7.5]], dtype=np.float16)
    np.testing.assert_array_equal(reduced.last_residual[compound_row], expected_last)
    # Its available positions are 4,5,6,7, whose mean position is 5.5.
    expected_mean = np.asarray([[5.5, 6.0], [5.5, 6.0]], dtype=np.float16)
    np.testing.assert_array_equal(reduced.mean_residual[compound_row], expected_mean)


def test_coverage_metrics_clean_boundary_and_systematic_gaps():
    clean = align_subwords_to_words(
        "clean",
        "ab cd",
        ("ab", "cd"),
        ((0, 2), (2, 5)),
        {0, 1},
    ).coverage
    assert (clean.clean_words, clean.total_words) == (2, 2)
    assert clean.boundary_straddling_subwords == 0
    assert clean.leading_single_subword_word_gaps == 0
    assert clean.last_word_gaps == 0

    boundary = align_subwords_to_words(
        "boundary",
        "ab cd",
        ("ab", "cd"),
        ((0, 2), (1, 4), (3, 5)),
        {0, 1, 2},
    ).coverage
    assert (boundary.clean_words, boundary.total_words) == (0, 2)
    assert boundary.boundary_straddling_subwords == 1
    assert boundary.leading_single_subword_word_gaps == 0
    assert boundary.last_word_gaps == 0

    leading_gap = align_subwords_to_words(
        "leading",
        "Ich geht",
        ("Ich", "geht"),
        ((0, 3), (3, 8)),
        {1},
    ).coverage
    assert (leading_gap.clean_words, leading_gap.total_words) == (1, 2)
    assert leading_gap.leading_single_subword_word_gaps == 1
    assert leading_gap.last_word_gaps == 0

    last_gap = align_subwords_to_words(
        "last",
        "Hallo Welt",
        ("Hallo", "Welt"),
        ((0, 5), (5, 10)),
        {0},
    ).coverage
    assert (last_gap.clean_words, last_gap.total_words) == (1, 2)
    assert last_gap.leading_single_subword_word_gaps == 0
    assert last_gap.last_word_gaps == 1


def test_build_dataset_chunks_merges_and_resumes(tmp_path, monkeypatch, capsys):
    bundle_path = tmp_path / "fixture-bundle"
    tokenizer_path = Path(f"{bundle_path}.tokenizer.json")
    tokenizer_path.write_bytes(TOKENIZER_PATH.read_bytes())
    data_root = tmp_path / "data"
    data_root.mkdir()
    sentences = [
        {"sent_id": f"fixture-s{index}", "text": TEXT, "tokens": TOKENS}
        for index in range(3)
    ]
    (data_root / "dev.jsonl").write_text(
        "".join(json.dumps(sentence) + "\n" for sentence in sentences), encoding="utf-8"
    )
    output_root = tmp_path / "output"
    calls = []

    def fake_run_fieldrun_batch(fieldrun, bundle, encoded, n, kcand, work_dir):
        calls.append(([item.sentence.sent_id for item in encoded], n, kcand))
        records = [
            _source_record(item.sentence.sent_id, position, item.dump_ids[position])
            for item in encoded
            for position in range(1, len(item.dump_ids) - 1)
        ]
        dump_path = work_dir / "source.jsonl"
        dump_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        return dump_path, float(len(encoded))

    monkeypatch.setattr(ground_dump, "run_fieldrun_batch", fake_run_fieldrun_batch)
    kwargs = {
        "model_tag": "fixture",
        "bundle_stem": str(bundle_path),
        "split": "dev",
        "limit": 3,
        "fieldrun_path": tmp_path / "fieldrun",
        "data_root": data_root,
        "output_root": output_root,
        "chunk_size": 2,
    }
    output_path, provenance = ground_dump.build_dataset(**kwargs)
    first_stdout = capsys.readouterr().out

    assert [call[0] for call in calls] == [["fixture-s0", "fixture-s1"], ["fixture-s2"]]
    assert calls[0][1] == calls[1][1]
    assert first_stdout.count("running fieldrun") == 2
    assert len(list(output_root.glob("*_partials/chunk*.npz"))) == 2
    assert provenance["fieldrun_subprocess_seconds"] == 3.0
    with np.load(output_path, allow_pickle=False) as result:
        assert set(result.files) == {
            "sent_ids",
            "token_index",
            "word_text",
            "last_subword_index",
            "checkpoint_layers",
            "checkpoint_fractions",
            "last_residual",
            "mean_residual",
            "provenance",
        }
        assert list(dict.fromkeys(result["sent_ids"].tolist())) == [
            "fixture-s0",
            "fixture-s1",
            "fixture-s2",
        ]

    resumed_path, resumed_provenance = ground_dump.build_dataset(**kwargs)
    second_stdout = capsys.readouterr().out
    assert resumed_path == output_path
    assert resumed_provenance == provenance
    assert len(calls) == 2
    assert second_stdout.count("skipping fieldrun") == 2


def test_build_dataset_rejects_nonpositive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ground_dump.build_dataset(
            model_tag="unused",
            bundle_stem="unused",
            split="dev",
            limit=1,
            chunk_size=0,
        )
