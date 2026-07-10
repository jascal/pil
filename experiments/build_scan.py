"""Build standalone SCAN pair datasets (WordCodec, no host LLM).

Source: Facebook SCAN (Lake & Baroni 2018) split files vendored under data/scan/.
Format per line:  IN: <command> OUT: <actions>

Writes for each split (length, addprim_jump, simple):
  data/alphabet_scan_{split}.json
  data/scan_{split}.pt   # train/test command_ids + action_ids lists
  data/scan_{split}_meta.json

Run: cd pil && .venv/bin/python experiments/build_scan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.alphabet import WordCodec  # noqa: E402

SCAN_ROOT = REPO / "data" / "scan"

# (split_tag, train_file, test_file) relative to SCAN_ROOT
SPLITS = [
    ("length", "length_split/tasks_train_length.txt", "length_split/tasks_test_length.txt"),
    ("addprim_jump", "add_prim_split/tasks_train_addprim_jump.txt",
     "add_prim_split/tasks_test_addprim_jump.txt"),
    ("simple", "simple_split/tasks_train_simple.txt", "simple_split/tasks_test_simple.txt"),
]


def parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line.startswith("IN:"):
        return None
    if " OUT:" not in line:
        return None
    left, right = line.split(" OUT:", 1)
    cmd = left[len("IN:"):].strip()
    act = right.strip()
    return cmd, act


def load_pairs(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        p = parse_line(line)
        if p:
            out.append(p)
    return out


def build_split(tag: str, train_rel: str, test_rel: str) -> None:
    train_p = SCAN_ROOT / train_rel
    test_p = SCAN_ROOT / test_rel
    if not train_p.exists() or not test_p.exists():
        print(f"skip {tag}: missing {train_p} or {test_p}")
        return
    train = load_pairs(train_p)
    test = load_pairs(test_p)
    # alphabet from TRAIN only (standalone: no test leakage into vocab)
    corpus = "\n".join(c + " " + a for c, a in train)
    codec = WordCodec.from_corpus(corpus)
    codec_path = REPO / "data" / f"alphabet_scan_{tag}.json"
    codec.save(codec_path)

    def enc_pairs(pairs):
        cins, aouts = [], []
        for c, a in pairs:
            ci = codec.encode(c)
            ao = codec.encode(a)
            if not ci or not ao:
                continue
            cins.append(ci)
            aouts.append(ao)
        return cins, aouts

    tr_in, tr_out = enc_pairs(train)
    te_in, te_out = enc_pairs(test)
    # pad to tensors for convenient storage (variable length → lists in object arrays)
    blob = {
        "split": tag,
        "alphabet": codec_path.name,
        "kind": "word",
        "train_in": tr_in,
        "train_out": tr_out,
        "test_in": te_in,
        "test_out": te_out,
        "n_train": len(tr_in),
        "n_test": len(te_in),
        "vocab_size": codec.vocab_size,
    }
    out_pt = REPO / "data" / f"scan_{tag}.pt"
    torch.save(blob, out_pt)
    meta = {
        "split": tag,
        "n_train": len(tr_in),
        "n_test": len(te_in),
        "vocab_size": codec.vocab_size,
        "alphabet_hash": codec.hash(),
        "train_file": train_rel,
        "test_file": test_rel,
        "max_cmd_len": max(len(x) for x in tr_in + te_in),
        "max_act_len": max(len(x) for x in tr_out + te_out),
        "origin": "standalone",
    }
    (REPO / "data" / f"scan_{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"scan/{tag}: train={len(tr_in)} test={len(te_in)} vocab={codec.vocab_size} "
        f"max_cmd={meta['max_cmd_len']} max_act={meta['max_act_len']} -> {out_pt.name}"
    )


def main():
    if not SCAN_ROOT.exists():
        raise SystemExit(
            f"missing {SCAN_ROOT} — clone brendenlake/SCAN and copy length_split, "
            "add_prim_split, simple_split under data/scan/"
        )
    for tag, tr, te in SPLITS:
        build_split(tag, tr, te)


if __name__ == "__main__":
    main()
