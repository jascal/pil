"""Build standalone CFQ pair datasets (WordCodec, no host LLM).

Source: HuggingFace ``google-research-datasets/cfq`` (Keysers et al., ICLR 2020).
Official Google GCS tarball requires auth; HF mirrors the same splits.

Configs (default): mcd1, mcd2, mcd3, random_split.

Writes per config tag:
  data/alphabet_cfq_{tag}.json
  data/cfq_{tag}.pt          # train/test question_ids + query_ids
  data/cfq_{tag}_meta.json

Requires: ``pip install datasets`` (HF datasets; optional, not a core pil dep).

Run: cd pil && .venv/bin/python experiments/build_cfq.py
Env:  CFQ_CONFIGS=mcd1,random_split   CFQ_MAX_TRAIN=20000 (optional subsample)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pil.alphabet import WordCodec  # noqa: E402

DEFAULT_CONFIGS = ["mcd1", "mcd2", "mcd3", "random_split"]
HF_NAME = "google-research-datasets/cfq"


def tag_of(config: str) -> str:
    return config.replace("_split", "")  # random_split -> random


def load_hf(config: str, split: str, max_n: int | None = None):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "CFQ build needs HuggingFace datasets: pip install datasets\n"
            f"({e})"
        ) from e
    spec = split if max_n is None else f"{split}[:{max_n}]"
    return load_dataset(HF_NAME, config, split=spec)


def build_config(config: str, max_train: int | None = None, max_test: int | None = None) -> None:
    tag = tag_of(config)
    print(f"loading CFQ/{config} ...", flush=True)
    train = load_hf(config, "train", max_train)
    test = load_hf(config, "test", max_test)
    pairs_tr = [(r["question"], r["query"]) for r in train]
    pairs_te = [(r["question"], r["query"]) for r in test]

    corpus = "\n".join(q + "\n" + a for q, a in pairs_tr)
    codec = WordCodec.from_corpus(corpus)
    codec_path = REPO / "data" / f"alphabet_cfq_{tag}.json"
    codec.save(codec_path)

    def enc_pairs(pairs):
        qs, sparqls = [], []
        for q, a in pairs:
            qi = codec.encode(q)
            ai = codec.encode(a)
            if not qi or not ai:
                continue
            qs.append(qi)
            sparqls.append(ai)
        return qs, sparqls

    tr_in, tr_out = enc_pairs(pairs_tr)
    te_in, te_out = enc_pairs(pairs_te)
    # keep raw strings for exact string baselines (tokenization can collide)
    blob = {
        "split": tag,
        "config": config,
        "alphabet": codec_path.name,
        "kind": "word",
        "train_in": tr_in,
        "train_out": tr_out,
        "test_in": te_in,
        "test_out": te_out,
        "train_q": [q for q, _ in pairs_tr],
        "train_sparql": [a for _, a in pairs_tr],
        "test_q": [q for q, _ in pairs_te],
        "test_sparql": [a for _, a in pairs_te],
        "n_train": len(tr_in),
        "n_test": len(te_in),
        "vocab_size": codec.vocab_size,
    }
    out_pt = REPO / "data" / f"cfq_{tag}.pt"
    torch.save(blob, out_pt)
    meta = {
        "split": tag,
        "config": config,
        "n_train": len(tr_in),
        "n_test": len(te_in),
        "vocab_size": codec.vocab_size,
        "alphabet_hash": codec.hash(),
        "max_q_len": max(len(x) for x in tr_in + te_in),
        "max_sparql_len": max(len(x) for x in tr_out + te_out),
        "origin": "standalone",
        "source": HF_NAME,
    }
    (REPO / "data" / f"cfq_{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"cfq/{tag}: train={len(tr_in)} test={len(te_in)} vocab={codec.vocab_size} "
        f"max_q={meta['max_q_len']} max_sparql={meta['max_sparql_len']} -> {out_pt.name}",
        flush=True,
    )


def main():
    configs = [c.strip() for c in os.environ.get("CFQ_CONFIGS", ",".join(DEFAULT_CONFIGS)).split(",")
               if c.strip()]
    max_train = os.environ.get("CFQ_MAX_TRAIN")
    max_test = os.environ.get("CFQ_MAX_TEST")
    mt = int(max_train) if max_train else None
    mte = int(max_test) if max_test else None
    (REPO / "data").mkdir(exist_ok=True)
    for c in configs:
        build_config(c, max_train=mt, max_test=mte)


if __name__ == "__main__":
    main()
