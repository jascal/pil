"""Pinned Hugging Face causal LMs for the certification studies (fp32; decode input = final-norm output).

Revisions are fixed in docs/notes/beyond_gpt2_prereg.md. Requires `transformers` (an optional dependency).
"""

from __future__ import annotations

import torch

REVISIONS = {
    "gpt2": None,  # the GPT-2 studies used the cached default revision
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "meta-llama/Llama-3.2-1B": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
}


def load(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = REVISIONS[model_id]
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.float32).to(device).eval()
    )
    return tok, model


def unembedding(model) -> torch.Tensor:
    return model.get_output_embeddings().weight.detach().float()


def decode_inputs(model, input_ids, attention_mask) -> torch.Tensor:
    """Final-norm hidden state at each row's last real position (the input to the tied unembedding)."""
    h = model.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    last = attention_mask.sum(1) - 1
    return h[torch.arange(len(last), device=h.device), last].float()


def single_token(tok, word: str) -> bool:
    return len(tok.encode(" " + word, add_special_tokens=False)) == 1


def bos_id(tok):
    """The BOS id the tokenizer prepends by default (Llama), else None (GPT-2, Qwen)."""
    ids = tok("x").input_ids
    return ids[0] if tok.bos_token_id is not None and ids and ids[0] == tok.bos_token_id else None
