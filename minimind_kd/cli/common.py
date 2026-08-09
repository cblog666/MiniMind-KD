from __future__ import annotations

from pathlib import Path
from typing import Any

from minimind_kd.config import ModelConfig
from minimind_kd.modeling.model import MiniMindKDForCausalLM


def load_tokenizer(path: str | Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the 'train' dependencies to load a tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer must define a dedicated pad_token_id. Train the bundled tokenizer or add a pad token "
            "and update model.vocab_size explicitly."
        )
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Tokenizer pad_token_id and eos_token_id must be distinct")
    return tokenizer


def synchronize_vocabulary(config: ModelConfig, tokenizer: Any) -> None:
    tokenizer_size = len(tokenizer)
    if config.vocab_size != tokenizer_size:
        raise ValueError(
            f"config vocab_size={config.vocab_size}, tokenizer size={tokenizer_size}. "
            "Set model.vocab_size to the exact tokenizer size before training."
        )
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id and eos_token_id")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Tokenizer pad_token_id and eos_token_id must be distinct")
    ids = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    for name, value in ids.items():
        if value is not None:
            setattr(config, name, int(value))


def load_model(config: ModelConfig, checkpoint: str | None) -> MiniMindKDForCausalLM:
    if checkpoint:
        return MiniMindKDForCausalLM.from_pretrained(checkpoint, config=config)
    return MiniMindKDForCausalLM(config)
