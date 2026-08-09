from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset


class TokenizerLike(Protocol):
    pad_token_id: int
    eos_token_id: int

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
            records.append(item)
    return records


def _pad_example(
    token_ids: list[int],
    labels: list[int],
    sequence_length: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    token_ids = token_ids[:sequence_length]
    labels = labels[:sequence_length]
    padding = sequence_length - len(token_ids)
    attention = [1] * len(token_ids) + [0] * padding
    token_ids += [pad_token_id] * padding
    labels += [-100] * padding
    return {
        "input_ids": torch.tensor(token_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.bool),
    }


class PackedPretrainDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: TokenizerLike,
        sequence_length: int,
        text_field: str = "text",
    ) -> None:
        stream: list[int] = []
        for record in read_jsonl(path):
            text = record.get(text_field)
            if not isinstance(text, str):
                raise ValueError(f"Each pre-training record needs a string '{text_field}'")
            stream.extend(tokenizer.encode(text, add_special_tokens=False))
            stream.append(tokenizer.eos_token_id)
        if not stream:
            raise ValueError("Pre-training data is empty")
        self.examples = [
            stream[index : index + sequence_length]
            for index in range(0, len(stream), sequence_length)
            if len(stream[index : index + sequence_length]) >= 2
        ]
        if not self.examples:
            raise ValueError("Pre-training data needs at least two tokens")
        self.sequence_length = sequence_length
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = list(self.examples[index])
        return _pad_example(tokens, list(tokens), self.sequence_length, self.pad_token_id)


def render_messages(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message.get("role", "user").strip().capitalize()
        content = message.get("content", "")
        chunks.append(f"{role}: {content}\n")
    return "".join(chunks)


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: TokenizerLike,
        sequence_length: int,
    ) -> None:
        self.records = read_jsonl(path)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        if "messages" in record:
            messages = record["messages"]
            if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
                raise ValueError("SFT messages must end with an assistant response")
            prompt = render_messages(messages[:-1]) + "Assistant: "
            response = str(messages[-1].get("content", ""))
        else:
            prompt = str(record.get("prompt", ""))
            response = str(record.get("response", ""))
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False) + [
            self.tokenizer.eos_token_id
        ]
        if len(response_ids) >= self.sequence_length:
            response_ids = response_ids[: self.sequence_length]
            response_ids[-1] = self.tokenizer.eos_token_id
            prompt_ids = []
        else:
            # Preserve the supervised answer when long prompts need truncation.
            prompt_budget = self.sequence_length - len(response_ids)
            prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        return _pad_example(
            input_ids,
            labels,
            self.sequence_length,
            self.tokenizer.pad_token_id,
        )


class PromptDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path) -> None:
        self.records = read_jsonl(path)
        for record in self.records:
            if not isinstance(record.get("prompt"), str):
                raise ValueError("Each prompt record needs a string 'prompt'")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]
