from __future__ import annotations

import argparse
from pathlib import Path

from minimind_kd.protocol import SPECIAL_TOKENS
from minimind_kd.training.data import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local ByteLevel BPE tokenizer")
    parser.add_argument("--data", required=True, help="JSONL containing a text field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=6400)
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError("Install the 'train' dependencies to train a tokenizer") from exc

    records = read_jsonl(args.data)

    def corpus():
        for record in records:
            text = record.get(args.text_field)
            if isinstance(text, str):
                yield text

    special = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", *SPECIAL_TOKENS]
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        corpus(),
        trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=special),
    )
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        additional_special_tokens=SPECIAL_TOKENS,
    )
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    wrapped.save_pretrained(destination)
    print(f"saved tokenizer with {len(wrapped)} tokens to {destination}")


if __name__ == "__main__":
    main()
