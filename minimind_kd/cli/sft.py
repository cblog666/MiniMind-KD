from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from minimind_kd.config import load_yaml_config
from minimind_kd.training.data import SFTDataset
from minimind_kd.training.supervised import train_supervised

from .common import load_model, load_tokenizer, synchronize_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain supervised fine-tuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_config, train_config, _ = load_yaml_config(args.config)
    tokenizer = load_tokenizer(args.tokenizer)
    synchronize_vocabulary(model_config, tokenizer)
    dataset = SFTDataset(args.data, tokenizer, train_config.sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = load_model(model_config, args.checkpoint)
    train_supervised(model, loader, train_config, args.output)


if __name__ == "__main__":
    main()
