from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from minimind_kd.config import load_yaml_config
from minimind_kd.training.data import PromptDataset
from minimind_kd.training.grpo import GRPOConfig, GRPOTrainer
from minimind_kd.training.optim import build_optimizer
from minimind_kd.training.rewards import WeightedRewards
from minimind_kd.training.supervised import resolve_device, seed_everything

from .common import load_model, load_tokenizer, synchronize_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a domain expert with GRPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True, help="Domain SFT checkpoint")
    parser.add_argument("--reference", help="Frozen reference; defaults to the SFT checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prompts-per-step", type=int, default=1)
    args = parser.parse_args()

    model_config, train_config, extra = load_yaml_config(args.config)
    tokenizer = load_tokenizer(args.tokenizer)
    synchronize_vocabulary(model_config, tokenizer)
    device = resolve_device(train_config.device)
    seed_everything(train_config.seed)
    policy = load_model(model_config, args.checkpoint).to(device)
    reference = (
        load_model(model_config, args.reference).to(device)
        if args.reference
        else copy.deepcopy(policy).to(device)
    )
    optimizer = build_optimizer(policy, train_config)
    grpo_config = GRPOConfig(**extra.get("grpo", {}))
    reward = WeightedRewards(extra.get("rewards", {"exact_match": 1.0, "reasoning_format": 0.1}))
    trainer = GRPOTrainer(policy, reference, tokenizer, optimizer, reward, grpo_config, device)
    dataset = PromptDataset(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "metrics.jsonl"
    records = [dataset[index] for index in range(len(dataset))]
    if not records:
        raise ValueError("GRPO dataset is empty")
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(args.steps):
            start = (step * args.prompts_per_step) % len(records)
            batch = [records[(start + offset) % len(records)] for offset in range(args.prompts_per_step)]
            metrics = trainer.step(batch)
            serializable = {key: value for key, value in metrics.items() if key != "completions"}
            serializable["step"] = step + 1
            log.write(json.dumps(serializable, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(serializable, ensure_ascii=False), flush=True)
    policy.save_pretrained(output / "final")


if __name__ == "__main__":
    main()
