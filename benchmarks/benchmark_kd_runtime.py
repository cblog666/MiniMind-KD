#!/usr/bin/env python3
"""Reproducible runtime benchmark for MiniMind-KD only.

This script intentionally does not benchmark upstream MiniMind. Its published
accuracy numbers are cited verbatim in the project README. Synthetic token IDs
remove tokenizer and storage noise from this architecture/runtime measurement;
they do not measure model quality.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from minimind_kd.config import load_yaml_config
from minimind_kd.modeling import MiniMindKDForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--forward-warmup-steps", type=int, default=2)
    parser.add_argument("--forward-steps", type=int, default=5)
    parser.add_argument("--train-warmup-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--decode-prompt-length", type=int, default=16)
    parser.add_argument("--decode-new-tokens", type=int, default=8)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, max_seq_len: int) -> None:
    positive = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "threads": args.threads,
        "forward_steps": args.forward_steps,
        "train_steps": args.train_steps,
        "decode_prompt_length": args.decode_prompt_length,
        "decode_new_tokens": args.decode_new_tokens,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Benchmark values must be positive: {', '.join(invalid)}")
    if args.forward_warmup_steps < 0 or args.train_warmup_steps < 0:
        raise ValueError("Warmup step counts cannot be negative")
    if args.sequence_length > max_seq_len:
        raise ValueError("sequence length exceeds model max_seq_len")
    if args.decode_prompt_length + args.decode_new_tokens > max_seq_len:
        raise ValueError("prompt plus decoded tokens exceeds model max_seq_len")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(call: Any, *, device: torch.device) -> tuple[Any, float]:
    synchronize(device)
    started = time.perf_counter()
    result = call()
    synchronize(device)
    return result, time.perf_counter() - started


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def parameter_counts(
    model: MiniMindKDForCausalLM,
) -> tuple[int, int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    routed = sum(
        parameter.numel() for name, parameter in model.named_parameters() if ".routed_experts." in name
    )
    config = model.config
    active = total - routed + routed * config.num_experts_per_token / config.num_routed_experts
    return total, int(round(active)), trainable


def optimizer_step(
    model: MiniMindKDForCausalLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids, labels=input_ids)
    if output.loss is None:
        raise RuntimeError("training benchmark did not produce a loss")
    output.loss.backward()
    optimizer.step()
    return float(output.loss.detach())


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)

    config, _, _ = load_yaml_config(args.config)
    validate_args(args, config.max_seq_len)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model = MiniMindKDForCausalLM(config).to(device=device, dtype=torch.float32)
    total, active, trainable = parameter_counts(model)
    input_ids = torch.randint(
        low=3,
        high=config.vocab_size,
        size=(args.batch_size, args.sequence_length),
        device=device,
    )

    model.eval()
    with torch.inference_mode():
        for _ in range(args.forward_warmup_steps):
            model(input_ids)

        def forward_loop() -> None:
            for _ in range(args.forward_steps):
                model(input_ids)

        _, forward_seconds = timed(forward_loop, device=device)

        prompt = input_ids[:, : args.decode_prompt_length]
        generated, decode_seconds = timed(
            lambda: model.generate(
                prompt,
                max_new_tokens=args.decode_new_tokens,
                temperature=0.0,
                eos_token_id=-1,
            ),
            device=device,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    model.train()
    for _ in range(args.train_warmup_steps):
        optimizer_step(model, optimizer, input_ids)

    losses: list[float] = []

    def training_loop() -> None:
        for _ in range(args.train_steps):
            losses.append(optimizer_step(model, optimizer, input_ids))

    _, train_seconds = timed(training_loop, device=device)
    tokens_per_forward_step = args.batch_size * args.sequence_length
    decoded_tokens = generated.shape[1] - prompt.shape[1]

    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "MiniMind-KD runtime only; synthetic tokens; not a quality benchmark",
        "config": str(args.config),
        "architecture": {
            "attention_schedule": [config.attention_type(index) for index in range(config.num_hidden_layers)],
            "total_parameters": total,
            "estimated_active_parameters_per_token": active,
            "trainable_parameters": trainable,
            "routed_experts": config.num_routed_experts,
            "experts_per_token": config.num_experts_per_token,
        },
        "workload": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "forward_warmup_steps": args.forward_warmup_steps,
            "forward_steps": args.forward_steps,
            "train_warmup_steps": args.train_warmup_steps,
            "train_steps": args.train_steps,
            "decode_prompt_length": args.decode_prompt_length,
            "decode_new_tokens": decoded_tokens,
            "optimizer": "AdamW",
            "precision": "fp32",
            "seed": args.seed,
        },
        "measurements": {
            "forward_seconds": forward_seconds,
            "forward_tokens_per_second": (tokens_per_forward_step * args.forward_steps / forward_seconds),
            "training_seconds": train_seconds,
            "training_tokens_per_second": (tokens_per_forward_step * args.train_steps / train_seconds),
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": decoded_tokens / decode_seconds,
            "last_training_loss": losses[-1],
            "peak_process_rss_mib": peak_rss_mib(),
        },
        "environment": {
            "device": str(device),
            "cpu": cpu_name(),
            "logical_cpus_visible": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
