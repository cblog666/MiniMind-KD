from __future__ import annotations

import contextlib
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minimind_kd.config import TrainConfig
from minimind_kd.modeling.model import MiniMindKDForCausalLM

from .optim import build_optimizer


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def cosine_learning_rate(step: int, total_steps: int, config: TrainConfig) -> float:
    warmup = max(1, int(total_steps * config.warmup_ratio))
    if step < warmup:
        return config.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


def train_supervised(
    model: MiniMindKDForCausalLM,
    loader: DataLoader,
    config: TrainConfig,
    output_dir: str | Path,
) -> list[dict[str, float]]:
    """Shared loop for pre-training and domain SFT."""

    seed_everything(config.seed)
    device = resolve_device(config.device)
    model.to(device).train()
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and config.precision == "fp16"))
    steps_per_epoch = math.ceil(len(loader) / config.gradient_accumulation_steps)
    total_steps = config.max_steps or config.epochs * steps_per_epoch
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    started = time.monotonic()

    for _epoch in range(config.epochs):
        for micro_step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, config.precision):
                result = model(**batch)
                if result.loss is None:
                    raise RuntimeError("Supervised batch did not produce a loss")
                loss = result.loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            should_update = micro_step % config.gradient_accumulation_steps == 0 or micro_step == len(loader)
            if not should_update:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            learning_rate = cosine_learning_rate(update_step, total_steps, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            record = {
                "step": float(update_step),
                "loss": float(result.loss.detach()),
                "causal_loss": float(result.causal_loss.detach()) if result.causal_loss is not None else 0.0,
                "mtp_loss": float(result.mtp_loss.detach()) if result.mtp_loss is not None else 0.0,
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(record)
            if update_step % config.log_every == 0:
                print(
                    f"step={update_step} loss={record['loss']:.4f} "
                    f"lr={learning_rate:.2e} grad={record['gradient_norm']:.3f}",
                    flush=True,
                )
            if update_step % config.save_every == 0:
                model.save_pretrained(output / f"step-{update_step}")
            if update_step >= total_steps:
                break
        if update_step >= total_steps:
            break
    model.save_pretrained(output / "final")
    return history
