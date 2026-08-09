from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from minimind_kd.config import TrainConfig


@torch.no_grad()
def zeroth_power_newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the polar factor used by Muon."""

    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz input must be a matrix")
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.float().mT if transposed else matrix.float()
    x = x / x.norm().clamp_min(1e-7)
    # Quintic coefficients used by the public Muon reference implementation.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.mT
        correction = b * gram + c * (gram @ gram)
        x = a * x + correction @ x
    return x.mT if transposed else x


@torch.no_grad()
def per_head_orthogonalize(update: torch.Tensor, num_heads: int, steps: int) -> torch.Tensor:
    if num_heads <= 1:
        return zeroth_power_newton_schulz(update, steps)
    if update.shape[0] % num_heads:
        raise ValueError("per-head Muon requires rows divisible by num_heads")
    rows_per_head = update.shape[0] // num_heads
    heads = update.view(num_heads, rows_per_head, update.shape[1])
    return torch.stack([zeroth_power_newton_schulz(head, steps) for head in heads], dim=0).reshape_as(update)


class MuonAdamW(torch.optim.Optimizer):
    """Muon for matrix weights plus AdamW for vectors/embeddings/LM heads."""

    def __init__(
        self,
        muon_params: Iterable[nn.Parameter],
        adamw_params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        ns_steps: int = 5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        update_rms: float = 0.18,
    ) -> None:
        groups = [
            {
                "params": list(muon_params),
                "algorithm": "muon",
                "lr": lr,
                "weight_decay": weight_decay,
            },
            {
                "params": list(adamw_params),
                "algorithm": "adamw",
                "lr": lr,
                "weight_decay": weight_decay,
            },
        ]
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "ns_steps": ns_steps,
            "betas": betas,
            "eps": eps,
            "update_rms": update_rms,
        }
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_muon(self, group: dict) -> None:
        momentum = self.defaults["momentum"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.float()
            if gradient.ndim != 2:
                raise RuntimeError("Muon parameter must be two-dimensional")
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(gradient)
            buffer = state["momentum_buffer"]
            buffer.mul_(momentum).add_(gradient, alpha=1.0 - momentum)
            nesterov = gradient.mul(1.0 - momentum).add(buffer, alpha=momentum)
            heads = int(getattr(parameter, "_muon_num_heads", 1))
            update = per_head_orthogonalize(nesterov, heads, self.defaults["ns_steps"])
            target_rms = self.defaults["update_rms"]
            update = update * (target_rms / update.square().mean().sqrt().clamp_min(1e-7))
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])

    def _step_adamw(self, group: dict) -> None:
        beta1, beta2 = self.defaults["betas"]
        eps = self.defaults["eps"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.float()
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(gradient)
                state["exp_avg_sq"] = torch.zeros_like(gradient)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            correction1 = 1.0 - beta1 ** state["step"]
            correction2 = 1.0 - beta2 ** state["step"]
            denominator = exp_avg_sq.sqrt() / math.sqrt(correction2)
            denominator.add_(eps)
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.addcdiv_(
                exp_avg.to(parameter.dtype), denominator.to(parameter.dtype), value=-group["lr"] / correction1
            )


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
        )
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        use_adamw = (
            parameter.ndim != 2
            or name.startswith("token_embedding")
            or name.startswith("lm_head")
            or name.startswith("mtp_head")
        )
        (adamw_params if use_adamw else muon_params).append(parameter)
    return MuonAdamW(
        muon_params,
        adamw_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.muon_momentum,
        ns_steps=config.muon_ns_steps,
    )
