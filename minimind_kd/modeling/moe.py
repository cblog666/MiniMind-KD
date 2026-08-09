from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .activations import SiTUAndMul
from .normalization import RMSNorm
from .quantization import QATLinear


@dataclass
class RouterMetrics:
    counts: torch.Tensor
    load_cv: torch.Tensor
    max_to_mean: torch.Tensor

    def detached(self) -> dict[str, float | list[int]]:
        return {
            "counts": self.counts.detach().cpu().tolist(),
            "load_cv": float(self.load_cv.detach()),
            "max_to_mean": float(self.max_to_mean.detach()),
        }


class ExpertMLP(nn.Module):
    def __init__(
        self,
        size: int,
        intermediate_size: int,
        beta: float,
        linear_beta: float,
        *,
        qat: bool = False,
        block_size: int = 32,
    ) -> None:
        super().__init__()

        def linear(input_size: int, output_size: int) -> QATLinear:
            return QATLinear(
                input_size,
                output_size,
                bias=False,
                qat=qat,
                block_size=block_size,
            )

        self.gate = linear(size, intermediate_size)
        self.up = linear(size, intermediate_size)
        self.down = linear(intermediate_size, size)
        self.activation = SiTUAndMul(beta, linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.activation(self.gate(x), self.up(x)))


class DenseSiTUMLP(ExpertMLP):
    pass


class StableLatentMoE(nn.Module):
    """Scaled Stable LatentMoE with exact minibatch Quantile Balancing.

    K3 estimates the quantile from globally reduced histograms. At MiniMind
    scale we can compute it exactly on the local minibatch. The updated bias is
    installed only after routing, so it affects the next optimizer step.
    """

    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        top_k: int,
        num_shared_experts: int,
        beta: float = 4.0,
        linear_beta: float = 25.0,
        norm_eps: float = 1e-6,
        quantile_balance: bool = True,
        expert_qat: bool = False,
        mxfp4_block_size: int = 32,
    ) -> None:
        super().__init__()
        if not 0 < top_k < num_experts:
            raise ValueError("top_k must be between 1 and num_experts - 1")
        self.num_experts = num_experts
        self.top_k = top_k
        self.quantile_balance = quantile_balance
        self.latent_down = nn.Linear(hidden_size, latent_size, bias=False)
        self.latent_norm = RMSNorm(latent_size, norm_eps)
        self.latent_up = nn.Linear(latent_size, hidden_size, bias=False)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(num_experts), persistent=True)

        expert_kwargs = {
            "beta": beta,
            "linear_beta": linear_beta,
            "qat": expert_qat,
            "block_size": mxfp4_block_size,
        }
        self.routed_experts = nn.ModuleList(
            [ExpertMLP(latent_size, intermediate_size, **expert_kwargs) for _ in range(num_experts)]
        )
        # Shared experts remain high precision in K3 deployment-aware training.
        self.shared_experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size,
                    shared_intermediate_size,
                    beta,
                    linear_beta,
                    qat=False,
                )
                for _ in range(num_shared_experts)
            ]
        )

    @torch.no_grad()
    def _next_quantile_bias(self, scores: torch.Tensor, selection_scores: torch.Tensor) -> None:
        if scores.shape[0] == 0:
            return
        cutoff = torch.topk(selection_scores, self.top_k + 1, dim=-1).values[:, self.top_k]
        margins = scores.float() - cutoff.float().unsqueeze(-1)
        quantile_level = 1.0 - self.top_k / self.num_experts
        new_bias = -torch.quantile(margins, quantile_level, dim=0)
        new_bias -= new_bias.mean()
        self.router_bias.copy_(new_bias.to(self.router_bias.dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RouterMetrics]:
        original_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, original_shape[-1])
        if token_mask is None:
            valid = torch.ones(flat_hidden.shape[0], device=flat_hidden.device, dtype=torch.bool)
        else:
            valid = token_mask.reshape(-1).to(torch.bool)

        shared_output = sum(
            (expert(flat_hidden) for expert in self.shared_experts), start=torch.zeros_like(flat_hidden)
        )
        routed_output = flat_hidden.new_zeros(flat_hidden.shape[0], self.latent_down.out_features)
        valid_hidden = flat_hidden[valid]
        scores = torch.sigmoid(self.router(valid_hidden).float())
        selection_scores = scores + self.router_bias.float()
        top = torch.topk(selection_scores, self.top_k, dim=-1)
        selected = top.indices
        weights = torch.gather(scores, 1, selected)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        valid_latent = self.latent_down(valid_hidden)
        mixed = valid_latent.new_zeros(valid_latent.shape)

        for expert_index, expert in enumerate(self.routed_experts):
            token_indices, route_slots = torch.where(selected == expert_index)
            if token_indices.numel() == 0:
                continue
            expert_values = expert(valid_latent[token_indices])
            weighted = expert_values * weights[token_indices, route_slots].to(expert_values.dtype).unsqueeze(
                -1
            )
            mixed.index_add_(0, token_indices, weighted)

        routed_output[valid] = mixed
        routed_output = self.latent_up(self.latent_norm(routed_output))
        output = (shared_output + routed_output).view(original_shape)
        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1).to(output.dtype)

        counts = torch.bincount(selected.reshape(-1), minlength=self.num_experts)
        counts_float = counts.float()
        mean = counts_float.mean().clamp_min(1.0)
        metrics = RouterMetrics(
            counts=counts,
            load_cv=counts_float.std(unbiased=False) / mean,
            max_to_mean=counts_float.max() / mean,
        )
        if self.training and self.quantile_balance:
            self._next_quantile_bias(scores, selection_scores)
        return output, metrics
