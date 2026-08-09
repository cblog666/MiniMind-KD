from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .normalization import RMSNorm


class DepthAttention(nn.Module):
    """Single-head attention over embedding/block representations."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.empty(hidden_size))
        self.key_norm = RMSNorm(hidden_size, eps)
        nn.init.normal_(self.pseudo_query, std=hidden_size**-0.5)

    def forward(
        self,
        completed_blocks: Sequence[torch.Tensor],
        partial_block: torch.Tensor | None,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        sources = list(completed_blocks)
        if partial_block is not None:
            sources.append(partial_block)
        if not sources:
            raise ValueError("DepthAttention requires at least one source")
        values = torch.stack(sources, dim=0)
        keys = self.key_norm(values)
        logits = torch.einsum("d,nbtd->nbt", self.pseudo_query, keys)
        weights = torch.softmax(logits.float(), dim=0).to(values.dtype)
        output = torch.einsum("nbt,nbtd->btd", weights, values)
        return (output, weights) if return_weights else output
