from __future__ import annotations

import torch
from torch import nn


class SiTUAndMul(nn.Module):
    """Sigmoid Tanh Unit GLU from the Kimi K3 report.

    Both branches are smoothly capped. With beta=4 and linear_beta=25 the
    scalar product is bounded by 100 while matching SwiGLU near the origin.
    """

    def __init__(self, beta: float = 4.0, linear_beta: float = 25.0) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        dtype = gate.dtype
        gate32 = gate.float()
        up32 = up.float()
        capped_gate = self.beta * torch.tanh(gate32 / self.beta) * torch.sigmoid(gate32)
        capped_up = self.linear_beta * torch.tanh(up32 / self.linear_beta)
        return (capped_gate * capped_up).to(dtype)
