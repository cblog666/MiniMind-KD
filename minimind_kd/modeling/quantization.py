from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fake_mxfp4(weight: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Block-scaled E2M1 fake quantization with a straight-through gradient.

    This is a portable training emulation, not a packed MXFP4 storage format or
    a vendor FP4 kernel. It is intentionally used only for routed-expert weights.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    original_shape = weight.shape
    flat = weight.float().reshape(-1)
    padding = (-flat.numel()) % block_size
    if padding:
        flat = F.pad(flat, (0, padding))
    blocks = flat.view(-1, block_size)
    max_abs = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    # MX scales are powers of two; floor prevents the largest value from
    # needlessly consuming a wider exponent bucket.
    scale = torch.pow(2.0, torch.floor(torch.log2(max_abs / 6.0)))
    normalized = blocks.abs() / scale
    levels = blocks.new_tensor(_E2M1_LEVELS)
    nearest = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    quantized = levels[nearest] * blocks.sign() * scale
    quantized = quantized.reshape(-1)[: weight.numel()].reshape(original_shape).to(weight.dtype)
    return weight + (quantized - weight).detach()


class QATLinear(nn.Linear):
    def __init__(self, *args, qat: bool = False, block_size: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.qat = qat
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = fake_mxfp4(self.weight, self.block_size) if self.qat else self.weight
        return F.linear(x, weight, self.bias)
