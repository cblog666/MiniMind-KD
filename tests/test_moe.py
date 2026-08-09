import torch

from minimind_kd.modeling.moe import StableLatentMoE
from minimind_kd.modeling.quantization import fake_mxfp4


def make_moe(**overrides):
    values = dict(
        hidden_size=24,
        latent_size=12,
        intermediate_size=20,
        shared_intermediate_size=32,
        num_experts=4,
        top_k=1,
        num_shared_experts=1,
    )
    values.update(overrides)
    return StableLatentMoE(**values)


def test_stable_latent_moe_forward_and_quantile_update():
    torch.manual_seed(11)
    moe = make_moe().train()
    inputs = torch.randn(3, 5, 24, requires_grad=True)
    before = moe.router_bias.clone()
    output, metrics = moe(inputs)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert metrics.counts.sum() == inputs.shape[0] * inputs.shape[1]
    assert not torch.equal(before, moe.router_bias)
    output.square().mean().backward()
    assert inputs.grad is not None


def test_fake_mxfp4_uses_straight_through_gradient():
    values = torch.linspace(-8, 8, 64, requires_grad=True)
    quantized = fake_mxfp4(values, block_size=16)
    assert torch.isfinite(quantized).all()
    assert torch.unique(quantized.detach()).numel() < values.numel()
    quantized.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
