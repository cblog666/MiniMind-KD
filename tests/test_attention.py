import torch

from minimind_kd.modeling.attn_res import DepthAttention
from minimind_kd.modeling.kda import KimiDeltaAttention
from minimind_kd.modeling.mla import GatedMLA


def test_kda_is_causal():
    torch.manual_seed(7)
    layer = KimiDeltaAttention(32, 4, 8, 8, 3, dropout=0.0).eval()
    values = torch.randn(2, 7, 32)
    full = layer(values)
    prefix = layer(values[:, :4])
    torch.testing.assert_close(full[:, :4], prefix, atol=1e-5, rtol=1e-5)


def test_gated_mla_is_causal_and_nope():
    torch.manual_seed(9)
    layer = GatedMLA(32, 4, 2, 8, kv_lora_rank=8, dropout=0.0).eval()
    values = torch.randn(2, 6, 32)
    full = layer(values)
    prefix = layer(values[:, :3])
    torch.testing.assert_close(full[:, :3], prefix, atol=1e-5, rtol=1e-5)
    assert not any("rope" in name.casefold() for name, _ in layer.named_parameters())


def test_attention_head_dimension_can_differ_from_model_width():
    layer = GatedMLA(
        30,
        4,
        2,
        8,
        kv_lora_rank=8,
        qk_direct_head_dim=4,
        v_head_dim=6,
        dropout=0.0,
    ).eval()
    values = torch.randn(2, 5, 30)
    assert layer(values).shape == values.shape


def test_depth_attention_weights_are_normalized():
    torch.manual_seed(3)
    residual = DepthAttention(16)
    sources = [torch.randn(2, 5, 16), torch.randn(2, 5, 16)]
    partial = torch.randn(2, 5, 16)
    output, weights = residual(sources, partial, return_weights=True)
    assert output.shape == partial.shape
    torch.testing.assert_close(weights.sum(dim=0), torch.ones_like(weights[0]))
