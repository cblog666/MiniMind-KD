from __future__ import annotations

from minimind_kd.config import ModelConfig


def tiny_config(**overrides) -> ModelConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_seq_len": 32,
        "kda_conv_kernel_size": 3,
        "kda_decay_rank": 8,
        "mla_kv_lora_rank": 8,
        "attnres_layers_per_block": 2,
        "num_dense_layers": 1,
        "num_routed_experts": 4,
        "num_experts_per_token": 1,
        "num_shared_experts": 1,
        "moe_latent_size": 16,
        "moe_intermediate_size": 24,
        "shared_intermediate_size": 48,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)
