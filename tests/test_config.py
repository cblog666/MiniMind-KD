import pytest
from conftest import tiny_config

from minimind_kd.config import ModelConfig, load_yaml_config


def test_hybrid_schedule_and_final_global_layer():
    config = tiny_config(num_hidden_layers=7)
    assert [config.attention_type(index) for index in range(7)] == [
        "kda",
        "kda",
        "kda",
        "mla",
        "kda",
        "kda",
        "mla",
    ]


def test_invalid_routing_configuration_is_rejected():
    with pytest.raises(ValueError, match="smaller"):
        tiny_config(num_routed_experts=2, num_experts_per_token=2)


def test_padding_and_end_of_sequence_tokens_must_be_distinct():
    with pytest.raises(ValueError, match="distinct"):
        tiny_config(pad_token_id=2, eos_token_id=2)


def test_unknown_configuration_field_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        ModelConfig.from_dict({"not_a_real_field": 1})


def test_k3_shape_reference_matches_public_release():
    config, _, _ = load_yaml_config("configs/k3_shape_reference.yaml")
    schedule = [config.attention_type(index) for index in range(config.num_hidden_layers)]
    assert schedule.count("kda") == 69
    assert schedule.count("mla") == 24
    assert config.hidden_size == 7168
    assert config.num_attention_heads == 96
    assert config.head_dim == 128
    assert config.mla_qk_nope_head_dim == 128
    assert config.mla_qk_direct_head_dim == 64
    assert config.mla_v_head_dim == 128
    assert config.vocab_size == 163840
    assert config.mla_q_lora_rank == 1536
    assert config.dense_intermediate_size == 33792
