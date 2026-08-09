from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


def _known_fields(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {', '.join(unknown)}")
    return values


@dataclass(slots=True)
class ModelConfig:
    """Configuration for the text backbone.

    Defaults deliberately describe a laptop-scale model, not Kimi K3 itself.
    The architecture ratios mirror the public K3 report while every width is
    independently scalable.
    """

    vocab_size: int = 6400
    hidden_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    # MiniMind and K3 allow the projected head width to differ from
    # hidden_size / num_attention_heads. Zero selects that usual default.
    head_dim: int = 0
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0

    # Kimi Delta Attention and 3:1 KDA/Gated-MLA hybrid schedule.
    kda_conv_kernel_size: int = 4
    kda_decay_rank: int = 64
    kda_log_decay_min: float = -5.0
    hybrid_kda_layers: int = 3
    hybrid_mla_layers: int = 1

    # NoPE Multi-head Latent Attention.
    mla_q_lora_rank: int = 0
    mla_kv_lora_rank: int = 64
    mla_qk_nope_head_dim: int = 0
    # K3's released config retains an uncompressed 64-wide Q/K channel named
    # qk_rope_head_dim, but applies NoPE. We call it "direct" here to avoid
    # implying that rotary embeddings are used.
    mla_qk_direct_head_dim: int = 0
    mla_v_head_dim: int = 0

    # Block Attention Residuals.
    attnres_layers_per_block: int = 4

    # Stable LatentMoE. The first layer is dense, as in Kimi K3.
    num_dense_layers: int = 1
    num_routed_experts: int = 8
    num_experts_per_token: int = 2
    num_shared_experts: int = 1
    moe_latent_size: int = 128
    moe_intermediate_size: int = 256
    dense_intermediate_size: int = 0
    shared_intermediate_size: int = 512
    quantile_balance: bool = True
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0

    # Pre-training objective and deployment-aware training.
    mtp_loss_weight: float = 0.3
    tie_word_embeddings: bool = True
    expert_qat: bool = False
    mxfp4_block_size: int = 32

    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.head_dim == 0:
            if self.hidden_size % self.num_attention_heads:
                raise ValueError(
                    "hidden_size must be divisible by num_attention_heads when head_dim is not set"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.dense_intermediate_size == 0:
            self.dense_intermediate_size = self.shared_intermediate_size
        if self.mla_qk_nope_head_dim == 0:
            self.mla_qk_nope_head_dim = self.head_dim
        if self.mla_v_head_dim == 0:
            self.mla_v_head_dim = self.head_dim
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len,
            "mla_qk_nope_head_dim": self.mla_qk_nope_head_dim,
            "mla_v_head_dim": self.mla_v_head_dim,
            "attnres_layers_per_block": self.attnres_layers_per_block,
            "num_routed_experts": self.num_routed_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "moe_latent_size": self.moe_latent_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "dense_intermediate_size": self.dense_intermediate_size,
            "shared_intermediate_size": self.shared_intermediate_size,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {', '.join(invalid)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.mla_qk_direct_head_dim < 0:
            raise ValueError("mla_qk_direct_head_dim must be non-negative")
        if self.num_experts_per_token >= self.num_routed_experts:
            raise ValueError("num_experts_per_token must be smaller than num_routed_experts for QB")
        if not 0 <= self.num_dense_layers <= self.num_hidden_layers:
            raise ValueError("num_dense_layers must be within the decoder depth")
        if self.kda_log_decay_min >= 0:
            raise ValueError("kda_log_decay_min must be negative")
        if self.hybrid_kda_layers < 0 or self.hybrid_mla_layers <= 0:
            raise ValueError("hybrid attention schedule is invalid")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.pad_token_id == self.eos_token_id:
            raise ValueError("pad_token_id and eos_token_id must be distinct")

    @property
    def hybrid_period(self) -> int:
        return self.hybrid_kda_layers + self.hybrid_mla_layers

    def attention_type(self, layer_index: int) -> str:
        if not 0 <= layer_index < self.num_hidden_layers:
            raise IndexError(layer_index)
        # K3 adds a final global layer even when the repeating pattern ends in KDA.
        if layer_index == self.num_hidden_layers - 1:
            return "mla"
        position = layer_index % self.hybrid_period
        return "kda" if position < self.hybrid_kda_layers else "mla"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        return cls(**_known_fields(cls, dict(values)))

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    sequence_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    max_steps: int = 0
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_ratio: float = 0.01
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    optimizer: str = "muon"
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    precision: str = "bf16"
    device: str = "auto"
    num_workers: int = 0
    log_every: int = 10
    save_every: int = 500

    def __post_init__(self) -> None:
        if self.sequence_length <= 1 or self.batch_size <= 0:
            raise ValueError("sequence_length and batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.optimizer not in {"muon", "adamw"}:
            raise ValueError("optimizer must be 'muon' or 'adamw'")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainConfig:
        return cls(**_known_fields(cls, dict(values)))


def load_yaml_config(path: str | Path) -> tuple[ModelConfig, TrainConfig, dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by packaging users
        raise RuntimeError("Install PyYAML to load YAML configuration files") from exc
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    model = ModelConfig.from_dict(raw.get("model", {}))
    training = TrainConfig.from_dict(raw.get("training", {}))
    extra = {key: value for key, value in raw.items() if key not in {"model", "training"}}
    return model, training, extra
