from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from minimind_kd.config import ModelConfig

from .attn_res import DepthAttention
from .kda import KimiDeltaAttention
from .mla import GatedMLA
from .moe import DenseSiTUMLP, RouterMetrics, StableLatentMoE
from .normalization import RMSNorm

_SAFE_CHECKPOINT_CONFIG_OVERRIDES = {
    # NoPE/KDA has no learned positional table, so curriculum stages may raise
    # the runtime limit without changing checkpoint tensor shapes.
    "max_seq_len",
    # Training/inference switches with no additional learned parameters.
    "mtp_loss_weight",
    "expert_qat",
    "mxfp4_block_size",
    "quantile_balance",
}


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    causal_loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    router_metrics: list[RouterMetrics] | None = None


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attention_type = config.attention_type(layer_index)
        self.pre_attention_residual = DepthAttention(config.hidden_size, config.rms_norm_eps)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.attention_type == "kda":
            self.attention = KimiDeltaAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.head_dim,
                config.kda_decay_rank,
                config.kda_conv_kernel_size,
                config.kda_log_decay_min,
                config.rms_norm_eps,
                config.dropout,
            )
        else:
            self.attention = GatedMLA(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.mla_qk_nope_head_dim,
                config.mla_kv_lora_rank,
                config.mla_q_lora_rank,
                config.mla_qk_direct_head_dim,
                config.mla_v_head_dim,
                config.rms_norm_eps,
                config.dropout,
            )
        self.pre_moe_residual = DepthAttention(config.hidden_size, config.rms_norm_eps)
        self.moe_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.is_dense = layer_index < config.num_dense_layers
        if self.is_dense:
            self.moe = DenseSiTUMLP(
                config.hidden_size,
                config.dense_intermediate_size,
                config.situ_beta,
                config.situ_linear_beta,
            )
        else:
            self.moe = StableLatentMoE(
                config.hidden_size,
                config.moe_latent_size,
                config.moe_intermediate_size,
                config.shared_intermediate_size,
                config.num_routed_experts,
                config.num_experts_per_token,
                config.num_shared_experts,
                config.situ_beta,
                config.situ_linear_beta,
                config.rms_norm_eps,
                config.quantile_balance,
                config.expert_qat,
                config.mxfp4_block_size,
            )

    def forward(
        self,
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, RouterMetrics | None]:
        hidden = self.pre_attention_residual(completed_blocks, partial_block)
        attention_output = self.attention(self.attention_norm(hidden), attention_mask)
        partial_block = attention_output if partial_block is None else partial_block + attention_output

        hidden = self.pre_moe_residual(completed_blocks, partial_block)
        normalized = self.moe_norm(hidden)
        if self.is_dense:
            moe_output = self.moe(normalized)
            metrics = None
        else:
            moe_output, metrics = self.moe(normalized, attention_mask)
        partial_block = partial_block + moe_output
        return partial_block, metrics


class MiniMindKDForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.final_residual = DepthAttention(config.hidden_size, config.rms_norm_eps)
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # A compact t+2 auxiliary head keeps MiniMind's training readable while
        # preserving K3's one-layer multi-token-prediction objective.
        self.mtp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mtp_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
            self.mtp_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        # Module-wide initialization runs after Embedding's constructor, so
        # restore the invariant expected for padding_idx.
        with torch.no_grad():
            self.token_embedding.weight[config.pad_token_id].zero_()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds configured {self.config.max_seq_len}"
            )
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        attention_mask = attention_mask.to(torch.bool)
        embedding = self.token_embedding(input_ids)
        completed_blocks: list[torch.Tensor] = [embedding]
        partial_block: torch.Tensor | None = None
        router_metrics: list[RouterMetrics] = []

        for index, layer in enumerate(self.layers):
            if index > 0 and index % self.config.attnres_layers_per_block == 0:
                if partial_block is None:
                    raise RuntimeError("AttnRes block ended without a partial representation")
                completed_blocks.append(partial_block)
                partial_block = None
            partial_block, metrics = layer(completed_blocks, partial_block, attention_mask)
            if metrics is not None:
                router_metrics.append(metrics)

        hidden = self.final_residual(completed_blocks, partial_block)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        causal_loss = mtp_loss = total_loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            causal_targets = labels[:, 1:].reshape(-1)
            if bool(causal_targets.ne(-100).any()):
                causal_loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, self.config.vocab_size),
                    causal_targets,
                    ignore_index=-100,
                )
            else:
                causal_loss = logits.sum() * 0.0
            if input_ids.shape[1] > 2 and self.config.mtp_loss_weight > 0:
                mtp_logits = self.mtp_head(self.mtp_norm(hidden[:, :-2]))
                mtp_targets = labels[:, 2:].reshape(-1)
                if bool(mtp_targets.ne(-100).any()):
                    mtp_loss = F.cross_entropy(
                        mtp_logits.reshape(-1, self.config.vocab_size),
                        mtp_targets,
                        ignore_index=-100,
                    )
                else:
                    mtp_loss = mtp_logits.sum() * 0.0
            else:
                mtp_loss = causal_loss.new_zeros(())
            total_loss = causal_loss + self.config.mtp_loss_weight * mtp_loss
        return CausalLMOutput(logits, total_loss, causal_loss, mtp_loss, router_metrics)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        generated = input_ids
        finished = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            window = generated[:, -self.config.max_seq_len :]
            logits = self(window).logits[:, -1]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if 0 < top_k < logits.shape[-1]:
                    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                if 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
                    logits = torch.full_like(logits, -torch.inf).scatter(1, sorted_indices, sorted_logits)
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, self.config.pad_token_id),
                next_token,
            )
            generated = torch.cat((generated, next_token), dim=1)
            finished |= next_token.squeeze(-1).eq(eos)
            if bool(finished.all()):
                break
        return generated

    def save_pretrained(self, directory: str | Path) -> None:
        from safetensors.torch import save_file

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.config.save_json(destination / "config.json")
        # save_file rejects shared storage from tied embeddings. Cloning
        # produces an unambiguous state dict; the constructor restores ties.
        state = {
            name: tensor.detach().contiguous().cpu().clone() for name, tensor in self.state_dict().items()
        }
        save_file(state, str(destination / "model.safetensors"))

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        config: ModelConfig | None = None,
        map_location: str | torch.device = "cpu",
    ) -> MiniMindKDForCausalLM:
        source = Path(directory)
        checkpoint_config = ModelConfig.from_dict(
            json.loads((source / "config.json").read_text(encoding="utf-8"))
        )
        if config is None:
            effective_config = checkpoint_config
        else:
            stored = checkpoint_config.to_dict()
            requested = config.to_dict()
            incompatible = sorted(
                key
                for key in stored
                if stored[key] != requested[key] and key not in _SAFE_CHECKPOINT_CONFIG_OVERRIDES
            )
            if incompatible:
                raise ValueError("Checkpoint architecture differs in: " + ", ".join(incompatible))
            effective_config = config
        model = cls(effective_config)
        safe_path = source / "model.safetensors"
        if not safe_path.exists():
            raise FileNotFoundError(f"Safe checkpoint not found: {safe_path}")
        from safetensors.torch import load_file

        state = load_file(str(safe_path), device=str(map_location))
        model.load_state_dict(state)
        return model
