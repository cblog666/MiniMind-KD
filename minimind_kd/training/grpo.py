from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from minimind_kd.modeling.model import MiniMindKDForCausalLM
from minimind_kd.protocol import effort_prompt

from .rewards import RewardFunction


class GenerationTokenizer(Protocol):
    pad_token_id: int
    eos_token_id: int

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = ...) -> str: ...


@dataclass(slots=True)
class GRPOConfig:
    num_generations: int = 4
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    clip_epsilon: float = 0.2
    kl_beta: float = 0.04
    advantage_epsilon: float = 1e-4
    max_gradient_norm: float = 1.0
    length_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.num_generations < 2:
            raise ValueError("GRPO requires at least two generations per prompt")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


def group_relative_advantages(
    rewards: torch.Tensor,
    group_size: int,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    if rewards.numel() % group_size:
        raise ValueError("reward count must be divisible by group_size")
    grouped = rewards.float().view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    return (centered / (grouped.std(dim=1, keepdim=True, unbiased=False) + epsilon)).reshape(-1)


def sequence_log_probs(
    model: MiniMindKDForCausalLM,
    sequences: torch.Tensor,
    prompt_lengths: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_mask = sequences.ne(pad_token_id)
    logits = model(sequences, attention_mask=attention_mask).logits[:, :-1]
    targets = sequences[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    selected = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)
    completion_mask = positions >= (prompt_lengths - 1).unsqueeze(1)
    completion_mask &= targets.ne(pad_token_id)
    return selected, completion_mask


def grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expected = policy_log_probs.shape
    if old_log_probs.shape != expected or reference_log_probs.shape != expected:
        raise ValueError("all log-probability tensors must have the same shape")
    if advantages.shape != (expected[0],):
        raise ValueError("advantages must have one value per completion")
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio.clamp(-20, 20))
    advantage = advantages.unsqueeze(-1)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    policy_objective = torch.minimum(unclipped, clipped)
    log_ref_over_policy = reference_log_probs - policy_log_probs
    kl = torch.exp(log_ref_over_policy.clamp(-20, 20)) - log_ref_over_policy - 1.0
    per_token = -policy_objective + kl_beta * kl
    mask = completion_mask.to(per_token.dtype)
    denominator = mask.sum().clamp_min(1.0)
    loss = (per_token * mask).sum() / denominator
    metrics = {
        "policy_objective": (policy_objective * mask).sum() / denominator,
        "reference_kl": (kl * mask).sum() / denominator,
        "clip_fraction": (((ratio - 1.0).abs() > clip_epsilon).to(mask.dtype) * mask).sum() / denominator,
    }
    return loss, metrics


class GRPOTrainer:
    """Small-scale on-policy specialist trainer with verifiable/local rewards."""

    def __init__(
        self,
        policy: MiniMindKDForCausalLM,
        reference: MiniMindKDForCausalLM,
        tokenizer: GenerationTokenizer,
        optimizer: torch.optim.Optimizer,
        reward: RewardFunction,
        config: GRPOConfig,
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.reference = reference.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.reward = reward
        self.config = config
        self.device = device

    @torch.no_grad()
    def _rollout(
        self, records: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        self.policy.eval()
        trajectories: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
        rewards: list[float] = []
        completions: list[str] = []
        for record in records:
            rendered = effort_prompt(record["prompt"], record.get("effort", "high"))
            prompt = torch.tensor(
                self.tokenizer.encode(rendered, add_special_tokens=False),
                dtype=torch.long,
                device=self.device,
            )
            repeated = prompt.unsqueeze(0).repeat(self.config.num_generations, 1)
            generated = self.policy.generate(
                repeated,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            for sequence in generated:
                completion_ids = sequence[prompt.numel() :].tolist()
                if self.tokenizer.eos_token_id in completion_ids:
                    completion_ids = completion_ids[: completion_ids.index(self.tokenizer.eos_token_id) + 1]
                while completion_ids and completion_ids[-1] == self.tokenizer.pad_token_id:
                    completion_ids.pop()
                completion = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
                score = self.reward(record, completion)
                target_tokens = int(record.get("target_tokens", self.config.max_new_tokens))
                excess = max(0, len(completion_ids) - target_tokens)
                score -= self.config.length_penalty * excess / max(1, target_tokens)
                trajectories.append(sequence.detach().cpu())
                prompt_lengths.append(prompt.numel())
                rewards.append(score)
                completions.append(completion)
        padded = pad_sequence(
            trajectories,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(self.device)
        return (
            padded,
            torch.tensor(prompt_lengths, device=self.device),
            torch.tensor(rewards, device=self.device),
            completions,
        )

    def step(self, records: list[dict[str, Any]]) -> dict[str, float | list[str]]:
        sequences, prompt_lengths, rewards, completions = self._rollout(records)
        with torch.no_grad():
            old_log_probs, completion_mask = sequence_log_probs(
                self.policy, sequences, prompt_lengths, self.tokenizer.pad_token_id
            )
            reference_log_probs, _ = sequence_log_probs(
                self.reference, sequences, prompt_lengths, self.tokenizer.pad_token_id
            )
        advantages = group_relative_advantages(
            rewards, self.config.num_generations, self.config.advantage_epsilon
        )
        self.policy.train()
        policy_log_probs, _ = sequence_log_probs(
            self.policy, sequences, prompt_lengths, self.tokenizer.pad_token_id
        )
        loss, metrics = grpo_loss(
            policy_log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
            completion_mask,
            clip_epsilon=self.config.clip_epsilon,
            kl_beta=self.config.kl_beta,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_gradient_norm
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "mean_reward": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "reference_kl": float(metrics["reference_kl"].detach()),
            "clip_fraction": float(metrics["clip_fraction"].detach()),
            "gradient_norm": float(gradient_norm),
            "completions": completions,
        }
