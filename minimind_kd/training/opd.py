from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from minimind_kd.modeling.model import MiniMindKDForCausalLM
from minimind_kd.protocol import effort_prompt


def reverse_kl_per_token(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    student_probs = student_log_probs.exp()
    return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)


def full_vocabulary_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact full-vocabulary KL(student || teacher), as used by V4 OPD."""

    per_token = reverse_kl_per_token(student_logits, teacher_logits)
    if token_mask is None:
        return per_token.mean()
    mask = token_mask.to(per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


class OPDTokenizer(Protocol):
    pad_token_id: int
    eos_token_id: int

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


@dataclass
class TeacherSpec:
    name: str
    model: MiniMindKDForCausalLM
    domains: set[str]
    weight: float = 1.0

    def applies_to(self, domain: str) -> bool:
        return "*" in self.domains or domain in self.domains


@dataclass(slots=True)
class OPDConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    max_gradient_norm: float = 1.0


class MultiTeacherOnPolicyDistiller:
    """Merge domain experts on trajectories sampled from the student."""

    def __init__(
        self,
        student: MiniMindKDForCausalLM,
        teachers: list[TeacherSpec],
        tokenizer: OPDTokenizer,
        optimizer: torch.optim.Optimizer,
        config: OPDConfig,
        device: torch.device,
    ) -> None:
        if not teachers:
            raise ValueError("OPD requires at least one teacher")
        self.student = student
        self.teachers = teachers
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config
        self.device = device
        for teacher in teachers:
            if teacher.model.config.vocab_size != student.config.vocab_size:
                raise ValueError(f"Teacher {teacher.name} has a different vocabulary")
            teacher.model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def _student_trajectories(self, records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        self.student.eval()
        sequences: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
        for record in records:
            rendered = effort_prompt(record["prompt"], record.get("effort", "high"))
            prompt = torch.tensor(
                self.tokenizer.encode(rendered, add_special_tokens=False),
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0)
            generated = self.student.generate(
                prompt,
                self.config.max_new_tokens,
                self.config.temperature,
                self.config.top_k,
                self.config.top_p,
                self.tokenizer.eos_token_id,
            )[0]
            sequences.append(generated.cpu())
            prompt_lengths.append(prompt.shape[1])
        return (
            pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(
                self.device
            ),
            torch.tensor(prompt_lengths, device=self.device),
        )

    def step(self, records: list[dict[str, Any]]) -> dict[str, float]:
        sequences, prompt_lengths = self._student_trajectories(records)
        attention_mask = sequences.ne(self.tokenizer.pad_token_id)
        targets = sequences[:, 1:]
        positions = torch.arange(targets.shape[1], device=self.device).unsqueeze(0)
        completion_mask = positions >= (prompt_lengths - 1).unsqueeze(1)
        completion_mask &= targets.ne(self.tokenizer.pad_token_id)

        self.student.train()
        student_logits = self.student(sequences, attention_mask=attention_mask).logits[:, :-1]
        sample_losses: list[torch.Tensor] = []
        sample_weights: list[torch.Tensor] = []
        for teacher in self.teachers:
            with torch.no_grad():
                teacher_logits = teacher.model(sequences, attention_mask=attention_mask).logits[:, :-1]
            token_kl = reverse_kl_per_token(student_logits, teacher_logits)
            per_sample = (token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)
            weights = torch.tensor(
                [
                    teacher.weight if teacher.applies_to(str(record.get("domain", "general"))) else 0.0
                    for record in records
                ],
                device=self.device,
            )
            sample_losses.append(per_sample)
            sample_weights.append(weights)
        loss_matrix = torch.stack(sample_losses, dim=1)
        weight_matrix = torch.stack(sample_weights, dim=1)
        totals = weight_matrix.sum(dim=1, keepdim=True)
        if bool((totals == 0).any()):
            missing = [
                str(records[index].get("domain", "general"))
                for index in torch.where(totals.squeeze(1) == 0)[0].tolist()
            ]
            raise ValueError(f"No OPD teacher covers domains: {', '.join(missing)}")
        normalized_weights = weight_matrix / totals
        loss = (loss_matrix * normalized_weights).sum(dim=1).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.student.parameters(), self.config.max_gradient_norm
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
            "mean_completion_tokens": float(completion_mask.sum(dim=1).float().mean()),
        }
