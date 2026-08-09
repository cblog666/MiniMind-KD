import copy

import torch
from conftest import tiny_config

from minimind_kd.modeling.model import MiniMindKDForCausalLM
from minimind_kd.training.grpo import (
    GRPOConfig,
    GRPOTrainer,
    group_relative_advantages,
    grpo_loss,
)
from minimind_kd.training.opd import (
    MultiTeacherOnPolicyDistiller,
    OPDConfig,
    TeacherSpec,
    full_vocabulary_reverse_kl,
)


class TinyGenerationTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [1, 3]

    def decode(self, token_ids, skip_special_tokens=False):
        return "42"


def test_group_advantages_are_centered_per_prompt():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 8.0, 12.0])
    advantages = group_relative_advantages(rewards, group_size=3).view(2, 3)
    torch.testing.assert_close(advantages.mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0)


def test_grpo_loss_is_differentiable():
    policy = torch.randn(4, 5, requires_grad=True)
    old = policy.detach().clone()
    reference = old - 0.1
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(4, 5, dtype=torch.bool)
    loss, metrics = grpo_loss(policy, old, reference, advantages, mask)
    assert torch.isfinite(loss)
    assert metrics["reference_kl"] >= 0
    loss.backward()
    assert policy.grad is not None


def test_full_vocabulary_reverse_kl_is_exact_for_identity():
    logits = torch.randn(2, 3, 11)
    identical = full_vocabulary_reverse_kl(logits, logits)
    assert abs(float(identical)) < 1e-6
    shifted = full_vocabulary_reverse_kl(logits, torch.randn_like(logits))
    assert shifted >= 0


def test_grpo_trainer_runs_one_on_policy_step():
    torch.manual_seed(23)
    policy = MiniMindKDForCausalLM(tiny_config(mtp_loss_weight=0.0))
    reference = copy.deepcopy(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    trainer = GRPOTrainer(
        policy,
        reference,
        TinyGenerationTokenizer(),
        optimizer,
        lambda record, completion: float(completion == str(record["answer"])),
        GRPOConfig(num_generations=2, max_new_tokens=2, top_k=0),
        torch.device("cpu"),
    )
    metrics = trainer.step([{"prompt": "What is six times seven?", "answer": 42, "effort": "none"}])
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["mean_reward"] == 1.0
    assert len(metrics["completions"]) == 2


def test_multi_teacher_opd_runs_one_full_vocabulary_step():
    torch.manual_seed(29)
    student = MiniMindKDForCausalLM(tiny_config(mtp_loss_weight=0.0))
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        teacher.lm_head.weight.add_(0.01 * torch.randn_like(teacher.lm_head.weight))
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    distiller = MultiTeacherOnPolicyDistiller(
        student,
        [TeacherSpec("math", teacher, {"math"})],
        TinyGenerationTokenizer(),
        optimizer,
        OPDConfig(max_new_tokens=2, top_k=0),
        torch.device("cpu"),
    )
    metrics = distiller.step([{"prompt": "Add two numbers", "domain": "math", "effort": "none"}])
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["mean_completion_tokens"] > 0
