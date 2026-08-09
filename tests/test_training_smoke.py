import pytest
import torch
from conftest import tiny_config
from torch.utils.data import DataLoader

from minimind_kd.config import TrainConfig
from minimind_kd.modeling.model import MiniMindKDForCausalLM
from minimind_kd.training.optim import build_optimizer
from minimind_kd.training.supervised import train_supervised


def test_muon_optimizer_updates_a_tiny_model():
    torch.manual_seed(17)
    model = MiniMindKDForCausalLM(tiny_config())
    training = TrainConfig(
        sequence_length=8,
        batch_size=1,
        gradient_accumulation_steps=1,
        epochs=1,
        learning_rate=1e-4,
        min_learning_rate=1e-5,
        optimizer="muon",
        precision="fp32",
        device="cpu",
        num_workers=0,
    )
    optimizer = build_optimizer(model, training)
    before = model.layers[0].attention.q_proj.weight.detach().clone()
    tokens = torch.randint(3, model.config.vocab_size, (1, 6))
    model(tokens, labels=tokens).loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.layers[0].attention.q_proj.weight)
    assert torch.isfinite(model.layers[0].attention.q_proj.weight).all()


def test_one_step_training_and_safe_checkpoint_round_trip(tmp_path):
    torch.manual_seed(19)
    config = tiny_config()
    model = MiniMindKDForCausalLM(config)
    tokens = torch.randint(3, config.vocab_size, (8,))
    dataset = [
        {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "attention_mask": torch.ones_like(tokens, dtype=torch.bool),
        }
    ]
    loader = DataLoader(dataset, batch_size=1)
    training = TrainConfig(
        sequence_length=8,
        batch_size=1,
        gradient_accumulation_steps=1,
        epochs=1,
        max_steps=1,
        learning_rate=1e-4,
        min_learning_rate=1e-5,
        optimizer="adamw",
        precision="fp32",
        device="cpu",
        num_workers=0,
        log_every=1,
        save_every=100,
    )
    history = train_supervised(model, loader, training, tmp_path)
    assert len(history) == 1
    checkpoint = tmp_path / "final"
    assert (checkpoint / "model.safetensors").exists()
    restored = MiniMindKDForCausalLM.from_pretrained(checkpoint)
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(tokens.unsqueeze(0)).logits, restored(tokens.unsqueeze(0)).logits)

    posttrain_config = type(config).from_dict(config.to_dict() | {"mtp_loss_weight": 0.0})
    posttrain_model = MiniMindKDForCausalLM.from_pretrained(checkpoint, config=posttrain_config)
    assert posttrain_model.config.mtp_loss_weight == 0.0

    incompatible = type(config).from_dict(config.to_dict() | {"hidden_size": 64})
    with pytest.raises(ValueError, match="hidden_size"):
        MiniMindKDForCausalLM.from_pretrained(checkpoint, config=incompatible)
