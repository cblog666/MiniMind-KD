import torch
from conftest import tiny_config

from minimind_kd.modeling.model import MiniMindKDForCausalLM


def test_model_forward_loss_backward_and_architecture():
    torch.manual_seed(13)
    config = tiny_config()
    model = MiniMindKDForCausalLM(config)
    torch.testing.assert_close(
        model.token_embedding.weight[config.pad_token_id],
        torch.zeros(config.hidden_size),
    )
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 8, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.mtp_loss is not None and torch.isfinite(output.mtp_loss)
    assert model.layers[-1].attention_type == "mla"
    assert len(output.router_metrics) == config.num_hidden_layers - config.num_dense_layers
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_greedy_generation_length():
    config = tiny_config()
    model = MiniMindKDForCausalLM(config).eval()
    prompt = torch.tensor([[3, 4, 5]])
    generated = model.generate(prompt, max_new_tokens=3, temperature=0.0, eos_token_id=-1)
    assert generated.shape == (1, 6)
