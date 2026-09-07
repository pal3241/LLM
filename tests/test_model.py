import torch

from minillm import MiniLLMForCausalLM, ModelConfig
from minillm.model import RMSNorm


def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=128, hidden_size=32, num_layers=2, num_attention_heads=4, num_kv_heads=2, intermediate_size=64, max_seq_len=32)


def test_rmsnorm_shape_and_gradient() -> None:
    x = torch.randn(2, 8, 32, requires_grad=True)
    result = RMSNorm(32)(x)
    assert result.shape == x.shape
    result.sum().backward()
    assert x.grad is not None


def test_model_forward_backward() -> None:
    model = MiniLLMForCausalLM(tiny_config())
    inputs = torch.randint(0, 128, (2, 16))
    output = model(inputs, inputs)
    assert output.logits.shape == (2, 16, 128)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_causal_attention() -> None:
    torch.manual_seed(7)
    model = MiniLLMForCausalLM(tiny_config()).eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 9, 8]])
    with torch.no_grad():
        logits_a = model(a).logits
        logits_b = model(b).logits
    torch.testing.assert_close(logits_a[:, :2], logits_b[:, :2])
