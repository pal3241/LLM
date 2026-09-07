import torch

from minillm.generation import sample_next_token


def test_greedy_sampling() -> None:
    logits = torch.tensor([[0.1, 0.5, 0.2]])
    assert sample_next_token(logits, temperature=0).item() == 1
