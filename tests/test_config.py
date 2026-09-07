import pytest

from minillm import ModelConfig


def test_invalid_gqa_ratio() -> None:
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=96, num_attention_heads=6, num_kv_heads=4)
