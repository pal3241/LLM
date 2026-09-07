from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 704
    max_seq_len: int = 256
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size harus habis dibagi num_attention_heads")
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads harus habis dibagi num_kv_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("head_dim harus genap untuk RoPE")
        for name in ("vocab_size", "hidden_size", "num_layers", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} harus lebih besar dari nol")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
