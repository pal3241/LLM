from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    "<|system|>", "<|user|>", "<|assistant|>",
]


class MiniTokenizer:
    def __init__(self, path: str | Path) -> None:
        self.backend = Tokenizer.from_file(str(path))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.backend.encode(text).ids
        if add_bos:
            ids.insert(0, self.token_to_id("<|bos|>"))
        if add_eos:
            ids.append(self.token_to_id("<|eos|>"))
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.backend.decode(ids, skip_special_tokens=False)

    def token_to_id(self, token: str) -> int:
        value = self.backend.token_to_id(token)
        if value is None:
            raise KeyError(f"Token tidak ditemukan: {token}")
        return value

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()
