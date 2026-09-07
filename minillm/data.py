from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import Dataset


def clean_text(text: str, preserve_code: bool = True) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(char for char in text if char in "\n\t" or unicodedata.category(char)[0] != "C")
    if preserve_code:
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(lines).strip()
    return re.sub(r"\s+", " ", text).strip()


def document_id(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def iter_text_files(paths: Iterable[str | Path]) -> Iterator[dict]:
    for value in paths:
        path = Path(value)
        files = path.rglob("*") if path.is_dir() else [path]
        for file_path in files:
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() == ".txt":
                text = clean_text(file_path.read_text(encoding="utf-8", errors="replace"))
                if text:
                    yield {"id": document_id(text), "source": str(file_path), "text": text}
            elif file_path.suffix.lower() in {".jsonl", ".json"}:
                lines = file_path.read_text(encoding="utf-8").splitlines()
                records = [json.loads(line) for line in lines if line.strip()] if file_path.suffix == ".jsonl" else json.loads("\n".join(lines))
                if isinstance(records, dict):
                    records = [records]
                for record in records:
                    text = clean_text(str(record.get("text", "")))
                    if text:
                        yield {**record, "id": record.get("id", document_id(text)), "text": text}


def write_clean_jsonl(inputs: Iterable[str | Path], output: str | Path) -> tuple[int, int]:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = duplicates = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in iter_text_files(inputs):
            if record["id"] in seen:
                duplicates += 1
                continue
            seen.add(record["id"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written, duplicates


class PackedTokenDataset(Dataset):
    def __init__(self, token_ids: list[int], sequence_length: int) -> None:
        if len(token_ids) < sequence_length + 1:
            raise ValueError("Token tidak cukup untuk membuat satu sequence")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.sequence_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.sequence_length
        chunk = self.tokens[start : start + self.sequence_length + 1]
        return chunk[:-1], chunk[1:]


def deterministic_split(records: list[dict], seed: int = 42, train_ratio: float = 0.995) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, min(len(shuffled), round(len(shuffled) * train_ratio)))
    return shuffled[:cut], shuffled[cut:]
