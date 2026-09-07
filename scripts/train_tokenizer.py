#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from minillm.tokenizer import SPECIAL_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latih byte-level BPE tokenizer MiniLLM")
    parser.add_argument("--input", nargs="+", required=True, help="File atau folder teks/JSONL")
    parser.add_argument("--output", default="tokenizer/artifacts/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    return parser.parse_args()


def discover_files(values: list[str]) -> list[str]:
    files: list[str] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            files.extend(str(item) for item in path.rglob("*") if item.suffix.lower() in {".txt", ".jsonl"})
        elif path.is_file():
            files.append(str(path))
    if not files:
        raise SystemExit("Tidak ada file input yang ditemukan")
    return files


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train(discover_files(args.input), trainer)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    print(f"Tokenizer disimpan: {output} ({tokenizer.get_vocab_size()} token)")


if __name__ == "__main__":
    main()
