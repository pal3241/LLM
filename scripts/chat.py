#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from minillm import MiniLLMForCausalLM, ModelConfig
from minillm.chat import format_chat
from minillm.checkpoint import load_checkpoint
from minillm.generation import generate_ids
from minillm.tokenizer import MiniTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MiniTokenizer(args.tokenizer)
    model = MiniLLMForCausalLM(ModelConfig.from_json(args.model_config)).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    history: list[dict[str, str]] = []
    print("MiniLLM Chat — /clear untuk reset, /exit untuk keluar")
    while True:
        prompt = input("You > ").strip()
        if prompt == "/exit":
            break
        if prompt == "/clear":
            history.clear()
            continue
        history.append({"role": "user", "content": prompt})
        encoded = tokenizer.encode(format_chat(history))[-model.config.max_seq_len :]
        ids = torch.tensor([encoded], dtype=torch.long, device=device)
        generated: list[int] = []
        print("AI  > ", end="", flush=True)
        for token_id in generate_ids(model, ids, eos_token_id=tokenizer.token_to_id("<|eos|>"), max_new_tokens=256):
            generated.append(token_id)
            text = tokenizer.decode(generated)
            print("\rAI  > " + text, end="", flush=True)
        answer = tokenizer.decode(generated).replace("<|eos|>", "").strip()
        print()
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
