#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minillm import MiniLLMForCausalLM, ModelConfig
from minillm.checkpoint import load_checkpoint, save_checkpoint
from minillm.data import PackedTokenDataset
from minillm.tokenizer import MiniTokenizer


def cosine_lambda(step: int, warmup: int, maximum: int, minimum_ratio: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
    return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretraining causal language model")
    parser.add_argument("--model", default="configs/model_tiny.json")
    parser.add_argument("--train", default="configs/pretrain_debug.json")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True, help="File teks UTF-8")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--resume")
    args = parser.parse_args()

    model_config = ModelConfig.from_json(args.model)
    train_config = json.loads(Path(args.train).read_text(encoding="utf-8"))
    random.seed(train_config["seed"])
    torch.manual_seed(train_config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MiniTokenizer(args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise SystemExit(f"vocab tokenizer ({tokenizer.vocab_size}) != config ({model_config.vocab_size})")
    text = Path(args.data).read_text(encoding="utf-8")
    dataset = PackedTokenDataset(tokenizer.encode(text, add_bos=True, add_eos=True), train_config["sequence_length"])
    loader = DataLoader(dataset, batch_size=train_config["micro_batch_size"], shuffle=True, drop_last=True)
    model = MiniLLMForCausalLM(model_config).to(device)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or "norm" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train_config["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=train_config["learning_rate"], betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lambda(step, train_config["warmup_steps"], train_config["max_steps"], train_config["min_learning_rate"] / train_config["learning_rate"]),
    )
    start_step = tokens_seen = 0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_step, tokens_seen = state["step"], state["tokens_seen"]

    iterator = iter(loader)
    accumulation = train_config["gradient_accumulation_steps"]
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, train_config["max_steps"] + 1):
        total_loss = 0.0
        for _ in range(accumulation):
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, labels = next(iterator)
            inputs, labels = inputs.to(device), labels.to(device)
            loss = model(inputs, labels).loss / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss non-finite pada step {step}")
            loss.backward()
            total_loss += float(loss)
            tokens_seen += inputs.numel()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["gradient_clip"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step % train_config["log_every"] == 0:
            print(json.dumps({"step": step, "loss": total_loss, "lr": scheduler.get_last_lr()[0], "tokens_seen": tokens_seen, "grad_norm": float(grad_norm)}))
        if step % train_config["checkpoint_every"] == 0 or step == train_config["max_steps"]:
            save_checkpoint(args.output, model, optimizer, scheduler, step, tokens_seen)


if __name__ == "__main__":
    main()
