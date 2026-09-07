from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch


def save_checkpoint(output_dir: str | Path, model, optimizer, scheduler, step: int, tokens_seen: int) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / f"step_{step:08d}.pt"
    temp_path = final_path.with_suffix(".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": model.config.__dict__,
    }
    torch.save(payload, temp_path)
    os.replace(temp_path, final_path)
    (output / "latest.json").write_text(json.dumps({"checkpoint": final_path.name}), encoding="utf-8")
    return final_path


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
