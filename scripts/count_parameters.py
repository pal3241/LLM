#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = SimpleNamespace(**json.load(handle))
    hidden = config.hidden_size
    head_dim = hidden // config.num_attention_heads
    kv_width = config.num_kv_heads * head_dim
    embedding = config.vocab_size * hidden
    attention_per_layer = hidden * hidden * 2 + hidden * kv_width * 2
    mlp_per_layer = hidden * config.intermediate_size * 3
    norms = config.num_layers * hidden * 2 + hidden
    lm_head = 0 if config.tie_word_embeddings else config.vocab_size * hidden
    groups = {
        "embedding": embedding,
        "attention": attention_per_layer * config.num_layers,
        "mlp": mlp_per_layer * config.num_layers,
        "norms": norms,
        "lm_head": lm_head,
    }
    for name, count in groups.items():
        print(f"{name:12}: {count:,}")
    total = sum(groups.values())
    print(f"{'total':12}: {total:,} ({total / 1e9:.3f}B)")


if __name__ == "__main__":
    main()
