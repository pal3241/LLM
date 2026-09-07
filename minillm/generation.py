from __future__ import annotations

from collections.abc import Iterator

import torch


def sample_next_token(logits: torch.Tensor, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probs, 1)
        return sorted_indices.gather(-1, sampled)
    return torch.multinomial(probabilities, 1)


@torch.inference_mode()
def generate_ids(model, input_ids: torch.Tensor, max_new_tokens: int = 64, eos_token_id: int | None = None, **sampling) -> Iterator[int]:
    model.eval()
    ids = input_ids
    for _ in range(max_new_tokens):
        context = ids[:, -model.config.max_seq_len :]
        token = sample_next_token(model(context).logits[:, -1], **sampling)
        token_id = int(token.item())
        yield token_id
        ids = torch.cat((ids, token), dim=1)
        if eos_token_id is not None and token_id == eos_token_id:
            break
