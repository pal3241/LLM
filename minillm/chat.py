from __future__ import annotations


def format_chat(messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    allowed = {"system", "user", "assistant"}
    parts = ["<|bos|>"]
    for message in messages:
        role = message.get("role")
        content = message.get("content", "").strip()
        if role not in allowed:
            raise ValueError(f"Role tidak valid: {role}")
        parts.append(f"<|{role}|>\n{content}\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)
