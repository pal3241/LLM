from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from minillm import MiniLLMForCausalLM, ModelConfig
from minillm.chat import format_chat
from minillm.checkpoint import load_checkpoint
from minillm.generation import generate_ids
from minillm.tokenizer import MiniTokenizer

STATE: dict = {"started": time.time()}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=1000)
    max_tokens: int = Field(256, ge=1, le=2048)


@asynccontextmanager
async def lifespan(_: FastAPI):
    config_path = os.environ.get("MINILLM_CONFIG")
    checkpoint_path = os.environ.get("MINILLM_CHECKPOINT")
    tokenizer_path = os.environ.get("MINILLM_TOKENIZER")
    if config_path and checkpoint_path and tokenizer_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = MiniTokenizer(tokenizer_path)
        model = MiniLLMForCausalLM(ModelConfig.from_json(config_path)).to(device)
        load_checkpoint(checkpoint_path, model, map_location=device)
        STATE.update(model=model, tokenizer=tokenizer, device=device)
    yield
    STATE.clear()


app = FastAPI(title="MiniLLM API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if "model" in STATE else "not_ready", "device": str(STATE.get("device", "none")), "uptime_seconds": round(time.time() - STATE.get("started", time.time()), 2)}


@app.post("/v1/chat/completions")
def chat(request: ChatRequest) -> dict:
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Model belum dimuat; atur MINILLM_CONFIG, MINILLM_CHECKPOINT, dan MINILLM_TOKENIZER")
    model, tokenizer, device = STATE["model"], STATE["tokenizer"], STATE["device"]
    prompt = format_chat([message.model_dump() for message in request.messages])
    encoded = tokenizer.encode(prompt)[-model.config.max_seq_len :]
    ids = torch.tensor([encoded], dtype=torch.long, device=device)
    output = list(generate_ids(model, ids, max_new_tokens=request.max_tokens, eos_token_id=tokenizer.token_to_id("<|eos|>"), temperature=request.temperature, top_p=request.top_p, top_k=request.top_k))
    text = tokenizer.decode(output).replace("<|eos|>", "").strip()
    return {"object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
