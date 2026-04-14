from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .webrtc_common import create_offer_handler

MODEL = None
TOKENIZER = None
MODEL_ID = ""
DEVICE = "cpu"
WEBRTC_PEERS: set[Any] = set()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 0.2
    max_tokens: int = Field(default=120, ge=1, le=1024)


def _resolve_device() -> str:
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _build_prompt(messages: list[ChatMessage]) -> str:
    global TOKENIZER
    assert TOKENIZER is not None

    plain = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
    try:
        rendered = TOKENIZER.apply_chat_template(
            [{"role": msg.role, "content": msg.content} for msg in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, str) and rendered.strip():
            return rendered
    except Exception:
        pass

    return plain + "\nassistant:"


def _generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    global MODEL, TOKENIZER, DEVICE
    assert MODEL is not None and TOKENIZER is not None

    inputs = TOKENIZER(prompt, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    do_sample = temperature > 0.0
    temperature = max(0.01, float(temperature))

    outputs = MODEL.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=TOKENIZER.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    return TOKENIZER.decode(generated_ids, skip_special_tokens=True).strip()


async def _stream_generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> AsyncIterator[str]:
    global MODEL, TOKENIZER, DEVICE
    assert MODEL is not None and TOKENIZER is not None

    inputs = TOKENIZER(prompt, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    do_sample = temperature > 0.0
    temperature = max(0.01, float(temperature))

    streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)

    kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "pad_token_id": TOKENIZER.eos_token_id,
        "streamer": streamer,
    }

    thread = threading.Thread(target=MODEL.generate, kwargs=kwargs, daemon=True)
    thread.start()

    loop = asyncio.get_running_loop()

    while True:
        token = await loop.run_in_executor(None, next, streamer, None)
        if token is None:
            break
        if token:
            yield token


def _chat_response_payload(model: str, content: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _messages_from_rpc_payload(payload: dict[str, object]) -> list[ChatMessage]:
    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        messages: list[ChatMessage] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role and content:
                messages.append(ChatMessage(role=role, content=content))
        if messages:
            return messages

    system_prompt = str(payload.get("system_prompt", "You are concise.")).strip()
    transcript = str(payload.get("transcript", "")).strip()
    if not transcript:
        raise ValueError("Missing transcript payload")

    return [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=transcript),
    ]


async def _webrtc_rpc_handler(
    op: str,
    payload: dict[str, object],
    stream: bool,
) -> dict[str, object] | AsyncIterator[dict[str, str]]:
    if op != "chat":
        raise ValueError(f"Unsupported operation: {op}")

    model_name = str(payload.get("model", MODEL_ID)).strip() or MODEL_ID
    max_tokens = int(payload.get("max_tokens", 120))
    temperature = float(payload.get("temperature", 0.2))
    messages = _messages_from_rpc_payload(payload)
    prompt = _build_prompt(messages)

    if not stream:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            _generate_text,
            prompt,
            max_tokens,
            temperature,
        )
        return {
            "model": model_name,
            "text": text,
        }

    async def _stream_tokens() -> AsyncIterator[dict[str, str]]:
        async for token in _stream_generate_text(
            prompt,
            max_tokens,
            temperature,
        ):
            if token:
                yield {"model": model_name, "token": token}

    return _stream_tokens()


def _stream_chunk_payload(model: str, content: str, first: bool = False, done: bool = False) -> dict[str, Any]:
    delta: dict[str, str] = {}
    if first:
        delta["role"] = "assistant"
    if content:
        delta["content"] = content

    finish_reason = "stop" if done else None

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER, MODEL_ID, DEVICE

    MODEL_ID = os.getenv("LLM_MODEL_ID", "HuggingFaceTB/SmolLM2-360M-Instruct")
    DEVICE = _resolve_device()

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)

    if DEVICE == "cuda":
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
    else:
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
        )

    # Warm-up for stable first-token latency.
    _ = _generate_text("assistant: hello", max_new_tokens=8, temperature=0.2)

    yield

    MODEL = None
    TOKENIZER = None


app = FastAPI(title="Local OpenAI-Compatible LLM Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok" if MODEL is not None else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
    }


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest):
    model_name = payload.model or MODEL_ID
    prompt = _build_prompt(payload.messages)

    if not payload.stream:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            _generate_text,
            prompt,
            payload.max_tokens,
            payload.temperature,
        )
        return JSONResponse(_chat_response_payload(model=model_name, content=text))

    async def event_stream() -> AsyncIterator[str]:
        first = True
        async for token in _stream_generate_text(
            prompt,
            payload.max_tokens,
            payload.temperature,
        ):
            chunk = _stream_chunk_payload(model=model_name, content=token, first=first, done=False)
            first = False
            yield f"data: {json.dumps(chunk, ensure_ascii=True)}\n\n"

        tail = _stream_chunk_payload(model=model_name, content="", first=False, done=True)
        yield f"data: {json.dumps(tail, ensure_ascii=True)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


app.post("/webrtc/offer")(create_offer_handler(WEBRTC_PEERS, _webrtc_rpc_handler))
