from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from .text_chunking import finalize_tail, normalize_stream_mode, split_ready_chunks
from .types import TurnRequest


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: str,
        api_key: str = "EMPTY",
        temperature: float = 0.2,
        max_tokens: int = 120,
        stream_mode: str = "sentence",
        aggressive_min_tokens: int = 5,
    ) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream_mode = normalize_stream_mode(stream_mode)
        self.aggressive_min_tokens = max(2, aggressive_min_tokens)

    async def stream_sentences(
        self,
        transcript: str,
        request: TurnRequest,
    ) -> AsyncIterator[str]:
        mode = normalize_stream_mode(request.llm_stream_mode or self.stream_mode)
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": transcript},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        buffer = ""
        async for chunk in stream:
            token = ""
            if chunk.choices and chunk.choices[0].delta is not None:
                token = chunk.choices[0].delta.content or ""
            if not token:
                continue

            buffer += token
            ready, buffer = split_ready_chunks(
                buffer,
                mode=mode,
                aggressive_min_tokens=self.aggressive_min_tokens,
            )
            for chunk in ready:
                yield chunk

        tail = finalize_tail(buffer)
        if tail is not None:
            yield tail
