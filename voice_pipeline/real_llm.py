from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from .types import TurnRequest


def _emit_complete_sentences(buffer: str) -> tuple[list[str], str]:
    sentences: list[str] = []
    start = 0

    for index, char in enumerate(buffer):
        if char not in ".!?\n":
            continue

        candidate = buffer[start : index + 1].strip()
        if candidate:
            sentences.append(candidate)
        start = index + 1

    return sentences, buffer[start:]


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: str,
        api_key: str = "EMPTY",
        temperature: float = 0.2,
        max_tokens: int = 120,
    ) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def stream_sentences(
        self,
        transcript: str,
        request: TurnRequest,
    ) -> AsyncIterator[str]:
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
            ready, buffer = _emit_complete_sentences(buffer)
            for sentence in ready:
                yield sentence

        tail = buffer.strip()
        if tail:
            yield tail if tail.endswith((".", "!", "?")) else f"{tail}."
