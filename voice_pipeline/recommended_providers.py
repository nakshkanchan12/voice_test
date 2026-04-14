from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI

from .http_providers import _segment_to_wav_bytes
from .text_chunking import finalize_tail, normalize_stream_mode, split_ready_chunks
from .types import SpeechSegment, TurnRequest


class NemotronHTTPASRProvider:
    """ASR provider for Nemotron-compatible HTTP transcription endpoints."""

    def __init__(
        self,
        base_url: str,
        language: str = "en",
        chunk_ms: int = 160,
        transcribe_path: str = "/transcribe",
        audio_field: str = "audio_file",
        model: str = "",
        api_key: str = "",
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.chunk_ms = chunk_ms
        normalized_path = transcribe_path.strip()
        self.transcribe_path = f"/{normalized_path.lstrip('/')}" if normalized_path else "/transcribe"
        self.audio_field = audio_field.strip() or "audio_file"
        self.model = model.strip()
        self.api_key = api_key.strip()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def transcribe(self, segment: SpeechSegment, request: TurnRequest) -> str:
        wav_bytes = _segment_to_wav_bytes(segment)
        files = {self.audio_field: ("segment.wav", wav_bytes, "audio/wav")}

        openai_style = self.audio_field == "file" or self.transcribe_path.endswith("/v1/audio/transcriptions")
        if openai_style:
            data: dict[str, str] = {
                "language": self.language,
            }
            if self.model:
                data["model"] = self.model
        else:
            data = {
                "call_id": request.call_id,
                "turn_id": request.turn_id,
                "language": self.language,
                "chunk_ms": str(self.chunk_ms),
            }
            if self.model:
                data["model"] = self.model

        headers: dict[str, str] = {}
        if self.api_key and self.api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = await self._client.post(
            f"{self.base_url}{self.transcribe_path}",
            files=files,
            data=data,
            headers=headers or None,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = response.json()
            if isinstance(payload, dict):
                text = str(payload.get("text") or payload.get("transcript") or payload.get("output_text") or "")
                if text.strip():
                    return text.strip()

                result = payload.get("result")
                if isinstance(result, dict):
                    nested = str(result.get("text") or result.get("transcript") or "")
                    if nested.strip():
                        return nested.strip()

                alternatives = payload.get("alternatives")
                if isinstance(alternatives, list) and alternatives:
                    first = alternatives[0]
                    if isinstance(first, dict):
                        alt_text = str(first.get("transcript") or first.get("text") or "")
                        if alt_text.strip():
                            return alt_text.strip()

        return response.text.strip()


class SGLangOpenAILLMProvider:
    """LLM provider for SGLang/vLLM OpenAI-compatible streaming endpoints."""

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


class Qwen3TTSProvider:
    """TTS provider supporting Qwen3-TTS style HTTP endpoints."""

    def __init__(
        self,
        base_url: str,
        mode: str = "http_synthesize",
        model: str = "piper/en_US-lessac-medium",
        voice: str = "Chelsie",
        split_granularity: str = "sentence",
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.model = model
        self.voice = voice
        self.split_granularity = split_granularity
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._stop_events: dict[str, asyncio.Event] = {}
        self._active_responses: dict[str, httpx.Response] = {}

    @staticmethod
    def _stream_key(request: TurnRequest) -> str:
        return f"{request.call_id}:{request.turn_id}"

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def stop_streaming(self, turn_id: str) -> None:
        for key, stop_event in list(self._stop_events.items()):
            if not key.endswith(f":{turn_id}"):
                continue

            stop_event.set()
            response = self._active_responses.get(key)
            if response is None:
                continue

            try:
                await response.aclose()
            except Exception:
                continue

    async def stream_audio(self, sentence: str, request: TurnRequest) -> AsyncIterator[bytes]:
        stream_key = self._stream_key(request)
        stop_event = asyncio.Event()
        self._stop_events[stream_key] = stop_event

        if self.mode == "openai_audio_speech":
            try:
                async for chunk in self._stream_openai_audio_speech(sentence, request=request):
                    if stop_event.is_set():
                        break
                    yield chunk
            finally:
                self._stop_events.pop(stream_key, None)
                self._active_responses.pop(stream_key, None)
            return

        payload: dict[str, Any] = {
            "text": sentence,
            "call_id": request.call_id,
            "turn_id": request.turn_id,
            "split_granularity": self.split_granularity,
        }

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/synthesize",
                json=payload,
            ) as response:
                self._active_responses[stream_key] = response
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if stop_event.is_set():
                        break
                    if chunk:
                        yield chunk
        finally:
            self._stop_events.pop(stream_key, None)
            self._active_responses.pop(stream_key, None)

    async def _stream_openai_audio_speech(self, sentence: str, request: TurnRequest) -> AsyncIterator[bytes]:
        stream_key = self._stream_key(request)
        payload = {
            "model": self.model,
            "input": sentence,
            "voice": self.voice,
            "response_format": "pcm",
            "stream": True,
            "stream_audio": True,
            "split_granularity": self.split_granularity,
        }

        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/audio/speech",
            json=payload,
        ) as response:
            self._active_responses[stream_key] = response
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
