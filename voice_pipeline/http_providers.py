from __future__ import annotations

import io
import wave
from typing import AsyncIterator

import httpx

from .types import SpeechSegment, TurnRequest


def _segment_to_wav_bytes(segment: SpeechSegment) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(segment.sample_rate_hz)
        wav.writeframes(b"".join(chunk.pcm for chunk in segment.chunks))
    return buffer.getvalue()


class HTTPASRProvider:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def transcribe(self, segment: SpeechSegment, request: TurnRequest) -> str:
        wav_bytes = _segment_to_wav_bytes(segment)
        files = {"audio_file": ("segment.wav", wav_bytes, "audio/wav")}
        data = {"call_id": request.call_id, "turn_id": request.turn_id}

        response = await self._client.post(f"{self.base_url}/transcribe", files=files, data=data)
        response.raise_for_status()

        payload = response.json()
        return str(payload.get("text", "")).strip()


class HTTPTTSProvider:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def stream_audio(self, sentence: str, request: TurnRequest) -> AsyncIterator[bytes]:
        payload = {
            "text": sentence,
            "call_id": request.call_id,
            "turn_id": request.turn_id,
        }

        async with self._client.stream(
            "POST",
            f"{self.base_url}/synthesize",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
