from __future__ import annotations

import asyncio
import base64
from typing import AsyncIterator

from .http_providers import _segment_to_wav_bytes
from .text_chunking import finalize_tail, normalize_stream_mode, split_ready_chunks
from .types import SpeechSegment, TurnRequest
from .webrtc_rpc import WebRTCRPCClient


class WebRTCASRProvider:
    def __init__(
        self,
        offer_url: str,
        language: str = "en",
        timeout_s: float = 120.0,
        reuse_session: bool = True,
    ) -> None:
        self.offer_url = offer_url
        self.language = language
        self.timeout_s = timeout_s
        self.reuse_session = reuse_session
        self._rpc = self._create_rpc() if self.reuse_session else None

    def _create_rpc(self) -> WebRTCRPCClient:
        return WebRTCRPCClient(
            offer_url=self.offer_url,
            channel_label="asr-rpc",
            timeout_s=self.timeout_s,
        )

    async def close(self) -> None:
        if self._rpc is not None:
            await self._rpc.close()

    async def transcribe(self, segment: SpeechSegment, request: TurnRequest) -> str:
        wav_bytes = _segment_to_wav_bytes(segment)
        payload = {
            "call_id": request.call_id,
            "turn_id": request.turn_id,
            "language": self.language,
            "audio_wav_b64": base64.b64encode(wav_bytes).decode("ascii"),
        }

        if self.reuse_session:
            assert self._rpc is not None
            result = await self._rpc.call_unary("transcribe", payload)
            return str(result.get("text", "")).strip()

        rpc = self._create_rpc()
        try:
            result = await rpc.call_unary("transcribe", payload)
        finally:
            await rpc.close()

        return str(result.get("text", "")).strip()


class WebRTCLLMProvider:
    def __init__(
        self,
        offer_url: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 120,
        stream_mode: str = "sentence",
        aggressive_min_tokens: int = 5,
        timeout_s: float = 120.0,
        reuse_session: bool = True,
    ) -> None:
        self.offer_url = offer_url
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream_mode = normalize_stream_mode(stream_mode)
        self.aggressive_min_tokens = max(2, aggressive_min_tokens)
        self.timeout_s = timeout_s
        self.reuse_session = reuse_session
        self._rpc = self._create_rpc() if self.reuse_session else None

    def _create_rpc(self) -> WebRTCRPCClient:
        return WebRTCRPCClient(
            offer_url=self.offer_url,
            channel_label="llm-rpc",
            timeout_s=self.timeout_s,
        )

    async def close(self) -> None:
        if self._rpc is not None:
            await self._rpc.close()

    async def stream_sentences(
        self,
        transcript: str,
        request: TurnRequest,
    ) -> AsyncIterator[str]:
        mode = normalize_stream_mode(request.llm_stream_mode or self.stream_mode)
        payload = {
            "call_id": request.call_id,
            "turn_id": request.turn_id,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "transcript": transcript,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream_mode": mode,
        }

        if self.reuse_session:
            assert self._rpc is not None
            stream = self._rpc.call_stream("chat", payload)
            rpc = None
        else:
            rpc = self._create_rpc()
            stream = rpc.call_stream("chat", payload)

        buffer = ""
        try:
            async for chunk in stream:
                token = str(chunk.get("token", ""))
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
        finally:
            if rpc is not None:
                await rpc.close()

        tail = finalize_tail(buffer)
        if tail is not None:
            yield tail


class WebRTCTTSProvider:
    def __init__(
        self,
        offer_url: str,
        model: str = "",
        voice: str = "",
        split_granularity: str = "sentence",
        timeout_s: float = 120.0,
        reuse_session: bool = True,
    ) -> None:
        self.offer_url = offer_url
        self.model = model
        self.voice = voice
        self.split_granularity = split_granularity
        self.timeout_s = timeout_s
        self.reuse_session = reuse_session
        self._rpc = self._create_rpc() if self.reuse_session else None
        self._stop_events: dict[str, asyncio.Event] = {}

    @staticmethod
    def _stream_key(request: TurnRequest) -> str:
        return f"{request.call_id}:{request.turn_id}"

    def _create_rpc(self) -> WebRTCRPCClient:
        return WebRTCRPCClient(
            offer_url=self.offer_url,
            channel_label="tts-rpc",
            timeout_s=self.timeout_s,
        )

    async def close(self) -> None:
        if self._rpc is not None:
            await self._rpc.close()

    async def stop_streaming(self, turn_id: str) -> None:
        for key, stop_event in list(self._stop_events.items()):
            if key.endswith(f":{turn_id}"):
                stop_event.set()

    async def stream_audio(self, sentence: str, request: TurnRequest) -> AsyncIterator[bytes]:
        stream_key = self._stream_key(request)
        stop_event = asyncio.Event()
        self._stop_events[stream_key] = stop_event

        payload = {
            "call_id": request.call_id,
            "turn_id": request.turn_id,
            "text": sentence,
            "model": self.model,
            "voice": self.voice,
            "split_granularity": self.split_granularity,
        }

        if self.reuse_session:
            assert self._rpc is not None
            stream = self._rpc.call_stream("synthesize", payload)
            rpc = None
        else:
            rpc = self._create_rpc()
            stream = rpc.call_stream("synthesize", payload)

        try:
            async for chunk in stream:
                if stop_event.is_set():
                    break

                encoded = str(chunk.get("audio_pcm_b64", ""))
                if not encoded:
                    continue

                try:
                    pcm = base64.b64decode(encoded.encode("ascii"), validate=True)
                except Exception:
                    continue

                if pcm:
                    yield pcm
        finally:
            self._stop_events.pop(stream_key, None)
            if rpc is not None:
                await rpc.close()
