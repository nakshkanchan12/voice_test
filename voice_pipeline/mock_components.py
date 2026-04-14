from __future__ import annotations

import asyncio
from time import perf_counter
from typing import AsyncIterator

from .asr_catalog import ASR_MODEL_PROFILES, DEFAULT_ASR_MODEL
from .text_chunking import finalize_tail, split_ready_chunks
from .types import AudioChunk, SpeechSegment, TurnRequest


class MockVAD:
    """Simple energy-flag VAD that emits a segment after sustained silence."""

    def __init__(self, min_speech_ms: int = 200, silence_ms: int = 300) -> None:
        self.min_speech_ms = min_speech_ms
        self.silence_ms = silence_ms

    async def stream_segments(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[SpeechSegment]:
        buffer: list[AudioChunk] = []
        speech_ms = 0
        silence_ms = 0
        last_speech_chunk_at: float | None = None

        async for chunk in audio_stream:
            if chunk.is_speech:
                buffer.append(chunk)
                speech_ms += chunk.duration_ms
                silence_ms = 0
                last_speech_chunk_at = perf_counter()
                continue

            silence_ms += chunk.duration_ms
            if buffer and silence_ms >= self.silence_ms:
                if speech_ms >= self.min_speech_ms:
                    yield SpeechSegment(
                        chunks=tuple(buffer),
                        sample_rate_hz=buffer[0].sample_rate_hz,
                        ended_by_silence=True,
                        last_speech_chunk_at=last_speech_chunk_at,
                        endpoint_at=perf_counter(),
                    )
                buffer = []
                speech_ms = 0
                silence_ms = 0
                last_speech_chunk_at = None

        if buffer and speech_ms >= self.min_speech_ms:
            yield SpeechSegment(
                chunks=tuple(buffer),
                sample_rate_hz=buffer[0].sample_rate_hz,
                ended_by_silence=False,
                last_speech_chunk_at=last_speech_chunk_at,
            )


class MockASR:
    def __init__(
        self,
        base_latency_ms: float | None = None,
        per_second_ms: float | None = None,
        model_name: str = DEFAULT_ASR_MODEL,
    ) -> None:
        profile = ASR_MODEL_PROFILES.get(model_name, ASR_MODEL_PROFILES[DEFAULT_ASR_MODEL])
        self.model_name = profile.name
        self.base_latency_ms = (
            float(base_latency_ms) if base_latency_ms is not None else profile.latency_base_ms
        )
        self.per_second_ms = (
            float(per_second_ms)
            if per_second_ms is not None
            else profile.latency_per_second_ms
        )
        self.transcript_bank = [
            "I can pay the EMI on Friday",
            "my last four digits are three seven two one",
            "the due date is tenth August",
            "I can do a partial payment today",
        ]

    async def transcribe(self, segment: SpeechSegment, request: TurnRequest) -> str:
        duration_s = max(segment.duration_ms, 20) / 1000.0
        delay_s = (self.base_latency_ms + (self.per_second_ms * duration_s)) / 1000.0
        await asyncio.sleep(delay_s)

        index = (segment.duration_ms + len(request.call_id) + len(request.turn_id)) % len(
            self.transcript_bank
        )
        return self.transcript_bank[index]


class MockLLM:
    def __init__(
        self,
        ttft_ms: float = 85.0,
        token_delay_ms: float = 8.0,
        aggressive_min_tokens: int = 5,
    ) -> None:
        self.ttft_ms = ttft_ms
        self.token_delay_ms = token_delay_ms
        self.aggressive_min_tokens = max(2, aggressive_min_tokens)

    async def stream_sentences(
        self,
        transcript: str,
        request: TurnRequest,
    ) -> AsyncIterator[str]:
        response = (
            "Thanks for confirming. "
            f"I heard: {transcript}. "
            "Please confirm whether you can make at least half of the amount today?"
        )

        await asyncio.sleep(self.ttft_ms / 1000.0)

        buffer = ""
        for token in response.split():
            await asyncio.sleep(self.token_delay_ms / 1000.0)

            if buffer:
                buffer = f"{buffer} {token}"
            else:
                buffer = token

            ready, buffer = split_ready_chunks(
                buffer,
                mode=request.llm_stream_mode,
                aggressive_min_tokens=self.aggressive_min_tokens,
            )
            for chunk in ready:
                yield chunk

        tail = finalize_tail(buffer)
        if tail is not None:
            yield tail


class MockTTS:
    def __init__(
        self,
        ttfb_ms: float = 95.0,
        chunk_delay_ms: float = 20.0,
        bytes_per_chunk: int = 640,
    ) -> None:
        self.ttfb_ms = ttfb_ms
        self.chunk_delay_ms = chunk_delay_ms
        self.bytes_per_chunk = bytes_per_chunk

    async def stream_audio(self, sentence: str, request: TurnRequest) -> AsyncIterator[bytes]:
        await asyncio.sleep(self.ttfb_ms / 1000.0)
        chunk_count = max(1, min(12, len(sentence) // 14))

        for _ in range(chunk_count):
            await asyncio.sleep(self.chunk_delay_ms / 1000.0)
            yield b"\x00" * self.bytes_per_chunk

    async def stop_streaming(self, turn_id: str) -> None:
        # Mock provider has no persistent stream handle to cancel.
        return
