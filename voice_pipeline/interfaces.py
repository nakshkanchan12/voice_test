from __future__ import annotations

from typing import AsyncIterator, Protocol

from .types import AudioChunk, SpeechSegment, TurnRequest


class VADProvider(Protocol):
    async def stream_segments(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[SpeechSegment]:
        ...


class ASRProvider(Protocol):
    async def transcribe(self, segment: SpeechSegment, request: TurnRequest) -> str:
        ...


class LLMProvider(Protocol):
    async def stream_sentences(
        self,
        transcript: str,
        request: TurnRequest,
    ) -> AsyncIterator[str]:
        ...


class TTSProvider(Protocol):
    async def stream_audio(
        self,
        sentence: str,
        request: TurnRequest,
    ) -> AsyncIterator[bytes]:
        ...
