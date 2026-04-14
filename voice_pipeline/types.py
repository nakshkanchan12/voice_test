from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass(frozen=True)
class AudioChunk:
    """Small audio unit (usually 20ms at 16kHz mono)."""

    pcm: bytes
    duration_ms: int = 20
    sample_rate_hz: int = 16000
    is_speech: bool = True


@dataclass(frozen=True)
class SpeechSegment:
    """Utterance fragment grouped by VAD."""

    chunks: tuple[AudioChunk, ...]
    sample_rate_hz: int = 16000
    ended_by_silence: bool = True

    @property
    def duration_ms(self) -> int:
        return sum(chunk.duration_ms for chunk in self.chunks)


@dataclass(frozen=True)
class TurnRequest:
    call_id: str
    turn_id: str
    llm_stream_mode: str = "aggressive"
    enable_barge_in: bool = True
    barge_in_min_speech_ms: int = 120
    metadata: dict[str, Any] = field(default_factory=dict)


async def iter_chunks(chunks: list[AudioChunk]) -> AsyncIterator[AudioChunk]:
    for chunk in chunks:
        yield chunk
