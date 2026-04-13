from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


def _elapsed_ms(started_at: float, ended_at: float) -> float:
    return round((ended_at - started_at) * 1000.0, 2)


@dataclass
class TurnLatency:
    call_id: str
    turn_id: str
    started_at: float = field(default_factory=perf_counter)
    speech_end_at: float | None = None
    first_asr_text_at: float | None = None
    first_llm_sentence_at: float | None = None
    first_tts_byte_at: float | None = None
    finished_at: float | None = None
    asr_queue_wait_ms: float = 0.0
    tts_queue_wait_ms: float = 0.0
    asr_segments: int = 0
    llm_sentences: int = 0
    tts_chunks: int = 0

    def mark_speech_end(self) -> None:
        if self.speech_end_at is None:
            self.speech_end_at = perf_counter()

    def mark_asr_text(self) -> None:
        if self.first_asr_text_at is None:
            self.first_asr_text_at = perf_counter()

    def mark_llm_sentence(self) -> None:
        if self.first_llm_sentence_at is None:
            self.first_llm_sentence_at = perf_counter()

    def mark_tts_byte(self) -> None:
        if self.first_tts_byte_at is None:
            self.first_tts_byte_at = perf_counter()

    def mark_finished(self) -> None:
        if self.finished_at is None:
            self.finished_at = perf_counter()

    def add_asr_queue_wait(self, wait_ms: float) -> None:
        self.asr_queue_wait_ms = round(self.asr_queue_wait_ms + wait_ms, 2)

    def add_tts_queue_wait(self, wait_ms: float) -> None:
        self.tts_queue_wait_ms = round(self.tts_queue_wait_ms + wait_ms, 2)

    def to_dict(self) -> dict[str, Any]:
        finished_at = self.finished_at or perf_counter()
        e2e_ms = _elapsed_ms(self.started_at, finished_at)

        speech_to_first_text_ms = None
        speech_to_first_audio_ms = None
        speech_to_done_ms = None

        if self.speech_end_at is not None and self.first_asr_text_at is not None:
            speech_to_first_text_ms = _elapsed_ms(
                self.speech_end_at,
                self.first_asr_text_at,
            )

        if self.speech_end_at is not None and self.first_tts_byte_at is not None:
            speech_to_first_audio_ms = _elapsed_ms(
                self.speech_end_at,
                self.first_tts_byte_at,
            )

        if self.speech_end_at is not None:
            speech_to_done_ms = _elapsed_ms(self.speech_end_at, finished_at)

        return {
            "call_id": self.call_id,
            "turn_id": self.turn_id,
            "e2e_ms": e2e_ms,
            "speech_to_first_text_ms": speech_to_first_text_ms,
            "speech_to_first_audio_ms": speech_to_first_audio_ms,
            "speech_to_done_ms": speech_to_done_ms,
            "asr_queue_wait_ms": self.asr_queue_wait_ms,
            "tts_queue_wait_ms": self.tts_queue_wait_ms,
            "asr_segments": self.asr_segments,
            "llm_sentences": self.llm_sentences,
            "tts_chunks": self.tts_chunks,
        }
