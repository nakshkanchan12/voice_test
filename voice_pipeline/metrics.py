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
    last_speech_chunk_at: float | None = None
    speech_end_at: float | None = None
    first_asr_text_at: float | None = None
    asr_complete_at: float | None = None
    llm_request_sent_at: float | None = None
    first_llm_sentence_at: float | None = None
    first_tts_byte_at: float | None = None
    finished_at: float | None = None
    asr_queue_wait_ms: float = 0.0
    tts_queue_wait_ms: float = 0.0
    asr_segments: int = 0
    llm_sentences: int = 0
    tts_chunks: int = 0
    eou_checks: int = 0
    eou_rejections: int = 0
    interrupted: bool = False
    cancelled_by_barge_in: bool = False

    def mark_speech_end(
        self,
        endpoint_at: float | None = None,
        last_speech_chunk_at: float | None = None,
    ) -> None:
        if self.last_speech_chunk_at is None and last_speech_chunk_at is not None:
            self.last_speech_chunk_at = last_speech_chunk_at
        if self.speech_end_at is None:
            self.speech_end_at = endpoint_at if endpoint_at is not None else perf_counter()

    def mark_asr_complete(self) -> None:
        if self.asr_complete_at is None:
            self.asr_complete_at = perf_counter()

    def mark_llm_request_sent(self) -> None:
        if self.llm_request_sent_at is None:
            self.llm_request_sent_at = perf_counter()

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

    def mark_interrupted(self) -> None:
        self.interrupted = True
        self.cancelled_by_barge_in = True

    def mark_cancelled_by_barge_in(self) -> None:
        self.mark_interrupted()

    def mark_eou_decision(self, is_done: bool) -> None:
        self.eou_checks += 1
        if not is_done:
            self.eou_rejections += 1

    def add_asr_queue_wait(self, wait_ms: float) -> None:
        self.asr_queue_wait_ms = round(self.asr_queue_wait_ms + wait_ms, 2)

    def add_tts_queue_wait(self, wait_ms: float) -> None:
        self.tts_queue_wait_ms = round(self.tts_queue_wait_ms + wait_ms, 2)

    def to_dict(self) -> dict[str, Any]:
        finished_at = self.finished_at or perf_counter()
        e2e_ms = _elapsed_ms(self.started_at, finished_at)

        vad_endpoint_ms = None
        asr_complete_ms = None
        llm_ttft_ms = None
        tts_ttfb_ms = None
        total_e2e_ms = None
        speech_to_first_text_ms = None
        speech_to_first_audio_ms = None
        speech_to_done_ms = None
        mic_to_first_audio_ms = None
        mic_to_first_audio_lt_700 = None
        mic_to_first_audio_lt_800 = None

        if self.last_speech_chunk_at is not None and self.speech_end_at is not None:
            vad_endpoint_ms = _elapsed_ms(self.last_speech_chunk_at, self.speech_end_at)

        if self.speech_end_at is not None and self.asr_complete_at is not None:
            asr_complete_ms = _elapsed_ms(self.speech_end_at, self.asr_complete_at)

        if self.llm_request_sent_at is not None and self.first_llm_sentence_at is not None:
            llm_ttft_ms = _elapsed_ms(self.llm_request_sent_at, self.first_llm_sentence_at)

        if self.first_llm_sentence_at is not None and self.first_tts_byte_at is not None:
            tts_ttfb_ms = _elapsed_ms(self.first_llm_sentence_at, self.first_tts_byte_at)

        if self.speech_end_at is not None and self.first_tts_byte_at is not None:
            total_e2e_ms = _elapsed_ms(self.speech_end_at, self.first_tts_byte_at)

        if self.speech_end_at is not None and self.first_asr_text_at is not None:
            speech_to_first_text_ms = _elapsed_ms(
                self.speech_end_at,
                self.first_asr_text_at,
            )

        speech_to_first_audio_ms = total_e2e_ms

        if self.first_tts_byte_at is not None:
            mic_to_first_audio_ms = _elapsed_ms(self.started_at, self.first_tts_byte_at)
            mic_to_first_audio_lt_700 = mic_to_first_audio_ms < 700
            mic_to_first_audio_lt_800 = mic_to_first_audio_ms < 800

        if self.speech_end_at is not None:
            speech_to_done_ms = _elapsed_ms(self.speech_end_at, finished_at)

        return {
            "call_id": self.call_id,
            "turn_id": self.turn_id,
            "vad_endpoint_ms": vad_endpoint_ms,
            "asr_complete_ms": asr_complete_ms,
            "llm_ttft_ms": llm_ttft_ms,
            "tts_ttfb_ms": tts_ttfb_ms,
            "total_e2e_ms": total_e2e_ms,
            "interrupted": self.interrupted,
            "e2e_ms": e2e_ms,
            "speech_to_first_text_ms": speech_to_first_text_ms,
            "speech_to_first_audio_ms": speech_to_first_audio_ms,
            "speech_to_done_ms": speech_to_done_ms,
            "mic_to_first_audio_ms": mic_to_first_audio_ms,
            "mic_to_first_audio_lt_700": mic_to_first_audio_lt_700,
            "mic_to_first_audio_lt_800": mic_to_first_audio_lt_800,
            "eou_checks": self.eou_checks,
            "eou_rejections": self.eou_rejections,
            "cancelled_by_barge_in": self.cancelled_by_barge_in,
            "asr_queue_wait_ms": self.asr_queue_wait_ms,
            "tts_queue_wait_ms": self.tts_queue_wait_ms,
            "asr_segments": self.asr_segments,
            "llm_sentences": self.llm_sentences,
            "tts_chunks": self.tts_chunks,
        }
