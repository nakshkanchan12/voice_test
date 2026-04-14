from __future__ import annotations

from time import perf_counter
from typing import AsyncIterator

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from .types import AudioChunk, SpeechSegment


class SileroVADProvider:
    """Real VAD using Silero model, emitting speech segments from streamed chunks."""

    def __init__(
        self,
        sample_rate_hz: int = 16000,
        min_speech_ms: int = 200,
        min_silence_ms: int = 300,
        threshold: float = 0.5,
        streaming: bool = True,
        partial_segment_ms: int = 320,
        max_stream_segment_ms: int = 1280,
        use_chunk_flags: bool = False,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.threshold = threshold
        self.streaming = streaming
        self.partial_segment_ms = max(40, partial_segment_ms)
        self.max_stream_segment_ms = max(self.partial_segment_ms, max_stream_segment_ms)
        self.use_chunk_flags = use_chunk_flags
        self.model = None if self.streaming else load_silero_vad()

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = load_silero_vad()

    def _detect_chunk_speech(self, chunk: AudioChunk) -> bool:
        if self.use_chunk_flags:
            return bool(chunk.is_speech)

        self._ensure_model()
        assert self.model is not None

        samples = np.frombuffer(chunk.pcm, dtype=np.int16)
        if samples.size == 0:
            return False

        normalized = samples.astype(np.float32) / 32768.0
        min_window_ms = max(10, min(80, max(1, chunk.duration_ms)))

        try:
            timestamps = get_speech_timestamps(
                torch.from_numpy(normalized),
                self.model,
                sampling_rate=chunk.sample_rate_hz,
                min_speech_duration_ms=min_window_ms,
                min_silence_duration_ms=min_window_ms,
                threshold=self.threshold,
            )
            return bool(timestamps)
        except Exception:
            return bool(chunk.is_speech)

    async def stream_segments(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[SpeechSegment]:
        if self.streaming:
            async for segment in self._stream_segments_incremental(audio_stream):
                yield segment
            return

        async for segment in self._stream_segments_batch(audio_stream):
            yield segment

    async def _stream_segments_incremental(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[SpeechSegment]:
        active_chunks: list[AudioChunk] = []
        active_ms = 0
        silence_ms = 0
        last_speech_chunk_at: float | None = None

        async for chunk in audio_stream:
            duration_ms = max(1, chunk.duration_ms)
            chunk_is_speech = self._detect_chunk_speech(chunk)

            if chunk_is_speech:
                active_chunks.append(chunk)
                active_ms += duration_ms
                silence_ms = 0
                last_speech_chunk_at = perf_counter()

                if active_ms < self.min_speech_ms:
                    continue

                if active_ms >= self.partial_segment_ms or active_ms >= self.max_stream_segment_ms:
                    yield SpeechSegment(
                        chunks=tuple(active_chunks),
                        sample_rate_hz=chunk.sample_rate_hz,
                        ended_by_silence=False,
                        last_speech_chunk_at=last_speech_chunk_at,
                    )
                    # Keep a tiny trailing context so silence endpointing can still close the turn.
                    active_chunks = active_chunks[-1:]
                    active_ms = sum(max(1, chunk.duration_ms) for chunk in active_chunks)
                continue

            if not active_chunks:
                continue

            silence_ms += duration_ms
            if silence_ms < self.min_silence_ms:
                continue

            if active_ms < self.min_speech_ms:
                active_chunks = []
                active_ms = 0
                silence_ms = 0
                continue

            yield SpeechSegment(
                chunks=tuple(active_chunks),
                sample_rate_hz=chunk.sample_rate_hz,
                ended_by_silence=True,
                last_speech_chunk_at=last_speech_chunk_at,
                endpoint_at=perf_counter(),
            )
            active_chunks = []
            active_ms = 0
            silence_ms = 0
            last_speech_chunk_at = None

        if active_chunks and active_ms >= self.min_speech_ms:
            yield SpeechSegment(
                chunks=tuple(active_chunks),
                sample_rate_hz=active_chunks[0].sample_rate_hz,
                ended_by_silence=False,
                last_speech_chunk_at=last_speech_chunk_at,
            )

    async def _stream_segments_batch(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[SpeechSegment]:
        self._ensure_model()
        assert self.model is not None

        chunks: list[AudioChunk] = []
        async for chunk in audio_stream:
            chunks.append(chunk)

        if not chunks:
            return

        pcm16 = b"".join(chunk.pcm for chunk in chunks)
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

        timestamps = get_speech_timestamps(
            torch.from_numpy(samples),
            self.model,
            sampling_rate=self.sample_rate_hz,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            threshold=self.threshold,
        )

        if not timestamps:
            speech_chunks = [chunk for chunk in chunks if chunk.is_speech]
            if speech_chunks:
                yield SpeechSegment(
                    chunks=tuple(speech_chunks),
                    sample_rate_hz=self.sample_rate_hz,
                    ended_by_silence=False,
                )
            return

        starts: list[int] = []
        ends: list[int] = []
        cursor = 0
        for chunk in chunks:
            starts.append(cursor)
            duration_samples = int((chunk.duration_ms / 1000.0) * self.sample_rate_hz)
            cursor += max(duration_samples, 1)
            ends.append(cursor)

        for stamp in timestamps:
            start_sample = int(stamp["start"])
            end_sample = int(stamp["end"])

            selected: list[AudioChunk] = []
            for idx, chunk in enumerate(chunks):
                if ends[idx] <= start_sample:
                    continue
                if starts[idx] >= end_sample:
                    break
                selected.append(chunk)

            if selected:
                yield SpeechSegment(
                    chunks=tuple(selected),
                    sample_rate_hz=self.sample_rate_hz,
                    ended_by_silence=True,
                    endpoint_at=perf_counter(),
                )
