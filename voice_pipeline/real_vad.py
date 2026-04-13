from __future__ import annotations

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
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.threshold = threshold
        self.model = load_silero_vad()

    async def stream_segments(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[SpeechSegment]:
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
                )
