from __future__ import annotations

import asyncio
import math
from typing import AsyncIterator, Sequence

from .types import AudioChunk


def pcm_to_chunks(
    pcm16: bytes,
    sample_rate_hz: int,
    chunk_ms: int = 20,
    trailing_silence_ms: int = 400,
) -> list[AudioChunk]:
    """Split mono PCM16 into fixed chunks plus trailing silence for endpoint detection."""

    bytes_per_sample = 2
    samples_per_chunk = max(1, int((sample_rate_hz * chunk_ms) / 1000))
    bytes_per_chunk = max(2, samples_per_chunk * bytes_per_sample)

    chunks: list[AudioChunk] = []
    for idx in range(0, len(pcm16), bytes_per_chunk):
        part = pcm16[idx : idx + bytes_per_chunk]
        if not part:
            continue

        if len(part) < bytes_per_chunk:
            part = part + (b"\x00" * (bytes_per_chunk - len(part)))

        chunks.append(
            AudioChunk(
                pcm=part,
                duration_ms=chunk_ms,
                sample_rate_hz=sample_rate_hz,
                is_speech=True,
                source="user",
            )
        )

    silence_chunks = max(0, int(math.ceil(max(0, trailing_silence_ms) / max(1, chunk_ms))))
    if silence_chunks > 0:
        silence = b"\x00" * bytes_per_chunk
        for _ in range(silence_chunks):
            chunks.append(
                AudioChunk(
                    pcm=silence,
                    duration_ms=chunk_ms,
                    sample_rate_hz=sample_rate_hz,
                    is_speech=False,
                    source="user",
                )
            )

    return chunks


def trim_chunks(chunks: Sequence[AudioChunk], max_duration_ms: int) -> list[AudioChunk]:
    if max_duration_ms <= 0:
        return []

    kept: list[AudioChunk] = []
    elapsed_ms = 0
    for chunk in chunks:
        if elapsed_ms >= max_duration_ms:
            break
        kept.append(chunk)
        elapsed_ms += max(1, chunk.duration_ms)

    return kept


async def simulate_live_stream(
    chunks: Sequence[AudioChunk],
    real_time: bool = True,
    speedup: float = 1.0,
) -> AsyncIterator[AudioChunk]:
    pacing = max(0.01, float(speedup))
    for chunk in chunks:
        yield chunk
        if not real_time:
            continue

        delay_s = (max(1, chunk.duration_ms) / 1000.0) / pacing
        await asyncio.sleep(delay_s)


async def simulate_interruptible_live_stream(
    leading_chunks: Sequence[AudioChunk],
    barge_in_chunks: Sequence[AudioChunk],
    *,
    barge_in_trigger: asyncio.Event | None,
    barge_in_delay_ms: int = 120,
    barge_in_timeout_ms: int = 5000,
    real_time: bool = True,
    speedup: float = 1.0,
) -> AsyncIterator[AudioChunk]:
    async for chunk in simulate_live_stream(leading_chunks, real_time=real_time, speedup=speedup):
        yield chunk

    if not barge_in_chunks:
        return

    if barge_in_trigger is not None:
        timeout_s = max(0.1, barge_in_timeout_ms / 1000.0)
        try:
            await asyncio.wait_for(barge_in_trigger.wait(), timeout=timeout_s)
        except TimeoutError:
            return

    if real_time and barge_in_delay_ms > 0:
        delay_s = (barge_in_delay_ms / 1000.0) / max(0.01, float(speedup))
        await asyncio.sleep(delay_s)

    async for chunk in simulate_live_stream(barge_in_chunks, real_time=real_time, speedup=speedup):
        yield chunk