from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import AsyncIterator, Awaitable, Callable

from .interfaces import ASRProvider, LLMProvider, TTSProvider, VADProvider
from .metrics import TurnLatency, _elapsed_ms
from .types import AudioChunk, TurnRequest, iter_chunks

AudioChunkCallback = Callable[[bytes], Awaitable[None]]


@dataclass
class TurnOutput:
    audio_bytes: bytes
    latency: TurnLatency


class StreamingVoicePipeline:
    """Runs ASR -> LLM -> TTS concurrently with bounded queues."""

    def __init__(
        self,
        vad: VADProvider,
        asr: ASRProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        asr_queue_size: int = 2,
        tts_queue_size: int = 2,
    ) -> None:
        self.vad = vad
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.asr_queue_size = asr_queue_size
        self.tts_queue_size = tts_queue_size

    async def run_turn(
        self,
        request: TurnRequest,
        audio_stream: AsyncIterator[AudioChunk],
        on_audio_chunk: AudioChunkCallback | None = None,
    ) -> TurnOutput:
        asr_queue: asyncio.Queue[tuple[int, str] | object] = asyncio.Queue(maxsize=self.asr_queue_size)
        tts_queue: asyncio.Queue[tuple[int, str] | object] = asyncio.Queue(maxsize=self.tts_queue_size)
        asr_done = object()
        tts_done = object()
        output_parts: list[bytes] = []
        latency = TurnLatency(call_id=request.call_id, turn_id=request.turn_id)
        active_generation = 0
        tts_playing = asyncio.Event()

        def _flush_tts_queue() -> None:
            preserved: list[object] = []
            while True:
                try:
                    item = tts_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if item is tts_done:
                    preserved.append(item)

            for item in preserved:
                tts_queue.put_nowait(item)

        async def _stop_tts_for_turn() -> None:
            stop_fn = getattr(self.tts, "stop_streaming", None)
            if stop_fn is None:
                return

            try:
                await stop_fn(request.turn_id)
            except TypeError:
                # Backward-compatible fallback for alternate stop signatures.
                try:
                    await stop_fn(request)
                except Exception:
                    return
            except Exception:
                return

        async def asr_stage() -> None:
            nonlocal active_generation
            async for segment in self.vad.stream_segments(audio_stream):
                if segment.ended_by_silence:
                    latency.mark_speech_end()

                if (
                    request.enable_barge_in
                    and tts_playing.is_set()
                    and latency.speech_end_at is not None
                    and segment.duration_ms >= max(1, request.barge_in_min_speech_ms)
                ):
                    active_generation += 1
                    latency.mark_cancelled_by_barge_in()
                    _flush_tts_queue()
                    await _stop_tts_for_turn()

                transcript = (await self.asr.transcribe(segment, request)).strip()
                if not transcript:
                    continue

                latency.asr_segments += 1
                latency.mark_asr_text()
                wait_started = perf_counter()
                await asr_queue.put((active_generation, transcript))
                latency.add_asr_queue_wait(_elapsed_ms(wait_started, perf_counter()))

            await asr_queue.put(asr_done)

        async def llm_stage() -> None:
            nonlocal active_generation
            while True:
                item = await asr_queue.get()
                if item is asr_done:
                    break

                transcript_generation, transcript = item
                if transcript_generation != active_generation:
                    continue

                async for sentence in self.llm.stream_sentences(transcript, request):
                    if transcript_generation != active_generation:
                        break

                    cleaned = sentence.strip()
                    if not cleaned:
                        continue

                    latency.llm_sentences += 1
                    latency.mark_llm_sentence()
                    wait_started = perf_counter()
                    await tts_queue.put((transcript_generation, cleaned))
                    latency.add_tts_queue_wait(_elapsed_ms(wait_started, perf_counter()))

            await tts_queue.put(tts_done)

        async def tts_stage() -> None:
            nonlocal active_generation
            while True:
                item = await tts_queue.get()
                if item is tts_done:
                    break

                sentence_generation, sentence = item
                if sentence_generation != active_generation:
                    continue

                tts_playing.set()
                try:
                    async for pcm in self.tts.stream_audio(sentence, request):
                        if sentence_generation != active_generation:
                            await _stop_tts_for_turn()
                            break

                        if not pcm:
                            continue

                        latency.tts_chunks += 1
                        latency.mark_tts_byte()
                        output_parts.append(pcm)
                        if on_audio_chunk is not None:
                            await on_audio_chunk(pcm)
                finally:
                    tts_playing.clear()

        await asyncio.gather(asr_stage(), llm_stage(), tts_stage())
        latency.mark_finished()

        if latency.speech_end_at is None:
            # Fallback for empty-speech turns where VAD never produced a segment.
            latency.mark_speech_end()

        return TurnOutput(audio_bytes=b"".join(output_parts), latency=latency)

    async def run_turn_from_chunks(
        self,
        request: TurnRequest,
        chunks: list[AudioChunk],
        on_audio_chunk: AudioChunkCallback | None = None,
    ) -> TurnOutput:
        return await self.run_turn(request, iter_chunks(chunks), on_audio_chunk=on_audio_chunk)
