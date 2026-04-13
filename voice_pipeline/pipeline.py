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
        asr_queue: asyncio.Queue[str | object] = asyncio.Queue(maxsize=self.asr_queue_size)
        tts_queue: asyncio.Queue[str | object] = asyncio.Queue(maxsize=self.tts_queue_size)
        asr_done = object()
        tts_done = object()
        output_parts: list[bytes] = []
        latency = TurnLatency(call_id=request.call_id, turn_id=request.turn_id)

        async def asr_stage() -> None:
            async for segment in self.vad.stream_segments(audio_stream):
                if segment.ended_by_silence:
                    latency.mark_speech_end()

                transcript = (await self.asr.transcribe(segment, request)).strip()
                if not transcript:
                    continue

                latency.asr_segments += 1
                latency.mark_asr_text()
                wait_started = perf_counter()
                await asr_queue.put(transcript)
                latency.add_asr_queue_wait(_elapsed_ms(wait_started, perf_counter()))

            await asr_queue.put(asr_done)

        async def llm_stage() -> None:
            while True:
                item = await asr_queue.get()
                if item is asr_done:
                    break

                transcript = str(item)
                async for sentence in self.llm.stream_sentences(transcript, request):
                    cleaned = sentence.strip()
                    if not cleaned:
                        continue

                    latency.llm_sentences += 1
                    latency.mark_llm_sentence()
                    wait_started = perf_counter()
                    await tts_queue.put(cleaned)
                    latency.add_tts_queue_wait(_elapsed_ms(wait_started, perf_counter()))

            await tts_queue.put(tts_done)

        async def tts_stage() -> None:
            while True:
                item = await tts_queue.get()
                if item is tts_done:
                    break

                sentence = str(item)
                async for pcm in self.tts.stream_audio(sentence, request):
                    if not pcm:
                        continue

                    latency.tts_chunks += 1
                    latency.mark_tts_byte()
                    output_parts.append(pcm)
                    if on_audio_chunk is not None:
                        await on_audio_chunk(pcm)

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
