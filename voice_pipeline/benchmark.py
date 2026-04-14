from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .factory import build_asr_provider, resolve_selected_asr_model
from .mock_components import MockLLM, MockTTS, MockVAD
from .pipeline import StreamingVoicePipeline
from .types import AudioChunk, TurnRequest


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)

    if lower == upper:
        return round(sorted_values[int(rank)], 2)

    lower_val = sorted_values[lower]
    upper_val = sorted_values[upper]
    blended = lower_val + (upper_val - lower_val) * (rank - lower)
    return round(blended, 2)


def build_mock_turn_chunks(index: int) -> list[AudioChunk]:
    # Vary utterance length slightly so p95 has meaningful spread.
    speech_chunks = 100 + (index % 25)  # 2.0s to 2.48s of speech
    silence_chunks = 18  # 360ms silence to trigger VAD endpoint

    chunks = [
        AudioChunk(pcm=b"\x00" * 640, duration_ms=20, is_speech=True)
        for _ in range(speech_chunks)
    ]
    chunks.extend(
        AudioChunk(pcm=b"\x00" * 640, duration_ms=20, is_speech=False)
        for _ in range(silence_chunks)
    )
    return chunks


def summarize(latencies: list[dict[str, Any]], concurrency: int) -> dict[str, Any]:
    speech_to_audio = [
        float(item["speech_to_first_audio_ms"])
        for item in latencies
        if item["speech_to_first_audio_ms"] is not None
    ]
    mic_to_audio = [
        float(item["mic_to_first_audio_ms"])
        for item in latencies
        if item.get("mic_to_first_audio_ms") is not None
    ]
    speech_to_done = [
        float(item["speech_to_done_ms"])
        for item in latencies
        if item["speech_to_done_ms"] is not None
    ]
    e2e = [float(item["e2e_ms"]) for item in latencies]

    under_700 = [value for value in speech_to_audio if value < 700.0]
    hit_rate = round((len(under_700) / len(speech_to_audio)) * 100.0, 2) if speech_to_audio else 0.0

    return {
        "concurrency": concurrency,
        "calls": len(latencies),
        "speech_to_first_audio_p50_ms": percentile(speech_to_audio, 50),
        "speech_to_first_audio_p95_ms": percentile(speech_to_audio, 95),
        "mic_to_first_audio_p50_ms": percentile(mic_to_audio, 50),
        "mic_to_first_audio_p95_ms": percentile(mic_to_audio, 95),
        "speech_to_done_p50_ms": percentile(speech_to_done, 50),
        "speech_to_done_p95_ms": percentile(speech_to_done, 95),
        "e2e_p50_ms": percentile(e2e, 50),
        "e2e_p95_ms": percentile(e2e, 95),
        "mean_asr_queue_wait_ms": round(mean(item["asr_queue_wait_ms"] for item in latencies), 2),
        "mean_tts_queue_wait_ms": round(mean(item["tts_queue_wait_ms"] for item in latencies), 2),
        "target_lt_700ms_hit_rate_pct": hit_rate,
        "lt_800ms_first_audio_hit_rate_pct": round(
            (sum(1 for value in speech_to_audio if value < 800.0) / max(1, len(speech_to_audio))) * 100.0,
            2,
        ),
    }


async def run_load_benchmark(
    total_calls: int = 40,
    concurrency: int = 20,
    asr_mode: str = "mock",
    asr_model: str | None = None,
) -> dict[str, Any]:
    resolved_model = asr_model or resolve_selected_asr_model()
    pipeline = StreamingVoicePipeline(
        vad=MockVAD(min_speech_ms=200, silence_ms=300),
        asr=build_asr_provider(mode=asr_mode, selected_model=resolved_model),
        llm=MockLLM(ttft_ms=85, token_delay_ms=8),
        tts=MockTTS(ttfb_ms=95, chunk_delay_ms=20),
        asr_queue_size=2,
        tts_queue_size=2,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def run_single(index: int) -> dict[str, Any]:
        async with semaphore:
            request = TurnRequest(
                call_id=f"call-{index // 2}",
                turn_id=f"turn-{index}",
                metadata={"scenario": "step1_mock"},
            )
            output = await pipeline.run_turn_from_chunks(
                request=request,
                chunks=build_mock_turn_chunks(index),
            )
            row = output.latency.to_dict()
            row["audio_bytes"] = len(output.audio_bytes)
            return row

    per_call = await asyncio.gather(*(run_single(i) for i in range(total_calls)))
    summary = summarize(per_call, concurrency=concurrency)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "step1_streaming_pipeline_mock",
        "asr_mode": asr_mode,
        "asr_model": resolved_model,
        "summary": summary,
        "calls": per_call,
    }


def save_results(report: dict[str, Any], output_path: str | None = None) -> str:
    if output_path:
        target = Path(output_path)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = Path("results") / f"step1_benchmark_{stamp}.json"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(target)
