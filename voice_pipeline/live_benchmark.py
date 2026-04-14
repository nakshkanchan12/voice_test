from __future__ import annotations

import argparse
import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .asr_dataset import load_audio_samples
from .http_providers import HTTPASRProvider, HTTPTTSProvider
from .pipeline import StreamingVoicePipeline
from .real_llm import OpenAICompatibleLLM
from .real_vad import SileroVADProvider
from .types import AudioChunk, TurnRequest
from .webrtc_rpc import default_webrtc_offer_url


async def _close_if_possible(instance: Any) -> None:
    close_fn = getattr(instance, "close", None)
    if close_fn is not None:
        await close_fn()


def _parse_bool_arg(value: str) -> bool:
    token = value.strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(ordered[int(rank)], 2)

    low_v = ordered[low]
    high_v = ordered[high]
    return round(low_v + (high_v - low_v) * (rank - low), 2)


def _pcm_to_chunks(pcm16: bytes, sample_rate_hz: int, chunk_ms: int = 20) -> list[AudioChunk]:
    bytes_per_sample = 2
    samples_per_chunk = int((sample_rate_hz * chunk_ms) / 1000)
    bytes_per_chunk = max(2, samples_per_chunk * bytes_per_sample)

    chunks: list[AudioChunk] = []
    for idx in range(0, len(pcm16), bytes_per_chunk):
        part = pcm16[idx : idx + bytes_per_chunk]
        if len(part) < bytes_per_chunk:
            part = part + (b"\x00" * (bytes_per_chunk - len(part)))

        chunks.append(
            AudioChunk(
                pcm=part,
                duration_ms=chunk_ms,
                sample_rate_hz=sample_rate_hz,
                is_speech=True,
            )
        )

    silence = b"\x00" * bytes_per_chunk
    for _ in range(20):
        chunks.append(
            AudioChunk(
                pcm=silence,
                duration_ms=chunk_ms,
                sample_rate_hz=sample_rate_hz,
                is_speech=False,
            )
        )

    return chunks


def _summarize(rows: list[dict[str, Any]], concurrency: int) -> dict[str, Any]:
    s2a = [float(r["speech_to_first_audio_ms"]) for r in rows if r["speech_to_first_audio_ms"] is not None]
    s2d = [float(r["speech_to_done_ms"]) for r in rows if r["speech_to_done_ms"] is not None]
    mic2a = [float(r["mic_to_first_audio_ms"]) for r in rows if r.get("mic_to_first_audio_ms") is not None]
    e2e = [float(r["e2e_ms"]) for r in rows]

    return {
        "concurrency": concurrency,
        "calls": len(rows),
        "speech_to_first_audio_p50_ms": _percentile(s2a, 50),
        "speech_to_first_audio_p95_ms": _percentile(s2a, 95),
        "mic_to_first_audio_p50_ms": _percentile(mic2a, 50),
        "mic_to_first_audio_p95_ms": _percentile(mic2a, 95),
        "speech_to_done_p50_ms": _percentile(s2d, 50),
        "speech_to_done_p95_ms": _percentile(s2d, 95),
        "e2e_p50_ms": _percentile(e2e, 50),
        "e2e_p95_ms": _percentile(e2e, 95),
        "mean_asr_queue_wait_ms": round(mean(r["asr_queue_wait_ms"] for r in rows), 2),
        "mean_tts_queue_wait_ms": round(mean(r["tts_queue_wait_ms"] for r in rows), 2),
        "lt_700ms_first_audio_hit_rate_pct": round(
            (sum(1 for x in s2a if x < 700.0) / max(1, len(s2a))) * 100.0,
            2,
        ),
        "lt_800ms_first_audio_hit_rate_pct": round(
            (sum(1 for x in s2a if x < 800.0) / max(1, len(s2a))) * 100.0,
            2,
        ),
    }


async def run_live_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    samples, source = load_audio_samples(
        source=args.dataset_source,
        limit=max(1, args.sample_pool),
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
    )
    if not samples:
        raise RuntimeError("No audio samples available for benchmark")

    vad = SileroVADProvider(
        sample_rate_hz=samples[0].sample_rate_hz,
        min_speech_ms=args.vad_min_speech_ms,
        min_silence_ms=args.vad_min_silence_ms,
        threshold=args.vad_threshold,
        streaming=args.vad_streaming,
        partial_segment_ms=args.vad_partial_segment_ms,
        max_stream_segment_ms=args.vad_max_stream_segment_ms,
    )
    if args.transport == "webrtc":
        from .webrtc_providers import WebRTCASRProvider, WebRTCLLMProvider, WebRTCTTSProvider

        asr_offer_url = args.asr_webrtc_offer_url or f"{args.asr_url.rstrip('/')}/webrtc/offer"
        llm_offer_url = args.llm_webrtc_offer_url or default_webrtc_offer_url(args.llm_base_url)
        tts_offer_url = args.tts_webrtc_offer_url or f"{args.tts_url.rstrip('/')}/webrtc/offer"

        asr = WebRTCASRProvider(
            offer_url=asr_offer_url,
            timeout_s=args.webrtc_timeout_s,
            reuse_session=args.webrtc_reuse_session,
        )
        llm = WebRTCLLMProvider(
            offer_url=llm_offer_url,
            model=args.llm_model,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream_mode=args.llm_stream_mode,
            aggressive_min_tokens=args.llm_aggressive_min_tokens,
            timeout_s=args.webrtc_timeout_s,
            reuse_session=args.webrtc_reuse_session,
        )
        tts = WebRTCTTSProvider(
            offer_url=tts_offer_url,
            timeout_s=args.webrtc_timeout_s,
            reuse_session=args.webrtc_reuse_session,
        )
    else:
        asr = HTTPASRProvider(base_url=args.asr_url)
        llm = OpenAICompatibleLLM(
            base_url=args.llm_base_url,
            model=args.llm_model,
            system_prompt=args.system_prompt,
            api_key=args.llm_api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream_mode=args.llm_stream_mode,
            aggressive_min_tokens=args.llm_aggressive_min_tokens,
        )
        tts = HTTPTTSProvider(base_url=args.tts_url)

    pipeline = StreamingVoicePipeline(vad=vad, asr=asr, llm=llm, tts=tts)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_one(index: int) -> dict[str, Any]:
        async with semaphore:
            sample = samples[index % len(samples)]
            chunks = _pcm_to_chunks(sample.pcm16, sample.sample_rate_hz)
            request = TurnRequest(
                call_id=f"call-{index}",
                turn_id=f"turn-{index}",
                llm_stream_mode=args.llm_stream_mode,
                enable_barge_in=args.enable_barge_in,
                barge_in_min_speech_ms=args.barge_in_min_speech_ms,
            )
            output = await pipeline.run_turn_from_chunks(request=request, chunks=chunks)
            row = output.latency.to_dict()
            row["audio_bytes"] = len(output.audio_bytes)
            row["source_id"] = sample.source_id
            return row

    rows = await asyncio.gather(*(run_one(i) for i in range(args.calls)))
    await _close_if_possible(asr)
    await _close_if_possible(llm)
    await _close_if_possible(tts)

    summary = _summarize(rows, concurrency=args.concurrency)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "live_real_pipeline",
        "transport": args.transport,
        "transport_config": {
            "webrtc_timeout_s": args.webrtc_timeout_s if args.transport == "webrtc" else None,
            "webrtc_reuse_session": args.webrtc_reuse_session if args.transport == "webrtc" else None,
        },
        "dataset_source_used": source,
        "summary": summary,
        "calls": rows,
    }


def _save(report: dict[str, Any], output_path: str) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent benchmark over live ASR/LLM/TTS endpoints.")
    parser.add_argument("--calls", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--sample-pool", type=int, default=2)

    parser.add_argument("--dataset-source", type=str, default="hf", choices=["hf", "kaggle", "tts", "mock"])
    parser.add_argument("--hf-dataset", type=str, default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--hf-config", type=str, default="clean")
    parser.add_argument("--hf-split", type=str, default="validation")

    parser.add_argument("--asr-url", type=str, default="http://127.0.0.1:8011")
    parser.add_argument("--llm-base-url", type=str, default="http://127.0.0.1:30000/v1")
    parser.add_argument("--llm-model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--llm-api-key", type=str, default="EMPTY")
    parser.add_argument("--system-prompt", type=str, default="You are a concise and compliant banking collection assistant.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--llm-stream-mode", type=str, default="aggressive", choices=["sentence", "aggressive"])
    parser.add_argument("--llm-aggressive-min-tokens", type=int, default=5)
    parser.add_argument("--tts-url", type=str, default="http://127.0.0.1:8012")
    parser.add_argument("--transport", type=str, default="http", choices=["http", "webrtc"])
    parser.add_argument("--asr-webrtc-offer-url", type=str, default="")
    parser.add_argument("--llm-webrtc-offer-url", type=str, default="")
    parser.add_argument("--tts-webrtc-offer-url", type=str, default="")
    parser.add_argument("--webrtc-timeout-s", type=float, default=120.0)
    parser.add_argument("--webrtc-reuse-session", type=_parse_bool_arg, default=True)

    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=200)
    parser.add_argument("--vad-min-silence-ms", type=int, default=300)
    parser.add_argument("--vad-streaming", type=_parse_bool_arg, default=True)
    parser.add_argument("--vad-partial-segment-ms", type=int, default=320)
    parser.add_argument("--vad-max-stream-segment-ms", type=int, default=1280)

    parser.add_argument("--enable-barge-in", type=_parse_bool_arg, default=True)
    parser.add_argument("--barge-in-min-speech-ms", type=int, default=120)

    parser.add_argument(
        "--output",
        type=str,
        default=f"results/live_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = asyncio.run(run_live_benchmark(args))
    path = _save(report, args.output)
    print(json.dumps(report["summary"], indent=2))
    print(f"results_saved_to={path}")
