from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

from .aec import ReferenceEchoCanceller
from .asr_dataset import load_audio_samples
from .eou import build_eou_decider
from .live_simulation import (
    pcm_to_chunks,
    simulate_interruptible_live_stream,
    simulate_live_stream,
    trim_chunks,
)
from .pipeline import StreamingVoicePipeline
from .real_vad import SileroVADProvider
from .recommended_providers import NemotronHTTPASRProvider, Qwen3TTSProvider, SGLangOpenAILLMProvider
from .types import TurnRequest
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


def _summarize(rows: list[dict[str, Any]], concurrency: int) -> dict[str, Any]:
    def _pluck(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    s2a = _pluck("speech_to_first_audio_ms")
    s2d = _pluck("speech_to_done_ms")
    mic2a = _pluck("mic_to_first_audio_ms")
    e2e = _pluck("e2e_ms")
    total_e2e = _pluck("total_e2e_ms")
    vad_endpoint = _pluck("vad_endpoint_ms")
    asr_complete = _pluck("asr_complete_ms")
    llm_ttft = _pluck("llm_ttft_ms")
    tts_ttfb = _pluck("tts_ttfb_ms")

    interrupted = sum(1 for row in rows if bool(row.get("interrupted")))

    return {
        "concurrency": concurrency,
        "calls": len(rows),
        "interrupted_calls": interrupted,
        "interrupted_rate_pct": round((interrupted / max(1, len(rows))) * 100.0, 2),
        "vad_endpoint_p50_ms": _percentile(vad_endpoint, 50),
        "vad_endpoint_p95_ms": _percentile(vad_endpoint, 95),
        "asr_complete_p50_ms": _percentile(asr_complete, 50),
        "asr_complete_p95_ms": _percentile(asr_complete, 95),
        "llm_ttft_p50_ms": _percentile(llm_ttft, 50),
        "llm_ttft_p95_ms": _percentile(llm_ttft, 95),
        "tts_ttfb_p50_ms": _percentile(tts_ttfb, 50),
        "tts_ttfb_p95_ms": _percentile(tts_ttfb, 95),
        "total_e2e_p50_ms": _percentile(total_e2e, 50),
        "total_e2e_p95_ms": _percentile(total_e2e, 95),
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


def _load_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt_file:
        return Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    return args.system_prompt


async def _wait_health(url: str, timeout_s: float = 180.0) -> None:
    async with httpx.AsyncClient(timeout=5.0) as client:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"Health check timed out for {url}")
            await asyncio.sleep(1.0)


async def run_recommended_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed >= 0:
        random.seed(args.seed)

    if args.wait_for_health:
        await _wait_health(f"{args.asr_url.rstrip('/')}/health", timeout_s=args.health_timeout)
        await _wait_health(f"{args.tts_url.rstrip('/')}/health", timeout_s=args.health_timeout)
        await _wait_health(f"{args.llm_base_url.rstrip('/v1').rstrip('/')}/health", timeout_s=args.health_timeout)

    samples, source = load_audio_samples(
        source=args.dataset_source,
        limit=max(1, args.sample_pool),
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
    )
    if not samples:
        raise RuntimeError("No audio samples available for benchmark")

    system_prompt = _load_system_prompt(args)
    shared_client: httpx.AsyncClient | None = None

    vad = SileroVADProvider(
        sample_rate_hz=samples[0].sample_rate_hz,
        min_speech_ms=args.vad_min_speech_ms,
        min_silence_ms=args.vad_min_silence_ms,
        threshold=args.vad_threshold,
        streaming=args.vad_streaming,
        partial_segment_ms=args.vad_partial_segment_ms,
        max_stream_segment_ms=args.vad_max_stream_segment_ms,
        use_chunk_flags=args.vad_use_chunk_flags,
    )

    eou_decider = build_eou_decider(
        args.eou_mode,
        model_name=args.eou_model,
        min_words=args.eou_min_words,
        min_chars=args.eou_min_chars,
    )

    aec_factory = None
    if args.aec_enabled:
        aec_factory = lambda: ReferenceEchoCanceller(
            history_ms=args.aec_history_ms,
            chunk_ms=args.audio_chunk_ms,
            correlation_threshold=args.aec_correlation_threshold,
            min_rms=args.aec_min_rms,
        )

    if args.transport == "webrtc":
        from .webrtc_providers import WebRTCASRProvider, WebRTCLLMProvider, WebRTCTTSProvider

        asr_offer_url = args.asr_webrtc_offer_url or f"{args.asr_url.rstrip('/')}/webrtc/offer"
        llm_offer_url = args.llm_webrtc_offer_url or default_webrtc_offer_url(args.llm_base_url)
        tts_offer_url = args.tts_webrtc_offer_url or f"{args.tts_url.rstrip('/')}/webrtc/offer"

        asr = WebRTCASRProvider(
            offer_url=asr_offer_url,
            language=args.asr_language,
            timeout_s=args.http_timeout,
        )
        llm = WebRTCLLMProvider(
            offer_url=llm_offer_url,
            model=args.llm_model,
            system_prompt=system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream_mode=args.llm_stream_mode,
            aggressive_min_tokens=args.llm_aggressive_min_tokens,
            timeout_s=args.http_timeout,
        )
        tts = WebRTCTTSProvider(
            offer_url=tts_offer_url,
            model=args.tts_model,
            voice=args.tts_voice,
            split_granularity=args.tts_split_granularity,
            timeout_s=args.http_timeout,
        )
    else:
        connector_limits = httpx.Limits(
            max_connections=max(20, args.concurrency * 3),
            max_keepalive_connections=max(20, args.concurrency * 3),
        )
        shared_client = httpx.AsyncClient(timeout=args.http_timeout, limits=connector_limits)

        asr = NemotronHTTPASRProvider(
            base_url=args.asr_url,
            language=args.asr_language,
            chunk_ms=args.asr_chunk_ms,
            transcribe_path=args.asr_transcribe_path,
            audio_field=args.asr_audio_field,
            model=args.asr_model,
            api_key=args.asr_api_key,
            timeout_s=args.http_timeout,
            client=shared_client,
        )
        llm = SGLangOpenAILLMProvider(
            base_url=args.llm_base_url,
            model=args.llm_model,
            system_prompt=system_prompt,
            api_key=args.llm_api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream_mode=args.llm_stream_mode,
            aggressive_min_tokens=args.llm_aggressive_min_tokens,
        )
        tts = Qwen3TTSProvider(
            base_url=args.tts_url,
            mode=args.tts_mode,
            model=args.tts_model,
            voice=args.tts_voice,
            split_granularity=args.tts_split_granularity,
            timeout_s=args.http_timeout,
            client=shared_client,
        )

    pipeline = StreamingVoicePipeline(
        vad=vad,
        asr=asr,
        llm=llm,
        tts=tts,
        asr_queue_size=args.asr_queue_size,
        tts_queue_size=args.tts_queue_size,
        eou_decider=eou_decider,
        aec_factory=aec_factory,
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_one(index: int) -> dict[str, Any]:
        async with semaphore:
            sample = samples[index % len(samples)]
            chunks = pcm_to_chunks(
                sample.pcm16,
                sample.sample_rate_hz,
                chunk_ms=args.audio_chunk_ms,
                trailing_silence_ms=args.trailing_silence_ms,
            )

            simulate_barge_in = (
                args.simulate_barge_in_rate > 0.0
                and random.random() < min(1.0, max(0.0, args.simulate_barge_in_rate))
            )

            request = TurnRequest(
                call_id=f"call-{index}",
                turn_id=f"turn-{index}",
                llm_stream_mode=args.llm_stream_mode,
                enable_barge_in=args.enable_barge_in,
                barge_in_min_speech_ms=args.barge_in_min_speech_ms,
            )

            if simulate_barge_in:
                tts_started = asyncio.Event()
                barge_chunks = trim_chunks(
                    pcm_to_chunks(
                        sample.pcm16,
                        sample.sample_rate_hz,
                        chunk_ms=args.audio_chunk_ms,
                        trailing_silence_ms=args.trailing_silence_ms,
                    ),
                    max_duration_ms=args.barge_in_utterance_ms,
                )

                async def _on_audio_chunk(_: bytes) -> None:
                    tts_started.set()

                audio_stream = simulate_interruptible_live_stream(
                    chunks,
                    barge_chunks,
                    barge_in_trigger=tts_started,
                    barge_in_delay_ms=args.barge_in_delay_ms,
                    barge_in_timeout_ms=args.barge_in_timeout_ms,
                    real_time=args.simulate_realtime,
                    speedup=args.realtime_speedup,
                )
                output = await pipeline.run_turn(
                    request=request,
                    audio_stream=audio_stream,
                    on_audio_chunk=_on_audio_chunk,
                )
            else:
                audio_stream = simulate_live_stream(
                    chunks,
                    real_time=args.simulate_realtime,
                    speedup=args.realtime_speedup,
                )
                output = await pipeline.run_turn(request=request, audio_stream=audio_stream)

            row = output.latency.to_dict()
            row["audio_bytes"] = len(output.audio_bytes)
            row["source_id"] = sample.source_id
            row["barge_in_simulated"] = simulate_barge_in
            return row

    if args.warmup_calls > 0:
        for i in range(args.warmup_calls):
            await run_one(-(i + 1))

    rows = await asyncio.gather(*(run_one(i) for i in range(args.calls)))

    await _close_if_possible(asr)
    await _close_if_possible(llm)
    await _close_if_possible(tts)
    if shared_client is not None:
        await shared_client.aclose()

    summary = _summarize(rows, concurrency=args.concurrency)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "recommended_stack_live_pipeline",
        "dataset_source_used": source,
        "runtime_config": {
            "simulate_realtime": args.simulate_realtime,
            "realtime_speedup": args.realtime_speedup,
            "audio_chunk_ms": args.audio_chunk_ms,
            "trailing_silence_ms": args.trailing_silence_ms,
            "simulate_barge_in_rate": args.simulate_barge_in_rate,
            "barge_in_delay_ms": args.barge_in_delay_ms,
            "eou_mode": args.eou_mode,
            "aec_enabled": args.aec_enabled,
        },
        "stack": {
            "transport": args.transport,
            "vad": "silero-v4",
            "asr": args.asr_url,
            "asr_transcribe_path": args.asr_transcribe_path,
            "asr_model": args.asr_model,
            "llm": args.llm_model,
            "llm_base_url": args.llm_base_url,
            "tts": args.tts_url,
            "tts_mode": args.tts_mode,
            "tts_model": args.tts_model,
        },
        "summary": summary,
        "calls": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommended live voice pipeline runner")

    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--sample-pool", type=int, default=5)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--dataset-source", type=str, default="hf", choices=["hf", "kaggle", "tts", "mock"])
    parser.add_argument("--hf-dataset", type=str, default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--hf-config", type=str, default="clean")
    parser.add_argument("--hf-split", type=str, default="validation")

    parser.add_argument("--asr-url", type=str, default="http://127.0.0.1:8011")
    parser.add_argument("--asr-language", type=str, default="en")
    parser.add_argument("--asr-chunk-ms", type=int, default=160)
    parser.add_argument("--asr-transcribe-path", type=str, default="/transcribe")
    parser.add_argument("--asr-audio-field", type=str, default="audio_file")
    parser.add_argument("--asr-model", type=str, default="")
    parser.add_argument("--asr-api-key", type=str, default="")

    parser.add_argument("--llm-base-url", type=str, default="http://127.0.0.1:30000/v1")
    parser.add_argument("--llm-model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--llm-api-key", type=str, default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--llm-stream-mode", type=str, default="aggressive", choices=["sentence", "aggressive"])
    parser.add_argument("--llm-aggressive-min-tokens", type=int, default=5)

    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a compliant banking collection assistant. Keep replies concise and factual.",
    )
    parser.add_argument("--system-prompt-file", type=str, default="")

    parser.add_argument("--tts-url", type=str, default="http://127.0.0.1:8012")
    parser.add_argument("--tts-mode", type=str, default="http_synthesize", choices=["http_synthesize", "openai_audio_speech"])
    parser.add_argument("--tts-model", type=str, default="piper/en_US-lessac-medium")
    parser.add_argument("--tts-voice", type=str, default="Chelsie")
    parser.add_argument("--tts-split-granularity", type=str, default="sentence")
    parser.add_argument("--transport", type=str, default="http", choices=["http", "webrtc"])
    parser.add_argument("--asr-webrtc-offer-url", type=str, default="")
    parser.add_argument("--llm-webrtc-offer-url", type=str, default="")
    parser.add_argument("--tts-webrtc-offer-url", type=str, default="")

    parser.add_argument("--audio-chunk-ms", type=int, default=20)
    parser.add_argument("--trailing-silence-ms", type=int, default=400)
    parser.add_argument("--simulate-realtime", type=_parse_bool_arg, default=True)
    parser.add_argument("--realtime-speedup", type=float, default=1.0)
    parser.add_argument("--simulate-barge-in-rate", type=float, default=0.0)
    parser.add_argument("--barge-in-delay-ms", type=int, default=120)
    parser.add_argument("--barge-in-timeout-ms", type=int, default=5000)
    parser.add_argument("--barge-in-utterance-ms", type=int, default=900)

    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=200)
    parser.add_argument("--vad-min-silence-ms", type=int, default=300)
    parser.add_argument("--vad-streaming", type=_parse_bool_arg, default=True)
    parser.add_argument("--vad-use-chunk-flags", type=_parse_bool_arg, default=False)
    parser.add_argument("--vad-partial-segment-ms", type=int, default=320)
    parser.add_argument("--vad-max-stream-segment-ms", type=int, default=1280)

    parser.add_argument("--eou-mode", type=str, default="heuristic", choices=["off", "heuristic", "parakeet"])
    parser.add_argument("--eou-model", type=str, default="nvidia/parakeet_realtime_eou_120m-v1")
    parser.add_argument("--eou-min-words", type=int, default=3)
    parser.add_argument("--eou-min-chars", type=int, default=8)

    parser.add_argument("--aec-enabled", type=_parse_bool_arg, default=True)
    parser.add_argument("--aec-history-ms", type=int, default=1500)
    parser.add_argument("--aec-correlation-threshold", type=float, default=0.92)
    parser.add_argument("--aec-min-rms", type=float, default=0.003)

    parser.add_argument("--enable-barge-in", type=_parse_bool_arg, default=True)
    parser.add_argument("--barge-in-min-speech-ms", type=int, default=120)

    parser.add_argument("--asr-queue-size", type=int, default=2)
    parser.add_argument("--tts-queue-size", type=int, default=2)
    parser.add_argument("--http-timeout", type=float, default=120.0)

    parser.add_argument("--wait-for-health", action="store_true")
    parser.add_argument("--health-timeout", type=float, default=180.0)

    parser.add_argument(
        "--output",
        type=str,
        default=f"results/recommended_stack_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
    )
    return parser.parse_args()


def _save(report: dict[str, Any], output_path: str) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(target)


if __name__ == "__main__":
    args = parse_args()
    report = asyncio.run(run_recommended_pipeline(args))
    target = _save(report, args.output)
    print(json.dumps(report["summary"], indent=2))
    print(f"results_saved_to={target}")
