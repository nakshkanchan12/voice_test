from __future__ import annotations

import argparse
import asyncio
import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .aec import ReferenceEchoCanceller
from .asr_dataset import load_audio_samples
from .eou import build_eou_decider
from .http_providers import HTTPASRProvider, HTTPTTSProvider
from .live_simulation import (
    pcm_to_chunks,
    simulate_interruptible_live_stream,
    simulate_live_stream,
    trim_chunks,
)
from .pipeline import StreamingVoicePipeline
from .real_llm import OpenAICompatibleLLM
from .real_vad import SileroVADProvider
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


def _record_microphone_audio(
    seconds: float,
    sample_rate_hz: int,
    channels: int,
    device: str,
) -> tuple[bytes, int]:
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Microphone capture requires sounddevice. Install with 'pip install sounddevice' "
            "and ensure PortAudio is available on your system."
        ) from exc

    duration_s = max(0.2, float(seconds))
    sample_rate = max(8000, int(sample_rate_hz))
    channel_count = max(1, int(channels))

    selected_device: int | str | None = None
    token = device.strip()
    if token:
        if token.lstrip("-").isdigit():
            selected_device = int(token)
        else:
            selected_device = token

    frame_count = int(duration_s * sample_rate)
    print(f"recording_from_mic_for_seconds={duration_s}")
    print("speak_now=true")
    recording = sd.rec(
        frame_count,
        samplerate=sample_rate,
        channels=channel_count,
        dtype="float32",
        device=selected_device,
    )
    sd.wait()

    mono = np.asarray(recording, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)

    pcm = np.clip(mono, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()
    return pcm16, sample_rate


def _write_pcm_wav(path: Path, pcm16: bytes, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm16)


def _save_latency(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


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


def _safe_slug(value: str) -> str:
    cleaned = [char if (char.isalnum() or char in {"-", "_"}) else "_" for char in value.strip()]
    slug = "".join(cleaned).strip("_")
    return slug or "live_call"


def _write_metrics_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _resolve_turn_output_path(base_output_path: Path, turn_index: int, multi_turn: bool) -> Path:
    if not multi_turn:
        return base_output_path

    suffix = base_output_path.suffix or ".wav"
    stem = base_output_path.stem or "live_turn_output"
    return base_output_path.with_name(f"{stem}_turn_{turn_index:03d}{suffix}")


def _summarize_call_metrics(rows: list[dict[str, Any]], call_id: str, source: str, transport: str) -> dict[str, Any]:
    s2a = [_to_float(row.get("speech_to_first_audio_ms")) for row in rows]
    s2a_values = [value for value in s2a if value is not None]
    mic2a = [_to_float(row.get("mic_to_first_audio_ms")) for row in rows]
    mic2a_values = [value for value in mic2a if value is not None]
    s2d = [_to_float(row.get("speech_to_done_ms")) for row in rows]
    s2d_values = [value for value in s2d if value is not None]
    e2e = [_to_float(row.get("e2e_ms")) for row in rows]
    e2e_values = [value for value in e2e if value is not None]

    queue_asr = [_to_float(row.get("asr_queue_wait_ms")) for row in rows]
    queue_asr_values = [value for value in queue_asr if value is not None]
    queue_tts = [_to_float(row.get("tts_queue_wait_ms")) for row in rows]
    queue_tts_values = [value for value in queue_tts if value is not None]

    cancelled_turns = sum(1 for row in rows if bool(row.get("cancelled_by_barge_in")))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "call_id": call_id,
        "dataset_source_used": source,
        "transport": transport,
        "turns": len(rows),
        "speech_to_first_audio_p50_ms": _percentile(s2a_values, 50),
        "speech_to_first_audio_p95_ms": _percentile(s2a_values, 95),
        "mic_to_first_audio_p50_ms": _percentile(mic2a_values, 50),
        "mic_to_first_audio_p95_ms": _percentile(mic2a_values, 95),
        "speech_to_done_p50_ms": _percentile(s2d_values, 50),
        "speech_to_done_p95_ms": _percentile(s2d_values, 95),
        "e2e_p50_ms": _percentile(e2e_values, 50),
        "e2e_p95_ms": _percentile(e2e_values, 95),
        "mean_asr_queue_wait_ms": round(sum(queue_asr_values) / max(1, len(queue_asr_values)), 2),
        "mean_tts_queue_wait_ms": round(sum(queue_tts_values) / max(1, len(queue_tts_values)), 2),
        "lt_700ms_first_audio_hit_rate_pct": round(
            (sum(1 for value in s2a_values if value < 700.0) / max(1, len(s2a_values))) * 100.0,
            2,
        ),
        "lt_800ms_first_audio_hit_rate_pct": round(
            (sum(1 for value in s2a_values if value < 800.0) / max(1, len(s2a_values))) * 100.0,
            2,
        ),
        "barge_in_cancelled_turns": cancelled_turns,
    }


async def run_live_turn(args: argparse.Namespace) -> None:
    if args.interactive_call and args.dataset_source != "mic":
        raise RuntimeError("--interactive-call requires --dataset-source mic")

    call_id = args.call_id.strip() or "live-call-1"

    input_sample_rate_hz: int
    source_pcm16: bytes | None = None
    source: str

    if args.dataset_source == "mic":
        source = "mic"
        if args.interactive_call:
            input_sample_rate_hz = max(8000, int(args.mic_sample_rate_hz))
        else:
            source_pcm16, input_sample_rate_hz = _record_microphone_audio(
                seconds=args.mic_seconds,
                sample_rate_hz=args.mic_sample_rate_hz,
                channels=args.mic_channels,
                device=args.mic_device,
            )
    else:
        samples, source = load_audio_samples(
            source=args.dataset_source,
            limit=1,
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
        )
        if not samples:
            raise RuntimeError("No input audio sample available for live turn")

        sample = samples[0]
        source_pcm16 = sample.pcm16
        input_sample_rate_hz = sample.sample_rate_hz

    vad = SileroVADProvider(
        sample_rate_hz=input_sample_rate_hz,
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

        asr = WebRTCASRProvider(offer_url=asr_offer_url)
        llm = WebRTCLLMProvider(
            offer_url=llm_offer_url,
            model=args.llm_model,
            system_prompt=args.system_prompt,
            stream_mode=args.llm_stream_mode,
            aggressive_min_tokens=args.llm_aggressive_min_tokens,
        )
        tts = WebRTCTTSProvider(offer_url=tts_offer_url)
    else:
        asr = HTTPASRProvider(base_url=args.asr_url)
        llm = OpenAICompatibleLLM(
            base_url=args.llm_base_url,
            model=args.llm_model,
            system_prompt=args.system_prompt,
            api_key=args.llm_api_key,
            stream_mode=args.llm_stream_mode,
            aggressive_min_tokens=args.llm_aggressive_min_tokens,
        )
        tts = HTTPTTSProvider(base_url=args.tts_url)

    pipeline = StreamingVoicePipeline(
        vad=vad,
        asr=asr,
        llm=llm,
        tts=tts,
        eou_decider=eou_decider,
        aec_factory=aec_factory,
    )

    base_output_path = Path(args.output_wav)
    base_output_path.parent.mkdir(parents=True, exist_ok=True)

    async def _run_turn_from_pcm(
        pcm16: bytes,
        sample_rate_hz: int,
        turn_index: int,
        multi_turn: bool,
        source_label: str,
    ) -> dict[str, Any]:
        chunks = pcm_to_chunks(
            pcm16,
            sample_rate_hz,
            chunk_ms=args.audio_chunk_ms,
            trailing_silence_ms=args.trailing_silence_ms,
        )
        barge_in_chunks = trim_chunks(
            pcm_to_chunks(
                pcm16,
                sample_rate_hz,
                chunk_ms=args.audio_chunk_ms,
                trailing_silence_ms=args.trailing_silence_ms,
            ),
            max_duration_ms=args.barge_in_utterance_ms,
        )

        request = TurnRequest(
            call_id=call_id,
            turn_id=f"live-turn-{turn_index}",
            llm_stream_mode=args.llm_stream_mode,
            enable_barge_in=args.enable_barge_in,
            barge_in_min_speech_ms=args.barge_in_min_speech_ms,
        )

        if args.simulate_barge_in:
            tts_started = asyncio.Event()

            async def _on_audio_chunk(_: bytes) -> None:
                tts_started.set()

            audio_stream = simulate_interruptible_live_stream(
                chunks,
                barge_in_chunks,
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

        turn_output_path = _resolve_turn_output_path(base_output_path, turn_index=turn_index, multi_turn=multi_turn)
        turn_output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pcm_wav(turn_output_path, output.audio_bytes, sample_rate_hz=args.tts_sample_rate_hz)

        row = output.latency.to_dict()
        row["turn_index"] = turn_index
        row["dataset_source_used"] = source_label
        row["transport"] = args.transport
        row["output_wav"] = str(turn_output_path)
        return row

    try:
        metrics_jsonl_path: Path | None = Path(args.metrics_jsonl) if args.metrics_jsonl else None

        if args.interactive_call:
            slug = _safe_slug(call_id)
            if metrics_jsonl_path is None:
                metrics_jsonl_path = Path("results") / f"live_call_metrics_{slug}.jsonl"

            metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_jsonl_path.write_text("", encoding="utf-8")
            print("interactive_call=true")
            print("call_metrics_jsonl=", str(metrics_jsonl_path))
            if args.max_turns <= 0:
                print("max_turns=unbounded")

            rows: list[dict[str, Any]] = []
            turn_index = 1
            while args.max_turns <= 0 or turn_index <= args.max_turns:
                print("turn_index=", turn_index)
                try:
                    turn_pcm16, turn_sample_rate_hz = _record_microphone_audio(
                        seconds=args.mic_seconds,
                        sample_rate_hz=args.mic_sample_rate_hz,
                        channels=args.mic_channels,
                        device=args.mic_device,
                    )
                    row = await _run_turn_from_pcm(
                        turn_pcm16,
                        turn_sample_rate_hz,
                        turn_index=turn_index,
                        multi_turn=True,
                        source_label=source,
                    )
                except KeyboardInterrupt:
                    print("call_interrupted=true")
                    break

                rows.append(row)
                _write_metrics_jsonl(metrics_jsonl_path, row)

                print("turn_output_wav=", row["output_wav"])
                print("turn_latency=", row)
                turn_index += 1

            summary_path = (
                Path(args.metrics_summary_json)
                if args.metrics_summary_json
                else Path("results") / f"live_call_summary_{slug}.json"
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_payload = _summarize_call_metrics(rows, call_id=call_id, source=source, transport=args.transport)
            summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
            print("call_metrics_summary_json=", str(summary_path))
            return

        if source_pcm16 is None:
            raise RuntimeError("No source PCM available for single-turn execution")

        row = await _run_turn_from_pcm(
            source_pcm16,
            input_sample_rate_hz,
            turn_index=1,
            multi_turn=False,
            source_label=source,
        )

        if metrics_jsonl_path is not None:
            _write_metrics_jsonl(metrics_jsonl_path, row)

        if args.metrics_summary_json:
            summary_path = Path(args.metrics_summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_payload = _summarize_call_metrics([row], call_id=call_id, source=source, transport=args.transport)
            summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
            print("call_metrics_summary_json=", str(summary_path))

        print("dataset_source_used=", source)
        print("transport=", args.transport)
        print("output_wav=", row["output_wav"])
        print("eou_mode=", args.eou_mode)
        print("aec_enabled=", args.aec_enabled)
        print("latency=", row)

        if args.latency_output:
            latency_path = Path(args.latency_output)
            _save_latency(latency_path, row)
            print("latency_output=", str(latency_path))
    finally:
        await _close_if_possible(asr)
        await _close_if_possible(llm)
        await _close_if_possible(tts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real end-to-end pipeline turn via live endpoints.")
    parser.add_argument("--dataset-source", type=str, default="hf", choices=["hf", "kaggle", "tts", "mock", "mic"])
    parser.add_argument("--hf-dataset", type=str, default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--hf-config", type=str, default="clean")
    parser.add_argument("--hf-split", type=str, default="validation")
    parser.add_argument("--audio-chunk-ms", type=int, default=20)
    parser.add_argument("--trailing-silence-ms", type=int, default=400)
    parser.add_argument("--simulate-realtime", type=_parse_bool_arg, default=True)
    parser.add_argument("--realtime-speedup", type=float, default=1.0)
    parser.add_argument("--simulate-barge-in", type=_parse_bool_arg, default=False)
    parser.add_argument("--barge-in-delay-ms", type=int, default=120)
    parser.add_argument("--barge-in-timeout-ms", type=int, default=5000)
    parser.add_argument("--barge-in-utterance-ms", type=int, default=900)
    parser.add_argument("--mic-seconds", type=float, default=5.0)
    parser.add_argument("--mic-sample-rate-hz", type=int, default=16000)
    parser.add_argument("--mic-channels", type=int, default=1)
    parser.add_argument("--mic-device", type=str, default="")
    parser.add_argument("--call-id", type=str, default="live-call-1")
    parser.add_argument("--interactive-call", type=_parse_bool_arg, default=False)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--metrics-jsonl", type=str, default="")
    parser.add_argument("--metrics-summary-json", type=str, default="")

    parser.add_argument("--asr-url", type=str, default="http://127.0.0.1:8011")
    parser.add_argument("--llm-base-url", type=str, default="http://127.0.0.1:30000/v1")
    parser.add_argument("--llm-model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--llm-api-key", type=str, default="EMPTY")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a concise and compliant banking collection assistant.",
    )
    parser.add_argument("--llm-stream-mode", type=str, default="aggressive", choices=["sentence", "aggressive"])
    parser.add_argument("--llm-aggressive-min-tokens", type=int, default=5)
    parser.add_argument("--tts-url", type=str, default="http://127.0.0.1:8012")
    parser.add_argument("--transport", type=str, default="http", choices=["http", "webrtc"])
    parser.add_argument("--asr-webrtc-offer-url", type=str, default="")
    parser.add_argument("--llm-webrtc-offer-url", type=str, default="")
    parser.add_argument("--tts-webrtc-offer-url", type=str, default="")
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
    parser.add_argument("--tts-sample-rate-hz", type=int, default=22050)
    parser.add_argument("--output-wav", type=str, default="results/live_turn_output.wav")
    parser.add_argument("--latency-output", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_live_turn(parse_args()))
