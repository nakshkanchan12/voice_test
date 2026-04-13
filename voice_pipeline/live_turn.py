from __future__ import annotations

import argparse
import asyncio
import wave
from pathlib import Path

from .asr_dataset import load_audio_samples
from .http_providers import HTTPASRProvider, HTTPTTSProvider
from .pipeline import StreamingVoicePipeline
from .real_llm import OpenAICompatibleLLM
from .real_vad import SileroVADProvider
from .types import AudioChunk, TurnRequest


def _pcm_to_chunks(pcm16: bytes, sample_rate_hz: int, chunk_ms: int = 20) -> list[AudioChunk]:
    bytes_per_sample = 2
    samples_per_chunk = int((sample_rate_hz * chunk_ms) / 1000)
    bytes_per_chunk = samples_per_chunk * bytes_per_sample

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
            )
        )

    # Add 400ms silence to trigger VAD endpoint.
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


def _write_pcm_wav(path: Path, pcm16: bytes, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm16)


async def run_live_turn(args: argparse.Namespace) -> None:
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
    chunks = _pcm_to_chunks(sample.pcm16, sample.sample_rate_hz)

    vad = SileroVADProvider(sample_rate_hz=sample.sample_rate_hz)
    asr = HTTPASRProvider(base_url=args.asr_url)
    llm = OpenAICompatibleLLM(
        base_url=args.llm_base_url,
        model=args.llm_model,
        system_prompt=args.system_prompt,
        api_key=args.llm_api_key,
    )
    tts = HTTPTTSProvider(base_url=args.tts_url)

    pipeline = StreamingVoicePipeline(vad=vad, asr=asr, llm=llm, tts=tts)

    request = TurnRequest(call_id="live-call-1", turn_id="live-turn-1")
    output = await pipeline.run_turn_from_chunks(request=request, chunks=chunks)

    output_pcm = output.audio_bytes
    output_path = Path(args.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pcm_wav(output_path, output_pcm, sample_rate_hz=args.tts_sample_rate_hz)

    print("dataset_source_used=", source)
    print("output_wav=", str(output_path))
    print("latency=", output.latency.to_dict())

    await asr.close()
    await tts.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real end-to-end pipeline turn via live endpoints.")
    parser.add_argument("--dataset-source", type=str, default="hf", choices=["hf", "kaggle", "tts", "mock"])
    parser.add_argument("--hf-dataset", type=str, default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--hf-config", type=str, default="clean")
    parser.add_argument("--hf-split", type=str, default="validation")

    parser.add_argument("--asr-url", type=str, default="http://127.0.0.1:8011")
    parser.add_argument("--llm-base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--llm-api-key", type=str, default="EMPTY")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a concise and compliant banking collection assistant.",
    )
    parser.add_argument("--tts-url", type=str, default="http://127.0.0.1:8012")
    parser.add_argument("--tts-sample-rate-hz", type=int, default=22050)
    parser.add_argument("--output-wav", type=str, default="results/live_turn_output.wav")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_live_turn(parse_args()))
