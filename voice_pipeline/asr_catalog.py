from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ASRModelProfile:
    name: str
    latency_base_ms: float
    latency_per_second_ms: float
    simulated_wer_pct: float


ASR_MODEL_PROFILES: dict[str, ASRModelProfile] = {
    "nemotron-fastconformer-160ms": ASRModelProfile(
        name="nemotron-fastconformer-160ms",
        latency_base_ms=80.0,
        latency_per_second_ms=22.0,
        simulated_wer_pct=3.9,
    ),
    "whisper-large-v3-turbo-int8-beam1": ASRModelProfile(
        name="whisper-large-v3-turbo-int8-beam1",
        latency_base_ms=150.0,
        latency_per_second_ms=35.0,
        simulated_wer_pct=4.8,
    ),
    "whisper-medium-int8-beam1": ASRModelProfile(
        name="whisper-medium-int8-beam1",
        latency_base_ms=170.0,
        latency_per_second_ms=32.0,
        simulated_wer_pct=5.6,
    ),
    "parakeet-tdt-1.1b": ASRModelProfile(
        name="parakeet-tdt-1.1b",
        latency_base_ms=142.0,
        latency_per_second_ms=30.0,
        simulated_wer_pct=6.2,
    ),
}

DEFAULT_ASR_MODEL = "whisper-large-v3-turbo-int8-beam1"
