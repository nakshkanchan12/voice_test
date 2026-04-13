from __future__ import annotations

import json
import os
from pathlib import Path

from .asr_catalog import ASR_MODEL_PROFILES, DEFAULT_ASR_MODEL
from .http_providers import HTTPASRProvider
from .mock_components import MockASR
from .real_asr import FasterWhisperASR


def resolve_selected_asr_model(path: str = "configs/selected_asr.json") -> str:
    target = Path(path)
    if not target.exists():
        return DEFAULT_ASR_MODEL

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_ASR_MODEL

    selected = str(payload.get("model_id", "") or payload.get("selected_model", "")).strip()
    if selected in ASR_MODEL_PROFILES:
        return selected

    # Real model ids (e.g., "large-v3-turbo") are allowed and should pass through.
    if selected:
        return selected

    return DEFAULT_ASR_MODEL


def load_selected_asr_config(path: str = "configs/selected_asr.json") -> dict[str, object]:
    target = Path(path)
    if not target.exists():
        return {}

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}
    return payload


def build_asr_provider(
    mode: str,
    selected_model: str | None = None,
):
    normalized = mode.strip().lower()
    model_name = selected_model or resolve_selected_asr_model()
    selected_config = load_selected_asr_config()

    if normalized == "real-faster-whisper":
        configured_device = str(selected_config.get("device", "auto"))
        configured_compute_type = str(selected_config.get("compute_type", "int8_float16"))
        configured_beam = int(selected_config.get("beam_size", 1))
        return FasterWhisperASR(
            model_name=model_name,
            device=configured_device,
            compute_type=configured_compute_type,
            beam_size=configured_beam,
        )

    if normalized == "http-asr":
        base_url = os.getenv("ASR_SERVER_URL", "http://127.0.0.1:8011")
        return HTTPASRProvider(base_url=base_url)

    return MockASR(model_name=model_name)
