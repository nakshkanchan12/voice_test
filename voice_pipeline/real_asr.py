from __future__ import annotations

import asyncio
import tempfile
import wave
from pathlib import Path
from typing import Any

from .types import SpeechSegment, TurnRequest


class FasterWhisperASR:
    """Optional real ASR provider. Requires faster-whisper to be installed."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "int8_float16",
        beam_size: int = 1,
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "faster-whisper is not installed. Install it to use real ASR mode."
            ) from exc

        resolved_device = self._resolve_device(device)
        resolved_compute_type = self._resolve_compute_type(resolved_device, compute_type)

        self._model = WhisperModel(
            model_name,
            device=resolved_device,
            compute_type=resolved_compute_type,
        )
        self._beam_size = beam_size

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = device.strip().lower()
        if normalized != "auto":
            return normalized

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def _resolve_compute_type(device: str, compute_type: str) -> str:
        if device == "cpu" and compute_type == "int8_float16":
            return "int8"
        return compute_type

    async def transcribe(self, segment: SpeechSegment, request: TurnRequest) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, segment)

    def _transcribe_sync(self, segment: SpeechSegment) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        try:
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(segment.sample_rate_hz)
                handle.writeframes(b"".join(chunk.pcm for chunk in segment.chunks))

            segments, _info = self._model.transcribe(
                str(wav_path),
                beam_size=self._beam_size,
                best_of=1,
                word_timestamps=False,
            )
            text = " ".join(item.text.strip() for item in segments if item.text.strip())
            return text.strip()
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass
