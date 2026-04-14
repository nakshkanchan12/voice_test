from __future__ import annotations

from collections import deque

import numpy as np

from .types import SpeechSegment


class ReferenceEchoCanceller:
    """Best-effort echo suppressor using recent TTS audio as reference."""

    def __init__(
        self,
        history_ms: int = 1500,
        chunk_ms: int = 20,
        correlation_threshold: float = 0.92,
        min_rms: float = 0.003,
    ) -> None:
        frame_count = max(1, int(max(1, history_ms) / max(1, chunk_ms)))
        self._recent_refs: deque[np.ndarray] = deque(maxlen=frame_count)
        self.correlation_threshold = correlation_threshold
        self.min_rms = min_rms

    @staticmethod
    def _pcm_to_float(pcm: bytes) -> np.ndarray:
        if not pcm:
            return np.empty((0,), dtype=np.float32)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return samples / 32768.0

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        n = min(a.size, b.size)
        if n < 32:
            return 0.0

        xa = a[-n:].astype(np.float32, copy=False)
        xb = b[-n:].astype(np.float32, copy=False)
        xa = xa - float(np.mean(xa))
        xb = xb - float(np.mean(xb))
        denom = float(np.linalg.norm(xa) * np.linalg.norm(xb))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(xa, xb) / denom)

    def register_tts_audio(self, pcm: bytes) -> None:
        frame = self._pcm_to_float(pcm)
        if frame.size < 32:
            return
        self._recent_refs.append(frame)

    def is_echo_segment(self, segment: SpeechSegment) -> bool:
        if not self._recent_refs or not segment.chunks:
            return False

        probe = self._pcm_to_float(segment.chunks[-1].pcm)
        if probe.size < 32:
            return False

        rms = float(np.sqrt(np.mean(np.square(probe))))
        if rms < self.min_rms:
            return False

        for ref in reversed(self._recent_refs):
            if self._corr(probe, ref) >= self.correlation_threshold:
                return True

        return False