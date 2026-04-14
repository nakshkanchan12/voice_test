from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .types import SpeechSegment, TurnRequest


class EndOfUtteranceDecider(Protocol):
    name: str

    async def is_end_of_utterance(
        self,
        transcript: str,
        *,
        request: TurnRequest,
        segment: SpeechSegment,
    ) -> bool:
        ...


@dataclass
class HeuristicEOUDecider:
    min_words: int = 3
    min_chars: int = 8
    name: str = "heuristic"

    async def is_end_of_utterance(
        self,
        transcript: str,
        *,
        request: TurnRequest,
        segment: SpeechSegment,
    ) -> bool:
        del request
        text = transcript.strip()
        if not text:
            return False

        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) < self.min_chars:
            return False

        words = [token for token in normalized.split(" ") if token]
        if len(words) < self.min_words:
            return False

        lowered = normalized.lower()
        if lowered.endswith((" and", " but", " because", " so", " then", " if", " uh", " um")):
            return False

        if normalized.endswith((".", "?", "!")):
            return True

        if len(words) >= (self.min_words + 6):
            return True

        return bool(segment.ended_by_silence)


@dataclass
class ParakeetEOUDecider:
    model_name: str = "nvidia/parakeet_realtime_eou_120m-v1"
    fallback: EndOfUtteranceDecider = field(default_factory=HeuristicEOUDecider)
    name: str = "parakeet"

    def __post_init__(self) -> None:
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        from nemo.collections.asr.models import EncDecCTCModelBPE  # type: ignore

        self._model = EncDecCTCModelBPE.from_pretrained(self.model_name)
        return self._model

    @staticmethod
    def _coerce_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) >= 0.5
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "done", "complete", "eou"}:
                return True
            if token in {"0", "false", "no", "continue", "incomplete"}:
                return False
            return None
        if isinstance(value, dict):
            for key in ("is_end", "eou", "done", "is_done", "complete"):
                if key in value:
                    return ParakeetEOUDecider._coerce_bool(value[key])
            return None
        if isinstance(value, (list, tuple)) and value:
            return ParakeetEOUDecider._coerce_bool(value[0])
        return None

    def _predict_with_model(self, transcript: str) -> bool | None:
        model = self._ensure_model()
        for method_name, call in (
            ("predict", lambda fn: fn(transcript)),
            ("predict", lambda fn: fn([transcript])),
            ("infer", lambda fn: fn(transcript)),
            ("infer", lambda fn: fn([transcript])),
        ):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                raw = call(method)
            except Exception:
                continue

            decision = self._coerce_bool(raw)
            if decision is not None:
                return decision

        return None

    async def is_end_of_utterance(
        self,
        transcript: str,
        *,
        request: TurnRequest,
        segment: SpeechSegment,
    ) -> bool:
        try:
            decision = self._predict_with_model(transcript)
            if decision is not None:
                return decision
        except Exception:
            pass

        return await self.fallback.is_end_of_utterance(
            transcript,
            request=request,
            segment=segment,
        )


def build_eou_decider(
    mode: str,
    *,
    model_name: str = "nvidia/parakeet_realtime_eou_120m-v1",
    min_words: int = 3,
    min_chars: int = 8,
) -> EndOfUtteranceDecider | None:
    normalized = mode.strip().lower()
    if normalized in {"", "off", "none", "disabled"}:
        return None

    heuristic = HeuristicEOUDecider(min_words=max(1, min_words), min_chars=max(1, min_chars))
    if normalized in {"heuristic", "simple"}:
        return heuristic
    if normalized in {"parakeet", "nemo"}:
        return ParakeetEOUDecider(model_name=model_name, fallback=heuristic)

    raise ValueError(f"Unsupported EOU mode: {mode}")