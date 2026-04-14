from .asr_compare import (
    ASRCompareConfig,
    compare_asr_models,
    save_compare_report,
    save_selected_model,
)
from .benchmark import run_load_benchmark, save_results
from .factory import resolve_selected_asr_model
from .pipeline import StreamingVoicePipeline, TurnOutput
from .recommended_providers import (
    NemotronHTTPASRProvider,
    Qwen3TTSProvider,
    SGLangOpenAILLMProvider,
)
from .types import AudioChunk, SpeechSegment, TurnRequest


async def run_recommended_pipeline(*args, **kwargs):
    # Lazy import keeps package import light and avoids runpy warnings when
    # running voice_pipeline.recommended_pipeline as a module.
    from .recommended_pipeline import run_recommended_pipeline as _run_recommended_pipeline

    return await _run_recommended_pipeline(*args, **kwargs)

__all__ = [
    "ASRCompareConfig",
    "compare_asr_models",
    "save_compare_report",
    "save_selected_model",
    "run_load_benchmark",
    "run_recommended_pipeline",
    "save_results",
    "resolve_selected_asr_model",
    "NemotronHTTPASRProvider",
    "SGLangOpenAILLMProvider",
    "Qwen3TTSProvider",
    "StreamingVoicePipeline",
    "TurnOutput",
    "AudioChunk",
    "SpeechSegment",
    "TurnRequest",
]
