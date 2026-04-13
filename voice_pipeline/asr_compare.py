from __future__ import annotations

import asyncio
import gc
import json
import math
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from jiwer import wer

from .asr_dataset import AudioTranscriptSample, load_audio_samples


@dataclass(frozen=True)
class ASRCompareCandidate:
    name: str
    model_id: str
    compute_type: str = "int8_float16"
    beam_size: int = 1


DEFAULT_CANDIDATES = [
    ASRCompareCandidate(
        name="whisper-tiny-int8-beam1",
        model_id="tiny",
        compute_type="int8_float16",
        beam_size=1,
    ),
    ASRCompareCandidate(
        name="whisper-large-v3-turbo-int8-beam1",
        model_id="large-v3-turbo",
        compute_type="int8_float16",
        beam_size=1,
    ),
    ASRCompareCandidate(
        name="whisper-medium-int8-beam1",
        model_id="medium",
        compute_type="int8_float16",
        beam_size=1,
    ),
    ASRCompareCandidate(
        name="whisper-small-int8-beam1",
        model_id="small",
        compute_type="int8_float16",
        beam_size=1,
    ),
]


@dataclass(frozen=True)
class ASRCompareConfig:
    dataset_source: str = "hf"
    sample_count: int = 12
    wer_threshold_pct: float = 5.0
    device: str = "cuda"
    hf_dataset: str = "hf-internal-testing/librispeech_asr_dummy"
    hf_config: str = "clean"
    hf_split: str = "validation"
    candidate_models: tuple[str, ...] = ("tiny", "small", "medium")
    model_load_timeout_s: float = 240.0
    sample_timeout_s: float = 120.0


def _pct(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    if len(values) == 1:
        return round(values[0], 2)

    rank = (len(values) - 1) * (percentile / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(values[low], 2)

    low_v = values[low]
    high_v = values[high]
    return round(low_v + (high_v - low_v) * (rank - low), 2)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized != "auto":
        return normalized

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_compute_type(device: str, desired: str) -> str:
    if device == "cpu" and desired == "int8_float16":
        return "int8"
    return desired


def _candidate_list(candidate_models: tuple[str, ...]) -> list[ASRCompareCandidate]:
    by_model = {candidate.model_id: candidate for candidate in DEFAULT_CANDIDATES}
    resolved: list[ASRCompareCandidate] = []

    for model_id in candidate_models:
        candidate = by_model.get(model_id)
        if candidate is not None:
            resolved.append(candidate)

    if resolved:
        return resolved
    return list(DEFAULT_CANDIDATES)


def _write_temp_wav(sample: AudioTranscriptSample) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = Path(handle.name)

    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample.sample_rate_hz)
        wav.writeframes(sample.pcm16)

    return wav_path


def _transcribe_file(
    model: Any,
    wav_path: Path,
    beam_size: int,
) -> str:
    segments, _info = model.transcribe(
        str(wav_path),
        beam_size=beam_size,
        best_of=1,
        word_timestamps=False,
    )
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


async def _evaluate_candidate(
    candidate: ASRCompareCandidate,
    samples: list[AudioTranscriptSample],
    device: str,
    model_load_timeout_s: float,
    sample_timeout_s: float,
) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        return {
            "model": candidate.name,
            "model_id": candidate.model_id,
            "status": "error",
            "error": f"faster-whisper missing: {exc}",
        }

    compute_type = _resolve_compute_type(device=device, desired=candidate.compute_type)

    loop = asyncio.get_running_loop()

    def _load_model() -> Any:
        return WhisperModel(candidate.model_id, device=device, compute_type=compute_type)

    try:
        model = await asyncio.wait_for(
            loop.run_in_executor(None, _load_model),
            timeout=max(1.0, model_load_timeout_s),
        )
    except asyncio.TimeoutError:
        return {
            "model": candidate.name,
            "model_id": candidate.model_id,
            "device": device,
            "compute_type": compute_type,
            "beam_size": candidate.beam_size,
            "status": "timeout",
            "error": f"model load exceeded {model_load_timeout_s:.1f}s",
        }
    except Exception as exc:
        return {
            "model": candidate.name,
            "model_id": candidate.model_id,
            "device": device,
            "compute_type": compute_type,
            "beam_size": candidate.beam_size,
            "status": "error",
            "error": f"model load failed: {exc}",
        }

    latencies: list[float] = []
    wers: list[float] = []

    # Warm-up is excluded from metrics to avoid one-time kernel and cache cost skew.
    warmup_path = _write_temp_wav(samples[0])
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, _transcribe_file, model, warmup_path, candidate.beam_size),
            timeout=max(1.0, sample_timeout_s),
        )
    finally:
        warmup_path.unlink(missing_ok=True)

    for sample in samples:
        wav_path = _write_temp_wav(sample)
        try:
            started = perf_counter()
            hypothesis = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    _transcribe_file,
                    model,
                    wav_path,
                    candidate.beam_size,
                ),
                timeout=max(1.0, sample_timeout_s),
            )
            ended = perf_counter()

            ref_text = _normalize_text(sample.transcript)
            hyp_text = _normalize_text(hypothesis)
            latencies.append((ended - started) * 1000.0)
            wers.append(float(wer(ref_text, hyp_text)) * 100.0)
        finally:
            wav_path.unlink(missing_ok=True)

    row = {
        "model": candidate.name,
        "model_id": candidate.model_id,
        "device": device,
        "compute_type": compute_type,
        "beam_size": candidate.beam_size,
        "samples": len(samples),
        "latency_p50_ms": _pct(latencies, 50),
        "latency_p95_ms": _pct(latencies, 95),
        "latency_mean_ms": round(mean(latencies), 2),
        "wer_pct": round(mean(wers), 2),
        "status": "ok",
    }

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return row


def _pick_winner(rows: list[dict[str, Any]], wer_threshold_pct: float) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return {
            "winner": "",
            "reason": "No successful model evaluations",
            "backend": "faster-whisper",
            "model_id": "",
        }

    passing = [row for row in ok_rows if float(row["wer_pct"]) <= wer_threshold_pct]
    ranked = passing if passing else ok_rows

    winner = sorted(ranked, key=lambda row: (float(row["latency_p50_ms"]), float(row["wer_pct"])))[0]
    reason = (
        f"Selected fastest model under WER threshold {wer_threshold_pct:.2f}%"
        if passing
        else "No model met WER threshold; selected lowest latency among successful runs"
    )

    return {
        "winner": winner["model"],
        "reason": reason,
        "backend": "faster-whisper",
        "model_id": winner["model_id"],
        "device": winner["device"],
        "compute_type": winner["compute_type"],
        "beam_size": winner["beam_size"],
    }


async def compare_asr_models(config: ASRCompareConfig) -> dict[str, Any]:
    samples, source_used = load_audio_samples(
        source=config.dataset_source,
        limit=max(1, config.sample_count),
        hf_dataset=config.hf_dataset,
        hf_config=config.hf_config,
        hf_split=config.hf_split,
    )

    resolved_device = _resolve_device(config.device)

    if not samples:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": "asr_model_compare_real",
            "dataset_source_requested": config.dataset_source,
            "dataset_source_used": source_used,
            "sample_count": 0,
            "wer_threshold_pct": config.wer_threshold_pct,
            "results": [],
            "selection": {
                "winner": "",
                "reason": "No audio samples available",
                "backend": "faster-whisper",
                "model_id": "",
            },
        }

    candidates = _candidate_list(config.candidate_models)
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        row = await _evaluate_candidate(
            candidate,
            samples=samples,
            device=resolved_device,
            model_load_timeout_s=config.model_load_timeout_s,
            sample_timeout_s=config.sample_timeout_s,
        )
        rows.append(row)

    rows = sorted(
        rows,
        key=lambda row: (
            1 if row.get("status") != "ok" else 0,
            float(row.get("latency_p50_ms", 9e9)),
            float(row.get("wer_pct", 9e9)),
        ),
    )

    selection = _pick_winner(rows=rows, wer_threshold_pct=config.wer_threshold_pct)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "asr_model_compare_real",
        "dataset_source_requested": config.dataset_source,
        "dataset_source_used": source_used,
        "sample_count": len(samples),
        "wer_threshold_pct": config.wer_threshold_pct,
        "device": resolved_device,
        "results": rows,
        "selection": selection,
    }


def save_compare_report(report: dict[str, Any], output_path: str | None = None) -> str:
    if output_path:
        target = Path(output_path)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = Path("results") / f"asr_compare_{stamp}.json"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(target)


def save_selected_model(report: dict[str, Any], output_path: str = "configs/selected_asr.json") -> str:
    selection = report.get("selection", {})
    winner = str(selection.get("winner", "")).strip()
    model_id = str(selection.get("model_id", "")).strip()

    if not winner or not model_id:
        raise ValueError("No selected winner in report")

    payload = {
        "selected_model": winner,
        "backend": str(selection.get("backend", "faster-whisper")),
        "model_id": model_id,
        "device": str(selection.get("device", "cuda")),
        "compute_type": str(selection.get("compute_type", "int8_float16")),
        "beam_size": int(selection.get("beam_size", 1)),
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "source_report": report.get("benchmark", "asr_model_compare_real"),
        "dataset_source_used": report.get("dataset_source_used", "unknown"),
        "wer_threshold_pct": report.get("wer_threshold_pct", 5.0),
    }

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(target)


def load_selected_model(path: str = "configs/selected_asr.json") -> str | None:
    target = Path(path)
    if not target.exists():
        return None

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None

    model_id = str(payload.get("model_id", "")).strip()
    if model_id:
        return model_id

    selected = str(payload.get("selected_model", "")).strip()
    return selected or None
