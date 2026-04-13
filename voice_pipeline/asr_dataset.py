from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioTranscriptSample:
    pcm16: bytes
    sample_rate_hz: int
    transcript: str
    source_id: str


def load_audio_samples(
    source: str,
    limit: int,
    hf_dataset: str = "hf-internal-testing/librispeech_asr_dummy",
    hf_config: str = "clean",
    hf_split: str = "validation",
) -> tuple[list[AudioTranscriptSample], str]:
    normalized = source.strip().lower()

    if normalized == "hf":
        samples = _load_from_hf(limit=limit, dataset_name=hf_dataset, config_name=hf_config, split=hf_split)
        if samples:
            return samples, "hf"
        return [], "hf-empty"

    if normalized == "kaggle":
        samples = _load_from_kaggle(limit=limit)
        if samples:
            return samples, "kaggle"
        return [], "kaggle-empty"

    if normalized == "mock":
        return _load_builtin_demo(limit=limit), "builtin-demo"

    # "tts" is currently supported via a user-provided dataset manifest created from TTS audio.
    if normalized == "tts":
        samples = _load_from_kaggle(limit=limit)
        if samples:
            return samples, "tts-manifest"
        return [], "tts-empty"

    samples = _load_from_kaggle(limit=limit)
    if samples:
        return samples, "local-manifest"

    return _load_builtin_demo(limit=limit), "builtin-demo"


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    scaled = (clipped * 32767.0).astype(np.int16)
    return scaled.tobytes()


def _read_audio_file(path: Path) -> tuple[bytes, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)

    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)

    return _float_to_pcm16(audio), int(sample_rate)


def _load_from_hf(
    limit: int,
    dataset_name: str,
    config_name: str,
    split: str,
) -> list[AudioTranscriptSample]:
    try:
        from datasets import Audio, load_dataset  # type: ignore
    except Exception:
        return []

    try:
        dataset = load_dataset(dataset_name, config_name, split=f"{split}[:{limit}]")
    except Exception:
        return []

    if "audio" not in dataset.column_names:
        return []

    transcript_col = "text" if "text" in dataset.column_names else "sentence"
    if transcript_col not in dataset.column_names:
        return []

    dataset = dataset.cast_column("audio", Audio(decode=False))

    samples: list[AudioTranscriptSample] = []
    for index, row in enumerate(dataset):
        audio_obj = row.get("audio")
        text = str(row.get(transcript_col, "")).strip().lower()
        if not audio_obj or not text:
            continue

        pcm16: bytes
        sample_rate: int

        source_path = audio_obj.get("path") if isinstance(audio_obj, dict) else None
        source_bytes = audio_obj.get("bytes") if isinstance(audio_obj, dict) else None

        source_path_obj = Path(str(source_path)) if source_path else None

        if source_path_obj is not None and source_path_obj.exists():
            pcm16, sample_rate = _read_audio_file(source_path_obj)
        elif source_bytes:
            with io.BytesIO(source_bytes) as blob:
                audio, sample_rate = sf.read(blob, dtype="float32", always_2d=False)
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio = audio.mean(axis=1)
            if not isinstance(audio, np.ndarray):
                audio = np.asarray(audio, dtype=np.float32)
            pcm16 = _float_to_pcm16(audio)
            sample_rate = int(sample_rate)
        else:
            continue

        samples.append(
            AudioTranscriptSample(
                pcm16=pcm16,
                sample_rate_hz=sample_rate,
                transcript=text,
                source_id=f"hf-{index}",
            )
        )

        if len(samples) >= limit:
            break

    return samples


def _load_from_kaggle(limit: int) -> list[AudioTranscriptSample]:
    manifest_candidates = [
        Path("data/kaggle_manifest.csv"),
        Path("data/asr_manifest.csv"),
        Path("kaggle/asr_manifest.csv"),
    ]

    manifest = next((path for path in manifest_candidates if path.exists()), None)
    if manifest is None:
        return []

    samples: list[AudioTranscriptSample] = []
    base_dir = manifest.parent

    with manifest.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            audio_path = str(row.get("audio_path", "")).strip()
            text = str(row.get("text", "")).strip().lower()
            if not audio_path or not text:
                continue

            full_path = Path(audio_path)
            if not full_path.is_absolute():
                full_path = base_dir / full_path
            if not full_path.exists():
                continue

            pcm16, sample_rate = _read_audio_file(full_path)
            samples.append(
                AudioTranscriptSample(
                    pcm16=pcm16,
                    sample_rate_hz=sample_rate,
                    transcript=text,
                    source_id=f"manifest-{index}",
                )
            )

            if len(samples) >= limit:
                break

    return samples


def _load_builtin_demo(limit: int) -> list[AudioTranscriptSample]:
    # Fallback only when no real dataset is present; intended for connectivity smoke checks.
    transcripts = [
        "please confirm your installment amount",
        "i can make a partial payment today",
        "my due date is tenth august",
    ]

    samples: list[AudioTranscriptSample] = []
    for idx, text in enumerate(transcripts[:limit]):
        sr = 16000
        duration_s = 1.2
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        waveform = 0.03 * np.sin(2 * np.pi * 220.0 * t)
        samples.append(
            AudioTranscriptSample(
                pcm16=_float_to_pcm16(waveform.astype(np.float32)),
                sample_rate_hz=sr,
                transcript=text,
                source_id=f"builtin-{idx}",
            )
        )

    return samples
