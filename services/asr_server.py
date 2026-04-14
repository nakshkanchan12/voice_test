from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, UploadFile

from .webrtc_common import create_offer_handler


MODEL = None
EXECUTOR: ThreadPoolExecutor | None = None
WEBRTC_PEERS: set[Any] = set()


def _resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized != "auto":
        return normalized

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_compute_type(device: str, compute_type: str) -> str:
    if device == "cpu" and compute_type == "int8_float16":
        return "int8"
    return compute_type


def _transcribe_file(audio_bytes: bytes, language: str | None, beam_size: int) -> str:
    global MODEL
    if MODEL is None:
        raise RuntimeError("ASR model is not loaded")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        wav_path = Path(tmp.name)

    try:
        segments, _info = MODEL.transcribe(
            str(wav_path),
            beam_size=beam_size,
            best_of=1,
            word_timestamps=False,
            language=language or None,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return text.strip()
    finally:
        wav_path.unlink(missing_ok=True)


async def _single_result(payload: dict[str, object]) -> AsyncIterator[dict[str, object]]:
    yield payload


async def _webrtc_rpc_handler(
    op: str,
    payload: dict[str, object],
    stream: bool,
) -> dict[str, object] | AsyncIterator[dict[str, object]]:
    if op != "transcribe":
        raise ValueError(f"Unsupported operation: {op}")

    if EXECUTOR is None:
        raise RuntimeError("ASR executor not initialized")

    audio_wav_b64 = str(payload.get("audio_wav_b64", "")).strip()
    if not audio_wav_b64:
        raise ValueError("Missing audio_wav_b64 payload")

    try:
        audio_bytes = base64.b64decode(audio_wav_b64.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 audio payload") from exc

    language = str(payload.get("language", "")).strip() or None
    beam_size = int(os.getenv("ASR_BEAM_SIZE", "1"))

    started = perf_counter()
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(
        EXECUTOR,
        _transcribe_file,
        audio_bytes,
        language,
        beam_size,
    )
    latency_ms = round((perf_counter() - started) * 1000.0, 2)

    result: dict[str, object] = {
        "text": text,
        "latency_ms": latency_ms,
        "call_id": str(payload.get("call_id", "")),
        "turn_id": str(payload.get("turn_id", "")),
    }

    if stream:
        return _single_result(result)

    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, EXECUTOR

    from faster_whisper import WhisperModel

    model_id = os.getenv("ASR_MODEL_ID", "tiny")
    desired_device = os.getenv("ASR_DEVICE", "auto")
    desired_compute = os.getenv("ASR_COMPUTE_TYPE", "int8_float16")
    workers = int(os.getenv("ASR_WORKERS", "2"))

    device = _resolve_device(desired_device)
    compute_type = _resolve_compute_type(device, desired_compute)

    MODEL = WhisperModel(model_id, device=device, compute_type=compute_type)
    EXECUTOR = ThreadPoolExecutor(max_workers=workers)

    yield

    if EXECUTOR is not None:
        EXECUTOR.shutdown(wait=False)
    EXECUTOR = None
    MODEL = None


app = FastAPI(title="Real ASR Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok" if MODEL is not None else "loading",
        "model": os.getenv("ASR_MODEL_ID", "tiny"),
    }


@app.post("/transcribe")
async def transcribe(
    audio_file: UploadFile = File(...),
    language: str = Form(default=""),
    call_id: str = Form(default=""),
    turn_id: str = Form(default=""),
) -> dict[str, object]:
    if EXECUTOR is None:
        raise RuntimeError("ASR executor not initialized")

    audio_bytes = await audio_file.read()
    beam_size = int(os.getenv("ASR_BEAM_SIZE", "1"))

    started = perf_counter()
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(
        EXECUTOR,
        _transcribe_file,
        audio_bytes,
        language.strip() or None,
        beam_size,
    )
    latency_ms = round((perf_counter() - started) * 1000.0, 2)

    return {
        "text": text,
        "latency_ms": latency_ms,
        "call_id": call_id,
        "turn_id": turn_id,
    }


app.post("/webrtc/offer")(create_offer_handler(WEBRTC_PEERS, _webrtc_rpc_handler))
