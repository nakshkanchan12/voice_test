from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

VOICE = None
SAMPLE_RATE = 22050


class SynthesizeRequest(BaseModel):
    text: str
    call_id: str = ""
    turn_id: str = ""


def _resolve_model_files() -> tuple[Path, Path]:
    explicit_model = os.getenv("PIPER_MODEL_PATH", "").strip()
    explicit_config = os.getenv("PIPER_CONFIG_PATH", "").strip()

    if explicit_model:
        model_path = Path(explicit_model)
        config_path = Path(explicit_config) if explicit_config else Path(f"{explicit_model}.json")
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                "Piper model or config not found. Set PIPER_MODEL_PATH and PIPER_CONFIG_PATH correctly."
            )
        return model_path, config_path

    from huggingface_hub import hf_hub_download

    repo_id = os.getenv("PIPER_HF_REPO", "rhasspy/piper-voices")
    model_file = os.getenv(
        "PIPER_HF_MODEL_FILE",
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    )

    model_path = Path(hf_hub_download(repo_id=repo_id, filename=model_file))
    config_path = Path(hf_hub_download(repo_id=repo_id, filename=f"{model_file}.json"))
    return model_path, config_path


def _stream_synthesis(text: str) -> AsyncIterator[bytes]:
    async def _iter() -> AsyncIterator[bytes]:
        global VOICE
        if VOICE is None:
            raise RuntimeError("Piper voice is not loaded")

        loop = asyncio.get_running_loop()

        def run_sync() -> list[bytes]:
            chunks: list[bytes] = []
            for chunk in VOICE.synthesize(text):
                pcm_bytes = chunk.audio_int16_bytes
                if pcm_bytes:
                    chunks.append(pcm_bytes)
            return chunks

        parts = await loop.run_in_executor(None, run_sync)
        for part in parts:
            yield part

    return _iter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global VOICE, SAMPLE_RATE

    from piper.voice import PiperVoice

    model_path, config_path = _resolve_model_files()
    use_cuda = os.getenv("PIPER_USE_CUDA", "1").strip().lower() in {"1", "true", "yes"}

    VOICE = PiperVoice.load(
        model_path=model_path,
        config_path=config_path,
        use_cuda=use_cuda,
    )
    SAMPLE_RATE = int(getattr(getattr(VOICE, "config", None), "sample_rate", 22050))

    yield

    VOICE = None


app = FastAPI(title="Piper TTS Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok" if VOICE is not None else "loading",
        "sample_rate_hz": SAMPLE_RATE,
    }


@app.post("/synthesize")
async def synthesize(payload: SynthesizeRequest) -> StreamingResponse:
    generator = _stream_synthesis(payload.text)
    return StreamingResponse(
        generator,
        media_type="application/octet-stream",
        headers={
            "x-audio-format": "pcm_s16le",
            "x-sample-rate-hz": str(SAMPLE_RATE),
            "x-channels": "1",
        },
    )
