# Recommended Voice AI Stack

This project now includes a concrete pipeline runner for the architecture:

- VAD: Silero v4 (in-process)
- ASR: HTTP transcription endpoint (faster-whisper tiny validated)
- LLM: OpenAI-compatible endpoint (SmolLM2 validated)
- TTS: HTTP synth endpoint (Piper lessac validated)
- Orchestration: full-duplex asyncio queue pipeline with aggressive chunking + barge-in cancellation
- Transport: switchable `http` or `webrtc` for direct latency comparison

## Pipeline Entry Point

Run:

python -m voice_pipeline.recommended_pipeline \
  --calls 20 \
  --concurrency 5 \
  --sample-pool 5 \
  --transport http \
  --llm-stream-mode aggressive \
  --enable-barge-in true \
  --barge-in-min-speech-ms 120 \
  --vad-streaming true \
  --vad-partial-segment-ms 320 \
  --asr-url http://127.0.0.1:8011 \
  --llm-base-url http://127.0.0.1:30000/v1 \
  --llm-model HuggingFaceTB/SmolLM2-360M-Instruct \
  --tts-url http://127.0.0.1:8012 \
  --tts-mode http_synthesize \
  --tts-model piper/en_US-lessac-medium \
  --wait-for-health

WebRTC run (same pipeline, different transport):

python -m voice_pipeline.recommended_pipeline \
  --calls 20 \
  --concurrency 5 \
  --sample-pool 5 \
  --transport webrtc \
  --llm-stream-mode aggressive \
  --enable-barge-in true \
  --barge-in-min-speech-ms 120 \
  --vad-streaming true \
  --vad-partial-segment-ms 320 \
  --asr-url http://127.0.0.1:8011 \
  --llm-base-url http://127.0.0.1:30000/v1 \
  --llm-model HuggingFaceTB/SmolLM2-360M-Instruct \
  --tts-url http://127.0.0.1:8012 \
  --tts-model piper/en_US-lessac-medium \
  --wait-for-health

Output:

- JSON metrics file with per-turn and p50/p95 latency values.
- Includes mic-based first-audio metrics and barge-in cancellation marker per turn.

## Hybrid Setup (Remote Nemotron ASR + Local LLM/TTS)

1) Start local LLM service:

```bash
HF_HUB_DISABLE_XET=1 LLM_MODEL_ID=HuggingFaceTB/SmolLM2-360M-Instruct \
uvicorn services.llm_server:app --host 127.0.0.1 --port 30000
```

2) Start local TTS service:

```bash
PIPER_MODEL_PATH=models/piper/en_US-lessac-medium.onnx \
PIPER_CONFIG_PATH=models/piper/en_US-lessac-medium.onnx.json \
uvicorn services.piper_tts_server:app --host 127.0.0.1 --port 8012
```

3) Point ASR to your remote Nemotron endpoint and run pipeline:

```bash
ASR_URL=http://<nemotron-host>:<port> \
LLM_BASE_URL=http://127.0.0.1:30000/v1 \
TTS_URL=http://127.0.0.1:8012 \
scripts/run_recommended_stack.sh
```

If your Nemotron deployment does not expose `GET /health`, disable health waiting:

```bash
ASR_URL=http://<nemotron-host>:<port> \
WAIT_FOR_HEALTH=0 \
scripts/run_recommended_stack.sh
```

Nemotron ASR contract expected by this pipeline:
- `POST /transcribe` with multipart field `audio_file`
- response JSON containing either `text` or `transcript`
- optional `POST /webrtc/offer` when using WebRTC transport

NVIDIA hosted Nemotron API pattern (Build API style) can be used with:

```bash
ASR_URL=https://<nvidia-api-host> \
ASR_TRANSCRIBE_PATH=/v1/audio/transcriptions \
ASR_AUDIO_FIELD=file \
ASR_MODEL=nvidia/nemotron-asr-streaming \
ASR_API_KEY=<your_nvidia_api_key> \
WAIT_FOR_HEALTH=0 \
scripts/run_recommended_stack.sh
```

Use `WAIT_FOR_HEALTH=0` for hosted APIs that do not expose `GET /health`.

## Startup Scripts

- scripts/start_sglang_phi35_awq.sh
- scripts/start_qwen3_tts_vllm_omni.sh
- scripts/start_nemotron_asr_stub.sh
- scripts/run_recommended_stack.sh

Environment and dependency templates:

- configs/recommended_stack.env.example
- requirements-recommended.txt

## Endpoint Contracts

ASR endpoint contract:
- POST /transcribe
- multipart field: audio_file
- returns: {"text": "..."} or {"transcript": "..."}
- POST /webrtc/offer for WebRTC data-channel negotiation

LLM endpoint contract:
- POST /v1/chat/completions
- OpenAI-compatible streaming chunks
- POST /webrtc/offer for WebRTC data-channel negotiation

TTS endpoint options:
- mode=http_synthesize: POST /synthesize (streaming PCM bytes)
- mode=openai_audio_speech: POST /v1/audio/speech (streaming PCM bytes)
- POST /webrtc/offer for WebRTC data-channel negotiation

## Queue Architecture

- ASR queue maxsize=2
- TTS queue maxsize=2
- Concurrent ASR/LLM/TTS stages with asyncio.gather
- LLM aggressive chunk streaming to start TTS before full sentence completion
- Generation-based barge-in cancellation to stop stale TTS when user interrupts
- Streaming VAD partial emission for earlier ASR dispatch
