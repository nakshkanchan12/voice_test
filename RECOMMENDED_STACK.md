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

## Live Microphone Input (No Static Dataset)

You can test the currently running models using your own live voice input.

1) Install microphone dependency (one-time):

```bash
pip install sounddevice
```

2) Run one live mic turn through the full pipeline:

```bash
scripts/run_live_mic_turn.sh
```

This records your microphone (default 5 seconds), runs ASR -> LLM -> TTS, saves the TTS output WAV, and prints turn latency metrics.

Useful overrides:

```bash
MIC_SECONDS=7 \
MIC_DEVICE=0 \
OUTPUT_WAV=results/live_turn_mic_custom.wav \
scripts/run_live_mic_turn.sh
```

Direct module invocation (equivalent):

```bash
python -m voice_pipeline.live_turn \
  --dataset-source mic \
  --mic-seconds 5 \
  --mic-sample-rate-hz 16000 \
  --mic-channels 1 \
  --asr-url http://127.0.0.1:8011 \
  --llm-base-url http://127.0.0.1:30000/v1 \
  --llm-model HuggingFaceTB/SmolLM2-360M-Instruct \
  --tts-url http://127.0.0.1:8012
```

Interactive call mode with continuous metrics logging:

```bash
INTERACTIVE_CALL=1 \
MAX_TURNS=10 \
METRICS_JSONL=results/live_call_metrics.jsonl \
METRICS_SUMMARY_JSON=results/live_call_summary.json \
scripts/run_live_mic_turn.sh
```

Behavior in interactive mode:
- records each mic turn, runs ASR -> LLM -> TTS, and writes `*_turn_XXX.wav` outputs
- appends one JSON metrics row per turn to `METRICS_JSONL` while the call is active
- writes aggregate p50/p95 and hit-rate summary to `METRICS_SUMMARY_JSON` when the call ends

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

By default, `scripts/run_recommended_stack.sh` now enables:
- aggressive LLM streaming (`LLM_STREAM_MODE=aggressive`)
- barge-in cancellation (`ENABLE_BARGE_IN=1`, `BARGE_IN_MIN_SPEECH_MS=120`)
- streaming VAD partial emission (`VAD_STREAMING=1`, `VAD_PARTIAL_SEGMENT_MS=320`)

Override these with environment variables when needed for A/B testing.

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

## Local Nemotron NIM (GPU)

If you want Nemotron running locally on your GPU, use the new helper script.

1) Export your NGC key and start NIM container:

```bash
export NGC_API_KEY=<your_ngc_key>
scripts/start_nemotron_asr_nim_local.sh
```

Defaults used by the launcher:
- Image: `nvcr.io/nim/nvidia/nemotron-asr-streaming-riva-v1:latest`
- HTTP: `http://127.0.0.1:9000`
- gRPC: `127.0.0.1:50051`

2) Start local LLM and TTS (same as hybrid):

```bash
HF_HUB_DISABLE_XET=1 LLM_MODEL_ID=HuggingFaceTB/SmolLM2-360M-Instruct \
uvicorn services.llm_server:app --host 127.0.0.1 --port 30000
```

```bash
PIPER_MODEL_PATH=models/piper/en_US-lessac-medium.onnx \
PIPER_CONFIG_PATH=models/piper/en_US-lessac-medium.onnx.json \
uvicorn services.piper_tts_server:app --host 127.0.0.1 --port 8012
```

3) Run the recommended pipeline pre-configured for local NIM ASR:

```bash
scripts/run_recommended_stack_nemotron_local.sh
```

This wrapper sets:
- `ASR_URL=http://127.0.0.1:9000`
- `ASR_TRANSCRIBE_PATH=/v1/audio/transcriptions`
- `ASR_AUDIO_FIELD=file`
- `ASR_MODEL=nvidia/nemotron-asr-streaming`
- `WAIT_FOR_HEALTH=0`

If your local NIM image/model name differs, override it:

```bash
NEMOTRON_ASR_MODEL=<your_model_name> scripts/run_recommended_stack_nemotron_local.sh
```

## Startup Scripts

- scripts/start_sglang_phi35_awq.sh
- scripts/start_qwen3_tts_vllm_omni.sh
- scripts/start_nemotron_asr_stub.sh
- scripts/start_nemotron_asr_nim_local.sh
- scripts/run_recommended_stack.sh
- scripts/run_recommended_stack_nemotron_local.sh
- scripts/run_live_mic_turn.sh

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
