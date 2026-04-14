# Real Voice AI Services

This folder contains live endpoints backed by real model runtimes.

## 1) ASR Server (faster-whisper)

Start:

```bash
ASR_MODEL_ID=small ASR_DEVICE=auto ASR_COMPUTE_TYPE=int8_float16 uvicorn services.asr_server:app --host 0.0.0.0 --port 8011
```

Health:

```bash
curl -s http://127.0.0.1:8011/health
```

Transcribe:

```bash
curl -s -X POST http://127.0.0.1:8011/transcribe \
  -F "audio_file=@sample.wav" \
  -F "language=en"
```

WebRTC offer endpoint:

```bash
curl -s -X POST http://127.0.0.1:8011/webrtc/offer \
  -H "Content-Type: application/json" \
  -d '{"type":"offer","sdp":"..."}'
```

## 2) TTS Server (Piper)

Start (auto-downloads a default voice from HF):

```bash
PIPER_USE_CUDA=1 uvicorn services.piper_tts_server:app --host 0.0.0.0 --port 8012
```

Health:

```bash
curl -s http://127.0.0.1:8012/health
```

Synthesize raw PCM:

```bash
curl -s -X POST http://127.0.0.1:8012/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"your payment is due tomorrow"}' \
  --output out.pcm
```

WebRTC offer endpoint:

```bash
curl -s -X POST http://127.0.0.1:8012/webrtc/offer \
  -H "Content-Type: application/json" \
  -d '{"type":"offer","sdp":"..."}'
```

## 3) Real E2E Single Turn

With ASR/TTS services running and an OpenAI-compatible LLM endpoint available:

```bash
python -m voice_pipeline.live_turn \
  --asr-url http://127.0.0.1:8011 \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model microsoft/Phi-3.5-mini-instruct \
  --tts-url http://127.0.0.1:8012
```

The script writes a real response waveform to `results/live_turn_output.wav`.

## 4) Local LLM Server (OpenAI-Compatible)

Start a local real model endpoint:

```bash
HF_HUB_DISABLE_XET=1 LLM_MODEL_ID=HuggingFaceTB/SmolLM2-360M-Instruct uvicorn services.llm_server:app --host 0.0.0.0 --port 8000
```

Health:

```bash
curl -s http://127.0.0.1:8000/health
```

OpenAI-compatible test:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "messages": [
      {"role":"system","content":"You are concise."},
      {"role":"user","content":"Write one sentence about EMI reminder."}
    ],
    "stream": false
  }'
```

WebRTC offer endpoint:

```bash
curl -s -X POST http://127.0.0.1:8000/webrtc/offer \
  -H "Content-Type: application/json" \
  -d '{"type":"offer","sdp":"..."}'
```

Then run full live turn with all local endpoints:

```bash
python -m voice_pipeline.live_turn \
  --asr-url http://127.0.0.1:8011 \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model HuggingFaceTB/SmolLM2-360M-Instruct \
  --tts-url http://127.0.0.1:8012
```

Concurrent live benchmark:

```bash
python -m voice_pipeline.live_benchmark \
  --calls 6 \
  --concurrency 2 \
  --transport http \
  --asr-url http://127.0.0.1:8011 \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model HuggingFaceTB/SmolLM2-360M-Instruct \
  --tts-url http://127.0.0.1:8012
```

Concurrent live benchmark over WebRTC data channels:

```bash
python -m voice_pipeline.live_benchmark \
  --calls 6 \
  --concurrency 2 \
  --transport webrtc \
  --asr-url http://127.0.0.1:8011 \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model HuggingFaceTB/SmolLM2-360M-Instruct \
  --tts-url http://127.0.0.1:8012
```

## 5) Recommended Production Stack Runner

Use the dedicated Nemotron + SGLang + Qwen3 pipeline runner:

```bash
python -m voice_pipeline.recommended_pipeline \
  --calls 20 \
  --concurrency 5 \
  --sample-pool 5 \
  --transport http \
  --asr-url http://127.0.0.1:8011 \
  --llm-base-url http://127.0.0.1:30000/v1 \
  --llm-model microsoft/Phi-3.5-mini-instruct \
  --tts-url http://127.0.0.1:8012 \
  --tts-mode http_synthesize \
  --tts-model Qwen/Qwen3-TTS-0.6B \
  --wait-for-health
```

WebRTC mode uses each service offer endpoint at `/webrtc/offer`. You can switch modes by changing `--transport`.

Helper scripts:

```bash
bash scripts/start_sglang_phi35_awq.sh
bash scripts/start_qwen3_tts_vllm_omni.sh
bash scripts/start_nemotron_asr_stub.sh
bash scripts/run_recommended_stack.sh
```
