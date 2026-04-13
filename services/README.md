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
