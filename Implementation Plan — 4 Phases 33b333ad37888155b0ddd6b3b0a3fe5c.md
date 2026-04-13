# Implementation Plan — 4 Phases

Four phases, each building on the last. Each phase has a clear deliverable and go/no-go gate before moving forward. Total timeline: 6–8 weeks.

---

# Phase 1 — streaming pipeline core

**Goal:** Replace the naive bulk-inference pipeline with a true streaming pipeline where ASR, LLM, and TTS run concurrently via asyncio queues. No LiveKit integration yet — test with audio files.

**Timeline:** Week 1–2

**Deliverable:** `streaming_pipeline.py` — a standalone script that takes a WAV file as input and produces streaming audio output, with per-stage latency logging.

## What to build

### 1.1 — Streaming ASR server

Build a FastAPI server wrapping faster-whisper that:

- Accepts audio as a stream (chunked HTTP or WebSocket)
- Uses `WhisperModel.transcribe()` with `word_timestamps=True`
- Emits partial transcripts via SSE as VAD detects speech end
- Pre-loads the model at startup (warm on first request)
- Uses `loop.run_in_executor(thread_pool, ...)` so inference doesn’t block asyncio

```python
# Target interface
async def transcribe_stream(audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    # yields partial transcript strings as they arrive
```

### 1.2 — Streaming LLM client

Build an async wrapper around vLLM’s OpenAI-compatible streaming endpoint:

- Always passes system prompt (POST_BOUNCE_AGENT instructions)
- Uses `stream=True` with SSE parsing
- Yields tokens as they arrive
- Implements sentence boundary detection: accumulates tokens until `['.', '!', '?', '\n']` then flushes

```python
async def generate_stream(transcript: str, system_prompt: str) -> AsyncIterator[str]:
    # yields complete sentences, not individual tokens
```

### 1.3 — Streaming TTS server

Build a FastAPI server wrapping Piper that:

- Accepts one sentence at a time
- Returns raw PCM audio bytes (16kHz, 16-bit, mono)
- Pre-loads model and ONNX session at startup
- Processes requests in a `ThreadPoolExecutor` (Piper is synchronous)

### 1.4 — Pipeline orchestrator

Chain the three stages with asyncio queues:

```python
asr_queue   = asyncio.Queue(maxsize=2)   # transcript sentences
tts_queue   = asyncio.Queue(maxsize=2)   # PCM audio chunks

async def run_pipeline(audio_input):
    await asyncio.gather(
        asr_stage(audio_input, asr_queue),
        llm_stage(asr_queue, tts_queue),
        tts_stage(tts_queue, audio_output),
    )
```

## Go/no-go gate

- [ ]  E2E turn latency (warm) < 1,500ms on a single call
- [ ]  All three model servers survive 5 restarts without memory leaks
- [ ]  Per-stage latency logged to JSON for every request

---

# Phase 2 — concurrency and pooling

**Goal:** Make the pipeline handle 20 simultaneous calls without latency degradation beyond 2× single-call baseline.

**Timeline:** Week 2–3

**Deliverable:** `pipeline_server.py` — a FastAPI service that accepts N concurrent pipeline requests and routes them through shared model servers with connection pooling.

## What to build

### 2.1 — Shared model server pool

Instead of one model server per call, run one model server per GPU and route all calls through it:

- **ASR:** Single faster-whisper server with `ThreadPoolExecutor(max_workers=4)`. Each worker handles one call’s audio concurrently.
- **LLM:** Single vLLM server. Continuous batching handles concurrency automatically — no changes needed.
- **TTS:** Single Piper server with `ThreadPoolExecutor(max_workers=8)` (CPU-bound, can have more workers).

### 2.2 — HTTP connection pool

All 20 concurrent calls share one `aiohttp.ClientSession` with a `TCPConnector`:

```python
# In server startup, not per-request
connector = aiohttp.TCPConnector(
    limit=50,          # max total connections
    limit_per_host=50, # max connections to one host (vLLM)
    ttl_dns_cache=300, # cache DNS resolution
    use_dns_cache=True,
)
HTTP_SESSION = aiohttp.ClientSession(connector=connector)
```

This avoids the overhead of TCP handshakes on every LLM/TTS request.

### 2.3 — asyncpg database pool

Replace SQLAlchemy `async_session` with `asyncpg` pool for transcript writes:

```python
DB_POOL = await asyncpg.create_pool(
    dsn=DATABASE_URL,
    min_size=5,
    max_size=20,
    command_timeout=10,
)
```

### 2.4 — Backpressure and circuit breaking

- If the ASR queue is full (model overloaded), return HTTP 503 immediately rather than queuing indefinitely
- Implement a simple circuit breaker: if > 3 requests timeout in 10s, stop accepting new calls and alert
- Use `asyncio.Semaphore(max_concurrent=20)` as a concurrency limiter at the API layer

## Go/no-go gate

- [ ]  20 concurrent calls complete with p95 turn latency < 2,000ms
- [ ]  No request timeouts at 20 concurrency
- [ ]  DB pool not exhausted under load (`asyncpg` pool stats logged)
- [ ]  Memory stable over 100 sequential calls (no leak)

---

# Phase 3 — kernel optimisations

**Goal:** Apply hardware-level optimisations to each model to reduce latency by 30–50% vs Phase 2 baseline.

**Timeline:** Week 3–4

**Deliverable:** Ablation report comparing baseline vs each optimisation in isolation vs all combined.

## What to apply

### 3.1 — ASR: int8 quantisation + beam_size=1

```python
# Before (float16, beam=5)
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
segments, _ = model.transcribe(audio, beam_size=5)

# After (int8_float16, beam=1 greedy)
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
segments, _ = model.transcribe(audio, beam_size=1, best_of=1)
```

Expected gain: 25–40% latency reduction.

### 3.2 — LLM: AWQ int4 quantisation

```bash
# Quantise Phi-3.5-mini to AWQ int4
pip install autoawq
python -m awq.quantize --model microsoft/Phi-3.5-mini-instruct --output ./phi35-awq-int4

# Serve with vLLM
python -m vllm.entrypoints.openai.api_server \
  --model ./phi35-awq-int4 \
  --quantization awq \
  --gpu-memory-utilization 0.4   # leaves room for Whisper on same GPU
```

Expected gain: 20–35% TTFT reduction, 50% VRAM reduction.

### 3.3 — LLM: SGLang RadixAttention for prefix caching

Swap vLLM for SGLang when the system prompt is long and repeated:

```bash
python -m sglang.launch_server \
  --model-path microsoft/Phi-3.5-mini-instruct \
  --port 30000 \
  --mem-fraction-static 0.4
```

Expected gain: ~300ms TTFT savings per call after first (prefix cache hit).

### 3.4 — TTS: TensorRT-EP for Piper ONNX

```python
# Convert Piper ONNX to TensorRT
import onnxruntime as ort

sess_options = ort.SessionOptions()
providers = [
    ('TensorrtExecutionProvider', {
        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,  # 2GB
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache',
    }),
    'CUDAExecutionProvider',
]
model = PiperVoice.load("en_US-lessac-medium.onnx", sess_options=sess_options, providers=providers)
```

Expected gain: 40–60% TTS TTFB reduction (first run slow due to TRT engine build; cached after).

### 3.5 — torch.compile for StyleTTS2 / PyTorch models

```python
import torch
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

Expected gain: 10–25% on PyTorch models after warm-up.

### 3.6 — LLM speculative decoding (stretch goal)

Use a small draft model (Phi-3.5-mini) to speculatively generate tokens verified by a larger model. Only worth exploring if Llama-3.1-8B is chosen over Phi-3.5-mini.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model microsoft/Phi-3.5-mini-instruct \
  --num-speculative-tokens 5
```

## Go/no-go gate

- [ ]  Phase 3 optimised pipeline achieves e2e turn latency p50 < 800ms (warm, single call)
- [ ]  p95 < 1,200ms under 20 concurrent calls
- [ ]  Ablation table completed (each optimisation measured in isolation)

---

# Phase 4 — LiveKit integration and live telephony

**Goal:** Wire the optimised pipeline into `pipeline_agent.py`, replacing Deepgram/OpenAI/Cartesia with local servers. Run real calls via SIP trunk.

**Timeline:** Week 4–6

**Deliverable:** Modified `pipeline_agent.py` that uses local OSS models, passes all existing tests, and successfully completes 20 concurrent real calls.

## What to build

### 4.1 — LiveKit plugin wrappers

Create three plugin classes matching the LiveKit agents SDK interface:

```python
# asr_plugin.py
class WhisperSTT(STT):
    async def recognize(self, buffer: AudioBuffer) -> SpeechEvent:
        # calls local faster-whisper server
        # returns SpeechEvent with transcript

# llm_plugin.py  
class LocalLLM(LLM):
    async def chat(self, messages: list[ChatMessage]) -> AsyncIterator[ChatChunk]:
        # calls local vLLM server with stream=True
        # yields ChatChunk objects token by token

# tts_plugin.py
class PiperTTS(TTS):
    async def synthesize(self, text: str) -> AsyncIterator[SynthesizedAudio]:
        # calls local Piper server
        # yields SynthesizedAudio chunks at sentence boundaries
```

### 4.2 — Swap plugins in pipeline_[agent.py](http://agent.py)

The change is surgical — only the session creation block changes:

```python
# Before
stt = deepgram.STT(model="nova-2-general")
llm_model = openai.LLM(model="gpt-3.5-turbo")
tts = cartesia.TTS(voice="79a125e8...")

# After
from asr_plugin import WhisperSTT
from llm_plugin import LocalLLM
from tts_plugin import PiperTTS

stt = WhisperSTT(server_url="http://localhost:8765")
llm_model = LocalLLM(server_url="http://localhost:8080", system_prompt=formatted_instructions)
tts = PiperTTS(server_url="http://localhost:8766")
```

### 4.3 — System prompt injection fix

The Pipeline 2 test exposed that system prompts must be explicitly passed through the plugin. Ensure `LocalLLM` always prepends the `formatted_instructions` from `OutboundCaller.__init__()` as a system message.

### 4.4 — Audio frame re-chunking for LiveKit

LiveKit expects `AudioFrame` objects at 20ms intervals (320 samples at 16kHz). Piper outputs variable-length PCM. Add a frame buffer:

```python
FRAME_SAMPLES = 320  # 20ms at 16kHz

async def rechunk_audio(pcm_stream: AsyncIterator[bytes]) -> AsyncIterator[AudioFrame]:
    buffer = b""
    async for chunk in pcm_stream:
        buffer += chunk
        while len(buffer) >= FRAME_SAMPLES * 2:  # 16-bit = 2 bytes/sample
            frame_bytes = buffer[:FRAME_SAMPLES * 2]
            buffer = buffer[FRAME_SAMPLES * 2:]
            yield AudioFrame(data=frame_bytes, sample_rate=16000, num_channels=1)
```

## Go/no-go gate

- [ ]  Single real call completes successfully with correct banking response
- [ ]  5 concurrent real calls complete
- [ ]  20 concurrent real calls complete with < 5% failure rate
- [ ]  Turn latency on real calls measured by `pipeline_agent_instrumentation.py` < 1,000ms p50