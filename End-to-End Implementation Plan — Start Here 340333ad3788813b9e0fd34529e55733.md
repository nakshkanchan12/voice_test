# End-to-End Implementation Plan — Start Here

This is the single source of truth for what to do, when, and why. Every decision here is grounded in real benchmark data already collected (see [KRIM.ai](http://KRIM.ai) Telephony — OSS Model Benchmark Report and Pipeline E2E Test Results). Read this top to bottom before writing any code.

> **Goal:** A production voice AI pipeline, fully open-source, streaming between ASR→LLM→TTS via asyncio queues, handling 20+ concurrent telephony calls with <1s turn latency. Deployed on Lightning AI (dev/ablation) and AWS SageMaker (production scale). Integrated into the existing LiveKit/KRIM telephony stack via `pipeline_agent.py`.
> 

> **Compute available:** Lightning AI (A10G instances) for iteration · AWS SageMaker (ml.g5.xlarge, LMI containers) for production · AWS g4dn.xlarge for TTS
> 

---

# What is already done

Before you write a single line of code, understand what ground truth you have:

- **Baseline measured:** Deepgram + GPT-3.5-turbo + Cartesia on 20 simulated concurrent calls → turn latency p50 = 727ms, cost $6.70/day at 10k calls
- **5 ASR models benchmarked** on latency (WER invalid — 100% across all, audio harness bug, must re-run with real audio)
- **7 LLM configs benchmarked:** Phi-3.5-mini best TTFT (49ms), Llama-3.1-8B best quality, GPT-4o mini worst (821ms, SaaS)
- **4 TTS models benchmarked:** Piper best TTFB (314ms), StyleTTS2 good quality (422ms), Kokoro broken warm-up (2.5s, fixable), Coqui too slow (3s)
- **2 real E2E pipelines run:** Pipeline 1 (Whisper+Phi+Piper) cold e2e = 2,454ms → warm estimate ~750ms ✅ · Pipeline 2 (Whisper+Llama+StyleTTS2) = 20,716ms ❌ (StyleTTS2 no-stream + system prompt not injected)
- **All code exists:** `pipeline_agent.py`, `benchmark_baseline.py`, `load_test_concurrent.py`, `pipeline_agent_instrumentation.py`

---

# Pre-work: read before coding (days 1–3)

Do not skip this. Understanding these five things will save you weeks of debugging.

**Day 1 — asyncio and concurrency model**

Read: Python asyncio docs (event loop, tasks, queues). Read: `loop.run_in_executor()` pattern. Understand: why a blocking `model.transcribe()` call freezes all 20 concurrent calls. Understand: `asyncio.Queue(maxsize=2)` creates backpressure — this is the core of your pipeline.

**Day 1 — study livekit/agents pipeline code**

Clone `livekit/agents`. Read `livekit/agents/pipeline/` directory. Understand how `AgentSession` chains STT→LLM→TTS already. Your job is to replace the plugin endpoints, not rebuild the session logic.

**Day 2 — vLLM PagedAttention + SGLang RadixAttention**

Read the vLLM paper abstract + sections 1–3. Key insight: continuous batching means 20 concurrent callers share one GPU forward pass. Read SGLang docs on RadixAttention. Key insight: your `POST_BOUNCE_AGENT` system prompt is ~500 tokens and identical for every call — SGLang caches it after the first request, saving ~300ms TTFT on every subsequent call.

**Day 2 — Nemotron ASR cache-aware streaming**

Read: [HuggingFace blog on Nemotron Speech ASR](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents). Key insight: Whisper re-encodes the full audio context on every chunk. Nemotron's FastConformer maintains encoder caches across all self-attention and convolution layers — each audio frame processed exactly once. 560 concurrent streams on H100 vs 180 for buffered baseline. This is a 3× concurrency improvement for the same GPU.

**Day 3 — Qwen3-TTS dual-track streaming architecture**

Read: [Qwen3-TTS Technical Report (arXiv:2601.15621)](https://arxiv.org/abs/2601.15621) sections 1–3. Key insight: Qwen3-TTS generates speech tokens autoregressively like an LLM — first audio packet at 97ms TTFB without waiting for full sentence synthesis. Read: [vLLM-Omni Qwen3-TTS serving docs](https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/examples/online_serving/qwen3_tts/). The WebSocket streaming endpoint does sentence-boundary splitting for you. Read: [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) for CUDA graph capture pattern.

**Day 3 — study NeMo cache-aware streaming tutorial**

Read: [NeMo Online ASR notebook](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb). Understand `CacheAwareStreamingAudioBuffer`, `get_initial_cache_state()`, and the chunk loop. This is the implementation you'll follow for Nemotron.

---

# Week 1 — fix existing pipeline and build streaming core

## Task 1.1 — fix the ASR WER test harness (day 1)

All ASR models returned 100% WER. This is a harness bug — reference transcripts were not loaded correctly or audio was synthetic. This is the most important unblocked bug because WER determines which ASR model you ship.

**What to do:**

1. Record 20 real utterances: EMI amounts ("my EMI is ₹7,200"), account numbers ("last four digits 3721"), dates ("payment due 10th August"), negotiation ("I can pay half now"), noisy variants (add 5dB background noise in Audacity)
2. Create reference transcript text files with exact expected transcripts
3. Re-run the ASR benchmark with `jiwer` WER calculation against real references
4. Add Nemotron (80ms and 160ms chunks) and Qwen3-TTS as candidates

**Decision from this task:** Which ASR model to use in production. Expected outcome: Nemotron at 160ms chunks should achieve WER < 8% and latency ~80ms, making it the clear winner over Whisper turbo (219ms) and Parakeet (142ms).

## Task 1.2 — fix Pipeline 1 pre-warming (day 2)

The Pipeline 1 cold-start e2e was 2,454ms. Warm estimate is 580–750ms. The gap is entirely cold-start. Fix this before any other optimisation.

**What to do:** Add FastAPI lifespan startup warmup to each model server:

```python
@asynccontextmanager
async def lifespan(app):
    # Send one dummy request to each model at startup
    WHISPER_MODEL.transcribe("warmup.wav", beam_size=1)   # 1s silent WAV
    list(PIPER_VOICE.synthesize("Hello.", wav_file=io.BytesIO()))
    # For vLLM: POST /v1/chat/completions with "say hi"
    yield
```

**Measure:** Run Pipeline 1 three times in a row. Request 1 (cold), request 2, request 3. Record e2e for each. Target: request 2 e2e < 900ms, request 3 < 750ms.

## Task 1.3 — fix Pipeline 2 system prompt injection (day 2)

Pipeline 2 Llama responded with "I can assist with banking queries" when it should be in `POST_BOUNCE_AGENT` mode. The system prompt was not passed.

**What to do:** Ensure every LLM API call includes:

```python
messages = [
    {"role": "system", "content": POST_BOUNCE_AGENT_INSTRUCTIONS},  # always identical — SGLang caches this
    {"role": "user", "content": f"Customer: {name}. EMI: {emi}. DPD: {dpd}. They said: {transcript}"},
]
```

Verify by running Pipeline 2 again and checking the LLM response mentions EMI amounts and banking context.

## Task 1.4 — build the asyncio streaming pipeline (days 3–5)

This is the core engineering task of the entire internship. Replace bulk inference with a three-stage concurrent pipeline using asyncio queues.

**Architecture:**

```
audio_input → [ASR stage] → Queue_1(maxsize=2) → [LLM stage] → Queue_2(maxsize=2) → [TTS stage] → audio_output
```

**Build in this order:**

1. `asr_server.py` — FastAPI + faster-whisper, `run_in_executor` for blocking inference, `/transcribe` endpoint returning SSE partial transcripts
2. `llm_client.py` — async wrapper around vLLM streaming endpoint, sentence boundary detector splitting at `['.','!','?']`, yields complete sentences not tokens
3. `tts_server.py` — FastAPI + Piper, `ThreadPoolExecutor(max_workers=8)`, returns PCM bytes per sentence
4. `streaming_pipeline.py` — orchestrator connecting the three via `asyncio.gather()` and `asyncio.Queue`

**Measure after building:** Run with a 5-sentence banking audio file. Record: time from audio-end to first TTS audio byte. Target < 800ms warm.

**This single task is the difference between 2,454ms and <800ms.** Do not move to Week 2 until this works.

---

# Week 2 — concurrency, pooling, and ablation group A (ASR)

## Task 2.1 — HTTP connection pool and asyncpg pool (day 1)

Before running any concurrency tests, set up shared resource pools. Without these, every concurrent call creates new TCP connections and DB connections, causing cascading failures at 20 concurrent.

**What to build:**

```python
# One shared aiohttp session across ALL concurrent calls
CONNECTOR = aiohttp.TCPConnector(limit=50, limit_per_host=50, keepalive_timeout=30)
HTTP_SESSION = aiohttp.ClientSession(connector=CONNECTOR)

# asyncpg pool for transcript writes
DB_POOL = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=5, max_size=20)

# Semaphore to cap concurrency at 20
CALL_SEMAPHORE = asyncio.Semaphore(20)
```

**Attach the latency probe** from `pipeline_agent_instrumentation.py` to every call session — this is how you get real per-turn data from live calls.

## Task 2.2 — ablation A1: Whisper beam_size (day 1)

Run experiment A1 from the ablations doc. Variants: beam_size ∈ {5, 3, 1} × compute_type ∈ {float16, int8_float16}. Measure latency and WER (using real audio from Task 1.1). **Save results to `results/exp_A1_<n>.json`.**

**Decision:** Use the smallest beam_size where WER < 5% on banking audio. Expected: beam_size=1, int8_float16 wins (~40% latency reduction).

## Task 2.3 — ablation A2-e: Nemotron vs Whisper (days 2–3)

This is the highest-impact ASR experiment. Deploy Nemotron via NeMo, implement `CacheAwareStreamingAudioBuffer`, run at 80ms and 160ms chunk sizes. Compare against Whisper turbo (int8_float16, beam=1).

**What to measure:** Latency (ms from speech-end to transcript), WER on Indian-accented banking audio, concurrent stream count at 20 calls.

**Decision rule:** If Nemotron WER < 5% on banking audio, use it — it's faster and 3× more concurrent. If WER is 8–12%, use Whisper turbo for accuracy.

## Task 2.4 — ablation A3: batch vs streaming ASR (day 3)

Test whether VAD-gated batch transcription (wait for utterance end, send full audio) is faster than chunk-by-chunk streaming for typical banking turns (2–5 seconds). For Nemotron, test both its native streaming mode and simulated batch mode.

**Decision:** For utterances < 5s, batch is almost always faster due to full phonetic context. Use streaming only if VAD endpoint detection is unreliable.

## Task 2.5 — ablation D1: concurrent call scaling (days 4–5)

Run `load_test_concurrent.py` at concurrency levels 1, 5, 10, 15, 20, 25, 30. Use `--delay 0.5` to respect SIP CPS limits. Measure p50, p95, p99 turn latency and failure rate at each level. Plot the curve — find where latency starts degrading non-linearly. That inflection point is your production concurrency limit per instance.

**Also run ablation D2** (ThreadPoolExecutor size for ASR): `max_workers` ∈ {1, 2, 4, 6, 8} at fixed 20 concurrent calls. Find optimal workers without GPU OOM.

---

# Week 3 — kernel optimisations and LLM/TTS ablations

## Task 3.1 — ablation B1: vLLM vs SGLang prefix caching (day 1)

This is the second-highest-impact experiment. Run 10 sequential requests with the `POST_BOUNCE_AGENT` system prompt through vLLM (baseline), then repeat through SGLang. Log TTFT per request index (1, 2, 5, 10) to show cache warm-up.

**Expected result:** SGLang request 5+ TTFT ~109ms vs vLLM ~481ms. The 372ms savings per turn × 4 turns per call × 10k calls/day = enormous aggregate impact.

**Decision:** Use SGLang if TTFT on request 5+ is >150ms faster. Given your system prompt pattern, it almost certainly will be.

## Task 3.2 — ablation B2: AWQ int4 quantisation for Phi-3.5-mini (day 1)

Quantise Phi-3.5-mini to AWQ int4 using `autoawq`. Deploy via vLLM with `--quantization awq`. Measure TTFT, VRAM usage, and response quality on 20 banking prompts (score 1–5 for accuracy, tone, compliance, no hallucination).

**Decision:** Use AWQ if quality score ≥ 3.5/5 on all categories. The VRAM reduction (8GB → 2.5GB) is critical — it means Whisper and Phi-3.5-mini fit on a single g5.xlarge together, cutting infra cost.

## Task 3.3 — ablation B3: LLM quality eval (days 2–3)

Create 30 banking test prompts: 10 standard EMI reminders, 5 partial payment negotiation, 5 hardship claims, 5 wrong party/opt-out, 5 hostile/escalation. Run through Phi-3.5-mini (AWQ) and Llama-3.1-8B (AWQ). Score both on hallucination rate (most important for compliance) and banking task completion.

**Decision:** If Phi-3.5-mini hallucination rate < 5% and quality ≥ 3.5/5, use it — it's 4× faster TTFT and cheaper. If hallucination rate > 5% on amounts/dates, switch to Llama-3.1-8B.

## Task 3.4 — ablation C3: fix Kokoro warm-up (day 2)

Kokoro showed 2,545ms TTFB in benchmarks vs expected <100ms. Fix: pre-load ONNX session at server startup with a dummy synthesis call. Re-measure TTFB on requests 1, 2, 3, 5, 10. If warm TTFB is <120ms, Kokoro becomes the best TTS option (beating Piper's 314ms by 2.5×).

## Task 3.5 — ablation C2-f: Qwen3-TTS streaming (days 3–4)

Deploy Qwen3-TTS 0.6B via vLLM-Omni. Use the WebSocket streaming endpoint with `stream_audio=true`, `split_granularity="sentence"`, `response_format="pcm"`. Measure TTFB (target: 97ms), RTF, and MOS score against Piper.

Also test `faster-qwen3-tts` with CUDA graph capture. Compare TTFA: CUDA graph path (~160ms) vs standard dynamic cache (~770ms).

**Decision:** If Qwen3-TTS TTFB < 150ms warm and MOS ≥ 3.5, replace Piper. The 97ms TTFB vs Piper's 314ms is a 3.2× improvement on the TTS bottleneck.

## Task 3.6 — ablation C1: sentence chunking strategy (day 4)

Test four LLM→TTS bridging strategies: full response → single TTS call, split at sentence end, split at clause boundary, token streaming at every 5 tokens. Measure perceived TTFB and audio naturalness (does it sound choppy?). The winner is almost certainly sentence-level splitting.

## Task 3.7 — kernel optimisations (day 5)

Apply in sequence, measure each in isolation:

1. **Whisper:** `compute_type="int8_float16"` + `beam_size=1` (already in A1, confirm as permanent baseline)
2. **vLLM LLM:** `--gpu-memory-utilization 0.4` to co-locate with Whisper on same GPU
3. **Piper TTS:** Switch ONNX from CPU → CUDAExecutionProvider. Then try TensorRT EP (compile engine once, cache it). Expected TRT gain: 40–60% TTFB reduction.
4. **torch.compile:** Apply `torch.compile(model, mode="reduce-overhead")` to any remaining PyTorch models (StyleTTS2, Kokoro PyTorch fallback)

---

# Week 4 — LiveKit integration and plugin wrappers

## Task 4.1 — build three LiveKit plugin wrappers (days 1–2)

Create `plugins/` directory with three files implementing the LiveKit agents SDK interfaces:

- `plugins/asr_plugin.py` — `WhisperSTT(STT)` or `NemotronSTT(STT)`: calls your local ASR server, returns `SpeechEvent` with transcript
- `plugins/llm_plugin.py` — `LocalLLM(LLM)`: calls SGLang/vLLM with `stream=True`, always injects `POST_BOUNCE_AGENT` system prompt, yields `ChatChunk` objects token by token
- `plugins/tts_plugin.py` — `QwenTTS(TTS)` or `PiperTTS(TTS)`: calls local TTS server, yields `SynthesizedAudio` chunks at sentence boundaries

## Task 4.2 — surgical swap in pipeline_[agent.py](http://agent.py) (day 2)

Change only the session creation block (lines ~736–776). Everything else — disposition tracking, transcript capture, Kafka events, interaction service, recording manager — stays identical.

```python
# Before (3 lines to change)
stt = deepgram.STT(model="nova-2-general")
llm_model = openai.LLM(model="gpt-3.5-turbo")
tts = cartesia.TTS(voice="79a125e8...")

# After
from plugins.asr_plugin import NemotronSTT   # or WhisperSTT
from plugins.llm_plugin import LocalLLM
from plugins.tts_plugin import QwenTTS       # or PiperTTS

stt = NemotronSTT(server_url=os.getenv("ASR_SERVER_URL"))
llm_model = LocalLLM(server_url=os.getenv("LLM_SERVER_URL"), system_prompt=formatted_instructions)
tts = QwenTTS(server_url=os.getenv("TTS_SERVER_URL"))
```

Also add `attach_latency_probe(session, agent, call_id)` immediately after session creation.

## Task 4.3 — audio frame re-chunking (day 2)

LiveKit expects `AudioFrame` at 20ms intervals (320 samples at 16kHz). Piper/Qwen3-TTS outputs variable-length PCM. Build the frame buffer adapter:

```python
FRAME_BYTES = 640  # 320 samples × 2 bytes (int16)
async def pcm_to_livekit_frames(pcm_stream) -> AsyncIterator[AudioFrame]:
    buffer = bytearray()
    async for chunk in pcm_stream:
        buffer.extend(chunk)
        while len(buffer) >= FRAME_BYTES:
            yield AudioFrame(data=bytes(buffer[:FRAME_BYTES]), sample_rate=16000, num_channels=1)
            del buffer[:FRAME_BYTES]
```

## Task 4.4 — single call smoke test on Lightning AI (day 3)

Start all 5 terminals: ASR server, LLM server, TTS server, queue processor, agent worker, API. Fire one real call via `load_test_concurrent.py --calls 1`. Listen: does greeting play within 2s? Does agent respond within 1s? Does it know the customer's EMI amount? No clicks or gaps in audio?

After call: `cat /tmp/latency_<call_id>.json` — verify per-turn latency logged.

## Task 4.5 — 20 concurrent calls live test (days 4–5)

Run `load_test_concurrent.py --calls 20 --delay 0.5`. Monitor GPU utilisation with `nvidia-smi` in a separate terminal. After all calls finish: run `benchmark_baseline.py parse-logs`. Save the result JSON.

**Targets for this to be a go:**

- p50 turn latency < 800ms
- p95 turn latency < 1,500ms
- Failure rate < 5%
- GPU utilisation 60–85% (below = CPU bottleneck, above = GPU bottleneck)

---

# Week 5 — SageMaker production deployment

## Task 5.1 — deploy LLM on SageMaker LMI (days 1–2)

Use SageMaker Large Model Inference containers (run vLLM internally). Deploy Phi-3.5-mini AWQ to `ml.g5.xlarge` in `ap-south-1` (Mumbai) for India data residency:

```python
hub = {
    'HF_MODEL_ID': 'microsoft/Phi-3.5-mini-instruct',
    'HF_MODEL_QUANTIZE': 'awq',
    'SM_NUM_GPUS': '1',
    'MAX_INPUT_LENGTH': '2048',
}
huggingface_model.deploy(instance_type='ml.g5.xlarge', endpoint_name='krim-phi35-awq')
```

Change `LLM_SERVER_URL` in `.env.local` to the SageMaker endpoint URL. The `LocalLLM` plugin requires no code changes — it's the same OpenAI-compatible API.

## Task 5.2 — deploy ASR on SageMaker (day 2)

Deploy Nemotron (or Whisper turbo) as a custom SageMaker endpoint with a NeMo container. Alternatively, if Lightning AI latency from SageMaker is acceptable, keep ASR on Lightning and only move LLM to SageMaker.

## Task 5.3 — ablation E1: co-location vs separate instances (day 3)

Test three deployment topologies at 20 concurrent calls:

- **E1-a:** ASR on g5.xlarge + LLM on g5.xlarge (two instances, $1,466/month)
- **E1-b:** ASR + LLM co-located on g5.2xlarge with AWQ ($885/month)
- **E1-c:** ASR + LLM + TTS all on g5.4xlarge ($1,186/month)

Measure e2e latency and cost per call at each topology. Expected winner: E1-b (AWQ allows co-location, saves $581/month).

## Task 5.4 — ablation E2: full pipeline latency budget (days 4–5)

This is the final integration test. Run the fully optimised pipeline (best config from all A/B/C/D/E experiments) on 100 real calls. Fill in the latency budget:

| Stage | Target | Actual p50 | Actual p95 |
| --- | --- | --- | --- |
| VAD endpoint | 30ms | ? | ? |
| ASR (Nemotron 160ms chunk) | 80ms | ? | ? |
| LLM TTFT (Phi-3.5 AWQ, SGLang) | 80ms | ? | ? |
| TTS TTFB (Qwen3-TTS 0.6B) | 100ms | ? | ? |
| LiveKit audio frame buffer | 20ms | ? | ? |
| **Total e2e turn** | **<400ms** | **?** | **?** |

**Ship criteria:** p50 < 600ms ✅ · p50 600–800ms acceptable · p50 > 800ms → find bottleneck and run targeted ablation.

---

# Experiment sequence summary

Every experiment saves a JSON to `results/exp_<id>_<n>.json`. Results feed into the final benchmark report.

| ID | Name | Week | Depends on | Key decision |
| --- | --- | --- | --- | --- |
| A1 | Whisper beam_size ablation | 2 | Real audio (1.1) | beam_size=1 wins? |
| A2-e | Nemotron vs Whisper | 2 | NeMo setup | Switch ASR model? |
| A3 | Batch vs streaming ASR | 2 | A2-e | Batch for <5s turns? |
| B1 | vLLM vs SGLang prefix cache | 3 | LLM servers warm | Switch to SGLang? |
| B2 | AWQ int4 Phi-3.5-mini | 3 | autoawq install | Use AWQ in prod? |
| B3 | LLM quality eval on banking | 3 | B2 | Phi vs Llama? |
| B4 | max_tokens + temperature | 3 | B3 | Tune generation params |
| C1 | Sentence chunking strategy | 3 | Streaming pipeline | Sentence-level wins? |
| C2-f | Qwen3-TTS vs Piper | 3 | vLLM-Omni setup | Switch TTS model? |
| C3 | Kokoro warm-up fix | 3 | C2-f | Kokoro as alt TTS? |
| D1 | Concurrent call scaling | 2 | Streaming pipeline | Max concurrency/instance |
| D2 | ThreadPool size ablation | 2 | D1 | Optimal workers |
| D3 | Connection pool size | 2 | D1 | Optimal pool size |
| E1 | Co-location topology | 5 | SageMaker deploy | Which instance config? |
| E2 | Full pipeline latency budget | 5 | All above | Ship decision |

---

# Compliance checklist (non-negotiable)

Check each of these before any production deployment. These are RBI/DPDP requirements.

- [ ]  All ASR inference on-prem or ap-south-1 (no audio to foreign cloud)
- [ ]  All LLM inference on-prem or ap-south-1 (no prompts to OpenAI/Anthropic)
- [ ]  All TTS inference on-prem or ap-south-1 (no text to Cartesia/ElevenLabs)
- [ ]  PII redaction (Aadhaar, account numbers) added before Kafka publish
- [ ]  `model_id` field added to Kafka event schema in `models/events.py`
- [ ]  Output guardrail: regex check on LLM response for numeric consistency (amounts, dates)
- [ ]  Hallucination rate < 5% on banking test set (from B3)
- [ ]  vLLM request logging disabled in production (`--disable-log-requests`)

---

# Final recommended stack (based on all benchmarks)

This is what you are building toward. Every ablation experiment is designed to validate or invalidate this recommendation.

| Component | Model | Serving | Instance | Monthly cost |
| --- | --- | --- | --- | --- |
| VAD | Silero (unchanged) | Runs in agent process | — | $0 |
| ASR | Nemotron-speech-streaming-0.6b | NeMo / NIM | g5.xlarge | $733 |
| LLM | Phi-3.5-mini-instruct AWQ | SGLang (RadixAttention) | g5.xlarge (co-located) | $0 extra |
| TTS | Qwen3-TTS-0.6B | vLLM-Omni WebSocket | g4dn.xlarge | $384 |
| Queue/API | Existing FastAPI + PostgreSQL | t3.medium | $30 |  |
| Kafka | Existing | t3.small | $15 |  |
| **Total** |  |  |  | **$1,162/month** |

**vs current SaaS baseline: ~$4,270/month → saving $3,108/month (73%).** Fully on-prem. RBI/DPDP compliant. Estimated warm turn latency: 290–380ms (VAD 30 + Nemotron 80 + Phi SGLang 80 + Qwen3-TTS 97 + framing 20 = 307ms theoretical).

---

# Daily log template

For every working day, save `logs/YYYY-MM-DD.md`:

```markdown
## Date: YYYY-MM-DD

### What I did
- Task X.Y: [description]

### Results
- Experiment A1-c: TTFT = 134ms, WER = 4.2% → WINNER
- [link to results JSON]

### Blockers
- Nemotron NeMo container OOM on g4dn.xlarge — need g5.xlarge

### Tomorrow
- Task 2.3: deploy Nemotron on g5.xlarge
```

This log is your internship deliverable. Keep it updated every day.