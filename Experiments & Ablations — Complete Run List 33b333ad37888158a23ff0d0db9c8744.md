# Experiments & Ablations — Complete Run List

Every experiment and ablation you need to run, in order. Each has a hypothesis, what to measure, how to measure it, and what the result means for your decision.

**Rule:** Change one variable at a time. Every experiment gets its own result JSON saved to `results/exp_<id>_<name>.json`.

---

# Experiment group A — ASR optimisations

## A1 — Whisper beam size ablation

**Hypothesis:** Reducing beam_size from 5 to 1 (greedy decoding) cuts ASR latency by 30–40% with <2% WER increase on banking audio.

**Variants:**

| Variant | beam_size | best_of | compute_type |
| --- | --- | --- | --- |
| A1-a (baseline) | 5 | 5 | float16 |
| A1-b | 3 | 3 | float16 |
| A1-c | 1 | 1 | float16 |
| A1-d | 1 | 1 | int8_float16 |

**Measure:** Latency (ms), WER on 20 banking utterances (real recorded audio)

**Decision rule:** Use the smallest beam_size where WER stays < 5% on banking vocab. If A1-c WER < 5%, use it. If not, use A1-b.

## A2 — ASR model size ablation

**Hypothesis:** Whisper `large-v3-turbo` gives the best latency/WER tradeoff. Parakeet-TDT 1.1B is faster but may have higher WER on Indian-accented English.

**Variants:**

| Variant | Model | Engine |
| --- | --- | --- |
| A2-a | Whisper large-v3-turbo | faster-whisper int8_float16 |
| A2-b | Whisper medium | faster-whisper int8_float16 |
| A2-c | Parakeet-TDT 1.1B | NeMo |
| A2-d | MMS-1b-all | HuggingFace |

**Measure:** Latency (ms), WER on: (1) clean English, (2) noisy English, (3) Indian-accented English (most important)

**Decision rule:** Fastest model with WER < 5% on Indian-accented banking audio wins.

## A3 — Streaming vs batch ASR

**Hypothesis:** VAD-gated batch transcription (wait for end of utterance, then transcribe) is lower latency than chunk-by-chunk streaming for short utterances (<5s).

**Variants:**

- A3-a: Batch — VAD detects end, send full utterance to Whisper
- A3-b: Streaming — send 500ms chunks, merge partial transcripts
- A3-c: Streaming with endpointing — WhisperLive-style server

**Measure:** Time from end of speech to final transcript available

**Decision rule:** For utterances < 5s (typical banking turn), batch is usually faster due to context. Use streaming only if VAD endpoint detection is unreliable.

---

# Experiment group B — LLM optimisations

## B1 — vLLM vs SGLang prefix caching

**Hypothesis:** SGLang RadixAttention reduces TTFT by 200–400ms after the first call due to system prompt caching. Effect is larger for longer system prompts.

**Protocol:**

1. Start vLLM server, run 10 sequential requests with same system prompt, log TTFT per request
2. Restart, run SGLang server, repeat
3. Compare TTFT on requests 1, 2, 5, 10 (shows cache warm-up effect)

**Measure:** TTFT (ms) per request index, memory usage

**Decision rule:** If SGLang TTFT on request 5+ is >150ms faster than vLLM, use SGLang.

## B2 — Quantisation ablation (Phi-3.5-mini)

**Hypothesis:** AWQ int4 reduces TTFT and VRAM by ~40% with <3% quality degradation on banking responses.

**Variants:**

| Variant | Precision | VRAM est. | Engine |
| --- | --- | --- | --- |
| B2-a | float16 | 8 GB | vLLM |
| B2-b | AWQ int4 | 2.5 GB | vLLM |
| B2-c | GPTQ int4 | 2.5 GB | vLLM |
| B2-d | float16 + torch.compile | 8 GB | vLLM |

**Measure:** TTFT (ms), total generation time, response quality score (human eval on 20 banking prompts: 1–5 scale for accuracy and tone)

**Decision rule:** Use AWQ if quality score >= float16 baseline − 0.5 points. VRAM saving is critical for co-locating ASR + LLM on same GPU.

## B3 — LLM model quality ablation (banking task)

**Hypothesis:** Phi-3.5-mini has sufficient quality for scripted banking conversations. Llama-3.1-8B is better for edge cases but not needed for standard EMI calls.

**Protocol:** Create 30 banking test prompts across 5 categories:

- Standard EMI reminder (10 prompts)
- Partial payment negotiation (5 prompts)
- Hardship claim handling (5 prompts)
- Wrong party / opt-out (5 prompts)
- Hostile / escalation scenarios (5 prompts)

Run each prompt through both models, score on: factual accuracy (amounts, dates correct), tone (professional, empathetic), compliance (no promises the bank can’t keep), hallucination (invented information).

**Measure:** Human eval score per category (1–5), hallucination rate (%)

**Decision rule:** Use Phi-3.5-mini if score >= 3.5/5 on all categories and hallucination rate < 5%. Otherwise use Llama-3.1-8B.

## B4 — Max tokens and temperature ablation

**Hypothesis:** Banking responses should be short (< 50 tokens). Setting `max_tokens=80` with `temperature=0.3` reduces generation time and hallucination without losing quality.

**Variants:** `max_tokens` ∈ {50, 80, 150, 256} × `temperature` ∈ {0.1, 0.3, 0.7}

**Measure:** Total generation time, response length (tokens), truncation rate (% of responses cut off)

---

# Experiment group C — TTS optimisations

## C1 — Sentence chunking strategy

**Hypothesis:** Splitting LLM output at sentence boundaries and starting TTS on the first sentence reduces perceived TTFB by 300–500ms vs waiting for the full response.

**Variants:**

- C1-a: Full response → single TTS call
- C1-b: Split at `['.', '!', '?']` → TTS per sentence
- C1-c: Split at `['.', '!', '?', ',']` → TTS per clause (more aggressive)
- C1-d: Token streaming → TTS at every 5 tokens (most aggressive, may sound choppy)

**Measure:** Time from LLM first token to first audio byte (perceived TTFB), audio naturalness score (does it sound choppy?)

**Decision rule:** Use C1-b if naturalness score >= 4/5. C1-c if speed is critical and naturalness >= 3.5/5.

## C2 — TTS engine ablation

**Variants:**

| Variant | Model | Engine | Expected TTFB |
| --- | --- | --- | --- |
| C2-a | Piper | ONNX CPU | ~314ms |
| C2-b | Piper | ONNX CUDA EP | ~150ms |
| C2-c | Piper | TensorRT EP | ~80ms |
| C2-d | Kokoro v0.19 | ONNX (pre-loaded) | ~80ms |
| C2-e | StyleTTS2 | PyTorch + sentence chunk | ~450ms per chunk |

**Measure:** TTFB (ms) per sentence, RTF, MOS score (Mean Opinion Score) — record and rate naturalness on a 1–5 scale with 5 listeners

**Decision rule:** Use fastest model with MOS >= 3.5/5.

## C3 — Kokoro warm-up investigation

**Hypothesis:** Kokoro’s 2,545ms TTFB was caused by ONNX session initialisation on first call, not the model itself. Pre-loading at startup should bring TTFB to <100ms.

**Protocol:**

1. Start Kokoro server, measure TTFB on requests 1, 2, 3, 5, 10
2. Confirm TTFB stabilises after request 1
3. Add `model.synthesize(" ")` dummy call at startup
4. Re-measure TTFB on request 1

**Measure:** TTFB per request index (before and after pre-loading fix)

---

# Experiment group D — concurrency and throughput

## D1 — Concurrent call scaling

**Hypothesis:** The pipeline handles 20 concurrent calls with p95 turn latency < 1,500ms. Latency degrades gracefully (linearly, not exponentially) as concurrency increases.

**Protocol:** Use `load_test_concurrent.py`. Run at concurrency levels: 1, 5, 10, 15, 20, 25, 30. Measure p50, p95, p99 turn latency and failure rate at each level. Plot latency vs concurrency.

**Decision rule:** Find the maximum concurrency where p95 < 1,500ms. This is your production concurrency limit per instance.

## D2 — Thread pool size ablation (ASR)

**Hypothesis:** Optimal `ThreadPoolExecutor` size for faster-whisper is 4 workers on a g5.xlarge (1 GPU). More workers causes GPU memory contention; fewer causes queuing.

**Variants:** `max_workers` ∈ {1, 2, 4, 6, 8} at 20 concurrent calls

**Measure:** Mean ASR latency, GPU utilisation (nvidia-smi), queue wait time

**Decision rule:** Use the max_workers that gives lowest mean ASR latency without GPU OOM.

## D3 — Connection pool size ablation

**Hypothesis:** `TCPConnector(limit=30)` is sufficient for 20 concurrent calls to vLLM. Smaller limits cause connection queuing; larger limits waste resources.

**Variants:** `limit` ∈ {10, 20, 30, 50} at 20 concurrent calls

**Measure:** HTTP connection wait time (time from request dispatch to connection acquired), request failure rate

---

# Experiment group E — end-to-end pipeline

## E1 — Co-location vs separate instances

**Hypothesis:** Running ASR + LLM on the same g5.xlarge (with AWQ quantisation) is cheaper and has similar latency to running on separate instances.

**Variants:**

- E1-a: ASR on g5.xlarge + LLM on separate g5.xlarge (two instances, $1,466/month)
- E1-b: ASR + LLM co-located on g5.2xlarge (one instance, $885/month)
- E1-c: ASR + LLM + TTS all on g5.4xlarge (one instance, $1,186/month)

**Measure:** E2E turn latency, cost per call, GPU memory headroom at 20 concurrent calls

## E2 — Full pipeline latency budget

This is the final integration test. Run the full optimised pipeline (best config from all A/B/C/D experiments) on 100 calls and produce a latency budget breakdown:

| Stage | Target | Actual p50 | Actual p95 |
| --- | --- | --- | --- |
| VAD endpoint detection | 30ms | ? | ? |
| ASR (Whisper turbo int8, beam=1) | 180ms | ? | ? |
| LLM TTFT (Phi-3.5-mini AWQ, SGLang) | 80ms | ? | ? |
| TTS TTFB (Piper, sentence 1, TRT-EP) | 80ms | ? | ? |
| Audio frame buffering + LiveKit send | 20ms | ? | ? |
| **Total e2e turn latency** | **<400ms** | ? | ? |

**Decision rule:** If p50 < 600ms, ship it. If p50 600–800ms, acceptable. If p50 > 800ms, find the bottleneck and run targeted ablation.

---

# Ablation result template

Save every experiment as:

```json
{
  "experiment_id": "B1",
  "hypothesis": "SGLang RadixAttention reduces TTFT after cache warm",
  "date": "2026-04-08",
  "compute": "Lightning AI / 1x A10G",
  "variants": [
    {
      "name": "vLLM baseline",
      "config": {"engine": "vllm", "model": "phi-3.5-mini"},
      "results": {"ttft_p50_ms": 481, "ttft_p95_ms": 620, "ttft_req10_ms": 465}
    },
    {
      "name": "SGLang RadixAttention",
      "config": {"engine": "sglang", "model": "phi-3.5-mini"},
      "results": {"ttft_p50_ms": 109, "ttft_p95_ms": 145, "ttft_req10_ms": 98}
    }
  ],
  "winner": "SGLang RadixAttention",
  "decision": "Use SGLang for all LLM serving. Cache warm effect confirmed."
}
```