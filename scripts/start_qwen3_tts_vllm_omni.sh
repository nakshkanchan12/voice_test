#!/usr/bin/env bash
set -euo pipefail

: "${QWEN3_TTS_MODEL:=Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
: "${QWEN3_TTS_PORT:=12001}"
: "${QWEN3_TTS_HOST:=0.0.0.0}"
: "${QWEN3_TTS_DTYPE:=float16}"
: "${QWEN3_TTS_GPU_MEM_UTIL:=0.45}"

python -m vllm_omni.entrypoints.openai.api_server \
  --host "${QWEN3_TTS_HOST}" \
  --port "${QWEN3_TTS_PORT}" \
  --model "${QWEN3_TTS_MODEL}" \
  --dtype "${QWEN3_TTS_DTYPE}" \
  --gpu-memory-utilization "${QWEN3_TTS_GPU_MEM_UTIL}" \
  --enable-auto-tool-choice
