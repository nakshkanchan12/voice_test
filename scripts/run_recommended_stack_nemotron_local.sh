#!/usr/bin/env bash
set -euo pipefail

# Run recommended pipeline with local Nemotron NIM ASR + local LLM/TTS.
: "${NEMOTRON_ASR_URL:=http://127.0.0.1:9000}"
: "${NEMOTRON_ASR_MODEL:=nvidia/nemotron-asr-streaming}"
: "${WAIT_FOR_HEALTH:=0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ASR_URL="${NEMOTRON_ASR_URL}" \
ASR_TRANSCRIBE_PATH="/v1/audio/transcriptions" \
ASR_AUDIO_FIELD="file" \
ASR_MODEL="${NEMOTRON_ASR_MODEL}" \
WAIT_FOR_HEALTH="${WAIT_FOR_HEALTH}" \
"${SCRIPT_DIR}/run_recommended_stack.sh"
