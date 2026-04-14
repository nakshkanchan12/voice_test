#!/usr/bin/env bash
set -euo pipefail

: "${CALLS:=20}"
: "${CONCURRENCY:=5}"
: "${SAMPLE_POOL:=5}"

: "${ASR_URL:=http://127.0.0.1:8011}"
: "${ASR_TRANSCRIBE_PATH:=/transcribe}"
: "${ASR_AUDIO_FIELD:=audio_file}"
: "${ASR_MODEL:=}"
: "${ASR_API_KEY:=}"
: "${LLM_BASE_URL:=http://127.0.0.1:30000/v1}"
: "${LLM_MODEL:=HuggingFaceTB/SmolLM2-360M-Instruct}"
: "${TTS_URL:=http://127.0.0.1:8012}"
: "${TTS_MODE:=http_synthesize}"
: "${TTS_MODEL:=piper/en_US-lessac-medium}"
: "${TRANSPORT:=http}"
: "${ASR_WEBRTC_OFFER_URL:=}"
: "${LLM_WEBRTC_OFFER_URL:=}"
: "${TTS_WEBRTC_OFFER_URL:=}"
: "${WAIT_FOR_HEALTH:=1}"

# Full-duplex defaults
: "${LLM_STREAM_MODE:=aggressive}"
: "${LLM_AGGRESSIVE_MIN_TOKENS:=5}"
: "${ENABLE_BARGE_IN:=1}"
: "${BARGE_IN_MIN_SPEECH_MS:=120}"
: "${VAD_STREAMING:=1}"
: "${VAD_PARTIAL_SEGMENT_MS:=320}"
: "${VAD_MAX_STREAM_SEGMENT_MS:=1280}"

extra_flags=()
if [[ "${WAIT_FOR_HEALTH}" == "1" ]]; then
  extra_flags+=(--wait-for-health)
fi

python -m voice_pipeline.recommended_pipeline \
  --calls "${CALLS}" \
  --concurrency "${CONCURRENCY}" \
  --sample-pool "${SAMPLE_POOL}" \
  --transport "${TRANSPORT}" \
  --asr-url "${ASR_URL}" \
  --asr-transcribe-path "${ASR_TRANSCRIBE_PATH}" \
  --asr-audio-field "${ASR_AUDIO_FIELD}" \
  --asr-model "${ASR_MODEL}" \
  --asr-api-key "${ASR_API_KEY}" \
  --asr-webrtc-offer-url "${ASR_WEBRTC_OFFER_URL}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --llm-webrtc-offer-url "${LLM_WEBRTC_OFFER_URL}" \
  --llm-model "${LLM_MODEL}" \
  --llm-stream-mode "${LLM_STREAM_MODE}" \
  --llm-aggressive-min-tokens "${LLM_AGGRESSIVE_MIN_TOKENS}" \
  --tts-url "${TTS_URL}" \
  --tts-webrtc-offer-url "${TTS_WEBRTC_OFFER_URL}" \
  --tts-mode "${TTS_MODE}" \
  --tts-model "${TTS_MODEL}" \
  --enable-barge-in "${ENABLE_BARGE_IN}" \
  --barge-in-min-speech-ms "${BARGE_IN_MIN_SPEECH_MS}" \
  --vad-streaming "${VAD_STREAMING}" \
  --vad-partial-segment-ms "${VAD_PARTIAL_SEGMENT_MS}" \
  --vad-max-stream-segment-ms "${VAD_MAX_STREAM_SEGMENT_MS}" \
  "${extra_flags[@]}"
