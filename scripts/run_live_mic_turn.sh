#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_SOURCE:=mic}"
: "${MIC_SECONDS:=5}"
: "${MIC_SAMPLE_RATE_HZ:=16000}"
: "${MIC_CHANNELS:=1}"
: "${MIC_DEVICE:=}"
: "${AUDIO_CHUNK_MS:=20}"
: "${CALL_ID:=live-call-1}"
: "${INTERACTIVE_CALL:=0}"
: "${MAX_TURNS:=0}"
: "${METRICS_JSONL:=}"
: "${METRICS_SUMMARY_JSON:=}"

: "${ASR_URL:=http://127.0.0.1:8011}"
: "${LLM_BASE_URL:=http://127.0.0.1:30000/v1}"
: "${LLM_MODEL:=HuggingFaceTB/SmolLM2-360M-Instruct}"
: "${LLM_API_KEY:=EMPTY}"
: "${LLM_STREAM_MODE:=aggressive}"
: "${LLM_AGGRESSIVE_MIN_TOKENS:=5}"

: "${TTS_URL:=http://127.0.0.1:8012}"
: "${TRANSPORT:=http}"
: "${ASR_WEBRTC_OFFER_URL:=}"
: "${LLM_WEBRTC_OFFER_URL:=}"
: "${TTS_WEBRTC_OFFER_URL:=}"

: "${VAD_STREAMING:=1}"
: "${VAD_PARTIAL_SEGMENT_MS:=320}"
: "${VAD_MAX_STREAM_SEGMENT_MS:=1280}"
: "${ENABLE_BARGE_IN:=1}"
: "${BARGE_IN_MIN_SPEECH_MS:=120}"

: "${TTS_SAMPLE_RATE_HZ:=22050}"
: "${OUTPUT_WAV:=results/live_turn_mic_output.wav}"

python -m voice_pipeline.live_turn \
  --dataset-source "${DATASET_SOURCE}" \
  --call-id "${CALL_ID}" \
  --interactive-call "${INTERACTIVE_CALL}" \
  --max-turns "${MAX_TURNS}" \
  --metrics-jsonl "${METRICS_JSONL}" \
  --metrics-summary-json "${METRICS_SUMMARY_JSON}" \
  --mic-seconds "${MIC_SECONDS}" \
  --mic-sample-rate-hz "${MIC_SAMPLE_RATE_HZ}" \
  --mic-channels "${MIC_CHANNELS}" \
  --mic-device "${MIC_DEVICE}" \
  --audio-chunk-ms "${AUDIO_CHUNK_MS}" \
  --asr-url "${ASR_URL}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --llm-model "${LLM_MODEL}" \
  --llm-api-key "${LLM_API_KEY}" \
  --llm-stream-mode "${LLM_STREAM_MODE}" \
  --llm-aggressive-min-tokens "${LLM_AGGRESSIVE_MIN_TOKENS}" \
  --tts-url "${TTS_URL}" \
  --transport "${TRANSPORT}" \
  --asr-webrtc-offer-url "${ASR_WEBRTC_OFFER_URL}" \
  --llm-webrtc-offer-url "${LLM_WEBRTC_OFFER_URL}" \
  --tts-webrtc-offer-url "${TTS_WEBRTC_OFFER_URL}" \
  --vad-streaming "${VAD_STREAMING}" \
  --vad-partial-segment-ms "${VAD_PARTIAL_SEGMENT_MS}" \
  --vad-max-stream-segment-ms "${VAD_MAX_STREAM_SEGMENT_MS}" \
  --enable-barge-in "${ENABLE_BARGE_IN}" \
  --barge-in-min-speech-ms "${BARGE_IN_MIN_SPEECH_MS}" \
  --tts-sample-rate-hz "${TTS_SAMPLE_RATE_HZ}" \
  --output-wav "${OUTPUT_WAV}"
