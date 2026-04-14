#!/usr/bin/env bash
set -euo pipefail

cat <<'INFO'
Nemotron ASR startup options:

1) Local GPU NIM container:
   scripts/start_nemotron_asr_nim_local.sh

2) Remote hosted/build endpoint:
   set ASR_URL / ASR_TRANSCRIBE_PATH / ASR_AUDIO_FIELD / ASR_MODEL / ASR_API_KEY
   and run scripts/run_recommended_stack.sh

3) Local hybrid runner pre-wired for NIM OpenAI audio endpoint:
   scripts/run_recommended_stack_nemotron_local.sh

Expected ASR API for this pipeline:
  POST /transcribe (multipart audio_file) OR
  POST /v1/audio/transcriptions (multipart file, model)
  response JSON containing text/transcript
INFO
