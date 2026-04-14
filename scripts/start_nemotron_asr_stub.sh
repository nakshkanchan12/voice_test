#!/usr/bin/env bash
set -euo pipefail

cat <<'INFO'
Nemotron ASR server startup is environment-specific (NeMo or NIM deployment).
Use your deployed Nemotron endpoint and make sure it exposes:
  POST /transcribe (multipart audio_file)
and returns JSON with either {"text": "..."} or {"transcript": "..."}

Then point the pipeline at:
  --asr-url http://<nemotron-host>:<port>
INFO
