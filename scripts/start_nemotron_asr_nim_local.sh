#!/usr/bin/env bash
set -euo pipefail

# Launch a local NVIDIA NIM ASR container for Nemotron streaming.
: "${NGC_API_KEY:=}"
: "${NEMOTRON_NIM_IMAGE:=nvcr.io/nim/nvidia/nemotron-asr-streaming-riva-v1:latest}"
: "${NEMOTRON_NIM_CONTAINER:=nemotron-asr-nim}"
: "${NEMOTRON_NIM_HTTP_PORT:=9000}"
: "${NEMOTRON_NIM_GRPC_PORT:=50051}"
: "${NEMOTRON_NIM_CACHE_DIR:=models/nim_cache}"
: "${NEMOTRON_NIM_TAGS_SELECTOR:=mode=str,vad=silero}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${NEMOTRON_NIM_CACHE_DIR}" = /* ]]; then
  CACHE_DIR="${NEMOTRON_NIM_CACHE_DIR}"
else
  CACHE_DIR="${REPO_ROOT}/${NEMOTRON_NIM_CACHE_DIR}"
fi

if [[ -z "${NGC_API_KEY}" ]]; then
  cat <<'ERR'
NGC_API_KEY is required.

Export it and rerun:
  export NGC_API_KEY=<your_ngc_key>
  scripts/start_nemotron_asr_nim_local.sh
ERR
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not in PATH." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. NVIDIA driver/runtime is required." >&2
  exit 1
fi

mkdir -p "${CACHE_DIR}"

echo "Logging in to nvcr.io..."
echo "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin >/dev/null

if docker ps -a --format '{{.Names}}' | grep -Fxq "${NEMOTRON_NIM_CONTAINER}"; then
  echo "Removing existing container ${NEMOTRON_NIM_CONTAINER}..."
  docker rm -f "${NEMOTRON_NIM_CONTAINER}" >/dev/null
fi

echo "Starting ${NEMOTRON_NIM_IMAGE}..."
docker run -d \
  --gpus all \
  --name "${NEMOTRON_NIM_CONTAINER}" \
  --shm-size=8g \
  -e NGC_API_KEY="${NGC_API_KEY}" \
  -e NIM_HTTP_API_PORT="${NEMOTRON_NIM_HTTP_PORT}" \
  -e NIM_GRPC_API_PORT="${NEMOTRON_NIM_GRPC_PORT}" \
  -e NIM_TAGS_SELECTOR="${NEMOTRON_NIM_TAGS_SELECTOR}" \
  -e NIM_CACHE_PATH=/opt/nim/.cache \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p "${NEMOTRON_NIM_HTTP_PORT}:${NEMOTRON_NIM_HTTP_PORT}" \
  -p "${NEMOTRON_NIM_GRPC_PORT}:${NEMOTRON_NIM_GRPC_PORT}" \
  -v "${CACHE_DIR}:/opt/nim/.cache" \
  "${NEMOTRON_NIM_IMAGE}" >/dev/null

cat <<INFO
Nemotron NIM container started.

Container name: ${NEMOTRON_NIM_CONTAINER}
HTTP endpoint : http://127.0.0.1:${NEMOTRON_NIM_HTTP_PORT}
GRPC endpoint : 127.0.0.1:${NEMOTRON_NIM_GRPC_PORT}

Follow logs:
  docker logs -f ${NEMOTRON_NIM_CONTAINER}

Stop container:
  docker rm -f ${NEMOTRON_NIM_CONTAINER}

Smoke test (after model warm-up):
  curl -sS -X POST "http://127.0.0.1:${NEMOTRON_NIM_HTTP_PORT}/v1/audio/transcriptions" \
    -F "model=nvidia/nemotron-asr-streaming" \
    -F "language=en" \
    -F "file=@data/hf_sample_wavs/audio_000.wav"
INFO
