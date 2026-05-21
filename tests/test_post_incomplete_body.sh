#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
QWEN3_TTS_DATA_ROOT="${QWEN3_TTS_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/qwen3-tts}"

SERVER_BIN="${1:-${REPO_ROOT}/build/qwen3-tts-server}"
MODEL_DIR="${QWEN3_TTS_MODEL_DIR:-${QWEN3_TTS_DATA_ROOT}/models}"
VOICE_DIR="${QWEN3_TTS_VOICE_DIR:-${QWEN3_TTS_DATA_ROOT}/voices}"
PORT="${QWEN3_TTS_TEST_PORT:-19000}"
LOG_FILE="${QWEN3_TTS_TEST_LOG:-/tmp/qwen3-tts-post-incomplete.log}"
FULL_BODY='{"text":"你好","language":"zh","voice":"lina"}'
PARTIAL_BODY='{"text":"你好","language":"zh"'

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_health() {
    local deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do
        local health
        health="$(curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" || true)"
        if [[ "$health" == *'"status":"ok"'* ]]; then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_idle() {
    local deadline=$((SECONDS + 10))
    while (( SECONDS < deadline )); do
        local health
        health="$(curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" || true)"
        if [[ "$health" == *'"busy":false'* && "$health" == *'"queue":0'* ]]; then
            return 0
        fi
        sleep 1
    done
    return 1
}

rm -f "$LOG_FILE" /tmp/qwen3-tts-post-incomplete.response
"$SERVER_BIN" -m "$MODEL_DIR" -v "$VOICE_DIR" -p "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

wait_for_health || {
    printf 'server did not become healthy\n' >&2
    cat "$LOG_FILE" >&2
    exit 1
}

{
    printf 'POST /api/tts HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n' "${#FULL_BODY}"
    printf '%s' "$PARTIAL_BODY"
} | socat - "TCP:127.0.0.1:${PORT}" >/tmp/qwen3-tts-post-incomplete.response || true

wait_for_idle || {
    printf 'server did not return to idle after incomplete POST\n' >&2
    cat "$LOG_FILE" >&2
    exit 1
}

if grep -q '\[TTS\] ' "$LOG_FILE"; then
    printf 'incomplete POST body was incorrectly synthesized\n' >&2
    cat "$LOG_FILE" >&2
    exit 1
fi

printf 'incomplete POST body rejected cleanly\n'
