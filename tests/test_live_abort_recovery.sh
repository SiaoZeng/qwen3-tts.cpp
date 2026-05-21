#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
QWEN3_TTS_DATA_ROOT="${QWEN3_TTS_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/qwen3-tts}"

SERVER_BIN="${1:-${REPO_ROOT}/build/qwen3-tts-server}"
MODEL_DIR="${QWEN3_TTS_MODEL_DIR:-${QWEN3_TTS_DATA_ROOT}/models}"
VOICE_DIR="${QWEN3_TTS_VOICE_DIR:-${QWEN3_TTS_DATA_ROOT}/voices}"
PORT="${QWEN3_TTS_TEST_PORT:-19000}"
LOG_FILE="${QWEN3_TTS_TEST_LOG:-/tmp/qwen3-tts-live-abort.log}"
TEXT="${QWEN3_TTS_TEST_TEXT:-夜裡的巷口有微微的風。小宇問母親，女人真正喜歡的是什麼。母親想了想，輕聲說，很多女人喜歡的，不只是浪漫的話，而是一個人願意長久地溫柔。是在她沉默時，仍然耐心陪伴；是在她不安時，給她安心；是在平凡的日子裡，還記得她的小習慣、小期待。花會凋謝，晚餐會吃完，可是被珍惜、被理解、被真心對待，會一直留在心裡。小宇望著夜色，忽然明白，最動人的愛，不是轟轟烈烈，而是有人一直把你放在心上。}"

cleanup() {
    if [[ -n "${LIVE_PID:-}" ]] && kill -0 "$LIVE_PID" 2>/dev/null; then
        kill "$LIVE_PID" 2>/dev/null || true
        wait "$LIVE_PID" 2>/dev/null || true
    fi
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
            printf '%s\n' "$health"
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_idle() {
    local deadline=$((SECONDS + 30))
    while (( SECONDS < deadline )); do
        local health
        health="$(curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" || true)"
        if [[ "$health" == *'"busy":false'* && "$health" == *'"queue":0'* ]]; then
            printf '%s\n' "$health"
            return 0
        fi
        sleep 1
    done
    return 1
}

rm -f "$LOG_FILE"
"$SERVER_BIN" -m "$MODEL_DIR" -v "$VOICE_DIR" -p "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

wait_for_health >/dev/null || {
    printf 'server did not become healthy\n' >&2
    cat "$LOG_FILE" >&2
    exit 1
}

LIVE_OUT="/tmp/qwen3-tts-live-abort.raw"
rm -f "$LIVE_OUT"
curl -fsSN --get \
    --data-urlencode "text=$TEXT" \
    --data-urlencode "lang=zh" \
    --data-urlencode "voice=lina" \
    "http://127.0.0.1:${PORT}/api/tts/live" >"$LIVE_OUT" &
LIVE_PID=$!

started_stream=0
deadline=$((SECONDS + 30))
while (( SECONDS < deadline )); do
    if [[ -f "$LIVE_OUT" ]] && [[ $(stat -c%s "$LIVE_OUT") -gt 0 ]]; then
        started_stream=1
        break
    fi
    sleep 1
done

if [[ "$started_stream" -ne 1 ]]; then
    printf 'live stream never produced bytes before abort\n' >&2
    cat "$LOG_FILE" >&2
    kill "$LIVE_PID" 2>/dev/null || true
    wait "$LIVE_PID" 2>/dev/null || true
    exit 1
fi

kill "$LIVE_PID" 2>/dev/null || true
wait "$LIVE_PID" 2>/dev/null || true

wait_for_idle >/dev/null || {
    printf 'server remained busy after aborted live request\n' >&2
    curl -sf "http://127.0.0.1:${PORT}/health" >&2 || true
    printf '\n--- server log ---\n' >&2
    cat "$LOG_FILE" >&2
    exit 1
}

STREAM_OUT="/tmp/qwen3-tts-live-abort-recovery.wav"
rm -f "$STREAM_OUT"
curl -sf --get \
    --data-urlencode "text=你好" \
    --data-urlencode "lang=zh" \
    --data-urlencode "voice=lina" \
    "http://127.0.0.1:${PORT}/api/tts/stream" >"$STREAM_OUT"
test -s "$STREAM_OUT"

printf 'live abort recovered cleanly\n'
