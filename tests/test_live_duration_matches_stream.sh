#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
QWEN3_TTS_DATA_ROOT="${QWEN3_TTS_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/qwen3-tts}"

SERVER_BIN="${1:-${REPO_ROOT}/build/qwen3-tts-server}"
MODEL_DIR="${QWEN3_TTS_MODEL_DIR:-${QWEN3_TTS_DATA_ROOT}/models}"
VOICE_DIR="${QWEN3_TTS_VOICE_DIR:-${QWEN3_TTS_DATA_ROOT}/voices}"
PORT="${QWEN3_TTS_TEST_PORT:-19000}"
LOG_FILE="${QWEN3_TTS_TEST_LOG:-/tmp/qwen3-tts-live-duration.log}"
TEXT="${QWEN3_TTS_TEST_TEXT:-經濟部商業署今(25)日辦理「產業競爭力輔導團－服務業 AI 升級」推廣說明會。}"
CHUNK_FRAMES="${QWEN3_TTS_STREAM_CHUNK_FRAMES:-4}"
OVERLAP_SAMPLES="${QWEN3_TTS_STREAM_OVERLAP_SAMPLES:-512}"
MAX_DIFF_SECONDS="${QWEN3_TTS_MAX_DURATION_DIFF:-0.05}"

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

duration_of() {
    ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"
}

rm -f "$LOG_FILE" /tmp/qwen3-tts-live-duration-stream.wav /tmp/qwen3-tts-live-duration.raw /tmp/qwen3-tts-live-duration.wav
QWEN3_TTS_STREAM_CHUNK_FRAMES="$CHUNK_FRAMES" \
QWEN3_TTS_STREAM_OVERLAP_SAMPLES="$OVERLAP_SAMPLES" \
QWEN3_TTS_TEMPERATURE=0 \
QWEN3_TTS_TOP_K=0 \
"$SERVER_BIN" -m "$MODEL_DIR" -v "$VOICE_DIR" -p "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

wait_for_health || {
    printf 'server did not become healthy\n' >&2
    cat "$LOG_FILE" >&2
    exit 1
}

curl -fsS --get \
    --data-urlencode "text=$TEXT" \
    --data-urlencode "lang=zh" \
    --data-urlencode "voice=lina" \
    "http://127.0.0.1:${PORT}/api/tts/stream" \
    -o /tmp/qwen3-tts-live-duration-stream.wav

curl -fsSN --get \
    --data-urlencode "text=$TEXT" \
    --data-urlencode "lang=zh" \
    --data-urlencode "voice=lina" \
    "http://127.0.0.1:${PORT}/api/tts/live" \
    -o /tmp/qwen3-tts-live-duration.raw

ffmpeg -f s16le -ar 24000 -ac 1 -i /tmp/qwen3-tts-live-duration.raw -y /tmp/qwen3-tts-live-duration.wav >/dev/null 2>&1

stream_duration="$(duration_of /tmp/qwen3-tts-live-duration-stream.wav)"
live_duration="$(duration_of /tmp/qwen3-tts-live-duration.wav)"

python3 - <<'PY' "$stream_duration" "$live_duration" "$MAX_DIFF_SECONDS" "$CHUNK_FRAMES" "$OVERLAP_SAMPLES"
import sys
stream = float(sys.argv[1])
live = float(sys.argv[2])
limit = float(sys.argv[3])
chunk_frames = sys.argv[4]
overlap = sys.argv[5]
diff = abs(stream - live)
print(f"stream={stream:.6f}s live={live:.6f}s diff={diff:.6f}s chunk_frames={chunk_frames} overlap={overlap}")
if diff > limit:
    sys.exit(1)
PY

printf 'live duration matches stream closely\n'
