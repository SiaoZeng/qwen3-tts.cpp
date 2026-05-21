#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
QWEN3_TTS_DATA_ROOT="${QWEN3_TTS_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/qwen3-tts}"

SERVER_BIN="${1:-${REPO_ROOT}/build/qwen3-tts-server}"
MODEL_DIR="${QWEN3_TTS_MODEL_DIR:-${QWEN3_TTS_DATA_ROOT}/models}"
VOICE_DIR="${QWEN3_TTS_VOICE_DIR:-${QWEN3_TTS_DATA_ROOT}/voices}"
PORT="${QWEN3_TTS_TEST_PORT:-19000}"
LOG_FILE="${QWEN3_TTS_TEST_LOG:-/tmp/qwen3-tts-live-cadence.log}"
TEXT="${QWEN3_TTS_TEST_TEXT:-經濟部商業署今(25)日辦理「產業競爭力輔導團－服務業 AI 升級」推廣說明會。}"
MAX_RATIO="${QWEN3_TTS_MAX_CADENCE_RATIO:-1.20}"

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

rm -f "$LOG_FILE" /tmp/qwen3-tts-live-cadence.raw
QWEN3_TTS_STREAM_TRACE_CHUNKS=1 "$SERVER_BIN" -m "$MODEL_DIR" -v "$VOICE_DIR" -p "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

wait_for_health || {
    printf 'server did not become healthy\n' >&2
    cat "$LOG_FILE" >&2
    exit 1
}

curl -fsSN --get \
    --data-urlencode "text=$TEXT" \
    --data-urlencode "lang=zh" \
    --data-urlencode "voice=lina" \
    "http://127.0.0.1:${PORT}/api/tts/live" -o /tmp/qwen3-tts-live-cadence.raw

python3 - <<'PY' "$LOG_FILE" "$MAX_RATIO"
import re
import sys

log_path = sys.argv[1]
max_ratio = float(sys.argv[2])
pattern = re.compile(r"\[TTS/live-chunk\].*chunk_ms=([0-9.]+) since_prev_ms=([0-9.]+).*")

ratios = []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        chunk_ms = float(m.group(1))
        since_prev_ms = float(m.group(2))
        if chunk_ms > 0:
            ratios.append(since_prev_ms / chunk_ms)

if not ratios:
    print('FAIL: no live chunk trace lines found', file=sys.stderr)
    sys.exit(1)

worst = max(ratios)
print(f'worst_chunk_cadence_ratio={worst:.3f}')
if worst > max_ratio:
    sys.exit(1)
PY

printf 'live chunk cadence stayed within realtime budget\n'
