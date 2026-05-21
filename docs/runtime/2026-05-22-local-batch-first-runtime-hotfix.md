# Local Batch-First TTS Runtime Hotfix

- Date: 2026-05-22
- Scope: local runtime only
- Status: applied and verified

## Purpose

Stabilize local `tts-say` usage by removing the default dependency on live-session creation and by making the deployed `tts-server` restart-stable against the current main build.

## Applied Runtime Changes

### 1. `tts-say` default path changed to batch-first

Target:
- `${HOME}/.local/bin/tts-say`

Behavior after the hotfix:
- default mode uses `GET /api/tts/stream`
- readiness is checked via `/health` before synthesis
- without `--output`, a temporary WAV is created, played inline, and removed afterward
- `ok` is printed only after playback succeeds
- default mode ignores `TTS_SESSION_URL`
- optional overrides supported:
  - `TTS_HEALTH_URL`
  - `TTS_READY_RETRY_COUNT`
  - `TTS_READY_RETRY_INTERVAL_SEC`
  - `TTS_BATCH_PLAYER_CMD`

Versioned patch artifact:
- `docs/runtime/2026-05-22-tts-say-batch-first.patch`

Backup kept at:
- `${HOME}/.local/bin/tts-say.pre-batch-first.bak`

### 2. Deployed `tts-server` binary switched to the current main build

Target:
- `${XDG_DATA_HOME:-$HOME/.local/share}/qwen3-tts/bin/qwen3-tts-server`

Applied change:
- runtime binary path now resolves to the repository main build:
  - `${HOME}/gh/qwen3-tts.cpp/build/qwen3-tts-server`

Reason:
- the previous deployed binary depended on worktree-local GGML shared-library paths and failed on restart with missing `libggml.so.0`

Backup kept at:
- `${XDG_DATA_HOME:-$HOME/.local/share}/qwen3-tts/bin/qwen3-tts-server.pre-main-symlink.bak`

## Verification Evidence

### Runtime checks

- `systemctl --user restart tts-server`
- `systemctl --user is-active tts-server` → `active`
- `curl http://127.0.0.1:8007/health` → `{"status":"ok","busy":false,"queue":0}`

### `tts-say` checks

- default mode with invalid `TTS_SESSION_URL` still succeeds
- German batch playback check succeeded
- Chinese batch playback check succeeded
- `--output` writes a valid WAV file and prints the output path
- service remained restartable after the change

## Rollback

### Restore `tts-say`

```bash
cp -a "$HOME/.local/bin/tts-say.pre-batch-first.bak" "$HOME/.local/bin/tts-say"
chmod 755 "$HOME/.local/bin/tts-say"
```

### Restore previous deployed server binary

```bash
cp -a \
  "${XDG_DATA_HOME:-$HOME/.local/share}/qwen3-tts/bin/qwen3-tts-server.pre-main-symlink.bak" \
  "${XDG_DATA_HOME:-$HOME/.local/share}/qwen3-tts/bin/qwen3-tts-server"
```

### Restart service

```bash
systemctl --user restart tts-server
```
