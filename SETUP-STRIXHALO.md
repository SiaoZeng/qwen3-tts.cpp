# Qwen3-TTS on AMD Strix Halo (gfx1151) — ONNX-free Setup

ONNX-free TTS stack running entirely on GGML with Vulkan GPU acceleration on AMD Radeon 8060S (RDNA 3.5, gfx1151).

## Performance

| Metric | Value |
|--------|-------|
| RTF | **0.72** (1.39x realtime) |
| Transformer | Vulkan GPU (Q8_0) |
| Decoder | Vulkan GPU (Q8_0) |
| ONNX Runtime | **Eliminated** |
| Model size | 1.6 GB (Q8_0) |
| Temperature | 0.3 |

## Hardware

- **CPU**: AMD Ryzen AI Max+ 395 (Zen 5, 16C/32T)
- **GPU**: Radeon 8060S (RDNA 3.5, gfx1151, 40 CUs)
- **RAM**: 128 GB DDR5 (shared memory, 120 GB GTT)
- **OS**: CachyOS (Arch-based), Kernel 6.19.9
- **Vulkan**: RADV (Mesa)

## Voices

| Name | Gender | Languages | Style |
|------|--------|-----------|-------|
| **Lina** (莉娜) | Female | TW-Mandarin, Deutsch, English | Warm Taiwanese accent |
| **Pem** (Pemberton) | Male | TW-Mandarin, Deutsch, British English | TW accent + RP British |

Voices are stored in `voices/` as WAV reference files for zero-shot voice cloning.

## Local Patches

### 1. Vulkan Decoder Fix (`src/audio_tokenizer_decoder.cpp`)

**Problem**: `normalize_codebooks()` wrote directly into Vulkan-mapped `tensor->data` causing segfault in `ggml_vk_buffer_write_2d`.

**Fix**: Download tensors to host → normalize on CPU → upload back to GPU:
```cpp
// Before: codebook->data pointed to Vulkan device memory
// ggml_fp16_t * cb_data = (ggml_fp16_t *)codebook->data; // CRASH

// After: download → normalize → upload
ggml_backend_tensor_get(codebook, cb_host.data(), 0, cb_bytes);
// ... normalize on host ...
ggml_backend_tensor_set(codebook, cb_host.data(), 0, cb_bytes);
```

### 2. Q8_0 Tokenizer Auto-Selection (`src/qwen3_tts.cpp`)

Added automatic preference for Q8_0 tokenizer GGUF alongside the existing Q8_0 TTS model preference.

## Build

```bash
# Prerequisites
sudo pacman -S vulkan-headers vulkan-icd-loader cmake

# Build GGML with Vulkan
cd ~/gh/qwen3-tts.cpp
git submodule update --init --recursive
cmake -S ggml -B ggml/build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(nproc)

# Build qwen3-tts.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target qwen3-tts-cli
```

## Model Setup

```bash
# Create venv for conversion scripts
python3 -m venv .venv && source .venv/bin/activate
pip install huggingface_hub gguf torch safetensors numpy tqdm

# Download + convert to F16 GGUF
python3 scripts/setup_pipeline_models.py --coreml off

# Quantize to Q8_0
python3 scripts/convert_tts_to_gguf.py \
    --input models/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3-tts-0.6b-q8_0.gguf --type q8_0

python3 scripts/convert_tokenizer_to_gguf.py \
    --input models/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3-tts-tokenizer-q8_0.gguf --type q8_0
```

## Usage

### CLI
```bash
# Default voice (no clone)
./build/qwen3-tts-cli -m models -t "Hello world" -l en -o out.wav -j 8

# With voice clone (Lina)
./build/qwen3-tts-cli -m models -t "Hallo Welt" -l de -r voices/lina.wav -o out.wav -j 8

# With Pem voice
./build/qwen3-tts-cli -m models -t "你好世界" -l zh -r voices/pem.wav -o out.wav -j 8
```

### Server (Port 8007)
```bash
# Start
systemctl --user start tts-server

# API
curl -X POST http://127.0.0.1:8007/api/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "Hallo", "language": "de", "voice": "lina"}'

# Health
curl http://127.0.0.1:8007/health
```

### tts-say CLI
```bash
tts-say "Hallo Welt"                    # Lina, Deutsch (default)
tts-say "Hello world" --lang en         # Lina, English
tts-say "你好世界" --voice pem --lang zh  # Pem, Mandarin
tts-say "Text" --voice default          # No clone
```

## Service Files

- `~/.config/systemd/user/tts-server.service` — Permanent server on port 8007
- `~/.local/bin/tts-server-ggml.sh` — HTTP wrapper around CLI
- `~/.local/bin/tts-say` — CLI with auto-play
- `~/.claude/skills/tts-speak/SKILL.md` — Claude Code skill

## Voice Design

Custom voices are created using the `qwen-tts` Python library (VoiceDesign model), then used as clone references:

```bash
# One-time: design a voice with natural language description
python3 -c "
from qwen_tts import Qwen3TTSModel
import torch, soundfile as sf
torch.backends.cudnn.enabled = False
model = Qwen3TTSModel.from_pretrained(
    'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
    device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
torch.manual_seed(777)
wavs, sr = model.generate_voice_design(
    text='你好，歡迎光臨。',
    language='Chinese',
    instruct='A warm Taiwanese woman with gentle 台灣腔 accent.')
sf.write('voices/my_voice.wav', wavs[0], sr)
"

# Then use as clone reference forever
./build/qwen3-tts-cli -m models -t "Text" -r voices/my_voice.wav -o out.wav
```

## Architecture

```
Text ──► [Tokenizer] ──► Token IDs
                              │
Speaker Clone ──► [Speaker Encoder] ──► Speaker Embedding (optional)
                              │
Token IDs + Embedding ──► [TTS Transformer 0.6B] ──► Speech Codes
                              │                        (Vulkan Q8_0)
                              │
Speech Codes ──► [WavTokenizer Decoder] ──► Audio 24kHz
                    (Vulkan Q8_0)
```

All components run on GGML — no ONNX Runtime, no PyTorch at inference time.

## Journey

| Stack | ONNX | RTF | Notes |
|-------|------|-----|-------|
| Coqui XTTS-v2 (PyTorch) | ✓ | 7.0 | MIOpen no solver DB for gfx1151 |
| XTTS-v2 optimized (cudnn=off+FP16) | ✓ | 0.6 | Best PyTorch result |
| cgisky/Qwen3-TTS-Rust Q8 | ✓ | 0.96 | ONNX CPU decoder |
| qwen3-tts.cpp CPU decoder | ✗ | 1.31 | Vulkan segfault in decoder |
| **qwen3-tts.cpp Full Vulkan Q8** | **✗** | **0.72** | **Production** |

## Key Findings: gfx1151 (RDNA 3.5)

- **MIOpen**: No solver DB → `torch.backends.cudnn.enabled = False` (PyTorch only)
- **FP8/FP4**: Hardware limit, no WMMA FP8 instructions (RDNA 4+ only)
- **BitsAndBytes INT8**: Slower than BF16 (dequant overhead > bandwidth savings)
- **GGML Vulkan**: Works natively via RADV, no HSA_OVERRIDE needed
- **Vulkan buffer_write**: Crashes when writing to device-mapped memory directly — must download→modify→upload
- **PyTorch ROCm**: pip wheels crash (ROCm 7.0 vs 7.2 ABI mismatch), use system package
