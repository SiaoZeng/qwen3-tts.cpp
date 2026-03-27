# Streaming TTS Spec — qwen3-tts.cpp on Strix Halo

## Ziel

First-Audio-Latenz von ~2900ms auf ~300ms reduzieren durch frame-by-frame Streaming.

## Ist-Zustand

```
Text → Tokenize (0ms) → Generate ALL Codes (2300ms) → Decode ALL (660ms) → WAV → Client
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         2960ms bevor der Client etwas hört
```

Alles blockiert bis zum Schluss. Client wartet 2.9s auf das komplette WAV.

## Soll-Zustand

```
Text → Tokenize → Generate Frame 1-4  → Decode Chunk 1 → EMIT → Client hört
                  Generate Frame 5-8  → Decode Chunk 2 → EMIT → Client hört weiter
                  Generate Frame 9-12 → Decode Chunk 3 → EMIT → ...
                  ...
                  EOS → Flush → Final Chunk → Done
```

First-Audio nach ~300ms (4 Frames × ~30ms/Frame Generate + ~200ms Decode Chunk).

## Architektur

### Drei Threads

```
┌─────────────────────────────────────────────────────────┐
│ Thread 1: Transformer (Vulkan GPU)                       │
│   generate() Schleife, Frame-by-Frame                    │
│   → schreibt Frames in Ring-Buffer                       │
│   → signalisiert Decoder-Thread alle N Frames            │
└──────────┬──────────────────────────────────────────────┘
           │ Ring-Buffer (lock-free, N=4 Frames Chunk)
┌──────────▼──────────────────────────────────────────────┐
│ Thread 2: Decoder (Vulkan GPU oder CPU)                  │
│   decode(chunk_codes, chunk_size) pro Chunk              │
│   → Crossfade mit vorherigem Chunk-Tail                  │
│   → schreibt PCM in Output-Queue                         │
└──────────┬──────────────────────────────────────────────┘
           │ PCM-Queue (SPSC)
┌──────────▼──────────────────────────────────────────────┐
│ Thread 3: HTTP Writer                                    │
│   Liest PCM-Chunks → schreibt Chunked HTTP Response      │
│   Erster Chunk: WAV-Header (unbekannte Länge = 0)        │
│   Folgende: Raw PCM Chunks                               │
└─────────────────────────────────────────────────────────┘
```

### Warum 3 Threads?

- Transformer und Decoder können **nicht gleichzeitig auf Vulkan** laufen (eine Queue)
- Aber: Decoder auf **CPU** während Transformer auf **GPU** → echte Parallelität
- HTTP Writer ist I/O-bound, blockiert nicht die Compute-Threads

### Alternative: 2 Threads (Transformer+Decoder sequentiell)

```
Thread 1: Generate 4 Frames → Decode 4 Frames → Emit → Generate 4 → Decode 4 → Emit
Thread 2: HTTP Writer
```

Einfacher, aber keine Parallelität. First-Audio = 4×30ms + 200ms = ~320ms.
Chunk-Latenz = 120ms Generate + 200ms Decode = 320ms pro Chunk.

**Empfehlung: 2-Thread-Ansatz zuerst**, da einfacher und First-Audio trotzdem ~320ms.

## Komponenten-Änderungen

### 1. `tts_transformer.cpp` — Frame-Callback

```cpp
// Neue Signatur mit Callback
using frame_callback_t = std::function<bool(const int32_t * frame_codes, int32_t n_codebooks, int32_t frame_idx)>;
// Return false = abort generation

bool generate_streaming(
    const int32_t * text_tokens, int32_t n_tokens,
    const float * speaker_embd, int32_t max_len,
    frame_callback_t on_frame,
    int32_t language_id, float repetition_penalty,
    float temperature, int32_t top_k);
```

**Änderung in der Schleife** (Zeile 2747-2749):

```cpp
// Vorher:
for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
    output.push_back(frame_codes[cb]);
}

// Nachher:
if (on_frame) {
    if (!on_frame(frame_codes, cfg.n_codebooks, frame)) {
        break;  // Callback said stop
    }
}
```

### 2. `audio_tokenizer_decoder.cpp` — Graph-Cache

```cpp
class AudioTokenizerDecoder {
    // Bestehend
    bool decode(const int32_t * codes, int32_t n_frames, std::vector<float> & samples);

    // Neu: Streaming mit gecachtem Graph
    bool init_streaming(int32_t chunk_size);  // Graph einmal für chunk_size bauen
    bool decode_chunk(const int32_t * codes, std::vector<float> & samples);  // Cached Graph nutzen
    void finish_streaming();

private:
    struct ggml_cgraph * cached_graph_ = nullptr;
    int32_t cached_chunk_size_ = 0;
    bool streaming_initialized_ = false;
};
```

`init_streaming(4)`:
```cpp
bool AudioTokenizerDecoder::init_streaming(int32_t chunk_size) {
    cached_chunk_size_ = chunk_size;
    cached_graph_ = build_graph(chunk_size);
    // Reserve scheduler für diese Graph-Größe
    ggml_backend_sched_reserve(state_.sched, cached_graph_);
    streaming_initialized_ = true;
    return true;
}
```

`decode_chunk(codes)`:
```cpp
bool AudioTokenizerDecoder::decode_chunk(const int32_t * codes, std::vector<float> & samples) {
    // Graph NICHT neu bauen — cached_graph_ wiederverwenden
    ggml_backend_sched_alloc_graph(state_.sched, cached_graph_);

    // Nur Inputs setzen
    for (int cb = 0; cb < 16; ++cb) {
        ggml_backend_tensor_set(cached_tensors_[cb], ...);
    }

    // Compute
    ggml_backend_sched_graph_compute(state_.sched, cached_graph_);

    // Output lesen
    ...
    ggml_backend_sched_reset(state_.sched);
    return true;
}
```

### 3. `crossfade.h` — Hann Window Blending

```cpp
// ~20 Zeilen
void crossfade_hann(float * prev_tail, const float * new_head, float * output,
                    int32_t overlap_samples) {
    for (int i = 0; i < overlap_samples; ++i) {
        float w = 0.5f * (1.0f - cosf(M_PI * (float)i / (float)overlap_samples));
        output[i] = prev_tail[i] * (1.0f - w) + new_head[i] * w;
    }
}
```

Parameter: `overlap_samples = 512` (21ms @ 24kHz).

### 4. `qwen3_tts.cpp` — Streaming Pipeline

```cpp
// Neue Methode
bool Qwen3TTS::synthesize_streaming(
    const std::string & text,
    const float * speaker_embd,
    const tts_params & params,
    std::function<void(const float * pcm, int32_t n_samples, bool is_final)> on_chunk)
{
    // 1. Tokenize
    auto tokens = tokenizer_.encode_for_tts(text);

    // 2. Init streaming decoder
    audio_decoder_.init_streaming(CHUNK_SIZE);

    // 3. Accumulator
    std::vector<int32_t> frame_buffer;
    std::vector<float> prev_tail(OVERLAP_SAMPLES, 0.0f);
    bool first_chunk = true;

    // 4. Generate with callback
    transformer_.generate_streaming(tokens.data(), tokens.size(),
        speaker_embd, params.max_audio_tokens,
        [&](const int32_t * frame_codes, int32_t n_cb, int32_t frame_idx) -> bool {

            // Accumulate
            for (int i = 0; i < n_cb; ++i)
                frame_buffer.push_back(frame_codes[i]);

            // Emit every CHUNK_SIZE frames
            if (frame_buffer.size() >= CHUNK_SIZE * n_cb) {
                std::vector<float> pcm;
                audio_decoder_.decode_chunk(frame_buffer.data(), pcm);

                // Crossfade
                if (!first_chunk && OVERLAP_SAMPLES > 0) {
                    crossfade_hann(prev_tail.data(), pcm.data(), pcm.data(), OVERLAP_SAMPLES);
                }

                // Save tail for next crossfade
                std::copy(pcm.end() - OVERLAP_SAMPLES, pcm.end(), prev_tail.begin());

                // Emit (trimmed)
                on_chunk(pcm.data(), pcm.size() - OVERLAP_SAMPLES, false);

                frame_buffer.clear();
                first_chunk = false;
            }
            return true;  // continue
        },
        params.language_id, params.repetition_penalty,
        params.temperature, params.top_k);

    // 5. Flush remaining frames
    if (!frame_buffer.empty()) {
        std::vector<float> pcm;
        audio_decoder_.decode(frame_buffer.data(), frame_buffer.size() / 16, pcm);
        // Crossfade + fade-out
        on_chunk(pcm.data(), pcm.size(), true);
    }

    audio_decoder_.finish_streaming();
    return true;
}
```

### 5. `server.cpp` — HTTP Chunked Streaming

```cpp
// GET /api/tts/live?text=...&lang=de&voice=lina
// Returns: chunked HTTP with WAV header + PCM chunks

// 1. Send WAV header with size=0 (unknown length)
write_wav_header(client_fd, 24000, 0);

// 2. Stream PCM chunks as they arrive
tts.synthesize_streaming(text, embedding, params,
    [&](const float * pcm, int32_t n, bool is_final) {
        // Convert float → int16
        std::vector<int16_t> samples(n);
        for (int i = 0; i < n; ++i)
            samples[i] = (int16_t)(pcm[i] * 32767.0f);
        write(client_fd, samples.data(), n * 2);
    });

// 3. Client: curl ... | pw-play --raw --rate 24000 --channels 1 --format s16
```

## Timing-Schätzung

### Generate-Timing pro Frame

Aktuell: ~2300ms für ~30 Frames = **~77ms/Frame**

Aufschlüsselung:
- Talker forward_step: ~50ms/Frame (28 Transformer-Layer auf Vulkan)
- Code predictor: ~27ms/Frame (5 Layer, 15 autoregressiv Steps)

### Decode-Timing pro Chunk

Aktuell: ~660ms für ~30 Frames = **~22ms/Frame**

Mit Graph-Cache für 4 Frames: ~22ms × 4 + ~50ms Overhead = **~140ms/Chunk**

### First-Audio-Latenz

```
4 Frames × 77ms/Frame = 308ms (Generate)
+ 140ms (Decode Chunk 1)
= ~450ms First-Audio
```

Mit Two-Phase (3 Frames erster Chunk):
```
3 × 77ms = 231ms + 120ms Decode = ~350ms First-Audio
```

### Gesamt-RTF

Sequentiell (2-Thread): RTF ~0.85 (etwas schlechter als batch wegen Chunk-Overhead)
Concurrent (3-Thread, Decoder auf CPU): RTF ~0.75 (ähnlich wie jetzt)

## Risiken

| Risiko | Mitigation |
|--------|-----------|
| Graph-Cache funktioniert nicht mit `sched` | Fallback: Graph jedes Mal bauen (aktueller Code) |
| Vulkan-Contention (Transformer + Decoder) | Decoder auf CPU, Transformer auf Vulkan |
| Crossfade-Artefakte | Overlap auf 1024 erhöhen, A/B testen |
| Chunk-Decoder-Qualität schlechter als Batch | Sliding Window: letzten 2 Chunks mit-decodieren, nur neuen Teil emittieren |
| HTTP Chunked + pw-play Kompatibilität | Raw PCM statt WAV, pw-play --raw |

## Implementierungs-Reihenfolge

```
Phase 1: Frame-Callback im Transformer (1h)
- [ ] generate_streaming() mit on_frame Callback
- [ ] Test: Callback wird pro Frame aufgerufen

Phase 2: Graph-Cache im Decoder (1h)
- [ ] init_streaming(chunk_size)
- [ ] decode_chunk() mit gecachtem Graph
- [ ] Benchmark: Chunk-Decode-Latenz vs Full-Decode

Phase 3: Crossfade (30min)
- [ ] crossfade_hann() implementieren
- [ ] A/B Test: mit/ohne Crossfade hörbar?

Phase 4: Streaming Pipeline (2h)
- [ ] synthesize_streaming() in Qwen3TTS
- [ ] Frame-Buffer + Chunk-Emission
- [ ] End-of-Stream Flush

Phase 5: HTTP Streaming (1h)
- [ ] /api/tts/live Endpoint
- [ ] Raw PCM Chunked Transfer
- [ ] tts-say Update für Live-Streaming

Phase 6: Optimierung (1h)
- [ ] Two-Phase (kleinerer erster Chunk)
- [ ] Concurrent Decode Thread
- [ ] Latenz-Messung End-to-End
```

## Abhängigkeiten

- Keine externen Libraries nötig
- Keine API-Änderungen an GGML
- Backward-kompatibel: bestehende batch-API bleibt unverändert
