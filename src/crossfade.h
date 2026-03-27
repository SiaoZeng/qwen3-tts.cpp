#pragma once
#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Hann-window crossfade between previous chunk tail and new chunk head
// Blends in-place: output[i] = prev[i] * (1 - w[i]) + curr[i] * w[i]
inline void crossfade_hann(const float * prev_tail, float * curr_head, int32_t overlap) {
    for (int32_t i = 0; i < overlap; ++i) {
        float w = 0.5f * (1.0f - cosf((float)M_PI * (float)i / (float)overlap));
        curr_head[i] = prev_tail[i] * (1.0f - w) + curr_head[i] * w;
    }
}

// Fade-in for first chunk (prevents pop)
inline void fade_in_hann(float * samples, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        float w = 0.5f * (1.0f - cosf((float)M_PI * (float)i / (float)n));
        samples[i] *= w;
    }
}

// Fade-out for last chunk (prevents pop)
inline void fade_out_hann(float * samples, int32_t n_samples, int32_t fade_len) {
    for (int32_t i = 0; i < fade_len && i < n_samples; ++i) {
        float w = 0.5f * (1.0f + cosf((float)M_PI * (float)i / (float)fade_len));
        samples[n_samples - fade_len + i] *= w;
    }
}
