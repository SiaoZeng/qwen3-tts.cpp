#include "qwen3_tts.h"
#include "audio_tokenizer_decoder.h"
#include "crossfade.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

static bool load_embedding_file(const std::string & path, std::vector<float> & embedding) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "FAIL: Could not open embedding file %s\n", path.c_str());
        return false;
    }

    int32_t size = 0;
    if (fread(&size, sizeof(size), 1, f) != 1 || size <= 0) {
        fclose(f);
        fprintf(stderr, "FAIL: Invalid embedding header in %s\n", path.c_str());
        return false;
    }

    embedding.resize(size);
    if (fread(embedding.data(), sizeof(float), embedding.size(), f) != embedding.size()) {
        fclose(f);
        fprintf(stderr, "FAIL: Could not read embedding data from %s\n", path.c_str());
        return false;
    }

    fclose(f);
    return true;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --models <dir> --embedding <file> [--text <text>] [--chunk-frames N] [--overlap-samples N]\n", prog);
}

static std::string pick_tokenizer_model(const std::string & model_dir) {
    const std::string q8 = model_dir + "/qwen3-tts-tokenizer-q8_0.gguf";
    FILE * q8f = fopen(q8.c_str(), "rb");
    if (q8f) {
        fclose(q8f);
        return q8;
    }
    return model_dir + "/qwen3-tts-tokenizer-f16.gguf";
}

int main(int argc, char ** argv) {
    auto getenv_int_or = [](const char * name, int fallback) {
        const char * val = getenv(name);
        if (!val || !*val) return fallback;
        char * end = nullptr;
        long parsed = strtol(val, &end, 10);
        if (end == val || *end != '\0' || parsed <= 0) return fallback;
        return (int)parsed;
    };
    std::string model_dir;
    std::string embedding_path;
    std::string text = "經濟部商業署今(25)日辦理「產業競爭力輔導團－服務業 AI 升級」推廣說明會。";
    int32_t chunk_frames = 8;
    int32_t overlap_samples = 512;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--embedding") == 0 && i + 1 < argc) {
            embedding_path = argv[++i];
        } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text = argv[++i];
        } else if (strcmp(argv[i], "--chunk-frames") == 0 && i + 1 < argc) {
            chunk_frames = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--overlap-samples") == 0 && i + 1 < argc) {
            overlap_samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_dir.empty() || embedding_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    qwen3_tts::Qwen3TTS tts;
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "FAIL: %s\n", tts.get_error().c_str());
        return 1;
    }

    std::vector<float> embedding;
    if (!load_embedding_file(embedding_path, embedding)) {
        return 1;
    }

    qwen3_tts::tts_params params;
    params.language_id = 2055;
    params.temperature = 0.0f;
    params.top_k = 0;
    params.repetition_penalty = 1.0f;
    params.print_progress = false;
    params.print_timing = false;

    {
        qwen3_tts::AudioTokenizerDecoder decoder;
        const std::string tokenizer_model = pick_tokenizer_model(model_dir);
        if (!decoder.load_model(tokenizer_model)) {
            fprintf(stderr, "FAIL: decoder load failed: %s\n", decoder.get_error().c_str());
            return 1;
        }

        const int32_t n_codebooks = decoder.get_config().n_codebooks;
        const int32_t context_frames = getenv_int_or("QWEN3_TTS_DECODER_TEST_CONTEXT_FRAMES", 1);
        const int32_t total_frames = chunk_frames * 2 + 3;
        std::vector<int32_t> codes(total_frames * n_codebooks, 0);
        for (int32_t frame = 0; frame < total_frames; ++frame) {
            for (int32_t cb = 0; cb < n_codebooks; ++cb) {
                codes[frame * n_codebooks + cb] = (frame + cb) % decoder.get_config().codebook_size;
            }
        }

        std::vector<float> batch_decoded;
        if (!decoder.decode(codes.data(), total_frames, batch_decoded)) {
            fprintf(stderr, "FAIL: decoder batch decode failed: %s\n", decoder.get_error().c_str());
            return 1;
        }

        {
            std::vector<float> direct_chunk_decoded;
            if (!decoder.decode(codes.data(), chunk_frames, direct_chunk_decoded)) {
                fprintf(stderr, "FAIL: decoder direct fixed chunk decode failed: %s\n", decoder.get_error().c_str());
                return 1;
            }
            if (!decoder.init_streaming(chunk_frames)) {
                fprintf(stderr, "FAIL: decoder init_streaming fixed chunk failed: %s\n", decoder.get_error().c_str());
                return 1;
            }
            std::vector<float> cached_chunk_decoded;
            if (!decoder.decode_chunk(codes.data(), chunk_frames, cached_chunk_decoded)) {
                fprintf(stderr, "FAIL: decoder cached fixed chunk decode failed: %s\n", decoder.get_error().c_str());
                return 1;
            }
            decoder.finish_streaming();
            if (direct_chunk_decoded.size() != cached_chunk_decoded.size()) {
                fprintf(stderr, "FAIL: cached chunk size mismatch direct=%zu cached=%zu\n",
                        direct_chunk_decoded.size(), cached_chunk_decoded.size());
                return 1;
            }
            double cached_sum_abs_diff = 0.0;
            double cached_max_abs_diff = 0.0;
            for (size_t i = 0; i < direct_chunk_decoded.size(); ++i) {
                const double d = fabs((double)direct_chunk_decoded[i] - (double)cached_chunk_decoded[i]);
                cached_sum_abs_diff += d;
                if (d > cached_max_abs_diff) {
                    cached_max_abs_diff = d;
                }
            }
            const double cached_mean_abs_diff = direct_chunk_decoded.empty() ? 0.0 : cached_sum_abs_diff / (double)direct_chunk_decoded.size();
            if (cached_mean_abs_diff > 0.0001) {
                fprintf(stderr, "FAIL: cached decode_chunk drift too large mean_abs_diff=%.8f max_abs_diff=%.8f\n",
                        cached_mean_abs_diff, cached_max_abs_diff);
                return 1;
            }
        }

        std::vector<float> context_probe;
        if (!decoder.decode(codes.data(), context_frames, context_probe)) {
            fprintf(stderr, "FAIL: decoder context probe failed: %s\n", decoder.get_error().c_str());
            return 1;
        }
        const int32_t context_samples = (int32_t)context_probe.size();
        std::vector<float> chunked_decoded;
        std::vector<int32_t> decode_window;
        std::vector<int32_t> previous_frame(n_codebooks, 0);
        std::vector<float> pending_chunk;
        bool have_previous_frame = false;

        for (int32_t frame = 0; frame < total_frames; frame += chunk_frames) {
            const int32_t fresh_frames = std::min(chunk_frames, total_frames - frame);
            std::vector<float> decoded_window;
            if (!have_previous_frame) {
                if (!decoder.decode(codes.data() + frame * n_codebooks, fresh_frames, decoded_window)) {
                    fprintf(stderr, "FAIL: decoder first chunk decode failed: %s\n", decoder.get_error().c_str());
                    return 1;
                }
                if (!decoder.init_streaming(chunk_frames + context_frames)) {
                    fprintf(stderr, "FAIL: decoder init_streaming failed: %s\n", decoder.get_error().c_str());
                    return 1;
                }
            } else {
                decode_window.clear();
                decode_window.insert(decode_window.end(), previous_frame.begin(), previous_frame.end());
                decode_window.insert(decode_window.end(),
                                     codes.begin() + frame * n_codebooks,
                                     codes.begin() + (frame + fresh_frames) * n_codebooks);

                if (!decoder.decode_chunk(decode_window.data(), fresh_frames + context_frames, decoded_window)) {
                    fprintf(stderr, "FAIL: decoder chunk decode failed: %s\n", decoder.get_error().c_str());
                    return 1;
                }
            }

            const size_t emit_offset = have_previous_frame ? (size_t)context_samples : 0;
            if (emit_offset > decoded_window.size()) {
                fprintf(stderr, "FAIL: decoder emit offset out of range offset=%zu size=%zu\n",
                        emit_offset, decoded_window.size());
                return 1;
            }
            std::vector<float> current_chunk(decoded_window.begin() + emit_offset, decoded_window.end());
            const size_t blend = std::min<size_t>(256, std::min(pending_chunk.size(), current_chunk.size()));
            if (!pending_chunk.empty()) {
                if (blend > 0) {
                    crossfade_hann(pending_chunk.data() + (pending_chunk.size() - blend), current_chunk.data(), (int32_t)blend);
                    chunked_decoded.insert(chunked_decoded.end(), pending_chunk.begin(), pending_chunk.end());
                    pending_chunk = std::move(current_chunk);
                } else {
                    chunked_decoded.insert(chunked_decoded.end(), pending_chunk.begin(), pending_chunk.end());
                    pending_chunk = std::move(current_chunk);
                }
            } else {
                pending_chunk = std::move(current_chunk);
            }

            previous_frame.assign(codes.begin() + (frame + fresh_frames - 1) * n_codebooks,
                                  codes.begin() + (frame + fresh_frames) * n_codebooks);
            have_previous_frame = true;
        }
        decoder.finish_streaming();

        chunked_decoded.insert(chunked_decoded.end(), pending_chunk.begin(), pending_chunk.end());

        if (chunked_decoded.size() != batch_decoded.size()) {
            fprintf(stderr, "FAIL: decoder chunked size mismatch batch=%zu chunked=%zu\n",
                    batch_decoded.size(), chunked_decoded.size());
            return 1;
        }

        double decoder_sum_abs_diff = 0.0;
        double decoder_max_abs_diff = 0.0;
        for (size_t i = 0; i < batch_decoded.size(); ++i) {
            const double d = fabs((double)batch_decoded[i] - (double)chunked_decoded[i]);
            decoder_sum_abs_diff += d;
            if (d > decoder_max_abs_diff) {
                decoder_max_abs_diff = d;
            }
        }
        const double decoder_mean_abs_diff = batch_decoded.empty() ? 0.0 : decoder_sum_abs_diff / (double)batch_decoded.size();
        if (decoder_mean_abs_diff > 0.03) {
            fprintf(stderr, "FAIL: decoder chunked waveform drift too large mean_abs_diff=%.8f max_abs_diff=%.8f\n",
                    decoder_mean_abs_diff, decoder_max_abs_diff);
            return 1;
        }
    }

    qwen3_tts::tts_result batch = tts.synthesize_with_embedding(text, embedding.data(), (int32_t)embedding.size(), params);
    if (!batch.success) {
        fprintf(stderr, "FAIL: batch synthesize failed: %s\n", batch.error_msg.c_str());
        return 1;
    }

    std::vector<float> streamed_audio;
    bool streaming_ok = tts.synthesize_streaming(
        text,
        embedding.data(),
        (int32_t)embedding.size(),
        params,
        [&](const float * pcm, int32_t n_samples, bool) -> bool {
            streamed_audio.insert(streamed_audio.end(), pcm, pcm + n_samples);
            return true;
        },
        chunk_frames,
        overlap_samples
    );

    if (!streaming_ok) {
        fprintf(stderr, "FAIL: streaming synthesize failed: %s\n", tts.get_error().c_str());
        return 1;
    }

    double batch_seconds = (double)batch.audio.size() / batch.sample_rate;
    double streamed_seconds = (double)streamed_audio.size() / batch.sample_rate;
    double diff_seconds = fabs(batch_seconds - streamed_seconds);
    double sum_abs_diff = 0.0;
    double max_abs_diff = 0.0;
    size_t n_compare = std::min(batch.audio.size(), streamed_audio.size());
    size_t batch_non_finite = 0;
    size_t streamed_non_finite = 0;

    for (float sample : batch.audio) {
        if (!std::isfinite(sample)) {
            batch_non_finite++;
        }
    }
    for (float sample : streamed_audio) {
        if (!std::isfinite(sample)) {
            streamed_non_finite++;
        }
    }

    if (batch_non_finite > 0 || streamed_non_finite > 0) {
        fprintf(stderr, "FAIL: non-finite audio samples batch=%zu streamed=%zu\n",
                batch_non_finite, streamed_non_finite);
        return 1;
    }

    for (size_t i = 0; i < n_compare; ++i) {
        double d = fabs((double)batch.audio[i] - (double)streamed_audio[i]);
        sum_abs_diff += d;
        if (d > max_abs_diff) {
            max_abs_diff = d;
        }
    }

    double mean_abs_diff = n_compare > 0 ? sum_abs_diff / (double)n_compare : 0.0;

    printf("batch=%zu streamed=%zu diff=%.6fs mean_abs_diff=%.8f max_abs_diff=%.8f chunk_frames=%d overlap=%d\n",
           batch.audio.size(), streamed_audio.size(), diff_seconds, mean_abs_diff, max_abs_diff, chunk_frames, overlap_samples);

    if (diff_seconds > 0.05) {
        fprintf(stderr, "FAIL: duration mismatch too large\n");
        return 1;
    }

    if (mean_abs_diff > 0.03) {
        fprintf(stderr, "FAIL: waveform drift too large mean_abs_diff=%.8f\n", mean_abs_diff);
        return 1;
    }

    printf("PASS: streaming duration matches batch closely\n");
    return 0;
}
