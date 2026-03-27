// Persistent TTS server — model stays loaded, accepts requests via HTTP
// Eliminates 620ms cold start per request

#include "qwen3_tts.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <signal.h>

static volatile bool g_running = true;

void signal_handler(int) { g_running = false; }

// Safe write that handles SIGPIPE / broken pipe gracefully
static bool safe_write(int fd, const void * data, size_t len) {
    ssize_t n = write(fd, data, len);
    return n >= 0 && (size_t)n == len;
}

// Simple JSON value extractor (no dependency needed)
static std::string json_get_string(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

// Base64 encode
static const char b64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::string base64_encode(const uint8_t * data, size_t len) {
    std::string result;
    result.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = ((uint32_t)data[i]) << 16;
        if (i + 1 < len) n |= ((uint32_t)data[i + 1]) << 8;
        if (i + 2 < len) n |= (uint32_t)data[i + 2];
        result += b64_table[(n >> 18) & 0x3F];
        result += b64_table[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? b64_table[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? b64_table[n & 0x3F] : '=';
    }
    return result;
}

static void send_response(int fd, int code, const std::string & content_type, const std::string & body) {
    std::string status = (code == 200) ? "OK" : (code == 400) ? "Bad Request" : "Internal Server Error";
    std::ostringstream resp;
    resp << "HTTP/1.1 " << code << " " << status << "\r\n"
         << "Content-Type: " << content_type << "\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Access-Control-Allow-Origin: *\r\n"
         << "Connection: close\r\n\r\n"
         << body;
    std::string r = resp.str();
    write(fd, r.c_str(), r.size());
}

static void handle_client(int client_fd, qwen3_tts::Qwen3TTS & tts,
                           const std::string & voices_dir,
                           const qwen3_tts::tts_params & default_params) {
    char buf[65536];
    int n = read(client_fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(client_fd); return; }
    buf[n] = 0;
    std::string request(buf);

    // GET /health
    if (request.find("GET /health") == 0) {
        send_response(client_fd, 200, "text/plain", "OK");
        close(client_fd);
        return;
    }

    // GET /api/tts/live?text=...&lang=de&voice=lina — streaming raw PCM s16le 24kHz mono
    if (request.find("GET /api/tts/live") != std::string::npos) {
        size_t q = request.find('?');
        size_t end = request.find(" HTTP");
        std::string query = (q != std::string::npos && end != std::string::npos) ?
            request.substr(q + 1, end - q - 1) : "";

        auto get_param = [&query](const std::string & key) -> std::string {
            size_t pos = query.find(key + "=");
            if (pos == std::string::npos) return "";
            pos += key.size() + 1;
            size_t amp = query.find('&', pos);
            std::string val = (amp != std::string::npos) ? query.substr(pos, amp - pos) : query.substr(pos);
            for (size_t i = 0; i < val.size(); i++) {
                if (val[i] == '+') val[i] = ' ';
                if (val[i] == '%' && i + 2 < val.size()) {
                    int c; sscanf(val.c_str() + i + 1, "%2x", &c);
                    val = val.substr(0, i) + (char)c + val.substr(i + 3);
                }
            }
            return val;
        };

        std::string text = get_param("text");
        std::string lang = get_param("lang");
        std::string voice = get_param("voice");
        if (voice.empty()) voice = "lina";
        if (lang.empty()) lang = "de";

        if (text.empty()) {
            send_response(client_fd, 400, "text/plain", "Missing text param");
            close(client_fd);
            return;
        }

        qwen3_tts::tts_params params = default_params;
        if      (lang == "de") params.language_id = 2053;
        else if (lang == "en") params.language_id = 2050;
        else if (lang == "zh") params.language_id = 2055;
        else if (lang == "ja") params.language_id = 2058;
        else if (lang == "ko") params.language_id = 2064;

        // Load embedding
        std::string emb_path = voices_dir + "/" + voice + ".bin";
        std::vector<float> emb;
        int32_t emb_size = 0;
        FILE * ef = fopen(emb_path.c_str(), "rb");
        if (ef) {
            fread(&emb_size, 4, 1, ef);
            emb.resize(emb_size);
            fread(emb.data(), 4, emb_size, ef);
            fclose(ef);
        }

        // Send HTTP header for raw PCM streaming
        std::string hdr = "HTTP/1.1 200 OK\r\n"
            "Content-Type: audio/pcm;rate=24000;channels=1;bits=16\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Connection: close\r\n\r\n";
        write(client_fd, hdr.c_str(), hdr.size());

        int32_t total_samples = 0;

        tts.synthesize_streaming(text,
            emb.empty() ? nullptr : emb.data(), emb_size, params,
            [&](const float * pcm, int32_t n_samples, bool is_final) {
                // Convert float → int16 PCM
                std::vector<int16_t> buf(n_samples);
                for (int32_t i = 0; i < n_samples; ++i) {
                    float s = pcm[i];
                    if (s > 1.0f) s = 1.0f;
                    if (s < -1.0f) s = -1.0f;
                    buf[i] = (int16_t)(s * 32767.0f);
                }

                // HTTP chunked encoding — handle broken pipe gracefully
                char chunk_hdr[32];
                int chunk_len = n_samples * 2;
                int hdr_len = snprintf(chunk_hdr, sizeof(chunk_hdr), "%x\r\n", chunk_len);
                if (!safe_write(client_fd, chunk_hdr, hdr_len) ||
                    !safe_write(client_fd, buf.data(), chunk_len) ||
                    !safe_write(client_fd, "\r\n", 2)) {
                    return;  // client disconnected, stop streaming
                }

                total_samples += n_samples;

                if (is_final) {
                    safe_write(client_fd, "0\r\n\r\n", 5);
                }
            },
            4,    // chunk_frames
            512   // overlap_samples
        );

        float dur = (float)total_samples / 24000.0f;
        fprintf(stderr, "[TTS/live] \"%s\" %.1fs streamed\n", text.c_str(), dur);
        close(client_fd);
        return;
    }

    // GET /api/tts/stream?text=...&lang=de&voice=lina — returns raw WAV directly
    if (request.find("GET /api/tts/stream") != std::string::npos) {
        // Parse query params from URL
        size_t q = request.find('?');
        size_t end = request.find(" HTTP");
        std::string query = (q != std::string::npos && end != std::string::npos) ?
            request.substr(q + 1, end - q - 1) : "";

        auto get_param = [&query](const std::string & key) -> std::string {
            size_t pos = query.find(key + "=");
            if (pos == std::string::npos) return "";
            pos += key.size() + 1;
            size_t amp = query.find('&', pos);
            std::string val = (amp != std::string::npos) ? query.substr(pos, amp - pos) : query.substr(pos);
            // URL decode spaces
            for (size_t i = 0; i < val.size(); i++) {
                if (val[i] == '+') val[i] = ' ';
                if (val[i] == '%' && i + 2 < val.size()) {
                    int c; sscanf(val.c_str() + i + 1, "%2x", &c);
                    val = val.substr(0, i) + (char)c + val.substr(i + 3);
                }
            }
            return val;
        };

        std::string text = get_param("text");
        std::string lang = get_param("lang");
        std::string voice = get_param("voice");
        if (voice.empty()) voice = "lina";
        if (lang.empty()) lang = "de";

        if (text.empty()) {
            send_response(client_fd, 400, "text/plain", "Missing text param");
            close(client_fd);
            return;
        }

        qwen3_tts::tts_params params = default_params;
        if      (lang == "de") params.language_id = 2053;
        else if (lang == "en") params.language_id = 2050;
        else if (lang == "zh") params.language_id = 2055;
        else if (lang == "ja") params.language_id = 2058;
        else if (lang == "ko") params.language_id = 2064;

        std::string emb_path = voices_dir + "/" + voice + ".bin";
        qwen3_tts::tts_result result;
        FILE * ef = fopen(emb_path.c_str(), "rb");
        if (ef) {
            int32_t es; fread(&es, 4, 1, ef);
            std::vector<float> emb(es); fread(emb.data(), 4, es, ef); fclose(ef);
            result = tts.synthesize_with_embedding(text, emb.data(), es, params);
        } else {
            result = tts.synthesize(text, params);
        }

        if (!result.success) {
            send_response(client_fd, 500, "text/plain", result.error_msg);
            close(client_fd);
            return;
        }

        // Build WAV in memory
        int32_t sr = result.sample_rate;
        int32_t ns = result.audio.size();
        int32_t ds = ns * 2;
        int32_t fs = 36 + ds;
        std::vector<uint8_t> wav(44 + ds);
        uint8_t * p = wav.data();
        memcpy(p, "RIFF", 4); p+=4; memcpy(p, &fs, 4); p+=4;
        memcpy(p, "WAVE", 4); p+=4; memcpy(p, "fmt ", 4); p+=4;
        int32_t fmts=16; memcpy(p,&fmts,4); p+=4;
        int16_t tag=1; memcpy(p,&tag,2); p+=2;
        int16_t ch=1; memcpy(p,&ch,2); p+=2;
        memcpy(p,&sr,4); p+=4;
        int32_t br=sr*2; memcpy(p,&br,4); p+=4;
        int16_t ba=2; memcpy(p,&ba,2); p+=2;
        int16_t bits=16; memcpy(p,&bits,2); p+=2;
        memcpy(p, "data", 4); p+=4; memcpy(p,&ds,4); p+=4;
        for (int32_t i = 0; i < ns; i++) {
            float s = std::max(-1.0f, std::min(1.0f, result.audio[i]));
            int16_t sample = (int16_t)(s * 32767.0f);
            memcpy(p, &sample, 2); p += 2;
        }

        // Send WAV directly — client can pipe to pw-play
        std::ostringstream hdr;
        hdr << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: audio/wav\r\n"
            << "Content-Length: " << wav.size() << "\r\n"
            << "Connection: close\r\n\r\n";
        std::string h = hdr.str();
        write(client_fd, h.c_str(), h.size());
        write(client_fd, wav.data(), wav.size());
        close(client_fd);

        float dur = (float)ns / sr;
        fprintf(stderr, "[TTS/stream] \"%s\" %.1fs in %lldms\n",
                text.c_str(), dur, (long long)result.t_total_ms);
        return;
    }

    // POST /api/tts
    if (request.find("POST /api/tts") == std::string::npos) {
        send_response(client_fd, 404, "text/plain", "Not Found");
        close(client_fd);
        return;
    }

    // Extract JSON body
    size_t body_start = request.find("\r\n\r\n");
    if (body_start == std::string::npos) {
        send_response(client_fd, 400, "application/json", "{\"success\":false,\"message\":\"No body\"}");
        close(client_fd);
        return;
    }
    std::string json_body = request.substr(body_start + 4);

    std::string text = json_get_string(json_body, "text");
    std::string lang = json_get_string(json_body, "language");
    std::string voice = json_get_string(json_body, "voice");

    if (text.empty()) {
        send_response(client_fd, 400, "application/json", "{\"success\":false,\"message\":\"Missing text\"}");
        close(client_fd);
        return;
    }

    if (voice.empty()) voice = "lina";
    if (lang.empty()) lang = "de";

    // Language mapping
    qwen3_tts::tts_params params = default_params;
    if      (lang == "de" || lang == "german")   params.language_id = 2053;
    else if (lang == "en" || lang == "english")  params.language_id = 2050;
    else if (lang == "zh" || lang == "chinese" || lang == "mandarin") params.language_id = 2055;
    else if (lang == "ja" || lang == "japanese") params.language_id = 2058;
    else if (lang == "ko" || lang == "korean")   params.language_id = 2064;
    else if (lang == "fr" || lang == "french")   params.language_id = 2061;
    else if (lang == "es" || lang == "spanish")  params.language_id = 2054;
    else if (lang == "ru" || lang == "russian")  params.language_id = 2069;

    // Load embedding
    std::string emb_path = voices_dir + "/" + voice + ".bin";
    qwen3_tts::tts_result result;

    FILE * emb_file = fopen(emb_path.c_str(), "rb");
    if (emb_file) {
        int32_t emb_size;
        fread(&emb_size, sizeof(int32_t), 1, emb_file);
        std::vector<float> emb(emb_size);
        fread(emb.data(), sizeof(float), emb_size, emb_file);
        fclose(emb_file);
        result = tts.synthesize_with_embedding(text, emb.data(), emb_size, params);
    } else {
        result = tts.synthesize(text, params);
    }

    if (!result.success) {
        std::string err = "{\"success\":false,\"message\":\"" + result.error_msg + "\"}";
        send_response(client_fd, 500, "application/json", err);
        close(client_fd);
        return;
    }

    // Encode WAV to base64
    std::vector<uint8_t> wav_data;
    {
        // WAV header
        int32_t sr = result.sample_rate;
        int32_t n_samples = result.audio.size();
        int32_t data_size = n_samples * 2; // 16-bit
        int32_t file_size = 36 + data_size;
        wav_data.resize(44 + data_size);
        uint8_t * p = wav_data.data();
        memcpy(p, "RIFF", 4); p += 4;
        memcpy(p, &file_size, 4); p += 4;
        memcpy(p, "WAVE", 4); p += 4;
        memcpy(p, "fmt ", 4); p += 4;
        int32_t fmt_size = 16; memcpy(p, &fmt_size, 4); p += 4;
        int16_t fmt_tag = 1; memcpy(p, &fmt_tag, 2); p += 2;
        int16_t channels = 1; memcpy(p, &channels, 2); p += 2;
        memcpy(p, &sr, 4); p += 4;
        int32_t byte_rate = sr * 2; memcpy(p, &byte_rate, 4); p += 4;
        int16_t block_align = 2; memcpy(p, &block_align, 2); p += 2;
        int16_t bits = 16; memcpy(p, &bits, 2); p += 2;
        memcpy(p, "data", 4); p += 4;
        memcpy(p, &data_size, 4); p += 4;
        for (int32_t i = 0; i < n_samples; i++) {
            float s = result.audio[i];
            if (s > 1.0f) s = 1.0f;
            if (s < -1.0f) s = -1.0f;
            int16_t sample = (int16_t)(s * 32767.0f);
            memcpy(p, &sample, 2); p += 2;
        }
    }

    std::string b64 = base64_encode(wav_data.data(), wav_data.size());
    std::string json_resp = "{\"success\":true,\"audio_base64\":\"" + b64 + "\",\"message\":null}";

    send_response(client_fd, 200, "application/json", json_resp);
    close(client_fd);

    float duration = (float)result.audio.size() / result.sample_rate;
    fprintf(stderr, "[TTS] \"%s\" lang=%s voice=%s | %.1fs audio in %lldms (RTF %.2f)\n",
            text.c_str(), lang.c_str(), voice.c_str(),
            duration, (long long)result.t_total_ms,
            (float)result.t_total_ms / (duration * 1000.0f));
}

int main(int argc, char ** argv) {
    std::string model_dir = "models";
    std::string voices_dir = "voices";
    int port = 8007;
    int threads = 8;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) model_dir = argv[++i];
        else if (arg == "-v" && i + 1 < argc) voices_dir = argv[++i];
        else if (arg == "-p" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (arg == "-j" && i + 1 < argc) threads = std::stoi(argv[++i]);
    }

    fprintf(stderr, "=== Qwen3-TTS Persistent Server (ONNX-free, GGML Vulkan) ===\n");
    fprintf(stderr, "Models: %s | Voices: %s | Port: %d | Threads: %d\n",
            model_dir.c_str(), voices_dir.c_str(), port, threads);

    qwen3_tts::Qwen3TTS tts;
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    fprintf(stderr, "Models loaded. Server ready.\n");

    qwen3_tts::tts_params default_params;
    default_params.temperature = 0.3f;
    default_params.n_threads = threads;
    default_params.language_id = 2053; // de

    // TCP server
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Error: bind failed on port %d\n", port);
        return 1;
    }
    listen(server_fd, 8);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);  // ignore broken pipe — safe_write handles it

    fprintf(stderr, "Listening on http://127.0.0.1:%d\n", port);
    fprintf(stderr, "  POST /api/tts  {\"text\":\"...\",\"language\":\"de\",\"voice\":\"lina\"}\n");
    fprintf(stderr, "  GET  /health\n");

    while (g_running) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) continue;
        handle_client(client_fd, tts, voices_dir, default_params);
    }

    close(server_fd);
    fprintf(stderr, "Server stopped.\n");
    return 0;
}
