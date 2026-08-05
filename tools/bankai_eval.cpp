// bankai_eval: stdin-driven probe evaluation tool for Bankai.
//
// Loads a GGUF model once, then reads commands from stdin:
//   PROBE <correct_id> <wrong_id> <n_tokens> <tok0> <tok1> ...
//     → runs one forward pass on the given prompt tokens, prints
//       "logit[correct] - logit[wrong]" to stdout
//   TOKENIZE <text...>
//     → tokenizes text using the model's tokenizer, prints
//       "<n_tokens> <tok0> <tok1> ..." to stdout
//   FLIP_ROW <tensor_name> <row_index>
//     → XOR all sign bits in the given row of a Q1_0_g128 tensor,
//       in-place on the GPU via ggml_backend_tensor_get/set.
//       XOR is self-inverse, so calling again reverts the flip.
//   NUM_ROWS <tensor_name>
//     → prints the number of rows (first dim) of a tensor
//   NUM_COLS <tensor_name>
//     → prints the number of cols (second dim) of a tensor
//   SCALES <tensor_name>
//     → prints the average absolute FP16 scale per row, space-separated
//   QUIT
//     → exits
//
// Designed to be driven by Python subprocess.Popen with bidirectional pipes.
// Weight manipulation stays in GPU memory — no file reloading, no restart.

#include "llama.h"
#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <unordered_map>

// Q1_0_g128 block layout (must match the fork):
//   bytes 0-1:   FP16 scale
//   bytes 2-17:  16 bytes = 128 sign bits
static constexpr int Q1_0_G128_BLOCK_SIZE = 18;
static constexpr int Q1_0_G128_GROUP_SIZE = 128;
static constexpr int Q1_0_G128_SCALE_BYTES = 2;
static constexpr int Q1_0_G128_BITS_OFFSET = 2;
static constexpr int Q1_0_G128_BITS_BYTES = 16;

static llama_model   * g_model = nullptr;
static llama_context * g_ctx   = nullptr;
static std::string     g_model_path;

// Tensor lookup cache — rebuilt on every model load
static std::unordered_map<std::string, struct ggml_tensor *> g_tensor_map;
static bool g_tensor_map_built = false;

static void free_model() {
    if (g_ctx)   { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
}

static bool load_model(const std::string & path) {
    free_model();
    g_tensor_map.clear();
    g_tensor_map_built = false;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;  // offload all layers to GPU
    mparams.use_mmap = false;    // critical: we modify the GGUF file between reloads

    g_model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_model) {
        fprintf(stderr, "[bankai_eval] failed to load model: %s\n", path.c_str());
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 2048;
    cparams.n_batch  = 2048;
    cparams.n_ubatch = 512;
    cparams.embeddings = false;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        fprintf(stderr, "[bankai_eval] failed to create context\n");
        return false;
    }

    return true;
}

// Internal llama.cpp function that returns a vector of (name, tensor*)
// pairs for all tensors in a model. It's not part of the public C API but
// is exported from libllama.so for llama-bench and friends. We cache the
// map on first use.
#include <utility>
const std::vector<std::pair<std::string, struct ggml_tensor *>> &
    llama_internal_get_tensor_map(const struct llama_model * model);

static void build_tensor_map() {
    const auto & tensors = llama_internal_get_tensor_map(g_model);
    for (const auto & pair : tensors) {
        g_tensor_map[pair.first] = pair.second;
    }
    g_tensor_map_built = true;
    fprintf(stderr, "[bankai_eval] indexed %zu tensors\n", g_tensor_map.size());
}

static ggml_tensor * find_tensor(const std::string & name) {
    if (!g_tensor_map_built) build_tensor_map();
    auto it = g_tensor_map.find(name);
    return (it != g_tensor_map.end()) ? it->second : nullptr;
}

// XOR the sign bits of a single row of a Q1_0_g128 tensor in-place on its backend.
// Rows are the first dim; each row contains (cols/128) blocks of 18 bytes.
static bool flip_row_q1_0_g128(const std::string & tensor_name, int64_t row) {
    ggml_tensor * t = find_tensor(tensor_name);
    if (!t) {
        fprintf(stderr, "[bankai_eval] tensor not found: %s\n", tensor_name.c_str());
        return false;
    }
    if (t->type != GGML_TYPE_Q1_0_g128) {
        fprintf(stderr, "[bankai_eval] tensor %s is not Q1_0_g128 (got %d)\n",
                tensor_name.c_str(), (int)t->type);
        return false;
    }

    // ggml tensor shape: t->ne[0] = cols (fastest dim), t->ne[1] = rows
    int64_t cols = t->ne[0];
    int64_t rows = t->ne[1];
    if (row < 0 || row >= rows) {
        fprintf(stderr, "[bankai_eval] row %lld out of range [0, %lld)\n",
                (long long)row, (long long)rows);
        return false;
    }

    int64_t groups_per_row = cols / Q1_0_G128_GROUP_SIZE;
    size_t  bytes_per_row  = (size_t)groups_per_row * Q1_0_G128_BLOCK_SIZE;
    size_t  row_offset     = (size_t)row * bytes_per_row;

    // Read current row bytes from backend storage
    std::vector<uint8_t> buf(bytes_per_row);
    ggml_backend_tensor_get(t, buf.data(), row_offset, bytes_per_row);

    // XOR the sign bits (bytes 2..18 of each 18-byte block)
    for (int64_t g = 0; g < groups_per_row; g++) {
        uint8_t * bits = buf.data() + g * Q1_0_G128_BLOCK_SIZE + Q1_0_G128_BITS_OFFSET;
        for (int i = 0; i < Q1_0_G128_BITS_BYTES; i++) {
            bits[i] ^= 0xFF;
        }
    }

    // Write back
    ggml_backend_tensor_set(t, buf.data(), row_offset, bytes_per_row);
    return true;
}

// XOR the sign bits of a single 128-bit group (within one row) in-place.
// This is the finer-grained version of flip_row — each call touches only
// 16 bytes (the sign bits of one 18-byte block), which is 32x more precise
// than row-level flips for Q1_0_g128 tensors with cols=4096.
static bool flip_group_q1_0_g128(const std::string & tensor_name,
                                 int64_t row, int64_t group) {
    ggml_tensor * t = find_tensor(tensor_name);
    if (!t || t->type != GGML_TYPE_Q1_0_g128) {
        fprintf(stderr, "[bankai_eval] bad tensor for flip_group: %s\n",
                tensor_name.c_str());
        return false;
    }

    int64_t cols = t->ne[0];
    int64_t rows = t->ne[1];
    int64_t groups_per_row = cols / Q1_0_G128_GROUP_SIZE;

    if (row < 0 || row >= rows || group < 0 || group >= groups_per_row) {
        fprintf(stderr, "[bankai_eval] flip_group out of range: row=%lld group=%lld\n",
                (long long)row, (long long)group);
        return false;
    }

    size_t block_offset = ((size_t)row * groups_per_row + (size_t)group)
                          * Q1_0_G128_BLOCK_SIZE
                          + Q1_0_G128_BITS_OFFSET;

    // Read the 16 sign-bit bytes for this single group
    uint8_t bits[Q1_0_G128_BITS_BYTES];
    ggml_backend_tensor_get(t, bits, block_offset, Q1_0_G128_BITS_BYTES);
    for (int i = 0; i < Q1_0_G128_BITS_BYTES; i++) {
        bits[i] ^= 0xFF;
    }
    ggml_backend_tensor_set(t, bits, block_offset, Q1_0_G128_BITS_BYTES);
    return true;
}

// Read average |fp16 scale| per row for a Q1_0_g128 tensor.
static bool row_scales_q1_0_g128(const std::string & tensor_name,
                                 std::vector<float> & out_scales) {
    ggml_tensor * t = find_tensor(tensor_name);
    if (!t || t->type != GGML_TYPE_Q1_0_g128) return false;

    int64_t cols = t->ne[0];
    int64_t rows = t->ne[1];
    int64_t groups_per_row = cols / Q1_0_G128_GROUP_SIZE;
    size_t  bytes_per_row  = (size_t)groups_per_row * Q1_0_G128_BLOCK_SIZE;
    size_t  total_bytes    = rows * bytes_per_row;

    std::vector<uint8_t> buf(total_bytes);
    ggml_backend_tensor_get(t, buf.data(), 0, total_bytes);

    out_scales.assign(rows, 0.0f);
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t * row_buf = buf.data() + r * bytes_per_row;
        double sum = 0.0;
        for (int64_t g = 0; g < groups_per_row; g++) {
            const uint8_t * block = row_buf + g * Q1_0_G128_BLOCK_SIZE;
            uint16_t bits;
            memcpy(&bits, block, 2);  // fp16 scale in first 2 bytes
            // Convert fp16 → fp32 using ggml helper
            float s = ggml_fp16_to_fp32((ggml_fp16_t)bits);
            sum += std::abs(s);
        }
        out_scales[r] = (float)(sum / (double)groups_per_row);
    }
    return true;
}

static float probe(int correct_id, int wrong_id, const std::vector<llama_token> & tokens) {
    // Clear KV cache so each probe is independent
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Build a batch containing all prompt tokens
    const int n = (int) tokens.size();
    llama_batch batch = llama_batch_init(n, 0, 1);

    for (int i = 0; i < n; i++) {
        batch.token[i]       = tokens[i];
        batch.pos[i]         = i;
        batch.n_seq_id[i]    = 1;
        batch.seq_id[i][0]   = 0;
        batch.logits[i]      = (i == n - 1) ? 1 : 0; // only care about last position
    }
    batch.n_tokens = n;

    if (llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        fprintf(stderr, "[bankai_eval] llama_decode failed\n");
        return 0.0f;
    }

    const float * logits = llama_get_logits_ith(g_ctx, n - 1);
    const float gap = logits[correct_id] - logits[wrong_id];

    llama_batch_free(batch);
    return gap;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    g_model_path = argv[1];

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    if (!load_model(g_model_path)) {
        return 1;
    }

    fprintf(stderr, "[bankai_eval] ready (model loaded)\n");
    fflush(stderr);
    // Signal readiness on stdout
    printf("READY\n");
    fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "PROBE") {
            int correct_id, wrong_id, n_tokens;
            iss >> correct_id >> wrong_id >> n_tokens;
            std::vector<llama_token> tokens(n_tokens);
            for (int i = 0; i < n_tokens; i++) {
                int t; iss >> t;
                tokens[i] = (llama_token) t;
            }
            float gap = probe(correct_id, wrong_id, tokens);
            printf("%.6f\n", gap);
            fflush(stdout);

        } else if (cmd == "TOKENIZE") {
            // Format: "TOKENIZE <text...>" — tokenize the rest of the line
            // (text can include spaces; we read everything after "TOKENIZE ")
            std::string text;
            // Advance past any leading whitespace in the stream
            iss >> std::ws;
            std::getline(iss, text);

            const llama_vocab * vocab = llama_model_get_vocab(g_model);
            // First pass to get token count
            int n = llama_tokenize(
                vocab, text.c_str(), (int)text.length(),
                nullptr, 0, /*add_special*/ false, /*parse_special*/ false
            );
            if (n < 0) n = -n;

            std::vector<llama_token> tokens(n);
            int n_tokens = llama_tokenize(
                vocab, text.c_str(), (int)text.length(),
                tokens.data(), n, false, false
            );
            if (n_tokens < 0) n_tokens = -n_tokens;

            printf("%d", n_tokens);
            for (int i = 0; i < n_tokens; i++) {
                printf(" %d", (int)tokens[i]);
            }
            printf("\n");
            fflush(stdout);

        } else if (cmd == "FLIP_ROW") {
            std::string tensor_name;
            int64_t row;
            iss >> tensor_name >> row;
            bool ok = flip_row_q1_0_g128(tensor_name, row);
            printf("%s\n", ok ? "OK" : "ERROR");
            fflush(stdout);

        } else if (cmd == "FLIP_GROUP") {
            std::string tensor_name;
            int64_t row, group;
            iss >> tensor_name >> row >> group;
            bool ok = flip_group_q1_0_g128(tensor_name, row, group);
            printf("%s\n", ok ? "OK" : "ERROR");
            fflush(stdout);

        } else if (cmd == "NUM_ROWS") {
            std::string tensor_name;
            iss >> tensor_name;
            ggml_tensor * t = find_tensor(tensor_name);
            if (!t) { printf("ERROR\n"); fflush(stdout); continue; }
            printf("%lld\n", (long long)t->ne[1]);
            fflush(stdout);

        } else if (cmd == "NUM_COLS") {
            std::string tensor_name;
            iss >> tensor_name;
            ggml_tensor * t = find_tensor(tensor_name);
            if (!t) { printf("ERROR\n"); fflush(stdout); continue; }
            printf("%lld\n", (long long)t->ne[0]);
            fflush(stdout);

        } else if (cmd == "SCALES") {
            std::string tensor_name;
            iss >> tensor_name;
            std::vector<float> scales;
            if (!row_scales_q1_0_g128(tensor_name, scales)) {
                printf("ERROR\n"); fflush(stdout); continue;
            }
            // Print count followed by values
            printf("%zu", scales.size());
            for (float s : scales) printf(" %.6g", s);
            printf("\n");
            fflush(stdout);

        } else if (cmd == "RELOAD") {
            if (!load_model(g_model_path)) {
                printf("RELOAD_FAILED\n");
                fflush(stdout);
                return 1;
            }
            printf("RELOADED\n");
            fflush(stdout);

        } else if (cmd == "QUIT") {
            break;

        } else {
            fprintf(stderr, "[bankai_eval] unknown command: %s\n", cmd.c_str());
            printf("ERROR\n");
            fflush(stdout);
        }
    }

    free_model();
    llama_backend_free();
    return 0;
}
