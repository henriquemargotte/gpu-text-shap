#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << what << "): " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

struct Layer {
    int in_dim = 0;
    int out_dim = 0;
    std::vector<float> W; // row-major [out_dim, in_dim]
    std::vector<float> b; // [out_dim]
};

struct Network {
    std::vector<Layer> layers;
};

struct Dataset {
    int n = 0;
    int seq_len = 0;
    std::vector<int32_t> labels;  // [n]
    std::vector<int32_t> token_ids; // [n, seq_len]
};

// Embedding matrix container: row-major [vocab_size, embed_dim]
struct Embedding {
    int vocab_size = 0;
    int embed_dim = 0;
    std::vector<float> data; // size = vocab_size * embed_dim
};

static Network load_weights(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open weights file: " + path);
    }

    Network net;
    int L = 0;
    in >> L;
    if (L <= 0) {
        throw std::runtime_error("Invalid layer count in weights file");
    }

    net.layers.resize(L);

    for (int i = 0; i < L; i++) {
        int in_dim = 0, out_dim = 0;
        in >> in_dim >> out_dim;
        if (in_dim <= 0 || out_dim <= 0) {
            throw std::runtime_error("Invalid layer dims in weights file");
        }
        Layer layer;
        layer.in_dim = in_dim;
        layer.out_dim = out_dim;
        layer.W.resize((size_t)in_dim * (size_t)out_dim);
        layer.b.resize((size_t)out_dim);

        for (size_t k = 0; k < layer.W.size(); k++) {
            in >> layer.W[k];
        }
        for (size_t k = 0; k < layer.b.size(); k++) {
            in >> layer.b[k];
        }
        net.layers[i] = std::move(layer);
    }

    return net;
}

static Embedding load_embedding(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open embedding file: " + path);
    }
    Embedding emb;
    in >> emb.vocab_size >> emb.embed_dim;
    if (emb.vocab_size <= 0 || emb.embed_dim <= 0) {
        throw std::runtime_error("Invalid embedding header");
    }
    emb.data.resize((size_t)emb.vocab_size * (size_t)emb.embed_dim);
    for (size_t i = 0; i < emb.data.size(); ++i) in >> emb.data[i];
    return emb;
}

static Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open dataset file: " + path);
    }

    Dataset ds;
    in >> ds.n >> ds.seq_len;
    if (ds.n <= 0 || ds.seq_len <= 0) {
        throw std::runtime_error("Invalid dataset header");
    }

    ds.labels.resize((size_t)ds.n);
    ds.token_ids.resize((size_t)ds.n * (size_t)ds.seq_len);

    for (int i = 0; i < ds.n; i++) {
        int lbl = 0;
        in >> lbl;
        ds.labels[(size_t)i] = (int32_t)lbl;
        for (int j = 0; j < ds.seq_len; j++) {
            int tid = 0;
            in >> tid;
            ds.token_ids[(size_t)i * (size_t)ds.seq_len + (size_t)j] = (int32_t)tid;
        }
    }

    return ds;
}

// Device helper: compute MLP layers given input/output shared buffers.
// Returns pointer to the buffer containing the final logits (either in_buf or out_buf).
// This helper runs synchronously across the block and uses __syncthreads(),
// so it must be called by all threads in the block.
__device__ float* mlp_compute_layers_device(
    float* in_buf,
    float* out_buf,
    const float* W_all,
    const float* b_all,
    const int* w_offsets,
    const int* b_offsets,
    const int* in_dims,
    const int* out_dims,
    int num_layers,
    int tx,
    int stride,
    int unused_i_feature
) {
    for (int l = 0; l < num_layers; l++) {
        int in_dim = in_dims[l];
        int out_dim = out_dims[l];
        const float* W = W_all + w_offsets[l];
        const float* b = b_all + b_offsets[l];

        for (int o = tx; o < out_dim; o += stride) {
            float acc = b[o];
            const float* wrow = W + (size_t)o * (size_t)in_dim;
            for (int k = 0; k < in_dim; k++) {
                acc += wrow[k] * in_buf[k];
            }
            // implement ReLU nonlinearity for all but the last layer
            if (l < num_layers - 1 && acc < 0.0f) acc = 0.0f;
            out_buf[o] = acc;
        }
        __syncthreads();

        float* tmp = in_buf;
        in_buf = out_buf;
        out_buf = tmp;
    }
    return in_buf;
}

// Kernel: perform per-permutation SHAP-style runs but compute mean-pooled embeddings
// incrementally on-device by looking up embeddings and maintaining a running sum.
__global__ void emb_mlp_forward_kernel(
    const int32_t* token_ids,
    int n_permutations,
    int seq_len,
    const float* embedding_matrix, // [vocab_size * embed_dim]
    int embed_dim,
    const float* W_all,
    const float* b_all,
    const int* w_offsets,
    const int* b_offsets,
    const int* in_dims,
    const int* out_dims,
    int num_layers,
    float* logits_out,
    float* shap_out,
    int out_dim_last,
    int max_dim
) {
    int block_id = blockIdx.x;
    int perm_idx = block_id % n_permutations;
    bool do_reverse = (block_id >= n_permutations);
    int sample = 0; // single sample
    int tx = threadIdx.x;
    int stride = blockDim.x;
    bool do_write = (tx == 0);

    extern __shared__ char s[];
    float* buf0 = reinterpret_cast<float*>(s);
    float* buf1 = reinterpret_cast<float*>(s + (size_t)max_dim * sizeof(float));
    int* perm = reinterpret_cast<int*>(s + (size_t)max_dim * 2 * sizeof(float));
    float* shap_block = reinterpret_cast<float*>(s + (size_t)max_dim * 2 * sizeof(float) + (size_t)seq_len * sizeof(int));
    float* running_sum = reinterpret_cast<float*>(s + (size_t)max_dim * 2 * sizeof(float) + (size_t)seq_len * sizeof(int) + (size_t)seq_len * sizeof(float));

    if (tx == 0) {
        for (int j = 0; j < seq_len; ++j) perm[j] = j;
        unsigned int state = (unsigned int)perm_idx * 1013904223u + 12345u;
        for (int j = seq_len - 1; j > 0; --j) {
            state = state * 1103515245u + 12345u;
            unsigned int r = state % (unsigned int)(j + 1);
            int tmp = perm[j]; perm[j] = perm[r]; perm[r] = tmp;
        }
    }
    __syncthreads();

    // Initialize running_sum to zero
    for (int d = tx; d < embed_dim; d += stride) running_sum[d] = 0.0f;
    __syncthreads();

    for (int i_feature = 0; i_feature < seq_len; i_feature++) {
        int perm_pos = do_reverse ? perm[seq_len - 1 - i_feature] : perm[i_feature];
        int32_t tok = token_ids[(size_t)sample * (size_t)seq_len + (size_t)perm_pos];

        // Add this token's embedding into running_sum cooperatively
        const float* emb_row = embedding_matrix + (size_t)tok * (size_t)embed_dim;
        for (int d = tx; d < embed_dim; d += stride) {
            float v = emb_row[d];
            atomicAdd(&running_sum[d], v);
        }
        __syncthreads();

        // Compute mean vector into buf0 (we will use first embed_dim entries)
        float denom = 1.0f / (float)(i_feature + 1);
        for (int d = tx; d < embed_dim; d += stride) {
            buf0[d] = running_sum[d] * denom;
        }
        __syncthreads();

        // Run MLP with input sized embed_dim
        float* final_buf = mlp_compute_layers_device(buf0, buf1,
                                                     W_all, b_all,
                                                     w_offsets, b_offsets,
                                                     in_dims, out_dims,
                                                     num_layers,
                                                     tx, stride, 0);
        __syncthreads();
        if (tx == 0) {
            shap_block[i_feature] = final_buf[0];
        }
        __syncthreads();
    }

    if (tx == 0){
        for (int i = seq_len - 1; i >= 1; i--) {
            shap_block[i] = shap_block[i] - shap_block[i-1];
        }
    }

    if (do_write) {
        logits_out[(size_t)block_id] = shap_block[seq_len-1];
    }
    __syncthreads();

    // Now atomically add per-block shap contributions to global shap_out (map permuted position -> original index)
    for (int j = tx; j < seq_len; j += stride) {
        int orig_idx = do_reverse ? perm[seq_len - 1 - j] : perm[j];
        atomicAdd(&shap_out[orig_idx], shap_block[j]);
    }
    __syncthreads();
}

int main(int argc, char** argv) {
    std::string weights_path = "out/mlp_weights.txt";
    std::string dataset_path = "out/tokenized_dataset.txt";
    std::string emb_path = "out/embedding_matrix.txt"; // embedding file
    int threads = 128;
    int max_print = 10;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (a == "--dataset" && i + 1 < argc) dataset_path = argv[++i];
        else if (a == "--embeddings" && i + 1 < argc) emb_path = argv[++i];
        else if (a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (a == "--print" && i + 1 < argc) max_print = std::stoi(argv[++i]);
        else {
            std::cerr << "Unknown/invalid arg: " << a << "\n";
            return 2;
        }
    }

    try {
        Network net = load_weights(weights_path);
        Dataset ds = load_dataset(dataset_path);
        Embedding emb = load_embedding(emb_path);

        if (net.layers.empty()) throw std::runtime_error("No layers loaded");
        if (emb.embed_dim <= 0) throw std::runtime_error("Invalid embedding dimension");

        int num_layers = (int)net.layers.size();
        int out_dim_last = net.layers.back().out_dim;
        int n_permutations = 10; // default

        // Re-parse argv to pick up --n-permutations if present
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--n-permutations" && i + 1 < argc) {
                n_permutations = std::stoi(argv[++i]);
            }
        }

        // Pack weights/biases into contiguous arrays, and create per-layer offsets.
        std::vector<int> w_offsets(num_layers);
        std::vector<int> b_offsets(num_layers);
        std::vector<int> in_dims(num_layers);
        std::vector<int> out_dims(num_layers);

        size_t total_w = 0;
        size_t total_b = 0;
        for (int l = 0; l < num_layers; l++) {
            in_dims[l] = net.layers[l].in_dim;
            out_dims[l] = net.layers[l].out_dim;
            w_offsets[l] = (int)total_w;
            b_offsets[l] = (int)total_b;
            total_w += (size_t)in_dims[l] * (size_t)out_dims[l];
            total_b += (size_t)out_dims[l];
        }

        std::vector<float> W_all(total_w, 0.0f);
        std::vector<float> b_all(total_b, 0.0f);

        for (int l = 0; l < num_layers; l++) {
            const auto& layer = net.layers[l];
            std::copy(layer.W.begin(), layer.W.end(), W_all.begin() + (size_t)w_offsets[l]);
            std::copy(layer.b.begin(), layer.b.end(), b_all.begin() + (size_t)b_offsets[l]);
        }

        int32_t* d_token_ids = nullptr;
        float* d_embedding = nullptr;
        float* d_W = nullptr;
        float* d_b = nullptr;
        int* d_w_offsets = nullptr;
        int* d_b_offsets = nullptr;
        int* d_in_dims = nullptr;
        int* d_out_dims = nullptr;
        float* d_logits = nullptr;

        checkCuda(cudaMalloc(&d_token_ids, ds.token_ids.size() * sizeof(int32_t)), "cudaMalloc token_ids");
        checkCuda(cudaMalloc(&d_embedding, emb.data.size() * sizeof(float)), "cudaMalloc embedding");
        checkCuda(cudaMalloc(&d_W, W_all.size() * sizeof(float)), "cudaMalloc W");
        checkCuda(cudaMalloc(&d_b, b_all.size() * sizeof(float)), "cudaMalloc b");
        checkCuda(cudaMalloc(&d_w_offsets, w_offsets.size() * sizeof(int)), "cudaMalloc w_offsets");
        checkCuda(cudaMalloc(&d_b_offsets, b_offsets.size() * sizeof(int)), "cudaMalloc b_offsets");
        checkCuda(cudaMalloc(&d_in_dims, in_dims.size() * sizeof(int)), "cudaMalloc in_dims");
        checkCuda(cudaMalloc(&d_out_dims, out_dims.size() * sizeof(int)), "cudaMalloc out_dims");
        checkCuda(cudaMalloc(&d_logits, (size_t)2 * (size_t)n_permutations * sizeof(float)), "cudaMalloc logits");

        checkCuda(cudaMemcpy(d_token_ids, ds.token_ids.data(), ds.token_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy token_ids");
        checkCuda(cudaMemcpy(d_embedding, emb.data.data(), emb.data.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy embedding");
        checkCuda(cudaMemcpy(d_W, W_all.data(), W_all.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy W");
        checkCuda(cudaMemcpy(d_b, b_all.data(), b_all.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy b");
        checkCuda(cudaMemcpy(d_w_offsets, w_offsets.data(), w_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy w_offsets");
        checkCuda(cudaMemcpy(d_b_offsets, b_offsets.data(), b_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy b_offsets");
        checkCuda(cudaMemcpy(d_in_dims, in_dims.data(), in_dims.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy in_dims");
        checkCuda(cudaMemcpy(d_out_dims, out_dims.data(), out_dims.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy out_dims");

        int max_dim = emb.embed_dim;
        for (const auto& layer : net.layers) {
            if (layer.out_dim > max_dim) max_dim = layer.out_dim;
            if (layer.in_dim > max_dim) max_dim = layer.in_dim;
        }

        size_t shmem = (size_t)max_dim * 2 * sizeof(float) + (size_t)ds.seq_len * sizeof(int) + (size_t)ds.seq_len * sizeof(float) + (size_t)emb.embed_dim * sizeof(float);

        dim3 grid((unsigned int)2 * (unsigned int)n_permutations);
        dim3 block(threads);

        // Allocate and zero device shap buffer (one aggregated vector of length seq_len)
        float* d_shap = nullptr;
        checkCuda(cudaMalloc(&d_shap, (size_t)ds.seq_len * sizeof(float)), "cudaMalloc d_shap");
        checkCuda(cudaMemset(d_shap, 0, (size_t)ds.seq_len * sizeof(float)), "memset d_shap");

        cudaEvent_t ev_start, ev_stop;
        checkCuda(cudaEventCreate(&ev_start), "create event start");
        checkCuda(cudaEventCreate(&ev_stop), "create event stop");
        checkCuda(cudaEventRecord(ev_start), "record event start");

        emb_mlp_forward_kernel<<<grid, block, shmem>>> (
            d_token_ids,
            n_permutations,
            ds.seq_len,
            d_embedding,
            emb.embed_dim,
            d_W,
            d_b,
            d_w_offsets,
            d_b_offsets,
            d_in_dims,
            d_out_dims,
            num_layers,
            d_logits,
            d_shap,
            out_dim_last,
            max_dim
        );
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaEventRecord(ev_stop), "record event stop");
        checkCuda(cudaEventSynchronize(ev_stop), "synchronize event stop");
        float elapsed_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop), "elapsed time");
        std::cout << "cuda_kernel_time_ms=" << elapsed_ms << "\n";
        checkCuda(cudaDeviceSynchronize(), "device sync");
        checkCuda(cudaEventDestroy(ev_start), "destroy event start");
        checkCuda(cudaEventDestroy(ev_stop), "destroy event stop");

        size_t out_count = (size_t)2 * (size_t)n_permutations;
        std::vector<float> h_logits(out_count);
        checkCuda(cudaMemcpy(h_logits.data(), d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "memcpy logits back");

        std::vector<float> h_shap((size_t)ds.seq_len);
        checkCuda(cudaMemcpy(h_shap.data(), d_shap, (size_t)ds.seq_len * sizeof(float), cudaMemcpyDeviceToHost), "memcpy shap back");
        std::vector<float> shap_values((size_t)ds.seq_len);
        for (int i = 0; i < ds.seq_len; ++i) {
            shap_values[(size_t)i] = h_shap[(size_t)i] / (float)out_count;
        }

        int to_print = std::min((int)out_count, max_print);
        for (size_t i = 0; i < (size_t)to_print; i++) {
            float logit = h_logits[i];
            std::cout << "block=" << i << " logit=" << logit << "\n";
        }

        int shap_print = std::min(ds.seq_len, 10);
        std::cout << "shap_values[0.." << shap_print - 1 << "] = ";
        for (int i = 0; i < shap_print; ++i) {
            std::cout << shap_values[(size_t)i] << (i + 1 < shap_print ? ", " : "\n");
        }

        std::string shap_out_path = dataset_path + std::string(".shap_values.csv");
        std::ofstream sf(shap_out_path);
        if (sf) {
            sf << "feature_idx,token_id,shap_value\n";
            for (int j = 0; j < ds.seq_len; ++j) {
                int32_t tok = ds.token_ids[(size_t)0 * (size_t)ds.seq_len + (size_t)j];
                sf << j << "," << tok << "," << shap_values[(size_t)j] << "\n";
            }
            sf.close();
            std::cout << "wrote shap values to " << shap_out_path << "\n";
        } else {
            std::cerr << "Failed to open shap output file: " << shap_out_path << "\n";
        }

        if (ds.n > 0) {
            std::vector<float> cur(emb.embed_dim);
            int sample = 0;
            for (int j = 0; j < ds.seq_len; ++j) {
                int32_t tok = ds.token_ids[(size_t)sample * (size_t)ds.seq_len + (size_t)j];
                for (int d = 0; d < emb.embed_dim; ++d) {
                    cur[d] += emb.data[(size_t)tok * (size_t)emb.embed_dim + (size_t)d];
                }
            }
            for (int d = 0; d < emb.embed_dim; ++d) cur[d] /= (float)ds.seq_len;
            std::vector<float> in_vec = std::move(cur);
            for (const auto& layer : net.layers) {
                std::vector<float> next(layer.out_dim);
                for (int o = 0; o < layer.out_dim; o++) {
                    float acc = layer.b[(size_t)o];
                    const float* wrow = layer.W.data() + (size_t)o * (size_t)layer.in_dim;
                    for (int k = 0; k < layer.in_dim; k++) {
                        acc += wrow[(size_t)k] * in_vec[(size_t)k];
                    }
                    next[(size_t)o] = acc;
                }
                in_vec = std::move(next);
            }
            float cpu_score = in_vec.empty() ? 0.0f : in_vec[0];
            std::cout << "cpu_check sample=0 score=" << cpu_score << "\n";
        }

        cudaFree(d_token_ids);
        cudaFree(d_embedding);
        cudaFree(d_W);
        cudaFree(d_b);
        cudaFree(d_w_offsets);
        cudaFree(d_b_offsets);
        cudaFree(d_in_dims);
        cudaFree(d_out_dims);
        cudaFree(d_logits);
        cudaFree(d_shap);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}