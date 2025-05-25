
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>

#include <cuda_runtime.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif


void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

void checkDeviceMemory() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Device memory (free/total) = %lld/%lld bytes\n", free, total);
}

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    float* dDataptr;
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;


void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    cudaCheck(cudaMalloc(&(s->x), sizeof(float)*p->dim)); cudaCheck(cudaMemset(s->x, 0, sizeof(float)*p->dim));
    cudaCheck(cudaMalloc(&(s->xb), sizeof(float)*p->dim)); cudaCheck(cudaMemset(s->xb, 0, sizeof(float)*p->dim));
    cudaCheck(cudaMalloc(&(s->xb2), sizeof(float)*p->dim)); cudaCheck(cudaMemset(s->xb2, 0, sizeof(float)*p->dim));
    cudaCheck(cudaMalloc(&(s->hb), sizeof(float)*p->hidden_dim)); cudaCheck(cudaMemset(s->hb, 0, sizeof(float)*p->hidden_dim));
    cudaCheck(cudaMalloc(&(s->hb2), sizeof(float)*p->hidden_dim)); cudaCheck(cudaMemset(s->hb2, 0, sizeof(float)*p->hidden_dim));
    cudaCheck(cudaMalloc(&(s->q), sizeof(float)*p->dim)); cudaCheck(cudaMemset(s->q, 0, sizeof(float)*p->dim));
    cudaCheck(cudaMalloc(&(s->key_cache), sizeof(float)*p->n_layers * p->seq_len * kv_dim)); cudaCheck(cudaMemset(s->key_cache, 0, sizeof(float)*p->n_layers * p->seq_len * kv_dim));
    cudaCheck(cudaMalloc(&(s->value_cache), sizeof(float)*p->n_layers * p->seq_len * kv_dim)); cudaCheck(cudaMemset(s->value_cache, 0, sizeof(float)*p->n_layers * p->seq_len * kv_dim));
    cudaCheck(cudaMalloc(&(s->att), sizeof(float)*p->n_heads * p->seq_len)); cudaCheck(cudaMemset(s->att, 0, sizeof(float)*p->n_heads * p->seq_len));
    cudaCheck(cudaMalloc(&(s->logits), sizeof(float)*p->vocab_size)); cudaCheck(cudaMemset(s->logits, 0, sizeof(float)*p->vocab_size));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    checkDeviceMemory();
}

void free_run_state(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->att);
    cudaFree(s->logits);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
}

size_t padded_float_count(size_t n_bytes) {
    return (n_bytes + sizeof(float) - 1) / sizeof(float);
}

void memory_map_weights(TransformerWeights *w, Config* p, unsigned long long ps, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;

    float* dp;
    size_t malloc_cnt = padded_float_count(ps);
    cudaCheck(cudaMalloc(&dp, sizeof(float)*malloc_cnt));
    cudaCheck(cudaMemset(dp, 0, sizeof(float)*malloc_cnt));

    w->dDataptr = dp;

    int ptr_stride = p->vocab_size * p->dim;
    w->token_embedding_table = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;

    ptr_stride = n_layers * p->dim;
    w->rms_att_weight = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;

    ptr_stride = n_layers * p->dim * (p->n_heads * head_size);
    w->wq = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;

    w->wk = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;

    w->wv = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;

    ptr_stride = n_layers * (p->n_heads * head_size) * p->dim;
    w->wo = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;
    
    ptr_stride = n_layers * p->dim;
    w->rms_ffn_weight = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;

    ptr_stride = n_layers * p->dim * p->hidden_dim;
    w->w1 = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;

    ptr_stride =n_layers * p->hidden_dim * p->dim;
    w->w2 = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;

    ptr_stride = n_layers * p->dim * p->hidden_dim;
    w->w3 = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;

    ptr_stride = p->dim;
    w->rms_final_weight = dp;
    cudaCheck(cudaMemcpy(dp, ptr, sizeof(float)*(ptr_stride), cudaMemcpyHostToDevice));
    ptr += ptr_stride;
    dp += ptr_stride;

    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    dp += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    dp += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)

    w->wcls = shared_weights ? w->token_embedding_table : dp;

    checkDeviceMemory();
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);

    ssize_t fs = *file_size;
    int cs = sizeof(Config);
    unsigned long long parameter_size = fs - cs;
    printf("전체 크기: %zd\n", fs);
    printf("config 크기: %d\n", cs);
    printf("파라미터 크기: %d\n", parameter_size);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, parameter_size, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
    // free GPU weights (CUDA)
    if (t->weights.dDataptr != NULL) {
        cudaError_t err = cudaFree(t->weights.dDataptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA free failed: %s\n", cudaGetErrorString(err));
        }
        t->weights.dDataptr = NULL;
    }
}

// ----------------------------------------------------------------------------
// CUDA kernel; the dynamics of the Transformer

__global__ void rms_norm_kernel(float* o, float* x, float* weight, int size) {
    // sum of squares를 한 블록에 대해 공유 메모리로 집계
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < size) ? x[i] * x[i] : 0.0f;

    // 각 스레드가 sdata에 자신의 값을 저장
    sdata[tid] = v;
    __syncthreads();

    // reduction (sum)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < size) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

        // 블록의 0번째 thread가 sum을 구함
    float ss = 0.0f;
    if (tid == 0) {
        ss = sdata[0] / size + 1e-5f;
        ss = 1.0f / sqrtf(ss);
        sdata[0] = ss; // normalization 계수 저장
    }
    __syncthreads();
    ss = sdata[0];

    // normalize & scale
    if (i < size) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

__global__ void matmul_kernel(float* xout, const float* x, const float* w, int n, int d) {
    extern __shared__ float x_shared[];
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 이 thread가 처리할 xout 인덱스

    if (i >= d) return; // 출력 범위 밖이면 즉시 리턴

    float val = 0.0f;

    // x를 chunk 단위로 shared memory에 올리고 곱셈
    for (int offset = 0; offset < n; offset += blockDim.x) {
        // shared memory에 blockDim.x개만큼 복사
        int j = offset + threadIdx.x;
        if (j < n)
            x_shared[threadIdx.x] = x[j];
        else
            x_shared[threadIdx.x] = 0.0f;
        __syncthreads();

        // chunk 내에서 연산 (blockDim.x로 잘림)
        int chunk = min(blockDim.x, n - offset);
        for (int k = 0; k < chunk; ++k) {
            val += w[i * n + offset + k] * x_shared[k];
        }
        __syncthreads(); // 다음 chunk 준비
    }

    xout[i] = val;
}

__global__ void rope_kernel(
    float* q, float* k,
    int pos, int n_heads, int head_size, int kv_dim
) {
    int h = blockIdx.y;                // head index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx * 2;                   // pair index (even only)

    if (h >= n_heads || i + 1 >= head_size) return;

    // q/k는 [n_heads, head_size]
    int q_offset = h * head_size;
    int k_offset = h * head_size;

    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    // 쿼리
    float v0 = q[q_offset + i];
    float v1 = q[q_offset + i + 1];
    q[q_offset + i]     = v0 * fcr - v1 * fci;
    q[q_offset + i + 1] = v0 * fci + v1 * fcr;

    // 키
    if (i < kv_dim && k != nullptr) {
        float k0 = k[k_offset + i];
        float k1 = k[k_offset + i + 1];
        k[k_offset + i]     = k0 * fcr - k1 * fci;
        k[k_offset + i + 1] = k0 * fci + k1 * fcr;
    }
}


__global__ void attn_score_kernel(
    float* att,             // (n_heads, seq_len)
    const float* q,         // (n_heads, head_size)
    const float* key_cache, // (layer_offset + t * kv_dim + head_offset)
    int n_heads,
    int seq_len,
    int head_size,
    int kv_dim,
    int pos,
    int loff,
    int kv_mul
) {
    int h = blockIdx.x;    // head index
    int t = blockIdx.y;    // time step (0 ~ pos)
    int i = threadIdx.x;   // vector dim (0 ~ head_size-1)

    if (h >= n_heads || t > pos) return;

    // 쿼리 벡터
    const float* qh = q + h * head_size;
    // 키 벡터
    int k_head_offset = (h / kv_mul) * head_size;
    const float* kht = key_cache + loff + t * kv_dim + k_head_offset;

    // dot product을 위한 부분합 구하기
    __shared__ float partsum[256]; // head_size 최대 256일 때
    float local = 0.0f;
    if (i < head_size) {
        local = qh[i] * kht[i];
    }
    partsum[i] = local;
    __syncthreads();

    // reduction (block 내 sum)
    // blockDim.x == head_size일 때만 안전
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) {
            partsum[i] += partsum[i + s];
        }
        __syncthreads();
    }

    if (i == 0) {
        float score = partsum[0] / sqrtf((float)head_size);
        att[h * seq_len + t] = score;
    }
}

__global__ void softmax_kernel(float* x, int size) {
    extern __shared__ float smem[]; // shared memory
    int tid = threadIdx.x;
    // 1. 각 thread가 하나의 원소 담당
    float val = (tid < size) ? x[tid] : -INFINITY;

    // 2. max값 찾기 (reduction)
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < size) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    float maxval = smem[0];
    __syncthreads();

    // 3. exp(x - max), sum
    float ex = (tid < size) ? expf(x[tid] - maxval) : 0.0f;
    smem[tid] = ex;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < size) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float sum = smem[0];
    __syncthreads();

    // 4. normalize
    if (tid < size && sum > 0.0f) {
        x[tid] = ex / sum;
    }
}

__global__ void residual_add_kernel(float* x, const float* y, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) x[i] += y[i];
}

__global__ void swiglu_kernel(float* out, const float* x1, const float* x2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x1[i];
        // SiLU(x) = x * sigmoid(x)
        float silu = v / (1.0f + expf(-v));
        out[i] = silu * x2[i];
    }
}

__global__ void weighted_sum_kernel(
    float* xb,             // [n_heads, head_size]
    const float* att,      // [n_heads, pos+1]
    const float* value_cache, // [n_layers, seq_len, kv_dim]
    int n_heads,
    int head_size,
    int kv_dim,
    int pos,
    int kv_mul,
    int loff
) {
    int h = blockIdx.x;     // head index
    int i = threadIdx.x;    // dim within head

    if (h >= n_heads || i >= head_size) return;

    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
        float a = att[h * (pos+1) + t];
        // value_cache 인덱싱은 기존과 동일
        int v_off = loff + t * kv_dim + (h / kv_mul) * head_size;
        sum += a * value_cache[v_off + i];
    }
    xb[h * head_size + i] = sum;
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t shared_mem = blockSize * sizeof(float);

    rms_norm_kernel<<<gridSize, blockSize, shared_mem>>>(o, x, weight, size);
    cudaCheck(cudaGetLastError());
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int blockSize = 128;
    int gridSize = (d + blockSize - 1) / blockSize;
    size_t shared_mem = blockSize * sizeof(float);

    matmul_kernel<<<gridSize, blockSize, shared_mem>>>(xout, x, w, n, d);
    cudaCheck(cudaGetLastError());

}

void rope(float* q, float* k, int pos, int dim, int head_size, int kv_dim, int n_heads) {
    int num_pairs = head_size / 2; // head_size: 각 head의 dim
    int blockSize = 128;
    int gridSize = (num_pairs + blockSize - 1) / blockSize;
    dim3 grid(gridSize, n_heads); // (pair, head)
    rope_kernel<<<grid, blockSize>>>(q, k, pos, n_heads, head_size, kv_dim);
    cudaCheck(cudaGetLastError());
}

void attn_score(
    float* att, float* q, float* key_cache,
    int n_heads, int seq_len, int head_size, int kv_dim,
    int pos, int loff, int kv_mul
) {
    // pos+1만큼만 처리 (0~pos)
    dim3 grid(n_heads, pos + 1);
    int block = head_size;
    attn_score_kernel<<<grid, block>>>(
        att, q, key_cache, n_heads, seq_len, head_size, kv_dim, pos, loff, kv_mul
    );
    cudaCheck(cudaGetLastError());
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    int block = 256;
    size_t shared_mem = block * sizeof(float);
    softmax_kernel<<<1, block, shared_mem>>>(x, size);
    cudaCheck(cudaGetLastError());
}

void softmax_host(float* x, int size) {
    float maxval = x[0];
    for (int i = 1; i < size; i++) if (x[i] > maxval) maxval = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - maxval);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

void residual_add(float* x, float* y, int dim) {
    int block = 128;
    int grid = (dim + block - 1) / block;
    residual_add_kernel<<<grid, block>>>(x, y, dim);
    cudaCheck(cudaGetLastError());
}

void swiglu(float* out, float* x1, float* x2, int n) {
    int block = 128;
    int grid = (n + block - 1) / block;
    swiglu_kernel<<<grid, block>>>(out, x1, x2, n);
    cudaCheck(cudaGetLastError());
}

void weighted_sum(
    float* xb, float* att, float* value_cache,
    int n_heads, int head_size, int kv_dim, int pos, int kv_mul, int loff
) {
    dim3 grid(n_heads);
    dim3 block(head_size);
    weighted_sum_kernel<<<grid, block>>>(
        xb, att, value_cache, n_heads, head_size, kv_dim, pos, kv_mul, loff
    );
    cudaCheck(cudaGetLastError());
}


float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // 1. copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    cudaCheck(cudaMemcpy(x, content_row, dim*sizeof(*x), cudaMemcpyDeviceToDevice));

    // 2. forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // 3. RoPE 모든 head에 동시에 적용
        rope(s->q, s->k, pos, dim, head_size, kv_dim, p->n_heads);

        // 4. 어텐션 스코어 계산
        attn_score(s->att, s->q, s->key_cache, p->n_heads, p->seq_len, head_size, kv_dim, pos, loff, kv_mul);

        // 5. softmax 각 head별로
        for (int h = 0; h < p->n_heads; h++) {
            float* att = s->att + h * (pos+1);
            softmax(att, pos+1);
        }

        // 6. weighted sum (이제 CUDA 커널로!)
        weighted_sum(s->xb, s->att, s->value_cache, p->n_heads, head_size, kv_dim, pos, kv_mul, loff);

        // 7. output projection (wo)
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // 8. residual connection (x += xb2)
        residual_add(x, s->xb2, dim);

        // 9. ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // 10. w1(x), w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // 11. SwiGLU (s->hb = silu(s->hb) * s->hb2)
        swiglu(s->hb, s->hb, s->hb2, hidden_dim);

        // 12. final FFN: xb = hb @ w2
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // 13. residual connection (x += xb)
        residual_add(x, s->xb, dim);
    }

    // 14. final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // 15. classifier
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);

    return s->logits; // Device pointer!

    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*)bsearch(
    &tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens
);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        // t->sorted_vocab = (TokenIndex*)(t->vocab_size * sizeof(TokenIndex));
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char*)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits_host) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits_host, sampler->vocab_size);
    } else {
        // temperature scaling
        for (int q=0; q<sampler->vocab_size; q++) {
            logits_host[q] /= sampler->temperature;
        }
        // host softmax
        softmax_host(logits_host, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits_host, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits_host, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}
// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // host buffer for logits
    float* logits_host = (float*)malloc(sizeof(float) * transformer->config.vocab_size);

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos); // device ptr
        // --- 반드시 host로 복사! ---
        cudaCheck(cudaMemcpy(logits_host, logits, sizeof(float)*transformer->config.vocab_size, cudaMemcpyDeviceToHost));

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits_host); // host softmax & sampling
        }
        pos++;
        if (next == 1) { break; }
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
    free(logits_host);
    free(prompt_tokens);
}


void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    // } else if (strcmp(mode, "chat") == 0) {
    //     chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif