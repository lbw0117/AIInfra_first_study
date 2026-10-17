#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// 分块大小
#define TILE_SIZE 32

// 分块矩阵乘法：C = A * B
// A 是 N x N，B 是 N x N，C 是 N x N
__global__ void matMulTiled(const float* A, const float* B, float* C, int N) {
    // 每个线程块负责 C 里的一个 TILE_SIZE x TILE_SIZE 的小块
    // 声明共享内存，大小是 TILE_SIZE x TILE_SIZE
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // 当前线程在 C 里负责的行和列
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 沿着 K 维度，一个 tile 一个 tile 地前进
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 协作加载 A 的一个 tile 到共享内存
        // A[row][t*TILE_SIZE + threadIdx.x] 对应的线性索引
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 协作加载 B 的一个 tile 到共享内存
        // B[t*TILE_SIZE + threadIdx.y][col] 对应的线性索引
        if (t * TILE_SIZE + threadIdx.y < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 必须同步，等所有线程都加载完才能开始算
        __syncthreads();

        // 在当前 tile 里做点积
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // 再次同步，防止下一轮 tile 的加载覆盖了还没算完的数据
        __syncthreads();
    }

    // 写回结果
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 4096;
    const size_t bytes = (size_t)N * N * sizeof(float);

    float* h_A = (float*)std::malloc(bytes);
    float* h_B = (float*)std::malloc(bytes);
    float* h_C = (float*)std::malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        std::fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    // warmup
    matMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int REPEAT = 5;
    float total_ms = 0.0f;
    for (int i = 0; i < REPEAT; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        matMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    std::printf("Tiled average time: %.3f ms\n", total_ms / REPEAT);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    float expected = (float)N * 2.0f;
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(h_C[i] - expected) > 1e-3) {
            std::printf("Mismatch at %d: got %f, expected %f\n", i, h_C[i], expected);
            ok = false;
            break;
        }
    }
    std::printf("%s\n", ok ? "OK" : "FAIL");

    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
