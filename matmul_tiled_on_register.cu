#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

// 定义错误检查宏
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ==========================================
// 核心 Kernel：Register Tiling + Shared Memory Tiling
// ==========================================
#define BM 64  // Block Tile M
#define BN 64  // Block Tile N
#define BK 16  // Block Tile K
#define TM 4   // Thread Tile M
#define TN 4   // Thread Tile N

__global__ void matmul_register_tiled(const float* __restrict__ A, 
                                      const float* __restrict__ B, 
                                      float* __restrict__ C, 
                                      int M, int N, int K) {
    // 1. 声明 Shared Memory，加1避免 Bank Conflict
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    int tx = threadIdx.x; // 0~15
    int ty = threadIdx.y; // 0~15
    int tid = ty * blockDim.x + tx; // 0~255

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int row = block_row + ty * TM;
    int col = block_col + tx * TN;

    // 2. 寄存器累加器
    float C_local[TM][TN] = {0.0f};
    float A_frag[TM];
    float B_frag[TN];

    // 3. 遍历 K 维度
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        
        // --- 阶段一：Global -> Shared (协作加载) ---
        for (int i = 0; i < BM * BK; i += 256) {
            int idx = i + tid;
            int r = idx / BK;
            int c = idx % BK;
            if (block_row + r < M && k_tile + c < K) {
                As[r][c] = A[(block_row + r) * K + (k_tile + c)];
            } else {
                As[r][c] = 0.0f;
            }
        }

        for (int i = 0; i < BK * BN; i += 256) {
            int idx = i + tid;
            int r = idx / BN;
            int c = idx % BN;
            if (k_tile + r < K && block_col + c < N) {
                Bs[r][c] = B[(k_tile + r) * N + (block_col + c)];
            } else {
                Bs[r][c] = 0.0f;
            }
        }
        
        __syncthreads();

        // --- 阶段二：Shared -> Register (细长条读取 + 外积计算) ---
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                A_frag[i] = As[ty * TM + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                B_frag[j] = Bs[k][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    C_local[i][j] += A_frag[i] * B_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // 4. 写回 Global Memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + (col + j)] = C_local[i][j];
            }
        }
    }
}

// ==========================================
// Main 函数：纯粹的 GPU 性能测试
// ==========================================
int main() {
    // 你可以在这里随意修改矩阵大小进行测试
    int M = 4096; 
    int N = 4096; 
    int K = 4096; 

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    std::cout << "========================================" << std::endl;
    std::cout << "GPU Register Tiling GEMM Benchmark" << std::endl;
    std::cout << "Matrix Size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // 1. 分配 Host 内存并初始化随机数（只做一次性初始化）
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& val : h_A) val = dis(gen);
    for (auto& val : h_B) val = dis(gen);

    // 2. 分配 Device 内存并拷贝初始数据
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size_C));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // 3. 配置 Kernel 启动参数
    dim3 blockDim(16, 16);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 4. 预热 (Warmup)
    int warmup_iter = 5;
    std::cout << "Warming up (" << warmup_iter << " iterations)..." << std::endl;
    for (int i = 0; i < warmup_iter; ++i) {
        matmul_register_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 5. CUDA Events 测速
    int test_iter = 50; // 增加测试轮数，让数据更稳定
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    for (int i = 0; i < test_iter; ++i) {
        matmul_register_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    // 6. 计算性能指标
    float avg_time_ms = milliseconds / test_iter;
    double flops = 2.0 * M * N * K; // 乘法和加法各算一次
    double gflops = (flops / (avg_time_ms / 1000.0)) / 1e9; // 转换为 GFLOPS

    std::cout << "\n=== GPU Performance Results ===" << std::endl;
    std::cout << "Total time for " << test_iter << " iterations: " << milliseconds << " ms" << std::endl;
    std::cout << "Average time per iteration: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "========================================" << std::endl;

    // 7. 清理资源
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
