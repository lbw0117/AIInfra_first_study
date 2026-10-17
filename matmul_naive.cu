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

// 朴素矩阵乘法：C = A * B
// 假设 A 是 M x K，B 是 K x N，C 是 M x N
// 为了简单，这里使用 N x N 的方阵
__global__ void matMulNaive(const float* A, const float* B, float* C, int N) {
    // 计算当前线程负责的 C 的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界保护：防止越界
    if (row < N && col < N) {
        float sum = 0.0f;
        // 朴素做法：每个线程独立算一行乘以一列的点积
        // 直接在 Global Memory 里遍历，内存访问非常频繁
        for (int k = 0; k < N; ++k) {
            // A[row][k] 的线性索引是 row * N + k
            // B[k][col] 的线性索引是 k * N + col
            sum += A[row * N + k] * B[k * N + col];
        }
        // C[row][col] 的线性索引是 row * N + col
        C[row * N + col] = sum;
    }
}

int main() {
    // 1. 设置矩阵大小 (N x N)
    const int N = 4096;
    const size_t bytes = (size_t)N * N * sizeof(float);

    // 2. 分配 CPU 内存
    float* h_A = (float*)std::malloc(bytes);
    float* h_B = (float*)std::malloc(bytes);
    float* h_C = (float*)std::malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        std::fprintf(stderr, "malloc failed\n");
        return 1;
    }

    // 3. 初始化数据
    // A 全为 1.0，B 全为 2.0，那么 C 的每个元素应该是 K * 1.0 * 2.0 = N * 2.0 = 1024.0
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 4. 分配 GPU 显存
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 5. CPU -> GPU 拷贝数据
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // 6. 配置线程块和网格
    // 每个 block 16x16 = 256 个线程
    dim3 threads(16, 16);
    // 网格尺寸 = (列数 / 16, 行数 / 16)，向上取整
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    // 7. 启动 kernel
    matMulNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // 8. 检查错误 + 同步
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 9. GPU -> CPU 拷回结果
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // 在 main 函数里，启动 kernel 之前：
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 记录开始时间
    CUDA_CHECK(cudaEventRecord(start));

    // 启动 kernel
    matMulNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // 记录结束时间
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); // 等 GPU 执行完

    // 计算耗时
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::printf("Kernel execution time: %.3f ms\n", milliseconds);

    // 销毁事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 10. 校验结果
    bool ok = true;
    float expected = (float)N * 2.0f;
    for (int i = 0; i < N * N; ++i) {
        // 浮点数比较要允许微小误差
        if (std::abs(h_C[i] - expected) > 1e-3) {
            std::printf("Mismatch at %d: got %f, expected %f\n", i, h_C[i], expected);
            ok = false;
            break;
        }
    }
    std::printf("%s\n", ok ? "OK" : "FAIL");

    // 11. 释放内存
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
