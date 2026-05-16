#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    return 1; \
  } \
} while (0)

__global__ void saxpy(float* y, const float* x, float a, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  int count = 0;
  CHECK(cudaGetDeviceCount(&count));
  std::printf("cuda_device_count=%d\n", count);
  for (int dev = 0; dev < count; ++dev) {
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, dev));
    std::printf("device=%d name=%s cc=%d.%d sms=%d global_mem_gib=%.2f l2_bytes=%zu bus_width_bits=%d\n",
      dev, p.name, p.major, p.minor, p.multiProcessorCount,
      (double)p.totalGlobalMem / 1024.0 / 1024.0 / 1024.0,
      (size_t)p.l2CacheSize, p.memoryBusWidth);
  }
  if (count == 0) return 0;

  CHECK(cudaSetDevice(0));
  size_t n = 256ULL * 1024ULL * 1024ULL / sizeof(float); // 256 MiB per vector
  size_t bytes = n * sizeof(float);
  float *dx = nullptr, *dy = nullptr;
  CHECK(cudaMalloc(&dx, bytes));
  CHECK(cudaMalloc(&dy, bytes));
  CHECK(cudaMemset(dx, 1, bytes));
  CHECK(cudaMemset(dy, 2, bytes));
  CHECK(cudaDeviceSynchronize());

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  for (int i = 0; i < 10; ++i) saxpy<<<grid, block>>>(dy, dx, 2.0f, n);
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));
  int iters = 100;
  for (int i = 0; i < iters; ++i) saxpy<<<grid, block>>>(dy, dx, 2.0f, n);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  // SAXPY touches x read + y read + y write = 3 * bytes per iteration.
  double gib = (double)iters * 3.0 * (double)bytes / 1024.0 / 1024.0 / 1024.0;
  double seconds = ms / 1000.0;
  std::printf("saxpy_bytes_per_vector=%zu iterations=%d seconds=%.6f approx_gib_per_s=%.3f\n", bytes, iters, seconds, gib / seconds);

  CHECK(cudaFree(dx));
  CHECK(cudaFree(dy));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return 0;
}
