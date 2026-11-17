// Device-pool commit probe for gfx1151 (Strix Halo).
//
// Build with the ROCm stack under test, not the Nix one:
//   /home/deck/llm-scripts/deps/bin/hipcc -O2 --offload-arch=gfx1151 \
//     -o /tmp/pool_commit_probe tools/bench/pool_commit_probe.hip
// Run: /tmp/pool_commit_probe --probe-gib 8
//
// Tests one documented hazard directly: the snapshot allocator in
// src/models/qwen38_flash_next/kernels/rocm/executor.cpp notes that ROCm's VM
// pool corrupts its heap when a page commit fails, so it refuses to request
// storage beyond the device headroom. This probe does the opposite on purpose -
// it takes a live queue, asks the pool for a block larger than the device can
// commit, and then checks whether the queue still executes. A wedged queue here
// means the pool failure damages state the runtime cannot recover, which is a
// queue fault that presents as an illegal opcode in the command stream rather
// than as a failed allocation.
//
// Every block it keeps is at most --probe-gib, and the over-commit request is
// never materialized, so the host is not filled.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

void Check(hipError_t status, const char* what) {
  if (status == hipSuccess) {
    return;
  }
  std::fprintf(stderr, "hip error: %s: %s\n", what, hipGetErrorString(status));
  std::exit(1);
}

__global__ void touch(float* data, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = data[i] * 1.5F + 1.0F;
  }
}

// A small, complete round trip. Returns false when the queue no longer
// executes, which is the failure this probe is looking for.
bool QueueHealthy(const char* label) {
  constexpr int kCount = 4096;
  float* buffer = nullptr;
  if (hipMalloc(&buffer, kCount * sizeof(float)) != hipSuccess) {
    std::printf("  [%s] hipMalloc failed\n", label);
    return false;
  }
  std::vector<float> host(kCount, 1.0F);
  if (hipMemcpy(buffer, host.data(), kCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess) {
    std::printf("  [%s] hipMemcpy H2D failed\n", label);
    return false;
  }
  touch<<<(kCount + 255) / 256, 256>>>(buffer, kCount);
  const hipError_t launch = hipGetLastError();
  const hipError_t sync = hipStreamSynchronize(nullptr);
  if (launch != hipSuccess || sync != hipSuccess) {
    std::printf("  [%s] queue faulted: launch=%s sync=%s\n", label,
                hipGetErrorString(launch), hipGetErrorString(sync));
    (void)hipFree(buffer);
    return false;
  }
  float value = 0.0F;
  if (hipMemcpy(&value, buffer, sizeof(float), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    std::printf("  [%s] hipMemcpy D2H failed\n", label);
    return false;
  }
  (void)hipFree(buffer);
  const bool ok = value == 2.5F;
  std::printf("  [%s] queue %s (value=%.3f)\n", label, ok ? "healthy" : "WRONG",
              value);
  return ok;
}

void ReportMemory(const char* label) {
  std::size_t free = 0;
  std::size_t total = 0;
  if (hipMemGetInfo(&free, &total) != hipSuccess) {
    std::printf("  [%s] hipMemGetInfo failed\n", label);
    return;
  }
  std::printf("  [%s] free=%.2f GiB total=%.2f GiB\n", label,
              static_cast<double>(free) / 1073741824.0,
              static_cast<double>(total) / 1073741824.0);
}

}  // namespace

int main(int argc, char** argv) {
  double probe_gib = 8.0;
  bool overcommit = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--probe-gib" && i + 1 < argc) {
      probe_gib = std::atof(argv[++i]);
    } else if (arg == "--no-overcommit") {
      overcommit = false;
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }

  int devices = 0;
  Check(hipGetDeviceCount(&devices), "hipGetDeviceCount");
  if (devices == 0) {
    std::fprintf(stderr, "no HIP device\n");
    return 1;
  }

  std::printf("pool commit probe: probe=%.2f GiB overcommit=%d\n", probe_gib,
              overcommit ? 1 : 0);
  ReportMemory("start");

  // The same pool the snapshot allocator builds: pinned device-location
  // storage, so commits land in the same backend path as gufo's checkpoints.
  hipMemPoolProps properties{};
  properties.allocType = hipMemAllocationTypePinned;
  properties.handleTypes = hipMemHandleTypeNone;
  properties.location.type = hipMemLocationTypeDevice;
  Check(hipGetDevice(&properties.location.id), "hipGetDevice");
  hipMemPool_t pool = nullptr;
  Check(hipMemPoolCreate(&pool, &properties), "hipMemPoolCreate");

  hipStream_t stream = nullptr;
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
        "hipStreamCreateWithFlags");

  if (!QueueHealthy("before")) {
    std::fprintf(stderr, "queue was already unhealthy before the probe\n");
    return 1;
  }

  // Phase 1: take real pool storage, the way checkpoint captures do.
  const std::uint64_t chunk = 256ULL << 20;
  const std::uint64_t wanted =
      static_cast<std::uint64_t>(probe_gib * 1073741824.0);
  std::vector<void*> blocks;
  for (std::uint64_t taken = 0; taken < wanted; taken += chunk) {
    void* block = nullptr;
    const hipError_t status =
        hipMallocFromPoolAsync(&block, chunk, pool, stream);
    if (status != hipSuccess) {
      std::printf("  allocation at %.2f GiB failed: %s\n",
                  static_cast<double>(taken) / 1073741824.0,
                  hipGetErrorString(status));
      break;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
      std::printf("  allocation sync at %.2f GiB failed\n",
                  static_cast<double>(taken) / 1073741824.0);
      break;
    }
    blocks.push_back(block);
  }
  std::printf("  took %zu pool blocks (%.2f GiB)\n", blocks.size(),
              static_cast<double>(blocks.size()) * chunk / 1073741824.0);
  ReportMemory("after take");

  // Phase 2: release them, as a checkpoint eviction does, then make the pool
  // serve the memory again immediately.
  for (void* block : blocks) {
    (void)hipFree(block);
  }
  blocks.clear();
  (void)hipStreamSynchronize(stream);
  ReportMemory("after release");

  // Phase 3: ask for more than the device can commit. gufo refuses this
  // request on purpose; the question is what it does to the queue when it
  // happens anyway.
  if (overcommit) {
    std::size_t free = 0;
    std::size_t total = 0;
    Check(hipMemGetInfo(&free, &total), "hipMemGetInfo");
    const std::uint64_t too_much =
        static_cast<std::uint64_t>(free) + (8ULL << 30);
    std::printf("  requesting %.2f GiB with %.2f GiB free\n",
                static_cast<double>(too_much) / 1073741824.0,
                static_cast<double>(free) / 1073741824.0);
    void* block = nullptr;
    const hipError_t status =
        hipMallocFromPoolAsync(&block, too_much, pool, stream);
    std::printf("  over-commit returned: %s\n", hipGetErrorString(status));
    const hipError_t sync = hipStreamSynchronize(stream);
    std::printf("  over-commit sync returned: %s\n", hipGetErrorString(sync));
    if (status == hipSuccess && block != nullptr) {
      (void)hipFree(block);
    }
    // The runtime may still hold an error from the failed commit.
    const hipError_t last = hipGetLastError();
    std::printf("  hipGetLastError after over-commit: %s\n",
                hipGetErrorString(last));
  }

  const bool healthy = QueueHealthy("after");
  (void)hipStreamDestroy(stream);
  (void)hipMemPoolDestroy(pool);
  std::printf("result: queue %s after pool commit failure\n",
              healthy ? "SURVIVED" : "CORRUPTED");
  return healthy ? 0 : 1;
}
