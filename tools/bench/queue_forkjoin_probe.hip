// Queue fork/join stress probe for gfx1151 (Strix Halo).
//
// Build with the ROCm stack under test, not the Nix one:
//   /home/deck/llm-scripts/deps/bin/hipcc -O2 --offload-arch=gfx1151 \
//     -o /tmp/queue_forkjoin_probe tools/bench/queue_forkjoin_probe.hip
// Run: /tmp/queue_forkjoin_probe --budget 120
//
// Reproduces the packet pattern a decode step leaves on one AQL queue when the
// shared-expert overlap is on: per layer, a barrier that depends on the other
// stream's signal, two GEMM-shaped dispatches, two elementwise-shaped
// dispatches, and a closing agent-scope barrier, then a stream synchronise per
// step. 44 of those groups make one step, and the run that aborted on
// 2026-11-16 17:16 queued 264 packets - 44 groups of six - before the command
// processor faulted. Every buffer here is a few MiB so the probe can share the
// device with a server.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

void Check(hipError_t status, const char* what) {
  if (status == hipSuccess) {
    return;
  }
  std::fprintf(stderr, "hip error: %s: %s\n", what,
               hipGetErrorString(status));
  std::exit(1);
}

// Same shape as the q8_0 shared-expert GEMMs: a 256-thread block over a
// two-row tile. The arithmetic is irrelevant, the packet is not.
__global__ void gemm_shaped(const float* __restrict__ in,
                            float* __restrict__ out, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows) {
    out[i] = in[i] * 1.0000001F + 1.0F;
  }
}

// Same shape as the routed-expert elementwise passes: 32-thread blocks over a
// large grid.
__global__ void elementwise_shaped(const float* __restrict__ in,
                                   float* __restrict__ out, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows) {
    out[i] = in[i] + 2.0F;
  }
}

std::uint32_t ComputeQueueCount() {
  std::uint32_t count = 0;
  std::error_code error;
  const auto root = std::filesystem::path("/sys/class/kfd/kfd/proc") /
                    std::to_string(::getpid()) / "queues";
  for (const auto& queue : std::filesystem::directory_iterator(
           root, std::filesystem::directory_options::skip_permission_denied,
           error)) {
    ++count;
  }
  return count;
}

struct Options {
  std::vector<int> streams{2};
  double budget_seconds = 120.0;
  int iters_per_step = 44;
  std::uint64_t steps = 0;
};

void LaunchGemm(hipStream_t stream, const float* in, float* out) {
  gemm_shaped<<<dim3(2560, 2, 1), dim3(256, 1, 1), 0, stream>>>(in, out, 5120);
  gemm_shaped<<<dim3(1024, 2, 1), dim3(256, 1, 1), 0, stream>>>(in, out, 2048);
}

void LaunchElementwise(hipStream_t stream, const float* in, float* out) {
  elementwise_shaped<<<dim3(20480, 1, 1), dim3(32, 1, 1), 0, stream>>>(
      in, out, 20480);
  elementwise_shaped<<<dim3(81920, 1, 1), dim3(32, 1, 1), 0, stream>>>(
      in, out, 81920);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> const char* {
      return i + 1 < argc ? argv[++i] : "";
    };
    if (arg == "--streams") {
      options.streams.clear();
      std::string list = next();
      while (!list.empty()) {
        const auto comma = list.find(',');
        options.streams.push_back(std::atoi(list.substr(0, comma).c_str()));
        list = comma == std::string::npos ? "" : list.substr(comma + 1);
      }
    } else if (arg == "--budget") {
      options.budget_seconds = std::atof(next());
    } else if (arg == "--iters") {
      options.iters_per_step = std::atoi(next());
    } else if (arg == "--steps") {
      options.steps = std::strtoull(next(), nullptr, 10);
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (options.streams.empty()) {
    std::fprintf(stderr, "--streams needs at least one value\n");
    return 2;
  }

  // The aborted run served under the queue plan's text cap.
  ::setenv("GPU_MAX_HW_QUEUES", "2", 0);

  int devices = 0;
  Check(hipGetDeviceCount(&devices), "hipGetDeviceCount");
  if (devices == 0) {
    std::fprintf(stderr, "no HIP device\n");
    return 1;
  }

  constexpr int kRows = 81920;
  std::vector<float> host(kRows, 0.5F);
  float* in = nullptr;
  float* out = nullptr;
  Check(hipMalloc(&in, kRows * sizeof(float)), "hipMalloc in");
  Check(hipMalloc(&out, kRows * sizeof(float)), "hipMalloc out");
  Check(hipMemcpy(in, host.data(), kRows * sizeof(float),
                  hipMemcpyHostToDevice),
        "hipMemcpy");

  const std::size_t stream_count = options.streams.size();
  std::vector<hipStream_t> streams(stream_count, nullptr);
  for (auto& stream : streams) {
    Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
          "hipStreamCreateWithFlags");
  }
  // Two events per stream, as the overlap uses one pair per layer.
  std::vector<hipEvent_t> start(stream_count, nullptr);
  std::vector<hipEvent_t> done(stream_count, nullptr);
  for (std::size_t s = 0; s < stream_count; ++s) {
    Check(hipEventCreateWithFlags(&start[s], hipEventDisableTiming),
          "hipEventCreate start");
    Check(hipEventCreateWithFlags(&done[s], hipEventDisableTiming),
          "hipEventCreate done");
  }

  std::atomic<std::uint64_t> steps{0};
  std::atomic<std::uint64_t> groups{0};
  std::atomic<bool> stop{false};
  std::thread watchdog([&] {
    const auto begin = std::chrono::steady_clock::now();
    while (!stop.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       begin)
              .count();
      std::printf("  t=%6.1fs steps=%llu groups=%llu queues=%u\n", elapsed,
                  static_cast<unsigned long long>(
                      steps.load(std::memory_order_relaxed)),
                  static_cast<unsigned long long>(
                      groups.load(std::memory_order_relaxed)),
                  ComputeQueueCount());
      std::fflush(stdout);
    }
  });

  std::printf("streams:");
  for (int n : options.streams) {
    std::printf(" %d", n);
  }
  std::printf(" iters/step=%d budget=%.0fs queues=%u\n",
              options.iters_per_step, options.budget_seconds,
              ComputeQueueCount());
  std::fflush(stdout);

  const auto begin = std::chrono::steady_clock::now();
  std::uint64_t step = 0;
  while (!stop.load(std::memory_order_relaxed)) {
    if (options.steps != 0 && step >= options.steps) {
      break;
    }
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - begin)
                               .count();
    if (elapsed > options.budget_seconds) {
      break;
    }
    for (int iteration = 0; iteration < options.iters_per_step; ++iteration) {
      for (std::size_t s = 0; s < stream_count; ++s) {
        hipStream_t stream = streams[s];
        hipStream_t other = streams[(s + 1) % stream_count];
        if (stream_count > 1) {
          // Fork: this stream waits for the previous stream's last packet.
          Check(hipEventRecord(start[s], other), "hipEventRecord fork");
          Check(hipStreamWaitEvent(stream, start[s], 0),
                "hipStreamWaitEvent fork");
        }
        LaunchGemm(stream, in, out);
        LaunchElementwise(stream, in, out);
        if (stream_count > 1) {
          Check(hipEventRecord(done[s], stream), "hipEventRecord join");
          Check(hipStreamWaitEvent(other, done[s], 0),
                "hipStreamWaitEvent join");
        }
      }
      groups.fetch_add(1, std::memory_order_relaxed);
    }
    // The forward synchronises once per step, as the real decode path does.
    Check(hipStreamSynchronize(streams[0]), "hipStreamSynchronize");
    const hipError_t last = hipGetLastError();
    if (last != hipSuccess) {
      std::fprintf(stderr, "step %llu: %s\n",
                   static_cast<unsigned long long>(step),
                   hipGetErrorString(last));
      return 1;
    }
    ++step;
    steps.store(step, std::memory_order_relaxed);
  }

  stop.store(true);
  watchdog.join();
  for (std::size_t s = 0; s < stream_count; ++s) {
    Check(hipStreamSynchronize(streams[s]), "final synchronise");
  }
  float result = 0.0F;
  Check(hipMemcpy(&result, out, sizeof(float), hipMemcpyDeviceToHost),
        "hipMemcpy result");
  std::printf("done: steps=%llu groups=%llu result=%.3f queues=%u\n",
              static_cast<unsigned long long>(step),
              static_cast<unsigned long long>(
                  groups.load(std::memory_order_relaxed)),
              result, ComputeQueueCount());
  for (std::size_t s = 0; s < stream_count; ++s) {
    (void)hipEventDestroy(done[s]);
    (void)hipEventDestroy(start[s]);
    (void)hipStreamDestroy(streams[s]);
  }
  (void)hipFree(in);
  (void)hipFree(out);
  return 0;
}
